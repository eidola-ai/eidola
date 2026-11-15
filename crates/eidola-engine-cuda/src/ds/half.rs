//! A logical device: the memory and kernels one half of the model runs on.
//!
//! The executor has two, an encoder's and a decoder's. Each half allocates its own
//! weights, pools, tables and scratch through its own [`Device`] and hands kernels only
//! addresses of that device, so nothing of one half is addressable from the other; what
//! crosses is a host value ([`super::boundary`]). Both halves may sit on one physical
//! GPU ([`CudaDevice::open`] twice on one ordinal: one context, two streams, disjoint
//! allocations) or on two; that is which ordinal each is opened on.

use std::any::Any;
use std::sync::Mutex;

use cudarc::driver::sys;
use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::sampling::Stream;
use eidola_engine::secret::KvKey;

use super::abi::*;
use super::attention::DsAttention;
use super::gemm_ref::DsGemmRef;
use super::head::DsHead;
use super::index::DsIndex;
use super::kernels::DsKernels;
use super::ops::DsOps;
use crate::kv_cipher::{CipherItem, CipherItems, KvCipher};
use crate::launch::dptr;
use crate::module::KernelDir;
use crate::sampler::{SampleLaunch, SampleRow, Sampler};
use crate::{CudaError, Gpu, Result, narrow};

/// What a device allocation holds, for the memory accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Weights,
    Kv,
    Tables,
    /// Per-slot model state.
    State,
    Scratch,
}

/// One sampling launch: row `r` samples the first `n` logits of row
/// `rows[r].logit_row` of `logits` (rows of `logits_stride` floats) into `tokens[r]`,
/// with its distribution in `probs[r · n ..]`; non-finite logits raise
/// [`crate::sampler::STATUS_NON_FINITE`] in `status`.
#[derive(Clone, Copy, Debug)]
pub struct SampleArgs<'a> {
    pub logits: u64,
    pub logits_stride: u32,
    /// Logit rows `logits` holds.
    pub logit_rows: u32,
    pub n: u32,
    pub rows: &'a [SampleRow],
    /// `[rows]` [`SampleRow`]s: the device copy of `rows`.
    pub rows_dev: u64,
    /// f64 `[rows, n]`.
    pub probs: u64,
    /// `[rows]`.
    pub tokens: u64,
    /// One word.
    pub status: u64,
    /// The stream a token is drawn on; none computes the distributions only.
    pub draw: Option<Stream>,
}

/// One chain-acceptance launch (`sampling::chain_accept`, a sequence per row): row `r`
/// has `num_drafts[r] = k` drafts, target distributions `target_row[r] ..= + k` of
/// `target`, and draft `i`'s distribution and id at row `draft_row[r] + i · draft_step`
/// of `draft` and of `drafts`. It writes `1 ..= k + 1` tokens to `out[r · stride ..]` and
/// their count to `counts[r]`.
#[derive(Clone, Copy, Debug)]
pub struct AcceptArgs {
    /// f64 `[*, n]`.
    pub target: u64,
    /// f64 `[*, n]`.
    pub draft: u64,
    pub n: u32,
    /// `[rows]` [`SampleRow`]s, `position` the first drafted position.
    pub rows: u64,
    pub target_row: u64,
    pub draft_row: u64,
    pub draft_step: u32,
    pub num_drafts: u64,
    pub drafts: u64,
    pub stride: u32,
    /// f64 `[rows, n]`.
    pub scratch: u64,
    pub out: u64,
    pub counts: u64,
    pub status: u64,
    pub num_rows: u32,
}

/// A device's memory and launches. Every operation is queued in order;
/// [`Device::download`] waits for everything before it.
pub trait Device {
    /// `bytes` zeroed bytes at an address aligned to 8, and what keeps them allocated.
    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)>;
    /// # Safety
    ///
    /// `addr .. addr + bytes` lies inside one live allocation of this device.
    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()>;
    /// # Safety
    ///
    /// As [`Device::zero`], for `src.len()` bytes.
    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()>;
    /// # Safety
    ///
    /// As [`Device::zero`], for `dst.len()` bytes.
    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()>;
    /// # Safety
    ///
    /// Both ranges lie inside live allocations of this device and do not overlap.
    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()>;
    /// XORs `words` 32-bit words at `addr` with the keystream of (`key`, `nonce`) from
    /// block 0 (`eidola_engine::kv_cipher`), keeping no copy of the key.
    ///
    /// # Safety
    ///
    /// As [`Device::zero`], for `4 · words` bytes at a word-aligned address.
    unsafe fn cipher(&self, addr: u64, words: u32, key: &KvKey, nonce: &KvNonce) -> Result<()>;
    /// # Safety
    ///
    /// Every address of `a` holds what its field documents.
    unsafe fn sample(&self, a: &SampleArgs<'_>) -> Result<()>;
    /// # Safety
    ///
    /// Every address of `a` holds what its field documents.
    unsafe fn accept(&self, a: &AcceptArgs) -> Result<()>;
    /// `bytes` zeroed bytes of host memory for the host tier, and what keeps them
    /// allocated. On a CUDA device the memory is page-locked: never paged out.
    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)>;
    /// Queues a copy of device bytes to host memory.
    ///
    /// # Safety
    ///
    /// `src` as [`Device::zero`]; `dst .. dst + bytes` lies inside one live
    /// [`Device::alloc_host`] allocation that nothing else reads or writes until
    /// [`Device::synchronize`].
    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()>;
    /// Queues a copy of host memory to device bytes.
    ///
    /// # Safety
    ///
    /// As [`Device::copy_to_host`], the other way.
    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()>;
    /// Zeroes host memory; the caller has synchronized.
    ///
    /// # Safety
    ///
    /// `dst .. dst + bytes` lies inside one live [`Device::alloc_host`] allocation.
    unsafe fn zero_host(&self, dst: *mut u8, bytes: usize) {
        // SAFETY: the caller's range.
        unsafe { std::ptr::write_bytes(dst, 0, bytes) }
    }
    /// Copies host memory out; the caller has synchronized.
    ///
    /// # Safety
    ///
    /// As [`Device::zero_host`], for `dst.len()` bytes at `src`.
    unsafe fn read_host(&self, src: *const u8, dst: &mut [u8]) {
        // SAFETY: the caller's range.
        dst.copy_from_slice(unsafe { std::slice::from_raw_parts(src, dst.len()) });
    }
    fn synchronize(&self) -> Result<()>;
    fn kernels(&self) -> &dyn DsKernels;
    /// Total device memory, when the device has a fixed amount.
    fn capacity(&self) -> Option<usize>;
}

/// An allocation of a half, freed when dropped.
pub struct Buffer {
    pub name: &'static str,
    pub kind: Kind,
    pub addr: u64,
    pub bytes: usize,
    _owner: Box<dyn Any>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Buffer({} {:?} {} bytes)",
            self.name, self.kind, self.bytes
        )
    }
}

/// Element types whose in-memory bytes are what a kernel reads: no padding, every bit
/// pattern valid.
pub trait Plain: Copy {}
impl Plain for u8 {}
impl Plain for u16 {}
impl Plain for u32 {}
impl Plain for i32 {}
impl Plain for u64 {}
impl Plain for f32 {}
impl Plain for f64 {}
impl Plain for SampleRow {}

/// `x` as its bytes.
pub fn bytes_of<T: Plain>(x: &[T]) -> &[u8] {
    // SAFETY: `T: Plain` has no padding; the slice covers exactly these bytes.
    unsafe { std::slice::from_raw_parts(x.as_ptr().cast(), std::mem::size_of_val(x)) }
}

fn bytes_of_mut<T: Plain>(x: &mut [T]) -> &mut [u8] {
    // SAFETY: as `bytes_of`, and every bit pattern is a valid `T`.
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr().cast(), std::mem::size_of_val(x)) }
}

impl Buffer {
    pub fn new(dev: &dyn Device, name: &'static str, kind: Kind, bytes: usize) -> Result<Buffer> {
        let (addr, owner) = dev.alloc(bytes.max(8))?;
        Ok(Buffer {
            name,
            kind,
            addr,
            bytes,
            _owner: owner,
        })
    }

    /// The address `offset` bytes in; `offset + bytes` must fit.
    pub fn at(&self, offset: usize, bytes: usize) -> u64 {
        assert!(
            offset.checked_add(bytes).is_some_and(|e| e <= self.bytes),
            "{}: {offset} + {bytes} past {}",
            self.name,
            self.bytes
        );
        self.addr + offset as u64
    }

    pub fn write<T: Plain>(&self, dev: &dyn Device, offset: usize, src: &[T]) -> Result<()> {
        let src = bytes_of(src);
        if src.is_empty() {
            return Ok(());
        }
        // SAFETY: `at` holds the range inside this allocation.
        unsafe { dev.upload(self.at(offset, src.len()), src) }
    }

    pub fn read<T: Plain + Default>(
        &self,
        dev: &dyn Device,
        offset: usize,
        len: usize,
    ) -> Result<Vec<T>> {
        let mut out = vec![T::default(); len];
        if len > 0 {
            let dst = bytes_of_mut(&mut out);
            // SAFETY: `at` holds the range inside this allocation.
            unsafe { dev.download(self.at(offset, dst.len()), dst) }?;
        }
        Ok(out)
    }

    pub fn clear(&self, dev: &dyn Device) -> Result<()> {
        // SAFETY: the whole allocation.
        unsafe { dev.zero(self.addr, self.bytes) }
    }
}

/// Bytes of one host-to-device copy.
const UPLOAD_PIECE: usize = 1 << 28;

/// A half on a CUDA device: its own stream of an opened device, the sampler and cipher
/// images, and the forward's kernel images behind their launch wrappers.
pub struct CudaDevice {
    gpu: Gpu,
    sampler: Sampler,
    cipher: Mutex<KvCipher>,
    ops: DsOps,
    gemm: DsGemmRef,
    head: DsHead,
    attention: DsAttention,
    index: DsIndex,
}

impl CudaDevice {
    /// Opens device `ordinal` with a fresh stream and loads and verifies the images
    /// every half needs.
    pub fn open(ordinal: usize, dir: &KernelDir) -> Result<CudaDevice> {
        let gpu = Gpu::open(ordinal)?;
        Ok(CudaDevice {
            sampler: Sampler::load(&gpu, dir)?,
            cipher: Mutex::new(KvCipher::load(&gpu, dir)?),
            ops: DsOps::load(&gpu, dir)?,
            gemm: DsGemmRef::load(&gpu, dir)?,
            head: DsHead::load(&gpu, dir)?,
            attention: DsAttention::load(&gpu, dir)?,
            index: DsIndex::load(&gpu, dir)?,
            gpu,
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn bind(&self) -> Result<()> {
        self.gpu.context().bind_to_thread()?;
        Ok(())
    }
}

impl Device for CudaDevice {
    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        let s = self.gpu.stream();
        let buf = s.alloc_zeros::<u64>(bytes.div_ceil(8))?;
        Ok((dptr(&buf, s), Box::new(buf)))
    }

    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's range.
        unsafe { sys::cuMemsetD8Async(addr, 0, bytes, self.gpu.stream().cu_stream()) }.result()?;
        Ok(())
    }

    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
        self.bind()?;
        // In pieces: a weight tensor may be tens of gigabytes of mapped file.
        let mut at = addr;
        for piece in src.chunks(UPLOAD_PIECE) {
            // SAFETY: the caller's range; a copy from pageable memory has left `piece`
            // when the call returns.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    at,
                    piece.as_ptr().cast(),
                    piece.len(),
                    self.gpu.stream().cu_stream(),
                )
            }
            .result()?;
            at += piece.len() as u64;
        }
        Ok(())
    }

    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's range; the stream is drained before `dst` is read.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr().cast(),
                addr,
                dst.len(),
                self.gpu.stream().cu_stream(),
            )
        }
        .result()?;
        self.gpu.synchronize()
    }

    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's ranges.
        unsafe { sys::cuMemcpyDtoDAsync_v2(dst, src, bytes, self.gpu.stream().cu_stream()) }
            .result()?;
        Ok(())
    }

    unsafe fn cipher(&self, addr: u64, words: u32, key: &KvKey, nonce: &KvNonce) -> Result<()> {
        let mut items = CipherItems::with_capacity(1);
        items.push(CipherItem::new(addr, words, 0, key, nonce));
        let mut cipher = self
            .cipher
            .lock()
            .map_err(|_| CudaError::new("cipher lock"))?;
        // SAFETY: one range, the caller's.
        unsafe { cipher.apply(&self.gpu, &mut items) }
    }

    unsafe fn sample(&self, a: &SampleArgs<'_>) -> Result<()> {
        if a.rows.iter().any(|r| r.logit_row >= a.logit_rows) {
            return Err(CudaError::new("sample: a logit row outside the step's"));
        }
        // SAFETY: the caller's addresses; the rows were checked above.
        unsafe {
            self.upload(a.rows_dev, bytes_of(a.rows))?;
            self.sampler.launch_sample(
                &self.gpu,
                SampleLaunch {
                    logits: a.logits,
                    logits_stride: u64::from(a.logits_stride),
                    n: a.n,
                    rows: a.rows_dev,
                    num_rows: narrow(a.rows.len(), "sample rows")?,
                    draw: a.draw,
                    probs: a.probs,
                    tokens: a.tokens,
                    status: a.status,
                },
            )
        }
    }

    unsafe fn accept(&self, a: &AcceptArgs) -> Result<()> {
        // SAFETY: the caller's addresses.
        unsafe {
            self.sampler.launch_accept(
                &self.gpu,
                crate::sampler::AcceptLaunch {
                    target: a.target,
                    draft: a.draft,
                    n: a.n,
                    rows: a.rows,
                    target_row: a.target_row,
                    draft_row: a.draft_row,
                    draft_step: a.draft_step,
                    num_drafts: a.num_drafts,
                    drafts: a.drafts,
                    stride: a.stride,
                    scratch: a.scratch,
                    out: a.out,
                    counts: a.counts,
                    status: a.status,
                    num_rows: a.num_rows,
                },
            )
        }
    }

    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)> {
        // SAFETY: the memory is zeroed below, before anything reads it.
        let mut mem = unsafe { self.gpu.context().alloc_pinned_with_flags::<u8>(bytes, 0) }?;
        let base = mem.as_mut_ptr()?;
        // SAFETY: the start of the `bytes`-long allocation just made.
        unsafe { std::ptr::write_bytes(base, 0, bytes) };
        Ok((base, Box::new(mem)))
    }

    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: the caller's ranges; the host range is page-locked and left alone
        // until the stream drains.
        unsafe { sys::cuMemcpyDtoHAsync_v2(dst.cast(), src, bytes, self.gpu.stream().cu_stream()) }
            .result()?;
        Ok(())
    }

    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: as `copy_to_host`.
        unsafe { sys::cuMemcpyHtoDAsync_v2(dst, src.cast(), bytes, self.gpu.stream().cu_stream()) }
            .result()?;
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.gpu.synchronize()
    }

    fn kernels(&self) -> &dyn DsKernels {
        self
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.gpu.info().total_mem_bytes)
    }
}

/// Each family is its wrapper's launch: the wrapper checks the struct and picks the grid.
macro_rules! launches {
    ($($name:ident: $params:ty => $field:ident . $method:ident),* $(,)?) => {
        impl DsKernels for CudaDevice {
            $(
                unsafe fn $name(&self, p: &$params) -> Result<()> {
                    // SAFETY: the caller's contract is the wrapper's.
                    unsafe { self.$field.$method(&self.gpu, p) }
                }
            )*
        }
    };
}

launches! {
    embed: EmbedParams => ops.embed,
    quant_fp8: QuantFp8Params => ops.quant_fp8,
    rot_quant: RotQuantParams => ops.rot_quant,
    rms_norm: RmsNormParams => ops.rms_norm,
    mhc_mix: MhcMixParams => ops.mhc_mix,
    collapse: CollapseParams => ops.collapse,
    expand: ExpandParams => ops.expand,
    lin8: Lin8Params => gemm.lin8,
    lin4: Lin4Params => gemm.lin4,
    grouped_dot: GroupedDotParams => gemm.grouped_dot,
    attention: AttnParams => attention.attend,
    linear: LinearParams => ops.linear,
    head: LinearParams => head.head,
    pool: PoolParams => ops.pool,
    index_score: IndexScoreParams => index.score,
    select: SelectParams => index.select,
    stream_mean: StreamMeanParams => ops.stream_mean,
    route: RouteParams => ops.route,
    place: PlaceParams => ops.place,
    place_rows: PlaceRowsParams => ops.place_rows,
    swiglu: SwigluParams => ops.swiglu,
    combine: CombineParams => ops.combine,
    engram_gather: EngramGatherParams => ops.engram_gather,
    engram_inject: EngramInjectParams => ops.engram_inject,
}
