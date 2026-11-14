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

use super::kernels::{DsKernels, Linked};
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

/// A half on a CUDA device: its own stream of an opened device, the sampler and cipher
/// images, and the forward's kernels.
pub struct CudaDevice {
    gpu: Gpu,
    sampler: Sampler,
    cipher: Mutex<KvCipher>,
    kernels: Box<dyn DsKernels>,
}

impl CudaDevice {
    /// Opens device `ordinal` with a fresh stream and loads the images every half
    /// needs. The forward's kernels are [`Linked`].
    pub fn open(ordinal: usize, dir: &KernelDir) -> Result<CudaDevice> {
        let gpu = Gpu::open(ordinal)?;
        Ok(CudaDevice {
            sampler: Sampler::load(&gpu, dir)?,
            cipher: Mutex::new(KvCipher::load(&gpu, dir)?),
            kernels: Box::new(Linked),
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
        // SAFETY: the caller's range; a copy from pageable memory has left `src` when
        // the call returns.
        unsafe {
            sys::cuMemcpyHtoDAsync_v2(
                addr,
                src.as_ptr().cast(),
                src.len(),
                self.gpu.stream().cu_stream(),
            )
        }
        .result()?;
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
                    draw: Some(Stream::Sample),
                    probs: a.probs,
                    tokens: a.tokens,
                    status: a.status,
                },
            )
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.gpu.synchronize()
    }

    fn kernels(&self) -> &dyn DsKernels {
        self.kernels.as_ref()
    }

    fn capacity(&self) -> Option<usize> {
        Some(self.gpu.info().total_mem_bytes)
    }
}
