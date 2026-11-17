//! [`HostDevice`]: host memory, each kernel launch run by a host function.
//!
//! The executor's tests give it the kernels' own bodies compiled for the host
//! (`oracle/qwen_launch_host.cc`): every launch first goes through its family's grid
//! function ([`super::kernels`]), exactly as a device launch does, and then runs the body
//! over that grid. So without a GPU the executor's scenarios exercise the structs it
//! fills, the wrappers' preconditions, the grids and the C++ a GPU compiles. It computes
//! only the exact profile. The row copy, the sampler and chain acceptance have no host
//! body: here they are this device's reading of `eidola_copy_rows` (every index bounded,
//! as the kernel bounds it) and of `eidola_sample` and `eidola_chain_accept`
//! (`eidola_engine::sampling`, which the device sampler equals bit for bit given equal
//! logits). It captures no graphs: the executor's padded launch sequence runs on it
//! directly.

use std::any::Any;
use std::ffi::c_void;

use eidola_engine::kv_cipher::{KvKeystream, KvNonce};
use eidola_engine::sampling::{self, Logits, SamplingParams};
use eidola_engine::secret::KvKey;

use super::abi::*;
use super::attention::Fa2Args;
use super::device::{QwenDevice, QwenProfile};
use super::kernels::*;
use crate::sampler::{STATUS_BAD_TOKEN as SAMPLE_BAD_TOKEN, STATUS_NON_FINITE, SampleRow};
use crate::{CudaError, Result};

/// Runs one launch: the family's name (the oracle's `qwen_host_<family>`), its parameter
/// struct, and the grid's x and y.
pub type HostLaunch = dyn Fn(&'static str, *const c_void, u32, u32) + Send + Sync;

/// Host memory with a host function per kernel family.
pub struct HostDevice {
    launch: Box<HostLaunch>,
    profile: QwenProfile,
}

impl HostDevice {
    pub fn new(launch: Box<HostLaunch>) -> HostDevice {
        HostDevice {
            launch,
            profile: QwenProfile::Exact,
        }
    }

    /// A device that plans as the fast profile does, for the plans' tests: every
    /// launch but FA2 computes as in the exact profile, and each FA2 launch is checked
    /// (its plan against its layer and scratch) and computes nothing, leaving its output
    /// as it was.
    pub fn planning_fast(launch: Box<HostLaunch>) -> HostDevice {
        HostDevice {
            launch,
            profile: QwenProfile::Fast,
        }
    }

    fn run<T>(&self, family: &'static str, p: &T, grid: Result<[u32; 3]>) -> Result<()> {
        let g = grid?;
        (self.launch)(family, (p as *const T).cast(), g[0], g[1]);
        Ok(())
    }
}

/// A host slice at `addr`.
///
/// # Safety
///
/// `addr .. addr + len · size_of::<T>()` lies inside one live allocation, aligned for
/// `T`, and nothing else holds a reference into it.
unsafe fn wr<'a, T>(addr: u64, len: usize) -> &'a mut [T] {
    // SAFETY: the caller's range.
    unsafe { std::slice::from_raw_parts_mut(addr as *mut T, len) }
}

impl QwenKernels for HostDevice {
    unsafe fn embed(&self, p: &EmbedParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("embed", p, embed_grid(p))
    }
    unsafe fn norm(&self, p: &NormParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("norm", p, norm_grid(p))
    }
    unsafe fn quant(&self, p: &QuantParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("quant", p, quant_grid(p))
    }
    unsafe fn lin8(&self, p: &Lin8Params) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("lin8", p, lin8_grid(p))
    }
    unsafe fn linear(&self, p: &LinearParams, _fast: Option<u64>) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("linear", p, linear_grid(p))
    }
    unsafe fn gate(&self, p: &GateParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("gate", p, gate_grid(p))
    }
    unsafe fn kv_write(&self, p: &KvWriteParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("kv_write", p, kv_write_grid(p))
    }
    unsafe fn attention(&self, p: &AttnParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("attn", p, attention_grid(p))
    }
    unsafe fn dn_prep(&self, p: &DnPrepParams) -> Result<()> {
        if p.positions == 0 {
            return Ok(());
        }
        self.run("dn_prep", p, dn_prep_grid(p))
    }
    unsafe fn dn_rec(&self, p: &DnRecParams) -> Result<()> {
        if p.seq_count == 0 {
            return Ok(());
        }
        self.run("dn_rec", p, dn_rec_grid(p))
    }
    unsafe fn dn_norm(&self, p: &DnNormParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("dn_norm", p, dn_norm_grid(p))
    }
    unsafe fn widen(&self, p: &WidenParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("widen", p, widen_grid(p))
    }
    unsafe fn round(&self, p: &RoundParams) -> Result<()> {
        if p.rows == 0 {
            return Ok(());
        }
        self.run("round", p, round_grid(p))
    }
    unsafe fn copy_rows(&self, a: &CopyArgs) -> Result<()> {
        copy_grid(a.items, a.width)?;
        let w = a.width as usize;
        for i in 0..a.items as usize {
            // SAFETY: the caller's item arrays of `items` words and buffers of
            // `rows[b] · width` words; every index is bounded below as the kernel does.
            unsafe {
                let sb = rd::<u32>(a.src_buf, i) as usize;
                let db = rd::<u32>(a.dst_buf, i) as usize;
                let sr = u64::from(rd::<u32>(a.src_row, i));
                let dr = u64::from(rd::<u32>(a.dst_row, i));
                if sb >= COPY_BUFFERS || db >= COPY_BUFFERS || sr >= a.rows[sb] || dr >= a.rows[db]
                {
                    *wr::<u32>(a.status, 1).first_mut().expect("a word") |= STATUS_BAD_INDEX;
                    continue;
                }
                let src = a.buf[sb] + sr * w as u64 * 4;
                let dst = a.buf[db] + dr * w as u64 * 4;
                std::ptr::copy(src as *const u32, dst as *mut u32, w);
            }
        }
        Ok(())
    }
    unsafe fn sample(&self, a: &SampleArgs) -> Result<()> {
        let n = a.n as usize;
        if a.num_rows == 0 {
            return Ok(());
        }
        if n == 0 || a.n as u64 > a.logits_stride || a.n > 1 << 20 {
            return Err(CudaError::new(format!("sample: vocabulary {n}")));
        }
        for r in 0..a.num_rows as usize {
            // SAFETY: the caller's rows, logits, distributions and tokens.
            unsafe {
                let row = rd::<SampleRow>(a.rows, r);
                let params = params_of(&row)?;
                let at = a.logits + u64::from(row.logit_row) * a.logits_stride * 4;
                let logits = wr::<f32>(at, n);
                if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY)
                    || (!params.is_greedy() && logits.iter().all(|v| *v == f32::NEG_INFINITY))
                {
                    *wr::<u32>(a.status, 1).first_mut().expect("a word") |= STATUS_NON_FINITE;
                    continue;
                }
                let l = Logits::new(logits, a.n);
                let probs = sampling::processed_probs(l, &params);
                wr::<f64>(a.probs + (r * n * 8) as u64, n).copy_from_slice(&probs);
                if let Some(stream) = a.draw {
                    let t = if params.is_greedy() {
                        sampling::argmax(l)
                    } else {
                        let u = sampling::uniform(params.seed(), u64::from(row.position), stream);
                        sampling::sample_from(&probs, u)
                    };
                    wr::<u32>(a.tokens + (r * 4) as u64, 1)[0] = t;
                }
            }
        }
        Ok(())
    }
    unsafe fn accept(&self, a: &AcceptLaunch) -> Result<()> {
        let n = a.n as usize;
        for r in 0..a.num_rows as usize {
            // SAFETY: the caller's rows, distributions, ids and outputs.
            unsafe {
                let row = rd::<SampleRow>(a.rows, r);
                let params = params_of(&row)?;
                let k = rd::<u32>(a.num_drafts, r) as usize;
                let t0 = rd::<u32>(a.target_row, r) as usize;
                let d0 = rd::<u32>(a.draft_row, r) as usize;
                let step = a.draft_step as usize;
                let dist = |base: u64, i: usize| wr::<f64>(base + (i * n * 8) as u64, n).to_vec();
                let mut ids = Vec::with_capacity(k);
                let mut bad = false;
                for i in 0..k {
                    let id = rd::<u32>(a.drafts, d0 + i * step);
                    bad |= id >= a.n;
                    ids.push(id);
                }
                let count = wr::<u32>(a.counts + (r * 4) as u64, 1);
                if bad {
                    *wr::<u32>(a.status, 1).first_mut().expect("a word") |= SAMPLE_BAD_TOKEN;
                    count[0] = 0;
                    continue;
                }
                let target: Vec<Vec<f64>> = (0..=k).map(|i| dist(a.target, t0 + i)).collect();
                let draft: Vec<Vec<f64>> = (0..k).map(|i| dist(a.draft, d0 + i * step)).collect();
                let out =
                    sampling::chain_accept(&target, &draft, &ids, &params, u64::from(row.position));
                let dst = wr::<u32>(a.out + (r * a.stride as usize * 4) as u64, out.len());
                dst.copy_from_slice(&out);
                count[0] = u32::try_from(out.len()).expect("at most k + 1 tokens");
            }
        }
        Ok(())
    }
    unsafe fn fa2(&self, a: &Fa2Args) -> Result<()> {
        if self.profile != QwenProfile::Fast {
            return Err(CudaError::new(
                "FA2 has no host body: the host device runs the exact profile",
            ));
        }
        let s = &a.view.shape;
        let l = &a.layer;
        let ok = s.split
            && l.window_left < 0
            && (s.tile == 16 || s.tile == 64)
            && l.num_kv_heads > 0
            && a.num_qo_heads / l.num_kv_heads == s.group_size
            && l.page_size == s.page_size
            && s.partial_rows <= a.partials.rows
            && a.partials.heads == a.num_qo_heads;
        if !ok {
            return Err(CudaError::new(format!("FA2 launch refused: {a:?}")));
        }
        // SAFETY: the view's work-item count word, in the table.
        let work = unsafe { rd::<u32>(a.view.work_items, 0) };
        if work > s.grid.max(1) * 1_000_000 {
            return Err(CudaError::new("FA2 work items past any plan"));
        }
        Ok(())
    }
}

/// Element `i` of a host array at `addr`.
///
/// # Safety
///
/// `addr` holds at least `i + 1` values of `T`, aligned.
unsafe fn rd<T: Copy>(addr: u64, i: usize) -> T {
    // SAFETY: the caller's.
    unsafe { *(addr as *const T).add(i) }
}

fn params_of(row: &SampleRow) -> Result<SamplingParams> {
    SamplingParams::new(row.temperature, row.top_k, row.top_p, row.min_p, row.seed)
        .map_err(|e| CudaError::new(format!("a sample row's parameters: {e:?}")))
}

/// Host allocations are aligned to this, as device ones are.
const ALIGN: usize = 256;

struct HostBuf(Vec<u8>);

fn host_alloc(bytes: usize) -> (u64, Box<dyn Any>) {
    let buf = HostBuf(vec![0u8; bytes + ALIGN]);
    let base = buf.0.as_ptr() as usize;
    let addr = base.next_multiple_of(ALIGN);
    (addr as u64, Box::new(buf))
}

impl QwenDevice for HostDevice {
    fn profile(&self) -> QwenProfile {
        self.profile
    }

    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        Ok(host_alloc(bytes))
    }

    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller's range, for this and every method below.
        unsafe { wr::<u8>(addr, bytes).fill(0) };
        Ok(())
    }

    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
        unsafe { wr::<u8>(addr, src.len()).copy_from_slice(src) };
        Ok(())
    }

    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        dst.copy_from_slice(unsafe { wr::<u8>(addr, dst.len()) });
        Ok(())
    }

    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        // SAFETY: non-overlapping ranges of live allocations (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes) };
        Ok(())
    }

    unsafe fn cipher(
        &self,
        addr: u64,
        words: u32,
        counter: u32,
        key: &KvKey,
        nonce: &KvNonce,
    ) -> Result<()> {
        KvKeystream::with_counter(key, nonce, counter)
            .apply_words(unsafe { wr::<u32>(addr, words as usize) });
        Ok(())
    }

    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)> {
        let (addr, keep) = host_alloc(bytes);
        Ok((addr as *mut u8, keep))
    }

    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst, bytes) };
        Ok(())
    }

    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        unsafe { std::ptr::copy_nonoverlapping(src, dst as *mut u8, bytes) };
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }

    fn capture(&self, _run: &mut dyn FnMut() -> Result<()>) -> Result<Box<dyn Any>> {
        Err(CudaError::new("the host device captures no graphs"))
    }

    fn replay(&self, _graph: &dyn Any) -> Result<()> {
        Err(CudaError::new("the host device replays no graphs"))
    }

    fn free_memory(&self) -> Result<i64> {
        Ok(0)
    }
}
