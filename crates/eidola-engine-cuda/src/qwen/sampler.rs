//! Qwen3.8's sampler and chain acceptance on the device (`qwen_sampling.cu`): the serving
//! core's `sampling` semantics bit for bit given equal logits, as the shared sampler
//! (`crate::sampler`) is, spread over the GPU and selecting top-k and top-p from a short
//! list of candidates (`eidola-engine-cuda/AGENTS.md` → Qwen3.8).
//!
//! A sample launch is four kernels over a work area of [`SAMPLE_WORK_ROW`] bytes a row
//! ([`sample_work_bytes`]); chain acceptance is three over [`ACCEPT_WORK_ROW`] bytes a
//! sequence ([`accept_work_bytes`]), in place of the shared kernel's residual rows.

use super::kernels::{AcceptLaunch, SampleArgs};
use crate::module::{Kernel, KernelDir, KernelModule};
use crate::{CudaError, Gpu, Result, launch};

/// The image.
pub const IMAGE: &str = "qwen_sampling";

/// Ids a row's chunked sums cover (`kMaxChunks` chunks of 1,024).
pub const MAX_VOCAB: u32 = 256 * 1024;

/// Bytes of one sample row's work area (`QwSampleWork`).
pub const SAMPLE_WORK_ROW: usize = 20_608;

/// Bytes of one sequence's acceptance work (`QwAcceptWork`).
pub const ACCEPT_WORK_ROW: usize = 2_064;

/// Ids a block of the chunk kernels covers (`SUM_CHUNK`).
const CHUNK: u32 = 1024;

/// Threads of `eidola_qwen_accept_decide`'s blocks, a sequence each.
const DECIDE_THREADS: u32 = 128;

/// The driver's grid limit in y.
const MAX_GRID_Y: u32 = 65_535;

/// The work area of a sample launch over `rows` rows.
pub fn sample_work_bytes(rows: usize) -> usize {
    rows.max(1) * SAMPLE_WORK_ROW
}

/// The work area of chain acceptance over `rows` sequences.
pub fn accept_work_bytes(rows: usize) -> usize {
    rows.max(1) * ACCEPT_WORK_ROW
}

/// Refuses a sample launch the kernels cannot take: a vocabulary past [`MAX_VOCAB`] or the
/// logits' stride, a missing work area, more rows than a grid's y.
pub fn sample_check(a: &SampleArgs) -> Result<()> {
    if a.n == 0 || u64::from(a.n) > a.logits_stride || a.n > MAX_VOCAB {
        return Err(CudaError::new(format!("sample: vocabulary {}", a.n)));
    }
    if a.work == 0 || a.num_rows > MAX_GRID_Y {
        return Err(CudaError::new(
            "sample: no work area, or rows past the grid",
        ));
    }
    Ok(())
}

/// Refuses an acceptance launch the kernels cannot take.
pub fn accept_check(a: &AcceptLaunch) -> Result<()> {
    if a.n == 0 || a.n > MAX_VOCAB {
        return Err(CudaError::new(format!("chain_accept: vocabulary {}", a.n)));
    }
    if a.scratch == 0 || a.num_rows > MAX_GRID_Y {
        return Err(CudaError::new(
            "chain_accept: no work area, or rows past the grid",
        ));
    }
    Ok(())
}

/// The sampler's kernels.
pub struct QwenSampler {
    _module: KernelModule,
    top: Kernel,
    exp: Kernel,
    norm: Kernel,
    pick: Kernel,
    decide: Kernel,
    sums: Kernel,
    finish: Kernel,
}

impl QwenSampler {
    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<QwenSampler> {
        QwenSampler::from_module(KernelModule::load(gpu, dir, IMAGE)?)
    }

    pub fn from_module(module: KernelModule) -> Result<QwenSampler> {
        let k = |s: &str| module.bound_kernel(IMAGE, s);
        Ok(QwenSampler {
            top: k("eidola_qwen_sample_top")?,
            exp: k("eidola_qwen_sample_exp")?,
            norm: k("eidola_qwen_sample_norm")?,
            pick: k("eidola_qwen_sample_pick")?,
            decide: k("eidola_qwen_accept_decide")?,
            sums: k("eidola_qwen_accept_sums")?,
            finish: k("eidola_qwen_accept_finish")?,
            _module: module,
        })
    }

    /// For each of `a.num_rows` rows: the processed distribution into `a.probs[r · n ..]`
    /// and, with `a.draw`, a token drawn from it into `a.tokens[r]`, as the shared
    /// sampler's `eidola_sample`.
    ///
    /// # Safety
    ///
    /// As [`super::kernels::QwenKernels::sample`]: every row's logit row inside
    /// `a.logits`, `a.probs` holding `num_rows · n` values, `a.tokens` `num_rows` (when
    /// drawing), `a.work` [`sample_work_bytes`]`(num_rows)` bytes and `a.status` a word.
    pub unsafe fn sample(&self, gpu: &Gpu, a: &SampleArgs) -> Result<()> {
        if a.num_rows == 0 {
            return Ok(());
        }
        sample_check(a)?;
        let stream = a.draw.map_or(u32::MAX, |d| d as u32);
        let chunks = a.n.div_ceil(CHUNK);
        let rows = a.num_rows;
        // SAFETY: arguments match the kernels; the caller's contract covers every address.
        unsafe {
            launch!(
                gpu,
                self.top,
                [rows, 1, 1],
                a.logits,
                a.logits_stride,
                a.n,
                a.rows,
                stream,
                a.tokens,
                a.work,
                a.status,
            )?;
            launch!(
                gpu,
                self.exp,
                [chunks, rows, 1],
                a.logits,
                a.logits_stride,
                a.n,
                a.rows,
                a.probs,
                a.work,
            )?;
            launch!(
                gpu,
                self.norm,
                [chunks, rows, 1],
                a.n,
                a.rows,
                a.probs,
                a.work
            )?;
            launch!(
                gpu,
                self.pick,
                [rows, 1, 1],
                a.n,
                a.rows,
                stream,
                a.probs,
                a.tokens,
                a.work,
                a.status,
            )
        }
    }

    /// Chain speculative acceptance (`sampling::chain_accept`), as the shared
    /// `eidola_chain_accept`, with `a.scratch` the work area.
    ///
    /// # Safety
    ///
    /// As [`crate::sampler::Sampler::launch_accept`], with `a.scratch` holding
    /// [`accept_work_bytes`]`(num_rows)` bytes.
    pub unsafe fn accept(&self, gpu: &Gpu, a: &AcceptLaunch) -> Result<()> {
        if a.num_rows == 0 {
            return Ok(());
        }
        accept_check(a)?;
        let chunks = a.n.div_ceil(CHUNK);
        let rows = a.num_rows;
        // SAFETY: arguments match the kernels; the caller's contract covers every address.
        unsafe {
            launch!(
                gpu,
                self.decide,
                [rows.div_ceil(DECIDE_THREADS), 1, 1],
                a.target,
                a.draft,
                a.n,
                a.rows,
                a.target_row,
                a.draft_row,
                a.draft_step,
                a.num_drafts,
                a.drafts,
                a.stride,
                a.scratch,
                a.out,
                a.counts,
                a.status,
                rows,
            )?;
            launch!(
                gpu,
                self.sums,
                [chunks, rows, 1],
                a.target,
                a.draft,
                a.n,
                a.rows,
                a.target_row,
                a.draft_row,
                a.draft_step,
                a.scratch,
            )?;
            launch!(
                gpu,
                self.finish,
                [rows, 1, 1],
                a.target,
                a.draft,
                a.n,
                a.rows,
                a.target_row,
                a.draft_row,
                a.draft_step,
                a.stride,
                a.scratch,
                a.out,
                a.counts,
            )
        }
    }
}
