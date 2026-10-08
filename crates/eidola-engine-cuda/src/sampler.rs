//! The device sampler (`sampling.cu`): the serving core's `sampling` semantics,
//! bit for bit given equal logits.

use crate::module::KernelDir;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use eidola_engine::sampling::{SamplingParams, Stream};

use crate::launch::dptr;
use crate::module::{Kernel, KernelModule};
use crate::{CudaError, Gpu, Result, launch};

/// One row's sampling parameters as the kernels read them (`EidolaSampleRow`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SampleRow {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    /// The position of the token being chosen (the first drafted position for
    /// chain acceptance).
    pub position: u32,
    /// The logits row this row samples from.
    pub logit_row: u32,
}

const _: () = assert!(std::mem::size_of::<SampleRow>() == 32);

// SAFETY: plain `repr(C)` data with the kernel's layout; all-zero bits are a
// valid (greedy) row.
unsafe impl DeviceRepr for SampleRow {}
unsafe impl ValidAsZeroBits for SampleRow {}

impl SampleRow {
    pub fn new(params: &SamplingParams, position: u32, logit_row: u32) -> SampleRow {
        SampleRow {
            temperature: params.temperature(),
            top_k: params.top_k(),
            top_p: params.top_p(),
            min_p: params.min_p(),
            seed: params.seed(),
            position,
            logit_row,
        }
    }
}

/// The status bit a row with NaN, `+inf` or (when not greedy) only `-inf`
/// logits sets.
pub const STATUS_NON_FINITE: u32 = 1;

/// The sampler kernels.
pub struct Sampler {
    _module: KernelModule,
    sample: Kernel,
    accept: Kernel,
}

impl Sampler {
    pub fn load(gpu: &Gpu, dir: &KernelDir) -> Result<Sampler> {
        Sampler::from_module(KernelModule::load(gpu, dir, "sampling")?)
    }

    pub fn from_module(module: KernelModule) -> Result<Sampler> {
        Ok(Sampler {
            sample: module.kernel("eidola_sample")?,
            accept: module.kernel("eidola_chain_accept")?,
            _module: module,
        })
    }

    /// For each of `num_rows` rows: the processed distribution over the first
    /// `n` logits of row `rows[r].logit_row` into `probs[r * n ..]`, and, with
    /// `draw`, a token drawn from it with that stream into `tokens[r]` (a
    /// greedy row takes the argmax and consumes no draw). Non-finite logits set
    /// [`STATUS_NON_FINITE`] in `status`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        gpu: &Gpu,
        logits: &CudaSlice<f32>,
        logits_stride: usize,
        n: u32,
        rows: &CudaSlice<SampleRow>,
        num_rows: u32,
        draw: Option<Stream>,
        probs: &mut CudaSlice<f64>,
        tokens: &mut CudaSlice<u32>,
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        if num_rows == 0 {
            return Ok(());
        }
        let r = num_rows as usize;
        if rows.len() < r || probs.len() < r * n as usize || tokens.len() < r || status.is_empty() {
            return Err(CudaError::new("sample: buffer too small"));
        }
        if n == 0 || n as usize > logits_stride || n > 1 << 20 || logits.len() < logits_stride {
            return Err(CudaError::new(format!("sample: vocabulary {n}")));
        }
        let s = gpu.stream();
        let stream = draw.map_or(u32::MAX, |d| d as u32);
        // SAFETY: arguments match `eidola_sample`; buffers checked above (the
        // logits rows named by `rows` are the caller's contract).
        unsafe {
            launch!(
                gpu,
                self.sample,
                [num_rows, 1, 1],
                dptr(logits, s),
                logits_stride as u64,
                n,
                dptr(rows, s),
                stream,
                dptr(probs, s),
                dptr(tokens, s),
                dptr(status, s),
            )
        }
    }

    /// Chain speculative acceptance for `num_rows` sequences: row `r` has
    /// `num_drafts[r]` drafts (`drafts[r * stride ..]`), target distributions
    /// `target[(target_row[r] + i) * n ..]` for `i` in `0..=k`, draft
    /// distributions `draft[(draft_row[r] + i) * n ..]` for `i` in `0..k`, and
    /// `rows[r].position` set to its first drafted position. Writes the tokens
    /// to `out[r * stride ..]` and their count to `counts[r]`.
    #[allow(clippy::too_many_arguments)]
    pub fn chain_accept(
        &self,
        gpu: &Gpu,
        target: &CudaSlice<f64>,
        draft: &CudaSlice<f64>,
        n: u32,
        rows: &CudaSlice<SampleRow>,
        target_row: &CudaSlice<u32>,
        draft_row: &CudaSlice<u32>,
        num_drafts: &CudaSlice<u32>,
        drafts: &CudaSlice<u32>,
        stride: u32,
        num_rows: u32,
        scratch: &mut CudaSlice<f64>,
        out: &mut CudaSlice<u32>,
        counts: &mut CudaSlice<u32>,
    ) -> Result<()> {
        if num_rows == 0 {
            return Ok(());
        }
        let r = num_rows as usize;
        // The kernel's chunked sums hold at most 1,024 chunks of 1,024.
        if n == 0 || n > 1 << 20 {
            return Err(CudaError::new(format!("chain_accept: vocabulary {n}")));
        }
        if stride == 0
            || drafts.len() < r * stride as usize
            || target_row.len() < r
            || draft_row.len() < r
            || num_drafts.len() < r
            || target.is_empty()
            || !target.len().is_multiple_of(n as usize)
            || !draft.len().is_multiple_of(n as usize)
        {
            return Err(CudaError::new(
                "chain_accept: row arrays too small or not whole rows",
            ));
        }
        if scratch.len() < r * n as usize
            || out.len() < r * stride as usize
            || counts.len() < r
            || rows.len() < r
        {
            return Err(CudaError::new("chain_accept: buffer too small"));
        }
        let s = gpu.stream();
        // SAFETY: arguments match `eidola_chain_accept`.
        unsafe {
            launch!(
                gpu,
                self.accept,
                [num_rows, 1, 1],
                dptr(target, s),
                dptr(draft, s),
                n,
                dptr(rows, s),
                dptr(target_row, s),
                dptr(draft_row, s),
                dptr(num_drafts, s),
                dptr(drafts, s),
                stride,
                dptr(scratch, s),
                dptr(out, s),
                dptr(counts, s),
            )
        }
    }
}
