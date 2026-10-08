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

    /// Chain speculative acceptance, one sequence per entry of `plan`: row
    /// `r` has `plan[r].drafts`, target distributions
    /// `target[(plan[r].target_row + i) * n ..]` for `i` in `0..=k`, draft
    /// distributions `draft[(plan[r].draft_row + i) * n ..]` for `i` in `0..k`,
    /// and `rows[r].position` set to its first drafted position. Writes the
    /// tokens to `out[r * stride ..]` and their count to `counts[r]`.
    ///
    /// Everything the kernel indexes by (draft counts, distribution rows,
    /// draft token ids) is checked here, on the host, against the buffers
    /// it will reach, and then uploaded into `inputs`: the device reads only
    /// values that passed.
    #[allow(clippy::too_many_arguments)]
    pub fn chain_accept(
        &self,
        gpu: &Gpu,
        target: &CudaSlice<f64>,
        draft: &CudaSlice<f64>,
        n: u32,
        rows: &CudaSlice<SampleRow>,
        plan: &[AcceptRow<'_>],
        inputs: &mut AcceptInputs,
        scratch: &mut CudaSlice<f64>,
        out: &mut CudaSlice<u32>,
        counts: &mut CudaSlice<u32>,
    ) -> Result<()> {
        if plan.is_empty() {
            return Ok(());
        }
        let r = plan.len();
        let stride = inputs.stride;
        // The kernel's chunked sums hold at most 1,024 chunks of 1,024.
        if n == 0 || n > 1 << 20 {
            return Err(CudaError::new(format!("chain_accept: vocabulary {n}")));
        }
        if !target.len().is_multiple_of(n as usize) || !draft.len().is_multiple_of(n as usize) {
            return Err(CudaError::new("chain_accept: distributions not whole rows"));
        }
        let (target_rows, draft_rows) = (target.len() / n as usize, draft.len() / n as usize);
        if r > inputs.rows
            || scratch.len() < r * n as usize
            || out.len() < r * stride as usize
            || counts.len() < r
            || rows.len() < r
        {
            return Err(CudaError::new("chain_accept: buffer too small"));
        }
        let mut drafts = vec![0u32; r * stride as usize];
        let (mut target_row, mut draft_row, mut num_drafts) = (
            Vec::with_capacity(r),
            Vec::with_capacity(r),
            Vec::with_capacity(r),
        );
        for (i, row) in plan.iter().enumerate() {
            let k = row.drafts.len();
            // Up to k accepted drafts and one more token: k + 1 slots.
            if k >= stride as usize {
                return Err(CudaError::new(format!(
                    "chain_accept: row {i} has {k} drafts, its output holds {stride} tokens"
                )));
            }
            // The kernel adds in u32; the sums must fit, and stay in bounds.
            let target_end = row.target_row.checked_add(k as u32 + 1);
            if target_end.is_none_or(|e| e as usize > target_rows) {
                return Err(CudaError::new(format!(
                    "chain_accept: row {i}'s target rows {}+{} beyond {target_rows}",
                    row.target_row,
                    k + 1
                )));
            }
            let draft_end = row.draft_row.checked_add(k as u32);
            if k > 0 && draft_end.is_none_or(|e| e as usize > draft_rows) {
                return Err(CudaError::new(format!(
                    "chain_accept: row {i}'s draft rows {}+{k} beyond {draft_rows}",
                    row.draft_row
                )));
            }
            if let Some(&d) = row.drafts.iter().find(|&&d| d >= n) {
                return Err(CudaError::new(format!(
                    "chain_accept: row {i} drafts token {d} of {n}"
                )));
            }
            drafts[i * stride as usize..][..k].copy_from_slice(row.drafts);
            target_row.push(row.target_row);
            draft_row.push(row.draft_row);
            num_drafts.push(k as u32);
        }
        let s = gpu.stream();
        s.memcpy_htod(&target_row, &mut inputs.target_row.slice_mut(..r))?;
        s.memcpy_htod(&draft_row, &mut inputs.draft_row.slice_mut(..r))?;
        s.memcpy_htod(&num_drafts, &mut inputs.num_drafts.slice_mut(..r))?;
        s.memcpy_htod(&drafts, &mut inputs.drafts.slice_mut(..r * stride as usize))?;
        // SAFETY: arguments match `eidola_chain_accept`; every index the
        // kernel derives was bounded above, and `inputs` holds exactly the
        // checked values (uploaded on this stream, before the launch).
        unsafe {
            launch!(
                gpu,
                self.accept,
                [r as u32, 1, 1],
                dptr(target, s),
                dptr(draft, s),
                n,
                dptr(rows, s),
                dptr(&inputs.target_row, s),
                dptr(&inputs.draft_row, s),
                dptr(&inputs.num_drafts, s),
                dptr(&inputs.drafts, s),
                stride,
                dptr(scratch, s),
                dptr(out, s),
                dptr(counts, s),
            )
        }
    }
}

/// One sequence's chain acceptance, as the host scheduled it: its drafts and
/// where its target and draft distributions start.
#[derive(Clone, Copy, Debug)]
pub struct AcceptRow<'a> {
    pub target_row: u32,
    pub draft_row: u32,
    pub drafts: &'a [u32],
}

/// The device copies of a chain acceptance's per-row inputs, for up to
/// `rows` sequences of at most `stride - 1` drafts. Only
/// [`Sampler::chain_accept`] writes them, with values it has checked.
pub struct AcceptInputs {
    rows: usize,
    stride: u32,
    target_row: CudaSlice<u32>,
    draft_row: CudaSlice<u32>,
    num_drafts: CudaSlice<u32>,
    drafts: CudaSlice<u32>,
}

impl AcceptInputs {
    pub fn new(gpu: &Gpu, rows: usize, stride: u32) -> Result<AcceptInputs> {
        let len = rows
            .checked_mul(stride as usize)
            .filter(|_| rows > 0 && stride > 0)
            .ok_or_else(|| CudaError::new(format!("accept inputs: {rows} x {stride}")))?;
        let s = gpu.stream();
        Ok(AcceptInputs {
            rows,
            stride,
            target_row: s.alloc_zeros(rows)?,
            draft_row: s.alloc_zeros(rows)?,
            num_drafts: s.alloc_zeros(rows)?,
            drafts: s.alloc_zeros(len)?,
        })
    }

    /// Tokens per output row: the most drafts a row may have, plus one.
    pub fn stride(&self) -> u32 {
        self.stride
    }
}
