//! The device sampler (`sampling.cu`): the serving core's `sampling` semantics,
//! bit for bit given equal logits.

use crate::module::KernelDir;
use cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits};
use eidola_engine::sampling::{SamplingParams, Stream};

use crate::launch::dptr;
use crate::module::{Kernel, KernelModule};
use crate::{CudaError, Gpu, Result, launch, narrow};

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

/// One `eidola_sample` launch over device addresses (see
/// [`Sampler::launch_sample`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SampleLaunch {
    pub logits: u64,
    pub logits_stride: u64,
    pub n: u32,
    pub rows: u64,
    pub num_rows: u32,
    pub draw: Option<Stream>,
    pub probs: u64,
    pub tokens: u64,
    pub status: u64,
}

/// The status bit a row with NaN, `+inf` or (when not greedy) only `-inf`
/// logits sets.
pub const STATUS_NON_FINITE: u32 = 1;

/// The status bit a token id at or past the sampleable vocabulary raises:
/// one `eidola_sample` would have written (it writes 0 instead), or a draft
/// `eidola_chain_accept` was given (the row ends with no tokens). Draft ids
/// a drafter produces on the device are bounded by these two kernels, where
/// they are made and where they are read, not on the host.
pub const STATUS_BAD_TOKEN: u32 = 2;

/// One `eidola_chain_accept` launch over device addresses (see
/// [`Sampler::launch_accept`]).
#[derive(Clone, Copy, Debug)]
pub struct AcceptLaunch {
    pub target: u64,
    pub draft: u64,
    pub n: u32,
    pub rows: u64,
    pub target_row: u64,
    pub draft_row: u64,
    pub draft_step: u32,
    pub num_drafts: u64,
    pub drafts: u64,
    pub stride: u32,
    pub scratch: u64,
    pub out: u64,
    pub counts: u64,
    pub status: u64,
    pub num_rows: u32,
}

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
        module.expect_image("sampling")?;
        Ok(Sampler {
            sample: module.kernel("eidola_sample")?,
            accept: module.kernel("eidola_chain_accept")?,
            _module: module,
        })
    }

    /// For each row of `rows`: the processed distribution over the first `n`
    /// logits of row `rows[r].logit_row` into `probs[r * n ..]`, and, with
    /// `draw`, a token drawn from it with that stream into `tokens[r]` (a
    /// greedy row takes the argmax and consumes no draw). Non-finite logits set
    /// [`STATUS_NON_FINITE`] in `status`. The rows are host values, checked
    /// (every `logit_row` inside `logits`) and then uploaded into `rows_dev`,
    /// so the kernel reads only rows that passed.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        gpu: &Gpu,
        logits: &CudaSlice<f32>,
        logits_stride: usize,
        n: u32,
        rows: &[SampleRow],
        rows_dev: &mut CudaSlice<SampleRow>,
        draw: Option<Stream>,
        probs: &mut CudaSlice<f64>,
        tokens: &mut CudaSlice<u32>,
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let r = rows.len();
        let num_rows: u32 = narrow(r, "sample rows")?;
        if rows_dev.len() < r
            || probs.len() < r * n as usize
            || tokens.len() < r
            || status.is_empty()
        {
            return Err(CudaError::new("sample: buffer too small"));
        }
        if n == 0 || n as usize > logits_stride || n > 1 << 20 || logits.len() < logits_stride {
            return Err(CudaError::new(format!("sample: vocabulary {n}")));
        }
        let logit_rows = logits.len() / logits_stride;
        if let Some(row) = rows.iter().find(|row| row.logit_row as usize >= logit_rows) {
            return Err(CudaError::new(format!(
                "sample: logit row {} of {logit_rows}",
                row.logit_row
            )));
        }
        let s = gpu.stream();
        s.memcpy_htod(rows, &mut rows_dev.slice_mut(..r))?;
        // SAFETY: buffers and every row's logit row checked above, and
        // `rows_dev` holds exactly those rows.
        unsafe {
            self.launch_sample(
                gpu,
                SampleLaunch {
                    logits: dptr(logits, s),
                    logits_stride: logits_stride as u64,
                    n,
                    rows: dptr(rows_dev, s),
                    num_rows,
                    draw,
                    probs: dptr(probs, s),
                    tokens: dptr(tokens, s),
                    status: dptr(status, s),
                },
            )
        }
    }

    /// The `eidola_sample` launch alone, over rows already on the device.
    ///
    /// # Safety
    ///
    /// `a.rows` must hold `a.num_rows` rows whose logit rows lie inside
    /// `a.logits` (rows of `a.logits_stride` floats, `a.n` of them read), and
    /// `a.probs`, `a.tokens` and `a.status` must hold `num_rows * n`,
    /// `num_rows` and one value, when the launch runs. `n` must be in
    /// `1..=2^20` and at most the stride.
    pub(crate) unsafe fn launch_sample(&self, gpu: &Gpu, a: SampleLaunch) -> Result<()> {
        if a.num_rows == 0 {
            return Ok(());
        }
        if a.n == 0 || u64::from(a.n) > a.logits_stride || a.n > 1 << 20 {
            return Err(CudaError::new(format!("sample: vocabulary {}", a.n)));
        }
        let stream = a.draw.map_or(u32::MAX, |d| d as u32);
        // SAFETY: arguments match `eidola_sample`; the caller's contract
        // covers every address.
        unsafe {
            launch!(
                gpu,
                self.sample,
                [a.num_rows, 1, 1],
                a.logits,
                a.logits_stride,
                a.n,
                a.rows,
                stream,
                a.probs,
                a.tokens,
                a.status,
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
    /// Everything the kernel indexes by (draft counts, distribution rows)
    /// is checked here, on the host, against the buffers it will reach, and
    /// then uploaded into `inputs`: the device reads only values that
    /// passed. Draft ids are checked here too, and again by the kernel (the
    /// drafted step's ids never reach the host), which raises
    /// [`STATUS_BAD_TOKEN`] in `status` for one out of range.
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
        status: &mut CudaSlice<u32>,
    ) -> Result<()> {
        if plan.is_empty() {
            return Ok(());
        }
        if status.is_empty() {
            return Err(CudaError::new("chain_accept: no status word"));
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
        let AcceptPlan {
            target_row,
            draft_row,
            num_drafts,
            ids,
        } = AcceptPlan::new(plan, n, stride, target_rows, draft_rows)?;
        let s = gpu.stream();
        if inputs.drafts.len() < ids.len() {
            inputs.drafts = s.alloc_zeros(ids.len())?;
        }
        s.memcpy_htod(&target_row, &mut inputs.target_row.slice_mut(..r))?;
        s.memcpy_htod(&draft_row, &mut inputs.draft_row.slice_mut(..r))?;
        s.memcpy_htod(&num_drafts, &mut inputs.num_drafts.slice_mut(..r))?;
        s.memcpy_htod(&ids, &mut inputs.drafts.slice_mut(..ids.len()))?;
        // SAFETY: arguments match `eidola_chain_accept`; every index the
        // kernel derives was bounded above, and `inputs` holds exactly the
        // checked values (uploaded on this stream, before the launch).
        unsafe {
            self.launch_accept(
                gpu,
                AcceptLaunch {
                    target: dptr(target, s),
                    draft: dptr(draft, s),
                    n,
                    rows: dptr(rows, s),
                    target_row: dptr(&inputs.target_row, s),
                    draft_row: dptr(&inputs.draft_row, s),
                    draft_step: 1,
                    num_drafts: dptr(&inputs.num_drafts, s),
                    drafts: dptr(&inputs.drafts, s),
                    stride,
                    scratch: dptr(scratch, s),
                    out: dptr(out, s),
                    counts: dptr(counts, s),
                    status: dptr(status, s),
                    num_rows: narrow(r, "acceptance rows")?,
                },
            )
        }
    }

    /// The `eidola_chain_accept` launch alone, over inputs already on the
    /// device.
    ///
    /// # Safety
    ///
    /// For each of `a.num_rows` rows `r`: `a.rows[r]` and `a.num_drafts[r] =
    /// k` must be readable; the target rows `a.target_row[r] ..= + k` and the
    /// draft rows `a.draft_row[r] + i * a.draft_step` (`i < k`) must lie
    /// inside `a.target` and `a.draft` (rows of `a.n` f64), the latter also
    /// inside `a.drafts`; `a.out` must hold `r * a.stride + k + 1` words,
    /// `a.counts` and `a.scratch` (rows of `a.n`) `a.num_rows` rows, and
    /// `a.status` one word. Draft ids are bounded by the kernel.
    pub unsafe fn launch_accept(&self, gpu: &Gpu, a: AcceptLaunch) -> Result<()> {
        if a.num_rows == 0 {
            return Ok(());
        }
        if a.n == 0 || a.n > 1 << 20 {
            return Err(CudaError::new(format!("chain_accept: vocabulary {}", a.n)));
        }
        // SAFETY: arguments match `eidola_chain_accept`; the caller's
        // contract covers every address.
        unsafe {
            launch!(
                gpu,
                self.accept,
                [a.num_rows, 1, 1],
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
            )
        }
    }
}

/// A host-scheduled chain acceptance, checked and laid out as the kernel
/// reads it: per row its target row, draft row and draft count, and the
/// draft ids. The kernel reads draft `i` of a row at the index of its
/// distribution (`draft_row + i`), so an id row belongs to a distribution
/// row: a draft is the token drawn from that distribution. Rows may share
/// distribution rows only with the same ids there; rows that name different
/// ids for one distribution row are refused, since one of them would be
/// accepted against the other's drafts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcceptPlan {
    pub target_row: Vec<u32>,
    pub draft_row: Vec<u32>,
    pub num_drafts: Vec<u32>,
    pub ids: Vec<u32>,
}

impl AcceptPlan {
    /// Checks `plan` against distributions of `target_rows` and `draft_rows`
    /// rows over a vocabulary of `n`, and outputs of `stride` tokens a row:
    /// every index the kernel derives in bounds, every draft id below `n`,
    /// and no id row given two different ids. Needs no device.
    pub fn new(
        plan: &[AcceptRow<'_>],
        n: u32,
        stride: u32,
        target_rows: usize,
        draft_rows: usize,
    ) -> Result<AcceptPlan> {
        let r = plan.len();
        let (mut target_row, mut draft_row, mut num_drafts) = (
            Vec::with_capacity(r),
            Vec::with_capacity(r),
            Vec::with_capacity(r),
        );
        let mut id_rows = 0usize;
        for (i, row) in plan.iter().enumerate() {
            let k = row.drafts.len();
            // Up to k accepted drafts and one more token: k + 1 slots.
            if k >= stride as usize {
                return Err(CudaError::new(format!(
                    "chain_accept: row {i} has {k} drafts, its output holds {stride} tokens"
                )));
            }
            // k < stride, a u32. The kernel adds in u32; the sums must fit,
            // and stay in bounds.
            let k32 = u32::try_from(k).expect("k < stride");
            let target_end = row.target_row.checked_add(k32 + 1);
            if target_end.is_none_or(|e| e as usize > target_rows) {
                return Err(CudaError::new(format!(
                    "chain_accept: row {i}'s target rows {}+{} beyond {target_rows}",
                    row.target_row,
                    k + 1
                )));
            }
            let draft_end = row.draft_row.checked_add(k32);
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
            if k > 0 {
                id_rows = id_rows.max(draft_end.map_or(0, |e| e as usize));
            }
            target_row.push(row.target_row);
            draft_row.push(row.draft_row);
            num_drafts.push(k32);
        }
        // Draft ids sit where their distributions do: draft `i` of row `r`
        // at id row `draft_row + i` (the kernel's `draft_step` of 1).
        let mut ids: Vec<Option<(usize, u32)>> = vec![None; id_rows.max(1)];
        for (i, row) in plan.iter().enumerate() {
            for (j, &d) in row.drafts.iter().enumerate() {
                let at = row.draft_row as usize + j;
                match ids[at] {
                    Some((other, e)) if e != d => {
                        return Err(CudaError::new(format!(
                            "chain_accept: rows {other} and {i} draft {e} and {d} from \
                             distribution row {at}"
                        )));
                    }
                    _ => ids[at] = Some((i, d)),
                }
            }
        }
        Ok(AcceptPlan {
            target_row,
            draft_row,
            num_drafts,
            ids: ids.into_iter().map(|x| x.map_or(0, |(_, d)| d)).collect(),
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(target_row: u32, draft_row: u32, drafts: &[u32]) -> AcceptRow<'_> {
        AcceptRow {
            target_row,
            draft_row,
            drafts,
        }
    }

    /// Each row's ids land at its distribution rows; rows may share
    /// distribution rows with the same ids, never with different ones (the
    /// kernel reads an id at its distribution's index, so the later row's
    /// ids would replace the earlier's).
    #[test]
    fn draft_ids_belong_to_their_distribution_rows() {
        let plan = AcceptPlan::new(&[row(0, 0, &[3, 4]), row(3, 2, &[5])], 8, 3, 6, 3).unwrap();
        assert_eq!(plan.ids, [3, 4, 5]);
        assert_eq!(plan.draft_row, [0, 2]);
        assert_eq!(plan.num_drafts, [2, 1]);
        // Shared rows, same ids: one layout serves both.
        let plan = AcceptPlan::new(&[row(0, 0, &[3, 4]), row(3, 1, &[4])], 8, 3, 6, 2).unwrap();
        assert_eq!(plan.ids, [3, 4]);
        // Shared rows, different ids: refused, naming both rows.
        let e = AcceptPlan::new(&[row(0, 0, &[3, 4]), row(3, 1, &[6])], 8, 3, 6, 2).unwrap_err();
        assert!(e.to_string().contains("rows 0 and 1"), "{e}");
        let e = AcceptPlan::new(&[row(0, 0, &[3]), row(3, 0, &[6])], 8, 3, 6, 2).unwrap_err();
        assert!(e.to_string().contains("distribution row 0"), "{e}");
        // A row without drafts names no id rows, wherever its draft row.
        let plan = AcceptPlan::new(&[row(0, u32::MAX, &[])], 8, 3, 6, 2).unwrap();
        assert_eq!(plan.ids, [0]);
    }
}
