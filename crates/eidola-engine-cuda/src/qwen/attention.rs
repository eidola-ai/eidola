//! The fast profile's attention: FlashInfer FA2 at head dimension 256
//! (`flashinfer_fa2_sink_paged_d256`), batch-invariant by MiMo's structure
//! ([`crate::mimo::attention`]): one KV warp over 64-key tiles, requests cut at
//! multiples of 64 query positions, every row's keys split at absolute multiples of
//! [`super::plan::SPLIT_KEYS`] (512) positions and the partials merged in chunk order,
//! work lists walked by fixed grids. The model has no sinks: every head's sink is `-inf`. The image has query tiles
//! 16 and 64 only, so a plan's tile is at most [`MAX_TILE`].
//!
//! **Two KV groups, one reduction.** The target's layers list their pages from position
//! 0 at the pool's block size. The drafter's layer holds no KV at position 0 (its first
//! row is at position 1, and its attention starts there), so its requests are made over
//! positions counted from 1: relative key `j` is position `j + 1`, a page is one position
//! (page size 1, both planes addressed a row at a time: [`drafter_page`]), and the split's
//! chunk boundaries and the 64-key cut fall at absolute positions `1 + 64·m`, a fixed
//! function of the position as the reduction requires.

use std::ffi::c_void;

use crate::mimo::attention::{
    AttnLayer, AttnParams, HostPlan, PagedKv, PagedParams, Partials, PassCaps, PlanView,
    UintFastdiv,
};
use crate::module::{Kernel, KernelModule};
use crate::{CudaError, Gpu, Result, narrow};

/// The image.
pub const IMAGE: &str = "flashinfer_fa2_sink_paged_d256";
/// Head dimension of queries, keys and values.
pub const HEAD_DIM: u32 = 256;
/// The largest query tile the image has.
pub const MAX_TILE: u32 = 64;
/// Bytes of one partial row per query head: the BF16 output and its f32 log-sum-exp.
pub const PARTIAL_BYTES_PER_HEAD: usize = HEAD_DIM as usize * 2 + 4;
/// The merge's launch-contract record.
const MERGE_META: &str = "eidola_fa2_merge_states_bf16_d256_meta";
/// Blocks of the persistent merge, at most.
const MERGE_BLOCKS: u32 = 148 * 4;

/// The page a drafter position lives in, at page size 1: block `b`'s image is the K
/// plane of `block_size` rows then the V plane, so position row `o` of block `b` is
/// page `b · 2 · block_size + o` of either plane (the V plane's base is one plane past
/// the K plane's, [`AttnLayer::v_base`]).
pub fn drafter_page(block: u32, offset: u32, block_size: u32) -> u64 {
    u64::from(block) * 2 * u64::from(block_size) + u64::from(offset)
}

/// One FA2 pass as the device runs it: a plan's view, the layer's KV, and the buffers.
#[derive(Clone, Copy, Debug)]
pub struct Fa2Args {
    pub view: PlanView,
    pub layer: AttnLayer,
    pub num_qo_heads: u32,
    /// `[rows, num_qo_heads, 256]` BF16.
    pub q: u64,
    /// `[rows, num_qo_heads, 256]` BF16.
    pub o: u64,
    pub partials: Partials,
    pub scale: f32,
}

/// Pass `index` of `plan`, padded to `caps` ([`HostPlan::padded_pass`]), as one table of
/// words in the order a launch's view reads them, and that view over the table at
/// `base` (a device address): what the step table holds for one FA2 launch, for a bench
/// or test that runs one pass alone.
pub fn pass_words(
    plan: &HostPlan,
    index: usize,
    caps: &PassCaps,
    base: u64,
) -> Result<(Vec<u32>, PlanView)> {
    let p = plan.padded_pass(index, caps)?;
    let mut w = Vec::new();
    let mut put = |v: &[u32]| {
        let at = base + 4 * w.len() as u64;
        w.extend_from_slice(v);
        at
    };
    let q_indptr = put(&p.q_indptr);
    let indices = put(&p.indices);
    let indptr = put(&p.indptr);
    let last_page_len = put(&p.last_page_len);
    let request_indices = put(&p.request_indices);
    let qo_tile_indices = put(&p.qo_tile_indices);
    let kv_tile_indices = put(&p.kv_tile_indices);
    let o_indptr = put(&p.o_indptr);
    let merge_indptr = put(&p.merge_indptr);
    let work_items = put(&[p.work_items, plan.split_keys()]);
    let view = PlanView {
        shape: caps.shape,
        out_row: p.out_row,
        q_indptr,
        indices,
        indptr,
        last_page_len,
        request_indices,
        qo_tile_indices,
        kv_tile_indices,
        o_indptr,
        merge_indptr,
        work_items,
        kv_chunk_size: work_items + 4,
    };
    Ok((w, view))
}

/// The d256 instances and their merge.
pub struct QwenAttention {
    _module: KernelModule,
    /// Split instances at query tiles 16 and 64.
    split: [Kernel; 2],
    merge: Kernel,
}

impl QwenAttention {
    pub fn from_module(module: KernelModule) -> Result<QwenAttention> {
        module.expect_image(IMAGE)?;
        let get = |s: &str| -> Result<Kernel> {
            let k = module.kernel(s)?;
            if k.meta().params_bytes as usize != std::mem::size_of::<AttnParams>() {
                return Err(CudaError::new(format!(
                    "{s}: its parameters are {} bytes in the image, {} here",
                    k.meta().params_bytes,
                    std::mem::size_of::<AttnParams>()
                )));
            }
            Ok(k)
        };
        let merge_symbol = module
            .cubin()
            .entry_for_meta(MERGE_META)
            .ok_or_else(|| CudaError::new("no d256 attention merge entry"))?
            .symbol
            .clone();
        let merge = module.kernel(&merge_symbol)?;
        if merge.meta().params_bytes != 0 {
            return Err(CudaError::new("the attention merge takes an argument list"));
        }
        Ok(QwenAttention {
            split: [
                get("eidola_fa2_sink_paged_d256_bf16_q16")?,
                get("eidola_fa2_sink_paged_d256_bf16_q64")?,
            ],
            merge,
            _module: module,
        })
    }

    /// One pass: the split attention into the partials, then their merge into `o`.
    ///
    /// # Safety
    ///
    /// The view's arrays hold a pass `HostPlan` made for its shape when the launches
    /// run; `q`, `o`, the layer's pool and its sinks cover what its requests address.
    pub unsafe fn run(&self, gpu: &Gpu, a: &Fa2Args) -> Result<()> {
        let shape = &a.view.shape;
        if shape.grid == 0 {
            return Ok(());
        }
        let layer = &a.layer;
        if !shape.split || layer.window_left >= 0 {
            return Err(CudaError::new(
                "d256 attention: split plans over global layers",
            ));
        }
        if layer.num_kv_heads == 0
            || !a.num_qo_heads.is_multiple_of(layer.num_kv_heads)
            || a.num_qo_heads / layer.num_kv_heads != shape.group_size
            || layer.page_size != shape.page_size
            || layer.k_page_stride == 0
            || layer.v_page_stride == 0
        {
            return Err(CudaError::new(format!(
                "d256 attention: a plan for groups of {} and pages of {} on {layer:?}",
                shape.group_size, shape.page_size
            )));
        }
        if shape.partial_rows > a.partials.rows || a.partials.heads != a.num_qo_heads {
            return Err(CudaError::new(format!(
                "d256 attention: {} partial rows past the scratch's {}",
                shape.partial_rows, a.partials.rows
            )));
        }
        let kernel = match shape.tile {
            16 => &self.split[0],
            64 => &self.split[1],
            t => return Err(CudaError::new(format!("d256 attention: no {t}-row tile"))),
        };
        let h = layer.num_kv_heads;
        let v = &a.view;
        let mut params = AttnParams {
            paged: PagedParams {
                q: a.q,
                paged_kv: PagedKv {
                    page_size: UintFastdiv::new(layer.page_size),
                    num_heads: h,
                    head_dim: HEAD_DIM,
                    batch_size: shape.num_requests,
                    stride_page: layer.k_page_stride,
                    stride_n: h * HEAD_DIM,
                    stride_h: HEAD_DIM,
                    v_stride_page: layer.v_page_stride,
                    v_stride_n: h * HEAD_DIM,
                    v_stride_h: HEAD_DIM,
                    k_data: layer.k_base,
                    v_data: layer.v_base,
                    indices: v.indices,
                    indptr: v.indptr,
                    last_page_len: v.last_page_len,
                    rope_pos_offset: 0,
                },
                q_indptr: v.q_indptr,
                o: a.partials.v,
                lse: a.partials.s,
                group_size: UintFastdiv::new(a.num_qo_heads / h),
                sink: layer.sink,
                sm_scale: f64::from(a.scale),
                num_qo_heads: a.num_qo_heads,
                q_stride_n: narrow(u64::from(a.num_qo_heads) * u64::from(HEAD_DIM), "q stride")?,
                q_stride_h: narrow(HEAD_DIM, "q head stride")?,
                window_left: -1,
                request_indices: v.request_indices,
                qo_tile_indices: v.qo_tile_indices,
                kv_tile_indices: v.kv_tile_indices,
                o_indptr: v.o_indptr,
                kv_chunk_size_ptr: v.kv_chunk_size,
                padded_batch_size: shape.grid,
                partition_kv: true,
                ..PagedParams::default()
            },
            work_items: v.work_items,
        };
        let mut args = [(&mut params as *mut AttnParams).cast::<c_void>()];
        // SAFETY: the kernel's by-value `EidolaAttnParamsD256` (size checked at load);
        // the addresses are the caller's.
        unsafe { kernel.launch(gpu.stream(), [shape.grid, 1, h], &mut args)? };
        let row_bytes = u64::from(a.num_qo_heads) * u64::from(HEAD_DIM) * 2;
        let pairs = u64::from(shape.merge_rows) * u64::from(a.num_qo_heads);
        let blocks: u32 = narrow(pairs.min(u64::from(MERGE_BLOCKS)), "merge blocks")?;
        if blocks == 0 {
            return Ok(());
        }
        // SAFETY: the merge reads the partial rows the launch above wrote (within the
        // scratch, checked) and writes the pass's rows of `o`.
        unsafe {
            crate::launch!(
                gpu,
                self.merge,
                [blocks, 1, 1],
                a.partials.v,
                a.partials.s,
                v.merge_indptr,
                a.o + u64::from(v.out_row) * row_bytes,
                0u64,
                shape.merge_rows,
                0u64,
                a.num_qo_heads
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimo::attention::{AttnRequest, HostPlan, PassCaps, Reduction, tile_for_max};

    #[test]
    fn plans_never_ask_for_the_missing_128_row_tile() {
        assert_eq!(tile_for_max(6, MAX_TILE), 16);
        assert_eq!(tile_for_max(24, MAX_TILE), 64);
        assert_eq!(tile_for_max(6 * 64, MAX_TILE), 64);
        let r = AttnRequest {
            q_start: 0,
            qo_len: 40,
            pages: (1..=3).collect(),
            kv_start: 0,
            kv_len: 40,
        };
        let plan = HostPlan::with_max_tile(&[r], 6, 16, Reduction::Split, 1 << 16, MAX_TILE)
            .expect("plan");
        assert_eq!(plan.tile(), 64);
        let caps = PassCaps::with_max_tile(Reduction::Split, 8, 4, 6, 16, 64, 4096, MAX_TILE)
            .expect("caps");
        assert_eq!(caps.shape.tile, 64);
    }

    #[test]
    fn drafter_pages_address_both_planes_a_row_at_a_time() {
        // Block 3 of 4-position blocks: K rows at 24..28, V rows at 28..32 (in rows of
        // one position); the V plane's base is 4 rows past the K plane's.
        assert_eq!(drafter_page(3, 0, 4), 24);
        assert_eq!(drafter_page(3, 3, 4), 27);
        assert_eq!(drafter_page(0, 1, 4), 1);
    }
}
