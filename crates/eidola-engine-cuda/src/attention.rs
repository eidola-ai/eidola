//! Paged attention: FlashInfer's FA2 prefill template with the attention-sink
//! variant (`flashinfer_fa2_sink_paged`), BF16 Q/KV/O, head dims 192/128,
//! causal, with an optional sliding window and per-head sink logits. One
//! kernel serves prefill and decode.
//!
//! The kernel takes FlashInfer's `PagedParams` by value; [`PagedParams`] is its
//! `repr(C)` mirror (296 bytes, checked against the image's launch contract).
//! The work list (which request, query tile and KV tile each CTA handles) is
//! what FlashInfer's `PrefillPlan` computes without KV splitting: one CTA per
//! (request, query tile) per KV head, each walking the request's whole visible
//! KV. Splitting long KV across CTAs (and the merge kernel) is not used yet.

use std::ffi::c_void;

use cudarc::driver::CudaSlice;

use crate::launch::dptr;
use crate::module::{Kernel, KernelModule};
use crate::{CudaError, Gpu, Result};

pub const HEAD_DIM_QK: u32 = 192;
pub const HEAD_DIM_VO: u32 = 128;

/// `cuda::fast_mod_div<uint32_t>` (CCCL 13.2) wrapped as FlashInfer's
/// `uint_fastdiv`: `{divisor, multiplier, add, shift}` then the plain divisor.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UintFastdiv {
    divisor: u32,
    multiplier: u32,
    add: u32,
    shift: i32,
    d: u32,
}

impl UintFastdiv {
    pub fn new(d: u32) -> UintFastdiv {
        let divisor = d.max(1);
        let shift = 31 - divisor.leading_zeros();
        let (multiplier, add) = if divisor.is_power_of_two() {
            (0, 0)
        } else {
            let k = 32 + shift;
            let mul = ((1u64 << k) + (1u64 << shift)) / divisor as u64;
            let low = (1u64 << k) / divisor as u64;
            (mul as u32, (low == mul) as u32)
        };
        UintFastdiv {
            divisor,
            multiplier,
            add,
            shift: shift as i32,
            d,
        }
    }
}

/// `paged_kv_t<nv_bfloat16, int32_t>` with independent K and V strides.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PagedKv {
    pub page_size: UintFastdiv,
    pub num_heads: u32,
    pub head_dim: u32,
    pub batch_size: u32,
    pub stride_page: u32,
    pub stride_n: u32,
    pub stride_h: u32,
    pub v_stride_page: u32,
    pub v_stride_n: u32,
    pub v_stride_h: u32,
    pub k_data: u64,
    pub v_data: u64,
    pub indices: u64,
    pub indptr: u64,
    pub last_page_len: u64,
    pub rope_pos_offset: u64,
}

/// FlashInfer's generated `PagedParams` for the sink variant.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PagedParams {
    pub q: u64,
    pub paged_kv: PagedKv,
    pub q_indptr: u64,
    pub o: u64,
    pub lse: u64,
    pub group_size: UintFastdiv,
    pub sink: u64,
    pub sm_scale: f64,
    pub num_qo_heads: u32,
    pub q_stride_n: i32,
    pub q_stride_h: i32,
    pub k_sf_stride_page: u32,
    pub k_sf_stride_n: u32,
    pub k_sf_stride_h: u32,
    pub v_sf_stride_page: u32,
    pub v_sf_stride_n: u32,
    pub v_sf_stride_h: u32,
    pub window_left: i32,
    pub request_indices: u64,
    pub qo_tile_indices: u64,
    pub kv_tile_indices: u64,
    pub merge_indptr: u64,
    pub o_indptr: u64,
    pub block_valid_mask: u64,
    pub kv_chunk_size_ptr: u64,
    pub max_total_num_rows: u32,
    pub total_num_rows: u64,
    pub padded_batch_size: u32,
    pub partition_kv: bool,
}

const _: () = {
    assert!(std::mem::size_of::<UintFastdiv>() == 20);
    assert!(std::mem::size_of::<PagedKv>() == 104);
    assert!(std::mem::size_of::<PagedParams>() == 296);
    assert!(std::mem::offset_of!(PagedParams, paged_kv) == 8);
    assert!(std::mem::offset_of!(PagedParams, q_indptr) == 112);
    assert!(std::mem::offset_of!(PagedParams, group_size) == 136);
    assert!(std::mem::offset_of!(PagedParams, sink) == 160);
    assert!(std::mem::offset_of!(PagedParams, window_left) == 212);
    assert!(std::mem::offset_of!(PagedParams, kv_chunk_size_ptr) == 264);
    assert!(std::mem::offset_of!(PagedParams, partition_kv) == 292);
};

/// One request (sequence row) of an attention call: its query rows and the
/// KV pages it reads, in position order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttnRequest {
    /// First query row in Q/O.
    pub q_start: u32,
    pub qo_len: u32,
    /// Physical blocks covering KV positions `first_page * page_size ..
    /// kv_end`; leading blocks wholly outside every query's window may be
    /// left out (positions are relative to the end, so the masks are
    /// unaffected).
    pub pages: Vec<u32>,
    /// KV positions covered, counted from the first listed page's start;
    /// the last query sits at `kv_len - 1`.
    pub kv_len: u32,
}

/// The per-step work list for one KV group (shared by all its layers).
pub struct AttnPlan {
    tile: u32,
    work_items: u32,
    num_requests: u32,
    q_indptr: CudaSlice<i32>,
    indices: CudaSlice<i32>,
    indptr: CudaSlice<i32>,
    last_page_len: CudaSlice<i32>,
    request_indices: CudaSlice<i32>,
    qo_tile_indices: CudaSlice<i32>,
    kv_tile_indices: CudaSlice<i32>,
    kv_chunk_size: CudaSlice<u32>,
}

/// One layer's KV and attention settings.
#[derive(Clone, Copy, Debug)]
pub struct AttnLayer {
    /// Device address of this layer's K inside block 0.
    pub k_base: u64,
    /// Device address of this layer's V inside block 0.
    pub v_base: u64,
    /// Elements per block (all layers): the page stride.
    pub block_elems: u32,
    pub num_kv_heads: u32,
    pub page_size: u32,
    /// Keys visible to a query besides itself (`window - 1`), or -1.
    pub window_left: i32,
    /// `f32[num_qo_heads]` sink logits (`-inf` for none).
    pub sink: u64,
}

/// The attention kernels.
pub struct Attention {
    _module: KernelModule,
    q16: Kernel,
    q64: Kernel,
    q128: Kernel,
}

impl Attention {
    pub fn from_module(module: KernelModule) -> Result<Attention> {
        let get = |s: &str| -> Result<Kernel> {
            let k = module.kernel(s)?;
            if k.meta().params_bytes as usize != std::mem::size_of::<PagedParams>() {
                return Err(CudaError::new(format!(
                    "{s}: PagedParams is {} bytes in the image, {} here",
                    k.meta().params_bytes,
                    std::mem::size_of::<PagedParams>()
                )));
            }
            Ok(k)
        };
        Ok(Attention {
            q16: get("eidola_fa2_sink_paged_bf16_q16")?,
            q64: get("eidola_fa2_sink_paged_bf16_q64")?,
            q128: get("eidola_fa2_sink_paged_bf16_q128")?,
            _module: module,
        })
    }

    /// Plan one step's attention for a group: query tile chosen from the
    /// longest packed query run (query rows × GQA group size), as FlashInfer
    /// does, and one work item per (request, query tile).
    pub fn plan(
        &self,
        gpu: &Gpu,
        requests: &[AttnRequest],
        group_size: u32,
        page_size: u32,
    ) -> Result<AttnPlan> {
        let s = gpu.stream();
        let packed = requests
            .iter()
            .map(|r| r.qo_len * group_size)
            .max()
            .unwrap_or(1);
        let tile = if packed <= 16 {
            16
        } else if packed <= 64 {
            64
        } else {
            128
        };
        let (mut q_indptr, mut indices, mut indptr, mut last) =
            (vec![0i32], vec![], vec![0i32], vec![]);
        let (mut req, mut qtile, mut kvtile) = (vec![], vec![], vec![]);
        for (i, r) in requests.iter().enumerate() {
            if r.kv_len == 0 || r.pages.is_empty() || r.kv_len > r.pages.len() as u32 * page_size {
                return Err(CudaError::new(format!("attention request {r:?}")));
            }
            if q_indptr.last().copied() != Some(r.q_start as i32) {
                return Err(CudaError::new(
                    "attention requests must cover Q rows in order",
                ));
            }
            q_indptr.push((r.q_start + r.qo_len) as i32);
            indices.extend(r.pages.iter().map(|&p| p as i32));
            indptr.push(indices.len() as i32);
            last.push((r.kv_len - (r.pages.len() as u32 - 1) * page_size) as i32);
            for t in 0..(r.qo_len * group_size).div_ceil(tile) {
                req.push(i as i32);
                qtile.push(t as i32);
                kvtile.push(0);
            }
        }
        let up = |v: &[i32]| -> Result<CudaSlice<i32>> {
            Ok(if v.is_empty() {
                s.alloc_zeros::<i32>(1)?
            } else {
                s.clone_htod(v)?
            })
        };
        Ok(AttnPlan {
            tile,
            work_items: req.len() as u32,
            num_requests: requests.len() as u32,
            q_indptr: up(&q_indptr)?,
            indices: up(&indices)?,
            indptr: up(&indptr)?,
            last_page_len: up(&last)?,
            request_indices: up(&req)?,
            qo_tile_indices: up(&qtile)?,
            kv_tile_indices: up(&kvtile)?,
            kv_chunk_size: s.alloc_zeros::<u32>(1)?,
        })
    }

    /// Attention for one layer: `q` is `[rows, num_qo_heads, 192]`, `o` is
    /// `[rows, num_qo_heads, 128]`, both BF16.
    ///
    /// # Safety
    ///
    /// `q`, `o`, the layer's pool and its sinks must cover what the plan's
    /// requests address.
    pub unsafe fn run(
        &self,
        gpu: &Gpu,
        plan: &AttnPlan,
        layer: &AttnLayer,
        num_qo_heads: u32,
        q: u64,
        o: u64,
    ) -> Result<()> {
        if plan.work_items == 0 {
            return Ok(());
        }
        let s = gpu.stream();
        let kernel = match plan.tile {
            16 => &self.q16,
            64 => &self.q64,
            _ => &self.q128,
        };
        let h = layer.num_kv_heads;
        let mut params = PagedParams {
            q,
            paged_kv: PagedKv {
                page_size: UintFastdiv::new(layer.page_size),
                num_heads: h,
                head_dim: HEAD_DIM_QK,
                batch_size: plan.num_requests,
                stride_page: layer.block_elems,
                stride_n: h * HEAD_DIM_QK,
                stride_h: HEAD_DIM_QK,
                v_stride_page: layer.block_elems,
                v_stride_n: h * HEAD_DIM_VO,
                v_stride_h: HEAD_DIM_VO,
                k_data: layer.k_base,
                v_data: layer.v_base,
                indices: dptr(&plan.indices, s),
                indptr: dptr(&plan.indptr, s),
                last_page_len: dptr(&plan.last_page_len, s),
                rope_pos_offset: 0,
            },
            q_indptr: dptr(&plan.q_indptr, s),
            o,
            lse: 0,
            group_size: UintFastdiv::new(num_qo_heads / h),
            sink: layer.sink,
            sm_scale: 1.0 / (HEAD_DIM_QK as f64).sqrt(),
            num_qo_heads,
            q_stride_n: (num_qo_heads * HEAD_DIM_QK) as i32,
            q_stride_h: HEAD_DIM_QK as i32,
            window_left: layer.window_left,
            request_indices: dptr(&plan.request_indices, s),
            qo_tile_indices: dptr(&plan.qo_tile_indices, s),
            kv_tile_indices: dptr(&plan.kv_tile_indices, s),
            o_indptr: dptr(&plan.q_indptr, s),
            kv_chunk_size_ptr: dptr(&plan.kv_chunk_size, s),
            padded_batch_size: plan.work_items,
            partition_kv: false,
            ..PagedParams::default()
        };
        let mut args = [&mut params as *mut PagedParams as *mut c_void];
        // SAFETY: the single by-value argument is the kernel's PagedParams
        // (size checked at load); addresses are the caller's.
        unsafe { kernel.launch(s, [plan.work_items, 1, h], &mut args) }
    }
}
