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
use crate::{CudaError, Gpu, Result, narrow};

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
            // `divisor` lies strictly between 2^shift and 2^(shift + 1), so
            // `mul` < 2^32.
            let mul32 = u32::try_from(mul).expect("fast_mod_div multiplier fits 32 bits");
            (mul32, u32::from(low == mul))
        };
        UintFastdiv {
            divisor,
            multiplier,
            add,
            shift: i32::try_from(shift).expect("a shift below 32"),
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

/// A step's work list as the host derives it, before upload. Every index
/// is checked to fit the kernel's `i32` arrays, and every request's pages to
/// cover exactly its KV: `(pages - 1) * page_size < kv_len <= pages *
/// page_size`, so the last page holds 1..=page_size positions. Its fields
/// are private: [`HostPlan::new`], which checks every relation the kernel
/// relies on, is the only way to make one, and [`Attention::upload`] takes
/// only this type.
///
/// ```compile_fail
/// // Not constructible outside the crate: every field is private.
/// let _ = eidola_engine_cuda::attention::HostPlan {
///     tile: 16,
///     q_indptr: vec![0, 1],
///     indices: vec![1],
///     indptr: vec![0, 1],
///     last_page_len: vec![1],
///     request_indices: vec![100],
///     qo_tile_indices: vec![0],
///     kv_tile_indices: vec![0],
/// };
/// ```
#[derive(Debug, PartialEq, Eq)]
pub struct HostPlan {
    tile: u32,
    q_indptr: Vec<i32>,
    indices: Vec<i32>,
    indptr: Vec<i32>,
    last_page_len: Vec<i32>,
    request_indices: Vec<i32>,
    qo_tile_indices: Vec<i32>,
    kv_tile_indices: Vec<i32>,
}

impl HostPlan {
    pub fn new(requests: &[AttnRequest], group_size: u32, page_size: u32) -> Result<HostPlan> {
        let bad =
            |r: &AttnRequest, why: &str| CudaError::new(format!("attention request {r:?}: {why}"));
        let int = |x: u64| i32::try_from(x).ok();
        if group_size == 0 || page_size == 0 {
            return Err(CudaError::new(format!(
                "attention plan: group size {group_size}, page size {page_size}"
            )));
        }
        let packed = requests
            .iter()
            .map(|r| r.qo_len as u64 * group_size as u64)
            .max()
            .unwrap_or(1);
        let tile = if packed <= 16 {
            16
        } else if packed <= 64 {
            64
        } else {
            128
        };
        let mut p = HostPlan {
            tile,
            q_indptr: vec![0],
            indices: vec![],
            indptr: vec![0],
            last_page_len: vec![],
            request_indices: vec![],
            qo_tile_indices: vec![],
            kv_tile_indices: vec![],
        };
        for (i, r) in requests.iter().enumerate() {
            let (kv, pages, ps) = (r.kv_len as u64, r.pages.len() as u64, page_size as u64);
            let before = pages.checked_sub(1).and_then(|n| n.checked_mul(ps));
            if before.is_none_or(|b| kv <= b || kv - b > ps) {
                return Err(bad(
                    r,
                    "its pages must cover its KV, the last one partly or wholly",
                ));
            }
            // Every query's own position is in its KV.
            if r.qo_len == 0 || r.qo_len > r.kv_len {
                return Err(bad(r, "1..=kv_len queries"));
            }
            if p.q_indptr.last().copied() != int(r.q_start as u64) {
                return Err(CudaError::new(
                    "attention requests must cover Q rows in order",
                ));
            }
            let q_end =
                int(r.q_start as u64 + r.qo_len as u64).ok_or_else(|| bad(r, "Q rows past i32"))?;
            p.q_indptr.push(q_end);
            for &page in &r.pages {
                p.indices
                    .push(int(page as u64).ok_or_else(|| bad(r, "a page past i32"))?);
            }
            p.indptr
                .push(int(p.indices.len() as u64).ok_or_else(|| bad(r, "pages past i32"))?);
            p.last_page_len.push(
                int(kv - before.expect("checked above"))
                    .ok_or_else(|| bad(r, "a page past i32"))?,
            );
            let tiles = (r.qo_len as u64 * group_size as u64).div_ceil(tile as u64);
            for t in 0..tiles {
                p.request_indices
                    .push(int(i as u64).ok_or_else(|| bad(r, "requests past i32"))?);
                p.qo_tile_indices
                    .push(int(t).ok_or_else(|| bad(r, "query tiles past i32"))?);
                p.kv_tile_indices.push(0);
            }
        }
        Ok(p)
    }
}

/// The per-step work list for one KV group (shared by all its layers).
pub struct AttnPlan {
    tile: u32,
    /// The page size `last_page_len` was computed for.
    page_size: u32,
    /// The GQA group size the query tiles were counted for.
    group_size: u32,
    /// Query rows the requests cover (`q_indptr`'s last entry).
    q_rows: u32,
    /// The largest page id listed, if any.
    max_page: Option<u32>,
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

impl AttnPlan {
    pub fn page_size(&self) -> u32 {
        self.page_size
    }

    pub fn group_size(&self) -> u32 {
        self.group_size
    }

    /// Query rows the plan's requests cover.
    pub fn q_rows(&self) -> u32 {
        self.q_rows
    }

    /// The largest page id it reads, if any.
    pub fn max_page(&self) -> Option<u32> {
        self.max_page
    }
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
        module.expect_image("flashinfer_fa2_sink_paged")?;
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

    /// Upload a host plan (see [`Attention::plan`]).
    pub fn upload(
        &self,
        gpu: &Gpu,
        host: HostPlan,
        group_size: u32,
        page_size: u32,
    ) -> Result<AttnPlan> {
        let s = gpu.stream();
        let HostPlan {
            tile,
            q_indptr,
            indices,
            indptr,
            last_page_len: last,
            request_indices: req,
            qo_tile_indices: qtile,
            kv_tile_indices: kvtile,
        } = host;
        let up = |v: &[i32]| -> Result<CudaSlice<i32>> {
            Ok(if v.is_empty() {
                s.alloc_zeros::<i32>(1)?
            } else {
                s.clone_htod(v)?
            })
        };
        let unsigned = |x: i32| u32::try_from(x).map_err(|_| CudaError::new("negative plan index"));
        Ok(AttnPlan {
            tile,
            page_size,
            group_size,
            q_rows: unsigned(*q_indptr.last().expect("q_indptr starts at 0"))?,
            max_page: indices.iter().copied().max().map(unsigned).transpose()?,
            work_items: narrow(req.len(), "attention work items")?,
            num_requests: narrow(indptr.len() - 1, "attention requests")?,
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
        let host = HostPlan::new(requests, group_size, page_size)?;
        self.upload(gpu, host, group_size, page_size)
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
        if layer.num_kv_heads == 0
            || !num_qo_heads.is_multiple_of(layer.num_kv_heads)
            || layer.page_size == 0
            || layer.block_elems == 0
        {
            return Err(CudaError::new(
                "attention: query heads must be whole GQA groups; pages non-empty",
            ));
        }
        if num_qo_heads / layer.num_kv_heads != plan.group_size {
            return Err(CudaError::new(format!(
                "attention: a plan for GQA groups of {} run on groups of {}",
                plan.group_size,
                num_qo_heads / layer.num_kv_heads
            )));
        }
        if layer.page_size != plan.page_size {
            return Err(CudaError::new(format!(
                "attention: a plan for {}-position pages run on {}",
                plan.page_size, layer.page_size
            )));
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
            q_stride_n: narrow(num_qo_heads as u64 * HEAD_DIM_QK as u64, "query row stride")?,
            q_stride_h: narrow(HEAD_DIM_QK, "query head stride")?,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn req(q_start: u32, qo_len: u32, pages: usize, kv_len: u32) -> AttnRequest {
        AttnRequest {
            q_start,
            qo_len,
            pages: (1..=u32::try_from(pages).unwrap()).collect(),
            kv_len,
        }
    }

    /// A request's pages cover exactly its KV, the last one with 1..=page_size
    /// positions; every index fits the kernel's `i32` arrays.
    #[test]
    fn plans_are_checked_on_the_host() {
        let p = HostPlan::new(&[req(0, 1, 2, 17), req(1, 3, 2, 32)], 16, 16).unwrap();
        assert_eq!(p.last_page_len, vec![1, 16]);
        assert_eq!(p.q_indptr, vec![0, 1, 4]);
        assert_eq!(p.indptr, vec![0, 2, 4]);
        assert_eq!(p.tile, 64);
        let refused = |rs: &[AttnRequest], group: u32, page: u32| {
            HostPlan::new(rs, group, page).expect_err(&format!("{rs:?}"));
        };
        refused(&[req(0, 1, 2, 1)], 16, 16);
        refused(&[req(0, 1, 2, 16)], 16, 16);
        refused(&[req(0, 1, 2, 33)], 16, 16);
        refused(&[req(0, 1, 0, 1)], 16, 16);
        refused(&[req(0, 0, 1, 1)], 16, 16);
        refused(&[req(0, 5, 1, 4)], 16, 16);
        refused(&[req(1, 1, 1, 1)], 16, 16);
        refused(&[req(0, 1, 1, 1), req(2, 1, 1, 1)], 16, 16);
        refused(&[req(0, 1, 1, 1)], 0, 16);
        refused(&[req(0, 1, 1, 1)], 16, 0);
        refused(
            &[AttnRequest {
                pages: vec![1 << 31],
                ..req(0, 1, 1, 1)
            }],
            16,
            16,
        );
        refused(
            &[req(0, 1, 1, 1), req(1, i32::MAX as u32, 1, u32::MAX)],
            16,
            u32::MAX,
        );
        // A page size past i32 leaves a last page whose length cannot be held.
        refused(&[req(0, 1, 1, 1 << 31)], 16, u32::MAX);
    }
}
