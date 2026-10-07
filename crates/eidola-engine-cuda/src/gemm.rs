//! The CUTLASS GEMMs' launch parameters, built in Rust.
//!
//! Both CUTLASS kernels (`cutlass_fp8_blockwise_gemm`, `cutlass_bf16_gemm`)
//! take CUTLASS 3.x's `GemmUniversal::Params` by value: 2048 bytes holding the
//! problem shape, the mainloop's TMA descriptors (and, for the blockwise
//! kernel, its scale-factor pointers and layouts), the epilogue's scalar and
//! output descriptor, the cluster-launch-control tile scheduler's parameters,
//! and the hardware info. CUTLASS builds them in host C++
//! (`GemmKernel::to_underlying_arguments`); this module builds the same bytes
//! field by field, encoding each descriptor through the driver exactly as CuTe
//! would ([`TmaSpec`]). No host C++ runs.
//!
//! The layout below was read from CUTLASS's types and is pinned by
//! `tests/cutlass_params.rs` against CUTLASS's own host path (the dev-only
//! oracle in `oracle/cutlass_params.cu`), byte for byte on every byte CUTLASS
//! defines; padding is zero here and undefined there.

use std::ffi::c_void;

use crate::module::Kernel;
use crate::tma::{TmaL2, TmaSpec, TmaSwizzle, TmaType};
use crate::{CudaError, Gpu, Result};

/// `sizeof(GemmKernel::Params)` for both kernels.
pub const PARAMS_BYTES: usize = 2048;

/// Which CUTLASS GEMM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmKind {
    /// `D (bf16) = (A·sfa) · (B·sfb)ᵀ`: FP8 e4m3 A `[M, K]` and B `[N, K]`,
    /// f32 scales per 1×128 of A (stored `[K/128][M]`, M contiguous) and per
    /// 128×128 of B (stored `[K/128][N/128]`, N blocks contiguous). M must be
    /// a multiple of 4 (the scale-factor TMA alignment).
    Fp8Blockwise,
    /// `D (f32) = alpha · A · Bᵀ`: BF16 A `[M, K]` and B `[N, K]`.
    Bf16,
}

/// One GEMM's operands (device addresses) and shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GemmArgs {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: u64,
    pub b: u64,
    pub d: u64,
    /// Blockwise kernel only.
    pub sfa: u64,
    /// Blockwise kernel only.
    pub sfb: u64,
    /// BF16 kernel only (the blockwise kernel's epilogue is the identity).
    pub alpha: f32,
}

// Field offsets inside Params (both kernels share the mainloop/epilogue/
// scheduler placement; the blockwise mainloop appends its scale fields).
const OFF_MODE: usize = 0;
const OFF_PROBLEM: usize = 4;
const OFF_TMA_A: usize = 128;
const OFF_TMA_B: usize = 384;
const OFF_TMA_A_FALLBACK: usize = 640;
const OFF_TMA_B_FALLBACK: usize = 896;
const OFF_PTR_SFA: usize = 1184;
const OFF_LAYOUT_SFA: usize = 1192;
const OFF_PTR_SFB: usize = 1216;
const OFF_LAYOUT_SFB: usize = 1224;
const OFF_ALPHA: usize = 1312;
const OFF_TMA_D: usize = 1664;
const OFF_SCHEDULER: usize = 1920;
const OFF_HW_INFO: usize = 1976;

/// The CTA tile (M, N).
const TILE_M: u32 = 128;
const TILE_N: u32 = 128;

impl GemmKind {
    /// The kernel name in the manifest and its entry symbol.
    pub fn kernel(self) -> (&'static str, &'static str) {
        match self {
            GemmKind::Fp8Blockwise => (
                "cutlass_fp8_blockwise_gemm",
                "eidola_cutlass_fp8_blockwise_gemm_bf16",
            ),
            GemmKind::Bf16 => ("cutlass_bf16_gemm", "eidola_cutlass_bf16_gemm_f32"),
        }
    }

    /// The descriptors CuTe builds for A, B and D, in that order.
    pub fn tma_specs(self, a: &GemmArgs) -> [TmaSpec; 3] {
        let (m, n, k) = (a.m as u64, a.n as u64, a.k as u64);
        let spec =
            |dtype, addr, dims: [u64; 3], row_bytes: u64, box_dims: [u32; 3], swizzle| TmaSpec {
                dtype,
                addr,
                dims: dims.to_vec(),
                strides: vec![row_bytes, 0],
                box_dims: box_dims.to_vec(),
                swizzle,
                l2: TmaL2::B128,
            };
        match self {
            GemmKind::Fp8Blockwise => [
                spec(
                    TmaType::U8,
                    a.a,
                    [k, m, 1],
                    k,
                    [128, 128, 1],
                    TmaSwizzle::B128,
                ),
                spec(
                    TmaType::U8,
                    a.b,
                    [k, n, 1],
                    k,
                    [128, 128, 1],
                    TmaSwizzle::B128,
                ),
                spec(
                    TmaType::Bf16,
                    a.d,
                    [n, m, 1],
                    2 * n,
                    [32, 128, 1],
                    TmaSwizzle::B64,
                ),
            ],
            GemmKind::Bf16 => [
                spec(
                    TmaType::Bf16,
                    a.a,
                    [k, m, 1],
                    2 * k,
                    [64, 128, 1],
                    TmaSwizzle::B128,
                ),
                spec(
                    TmaType::Bf16,
                    a.b,
                    [k, n, 1],
                    2 * k,
                    [64, 128, 1],
                    TmaSwizzle::B128,
                ),
                spec(
                    TmaType::F32,
                    a.d,
                    [n, m, 1],
                    4 * n,
                    [32, 128, 1],
                    TmaSwizzle::B128,
                ),
            ],
        }
    }

    /// Whether CUTLASS accepts the shape (`can_implement`): K a multiple of
    /// the 16-byte TMA alignment and, for the blockwise kernel, of the
    /// 128-wide scale block, with M a multiple of 4.
    pub fn check(self, a: &GemmArgs) -> Result<()> {
        let ok = match self {
            GemmKind::Fp8Blockwise => {
                a.m.is_multiple_of(4)
                    && a.k.is_multiple_of(128)
                    && a.n.is_multiple_of(16)
                    && a.m > 0
                    && a.n > 0
            }
            GemmKind::Bf16 => a.k.is_multiple_of(8) && a.n.is_multiple_of(8) && a.m > 0 && a.k > 0,
        };
        if ok {
            Ok(())
        } else {
            Err(CudaError::new(format!("{self:?} GEMM cannot run {a:?}")))
        }
    }

    /// The cluster-launch-control scheduler's rasterization (`Heuristic`):
    /// along M when there are more N tiles than M tiles, else along N unless
    /// that would overflow the grid's Y limit. Returns (along_n, grid).
    pub fn grid(self, a: &GemmArgs) -> (bool, [u32; 3]) {
        let tiles_m = a.m.div_ceil(TILE_M);
        let tiles_n = a.n.div_ceil(TILE_N);
        let along_n = tiles_n <= tiles_m && tiles_m <= 0xffff;
        let grid = if along_n {
            [tiles_n, tiles_m, 1]
        } else {
            [tiles_m, tiles_n, 1]
        };
        (along_n, grid)
    }

    /// The Params bytes and the grid, with each descriptor produced by
    /// `encode` (the driver in production; tests substitute recorded ones).
    pub fn params_with(
        self,
        a: &GemmArgs,
        sm_count: u32,
        mut encode: impl FnMut(&TmaSpec) -> Result<[u8; 128]>,
    ) -> Result<([u8; PARAMS_BYTES], [u32; 3])> {
        self.check(a)?;
        let mut p = [0u8; PARAMS_BYTES];
        let put32 = |p: &mut [u8; PARAMS_BYTES], off: usize, v: u32| {
            p[off..off + 4].copy_from_slice(&v.to_le_bytes())
        };
        let put64 = |p: &mut [u8; PARAMS_BYTES], off: usize, v: u64| {
            p[off..off + 8].copy_from_slice(&v.to_le_bytes())
        };
        // mode = kGemm (0); problem shape (M, N, K, L).
        put32(&mut p, OFF_MODE, 0);
        for (i, v) in [a.m, a.n, a.k, 1].into_iter().enumerate() {
            put32(&mut p, OFF_PROBLEM + 4 * i, v);
        }
        let [sa, sb, sd] = self.tma_specs(a);
        let mut desc = |s: &TmaSpec| -> Result<[u8; 128]> {
            let mut d = encode(s)?;
            s.cute_fixup(&mut d);
            Ok(d)
        };
        let (da, db, dd) = (desc(&sa)?, desc(&sb)?, desc(&sd)?);
        // The cluster shape is static (1x1x1), so the fallback descriptors are
        // built from the same inputs and equal the primary ones.
        for (off, d) in [
            (OFF_TMA_A, &da),
            (OFF_TMA_B, &db),
            (OFF_TMA_A_FALLBACK, &da),
            (OFF_TMA_B_FALLBACK, &db),
            (OFF_TMA_D, &dd),
        ] {
            p[off..off + 128].copy_from_slice(d);
        }
        match self {
            GemmKind::Fp8Blockwise => {
                let kb = a.k.div_ceil(128);
                let nb = a.n.div_ceil(128);
                // layout_SFA: ((1, M), (128, K/128), L) : ((0, 1), (0, M), M·K/128)
                put64(&mut p, OFF_PTR_SFA, a.sfa);
                for (i, v) in [a.m, kb, 1, a.m, a.m * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFA + 4 * i, v);
                }
                // layout_SFB: ((128, N/128), (128, K/128), L) : ((0, 1), (0, N/128), ·)
                put64(&mut p, OFF_PTR_SFB, a.sfb);
                for (i, v) in [nb, kb, 1, nb, nb * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFB + 4 * i, v);
                }
                put32(&mut p, OFF_ALPHA, 1.0f32.to_bits());
            }
            GemmKind::Bf16 => put32(&mut p, OFF_ALPHA, a.alpha.to_bits()),
        }
        // Scheduler: tiles (M, N, L), FastDivmod(cluster m = 1), FastDivmod(
        // cluster n = 1), swizzle divmod unused (divisor 0), raster order,
        // log swizzle 0.
        let (along_n, grid) = self.grid(a);
        let s = OFF_SCHEDULER;
        put32(&mut p, s, a.m.div_ceil(TILE_M));
        put32(&mut p, s + 4, a.n.div_ceil(TILE_N));
        put32(&mut p, s + 8, 1);
        put32(&mut p, s + 12, 1);
        put32(&mut p, s + 24, 1);
        put32(&mut p, s + 48, along_n as u32);
        // hw_info: device 0, SM count; no cluster overrides.
        put32(&mut p, OFF_HW_INFO + 4, sm_count);
        Ok((p, grid))
    }
}

/// A loaded CUTLASS GEMM kernel.
pub struct Gemm {
    kind: GemmKind,
    kernel: Kernel,
}

impl Gemm {
    pub fn new(kind: GemmKind, kernel: Kernel) -> Result<Gemm> {
        if kernel.meta().params_bytes as usize != PARAMS_BYTES {
            return Err(CudaError::new(format!(
                "{kind:?}: the image's Params is {} bytes, the host mirror {PARAMS_BYTES}",
                kernel.meta().params_bytes
            )));
        }
        Ok(Gemm { kind, kernel })
    }

    pub fn kind(&self) -> GemmKind {
        self.kind
    }

    /// Launch on the GPU's stream.
    ///
    /// # Safety
    ///
    /// The operand addresses must cover the shapes [`GemmKind`] documents
    /// (the blockwise kernel reads `M` rows of A and of `sfa`, rounded up to
    /// what its tiles touch through TMA, which clamps at the extents).
    pub unsafe fn launch(&self, gpu: &Gpu, a: &GemmArgs) -> Result<()> {
        let (mut params, grid) = self
            .kind
            .params_with(a, gpu.info().sm_count, |s| s.encode())?;
        let mut args = [params.as_mut_ptr() as *mut c_void];
        // SAFETY: the single argument is the kernel's by-value Params, whose
        // size the launch contract confirmed; operands are the caller's.
        unsafe { self.kernel.launch(gpu.stream(), grid, &mut args) }
    }
}
