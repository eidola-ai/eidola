//! The CUTLASS GEMMs' launch parameters, built in Rust.
//!
//! The CUTLASS kernels (`cutlass_fp8_blockwise_gemm`, `cutlass_bf16_gemm`, the
//! swapped FP8 instances of `cutlass_fp8_blockwise_swap`, and the split FP8
//! instances of `cutlass_fp8_blockwise_split`) take CUTLASS 3.x's
//! `GemmUniversal::Params` by value: 2048 bytes holding the
//! problem shape, the mainloop's TMA descriptors (and, for the blockwise
//! kernel, its scale-factor pointers and layouts), the epilogue's scalar and
//! output descriptor, the cluster-launch-control tile scheduler's parameters,
//! and the hardware info. CUTLASS builds them in host C++
//! (`GemmKernel::to_underlying_arguments`); this module builds the same bytes
//! field by field, encoding each descriptor through the driver exactly as CuTe
//! would ([`TmaSpec`]). No host C++ runs.
//!
//! The layout below was read from CUTLASS's types and is pinned by
//! `tests/mimo_cutlass_params.rs` against CUTLASS's own host path (the dev-only
//! oracle in `oracle/cutlass_params.cu`), byte for byte on every byte CUTLASS
//! defines; padding is zero here and undefined there.

use std::ffi::c_void;

use crate::mimo::tma::{TmaL2, TmaSpec, TmaSwizzle, TmaType};
use crate::module::{Kernel, KernelModule};
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
    /// [`GemmKind::Fp8Blockwise`]'s product with the same operands and scale
    /// arrays, computed transposed (`cutlass_fp8_blockwise_swap`): CUTLASS's A
    /// is the weight, its B the `M` activation rows, tiled 16, 32 or 64 rows at
    /// a time, its D `[N, M]` column-major (the same bytes as `[M, N]`
    /// row-major). For few rows.
    Fp8Swap(SwapTile),
    /// The FP8 product of [`GemmKind::Fp8Blockwise`] over `splits` equal ranges
    /// of K at once (`cutlass_fp8_blockwise_split`), each range's sums left
    /// unrounded: `D[s]` (f32 `[M, N]`, the ranges' arrays one after another)
    /// is the product of K columns `s · K/splits .. (s + 1) · K/splits` of A and
    /// B. CUTLASS runs each range as one batch of its problem (its L mode). The
    /// tile names the instance: the swapped kernel's operands at 16, 32 or 64
    /// activation rows, or the unswapped kernel's 128 × 128.
    Fp8Split(SplitTile, u32),
}

/// The instance of a split FP8 GEMM ([`GemmKind::Fp8Split`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SplitTile {
    /// Swapped, at [`SwapTile`]'s activation rows.
    Swap(SwapTile),
    /// Unswapped, 128 × 128 tiles.
    Wide,
}

impl SplitTile {
    pub const ALL: [SplitTile; 4] = [
        SplitTile::Swap(SwapTile::T16),
        SplitTile::Swap(SwapTile::T32),
        SplitTile::Swap(SwapTile::T64),
        SplitTile::Wide,
    ];
}

/// The activation rows a swapped FP8 GEMM's tile holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwapTile {
    T16,
    T32,
    T64,
}

impl SwapTile {
    pub const ALL: [SwapTile; 3] = [SwapTile::T16, SwapTile::T32, SwapTile::T64];

    pub fn rows(self) -> u32 {
        match self {
            SwapTile::T16 => 16,
            SwapTile::T32 => 32,
            SwapTile::T64 => 64,
        }
    }
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
            GemmKind::Fp8Swap(t) => (
                "cutlass_fp8_blockwise_swap",
                match t {
                    SwapTile::T16 => "eidola_cutlass_fp8_blockwise_swap16_bf16",
                    SwapTile::T32 => "eidola_cutlass_fp8_blockwise_swap32_bf16",
                    SwapTile::T64 => "eidola_cutlass_fp8_blockwise_swap64_bf16",
                },
            ),
            GemmKind::Fp8Split(t, _) => (
                "cutlass_fp8_blockwise_split",
                match t {
                    SplitTile::Swap(SwapTile::T16) => "eidola_cutlass_fp8_blockwise_split16_f32",
                    SplitTile::Swap(SwapTile::T32) => "eidola_cutlass_fp8_blockwise_split32_f32",
                    SplitTile::Swap(SwapTile::T64) => "eidola_cutlass_fp8_blockwise_split64_f32",
                    SplitTile::Wide => "eidola_cutlass_fp8_blockwise_split128_f32",
                },
            ),
        }
    }

    /// Batches of CUTLASS's problem (its L): a split kind's ranges of K, else 1.
    pub fn splits(self) -> u32 {
        match self {
            GemmKind::Fp8Split(_, s) => s,
            _ => 1,
        }
    }

    /// Whether CUTLASS's A is the weight (the swapped instances).
    fn swapped(self) -> bool {
        matches!(
            self,
            GemmKind::Fp8Swap(_) | GemmKind::Fp8Split(SplitTile::Swap(_), _)
        )
    }

    /// CUTLASS's own problem (M, N): the transposed one for a swapped kind.
    fn cutlass_mn(self, a: &GemmArgs) -> (u32, u32) {
        if self.swapped() {
            (a.n, a.m)
        } else {
            (a.m, a.n)
        }
    }

    /// CUTLASS's CTA tile (M, N).
    fn tile(self) -> (u32, u32) {
        match self {
            GemmKind::Fp8Swap(t) | GemmKind::Fp8Split(SplitTile::Swap(t), _) => (TILE_M, t.rows()),
            _ => (TILE_M, TILE_N),
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
            GemmKind::Fp8Swap(t) => [
                spec(
                    TmaType::U8,
                    a.b,
                    [k, n, 1],
                    k,
                    [128, 128, 1],
                    TmaSwizzle::B128,
                ),
                spec(
                    TmaType::U8,
                    a.a,
                    [k, m, 1],
                    k,
                    [128, t.rows(), 1],
                    TmaSwizzle::B128,
                ),
                spec(
                    TmaType::Bf16,
                    a.d,
                    [n, m, 1],
                    2 * n,
                    [64, t.rows().min(32), 1],
                    TmaSwizzle::B128,
                ),
            ],
            GemmKind::Fp8Split(t, splits) => {
                // Each batch is one range of K: `k / splits` columns of the same rows,
                // the next batch `k / splits` elements on; its output the next array.
                let s = u64::from(splits.max(1));
                let ks = k / s;
                let split =
                    |dtype, addr, rows: u64, row_bytes: u64, step: u64, box_dims, swizzle| {
                        TmaSpec {
                            dtype,
                            addr,
                            dims: vec![if dtype == TmaType::F32 { n } else { ks }, rows, s],
                            // The output's packed stride is zero for one batch
                            // (`make_cute_packed_stride`); the operands' is the range's.
                            strides: vec![
                                row_bytes,
                                if s > 1 || dtype != TmaType::F32 {
                                    step
                                } else {
                                    0
                                },
                            ],
                            box_dims: Vec::from(box_dims),
                            swizzle,
                            l2: TmaL2::B128,
                        }
                    };
                let rows = match t {
                    SplitTile::Swap(t) => t.rows(),
                    SplitTile::Wide => TILE_N,
                };
                let weight = split(TmaType::U8, a.b, n, k, ks, [128, 128, 1], TmaSwizzle::B128);
                let act = split(TmaType::U8, a.a, m, k, ks, [128, rows, 1], TmaSwizzle::B128);
                match t {
                    SplitTile::Swap(_) => {
                        let out = split(
                            TmaType::F32,
                            a.d,
                            m,
                            4 * n,
                            4 * n * m,
                            [32, rows.min(32), 1],
                            TmaSwizzle::B128Atom32,
                        );
                        [weight, act, out]
                    }
                    SplitTile::Wide => {
                        let out = split(
                            TmaType::F32,
                            a.d,
                            m,
                            4 * n,
                            4 * n * m,
                            [32, 128, 1],
                            TmaSwizzle::B128,
                        );
                        [act, weight, out]
                    }
                }
            }
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
    /// the 16-byte TMA alignment and, for the blockwise kernels, of the
    /// 128-wide scale block, with M a multiple of 4 for the unswapped one.
    pub fn check(self, a: &GemmArgs) -> Result<()> {
        let ok = match self {
            // The swapped kernel's activation scales are not a TMA operand, so
            // any row count serves.
            GemmKind::Fp8Swap(_) => {
                a.k.is_multiple_of(128) && a.n.is_multiple_of(16) && a.m > 0 && a.n > 0
            }
            GemmKind::Fp8Blockwise => {
                a.m.is_multiple_of(4)
                    && a.k.is_multiple_of(128)
                    && a.n.is_multiple_of(16)
                    && a.m > 0
                    && a.n > 0
            }
            GemmKind::Bf16 => a.k.is_multiple_of(8) && a.n.is_multiple_of(8) && a.m > 0 && a.k > 0,
            // Every range a whole number of 128-wide scale blocks; the unswapped
            // instance's rows a multiple of 4, as for its BF16 form.
            GemmKind::Fp8Split(t, splits) => {
                splits > 0
                    && a.k.is_multiple_of(128 * splits)
                    && a.n.is_multiple_of(16)
                    && a.m > 0
                    && a.n > 0
                    && (t != SplitTile::Wide || a.m.is_multiple_of(4))
            }
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
        let (tiles_m, tiles_n) = self.tiles(a);
        let along_n = tiles_n <= tiles_m && tiles_m <= 0xffff;
        let l = self.splits();
        let grid = if along_n {
            [tiles_n, tiles_m, l]
        } else {
            [tiles_m, tiles_n, l]
        };
        (along_n, grid)
    }

    /// CUTLASS's output tiles along its M and N.
    fn tiles(self, a: &GemmArgs) -> (u32, u32) {
        let (m, n) = self.cutlass_mn(a);
        let (tm, tn) = self.tile();
        (m.div_ceil(tm), n.div_ceil(tn))
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
        let (cm, cn) = self.cutlass_mn(a);
        let splits = self.splits();
        for (i, v) in [cm, cn, a.k / splits, splits].into_iter().enumerate() {
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
            GemmKind::Fp8Split(SplitTile::Wide, _) => {
                // A range's blocks: the K-block mode spans one range, the L mode
                // steps from range to range (each scale array is K-block major).
                let kb = a.k / 128 / splits;
                let nb = a.n.div_ceil(128);
                put64(&mut p, OFF_PTR_SFA, a.sfa);
                for (i, v) in [a.m, kb, splits, a.m, a.m * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFA + 4 * i, v);
                }
                put64(&mut p, OFF_PTR_SFB, a.sfb);
                for (i, v) in [nb, kb, splits, nb, nb * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFB + 4 * i, v);
                }
                put32(&mut p, OFF_ALPHA, 1.0f32.to_bits());
            }
            GemmKind::Fp8Split(SplitTile::Swap(_), _) => {
                let kb = a.k / 128 / splits;
                let nb = a.n.div_ceil(128);
                put64(&mut p, OFF_PTR_SFA, a.sfb);
                for (i, v) in [nb, kb, splits, nb, nb * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFA + 4 * i, v);
                }
                put64(&mut p, OFF_PTR_SFB, a.sfa);
                for (i, v) in [a.m, kb, splits, a.m, a.m * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFB + 4 * i, v);
                }
                put32(&mut p, OFF_ALPHA, 1.0f32.to_bits());
            }
            GemmKind::Fp8Swap(_) => {
                let kb = a.k.div_ceil(128);
                let nb = a.n.div_ceil(128);
                // CUTLASS's SFA is the weight's scales: ((128, N/128), (128, K/128), L) :
                // ((0, 1), (0, N/128), ·).
                put64(&mut p, OFF_PTR_SFA, a.sfb);
                for (i, v) in [nb, kb, 1, nb, nb * kb].into_iter().enumerate() {
                    put32(&mut p, OFF_LAYOUT_SFA + 4 * i, v);
                }
                // Its SFB the activations': ((1, M), (128, K/128), L) : ((0, 1), (0, M), ·).
                put64(&mut p, OFF_PTR_SFB, a.sfa);
                for (i, v) in [a.m, kb, 1, a.m, a.m * kb].into_iter().enumerate() {
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
        let (tiles_m, tiles_n) = self.tiles(a);
        let s = OFF_SCHEDULER;
        put32(&mut p, s, tiles_m);
        put32(&mut p, s + 4, tiles_n);
        put32(&mut p, s + 8, splits);
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
    /// The GEMM of `kind`, from its own image and entry only: the Params
    /// layout built for one kind never reaches another kind's kernel.
    pub fn from_module(kind: GemmKind, module: &KernelModule) -> Result<Gemm> {
        let (image, entry) = kind.kernel();
        let kernel = module.bound_kernel(image, entry)?;
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
        // SAFETY: the caller's.
        unsafe { self.launch_as(gpu, self.kind, a) }
    }

    /// A split kind's launch at `splits` ranges of K: the instance is the loaded one,
    /// the split a field of its Params.
    ///
    /// # Safety
    ///
    /// As [`Gemm::launch`], `D` holding `splits` arrays of `M × N` f32.
    pub unsafe fn launch_splits(&self, gpu: &Gpu, splits: u32, a: &GemmArgs) -> Result<()> {
        let GemmKind::Fp8Split(t, _) = self.kind else {
            return Err(CudaError::new(format!(
                "{:?} is not a split GEMM",
                self.kind
            )));
        };
        // SAFETY: the caller's.
        unsafe { self.launch_as(gpu, GemmKind::Fp8Split(t, splits), a) }
    }

    /// # Safety
    ///
    /// As [`Gemm::launch`]; `kind` names this kernel's image and entry.
    unsafe fn launch_as(&self, gpu: &Gpu, kind: GemmKind, a: &GemmArgs) -> Result<()> {
        debug_assert_eq!(kind.kernel(), self.kind.kernel());
        // Descriptor encoding needs this thread's current context.
        gpu.context().bind_to_thread()?;
        let (mut params, grid) = kind.params_with(a, gpu.info().sm_count, |s| s.encode())?;
        let mut args = [params.as_mut_ptr() as *mut c_void];
        // SAFETY: the single argument is the kernel's by-value Params, whose
        // size the launch contract confirmed; operands are the caller's.
        unsafe { self.kernel.launch(gpu.stream(), grid, &mut args) }
    }
}
