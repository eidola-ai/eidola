//! The routed-expert grouped GEMM: DeepGEMM's `sm100_fp8_fp4_gemm_1d1d`
//! instances (FP8 activations × MXFP4 expert weights, BF16 out), launched with
//! the five TMA descriptors DeepGEMM's host code builds
//! (`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`, `runtime_utils.hpp`),
//! encoded here in Rust.
//!
//! The instances fix `N`, `K`, 256 groups, 128×128×128 blocks, a 2-CTA cluster
//! multicasting A (so each CTA loads 64 rows of A), swap-AB, and 128-byte
//! swizzles; M is a runtime argument. Operands, per group `g`:
//!
//! * **A** FP8 e4m3, K-major rows. Scales **SFA**: UE8M0 per (row, 128 of K),
//!   packed four K blocks per `i32` (byte `j` of word `w` covers K block
//!   `4w + j`), stored `[K/512][M']` with rows contiguous (`M'` = M rounded up
//!   to 4).
//! * **B** MXFP4: `[N][K/2]` bytes, element `2i` in the low nibble. Scales
//!   **SFB**: UE8M0 per (row, 32 of K) packed four per `i32` (one word per 128
//!   of K), stored `[K/128][N]`.
//! * **D** BF16 `[M][N]`.
//!
//! Layouts:
//!
//! * **Contiguous** (`MGroupedContiguous`, prefill): A, SFA and D hold every
//!   group's rows back to back, each group's run starting on a multiple of 128
//!   rows; `grouped_layout[row]` is the row's group, or `-1` for padding rows.
//! * **Masked** (`MGroupedMasked`, decode): A is `[G][M][K]`, SFA `[G][K/512][M']`,
//!   D `[G][M][N]`; `grouped_layout[g]` is the number of valid rows of group `g`.
//!
//! B and SFB hold all 256 groups in both layouts.

use std::ffi::c_void;

use crate::module::{Kernel, KernelModule};
use crate::ops::deepgemm_grid;
use crate::tma::{TmaL2, TmaSpec, TmaSwizzle, TmaType};
use crate::{CudaError, Gpu, Result};

/// Groups (routed experts) every instance is built for.
pub const GROUPS: u32 = 256;
/// The block M of the instances: contiguous-layout groups start on multiples
/// of it.
pub const BLOCK_M: u32 = 128;

/// Which grouped layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeLayout {
    /// `m` is the total row count (a multiple of [`BLOCK_M`]).
    Contiguous,
    /// `m` is the per-group row capacity.
    Masked,
}

/// Which projection: the fused gate/up (`N = 4096`, `K = 4096`) or down
/// (`N = 4096`, `K = 2048`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeProj {
    GateUp,
    Down,
}

impl MoeProj {
    pub fn n(self) -> u32 {
        4096
    }

    pub fn k(self) -> u32 {
        match self {
            MoeProj::GateUp => 4096,
            MoeProj::Down => 2048,
        }
    }

    fn meta(self, layout: MoeLayout) -> &'static str {
        match (layout, self) {
            (MoeLayout::Contiguous, MoeProj::GateUp) => {
                "eidola_deepgemm_fp8_fp4_contiguous_gate_up_meta"
            }
            (MoeLayout::Contiguous, MoeProj::Down) => {
                "eidola_deepgemm_fp8_fp4_contiguous_down_meta"
            }
            (MoeLayout::Masked, MoeProj::GateUp) => "eidola_deepgemm_fp8_fp4_masked_gate_up_meta",
            (MoeLayout::Masked, MoeProj::Down) => "eidola_deepgemm_fp8_fp4_masked_down_meta",
        }
    }
}

/// One grouped GEMM's operands (device addresses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeGemmArgs {
    pub layout: MoeLayout,
    pub proj: MoeProj,
    /// Rows: total (contiguous) or per group (masked).
    pub m: u32,
    /// `i32` per row (contiguous) or per group (masked).
    pub grouped_layout: u64,
    pub a: u64,
    pub sfa: u64,
    pub b: u64,
    pub sfb: u64,
    pub d: u64,
}

/// `EpilogueOperatorArgs` (the Identity epilogue adds no state).
#[repr(C)]
#[derive(Clone, Copy)]
struct EpilogueArgs {
    sfd: u64,
    sfd_stride: u32,
    shape_m: u32,
    shape_n: u32,
    alpha: f32,
}

const _: () = assert!(std::mem::size_of::<EpilogueArgs>() == 24);

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct TensorMap([u8; 128]);

impl MoeGemmArgs {
    /// The descriptors DeepGEMM builds: A, B, SFA, SFB, CD.
    pub fn tma_specs(&self) -> [TmaSpec; 5] {
        let (n, k) = (self.proj.n() as u64, self.proj.k() as u64);
        let m = self.m as u64;
        let g = GROUPS as u64;
        let groups_m = match self.layout {
            MoeLayout::Contiguous => 1,
            MoeLayout::Masked => g,
        };
        let m4 = m.div_ceil(4) * 4;
        let spec =
            |dtype, addr, dims: [u64; 2], stride: u64, box_dims: [u32; 2], swizzle| TmaSpec {
                dtype,
                addr,
                dims: dims.to_vec(),
                strides: vec![stride],
                box_dims: box_dims.to_vec(),
                swizzle,
                l2: TmaL2::B256,
            };
        [
            // A: K-major, box (BLOCK_K, LOAD_BLOCK_M = 64).
            spec(
                TmaType::U8,
                self.a,
                [k, m * groups_m],
                k,
                [128, 64],
                TmaSwizzle::B128,
            ),
            // B: packed FP4, K-major, every group stacked along N.
            spec(
                TmaType::Fp4Unpacked,
                self.b,
                [k, n * g],
                k / 2,
                [128, 128],
                TmaSwizzle::B128,
            ),
            // SFA: one word per 512 of K, rows contiguous.
            spec(
                TmaType::I32,
                self.sfa,
                [m4, (k / 512) * groups_m],
                m4 * 4,
                [128, 1],
                TmaSwizzle::None,
            ),
            // SFB: one word per 128 of K.
            spec(
                TmaType::I32,
                self.sfb,
                [n, (k / 128) * g],
                n * 4,
                [128, 1],
                TmaSwizzle::None,
            ),
            // CD: box (128-byte swizzle / 2 bytes, STORE_BLOCK_M = 16).
            spec(
                TmaType::Bf16,
                self.d,
                [n, m * groups_m],
                n * 2,
                [64, 16],
                TmaSwizzle::B128,
            ),
        ]
    }

    pub fn check(&self) -> Result<()> {
        let ok = self.m > 0
            && match self.layout {
                MoeLayout::Contiguous => self.m.is_multiple_of(BLOCK_M),
                MoeLayout::Masked => true,
            };
        if ok {
            Ok(())
        } else {
            Err(CudaError::new(format!("grouped GEMM cannot run {self:?}")))
        }
    }
}

/// The four grouped-GEMM instances.
pub struct MoeGemm {
    _module: KernelModule,
    kernels: Vec<((MoeLayout, MoeProj), Kernel)>,
}

impl MoeGemm {
    pub fn from_module(module: KernelModule) -> Result<MoeGemm> {
        let mut kernels = Vec::new();
        for layout in [MoeLayout::Contiguous, MoeLayout::Masked] {
            for proj in [MoeProj::GateUp, MoeProj::Down] {
                let entry = module
                    .cubin()
                    .entry_for_meta(proj.meta(layout))
                    .ok_or_else(|| CudaError::new(format!("no entry for {}", proj.meta(layout))))?;
                kernels.push(((layout, proj), module.kernel(&entry.symbol)?));
            }
        }
        Ok(MoeGemm {
            _module: module,
            kernels,
        })
    }

    /// Launch on the GPU's stream.
    ///
    /// # Safety
    ///
    /// The operands must hold the layouts the module documents for `args.m`.
    pub unsafe fn launch(&self, gpu: &Gpu, args: &MoeGemmArgs) -> Result<()> {
        args.check()?;
        let kernel = &self
            .kernels
            .iter()
            .find(|(key, _)| *key == (args.layout, args.proj))
            .expect("every instance loaded")
            .1;
        let grid = deepgemm_grid(gpu)?;
        let mut maps = [TensorMap([0; 128]); 5];
        for (map, spec) in maps.iter_mut().zip(args.tma_specs()) {
            map.0 = spec.encode()?;
        }
        let mut grouped_layout = args.grouped_layout;
        let (mut m, mut n, mut k) = (args.m, args.proj.n(), args.proj.k());
        let mut epilogue = EpilogueArgs {
            sfd: 0,
            sfd_stride: 0,
            shape_m: 0,
            shape_n: 0,
            alpha: 1.0,
        };
        let mut params: Vec<*mut c_void> = vec![
            &mut grouped_layout as *mut _ as *mut c_void,
            &mut m as *mut _ as *mut c_void,
            &mut n as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut epilogue as *mut _ as *mut c_void,
        ];
        for map in &mut maps {
            params.push(map as *mut _ as *mut c_void);
        }
        // SAFETY: the parameter list matches the instances' signature
        // `(int*, u32, u32, u32, Identity, CUtensorMap x5)`; operands are the
        // caller's.
        unsafe { kernel.launch(gpu.stream(), grid, &mut params) }
    }
}
