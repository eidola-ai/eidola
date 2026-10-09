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
//! The layout is DeepGEMM's psum one (`MGroupedContiguousWithPsumLayout`) at
//! every token count: A, SFA and D hold every group's rows back to back in
//! group order, group `g`'s run starting at the previous run's end rounded up
//! to 128 rows; `grouped_layout[g]` is the end of group `g`'s rows (a prefix
//! sum over the groups, 256 words). The scheduler visits only each group's
//! own blocks and narrows a group's last block to its rows rounded up to 16,
//! so the work follows the routed rows, not M. M (the layout's row bound) is
//! the launch's only shape argument; the routing is device data, read by the
//! kernel. B and SFB hold all 256 groups.

use std::ffi::c_void;

use crate::module::{Kernel, KernelModule};
use crate::ops::deepgemm_grid;
use crate::tma::{TmaL2, TmaSpec, TmaSwizzle, TmaType};
use crate::{CudaError, Gpu, Result};

/// Groups (routed experts) every instance is built for.
pub const GROUPS: u32 = 256;
/// The block M of the instances: groups start on multiples of it.
pub const BLOCK_M: u32 = 128;

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

    fn meta(self) -> &'static str {
        match self {
            MoeProj::GateUp => "eidola_deepgemm_fp8_fp4_psum_gate_up_meta",
            MoeProj::Down => "eidola_deepgemm_fp8_fp4_psum_down_meta",
        }
    }
}

/// One grouped GEMM's operands (device addresses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeGemmArgs {
    pub proj: MoeProj,
    /// The rows the layout holds (a multiple of [`BLOCK_M`]): the
    /// descriptors' bound, not the work.
    pub m: u32,
    /// `i32` per group: the end of its rows.
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
    /// The descriptors DeepGEMM builds for the executor's instances: A, B,
    /// SFA, SFB, CD.
    pub fn tma_specs(&self) -> [TmaSpec; 5] {
        self.tma_specs_for(MoeTile::EXECUTOR)
    }

    /// The descriptors for an instance with `tile`: only A's box depends on
    /// it (each CTA loads `block_m / cluster` rows of A).
    pub fn tma_specs_for(&self, tile: MoeTile) -> [TmaSpec; 5] {
        let (n, k) = (self.proj.n() as u64, self.proj.k() as u64);
        let m = self.m as u64;
        let g = GROUPS as u64;
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
            // A: K-major, box (BLOCK_K, LOAD_BLOCK_M; 64 for the executor's).
            spec(
                TmaType::U8,
                self.a,
                [k, m],
                k,
                [128, tile.load_block_m()],
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
                [m4, k / 512],
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
                [n, m],
                n * 2,
                [64, 16],
                TmaSwizzle::B128,
            ),
        ]
    }

    pub fn check(&self) -> Result<()> {
        self.check_for(MoeTile::EXECUTOR)
    }

    /// The layout's bound is whole blocks of `tile.block_m` rows.
    pub fn check_for(&self, tile: MoeTile) -> Result<()> {
        if self.m > 0 && self.m.is_multiple_of(tile.block_m) {
            Ok(())
        } else {
            Err(CudaError::new(format!(
                "grouped GEMM ({tile:?}) cannot run {self:?}"
            )))
        }
    }
}

/// What a grouped-GEMM instance's host side depends on: its block M (the
/// psum layout's run alignment, and the unit of the layout's bound) and its
/// cluster size (the CTAs that share an A tile, each loading
/// `block_m / cluster` of its rows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeTile {
    pub block_m: u32,
    pub cluster: u32,
}

impl MoeTile {
    /// The executor's instances: block M 128, a 2-CTA cluster.
    pub const EXECUTOR: MoeTile = MoeTile {
        block_m: BLOCK_M,
        cluster: 2,
    };

    pub fn load_block_m(self) -> u32 {
        self.block_m / self.cluster
    }
}

/// The two grouped-GEMM instances.
pub struct MoeGemm {
    _module: KernelModule,
    kernels: Vec<(MoeProj, Kernel)>,
}

impl MoeGemm {
    pub fn from_module(module: KernelModule) -> Result<MoeGemm> {
        module.expect_image("deepgemm_fp8_fp4_grouped")?;
        let mut kernels = Vec::new();
        for proj in [MoeProj::GateUp, MoeProj::Down] {
            let entry = module
                .cubin()
                .entry_for_meta(proj.meta())
                .ok_or_else(|| CudaError::new(format!("no entry for {}", proj.meta())))?;
            kernels.push((proj, module.kernel(&entry.symbol)?));
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
        let kernel = &self
            .kernels
            .iter()
            .find(|(proj, _)| *proj == args.proj)
            .expect("every instance loaded")
            .1;
        // SAFETY: the caller's contract.
        unsafe { launch_instance(gpu, kernel, MoeTile::EXECUTOR, args) }
    }
}

/// The experimental tilings of the grouped GEMM in the
/// `deepgemm_fp8_fp4_grouped_variants` image (its source lists each one's
/// template arguments and shared memory), for the kernel bench only: the
/// executor runs [`MoeGemm`]'s instances and never loads this image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeVariant {
    /// Block M 64, 2-CTA cluster, 8 stages (the executor's stage count).
    M64S8,
    /// Block M 64, 2-CTA cluster, 10 stages.
    M64S10,
    /// Block M 32, 2-CTA cluster, 11 stages.
    M32S11,
    /// Block M 64, no cluster (1-CTA UMMA), 8 stages.
    M64S8Cta1,
}

impl MoeVariant {
    pub const ALL: [MoeVariant; 4] = [
        MoeVariant::M64S8,
        MoeVariant::M64S10,
        MoeVariant::M32S11,
        MoeVariant::M64S8Cta1,
    ];

    /// The name its launch-contract records carry.
    pub fn name(self) -> &'static str {
        match self {
            MoeVariant::M64S8 => "m64s8",
            MoeVariant::M64S10 => "m64s10",
            MoeVariant::M32S11 => "m32s11",
            MoeVariant::M64S8Cta1 => "m64s8c1",
        }
    }

    pub fn tile(self) -> MoeTile {
        let (block_m, cluster) = match self {
            MoeVariant::M64S8 | MoeVariant::M64S10 => (64, 2),
            MoeVariant::M32S11 => (32, 2),
            MoeVariant::M64S8Cta1 => (64, 1),
        };
        MoeTile { block_m, cluster }
    }

    /// Pipeline stages it was built with (informational: nothing on the
    /// host depends on it).
    pub fn stages(self) -> u32 {
        match self {
            MoeVariant::M64S8 | MoeVariant::M64S8Cta1 => 8,
            MoeVariant::M64S10 => 10,
            MoeVariant::M32S11 => 11,
        }
    }

    /// The launch-contract record of its `proj` instance.
    pub fn meta(self, proj: MoeProj) -> String {
        let proj = match proj {
            MoeProj::GateUp => "gate_up",
            MoeProj::Down => "down",
        };
        format!("eidola_deepgemm_variant_{}_{proj}_meta", self.name())
    }
}

/// Every instance of the variants image.
pub struct MoeGemmVariants {
    _module: KernelModule,
    kernels: Vec<(MoeVariant, MoeProj, Kernel)>,
}

impl MoeGemmVariants {
    pub const IMAGE: &str = "deepgemm_fp8_fp4_grouped_variants";

    pub fn from_module(module: KernelModule) -> Result<MoeGemmVariants> {
        module.expect_image(Self::IMAGE)?;
        let mut kernels = Vec::new();
        for variant in MoeVariant::ALL {
            for proj in [MoeProj::GateUp, MoeProj::Down] {
                let meta = variant.meta(proj);
                let entry = module
                    .cubin()
                    .entry_for_meta(&meta)
                    .ok_or_else(|| CudaError::new(format!("no entry for {meta}")))?;
                kernels.push((variant, proj, module.kernel(&entry.symbol)?));
            }
        }
        Ok(MoeGemmVariants {
            _module: module,
            kernels,
        })
    }

    /// Launch `variant` on the GPU's stream.
    ///
    /// # Safety
    ///
    /// The operands must hold the layouts the module documents for `args.m`,
    /// with every run starting on a multiple of `variant.tile().block_m`
    /// rows (`engine_ops::expert_placement_aligned`).
    pub unsafe fn launch(&self, gpu: &Gpu, variant: MoeVariant, args: &MoeGemmArgs) -> Result<()> {
        let kernel = &self
            .kernels
            .iter()
            .find(|(v, proj, _)| *v == variant && *proj == args.proj)
            .expect("every instance loaded")
            .2;
        // SAFETY: the caller's contract.
        unsafe { launch_instance(gpu, kernel, variant.tile(), args) }
    }
}

/// Launch one instance built with `tile` over `args`.
///
/// # Safety
///
/// `kernel` is a `sm100_fp8_fp4_gemm_1d1d_impl` instance for `args.proj`
/// built with `tile`, and the operands hold the layouts the module documents
/// for `args.m`, with runs aligned to `tile.block_m`.
unsafe fn launch_instance(
    gpu: &Gpu,
    kernel: &Kernel,
    tile: MoeTile,
    args: &MoeGemmArgs,
) -> Result<()> {
    args.check_for(tile)?;
    let grid = deepgemm_grid(gpu)?;
    // Descriptor encoding needs this thread's current context.
    gpu.context().bind_to_thread()?;
    let mut maps = [TensorMap([0; 128]); 5];
    for (map, spec) in maps.iter_mut().zip(args.tma_specs_for(tile)) {
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
