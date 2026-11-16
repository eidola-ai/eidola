//! A host device whose kernels are the device kernels' own bodies, compiled for the
//! host (`oracle/ds_launch_host.cc` over `eidola-engine-kernels`' `csrc/ds_*.cuh`).
//!
//! Each launch goes through what its wrapper does before a device launch (the
//! parameter checks and the grid, the functions the wrappers themselves call) and then
//! runs the body over that grid. So the executor's scenarios exercise the structs it
//! fills, the wrappers' preconditions, the grids and the C++ a GPU compiles; what is
//! left to a device is the entry points' thread placement and what the compiler emits.

use std::any::Any;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;
use eidola_engine_cuda::deepseek::abi::*;
use eidola_engine_cuda::deepseek::gemm::GemmForm;
use eidola_engine_cuda::deepseek::half::{AcceptArgs, Device, SampleArgs};
use eidola_engine_cuda::deepseek::host::HostDevice;
use eidola_engine_cuda::deepseek::kernels::DsKernels;
use eidola_engine_cuda::deepseek::{attention, gemm_ref, head, index, ops};
use eidola_engine_cuda::{CudaError, Result};

/// The compiled bodies, or `None` (after saying why) without a C++ compiler.
pub fn library() -> Option<Arc<libloading::Library>> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("ds_launch_host: no C++ compiler ({cxx}); the compiled-kernel run did not run");
        return None;
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // One build per test process: tests of one binary share the file.
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "libds_launch_host-{}.{}",
        std::process::id(),
        std::env::consts::DLL_EXTENSION
    ));
    static BUILD: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = BUILD.lock().unwrap_or_else(|e| e.into_inner());
    if !out.exists() {
        let status = Command::new(&cxx)
            // No contraction, as the device build.
            .args([
                "-std=c++17",
                "-O2",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-shared",
                "-fPIC",
            ])
            .arg("-I")
            .arg(crate_dir.join("../eidola-engine-kernels/csrc"))
            .arg(crate_dir.join("oracle/ds_launch_host.cc"))
            .arg("-o")
            .arg(&out)
            .status()
            .expect("run the C++ compiler");
        assert!(status.success(), "{cxx} failed on oracle/ds_launch_host.cc");
    }
    // SAFETY: a library this test just built from the crate's own source; it has no
    // initialisers.
    Some(Arc::new(
        unsafe { libloading::Library::new(&out) }.expect("load the kernel bodies"),
    ))
}

/// Host memory with the compiled kernel bodies.
pub struct CompiledDevice {
    lib: Arc<libloading::Library>,
    host: HostDevice,
}

impl CompiledDevice {
    pub fn new(lib: Arc<libloading::Library>) -> CompiledDevice {
        CompiledDevice {
            lib,
            host: HostDevice,
        }
    }

    /// Runs `symbol` over `grid`.
    ///
    /// # Safety
    ///
    /// `symbol` takes a `P` and `p`'s addresses hold what the kernel reads and writes.
    unsafe fn run<P>(&self, symbol: &str, p: &P, grid: [u32; 3]) -> Result<()> {
        type Launch<P> = unsafe extern "C" fn(*const P, u32, u32, u32);
        // SAFETY: the library defines every `dsh_*` with this signature.
        let f = unsafe { self.lib.get::<Launch<P>>(symbol.as_bytes()) }
            .map_err(|e| CudaError::new(format!("{symbol}: {e}")))?;
        // SAFETY: the caller's.
        unsafe { f(p, grid[0], grid[1], grid[2]) };
        Ok(())
    }
}

/// A family as its wrapper launches it: nothing for no rows, else the wrapper's checks
/// and grid, then the body.
macro_rules! families {
    ($($name:ident: $params:ty, $rows:ident => $grid:expr),* $(,)?) => {
        impl DsKernels for CompiledDevice {
            $(
                unsafe fn $name(&self, p: &$params) -> Result<()> {
                    if p.$rows == 0 {
                        return Ok(());
                    }
                    let grid: fn(&$params) -> Result<[u32; 3]> = $grid;
                    // SAFETY: the caller's contract.
                    unsafe { self.run(concat!("dsh_", stringify!($name)), p, grid(p)?) }
                }
            )*
        }
    };
}

families! {
    embed: EmbedParams, rows => ops::embed_grid,
    quant_fp8: QuantFp8Params, rows => ops::quant_fp8_grid,
    rot_quant: RotQuantParams, rows => ops::rot_quant_grid,
    rms_norm: RmsNormParams, rows => ops::rms_norm_grid,
    mhc_mix: MhcMixParams, rows => ops::mhc_mix_grid,
    collapse: CollapseParams, rows => ops::collapse_grid,
    expand: ExpandParams, rows => ops::expand_grid,
    lin8: Lin8Params, rows => gemm_ref::lin8_grid,
    lin4: Lin4Params, layout_rows => gemm_ref::lin4_grid,
    grouped_dot: GroupedDotParams, rows => gemm_ref::grouped_dot_grid,
    attention: AttnParams, rows => |p| {
        attention::check_attn(p)?;
        attention::attn_grid(p, attention::heads_per_block(p.rows, p.heads))
    },
    linear: LinearParams, rows => ops::linear_grid,
    head: LinearParams, rows => head::head_grid,
    pool: PoolParams, entries => ops::pool_grid,
    index_score: IndexScoreParams, rows => |p| {
        index::check_score(p)?;
        Ok(index::score_grid(p))
    },
    select: SelectParams, rows => |p| {
        index::check_select(p)?;
        Ok([p.rows, 1, 1])
    },
    stream_mean: StreamMeanParams, rows => ops::stream_mean_grid,
    route: RouteParams, rows => ops::route_grid,
    // The placement launches whatever the row count, as its wrapper does.
    place: PlaceParams, groups => ops::place_grid,
    place_rows: PlaceRowsParams, rows => ops::place_rows_grid,
    swiglu: SwigluParams, rows => ops::swiglu_grid,
    combine: CombineParams, rows => ops::combine_grid,
    engram_gather: EngramGatherParams, rows => ops::engram_gather_grid,
    engram_inject: EngramInjectParams, rows => ops::engram_inject_grid,
}

impl Device for CompiledDevice {
    /// The reference bodies: the exact form, and the banded form, whose outputs are
    /// the reference's bit for bit.
    fn select_gemm(&mut self, form: GemmForm) -> Result<()> {
        match form {
            GemmForm::Exact | GemmForm::Banded => Ok(()),
            GemmForm::Tolerance => Err(CudaError::new("the compiled bodies are exact")),
        }
    }

    fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn Any>)> {
        self.host.alloc(bytes)
    }
    unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
        // SAFETY: the caller's, for this and every method below.
        unsafe { self.host.zero(addr, bytes) }
    }
    unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
        unsafe { self.host.upload(addr, src) }
    }
    unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
        unsafe { self.host.download(addr, dst) }
    }
    unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
        unsafe { self.host.copy(dst, src, bytes) }
    }
    unsafe fn cipher(&self, addr: u64, words: u32, key: &KvKey, nonce: &KvNonce) -> Result<()> {
        unsafe { self.host.cipher(addr, words, key, nonce) }
    }
    unsafe fn sample(&self, a: &SampleArgs<'_>) -> Result<()> {
        unsafe { self.host.sample(a) }
    }
    unsafe fn accept(&self, a: &AcceptArgs) -> Result<()> {
        unsafe { self.host.accept(a) }
    }
    fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn Any>)> {
        self.host.alloc_host(bytes)
    }
    unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
        unsafe { self.host.copy_to_host(dst, src, bytes) }
    }
    unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
        unsafe { self.host.copy_from_host(dst, src, bytes) }
    }
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn kernels(&self) -> &dyn DsKernels {
        self
    }
    fn capacity(&self) -> Option<usize> {
        None
    }
}
