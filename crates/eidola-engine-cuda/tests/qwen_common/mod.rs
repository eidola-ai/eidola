//! What the Qwen3.8 tests share: the kernels' bodies compiled for the host
//! (`oracle/qwen_launch_host.cc` over `eidola-engine-kernels`' `csrc/qwen_ops.cuh`,
//! `qwen_fast.cuh` and `qwen_fast_deltanet.cuh`) as a
//! [`HostDevice`], buffers on any [`QwenDevice`], a seeded generator, and the loud skip of
//! a device test.
#![allow(dead_code)]

pub mod cases;
pub mod chunk;
pub mod invariance;
pub mod pair;
pub mod scenarios;

use std::any::Any;
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use eidola_engine_cuda::qwen::host::HostDevice;
use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath, QwenCudaExecutor, QwenDevice};

/// The compiled bodies, or `None` (after saying why) without a C++ compiler.
pub fn library() -> Option<Arc<libloading::Library>> {
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("qwen_launch_host: no C++ compiler ({cxx}); the compiled-kernel run did not run");
        return None;
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Built once per test process, always afresh: a file left by an earlier process (a
    // reused process id) could hold other bodies.
    static BUILT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    let out = BUILT
        .get_or_init(|| {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos());
            let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
                "libqwen_launch_host-{}-{nanos}.{}",
                std::process::id(),
                std::env::consts::DLL_EXTENSION
            ));
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
                .arg(crate_dir.join("oracle/qwen_launch_host.cc"))
                .arg("-o")
                .arg(&out)
                .status()
                .expect("run the C++ compiler");
            assert!(
                status.success(),
                "{cxx} failed on oracle/qwen_launch_host.cc"
            );
            out
        })
        .clone();
    // SAFETY: a library this test just built from the crate's own source; it has no
    // initialisers.
    Some(Arc::new(
        unsafe { libloading::Library::new(&out) }.expect("load the kernel bodies"),
    ))
}

type HostFn = unsafe extern "C" fn(*const c_void, u32, u32);

/// A [`HostDevice`] whose launches run the compiled bodies.
pub fn host_device(lib: Arc<libloading::Library>) -> HostDevice {
    HostDevice::new(host_launch(lib))
}

/// Each launch run by its family's host body (`qwen_host_<family>`).
pub fn host_launch(
    lib: Arc<libloading::Library>,
) -> Box<eidola_engine_cuda::qwen::host::HostLaunch> {
    Box::new(move |family, params, gx, gy| {
        let name = format!("qwen_host_{family}\0");
        // SAFETY: every `qwen_host_*` symbol has this signature, and takes the family's
        // parameter struct, which the device built for it.
        unsafe {
            let f: libloading::Symbol<HostFn> =
                lib.get(name.as_bytes()).expect("a host body per family");
            f(params, gx, gy);
        }
    })
}

/// A [`HostDevice`] whose launches run the compiled bodies, counting the launches of each
/// family into `counts`.
pub fn counting_device(
    lib: Arc<libloading::Library>,
    counts: Arc<std::sync::Mutex<std::collections::HashMap<&'static str, u64>>>,
) -> HostDevice {
    let run = host_launch(lib);
    HostDevice::new(Box::new(move |family, params, gx, gy| {
        *counts.lock().expect("counts").entry(family).or_default() += 1;
        run(family, params, gx, gy);
    }))
}

/// The quantiser's reference form (`quant_thread`) compiled for the host, over host
/// buffers: `(params, grid x, grid y)`, the grid a thread per 128-wide block.
pub fn quant_thread_reference(
    lib: Arc<libloading::Library>,
) -> impl Fn(&eidola_engine_cuda::qwen::abi::QuantParams, u32, u32) {
    move |p, gx, gy| {
        // SAFETY: `qwen_host_quant_thread` takes a `QwQuantParams` and the grid; the
        // caller's addresses are host buffers holding what the struct names.
        unsafe {
            let f: libloading::Symbol<HostFn> = lib
                .get(b"qwen_host_quant_thread\0")
                .expect("the reference quantiser");
            f(
                (p as *const eidola_engine_cuda::qwen::abi::QuantParams).cast(),
                gx,
                gy,
            );
        }
    }
}

/// A body compiled for the host, by its symbol (`qwen_host_*`), over host buffers:
/// `(params, grid x, grid y)`. The reference forms of the kernels arranged for speed
/// (`qwen_host_norm_reference`, `qwen_host_dn_norm_reference`,
/// `qwen_host_widen_reference`) run over their own grids: a CTA a (unit, row), a CTA a
/// (head, row), and an element a thread.
pub fn host_body<P>(lib: Arc<libloading::Library>, symbol: &'static str) -> impl Fn(&P, u32, u32) {
    move |p, gx, gy| {
        let name = format!("{symbol}\0");
        // SAFETY: every `qwen_host_*` launch symbol takes its family's parameter struct
        // and the grid; the caller's addresses are host buffers holding what it names.
        unsafe {
            let f: libloading::Symbol<HostFn> = lib.get(name.as_bytes()).expect("a host body");
            f((p as *const P).cast(), gx, gy);
        }
    }
}

/// `qwk::e4m3_code` against `ds::e4m3_encode` over every f32 bit pattern, in eight
/// ranges at once: the patterns that differ (none, or the first of each range's).
pub fn e4m3_code_mismatches(lib: &libloading::Library) -> Vec<(u64, u32)> {
    type Check = unsafe extern "C" fn(u32, u32, *mut u32) -> u64;
    // SAFETY: `qwen_host_e4m3_check` has this signature and touches nothing but `at`.
    let check: libloading::Symbol<Check> =
        unsafe { lib.get(b"qwen_host_e4m3_check\0") }.expect("the encoder check");
    let check = *check;
    std::thread::scope(|s| {
        let parts: Vec<_> = (0..8u64)
            .map(|i| {
                s.spawn(move || {
                    let first = u32::try_from(i << 29).expect("a 32-bit pattern");
                    let last = u32::try_from(((i + 1) << 29) - 1).expect("a 32-bit pattern");
                    let mut at = 0u32;
                    // SAFETY: as above; `at` is this thread's own.
                    let bad = unsafe { check(first, last, &mut at) };
                    (bad, at)
                })
            })
            .collect();
        parts
            .into_iter()
            .map(|h| h.join().expect("a check thread"))
            .filter(|&(bad, _)| bad > 0)
            .collect()
    })
}

/// The fast profile's small-N GEMM body compiled for the host (`qwen_host_fast_gemm`),
/// over host buffers: a [`cases::small_gemm`] runner.
pub fn small_gemm_host(lib: Arc<libloading::Library>) -> impl cases::SmallGemmRun {
    small_gemm_host_form(lib, false)
}

/// The small-N GEMM's tiled form compiled for the host (`qwen_host_fast_gemm_tiled`):
/// a [`cases::small_gemm`] runner.
pub fn small_gemm_tiled_host(lib: Arc<libloading::Library>) -> impl cases::SmallGemmRun {
    small_gemm_host_form(lib, true)
}

fn small_gemm_host_form(lib: Arc<libloading::Library>, tiled: bool) -> impl cases::SmallGemmRun {
    move |x, w, rows, first, n, k| {
        let mut out = vec![0.0f32; rows * n];
        let p = eidola_engine_cuda::qwen::abi::LinearParams {
            x: x[first * k..].as_ptr() as u64,
            w: w.as_ptr() as u64,
            out: out.as_mut_ptr() as u64,
            rows: u32::try_from(rows).expect("rows"),
            n: u32::try_from(n).expect("n"),
            k: u32::try_from(k).expect("k"),
            x_stride: u32::try_from(k).expect("k"),
            out_stride: u32::try_from(n).expect("n"),
            out_f32: 1,
        };
        use eidola_engine_cuda::qwen::kernels::{small_gemm_grid, small_gemm_tiled_grid};
        let (grid, symbol): (_, &[u8]) = if tiled {
            (small_gemm_tiled_grid(&p), b"qwen_host_fast_gemm_tiled\0")
        } else {
            (small_gemm_grid(&p), b"qwen_host_fast_gemm\0")
        };
        let [gx, gy, _] = grid.map_err(|e| e.to_string())?;
        // SAFETY: `qwen_host_fast_gemm` and `qwen_host_fast_gemm_tiled` take a
        // `QwLinearParams` and the grid; the addresses are the host vectors above, alive for
        // the call.
        unsafe {
            let f: libloading::Symbol<HostFn> = lib.get(symbol).expect("the small-N GEMM body");
            f(
                (&p as *const eidola_engine_cuda::qwen::abi::LinearParams).cast(),
                gx,
                gy,
            );
        }
        Ok(out)
    }
}

/// A [`HostDevice`] planning as the fast profile: the compiled bodies, the DeltaNet in
/// blocks, and FA2's launches checked and computing nothing.
pub fn planning_fast(lib: Arc<libloading::Library>) -> HostDevice {
    HostDevice::planning_fast(host_launch(lib))
}

/// A [`HostDevice`] computing the exact profile but for the DeltaNet, which runs in
/// blocks as the fast profile's does (the intra kernel's body and the state kernel's host
/// form).
pub fn block_deltanet(lib: Arc<libloading::Library>) -> HostDevice {
    HostDevice::block_deltanet(host_launch(lib))
}

/// The intra kernel's body compiled for the host (`qwen_host_dn_intra`), over host
/// buffers.
pub fn dn_intra_host(
    lib: Arc<libloading::Library>,
) -> impl Fn(&eidola_engine_cuda::qwen::abi::DnChunkParams) -> Result<(), String> {
    move |p| {
        let [gx, gy, _] =
            eidola_engine_cuda::qwen::kernels::dn_intra_grid(p).map_err(|e| e.to_string())?;
        // SAFETY: `qwen_host_dn_intra` takes a `QwDnChunkParams` and the grid; the caller's
        // addresses are host buffers holding what the struct names.
        unsafe {
            let f: libloading::Symbol<HostFn> = lib
                .get(b"qwen_host_dn_intra\0")
                .expect("the intra kernel's body");
            f(
                (p as *const eidola_engine_cuda::qwen::abi::DnChunkParams).cast(),
                gx,
                gy,
            );
        }
        Ok(())
    }
}

/// The compiled-bodies device, or `None` without a compiler.
pub fn compiled() -> Option<HostDevice> {
    library().map(host_device)
}

/// Raises the soft memory-lock limit to the hard one, for the reference executor's
/// locked host tier, and serializes the tests of one binary that build one.
pub fn memlock_guard() -> std::sync::MutexGuard<'static, ()> {
    use rustix::process::{Resource, Rlimit, getrlimit, setrlimit};
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let l = getrlimit(Resource::Memlock);
    if l.current.is_some() && l.current != l.maximum {
        let _ = setrlimit(
            Resource::Memlock,
            Rlimit {
                current: l.maximum,
                maximum: l.maximum,
            },
        );
    }
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A device allocation holding `data`, kept alive with it.
pub struct Buf {
    pub addr: u64,
    pub bytes: usize,
    _keep: Box<dyn Any>,
}

pub fn buf<D: QwenDevice, T: Copy>(dev: &D, data: &[T]) -> Buf {
    let bytes = std::mem::size_of_val(data).max(8);
    let (addr, keep) = dev.alloc(bytes).expect("alloc");
    // SAFETY: a fresh allocation of `bytes`.
    unsafe {
        let src =
            std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data));
        dev.upload(addr, src).expect("upload");
    }
    Buf {
        addr,
        bytes,
        _keep: keep,
    }
}

pub fn zeros<D: QwenDevice>(dev: &D, bytes: usize) -> Buf {
    let (addr, keep) = dev.alloc(bytes.max(8)).expect("alloc");
    Buf {
        addr,
        bytes,
        _keep: keep,
    }
}

pub fn read<D: QwenDevice, T: Copy + Default>(dev: &D, b: &Buf, n: usize) -> Vec<T> {
    let mut out = vec![T::default(); n];
    // SAFETY: `n` elements of `b`, which the caller sized.
    unsafe {
        let dst = std::slice::from_raw_parts_mut(
            out.as_mut_ptr().cast::<u8>(),
            n * std::mem::size_of::<T>(),
        );
        dev.download(b.addr, dst).expect("download");
    }
    out
}

/// A seeded LCG, drawing from the state's high bits.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform in [-1, 1).
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    pub fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 33) % n
    }

    /// A BF16 value of magnitude up to `scale`.
    pub fn bf16(&mut self, scale: f32) -> f32 {
        eidola_engine_cuda::bf16::to_f32(eidola_engine_cuda::bf16::from_f32(self.f32() * scale))
    }
}

pub fn bits(x: &[f32]) -> Vec<u16> {
    x.iter()
        .map(|&v| eidola_engine_cuda::bf16::from_f32(v))
        .collect()
}

pub fn widen(x: &[u16]) -> Vec<f32> {
    x.iter()
        .map(|&b| eidola_engine_cuda::bf16::to_f32(b))
        .collect()
}

/// The kernel directory for device tests, or `None` after saying loudly why a device test
/// does not run here.
pub fn device_kernels(test: &str) -> Option<String> {
    if !eidola_engine_cuda::Gpu::available() {
        eprintln!("SKIPPED {test}: no CUDA device on this machine (a GPU session runs it)");
        return None;
    }
    match std::env::var("EIDOLA_ENGINE_KERNELS_DIR") {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("SKIPPED {test}: EIDOLA_ENGINE_KERNELS_DIR is not set");
            None
        }
    }
}

/// Writes a measurement under `EIDOLA_QWEN_SUMMARY_DIR` (when set) as `<name>.txt`.
pub fn record(name: &str, text: &str) {
    if let Ok(dir) = std::env::var("EIDOLA_QWEN_SUMMARY_DIR") {
        let path = PathBuf::from(dir).join(format!("{name}.txt"));
        std::fs::write(&path, text).expect("write the measurement");
        eprintln!("recorded {}", path.display());
    }
}

thread_local! {
    static PATH: std::cell::Cell<(CudaGraphs, DecodePath)> =
        const { std::cell::Cell::new((CudaGraphs::Off, DecodePath::Eager)) };
    static RUNG_STEPS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static FIXTURE_256: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static BLOCK_SIZE: std::cell::Cell<u32> = const { std::cell::Cell::new(4) };
}

/// Runs `f` with every executor the harnesses build on this thread at KV blocks of `bs`
/// positions (4 otherwise).
pub fn with_block_size<R>(bs: u32, f: impl FnOnce() -> R) -> R {
    let old = BLOCK_SIZE.with(|c| c.replace(bs));
    let r = f();
    BLOCK_SIZE.with(|c| c.set(old));
    r
}

/// The block size executors built now take.
pub fn block_size_now() -> u32 {
    BLOCK_SIZE.with(std::cell::Cell::get)
}

/// Runs `f` with every harness on this thread over the fixture with 256-wide attention
/// heads (`qwen_tiny256`).
pub fn with_fixture_256<R>(f: impl FnOnce() -> R) -> R {
    let old = FIXTURE_256.with(|c| c.replace(true));
    let r = f();
    FIXTURE_256.with(|c| c.set(old));
    r
}

/// Whether harnesses built now use `qwen_tiny256`.
pub fn fixture_256() -> bool {
    FIXTURE_256.with(std::cell::Cell::get)
}

/// Runs `f` with every executor the harnesses build on this thread capturing graphs as
/// `graphs` says and running decode steps that fit a rung by `path`.
pub fn with_path<R>(graphs: CudaGraphs, path: DecodePath, f: impl FnOnce() -> R) -> R {
    let old = PATH.with(|p| p.replace((graphs, path)));
    let r = f();
    PATH.with(|p| p.set(old));
    r
}

/// The graphs setting executors built now take.
pub fn graphs_now() -> CudaGraphs {
    PATH.with(|p| p.get().0)
}

/// Sets the current decode path on a freshly built executor.
pub fn apply_path<D: QwenDevice>(ex: &mut QwenCudaExecutor<D>) {
    let path = PATH.with(|p| p.get().1);
    ex.set_decode_path(path).expect("the decode path");
}

/// Counts steps a rung ran (directly or replayed), on this thread.
pub fn note_rung_steps(n: u64) {
    RUNG_STEPS.with(|c| c.set(c.get() + n));
}

/// Steps a rung ran on this thread so far.
pub fn rung_steps() -> u64 {
    RUNG_STEPS.with(std::cell::Cell::get)
}
