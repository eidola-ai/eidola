//! What the Qwen3.8 tests share: the kernels' bodies compiled for the host
//! (`oracle/qwen_launch_host.cc` over `eidola-engine-kernels`' `csrc/qwen_ops.cuh`) as a
//! [`HostDevice`], buffers on any [`QwenDevice`], a seeded generator, and the loud skip of
//! a device test.
#![allow(dead_code)]

pub mod cases;
pub mod invariance;
pub mod pair;
pub mod scenarios;

use std::any::Any;
use std::ffi::c_void;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use eidola_engine_cuda::qwen::QwenDevice;
use eidola_engine_cuda::qwen::host::HostDevice;

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
    HostDevice::new(Box::new(move |family, params, gx, gy| {
        let name = format!("qwen_host_{family}\0");
        // SAFETY: every `qwen_host_*` symbol has this signature, and takes the family's
        // parameter struct, which the device built for it.
        unsafe {
            let f: libloading::Symbol<HostFn> =
                lib.get(name.as_bytes()).expect("a host body per family");
            f(params, gx, gy);
        }
    }))
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
