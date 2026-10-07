//! GPU test setup: a device, a kernel build output, and the images to test.
#![allow(dead_code)]

pub mod qref;

use eidola_engine_cuda::{Gpu, ImageArch, ImageSource, KernelModule};
use eidola_engine_kernels::{ArtifactDir, Manifest};

pub struct Setup {
    pub gpu: Gpu,
    pub dir: ArtifactDir<'static>,
    /// Every image a correctness test runs against: the device's exact cubin
    /// and the family cubin (what a CC 10.0 part other than the B200 would
    /// run, and the only way this device exercises a second binary).
    pub archs: Vec<ImageArch>,
}

impl Setup {
    pub fn module(&self, name: &str, arch: ImageArch) -> KernelModule {
        KernelModule::load_from(&self.gpu, &self.dir, name, ImageSource::Cubin(arch))
            .unwrap_or_else(|e| panic!("{name} {arch:?}: {e}"))
    }
}

/// `None` (after saying why) without a device or without
/// `EIDOLA_ENGINE_KERNELS_DIR`.
pub fn setup() -> Option<Setup> {
    let Some(dir) = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR") else {
        eprintln!("skipping: EIDOLA_ENGINE_KERNELS_DIR is not set");
        return None;
    };
    if !Gpu::available() {
        eprintln!("skipping: no CUDA device");
        return None;
    }
    let gpu = Gpu::open(0).expect("open device 0");
    let exact = gpu.image_arch()?;
    let mut archs = vec![exact];
    if exact != ImageArch::Sm100f {
        archs.push(ImageArch::Sm100f);
    }
    Some(Setup {
        gpu,
        dir: ArtifactDir::new(dir, Manifest::embedded()),
        archs,
    })
}

/// A deterministic generator for test data.
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
        self.next_u64() % n
    }
}
