//! Device smoke tests. They need a CC 10.x GPU and a kernel build output
//! (`EIDOLA_ENGINE_KERNELS_DIR`); without either they print why and pass.

use eidola_engine_cuda::{Gpu, ImageArch, ImageSource, KernelModule, bf16, ops};
use eidola_engine_kernels::{ArtifactDir, Manifest};

fn setup() -> Option<(Gpu, ArtifactDir<'static>)> {
    let Some(dir) = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR") else {
        eprintln!("skipping: EIDOLA_ENGINE_KERNELS_DIR is not set");
        return None;
    };
    if !Gpu::available() {
        eprintln!("skipping: no CUDA device");
        return None;
    }
    let gpu = Gpu::open(0).expect("open device 0");
    gpu.image_arch()?;
    Some((gpu, ArtifactDir::new(dir, Manifest::embedded())))
}

/// Every entry of every kernel resolves (mangled, internal-linkage template
/// instances included) and its launch contract reads back, from the exact
/// cubin, the family cubin and the fatbin.
#[test]
fn every_entry_and_contract_resolves() {
    let Some((gpu, dir)) = setup() else { return };
    let exact = gpu.image_arch().unwrap();
    for source in [
        ImageSource::Cubin(exact),
        ImageSource::Cubin(ImageArch::Sm100f),
        ImageSource::Fatbin(exact),
    ] {
        for fatbin in &Manifest::embedded().fatbins {
            let module = KernelModule::load_from(&gpu, &dir, &fatbin.name, source).unwrap();
            for entry in &module.cubin().entries {
                let kernel = module.kernel(&entry.symbol).unwrap();
                assert!(kernel.meta().block.iter().all(|&b| b > 0));
                assert!(kernel.meta().dynamic_smem_bytes <= gpu.info().max_smem_per_block_optin);
            }
        }
    }
}

/// An architecture-specific image for another part is refused, not JITted:
/// the images carry no PTX.
#[test]
fn foreign_arch_image_is_refused() {
    let Some((gpu, dir)) = setup() else { return };
    let foreign = match gpu.image_arch().unwrap() {
        ImageArch::Sm103a => ImageArch::Sm100a,
        _ => ImageArch::Sm103a,
    };
    assert!(KernelModule::load_from(&gpu, &dir, "rmsnorm", ImageSource::Cubin(foreign)).is_err());
}

/// RMSNorm matches the f32 reference rounded to BF16 within one BF16 ulp
/// (the kernel uses `rsqrtf` and a different summation order).
#[test]
fn rmsnorm_matches_reference() {
    let Some((gpu, dir)) = setup() else { return };
    let (rows, hidden) = (37usize, 4096usize);
    let mut s = 1u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    };
    let x: Vec<u16> = (0..rows * hidden).map(|_| bf16::from_f32(next())).collect();
    let w: Vec<u16> = (0..hidden).map(|_| bf16::from_f32(next())).collect();
    let stream = gpu.stream();
    let (dx, dw) = (
        stream.clone_htod(&x).unwrap(),
        stream.clone_htod(&w).unwrap(),
    );
    // The device's own cubin and the family cubin.
    let mut archs = vec![gpu.image_arch().unwrap()];
    if archs[0] != ImageArch::Sm100f {
        archs.push(ImageArch::Sm100f);
    }
    for arch in archs {
        let module =
            KernelModule::load_from(&gpu, &dir, "rmsnorm", ImageSource::Cubin(arch)).unwrap();
        let kernel = module.kernel("eidola_rmsnorm_bf16").unwrap();
        let mut dout = stream.alloc_zeros::<u16>(rows * hidden).unwrap();
        ops::rmsnorm_bf16(&gpu, &kernel, &mut dout, &dx, &dw, hidden as u32, 1e-6).unwrap();
        check_rmsnorm(&stream.clone_dtoh(&dout).unwrap(), &x, &w, rows, hidden);
    }
}

fn check_rmsnorm(out: &[u16], x: &[u16], w: &[u16], rows: usize, hidden: usize) {
    let wf: Vec<f32> = w.iter().map(|&b| bf16::to_f32(b)).collect();
    let mut reference = vec![0f32; hidden];
    for r in 0..rows {
        let row: Vec<f32> = x[r * hidden..(r + 1) * hidden]
            .iter()
            .map(|&b| bf16::to_f32(b))
            .collect();
        eidola_engine_model::tensor::rms_norm(&row, &wf, 1e-6, &mut reference);
        for (i, &want) in reference.iter().enumerate() {
            let got = out[r * hidden + i];
            let diff = (got as i32 - bf16::from_f32(want) as i32).abs();
            assert!(diff <= 1, "row {r} col {i}: {got:#06x} vs {want}");
        }
    }
}
