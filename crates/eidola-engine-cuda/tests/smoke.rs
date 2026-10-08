//! Device smoke tests. They need a CC 10.x GPU and a kernel build output
//! (`EIDOLA_ENGINE_KERNELS_DIR`); without either they print why and pass.

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::ops::RmsNorm;
use eidola_engine_cuda::{Gpu, ImageArch, ImageSource, KernelModule, bf16};
use eidola_engine_kernels::Manifest;

fn setup() -> Option<(Gpu, KernelDir)> {
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
    Some((gpu, KernelDir::new(dir)))
}

/// Every entry of every kernel resolves (mangled, internal-linkage template
/// instances included) and its launch contract reads back, from the exact
/// cubin, the family cubin and the fatbin. The contracts also pin the
/// device-side requirements `check_device` holds: the largest launch's shared
/// memory is exactly `REQUIRED_SMEM_PER_BLOCK`, and only DeepGEMM's
/// instances (the MoE path) launch clusters.
#[test]
fn every_entry_and_contract_resolves() {
    use eidola_engine_cuda::support::REQUIRED_SMEM_PER_BLOCK;
    let Some((gpu, dir)) = setup() else { return };
    let exact = gpu.image_arch().unwrap();
    for source in [
        ImageSource::Cubin(exact),
        ImageSource::Cubin(ImageArch::Sm100f),
        ImageSource::Fatbin(exact),
    ] {
        let mut largest = (0, String::new());
        for fatbin in &Manifest::embedded().fatbins {
            let module = KernelModule::load_from(&gpu, &dir, &fatbin.name, source).unwrap();
            for entry in &module.cubin().entries {
                let kernel = module.kernel(&entry.symbol).unwrap();
                let meta = kernel.meta();
                assert!(meta.block.iter().all(|&b| b > 0));
                let smem = meta.dynamic_smem_bytes + kernel.static_smem_bytes().unwrap();
                if smem > largest.0 {
                    largest = (smem, entry.symbol.clone());
                }
                if meta.cluster != [1, 1, 1] {
                    assert_eq!(fatbin.name, "deepgemm_fp8_fp4_grouped", "{}", entry.symbol);
                }
            }
        }
        assert_eq!(
            largest.0, REQUIRED_SMEM_PER_BLOCK,
            "{source:?}: {}",
            largest.1
        );
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
        let norm = RmsNorm::from_module(&module).unwrap();
        let mut dout = stream.alloc_zeros::<u16>(rows * hidden).unwrap();
        norm.launch(&gpu, &mut dout, &dx, &dw, hidden as u32, 1e-6)
            .unwrap();
        check_rmsnorm(&stream.clone_dtoh(&dout).unwrap(), &x, &w, rows, hidden);
    }
}

/// A resolved kernel keeps its image loaded: one resolved from a module
/// that is then dropped (here, a temporary) still reads its attributes and
/// launches, correctly, after other images have loaded in its place.
#[test]
fn kernels_outlive_their_module_handle() {
    let Some((gpu, dir)) = setup() else { return };
    let (rows, hidden) = (3usize, 4096usize);
    let x: Vec<u16> = (0..rows * hidden)
        .map(|i| bf16::from_f32((i % 7) as f32 - 3.0))
        .collect();
    let w: Vec<u16> = (0..hidden)
        .map(|i| bf16::from_f32(0.5 + (i % 3) as f32))
        .collect();
    let stream = gpu.stream();
    let (dx, dw) = (
        stream.clone_htod(&x).unwrap(),
        stream.clone_htod(&w).unwrap(),
    );
    let norm = RmsNorm::from_module(&KernelModule::load(&gpu, &dir, "rmsnorm").unwrap()).unwrap();
    let others: Vec<_> = ["engine_ops", "sampling", "rmsnorm"]
        .into_iter()
        .map(|name| KernelModule::load(&gpu, &dir, name).unwrap())
        .collect();
    drop(others);
    norm.kernel().static_smem_bytes().unwrap();
    let mut dout = stream.alloc_zeros::<u16>(rows * hidden).unwrap();
    norm.launch(&gpu, &mut dout, &dx, &dw, hidden as u32, 1e-6)
        .unwrap();
    check_rmsnorm(&stream.clone_dtoh(&dout).unwrap(), &x, &w, rows, hidden);
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

/// Kernels loaded on one thread launch from another: every launch path binds
/// the module's context (current contexts are per thread), including the
/// ones that make other driver calls first (descriptor encoding, memsets).
#[test]
fn launches_from_another_thread() {
    use eidola_engine::executor::Maintenance;
    use eidola_engine_cuda::gemm::{Gemm, GemmArgs, GemmKind};
    use eidola_engine_cuda::kv::{GroupGeometry, KvStore};
    use eidola_engine_cuda::launch::dptr;

    let Some((gpu, dir)) = setup() else { return };
    let module = KernelModule::load(&gpu, &dir, "rmsnorm").unwrap();
    let norm = RmsNorm::from_module(&module).unwrap();
    let (name, _) = GemmKind::Bf16.kernel();
    let gemm_module = KernelModule::load(&gpu, &dir, name).unwrap();
    let gemm = Gemm::from_module(GemmKind::Bf16, &gemm_module).unwrap();
    let (rows, hidden) = (4usize, 256usize);
    let x: Vec<u16> = (0..rows * hidden)
        .map(|i| bf16::from_f32((i % 7) as f32 - 3.0))
        .collect();
    let w = vec![bf16::from_f32(1.0); hidden * hidden];
    let stream = gpu.stream();
    let (dx, dw) = (
        stream.clone_htod(&x).unwrap(),
        stream.clone_htod(&w).unwrap(),
    );
    let dout = stream.alloc_zeros::<u16>(rows * hidden).unwrap();
    let dgemm = stream.alloc_zeros::<f32>(rows * hidden).unwrap();
    let mut kv = KvStore::new(
        &gpu,
        vec![GroupGeometry {
            num_layers: 1,
            num_kv_heads: 1,
            head_dim_qk: 192,
            head_dim_v: 128,
            block_size: 16,
            num_blocks: 2,
        }],
        1,
        1,
        0,
    )
    .unwrap();
    let ptrs = [dptr(&dout, stream), dptr(&dx, stream), dptr(&dw, stream)];
    let gemm_args = GemmArgs {
        m: rows as u32,
        n: hidden as u32,
        k: hidden as u32,
        a: dptr(&dx, stream),
        b: dptr(&dw, stream),
        d: dptr(&dgemm, stream),
        sfa: 0,
        sfb: 0,
        alpha: 1.0,
    };
    gpu.synchronize().unwrap();
    std::thread::scope(|s| {
        s.spawn(|| {
            // Nothing on this thread has made a context current; each call
            // below is the first driver call of its kind here.
            let (mut o, mut i, mut wt, mut h, mut eps) =
                (ptrs[0], ptrs[1], ptrs[2], hidden as u32, 1e-6f32);
            let mut args = [
                &mut o as *mut _ as *mut std::ffi::c_void,
                &mut i as *mut _ as *mut std::ffi::c_void,
                &mut wt as *mut _ as *mut std::ffi::c_void,
                &mut h as *mut _ as *mut std::ffi::c_void,
                &mut eps as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                norm.kernel()
                    .launch(gpu.stream(), [rows as u32, 1, 1], &mut args)
            }
            .unwrap();
            unsafe { gemm.launch(&gpu, &gemm_args) }.unwrap();
            kv.apply(&gpu, &[Maintenance::Zero { group: 0, block: 1 }], &[])
                .unwrap();
        });
    });
    gpu.synchronize().unwrap();
    assert!(
        stream.clone_dtoh(&dout).unwrap().iter().any(|&v| v != 0),
        "the norm ran"
    );
    assert!(
        stream.clone_dtoh(&dgemm).unwrap().iter().any(|&v| v != 0.0),
        "the GEMM ran"
    );
}

/// Only the compiled-in manifest approves bytes: a cubin altered by one byte
/// is refused before it reaches the driver.
#[test]
fn altered_images_are_refused() {
    let Some((gpu, _)) = setup() else { return };
    let src = std::path::PathBuf::from(std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR").unwrap());
    let arch = gpu.image_arch().unwrap();
    let cubin = Manifest::embedded()
        .cubin("rmsnorm", arch.as_str())
        .unwrap();
    let tmp = std::env::temp_dir().join(format!("eidola-altered-{}", std::process::id()));
    let dst = tmp.join(&cubin.file);
    std::fs::create_dir_all(dst.parent().unwrap()).unwrap();
    let mut bytes = std::fs::read(src.join(&cubin.file)).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&dst, &bytes).unwrap();
    let e = KernelModule::load(&gpu, &KernelDir::new(&tmp), "rmsnorm")
        .err()
        .expect("refused");
    std::fs::remove_dir_all(&tmp).unwrap();
    assert!(e.to_string().contains("sha256"), "{e}");
}
