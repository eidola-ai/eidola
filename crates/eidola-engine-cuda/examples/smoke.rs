//! Device smoke check: report what the driver says about the device, load
//! every manifest image this device can run (and show which it cannot),
//! resolve every entry and launch-contract record, and run RMSNorm against
//! the f32 reference.
//!
//! ```sh
//! cargo run --release -p eidola-engine-cuda --example smoke -- <kernel build output>
//! ```

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::{Gpu, ImageArch, ImageSource, KernelModule, bf16, ops};
use eidola_engine_kernels::Manifest;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: smoke <kernel build output dir>");
    let manifest = Manifest::embedded();
    let artifacts = KernelDir::new(&dir);
    artifacts
        .verify_all()
        .expect("every image matches the manifest");
    println!("artifacts: every image in {dir} matches the compiled-in manifest");

    let gpu = Gpu::open(0).expect("open device 0");
    let info = gpu.info();
    println!("device: {info:#?}");
    println!(
        "driver API {}.{}; exact-match image {:?}",
        info.driver_version / 1000,
        info.driver_version % 1000 / 10,
        gpu.image_arch()
    );

    let names: Vec<&str> = manifest.fatbins.iter().map(|f| f.name.as_str()).collect();
    let sources = [
        ImageSource::Cubin(ImageArch::Sm103a),
        ImageSource::Cubin(ImageArch::Sm100f),
        ImageSource::Cubin(ImageArch::Sm100a),
        ImageSource::Fatbin(gpu.image_arch().unwrap_or(ImageArch::Sm100f)),
    ];
    for source in sources {
        for name in &names {
            match KernelModule::load_from(&gpu, &artifacts, name, source) {
                Ok(module) => {
                    for entry in &module.cubin().entries {
                        match module.kernel(&entry.symbol) {
                            Ok(k) => {
                                println!("  {source:?} {name}: {} -> {:?}", entry.meta, k.meta())
                            }
                            Err(e) => println!("  {source:?} {name}: {} FAILED {e}", entry.meta),
                        }
                    }
                }
                Err(e) => println!("  {source:?} {name}: load FAILED {e}"),
            }
        }
    }

    for source in [
        ImageSource::Cubin(ImageArch::Sm103a),
        ImageSource::Cubin(ImageArch::Sm100f),
        ImageSource::Fatbin(ImageArch::Sm103a),
    ] {
        let Ok(module) = KernelModule::load_from(&gpu, &artifacts, "rmsnorm", source) else {
            println!("rmsnorm {source:?}: not loadable on this device");
            continue;
        };
        let kernel = module.kernel("eidola_rmsnorm_bf16").expect("entry");
        for (rows, hidden) in [(1u32, 4096u32), (37, 4096), (5, 1000), (3, 256)] {
            let (max_ulps, mismatches) = rmsnorm_vs_reference(&gpu, &kernel, rows, hidden);
            println!(
                "rmsnorm {source:?} rows {rows} hidden {hidden}: {mismatches} of {} outputs differ from the f32 reference rounded to bf16, max {max_ulps} bf16 ulp",
                rows * hidden
            );
        }
    }
}

fn rmsnorm_vs_reference(
    gpu: &Gpu,
    kernel: &eidola_engine_cuda::Kernel,
    rows: u32,
    hidden: u32,
) -> (u32, usize) {
    let n = (rows * hidden) as usize;
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ n as u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    };
    let x: Vec<u16> = (0..n).map(|_| bf16::from_f32(next())).collect();
    let w: Vec<u16> = (0..hidden).map(|_| bf16::from_f32(next())).collect();
    let stream = gpu.stream();
    let dx = stream.clone_htod(&x).unwrap();
    let dw = stream.clone_htod(&w).unwrap();
    let mut dout = stream.alloc_zeros::<u16>(n).unwrap();
    ops::rmsnorm_bf16(gpu, kernel, &mut dout, &dx, &dw, hidden, 1e-6).unwrap();
    let out = stream.clone_dtoh(&dout).unwrap();

    let wf: Vec<f32> = w.iter().map(|&b| bf16::to_f32(b)).collect();
    let mut max_ulps = 0u32;
    let mut mismatches = 0;
    let mut reference = vec![0f32; hidden as usize];
    for r in 0..rows as usize {
        let row: Vec<f32> = x[r * hidden as usize..(r + 1) * hidden as usize]
            .iter()
            .map(|&b| bf16::to_f32(b))
            .collect();
        eidola_engine_model::tensor::rms_norm(&row, &wf, 1e-6, &mut reference);
        for (i, &want) in reference.iter().enumerate() {
            let got = out[r * hidden as usize + i];
            let want = bf16::from_f32(want);
            if got != want {
                mismatches += 1;
                max_ulps = max_ulps.max((got as i32 - want as i32).unsigned_abs());
            }
        }
    }
    (max_ulps, mismatches)
}
