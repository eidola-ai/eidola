//! The Qwen3.8 kernels on a GPU: every family against the model crate bit for bit on the
//! device's own image and on the `sm_100f` family image (the exact profile), the
//! quantiser's kernel against its reference form compiled for the host (codes and scales
//! bit for bit, edge blocks included), then the fast profile's FP8 linears and BF16 GEMM
//! (the DeltaNet's stacked `a` and `b`) against the exact ones (deviation recorded, and
//! bounded for the BF16 GEMM; a row's bits alone and in every batch asserted equal). Skipped, saying why, without a device or
//! `EIDOLA_ENGINE_KERNELS_DIR`.

mod qwen_common;

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::QwenProfile;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use qwen_common::cases;

#[test]
fn qwen_kernels_on_the_device() {
    let Some(dir) = qwen_common::device_kernels("qwen_kernels_on_the_device") else {
        return;
    };
    let dir = KernelDir::new(dir);
    let exact = CudaDevice::open(0, &dir, QwenProfile::Exact).expect("the device");
    let mut failed = Vec::new();
    for (name, outcome) in cases::all(&exact) {
        match outcome {
            Ok(()) => eprintln!("ok      {name}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    match qwen_common::library() {
        Some(lib) => {
            let reference = qwen_common::quant_thread_reference(lib);
            match cases::quant_forms(&exact, &reference, 12) {
                Ok(()) => eprintln!("ok      quant forms"),
                Err(e) => {
                    eprintln!("FAILED  quant forms: {e}");
                    failed.push("quant forms");
                }
            }
        }
        None => {
            eprintln!("FAILED  quant forms: no C++ compiler for the reference form");
            failed.push("quant forms");
        }
    }
    let fast = CudaDevice::open(0, &dir, QwenProfile::Fast).expect("the device");
    match cases::fast_bf16_gemm(&exact, &fast, 13) {
        Ok(report) => {
            eprintln!("ok      fast_bf16_gemm: {report}");
            qwen_common::record("qwen_ops_fast_bf16_gemm", &report);
        }
        Err(e) => {
            eprintln!("FAILED  fast_bf16_gemm: {e}");
            failed.push("fast_bf16_gemm");
        }
    }
    match cases::fast_linears(&exact, &fast, 11) {
        Ok(report) => {
            eprintln!("ok      fast_linears: {report}");
            qwen_common::record("qwen_ops_fast_linears", &report);
        }
        Err(e) => {
            eprintln!("FAILED  fast_linears: {e}");
            failed.push("fast_linears");
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
