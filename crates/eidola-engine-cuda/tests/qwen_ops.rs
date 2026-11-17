//! The Qwen3.8 kernels on a GPU: every family against the model crate bit for bit on the
//! device's own image and on the `sm_100f` family image (the exact profile), then the fast
//! profile's linears against the exact ones (deviation recorded; a row's bits alone and in
//! every batch asserted equal). Skipped, saying why, without a device or
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
    let fast = CudaDevice::open(0, &dir, QwenProfile::Fast).expect("the device");
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
