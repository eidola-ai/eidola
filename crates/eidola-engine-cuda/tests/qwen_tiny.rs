//! The Qwen3.8 CUDA executor in its exact profile on a GPU, held to the CPU reference
//! executor step for step on the committed fixture through the real engine: the same
//! scenarios as `tests/qwen_host.rs` (`qwen_common::scenarios`). Skipped, saying why,
//! without a device or `EIDOLA_ENGINE_KERNELS_DIR`.

mod qwen_common;

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::QwenProfile;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use qwen_common::scenarios;

fn device(dir: &str) -> CudaDevice {
    CudaDevice::open(0, &KernelDir::new(dir), QwenProfile::Exact).expect("the device")
}

#[test]
fn qwen_tiny_scenarios_on_the_device() {
    let Some(dir) = qwen_common::device_kernels("qwen_tiny_scenarios_on_the_device") else {
        return;
    };
    let make = || device(&dir);
    scenarios::one_request(make());
    eprintln!("ok      one_request");
    scenarios::chunkings(&make);
    eprintln!("ok      chunkings");
    scenarios::hits_and_pressure(&make);
    eprintln!("ok      hits_and_pressure");
    scenarios::alone_and_crowded(&make);
    eprintln!("ok      alone_and_crowded");
}
