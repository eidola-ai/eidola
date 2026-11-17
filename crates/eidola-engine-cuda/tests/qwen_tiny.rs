//! The Qwen3.8 CUDA executor in its exact profile on a GPU, held to the CPU reference
//! executor step for step on the committed fixture through the real engine: the same
//! scenarios as `tests/qwen_host.rs` (`qwen_common::scenarios`), eagerly and with decode
//! steps replaying their captured graphs (the graphs keep the exact profile equal to the
//! CPU executor). Skipped, saying why, without a device or `EIDOLA_ENGINE_KERNELS_DIR`.

mod qwen_common;

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath, QwenProfile};
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

#[test]
fn qwen_tiny_scenarios_replaying_graphs() {
    let Some(dir) = qwen_common::device_kernels("qwen_tiny_scenarios_replaying_graphs") else {
        return;
    };
    let make = || device(&dir);
    let before = qwen_common::rung_steps();
    qwen_common::with_path(CudaGraphs::On, DecodePath::Replay, || {
        scenarios::one_request(make());
        scenarios::chunkings(&make);
        eprintln!("ok      chunkings (graphs)");
        scenarios::hits_and_pressure(&make);
        eprintln!("ok      hits_and_pressure (graphs)");
        scenarios::alone_and_crowded(&make);
        eprintln!("ok      alone_and_crowded (graphs)");
    });
    qwen_common::with_fixture_256(|| {
        qwen_common::with_path(CudaGraphs::On, DecodePath::Replay, || {
            scenarios::chunkings(&make);
            eprintln!("ok      chunkings (graphs, 256-wide heads)");
        });
    });
    let n = qwen_common::rung_steps() - before;
    eprintln!("{n} steps replayed a graph");
    assert!(n > 100, "only {n} steps replayed a graph");
}
