//! The DeepSeek-V4.1 executor on a GPU: the committed tiny fixture and its wide-window
//! configuration through the real engine, both logical devices on device 0, every step
//! held equal to the CPU reference executor's (`ds_common`): tokens, logits bit for
//! bit, pool bytes block for block, the crossing. Skipped (after saying why) without a
//! device or without `EIDOLA_ENGINE_KERNELS_DIR`.
//!
//!     EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> \
//!     EIDOLA_DS_TINY_SUMMARY=ds_tiny.json \
//!         cargo test --release -p eidola-engine-cuda --test ds_tiny -- --nocapture
//!
//! prints a line per scenario and writes the JSON summary (what was compared, and each
//! step's time part by part) to `EIDOLA_DS_TINY_SUMMARY`, or to `ds_tiny.json` in the
//! test's temporary directory. A failure is a panic naming the first difference; no
//! summary is written then, nor when the test is skipped.

mod ds_common;

use std::path::PathBuf;

use ds_common::*;
use eidola_engine_cuda::ds::half::{CudaDevice, Device};
use eidola_engine_cuda::{Gpu, KernelDir};

#[test]
fn the_tiny_model_equals_the_cpu_reference_executor() {
    let Some(dir) = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR") else {
        eprintln!("skipping: EIDOLA_ENGINE_KERNELS_DIR is not set");
        return;
    };
    if !Gpu::available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let info = Gpu::open(0).expect("open device 0").info().clone();
    eprintln!("{} (driver {})", info.name, info.driver_version);
    let device = || -> Box<dyn Device> {
        Box::new(CudaDevice::open(0, &KernelDir::new(dir.clone())).expect("open a half"))
    };
    let mut outcomes = Vec::new();
    for wide in [false, true] {
        let model = model(wide);
        let mut run = Run {
            model: &model,
            device: &device,
            outcomes: Vec::new(),
        };
        all_scenarios(&mut run, wide);
        outcomes.extend(run.outcomes);
    }
    let path = std::env::var_os("EIDOLA_DS_TINY_SUMMARY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("ds_tiny.json"));
    let json = serde_json::to_string_pretty(&summary(&info.name, &outcomes)).unwrap();
    std::fs::write(&path, json).expect("write the summary");
    eprintln!("summary: {}", path.display());
}
