//! The DeepSeek-V4.1 executor on a GPU: the committed tiny fixture and its wide-window
//! configuration through the real engine, both logical devices on device 0 (or the
//! encoder's on one GPU and the decoder's on another: `EIDOLA_DS_TINY_DEVICES=0,1`), every step
//! held equal to the CPU reference executor's (`deepseek_common`): tokens, logits bit for
//! bit, pool bytes block for block, the crossing, and with drafts on every draft slot's
//! logits and the drafted tokens. Skipped (after saying why) without a device or
//! without `EIDOLA_ENGINE_KERNELS_DIR`.
//!
//!     EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> EIDOLA_DS_SUMMARY_DIR=<dir> \
//!         cargo test --release -p eidola-engine-cuda --test deepseek_tiny -- --nocapture
//!
//! Three tests, each printing a line per scenario and writing a JSON summary (what was
//! compared, and each step's time part by part) into `EIDOLA_DS_SUMMARY_DIR`, or the
//! test's temporary directory: the plain scenarios (`ds_tiny.json`, or the file
//! `EIDOLA_DS_TINY_SUMMARY` names), the scenarios with every draft slot drafted
//! (`ds_tiny_drafting.json`), and the memory scenarios with a page-locked host tier
//! (`ds_tiny_memory.json`). A failure is a panic naming the first difference; no summary
//! is written then, nor when the test is skipped.

mod deepseek_common;

use std::path::PathBuf;

use deepseek_common::*;
use eidola_engine_cuda::deepseek::half::{CudaDevice, Device};
use eidola_engine_cuda::{Gpu, KernelDir};

/// The encoder's and the decoder's GPU: `EIDOLA_DS_TINY_DEVICES` (`e,d`), else both 0.
fn devices() -> [usize; 2] {
    match std::env::var("EIDOLA_DS_TINY_DEVICES") {
        Ok(v) if !v.is_empty() => {
            let d: Vec<usize> = v
                .split(',')
                .map(|d| {
                    d.trim()
                        .parse()
                        .expect("EIDOLA_DS_TINY_DEVICES: two ordinals")
                })
                .collect();
            assert_eq!(d.len(), 2, "EIDOLA_DS_TINY_DEVICES: two ordinals");
            [d[0], d[1]]
        }
        _ => [0, 0],
    }
}

/// Runs `scenarios` over the fixture and its wide-window configuration on the devices
/// of [`devices`] and writes their summary to `file`.
fn on_device(file: &str, drafting: bool, scenarios: impl Fn(&mut Run<'_>, bool)) {
    let Some(dir) = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR") else {
        eprintln!("skipping: EIDOLA_ENGINE_KERNELS_DIR is not set");
        return;
    };
    if !Gpu::available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let ordinals = devices();
    let info = Gpu::open(ordinals[0])
        .expect("open the encoder's device")
        .info()
        .clone();
    eprintln!(
        "{} (driver {}), halves on {ordinals:?}",
        info.name, info.driver_version
    );
    // Every executor opens its encoder's half first, then its decoder's.
    let opened = std::cell::Cell::new(0usize);
    let device = || -> Box<dyn Device> {
        let n = opened.get();
        opened.set(n + 1);
        Box::new(
            CudaDevice::open(ordinals[n % 2], &KernelDir::new(dir.clone())).expect("open a half"),
        )
    };
    let mut outcomes = Vec::new();
    for wide in [false, true] {
        let model = model(wide);
        let mut run = Run {
            model: &model,
            device: &device,
            outcomes: Vec::new(),
            width: if drafting { model.slots } else { 0 },
        };
        scenarios(&mut run, wide);
        outcomes.extend(run.outcomes);
    }
    let path = std::env::var_os("EIDOLA_DS_SUMMARY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")))
        .join(file);
    let path = match (file, std::env::var_os("EIDOLA_DS_TINY_SUMMARY")) {
        ("ds_tiny.json", Some(named)) => PathBuf::from(named),
        _ => path,
    };
    let json = serde_json::to_string_pretty(&summary(&info.name, &outcomes)).unwrap();
    std::fs::write(&path, json).expect("write the summary");
    eprintln!("summary: {}", path.display());
}

#[test]
fn the_tiny_model_equals_the_cpu_reference_executor() {
    on_device("ds_tiny.json", false, all_scenarios);
}

/// Every draft slot drafted: all widths and acceptance patterns, seeded sampling, and
/// the plain scenarios again with drafts on.
#[test]
fn the_tiny_model_drafts_as_the_cpu_reference_executor() {
    on_device("ds_tiny_drafting.json", true, drafting_scenarios);
}

/// Idle encryption, resealing, the host tier's round trip and randomized traffic over
/// a host tier, with drafts on; the host tier is page-locked host memory.
#[test]
fn the_tiny_model_keeps_its_memory_as_the_cpu_reference_executor() {
    on_device("ds_tiny_memory.json", true, memory_scenarios);
}
