//! The DeepSeek-V4.1 executor with the backbone's routed experts staged (`deepseek::staged`,
//! dev-only): the shared scenarios (`deepseek_common`) with each half's expert cache smaller
//! than its layers' experts, every step held equal to the CPU reference executor's
//! exactly as the unstaged executor is (tokens, logits bit for bit, pool bytes, the
//! crossing, draft logits). The fixture has 8 experts a layer, 2 per token and 4 layers
//! a half: a cache of 2 holds one row's experts and no more, so nearly every round
//! evicts; a cache of 5 runs most steps in several rounds.
//!
//! On the host (`HostDevice`, and the device kernels' bodies compiled for the host) the
//! tests run anywhere. `the_tiny_model_staged_on_a_device_*` run the same on a GPU, both
//! halves on device 0, and are skipped (after saying why) without a device or
//! `EIDOLA_ENGINE_KERNELS_DIR`:
//!
//!     EIDOLA_ENGINE_KERNELS_DIR=<kernel build output> EIDOLA_DS_SUMMARY_DIR=<dir> \
//!         cargo test --release -p eidola-engine-cuda --features dev-staged-experts \
//!         --test deepseek_staged -- --nocapture --test-threads 1

mod deepseek_common;

use std::path::PathBuf;

use deepseek_common::*;
use eidola_engine_cuda::deepseek::half::{CudaDevice, Device};
use eidola_engine_cuda::deepseek::host::HostDevice;
use eidola_engine_cuda::deepseek::staged::HostMode;
use eidola_engine_cuda::{Gpu, KernelDir};

fn host() -> Box<dyn Device> {
    Box::new(HostDevice)
}

/// Runs `scenario` over the fixture (or its wide configuration) with caches of
/// `capacity` experts per half read `mode`.
fn staged(
    device: &dyn Fn() -> Box<dyn Device>,
    wide: bool,
    width: u32,
    mode: HostMode,
    capacity: [u32; 2],
    scenario: impl FnOnce(&mut Run<'_>),
) -> Vec<Outcome> {
    STAGED.set(Some((mode, capacity)));
    let model = model(wide);
    let mut run = Run {
        model: &model,
        device,
        outcomes: Vec::new(),
        width,
    };
    scenario(&mut run);
    STAGED.set(None);
    assert!(run.outcomes.iter().all(|o| o.compared.blocks > 0));
    run.outcomes
}

const READ: HostMode = HostMode::Read { threads: 3 };

#[test]
fn a_cache_of_one_rows_experts() {
    staged(&host, false, 0, READ, [2, 2], |r| {
        r.single_requests();
        r.chunked_prefill(&[1, 7, 48, 97, 1000]);
        r.batch_invariance();
    });
}

#[test]
fn a_cache_of_some_experts_from_the_mapped_checkpoint() {
    staged(&host, false, 0, HostMode::Mapped, [5, 3], |r| {
        r.single_requests();
        r.one_triangle();
        r.hits();
        r.preemption();
        r.randomized(0..2);
    });
}

#[test]
fn a_cache_of_every_expert() {
    staged(&host, false, 0, READ, [32, 32], |r| r.single_requests());
}

#[test]
fn staged_drafting() {
    staged(&host, false, 4, READ, [3, 2], |r| {
        r.greedy_speculation();
        r.sampled_speculation();
    });
}

#[test]
fn staged_drafting_past_one_attention_block() {
    staged(&host, true, 5, READ, [4, 4], |r| {
        drafting_scenarios(r, true)
    });
}

#[test]
fn staged_memory() {
    staged(&host, false, 4, READ, [2, 5], |r| {
        memory_scenarios(r, false)
    });
}

/// The device kernels' bodies compiled for the host behind the wrappers' checks and
/// grids: the placement and the products over `capacity` groups. Skipped (after saying
/// why) without a C++ compiler.
#[test]
fn the_compiled_kernel_bodies_run_staged() {
    let Some(lib) = compiled::library() else {
        return;
    };
    let device =
        move || -> Box<dyn Device> { Box::new(compiled::CompiledDevice::new(lib.clone())) };
    staged(&device, false, 0, READ, [2, 3], |r| all_scenarios(r, false));
    staged(&device, true, 5, READ, [4, 2], |r| {
        drafting_scenarios(r, true)
    });
}

/// Runs the staged scenarios on device 0 and writes their summary to `file`.
fn on_device(file: &str, drafting: bool, capacity: [u32; 2]) {
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
        let slots = if wide { 5 } else { 4 };
        let width = if drafting { slots } else { 0 };
        outcomes.extend(staged(&device, wide, width, READ, capacity, |r| {
            if drafting {
                drafting_scenarios(r, wide);
                memory_scenarios(r, wide);
            } else {
                all_scenarios(r, wide);
            }
        }));
    }
    let path = std::env::var_os("EIDOLA_DS_SUMMARY_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_TARGET_TMPDIR")))
        .join(file);
    let json = serde_json::to_string_pretty(&summary(&info.name, &outcomes)).unwrap();
    std::fs::write(&path, json).expect("write the summary");
    eprintln!("summary: {}", path.display());
}

#[test]
fn the_tiny_model_staged_on_a_device_equals_the_cpu_reference_executor() {
    on_device("ds_staged.json", false, [2, 3]);
}

#[test]
fn the_tiny_model_staged_on_a_device_drafts_and_keeps_its_memory() {
    on_device("ds_staged_drafting.json", true, [3, 2]);
}
