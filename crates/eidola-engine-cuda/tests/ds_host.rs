//! The DeepSeek-V4.1 executor over the host device (`ds::host`): host memory and the
//! reference numerics behind the same launches, memory layouts, planner and boundary
//! the device run uses. Every step is held equal to the CPU reference executor's
//! (`ds_common`): tokens, logits bit for bit, pool bytes block for block, the crossing.
//! Runs anywhere; `ds_tiny.rs` is the same scenarios on a GPU.

mod ds_common;

use ds_common::*;
use eidola_engine_cuda::ds::half::Device;
use eidola_engine_cuda::ds::host::HostDevice;

fn host() -> Box<dyn Device> {
    Box::new(HostDevice)
}

fn run(wide: bool, scenario: impl FnOnce(&mut Run<'_>)) {
    drafting(wide, 0, scenario);
}

fn drafting(wide: bool, width: u32, scenario: impl FnOnce(&mut Run<'_>)) {
    let model = model(wide);
    let mut run = Run {
        model: &model,
        device: &host,
        outcomes: Vec::new(),
        width,
    };
    scenario(&mut run);
    assert!(run.outcomes.iter().all(|o| o.compared.blocks > 0));
}

#[test]
fn single_requests() {
    run(false, |r| r.single_requests());
}

#[test]
fn chunked_prefill_is_identical_to_unchunked() {
    run(false, |r| r.chunked_prefill(&[1, 7, 48, 97, 1000]));
}

#[test]
fn a_long_prompt_costs_one_triangle() {
    run(false, |r| r.one_triangle());
}

#[test]
fn hits_resume_at_retained_points() {
    run(false, |r| r.hits());
}

#[test]
fn a_point_without_full_depth_windows_is_not_retained() {
    run(false, |r| r.cancelled_mid_triangle());
}

#[test]
fn preemption_and_resume() {
    run(false, |r| r.preemption());
}

#[test]
fn kv_pressure_sheds_before_it_ends_a_sequence() {
    run(false, |r| r.shedding());
}

#[test]
fn a_request_does_not_depend_on_its_batch() {
    run(false, |r| r.batch_invariance());
}

#[test]
fn randomized_interleavings() {
    run(false, |r| r.randomized(0..3));
}

#[test]
fn greedy_speculation_equals_plain_decoding_at_every_width() {
    run(false, |r| r.greedy_speculation());
}

#[test]
fn drafting_at_the_length_limit() {
    drafting(false, 4, |r| r.length_limit());
}

#[test]
fn seeded_sampled_speculation() {
    run(false, |r| r.sampled_speculation());
}

/// Every scenario with the drafter's every slot drafted.
#[test]
fn the_scenarios_with_drafts() {
    drafting(false, 4, |r| all_scenarios(r, false));
}

/// The window of 80 with five slots: the drafter's context list spans two blocks.
#[test]
fn drafting_past_one_attention_block() {
    drafting(true, 5, |r| drafting_scenarios(r, true));
}

/// Idle KV is ciphertext, a recomputed block reseals to the same bytes, and the host
/// tier round-trips every group's ciphertext.
#[test]
fn the_memory_scenarios() {
    drafting(false, 4, |r| memory_scenarios(r, false));
}

/// The window of 80: a query's list spans two 64-slot attention blocks.
#[test]
fn a_window_past_one_attention_block() {
    run(true, |r| {
        r.chunked_prefill(&[7, 1000]);
        r.hits();
        r.cancelled_mid_triangle();
        r.batch_invariance();
    });
}

/// The summary the device run writes, from a host run.
#[test]
fn the_summary_records_every_scenario() {
    let model = model(false);
    let mut run = Run {
        model: &model,
        device: &host,
        outcomes: Vec::new(),
        width: 0,
    };
    run.single_requests();
    run.shedding();
    let json = summary("host", &run.outcomes);
    let scenarios = json["scenarios"].as_array().unwrap();
    assert_eq!(scenarios.len(), 2);
    assert_eq!(scenarios[0]["name"], "tiny/single_requests");
    assert!(scenarios[0]["logit_rows_equal"].as_u64().unwrap() > 0);
    let steps = scenarios[0]["per_step"].as_array().unwrap();
    assert!(!steps.is_empty());
    assert!(steps.iter().all(|s| s["total_ms"].as_f64().unwrap() > 0.0));
    assert!(scenarios[0]["step_ms"]["encoder"].as_f64().unwrap() > 0.0);
}

/// What the device run runs, with every launch served by the device kernels' own
/// bodies, compiled for the host, behind the launch wrappers' checks and grids: the
/// fixture, then the window of 80. Skipped (after saying why) without a C++ compiler.
fn compiled_bodies(drafting: bool, scenarios: impl Fn(&mut Run<'_>, bool)) {
    let Some(lib) = compiled::library() else {
        return;
    };
    let device =
        move || -> Box<dyn Device> { Box::new(compiled::CompiledDevice::new(lib.clone())) };
    for wide in [false, true] {
        let model = model(wide);
        let mut run = Run {
            model: &model,
            device: &device,
            outcomes: Vec::new(),
            width: if drafting { model.slots } else { 0 },
        };
        scenarios(&mut run, wide);
        assert!(run.outcomes.iter().all(|o| o.compared.blocks > 0));
    }
}

#[test]
fn the_compiled_kernel_bodies_run_the_scenarios() {
    compiled_bodies(false, all_scenarios);
}

#[test]
fn the_compiled_kernel_bodies_draft() {
    compiled_bodies(true, drafting_scenarios);
}

#[test]
fn the_compiled_kernel_bodies_keep_their_memory() {
    compiled_bodies(true, memory_scenarios);
}
