//! The Qwen3.8 CUDA executor in its exact profile on host memory, every launch the
//! kernels' own bodies compiled for the host, held to the CPU reference executor step for
//! step through the real engine (`qwen_common::pair`, `qwen_common::scenarios`): tokens,
//! logits bits, state images bytes and encryption, and the memory scan after every step;
//! then reproducibility between calls (`qwen_common::invariance`). Skipped, saying why,
//! without a C++ compiler.

mod qwen_common;

use qwen_common::scenarios;

#[test]
fn one_request_matches_the_reference() {
    if let Some(dev) = qwen_common::compiled() {
        scenarios::one_request(dev);
    }
}

#[test]
fn requests_match_the_reference_at_every_chunking_and_strategy() {
    if let Some(lib) = qwen_common::library() {
        scenarios::chunkings(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn prefix_hits_preemption_and_the_host_tier_match_the_reference() {
    if let Some(lib) = qwen_common::library() {
        scenarios::hits_and_pressure(&|| qwen_common::host_device(lib.clone()));
    }
}

#[test]
fn a_request_alone_and_beside_others_gives_the_same_tokens() {
    if let Some(lib) = qwen_common::library() {
        scenarios::alone_and_crowded(&|| qwen_common::host_device(lib.clone()));
    }
}

/// Reproducibility between calls in the exact profile on the host-compiled kernels: the
/// contract the fast profile is asserted against on a GPU (`tests/qwen_fast.rs`).
#[test]
fn invariance_holds_on_the_compiled_kernels() {
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::host_device(lib.clone());
    let mut failed = Vec::new();
    for (name, r) in qwen_common::invariance::all(&make) {
        match r {
            Ok(what) => eprintln!("ok      {name}: {what}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The real-weights measurement's path (`QwenCudaExecutor::dense_logits`) against the
/// model crate's forward over one sequence, every position, bit for bit; the slot and
/// blocks it borrowed left zero.
#[test]
fn dense_logits_equal_the_reference_forward() {
    use eidola_engine_cuda::qwen::QwenCudaExecutor;
    use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions};
    use qwen_common::pair::*;
    let Some(dev) = qwen_common::compiled() else {
        return;
    };
    let model = fixture();
    let mut ex =
        QwenCudaExecutor::new(dev, &model, cuda_config(&Geometry::plain(0))).expect("executor");
    let tokens = TestRng(5).tokens(37, VOCAB);
    let got = ex.dense_logits(3, &tokens).expect("logits");
    let opts = QwenForwardOptions {
        logits: LogitRows::All,
        keep_streams: false,
    };
    let want = model
        .forward(&mut model.new_state(), &tokens, &opts)
        .expect("forward");
    for (q, row) in got.iter().enumerate() {
        let w: Vec<u32> = want.logits.row(q).iter().map(|v| v.to_bits()).collect();
        let g: Vec<u32> = row.iter().map(|v| v.to_bits()).collect();
        assert_eq!(g, w, "position {q}");
    }
    assert!(ex.slot_state_is_zero(3));
    assert!((1..=10).all(|b| ex.block_is_zero(0, b)));
}

/// The padded rung path (a decode rung's launch sequence over its table, as a graph
/// records it, run directly: the host captures no graphs) against the CPU reference
/// executor: the same scenarios, and rungs must have run.
#[test]
fn the_padded_rung_path_matches_the_reference() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::host_device(lib.clone());
    let before = qwen_common::rung_steps();
    qwen_common::with_path(CudaGraphs::Off, DecodePath::Direct, || {
        scenarios::one_request(make());
        scenarios::chunkings(&make);
        scenarios::hits_and_pressure(&make);
        scenarios::alone_and_crowded(&make);
    });
    let n = qwen_common::rung_steps() - before;
    eprintln!("{n} steps ran a rung");
    assert!(n > 100, "only {n} steps ran a rung");
}

/// Reproducibility between calls across the paths: the same workloads eagerly and on the
/// padded rung path give the same tokens and logits bits, and every invariance scenario
/// holds on the rung path.
#[test]
fn the_rung_path_and_the_eager_path_agree() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::host_device(lib.clone());
    let mut failed = Vec::new();
    let paths = [
        (CudaGraphs::Off, DecodePath::Eager),
        (CudaGraphs::Off, DecodePath::Direct),
    ];
    for (name, r) in qwen_common::invariance::across_paths(&make, &paths) {
        match r {
            Ok(what) => eprintln!("ok      {name}: {what}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    qwen_common::with_path(CudaGraphs::Off, DecodePath::Direct, || {
        for (name, r) in qwen_common::invariance::all(&make) {
            match r {
                Ok(what) => eprintln!("ok      direct {name}: {what}"),
                Err(e) => {
                    eprintln!("FAILED  direct {name}: {e}");
                    failed.push(name.to_string());
                }
            }
        }
    });
    assert!(failed.is_empty(), "{failed:?}");
}

/// The fast profile's plans on every path without a device, on the fixture with 256-wide
/// heads: FA2's work lists built and packed for every step (prefill chunks, decode rows,
/// the drafter's position pages, the rungs' reservations) and each launch checked,
/// computing nothing. Its attention reads as zeros, so the logits are not the model's,
/// but they are one function of the requests on the eager and rung paths alike, which
/// must agree bit for bit.
#[test]
fn fast_profile_plans_pack_on_every_path() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let make = || qwen_common::planning_fast(lib.clone());
    let paths = [
        (CudaGraphs::Off, DecodePath::Eager),
        (CudaGraphs::Off, DecodePath::Direct),
    ];
    let mut failed = Vec::new();
    let results =
        qwen_common::with_fixture_256(|| qwen_common::invariance::across_paths(&make, &paths));
    for (name, r) in results {
        match r {
            Ok(what) => eprintln!("ok      {name}: {what}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The fixture with 256-wide heads: the model crate's forward equals its goldens (the
/// released model code, pinned) bit for bit, and the exact profile on it equals the CPU
/// reference executor through the scheduler, eagerly and on the padded rung path.
#[test]
fn the_256_wide_fixture_runs_exact() {
    use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath};
    use eidola_engine_model::WeightSet;
    use eidola_engine_model::qwen::{LogitRows, QwenForwardOptions};
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let dir = qwen_common::pair::fixture_dir("qwen_tiny256");
    let golden = WeightSet::open_files(&[dir.join("golden.safetensors")]).expect("goldens");
    let tokens: Vec<u32> = golden
        .get("tokens")
        .and_then(|t| t.to_i64("tokens"))
        .expect("tokens")
        .into_iter()
        .map(|t| u32::try_from(t).expect("a token"))
        .collect();
    let want = golden
        .get("prefill.logits")
        .and_then(|t| t.to_f32("prefill.logits"))
        .expect("logits");
    qwen_common::with_fixture_256(|| {
        let model = qwen_common::pair::fixture();
        assert_eq!(model.config().attention.head_dim, 256);
        let opts = QwenForwardOptions {
            logits: LogitRows::All,
            keep_streams: false,
        };
        let got = model
            .forward(&mut model.new_state(), &tokens, &opts)
            .expect("forward");
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&got.logits.data),
            bits(&want),
            "the forward against its goldens"
        );
        let make = || qwen_common::host_device(lib.clone());
        for (graphs, path) in [
            (CudaGraphs::Off, DecodePath::Eager),
            (CudaGraphs::Off, DecodePath::Direct),
        ] {
            qwen_common::with_path(graphs, path, || {
                scenarios::chunkings(&make);
                scenarios::alone_and_crowded(&make);
            });
        }
    });
}
