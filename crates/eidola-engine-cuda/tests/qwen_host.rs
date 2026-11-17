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
fn one_token_blocks_without_drafts_keep_no_tail() {
    if let Some(lib) = qwen_common::library() {
        scenarios::empty_tails(&|| qwen_common::host_device(lib.clone()));
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
