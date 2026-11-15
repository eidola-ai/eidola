//! The DeepSeek-V4.1 generation runner (`examples/ds_generate`) over the DeepSeek dev
//! model: its committed prompts generate through the node's loader, pipeline, executor
//! configuration and decoder; every request is checkpointed and a rerun skips it; the
//! width-0 pass matches the drafting one token for token; the draft counts per slot add
//! up to the engine's own; the time limit refuses after calibration, and before any work
//! once calibrated. The dev model's output is gibberish: this tests the plumbing.

mod common;
#[allow(dead_code)]
#[path = "../examples/ds_generate/runner.rs"]
mod runner;

use std::path::Path;

use common::{deepseek_model_dir, deepseek_weights_hash};
use runner::{Outcome, PromptSpec, RunConfig, run};
use serde_json::Value;

fn config(out: &Path) -> RunConfig {
    RunConfig {
        weights_dir: deepseek_model_dir().to_path_buf(),
        weights_sha256: deepseek_weights_hash().to_string(),
        out_dir: out.to_path_buf(),
        prompts: PromptSpec::parse_file(include_bytes!("../examples/ds_generate/prompts.json"))
            .unwrap(),
        max_new_tokens: 8,
        draft_width: None,
        time_limit_secs: 3600.0,
        prefill_chunk: 512,
        block_size: 16,
        min_available_bytes: 0,
    }
}

fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn the_runner_generates_checkpoints_and_resumes() {
    let out = tempfile::tempdir().unwrap();
    let cfg = config(out.path());
    assert_eq!(run(&cfg).unwrap(), Outcome::Complete { mismatches: 0 });

    let summary = read(&out.path().join("summary.json"));
    // The dev model's drafter has four slots: the default width.
    assert_eq!(summary["draft_width"], 4);
    assert_eq!(summary["weights_sha256"], deepseek_weights_hash());
    let requests = summary["requests"].as_array().unwrap();
    let names: Vec<String> = requests
        .iter()
        .map(|r| format!("{}.w{}", r["name"].as_str().unwrap(), r["draft_width"]))
        .collect();
    assert_eq!(
        names,
        [
            "plain.w4",
            "reasoning.w4",
            "code.w4",
            "tool.w4",
            "plain.w0",
            "reasoning.w0"
        ]
    );
    for r in requests {
        let generated = r["generated_tokens"].as_u64().unwrap();
        assert!((1..=8).contains(&generated), "{r}");
        assert_eq!(r["token_ids"].as_array().unwrap().len() as u64, generated);
        assert_eq!(r["prefill"]["steps"], 1, "one prefill step: {r}");
        let slots = &r["drafts_by_slot"];
        let width = r["draft_width"].as_u64().unwrap();
        // The per-slot counts, read from the executor's rows, are the engine's own.
        assert_eq!(slots["total_proposed"], r["engine_stats"]["drafted"], "{r}");
        if width == 0 {
            assert_eq!(slots["total_proposed"], 0);
        } else if generated > 1 {
            let proposed = slots["proposed"].as_array().unwrap();
            assert_eq!(proposed.len() as u64, width);
            assert!(proposed.iter().all(|p| p == &proposed[0]));
            assert_eq!(proposed[0], r["decode"]["steps"]);
        }
        let reached = slots["reached"].as_array().unwrap();
        let accepted = slots["accepted"].as_array().unwrap();
        for i in 1..reached.len() {
            assert_eq!(
                reached[i],
                accepted[i - 1],
                "slot {i} is reached by acceptance of {}",
                i - 1
            );
        }
        let name = format!("{}.w{width}", r["name"].as_str().unwrap());
        let transcript =
            std::fs::read_to_string(out.path().join("transcripts").join(format!("{name}.txt")))
                .unwrap();
        assert!(
            transcript.contains("<｜begin▁of▁sentence｜>"),
            "{transcript}"
        );
    }
    let checks = summary["width0_checks"].as_array().unwrap();
    assert_eq!(checks.len(), 2);
    assert!(checks.iter().all(|c| c["identical"] == true), "{checks:?}");
    let tool = requests.iter().find(|r| r["name"] == "tool").unwrap();
    assert!(
        tool["prompt_text"].is_null(),
        "the summary leaves the prompt to the transcripts"
    );
    let tool_record = read(&out.path().join("requests/tool.w4.json"));
    assert!(
        tool_record["prompt_text"]
            .as_str()
            .unwrap()
            .contains("get_weather"),
        "the tool is on the system message the format reads"
    );

    // A rerun skips every finished request; one removed is run again, equally.
    let before: Vec<Vec<u8>> = names
        .iter()
        .map(|n| std::fs::read(out.path().join(format!("requests/{n}.json"))).unwrap())
        .collect();
    let code = out.path().join("requests/code.w4.json");
    let code_before = read(&code);
    std::fs::remove_file(&code).unwrap();
    assert_eq!(run(&cfg).unwrap(), Outcome::Complete { mismatches: 0 });
    for (n, bytes) in names.iter().zip(&before) {
        let now = std::fs::read(out.path().join(format!("requests/{n}.json"))).unwrap();
        if n == "code.w4" {
            assert_eq!(read(&code)["token_ids"], code_before["token_ids"]);
            assert_ne!(&now, bytes, "rerun");
        } else {
            assert_eq!(&now, bytes, "{n} was not run again");
        }
    }

    // The wrong hash is refused by the node's loader.
    let mut wrong = config(out.path());
    wrong.weights_sha256 = "0".repeat(64);
    let e = run(&wrong).unwrap_err();
    assert!(e.contains("weights hash"), "{e}");
}

#[test]
fn the_time_limit_refuses_after_calibration_and_then_up_front() {
    let out = tempfile::tempdir().unwrap();
    let mut cfg = config(out.path());
    cfg.time_limit_secs = 1e-6;
    let Outcome::Refused(why) = run(&cfg).unwrap() else {
        panic!("not refused");
    };
    assert!(why.contains("over the limit"), "{why}");
    let cal = read(&out.path().join("calibration.json"));
    assert!(cal["prefill_secs_per_position"].as_f64().unwrap() > 0.0);
    assert!(cal["step_secs_by_width"]["4"].as_f64().unwrap() > 0.0);
    let finished = std::fs::read_dir(out.path().join("requests"))
        .unwrap()
        .count();
    assert_eq!(finished, 0, "the calibrating request is not checkpointed");

    // Calibrated: refused before any request runs.
    let Outcome::Refused(why) = run(&cfg).unwrap() else {
        panic!("not refused");
    };
    assert!(why.contains("calibrated earlier"), "{why}");
}
