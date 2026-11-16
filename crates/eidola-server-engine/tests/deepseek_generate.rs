//! The DeepSeek-V4.1 generation runner (`examples/deepseek_generate`) over the DeepSeek dev
//! model: its committed prompts generate through the node's loader, pipeline, executor
//! configuration and decoder; every request is checkpointed and a rerun skips it; the
//! width-0 pass matches the drafting one token for token; the draft counts per slot add
//! up to the engine's own; the time limit refuses after calibration, and before any work
//! once calibrated. The dev model's output is gibberish: this tests the plumbing.

mod common;
#[allow(dead_code)]
#[path = "../examples/deepseek_generate/runner.rs"]
mod runner;

use std::path::Path;

use common::{deepseek_loaded, deepseek_model_dir, deepseek_weights_hash};
use runner::{Outcome, PromptSpec, RunConfig, run};
use serde_json::Value;

fn config(out: &Path) -> RunConfig {
    RunConfig {
        weights_dir: deepseek_model_dir().to_path_buf(),
        weights_sha256: deepseek_weights_hash().to_string(),
        out_dir: out.to_path_buf(),
        prompts: PromptSpec::parse_file(include_bytes!(
            "../examples/deepseek_generate/prompts.json"
        ))
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
        // The prompt and the output as ids, with each generated token's own piece.
        let saved = read(&out.path().join(format!(
            "requests/{}.w{width}.json",
            r["name"].as_str().unwrap()
        )));
        assert_eq!(
            saved["prompt_token_ids"].as_array().unwrap().len() as u64,
            saved["prompt_tokens"].as_u64().unwrap()
        );
        let pieces = saved["generated_pieces"].as_array().unwrap();
        let ids: Vec<&Value> = pieces.iter().map(|p| &p["id"]).collect();
        assert_eq!(
            ids,
            saved["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        );
        assert!(
            pieces
                .iter()
                .all(|p| p["text"].is_string() || p["bytes"].is_string())
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

    // A saved result made from other inputs is run again, never relabelled. A new token
    // limit reruns every request (greedy: each a prefix of the longer one's tokens).
    let snapshot = |out: &Path| -> Vec<Vec<u8>> {
        names
            .iter()
            .map(|n| std::fs::read(out.join(format!("requests/{n}.json"))).unwrap())
            .collect()
    };
    let longer: Vec<Value> = names
        .iter()
        .map(|n| read(&out.path().join(format!("requests/{n}.json"))))
        .collect();
    let mut shorter = config(out.path());
    shorter.max_new_tokens = 6;
    assert_eq!(run(&shorter).unwrap(), Outcome::Complete { mismatches: 0 });
    for (n, long) in names.iter().zip(&longer) {
        let short = read(&out.path().join(format!("requests/{n}.json")));
        let (s, l) = (
            short["token_ids"].as_array().unwrap(),
            long["token_ids"].as_array().unwrap(),
        );
        assert!(s.len() <= 6 && l.starts_with(s), "{n} was run again");
        assert_ne!(short["fingerprint"], long["fingerprint"]);
    }
    // A changed prompt reruns its own requests only.
    let mut edited = shorter.clone();
    edited.prompts[2].messages = eidola_engine_chat::json::parse(
        r#"[{"role": "user", "content": "Write a Rust function that sums a slice."}]"#,
    )
    .unwrap();
    let before = snapshot(out.path());
    assert_eq!(run(&edited).unwrap(), Outcome::Complete { mismatches: 0 });
    for (n, (now, then)) in names.iter().zip(snapshot(out.path()).iter().zip(&before)) {
        assert_eq!(now != then, n == "code.w4", "{n}");
    }
    // A result whose fingerprint is not this run's (another build or checkpoint made it)
    // is run again, and the summary carries this run's.
    let plain0 = out.path().join("requests/plain.w0.json");
    let mut foreign = read(&plain0);
    foreign["fingerprint"] = "0".repeat(64).into();
    foreign["token_ids"] = serde_json::json!([1, 2, 3]);
    std::fs::write(&plain0, serde_json::to_vec(&foreign).unwrap()).unwrap();
    let before = snapshot(out.path());
    assert_eq!(run(&edited).unwrap(), Outcome::Complete { mismatches: 0 });
    for (n, (now, then)) in names.iter().zip(snapshot(out.path()).iter().zip(&before)) {
        assert_eq!(now != then, n == "plain.w0", "{n}");
    }
    let summary = read(&out.path().join("summary.json"));
    let plain = summary["requests"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "plain" && r["draft_width"] == 0)
        .unwrap();
    assert_ne!(plain["token_ids"], serde_json::json!([1, 2, 3]));
    assert!(
        summary["width0_checks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["identical"] == true)
    );

    // The text summary's transcripts are this run's requests', whatever else the
    // directory holds.
    std::fs::write(out.path().join("transcripts/gone.w9.txt"), "STALE").unwrap();
    assert_eq!(run(&edited).unwrap(), Outcome::Complete { mismatches: 0 });
    let text = std::fs::read_to_string(out.path().join("transcripts.txt")).unwrap();
    assert!(!text.contains("STALE"));
    assert_eq!(text.matches("(draft width ").count(), names.len());
    for n in &names {
        let (name, width) = n.split_once(".w").unwrap();
        assert!(
            text.contains(&format!("== {name} (draft width {width})")),
            "{n}"
        );
    }
    let cal = read(&out.path().join("calibration.json"));
    assert!(cal["step_secs_by_width"]["4"].as_f64().unwrap() > 0.0);
    assert!(cal["step_secs_by_width"]["0"].as_f64().unwrap() > 0.0);

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
    assert_eq!(
        cal["step_secs_by_width"],
        serde_json::json!({}),
        "refused at the prefill, before any decode step"
    );
    let finished = std::fs::read_dir(out.path().join("requests"))
        .unwrap()
        .count();
    assert_eq!(finished, 0, "the calibrating request is not checkpointed");

    // Calibrated: refused before the checkpoint is read (here, one that is not there).
    let mut elsewhere = cfg.clone();
    elsewhere.weights_dir = out.path().join("no-checkpoint-here");
    let Outcome::Refused(why) = run(&elsewhere).unwrap() else {
        panic!("not refused");
    };
    assert!(why.contains("nothing was loaded"), "{why}");
    // A change to the requests leaves their lengths unknown before loading, so the
    // refusal waits for the load: the checkpoint is read again.
    let mut changed = elsewhere.clone();
    changed.max_new_tokens = 7;
    let e = run(&changed).unwrap_err();
    assert!(e.contains("weights directory"), "{e}");
}

/// With one new token every request ends at its prefill: the prefill's measured cost
/// alone refuses a run over the limit, though no decode step is ever timed.
#[test]
fn one_new_token_is_bounded_by_the_prefill_alone() {
    let out = tempfile::tempdir().unwrap();
    let mut cfg = config(out.path());
    cfg.max_new_tokens = 1;
    cfg.time_limit_secs = 1e-6;
    let Outcome::Refused(why) = run(&cfg).unwrap() else {
        panic!("not refused");
    };
    assert!(why.contains("over the limit"), "{why}");
    let cal = read(&out.path().join("calibration.json"));
    assert!(cal["prefill_secs_per_position"].as_f64().unwrap() > 0.0);
    assert_eq!(cal["step_secs_by_width"], serde_json::json!({}));
    let finished = std::fs::read_dir(out.path().join("requests"))
        .unwrap()
        .count();
    assert_eq!(finished, 0);

    cfg.time_limit_secs = 3600.0;
    assert_eq!(run(&cfg).unwrap(), Outcome::Complete { mismatches: 0 });
    let summary = read(&out.path().join("summary.json"));
    for r in summary["requests"].as_array().unwrap() {
        assert_eq!(r["generated_tokens"], 1, "{r}");
        assert_eq!(r["decode"]["steps"], 0, "{r}");
    }
}

/// The runner's prompt is the node's for the prompts file's text: a tool's schema keeps
/// the key order the file gives it, as the format renders it from the text a client sent.
#[test]
fn a_tool_prompt_is_the_one_the_node_renders_from_the_files_text() {
    // The file's `tool` entry, as a client sends it: every key where the file has it.
    let messages = r#"[{"role": "user", "content": "What is the weather in Paris right now?"}]"#;
    let tools = r#"[{"type": "function", "function": {"name": "get_weather", "description": "The current weather in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The city's name."}}, "required": ["city"]}}}]"#;
    let file = PromptSpec::parse_file(include_bytes!("../examples/deepseek_generate/prompts.json"))
        .unwrap();
    let tool = file.iter().find(|p| p.name == "tool").unwrap();
    let parse = |t: &str| eidola_engine_chat::json::parse(t).unwrap();
    assert_eq!(tool.messages, parse(messages));
    assert_eq!(tool.tools.as_ref(), Some(&parse(tools)));

    let body = format!(
        r#"{{"model": "m", "messages": {messages}, "tools": {tools}, "max_completion_tokens": 1, "temperature": 0.0, "stream": true, "stream_options": {{"include_usage": true}}}}"#
    );
    let valid = eidola_server_engine::api::parse_request(body.as_bytes(), "m").unwrap();
    let node = eidola_server_engine::pipeline::prepare(
        &deepseek_loaded(),
        &eidola_engine::secret::SaltDeriver::new(),
        1024,
        1,
        valid,
    )
    .unwrap()
    .request
    .prompt;

    let out = tempfile::tempdir().unwrap();
    let mut cfg = config(out.path());
    cfg.prompts = vec![tool.clone()];
    cfg.max_new_tokens = 1;
    cfg.draft_width = Some(0);
    assert_eq!(run(&cfg).unwrap(), Outcome::Complete { mismatches: 0 });
    let saved = read(&out.path().join("requests/tool.w0.json"));
    let ours: Vec<u32> = saved["prompt_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ours, node);
    assert!(
        saved["prompt_text"]
            .as_str()
            .unwrap()
            .contains(r#"{"name": "get_weather", "description": "The current weather in a city.", "parameters": {"type": "object""#),
        "the schema in the file's key order"
    );
}
