//! Qwen3.8 output parsing against vLLM's `qwen3` / `qwen3_coder` parser, as
//! recorded by `dev/qwen_gen_vllm_fixtures.py` in
//! `fixtures/qwen/vllm_parser_cases.json`.
//!
//! On every well-formed output (an assistant turn as the model's template
//! renders it) the split into reasoning, content and calls, and every call's
//! arguments, are vLLM's. On truncated and malformed outputs the two rules
//! part where vLLM's is lossy; each such case is listed below with the reason,
//! and every case not listed agrees.

use std::path::PathBuf;

use eidola_engine_chat::json;
use eidola_engine_chat::pysem::py_strip;
use eidola_engine_chat::{
    ChatDelta, OutputConfig, OutputParser, PrefixedCallIds, StopCause, ToolSchemas,
};
use serde_json::Value;

/// Where this crate's rule (the crate's `AGENTS.md`, Qwen3.8) and vLLM's part.
const DIFFERENCES: &[(&str, &str)] = &[
    (
        "edge/truncated_in_call",
        "an unclosed block is the answer's text; vLLM emits a call from it",
    ),
    (
        "edge/truncated_after_function",
        "an unclosed block is the answer's text; vLLM emits a call from it",
    ),
    (
        "edge/think_open_mid_reasoning",
        "only leading <think> tags are dropped; vLLM drops one anywhere in the reasoning",
    ),
    (
        "edge/repeated_think_end",
        "the answer keeps a later </think>; vLLM drops it",
    ),
    (
        "edge/function_without_tool_call",
        "a call is a <tool_call> block; vLLM also reads a bare <function=",
    ),
    (
        "edge/undeclared_function",
        "an undeclared function's block is the answer's text; vLLM calls it",
    ),
    (
        "edge/missing_parameter_end",
        "a value ends at </parameter>; vLLM also ends one at the next <parameter=",
    ),
    (
        "edge/spaced_names",
        "names are stripped of whitespace; vLLM keeps a parameter name's",
    ),
    (
        "edge/text_after_call",
        "the answer keeps text after a call; vLLM drops it",
    ),
    (
        "edge/text_between_calls",
        "the answer keeps text between calls; vLLM drops it",
    ),
    (
        "edge/prose_mentions_tag",
        "prose that mentions <tool_call> is the answer's text; vLLM drops what follows it",
    ),
    (
        "edge/empty_function_name",
        "a block that is not a call is the answer's text; vLLM drops it",
    ),
    (
        "edge/thinking_off_tags",
        "with thinking off the answer is verbatim; vLLM drops </think> from it",
    ),
    (
        "edge/no_tools_call_markup",
        "with no tools offered markup is the answer's text; vLLM still parses calls",
    ),
];

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen")
}

/// Our parse of a whole output, fed in `chunk`-character pieces.
fn parse(case: &Value, chunk: usize) -> ChatDelta {
    let thinking = case["thinking"].as_bool().unwrap();
    let tools = case["tools"]
        .as_str()
        .map(|t| ToolSchemas::from_tools(&json::parse(t).unwrap()));
    let mut parser = OutputParser::new(
        OutputConfig::qwen3_8((!thinking).then_some(false), tools),
        PrefixedCallIds("call_".into()),
    );
    let output: Vec<char> = case["output"].as_str().unwrap().chars().collect();
    let mut all = ChatDelta::default();
    let mut add = |delta: ChatDelta| {
        all.reasoning_content.push_str(&delta.reasoning_content);
        all.content.push_str(&delta.content);
        all.tool_calls.extend(delta.tool_calls);
    };
    for piece in output.chunks(chunk) {
        add(parser.push_text(&piece.iter().collect::<String>()));
    }
    add(parser.finish(StopCause::EndOfSequence).0);
    all
}

/// Whether our parse is vLLM's: reasoning and calls exactly, and the content
/// exactly, or, when vLLM called tools (where it strips the content), stripped.
fn agrees(ours: &ChatDelta, vllm: &Value) -> bool {
    let text = |v: &Value| v.as_str().unwrap_or("").to_string();
    let calls: Vec<(String, String)> = vllm["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (text(&c[0]), text(&c[1])))
        .collect();
    let our_calls: Vec<(String, String)> = ours
        .tool_calls
        .iter()
        .map(|c| (c.name.clone(), c.arguments.clone()))
        .collect();
    let content = if calls.is_empty() {
        ours.content.as_str()
    } else {
        py_strip(&ours.content)
    };
    ours.reasoning_content == text(&vllm["reasoning"])
        && content == text(&vllm["content"])
        && our_calls == calls
}

#[test]
fn every_case_agrees_with_vllm_or_is_a_listed_difference() {
    let corpus: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("vllm_parser_cases.json")).unwrap(),
    )
    .unwrap();
    let (mut well_formed, mut failures) = (0, Vec::new());
    for case in corpus["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ours = parse(case, usize::MAX);
        // Streaming in any pieces is the same parse.
        for chunk in [1, 3, 7] {
            assert_eq!(
                parse(case, chunk),
                ours,
                "{name} in {chunk}-character pieces"
            );
        }
        let listed = DIFFERENCES.iter().any(|(n, _)| *n == name);
        let same = agrees(&ours, &case["vllm"]);
        if case["well_formed"].as_bool().unwrap() {
            well_formed += 1;
            assert!(!listed, "{name} is well-formed");
        }
        if same == listed {
            failures.push(format!(
                "{name}: {} vLLM\n  ours: {ours:?}\n  vLLM: {}",
                if same { "agrees with" } else { "differs from" },
                case["vllm"]
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} cases:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(well_formed >= 140, "only {well_formed} well-formed cases");
}
