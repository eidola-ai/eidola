//! Differential tests against SGLang's MiMo tool-call detector.
//!
//! Expected values were produced by executing SGLang's own sources (see
//! `dev/gen_sglang_fixtures.py`), with the per-parameter failure containment
//! documented in `src/args.rs` applied around them.
//!
//! The parser's own typing rule is the round-trip one; these tests select
//! SGLang's rule so that the block structure and the Python-semantics pieces
//! both rules share can be compared value for value with SGLang.

use std::path::PathBuf;

use eidola_engine_chat::args::{ArgumentTyping, ToolSchemas, convert_parameter_sglang};
use eidola_engine_chat::json::{self, Json};
use eidola_engine_chat::tool_call::{
    ParsedToolCalls, PrefixedCallIds, ToolCallParser, parse_complete,
};

fn fixture(name: &str) -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn schemas(tools: &str) -> ToolSchemas {
    ToolSchemas::from_tools(&json::parse(tools).unwrap()).with_typing(ArgumentTyping::Sglang)
}

#[test]
fn parameter_conversion_matches_sglang() {
    let corpus = fixture("sglang_param_cases.json");
    let mut failures = Vec::new();
    let mut checked = 0;
    for section in corpus["sections"].as_array().unwrap() {
        let values = section["values"].as_array().unwrap();
        for entry in section["types"].as_array().unwrap() {
            let tools = entry["tools"].as_str().unwrap();
            let schemas = schemas(tools);
            let param_type = schemas.param_type("f", "p");
            for (value, expected) in values.iter().zip(entry["expected"].as_array().unwrap()) {
                let raw = value.as_str().unwrap();
                let expected = match expected.as_str() {
                    Some(text) => text.to_string(),
                    None => json::dumps_default(&Json::Str(raw.to_string())),
                };
                let got = json::dumps_default(&convert_parameter_sglang(raw, &param_type));
                if got != expected {
                    failures.push(format!(
                        "type {tools} ({param_type}) raw {:?}: got {:?}, sglang {:?}",
                        truncate(raw),
                        truncate(&got),
                        truncate(&expected)
                    ));
                }
                checked += 1;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {checked} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(checked > 25_000);
}

fn truncate(s: &str) -> String {
    if s.chars().count() > 120 {
        format!("{}…", s.chars().take(120).collect::<String>())
    } else {
        s.to_string()
    }
}

fn streamed(text: &str, schemas: &ToolSchemas, cut: impl Fn(usize) -> bool) -> ParsedToolCalls {
    let mut parser = ToolCallParser::new(schemas.clone(), PrefixedCallIds("call_".into()));
    let mut out = ParsedToolCalls::default();
    let mut start = 0;
    let mut deltas = Vec::new();
    for (i, _) in text.char_indices().skip(1) {
        if cut(i) {
            deltas.push(parser.push(&text[start..i]));
            start = i;
        }
    }
    deltas.push(parser.push(&text[start..]));
    deltas.push(parser.finish());
    for d in deltas {
        out.content.push_str(&d.content);
        for call in d.calls {
            assert_eq!(call.index, out.calls.len());
            out.calls.push((call.name, call.arguments));
        }
    }
    out
}

#[test]
fn complete_outputs_match_sglang_and_streaming_agrees() {
    let corpus = fixture("sglang_text_cases.json");
    let schemas = schemas(corpus["tools"].as_str().unwrap());
    let mut failures = Vec::new();
    let cases = corpus["cases"].as_array().unwrap();
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let expected = ParsedToolCalls {
            content: case["expected_content"].as_str().unwrap().to_string(),
            calls: case["expected_calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| {
                    (
                        c[0].as_str().unwrap().to_string(),
                        c[1].as_str().unwrap().to_string(),
                    )
                })
                .collect(),
        };
        let got = parse_complete(text, &schemas);
        if got != expected {
            failures.push(format!(
                "{text:?}\n  got:    {got:?}\n  sglang: {expected:?}"
            ));
            continue;
        }
        for (label, every) in [("char", 1usize), ("3", 3), ("7", 7)] {
            let s = streamed(text, &schemas, |i| i % every == 0);
            if s != expected {
                failures.push(format!("{text:?} streamed by {label}: {s:?}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(cases.len() > 500);
}
