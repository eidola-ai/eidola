//! Byte-exact comparison of Qwen3.8 prompts against
//! `transformers.apply_chat_template`.
//!
//! The expected strings in `fixtures/qwen/template_cases.json` were produced by
//! the reference implementation (`dev/qwen_gen_template_fixtures.py`).

use std::path::PathBuf;

use eidola_engine_chat::json::{self, Json};
use eidola_engine_chat::qwen::{ChatTemplate, RenderOptions};
use eidola_engine_chat::{ChatError, ChatInput};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen")
}

fn first_difference(a: &str, b: &str) -> String {
    let index = a
        .char_indices()
        .zip(b.chars())
        .find(|((_, x), y)| x != y)
        .map(|((i, _), _)| i)
        .unwrap_or(a.len().min(b.len()));
    let start = a.floor_char_boundary(index.saturating_sub(40));
    format!(
        "at byte {index}:\n  rust:   {:?}\n  python: {:?}",
        &a[start..a.ceil_char_boundary((index + 40).min(a.len()))],
        &b[b.floor_char_boundary(start.min(b.len()))
            ..b.ceil_char_boundary((index + 40).min(b.len()))]
    )
}

/// A case's `apply_chat_template` keyword arguments, each absent unless set.
fn options(case: &serde_json::Value) -> RenderOptions {
    RenderOptions {
        add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
        enable_thinking: case["enable_thinking"].as_bool(),
        preserve_thinking: case["preserve_thinking"].as_bool(),
        reasoning_effort: case["reasoning_effort"].as_str().map(str::to_string),
    }
}

fn corpus() -> serde_json::Value {
    let text = std::fs::read_to_string(fixtures().join("template_cases.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn input(case: &serde_json::Value) -> Result<ChatInput, ChatError> {
    ChatInput::from_json(case["messages"].as_str().unwrap(), case["tools"].as_str())
}

#[test]
fn renders_every_fixture_byte_exactly() {
    let corpus = corpus();
    let template = ChatTemplate::from_model_dir(&fixtures()).expect("pinned template");
    assert_eq!(
        corpus["meta"]["template_sha256"].as_str(),
        Some(template.sha256())
    );
    let mut failures = Vec::new();
    let (mut rendered, mut refused) = (0, 0);
    for case in corpus["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let result = input(case).and_then(|input| template.render(&input, &options(case)));
        match (case.get("expected").and_then(|e| e.as_str()), result) {
            (Some(expected), Ok(actual)) => {
                rendered += 1;
                if actual != expected {
                    failures.push(format!("{name}: {}", first_difference(&actual, expected)));
                }
            }
            (Some(_), Err(e)) => failures.push(format!("{name}: unexpected error {e}")),
            (None, Ok(actual)) => {
                failures.push(format!("{name}: expected an error, rendered {actual:?}"))
            }
            (None, Err(e)) => {
                refused += 1;
                let kind_matches = match case["reject"].as_str().unwrap() {
                    "unsupported" => matches!(e, ChatError::UnsupportedContent(_)),
                    "invalid" => matches!(e, ChatError::InvalidInput(_)),
                    "template" => matches!(e, ChatError::Template(_)),
                    other => panic!("{name}: unknown refusal {other}"),
                };
                if !kind_matches {
                    failures.push(format!("{name}: refused as {e:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(rendered >= 500, "only {rendered} rendered fixtures");
    assert!(refused >= 30, "only {refused} refused fixtures");
}

/// Re-encodes every tool call's object `arguments` as the JSON string an
/// OpenAI client would send, and returns how many it changed.
fn stringify_arguments(messages: &mut Json) -> usize {
    let mut changed = 0;
    let Json::Array(messages) = messages else {
        return 0;
    };
    for message in messages {
        let Json::Object(fields) = message else {
            continue;
        };
        let Some((_, Json::Array(calls))) = fields.iter_mut().find(|(k, _)| k == "tool_calls")
        else {
            continue;
        };
        for call in calls {
            let Json::Object(call) = call else { continue };
            let body = match call.iter_mut().find(|(k, _)| k == "function") {
                Some((_, Json::Object(body))) => body,
                Some(_) => continue,
                None => call,
            };
            if let Some((_, arguments)) = body.iter_mut().find(|(k, _)| k == "arguments")
                && matches!(arguments, Json::Object(_))
            {
                *arguments = Json::Str(json::dumps_default(arguments));
                changed += 1;
            }
        }
    }
    changed
}

/// Arguments sent as a JSON string render exactly as the object they encode.
#[test]
fn string_arguments_render_as_the_object_they_encode() {
    let corpus = corpus();
    let template = ChatTemplate::from_model_dir(&fixtures()).unwrap();
    let mut checked = 0;
    for case in corpus["cases"].as_array().unwrap() {
        let Some(expected) = case["expected"].as_str() else {
            continue;
        };
        let mut input = input(case).unwrap();
        if stringify_arguments(&mut input.messages) == 0 {
            continue;
        }
        let actual = template.render(&input, &options(case)).unwrap();
        assert_eq!(actual, expected, "{}", case["name"]);
        checked += 1;
    }
    assert!(checked >= 60, "only {checked} cases with arguments");
}
