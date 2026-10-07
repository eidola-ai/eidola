//! Byte-exact comparison against `transformers.apply_chat_template`.
//!
//! The expected strings in `fixtures/template_cases.json` were produced by the
//! reference implementation (see `dev/gen_template_fixtures.py`).

use std::path::PathBuf;

use eidola_engine_chat::json::{self, Json};
use eidola_engine_chat::{ChatInput, ChatTemplate, RenderOptions};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn load_template() -> ChatTemplate {
    ChatTemplate::from_model_dir(&fixtures()).expect("pinned template")
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

#[test]
fn renders_every_fixture_byte_exactly() {
    let text = std::fs::read_to_string(fixtures().join("template_cases.json")).unwrap();
    let corpus: serde_json::Value = serde_json::from_str(&text).unwrap();
    let template = load_template();
    assert_eq!(
        corpus["meta"]["template_sha256"].as_str(),
        Some(template.sha256())
    );
    let cases = corpus["cases"].as_array().unwrap();
    let mut failures = Vec::new();
    let mut rendered = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let options = RenderOptions {
            add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
            enable_thinking: case["enable_thinking"].as_bool(),
        };
        let result =
            ChatInput::from_json(case["messages"].as_str().unwrap(), case["tools"].as_str())
                .and_then(|input| template.render(&input, options));
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
            (None, Err(_)) => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(rendered >= 200, "only {rendered} rendered fixtures");
}

/// Re-encodes every tool call's object `arguments` as the JSON string an OpenAI
/// client would send, in whichever wrapper the call uses (`function`, `custom`,
/// or the bare call object), and returns how many it changed. Bodies whose
/// `input` is a string are left alone, as the template renders `input` there.
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
            let key = ["function", "custom"]
                .into_iter()
                .find(|key| call.get(key).is_some());
            let Json::Object(call_fields) = call else {
                continue;
            };
            let body = match key {
                Some(key) => match call_fields.iter_mut().find(|(k, _)| k == key) {
                    Some((_, Json::Object(body))) => body,
                    _ => continue,
                },
                None => call_fields,
            };
            if body
                .iter()
                .any(|(k, v)| k == "input" && matches!(v, Json::Str(_)))
            {
                continue;
            }
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

/// String-valued `arguments`, in every tool-call wrapper the template accepts,
/// normalize to exactly what Python renders for the decoded object: every
/// fixture with object arguments is re-sent with them JSON-encoded, normalized,
/// and compared with the reference rendering byte for byte.
#[test]
fn normalized_string_arguments_render_like_python_objects() {
    let text = std::fs::read_to_string(fixtures().join("template_cases.json")).unwrap();
    let corpus: serde_json::Value = serde_json::from_str(&text).unwrap();
    let template = load_template();
    let mut failures = Vec::new();
    let mut per_wrapper = std::collections::BTreeMap::<&str, usize>::new();
    for case in corpus["cases"].as_array().unwrap() {
        let Some(expected) = case["expected"].as_str() else {
            continue;
        };
        let name = case["name"].as_str().unwrap();
        let mut input =
            ChatInput::from_json(case["messages"].as_str().unwrap(), case["tools"].as_str())
                .unwrap();
        if stringify_arguments(&mut input.messages) == 0 {
            continue;
        }
        for wrapper in ["\"custom\"", "\"function\""] {
            if case["messages"].as_str().unwrap().contains(wrapper) {
                *per_wrapper.entry(wrapper).or_default() += 1;
            }
        }
        if name.contains("unwrapped_call") {
            *per_wrapper.entry("bare").or_default() += 1;
        }
        let options = RenderOptions {
            add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
            enable_thinking: case["enable_thinking"].as_bool(),
        };
        let result = input
            .normalize_tool_call_arguments()
            .and_then(|()| template.render(&input, options));
        match result {
            Ok(actual) if actual == expected => {}
            Ok(actual) => failures.push(format!("{name}: {}", first_difference(&actual, expected))),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    for wrapper in ["\"function\"", "\"custom\"", "bare"] {
        assert!(
            per_wrapper.get(wrapper).copied().unwrap_or(0) > 0,
            "{wrapper}: {per_wrapper:?}"
        );
    }
}
