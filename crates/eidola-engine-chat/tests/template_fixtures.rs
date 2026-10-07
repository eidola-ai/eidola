//! Byte-exact comparison against `transformers.apply_chat_template`.
//!
//! The expected strings in `fixtures/template_cases.json` were produced by the
//! reference implementation (see `dev/gen_template_fixtures.py`).

use std::path::PathBuf;

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
