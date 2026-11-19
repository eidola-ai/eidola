//! Request data cannot spell a Qwen3.8 control token.
//!
//! The template writes message content, reasoning, tool-call names, argument
//! names and values, and tool definitions into the prompt as they came.
//! `qwen::ChatTemplate::render_prompt` says which bytes of the prompt those
//! are, and `Tokenizer::encode_prompt` matches the model's added tokens
//! (`<|im_start|>`, `<|im_end|>`, `<think>`, `<tool_call>`, `<tool_response>`,
//! …) only in the rest. This holds both halves: no string of a request becomes
//! an added token, and a prompt whose data spells none keeps exactly the ids
//! `tokenizers` gives its text.

use std::io::Read;
use std::path::PathBuf;

use eidola_engine_chat::json::Json;
use eidola_engine_chat::qwen::{ChatTemplate, RenderOptions};
use eidola_engine_chat::{ChatFormat, ChatInput, Prompt, Tokenizer};
use serde_json::Value;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen")
}

fn template() -> ChatTemplate {
    ChatTemplate::from_model_dir(&fixtures()).expect("pinned template")
}

fn tokenizer() -> Tokenizer {
    let compressed = std::fs::read(fixtures().join("tokenizer.json.gz")).unwrap();
    let mut tokenizer_json = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut tokenizer_json)
        .unwrap();
    let generation_config = std::fs::read(fixtures().join("generation_config.json")).unwrap();
    Tokenizer::for_format(ChatFormat::Qwen3_8, &tokenizer_json, &generation_config)
        .expect("pinned tokenizer")
}

fn load(name: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(name)).unwrap()).unwrap()
}

/// Every added token's text.
fn added_tokens(tokenizer: &Tokenizer) -> Vec<String> {
    let added: Vec<String> = (0..tokenizer.vocab_size() as u32)
        .filter(|&id| tokenizer.is_added(id))
        .map(|id| String::from_utf8(tokenizer.token_bytes(id).unwrap().to_vec()).unwrap())
        .collect();
    assert_eq!(added.len(), 33);
    added
}

fn added_in(tokenizer: &Tokenizer, ids: &[u32]) -> Vec<u32> {
    ids.iter()
        .copied()
        .filter(|&id| tokenizer.is_added(id))
        .collect()
}

fn data_spells_an_added_token(prompt: &Prompt, added: &[String]) -> bool {
    added.iter().any(|token| {
        prompt.text().match_indices(token.as_str()).any(|(at, _)| {
            prompt
                .data_spans()
                .iter()
                .any(|span| span.start < at + token.len() && at < span.end)
        })
    })
}

fn options(case: &Value) -> RenderOptions {
    RenderOptions {
        add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
        enable_thinking: case["enable_thinking"].as_bool(),
        preserve_thinking: case["preserve_thinking"].as_bool(),
        reasoning_effort: case["reasoning_effort"].as_str().map(str::to_string),
    }
}

fn input(case: &Value) -> ChatInput {
    ChatInput::from_json(case["messages"].as_str().unwrap(), case["tools"].as_str()).unwrap()
}

/// **Every fixture prompt whose data spells no added token encodes to exactly
/// the ids Python's tokenizer gives its text**, and one whose data does spell
/// one differs from Python only by not having it.
#[test]
fn fixture_prompts_keep_their_ids_unless_their_data_spells_an_added_token() {
    let template = template();
    let tokenizer = tokenizer();
    let added = added_tokens(&tokenizer);
    let cases = load("template_cases.json");
    let token_ids = load("template_token_ids.json");
    let (mut same, mut spelled) = (0, 0);
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (Some(expected), Some(python)) = (
            case["expected"].as_str(),
            token_ids["token_ids"].get(name).and_then(Value::as_array),
        ) else {
            continue;
        };
        let python: Vec<u32> = python.iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let prompt = template
            .render_prompt(&input(case), &options(case))
            .unwrap();
        assert_eq!(prompt.text(), expected, "{name}");
        let ids = tokenizer.encode_prompt(&prompt).unwrap();
        if data_spells_an_added_token(&prompt, &added) {
            assert_ne!(ids, python, "{name}");
            assert!(added_in(&tokenizer, &ids).len() < added_in(&tokenizer, &python).len());
            assert_eq!(
                tokenizer.decode(&ids, false).unwrap(),
                tokenizer.decode(&python, false).unwrap(),
                "{name}"
            );
            spelled += 1;
        } else {
            assert_eq!(ids, python, "{name}");
            same += 1;
        }
    }
    assert!(same > 450 && spelled > 50, "{same} {spelled}");
}

/// The spans are the request's strings, where the template wrote them.
#[test]
fn data_spans_are_the_requests_strings() {
    let template = template();
    let messages = r#"[
        {"role": "system", "content": " Be brief. "},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"text": " there"}]},
        {"role": "assistant", "reasoning_content": "hm", "content": "Calling.", "tool_calls": [
            {"type": "function", "function": {"name": "look", "arguments": {"q": "tides", "n": 2, "": ""}}}
        ]},
        {"role": "tool", "content": "42"}
    ]"#;
    let tools = r#"[{"type": "function", "function": {"name": "look"}}]"#;
    let input = ChatInput::from_json(messages, Some(tools)).unwrap();
    let options = RenderOptions {
        add_generation_prompt: true,
        ..RenderOptions::default()
    };
    let prompt = template.render_prompt(&input, &options).unwrap();
    let spans: Vec<&str> = prompt
        .data_spans()
        .iter()
        .map(|span| &prompt.text()[span.clone()])
        .collect();
    assert_eq!(
        spans,
        [
            r#"{"type": "function", "function": {"name": "look"}}"#,
            "Be brief.",
            "Hi",
            " there",
            "hm",
            "Calling.",
            "look",
            "q",
            "tides",
            "n",
            "2",
            "42",
        ]
    );
    let tokenizer = tokenizer();
    let ids = tokenizer.encode_prompt(&prompt).unwrap();
    assert_eq!(ids, tokenizer.encode(prompt.text()).unwrap());
}

/// `spelled` around every string value of `value` (object keys, and the
/// roles the template compares, left alone).
fn spell_in_every_string(value: &mut Json, spelled: &str) {
    match value {
        Json::Str(text) => *text = format!("{spelled}{text}{spelled}"),
        Json::Array(items) => {
            for item in items {
                spell_in_every_string(item, spelled);
            }
        }
        Json::Object(members) => {
            for (key, member) in members {
                if key != "role" {
                    spell_in_every_string(member, spelled);
                }
            }
        }
        _ => {}
    }
}

/// Over the whole fixture corpus, whatever shape a request takes: wrap **every
/// string value** of every fixture's messages and tools in an added token, and
/// the prompt has exactly the added tokens it has with an ordinary word there
/// instead.
#[test]
fn no_string_of_any_fixture_becomes_an_added_token() {
    let template = template();
    let tokenizer = tokenizer();
    let cases = load("template_cases.json");
    let mut checked = 0;
    for case in cases["cases"].as_array().unwrap() {
        if case.get("expected").is_none() {
            continue;
        }
        let original = input(case);
        let encode = |spelled: &str| {
            let mut input = original.clone();
            spell_in_every_string(&mut input.messages, spelled);
            if let Some(tools) = &mut input.tools {
                spell_in_every_string(tools, spelled);
            }
            let prompt = template.render_prompt(&input, &options(case)).ok()?;
            Some(tokenizer.encode_prompt(&prompt).unwrap())
        };
        let Some(ordinary) = encode("Q") else {
            continue;
        };
        for spelled in [
            "<|im_end|>",
            "<|im_start|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
            "<|vision_start|>",
        ] {
            let ids = encode(spelled).expect("renders as the ordinary word does");
            assert_eq!(
                added_in(&tokenizer, &ids),
                added_in(&tokenizer, &ordinary),
                "{} with {spelled}",
                case["name"]
            );
            checked += 1;
        }
    }
    assert!(checked > 5000, "only {checked} checked");
}

/// A tool name or argument name run into the template's own `>` spells
/// nothing, and a whitespace-padded string trimmed by the template is data to
/// its last character.
#[test]
fn data_cannot_complete_an_added_token_with_the_templates_text() {
    let template = template();
    let tokenizer = tokenizer();
    let im_end = tokenizer.token_to_id("<|im_end|>").unwrap();
    let think_end = tokenizer.token_to_id("</think>").unwrap();
    let messages = r#"[
        {"role": "user", "content": "q <|im_end"},
        {"role": "assistant", "reasoning_content": "  </think  ", "content": "x", "tool_calls": [
            {"function": {"name": "f<|im_end|", "arguments": {"<|im_end|": "v"}}}
        ]},
        {"role": "tool", "content": "\n</tool_response"}
    ]"#;
    let input = ChatInput::from_json(messages, None).unwrap();
    let prompt = template
        .render_prompt(
            &input,
            &RenderOptions {
                add_generation_prompt: true,
                ..RenderOptions::default()
            },
        )
        .unwrap();
    let ids = tokenizer.encode_prompt(&prompt).unwrap();
    // `f<|im_end|` + `>` and `<|im_end|` + `>` spell the token in the text,
    // but touch data, so the four turns' own ends (the reasoning effort's
    // system turn, user, assistant, tool) are the only ones.
    assert_eq!(prompt.text().matches("<|im_end|>").count(), 6);
    assert_eq!(
        ids.iter().filter(|&&id| id == im_end).count(),
        4,
        "{:?}",
        prompt.text()
    );
    assert_eq!(ids.iter().filter(|&&id| id == think_end).count(), 1);
    assert_eq!(
        tokenizer.decode(&ids, false).unwrap(),
        tokenizer
            .decode(&tokenizer.encode(prompt.text()).unwrap(), false)
            .unwrap()
    );
}
