//! Byte-exact comparison of the DeepSeek-V4.1 prompt encoder against the
//! model repository's reference (`encoding/encoding.py`), and of the
//! tokenizer against Python `tokenizers`.
//!
//! The expected prompts and token ids in `fixtures/deepseek_v4_1/` were
//! produced by the reference (see `dev/gen_deepseek_fixtures.py`).

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use eidola_engine_chat::deepseek_v4_1::{self, EncodeOptions, ReasoningEffort};
use eidola_engine_chat::json::{self, Json};
use eidola_engine_chat::{ChatError, ChatFormat, ChatInput, Prompt, Tokenizer};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deepseek_v4_1")
}

fn corpus(name: &str) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(fixtures().join(name)).unwrap()).unwrap()
}

/// The pinned tokenizer: the committed copy, or the model directory named by
/// `EIDOLA_DEEPSEEK_MODEL_DIR`.
fn tokenizer() -> Tokenizer {
    if let Some(dir) = std::env::var_os("EIDOLA_DEEPSEEK_MODEL_DIR") {
        return Tokenizer::for_format_from_model_dir(ChatFormat::DeepSeekV4_1, &PathBuf::from(dir))
            .expect("pinned tokenizer");
    }
    let compressed = std::fs::read(fixtures().join("tokenizer.json.gz")).unwrap();
    let mut tokenizer_json = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut tokenizer_json)
        .unwrap();
    let config = std::fs::read(fixtures().join("config.json")).unwrap();
    Tokenizer::for_format(ChatFormat::DeepSeekV4_1, &tokenizer_json, &config)
        .expect("pinned tokenizer")
}

fn options(case: &serde_json::Value) -> EncodeOptions {
    let reasoning_effort = match &case["reasoning_effort"] {
        serde_json::Value::Null => ReasoningEffort::default(),
        serde_json::Value::String(level) => ReasoningEffort::level(level).unwrap(),
        budget => ReasoningEffort::budget(budget.as_u64().unwrap() as u8).unwrap(),
    };
    EncodeOptions {
        thinking: case["thinking"].as_bool().unwrap(),
        reasoning_effort,
        drop_thinking: case["drop_thinking"].as_bool().unwrap(),
    }
}

fn input(case: &serde_json::Value) -> Result<ChatInput, ChatError> {
    ChatInput::from_json(case["messages"].as_str().unwrap(), case["tools"].as_str())
}

fn encode(case: &serde_json::Value) -> Result<Prompt, ChatError> {
    input(case).and_then(|input| deepseek_v4_1::encode(&input, options(case)))
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

fn ids(value: &serde_json::Value) -> Vec<u32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|id| id.as_u64().unwrap() as u32)
        .collect()
}

#[test]
fn encodes_every_fixture_byte_exactly() {
    let corpus = corpus("encoding_cases.json");
    assert_eq!(
        corpus["meta"]["repository"],
        "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    );
    assert_eq!(corpus["meta"]["revision"].as_str().unwrap().len(), 40);
    let mut failures = Vec::new();
    let mut rendered: HashMap<&str, usize> = HashMap::new();
    let (mut refused, mut images) = (0, 0);
    for case in corpus["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let result = encode(case);
        match (case["reject"].as_str(), case["expected"].as_str(), result) {
            (None, Some(expected), Ok(prompt)) => {
                *rendered
                    .entry(case["category"].as_str().unwrap())
                    .or_default() += 1;
                if prompt.text() != expected {
                    failures.push(format!(
                        "{name}: {}",
                        first_difference(prompt.text(), expected)
                    ));
                }
            }
            (None, _, Err(e)) => failures.push(format!("{name}: unexpected error {e}")),
            (Some("image"), _, Err(ChatError::UnsupportedContent(_))) => {
                refused += 1;
                images += 1;
            }
            (Some("invalid"), _, Err(ChatError::InvalidInput(_))) => refused += 1,
            (Some(_), _, Err(e)) => failures.push(format!("{name}: refused with the wrong {e:?}")),
            (Some(_), _, Ok(prompt)) => failures.push(format!(
                "{name}: expected a refusal, rendered {:?}",
                prompt.text()
            )),
            (None, None, Ok(_)) => failures.push(format!("{name}: no expected prompt")),
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The reference's own golden pairs: four rendered, and the vision one
    // refused as an image.
    assert_eq!(rendered["shipped"], 4);
    let generated: usize = rendered.values().sum::<usize>() - rendered["shipped"];
    assert!(generated >= 600, "only {generated} generated fixtures");
    for category in [
        "basic",
        "task",
        "effort",
        "reasoning",
        "mid_system",
        "tools",
        "tool_calls",
        "tool_results",
        "content",
        "text",
        "random",
    ] {
        assert!(rendered[category] >= 20, "{category}");
    }
    assert!(refused >= 60 && images >= 14, "{refused} {images}");
}

/// The byte range of each token in the text its ids decode to.
fn token_spans(tokenizer: &Tokenizer, ids: &[u32]) -> Vec<(u32, std::ops::Range<usize>)> {
    let mut at = 0;
    ids.iter()
        .map(|&id| {
            let len = tokenizer.token_bytes(id).unwrap().len();
            at += len;
            (id, at - len..at)
        })
        .collect()
}

#[test]
fn prompts_and_strings_encode_like_python_tokenizers() {
    let tokenizer = tokenizer();
    let cases = corpus("encoding_cases.json");
    let fixtures = corpus("token_ids.json");
    assert_eq!(
        fixtures["meta"]["tokenizer_sha256"].as_str(),
        Some(tokenizer.sha256())
    );
    let mut failures = Vec::new();
    let (mut checked, mut same, mut spelled) = (0, 0, 0);
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let Some(expected) = fixtures["token_ids"].get(name) else {
            continue;
        };
        let expected = ids(expected);
        let prompt = encode(case).unwrap();
        if tokenizer.encode(prompt.text()).unwrap() != expected {
            failures.push(name.to_string());
        }
        checked += 1;

        // As serving encodes it: the same bytes, no added token on request
        // data, and Python's ids wherever the data spells none.
        let served = tokenizer.encode_prompt(&prompt).unwrap();
        let spans = token_spans(&tokenizer, &served);
        let bytes: Vec<u8> = served
            .iter()
            .flat_map(|&id| tokenizer.token_bytes(id).unwrap().to_vec())
            .collect();
        assert_eq!(bytes, prompt.text().as_bytes(), "{name}");
        for (id, span) in &spans {
            if tokenizer.is_added(*id) {
                let touches_data = prompt
                    .data_spans()
                    .iter()
                    .any(|data| data.start < span.end && span.start < data.end);
                assert!(!touches_data, "{name}: added token {id} on request data");
            }
        }
        if served == expected {
            same += 1;
        } else {
            spelled += 1;
        }
    }
    for pair in fixtures["strings"].as_array().unwrap() {
        let text = pair[0].as_str().unwrap();
        if tokenizer.encode(text).unwrap() != ids(&pair[1]) {
            failures.push(format!("string {text:?}"));
        }
        checked += 1;
    }
    assert!(
        failures.is_empty(),
        "{} mismatches: {failures:#?}",
        failures.len()
    );
    assert!(checked > 1000, "{checked}");
    // Most prompts' data spells no added token and keeps Python's ids; the
    // ones that differ are the cases written to spell one.
    assert!(same > 500 && spelled > 20, "{same} {spelled}");
}

/// Appends `token` to every string a request supplies, leaving alone the
/// members that select structure rather than supply text.
fn spell(value: &mut Json, token: &str, in_data: bool) {
    match value {
        Json::Str(text) => text.push_str(token),
        Json::Array(items) => items.iter_mut().for_each(|v| spell(v, token, in_data)),
        Json::Object(fields) => {
            for (key, value) in fields.iter_mut() {
                let structural = matches!(
                    key.as_str(),
                    "role" | "type" | "task" | "id" | "tool_call_id" | "tool_use_id" | "namespace"
                );
                // String arguments that are JSON would stop being JSON, and
                // the call would be written with other parameters.
                let json_arguments = key == "arguments"
                    && value
                        .as_str()
                        .is_some_and(|text| json::parse_python_lenient(text).is_ok());
                if (in_data || !structural) && !json_arguments {
                    let data = in_data || matches!(key.as_str(), "arguments" | "parameters");
                    spell(value, token, data);
                }
                if in_data {
                    key.push_str(token);
                }
            }
        }
        _ => {}
    }
}

/// **Request data is never a control token**: with a token appended to every
/// string of every fixture, the prompt holds exactly the tokens it held
/// without.
#[test]
fn request_data_spells_no_added_token() {
    let tokenizer = tokenizer();
    let cases = corpus("encoding_cases.json");
    let mut checked = 0;
    for token in [
        "<｜User｜>",
        "<｜Assistant｜>",
        "</think>",
        "<｜end▁of▁sentence｜>",
        "｜DSML｜",
    ] {
        let id = tokenizer.token_to_id(token).unwrap();
        assert!(tokenizer.is_added(id));
        let count = |prompt: &Prompt| {
            let ids = tokenizer.encode_prompt(prompt).unwrap();
            ids.iter().filter(|&&i| i == id).count()
        };
        for case in cases["cases"].as_array().unwrap() {
            if !case["reject"].is_null() {
                continue;
            }
            let original = input(case).unwrap();
            let mut spelled = original.clone();
            spell(&mut spelled.messages, token, false);
            if let Some(tools) = &mut spelled.tools {
                spell(tools, token, false);
            }
            let options = options(case);
            let plain = deepseek_v4_1::encode(&original, options).unwrap();
            let Ok(prompt) = deepseek_v4_1::encode(&spelled, options) else {
                continue;
            };
            let name = case["name"].as_str().unwrap();
            // A conversation with no text of its own has nothing to spell.
            if prompt.text().matches(token).count() == plain.text().matches(token).count() {
                continue;
            }
            assert_eq!(count(&prompt), count(&plain), "{name}: {token}");
            checked += 1;
        }
    }
    assert!(checked > 3000, "{checked}");
}

/// The format writes `<` or `/` before each `｜DSML｜` of its own, so request
/// data never runs into one: the overlap `Tokenizer::encode_prompt` refuses
/// (a data `｜DSML` completed by the format's `｜DSML｜`) cannot be rendered.
#[test]
fn the_formats_own_dsml_never_follows_request_data() {
    const DSML: &str = "｜DSML｜";
    let cases = corpus("encoding_cases.json");
    let mut own = 0;
    for case in cases["cases"].as_array().unwrap() {
        if !case["reject"].is_null() {
            continue;
        }
        let prompt = encode(case).unwrap();
        let text = prompt.text();
        for (at, _) in text.match_indices(DSML) {
            let in_data = prompt
                .data_spans()
                .iter()
                .any(|data| data.start < at + DSML.len() && at < data.end);
            if !in_data {
                assert!(text[..at].ends_with('<') || text[..at].ends_with('/'));
                assert!(!prompt.data_spans().iter().any(|data| data.end == at));
                own += 1;
            }
        }
    }
    assert!(own > 1000, "{own}");
}

#[test]
fn request_options_follow_the_chat_template() {
    let effort = |text: &str| json::parse(text).unwrap();
    let resolve = |thinking, text: Option<&str>| {
        EncodeOptions::from_request(thinking, text.map(effort).as_ref())
            .map(|o| (o.thinking, o.reasoning_effort.get(), o.drop_thinking))
    };
    assert_eq!(resolve(None, None), Ok((true, 75, true)));
    assert_eq!(resolve(None, Some("null")), Ok((true, 75, true)));
    assert_eq!(resolve(Some(false), None), Ok((false, 75, true)));
    for (name, budget) in [
        ("minimal", 50),
        ("low", 50),
        ("medium", 75),
        ("high", 75),
        ("xhigh", 75),
        ("max", 100),
    ] {
        assert_eq!(
            resolve(None, Some(&format!("\"{name}\""))),
            Ok((true, budget, true))
        );
    }
    // `none` turns thinking off unless the request set it.
    assert_eq!(resolve(None, Some("\"none\"")), Ok((false, 75, true)));
    assert_eq!(resolve(Some(true), Some("\"none\"")), Ok((true, 75, true)));
    for budget in [1, 42, 100] {
        assert_eq!(
            resolve(None, Some(&budget.to_string())),
            Ok((true, budget, true))
        );
    }
    for refused in [
        "0",
        "101",
        "-1",
        "1.5",
        "1.0",
        "true",
        "\"ultra\"",
        "\"\"",
        "[50]",
        "1e2",
    ] {
        assert!(
            matches!(
                resolve(None, Some(refused)),
                Err(ChatError::InvalidInput(_))
            ),
            "{refused}"
        );
    }
    assert_eq!(ReasoningEffort::budget(0), None);
    assert_eq!(ReasoningEffort::budget(101), None);
    assert_eq!(ReasoningEffort::level("medium"), None);
}

#[test]
fn declared_tools_are_named_as_the_model_calls_them() {
    let tools = json::parse(
        r#"[{"type":"function","function":{"name":"plain"}},
            {"type":"function","namespace":"files","function":{"name":"read"}},
            {"type":"function","function":{"name":"search::lookup"}}]"#,
    )
    .unwrap();
    let schemas = deepseek_v4_1::tool_schemas(&tools).unwrap();
    for name in ["plain", "files::read", "search::lookup"] {
        assert!(schemas.contains(name), "{name}");
    }
    assert!(!schemas.contains("read"));
    assert!(
        deepseek_v4_1::tool_schemas(&json::parse(r#"[{"function":{"name":"a::b::c"}}]"#).unwrap())
            .is_err()
    );
}
