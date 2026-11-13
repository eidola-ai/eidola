//! The DeepSeek-V4.1 output parsers: agreement with the reference's
//! `parse_message_from_completion_text` on the output it accepts
//! (`fixtures/deepseek_v4_1/output_cases.json`, from
//! `dev/gen_deepseek_fixtures.py`), and property tests that no input panics,
//! that streaming equals parsing the whole output for every split, that an
//! answer loses only its calls, and that parsing inverts the encoder.

use std::io::Read;
use std::path::PathBuf;
use std::sync::LazyLock;

use eidola_engine_chat::deepseek_v4_1::output::{
    DsmlParser, ReasoningSplitter, parse_complete, split_reasoning,
};
use eidola_engine_chat::deepseek_v4_1::{self, EncodeOptions};
use eidola_engine_chat::json::{self, Json};
use eidola_engine_chat::tool_call::ParsedToolCalls;
use eidola_engine_chat::{
    ChatDelta, ChatError, ChatFormat, ChatInput, FinishReason, OutputConfig, OutputParser,
    PrefixedCallIds, StopCause, Tokenizer, ToolSchemas,
};
use proptest::prelude::*;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/deepseek_v4_1")
}

static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    let compressed = std::fs::read(fixtures().join("tokenizer.json.gz")).unwrap();
    let mut tokenizer_json = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut tokenizer_json)
        .unwrap();
    let config = std::fs::read(fixtures().join("config.json")).unwrap();
    Tokenizer::for_format(ChatFormat::DeepSeekV4_1, &tokenizer_json, &config).unwrap()
});

fn declare<'a>(names: impl IntoIterator<Item = &'a str>) -> ToolSchemas {
    let tools = names
        .into_iter()
        .map(|name| {
            Json::Object(vec![(
                "function".into(),
                Json::Object(vec![("name".into(), Json::Str(name.into()))]),
            )])
        })
        .collect();
    deepseek_v4_1::tool_schemas(&Json::Array(tools)).unwrap()
}

fn parser(thinking: bool, tools: Option<ToolSchemas>) -> OutputParser {
    OutputParser::new(
        OutputConfig::deepseek_v4_1(thinking, tools),
        PrefixedCallIds("call_".into()),
    )
}

/// Byte offsets into `text` for a list of cut points, on character boundaries.
fn chunks<'a>(text: &'a str, cuts: &[usize]) -> Vec<&'a str> {
    let mut cuts: Vec<usize> = cuts
        .iter()
        .map(|&cut| text.floor_char_boundary(cut.min(text.len())))
        .collect();
    cuts.push(text.len());
    cuts.sort();
    let mut from = 0;
    cuts.into_iter()
        .map(|cut| {
            let chunk = &text[from..cut];
            from = cut;
            chunk
        })
        .collect()
}

fn every(text: &str, size: usize) -> Vec<usize> {
    (0..text.len()).step_by(size).collect()
}

/// The whole pipeline over `text` cut at `cuts`: everything it gave out, and
/// the finish reason.
fn run(
    text: &str,
    cuts: &[usize],
    thinking: bool,
    tools: Option<ToolSchemas>,
) -> (ChatDelta, FinishReason) {
    let mut parser = parser(thinking, tools);
    let mut all = ChatDelta::default();
    let mut add = |delta: ChatDelta| {
        all.reasoning_content.push_str(&delta.reasoning_content);
        all.content.push_str(&delta.content);
        all.tool_calls.extend(delta.tool_calls);
    };
    for chunk in chunks(text, cuts) {
        add(parser.push_text(chunk));
    }
    let (rest, reason) = parser.finish(StopCause::EndOfSequence);
    add(rest);
    assert!(!parser.resumes_exactly());
    (all, reason)
}

fn stream_calls(text: &str, cuts: &[usize], schemas: &ToolSchemas) -> ParsedToolCalls {
    let mut parser = DsmlParser::new(schemas.clone(), PrefixedCallIds("c".into()));
    let mut out = ParsedToolCalls::default();
    let mut index = 0;
    let mut add = |delta: eidola_engine_chat::tool_call::ToolCallDelta| {
        out.content.push_str(&delta.content);
        for call in delta.calls {
            assert_eq!(call.index, index);
            assert_eq!(call.id, format!("c{index}"));
            index += 1;
            out.calls.push((call.name, call.arguments));
        }
    };
    for chunk in chunks(text, cuts) {
        add(parser.push(chunk));
    }
    add(parser.finish());
    out
}

fn stream_reasoning(text: &str, cuts: &[usize], thinking: bool) -> (String, String) {
    let mut splitter = ReasoningSplitter::new(thinking);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for chunk in chunks(text, cuts) {
        let delta = splitter.push(chunk);
        reasoning.push_str(&delta.reasoning);
        content.push_str(&delta.content);
    }
    let delta = splitter.finish();
    reasoning.push_str(&delta.reasoning);
    content.push_str(&delta.content);
    (reasoning, content)
}

#[test]
fn reads_what_the_reference_reads() {
    let text = std::fs::read_to_string(fixtures().join("output_cases.json")).unwrap();
    let corpus: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut failures = Vec::new();
    let (mut accepted, mut with_calls, mut refused) = (0, 0, 0);
    for case in corpus["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let thinking = case["thinking"].as_bool().unwrap();
        let declared = || {
            // The tools the output calls, and one more so that tool calls are
            // parsed in an output that calls none.
            let names = case["declared"].as_array().unwrap();
            declare(
                names
                    .iter()
                    .map(|n| n.as_str().unwrap())
                    .chain(["never_called"]),
            )
        };
        let (whole, reason) = run(text, &[], thinking, Some(declared()));
        for size in [1, 3, 7] {
            let (streamed, streamed_reason) =
                run(text, &every(text, size), thinking, Some(declared()));
            if (&streamed, streamed_reason) != (&whole, reason) {
                failures.push(format!("{name}: {size}-byte chunks parse differently"));
            }
        }
        for call in &whole.tool_calls {
            if !matches!(json::parse_strict(&call.arguments), Ok(Json::Object(_))) {
                failures.push(format!("{name}: arguments {:?}", call.arguments));
            }
        }
        let Some(reference) = case.get("reference") else {
            refused += 1;
            continue;
        };
        accepted += 1;
        let calls = reference["tool_calls"].as_array().unwrap();
        with_calls += usize::from(!calls.is_empty());
        let strict = reference["strict_json"].as_bool().unwrap();
        let same_calls = whole.tool_calls.len() == calls.len()
            && whole.tool_calls.iter().zip(calls).all(|(ours, theirs)| {
                ours.name == theirs["name"].as_str().unwrap()
                    && (!strict || ours.arguments == theirs["arguments"].as_str().unwrap())
            });
        let expected_reason = if calls.is_empty() {
            FinishReason::Stop
        } else {
            FinishReason::ToolCalls
        };
        if whole.reasoning_content != reference["reasoning_content"].as_str().unwrap()
            || whole.content != reference["content"].as_str().unwrap()
            || !same_calls
            || reason != expected_reason
        {
            failures.push(format!("{name}: {whole:?}\n  reference: {reference}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        accepted >= 80 && with_calls >= 40 && refused >= 15,
        "{accepted} {with_calls} {refused}"
    );
}

const BLOCK: &str = "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"lookup\">\n<｜DSML｜ parameter name=\"query\" string=\"true\">value</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">2</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";

/// **The answer loses only its calls**: what the reference refuses is given
/// out as content.
#[test]
fn output_the_reference_refuses_is_kept() {
    let tools = || Some(declare(["lookup"]));
    let parse = |text: &str| {
        let (delta, reason) = run(text, &[], true, tools());
        let calls: Vec<(String, String)> = delta
            .tool_calls
            .into_iter()
            .map(|call| (call.name, call.arguments))
            .collect();
        (delta.reasoning_content, delta.content, calls, reason)
    };
    let lookup = || {
        vec![(
            "lookup".to_string(),
            r#"{"query": "value", "limit": 2}"#.to_string(),
        )]
    };
    let none = Vec::new;
    let (r, c) = (|| "r".to_string(), |text: &str| text.to_string());

    // Reasoning never closed: all reasoning.
    assert_eq!(
        parse("thoughts"),
        ("thoughts".into(), c(""), none(), FinishReason::Stop)
    );
    // Calls before `</think>`: the reasoning ends at the block.
    assert_eq!(
        parse(&format!("plan{BLOCK}")),
        ("plan".into(), c(""), lookup(), FinishReason::ToolCalls)
    );
    // Text after the calls, and a second block.
    assert_eq!(
        parse(&format!("r</think>a{BLOCK}tail")),
        (r(), c("atail"), lookup(), FinishReason::ToolCalls)
    );
    let twice = [lookup(), lookup()].concat();
    assert_eq!(
        parse(&format!("r</think>a{BLOCK}{BLOCK}")),
        (r(), c("a"), twice, FinishReason::ToolCalls)
    );
    // No blank line before the block, or more than one.
    assert_eq!(
        parse(&format!("r</think>a{}", &BLOCK[2..])),
        (r(), c("a"), lookup(), FinishReason::ToolCalls)
    );
    assert_eq!(
        parse(&format!("r</think>a\n{BLOCK}")),
        (r(), c("a\n"), lookup(), FinishReason::ToolCalls)
    );
    // An unclosed block, an undeclared tool, a malformed parameter, an empty
    // block: content, blank line and all.
    let unclosed = &BLOCK[..BLOCK.len() - 4];
    assert_eq!(
        parse(&format!("r</think>a{unclosed}")),
        (r(), format!("a{unclosed}"), none(), FinishReason::Stop)
    );
    for broken in [
        BLOCK.replace("lookup", "other"),
        BLOCK.replace("string=\"false\"", "string=\"maybe\""),
        BLOCK.replace("name=\"limit\"", "name=\"query\""),
        BLOCK.replace("</｜DSML｜ invoke>\n", ""),
    ] {
        assert_eq!(
            parse(&format!("r</think>a{broken}")),
            (r(), format!("a{broken}"), none(), FinishReason::Stop),
            "{broken}"
        );
    }
    // A block of no calls is dropped, as the reference drops it.
    assert_eq!(
        parse("r</think>a\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>"),
        (r(), c("a"), none(), FinishReason::Stop)
    );
    // A value that is not strict JSON is the string it spells.
    for (value, expected) in [
        ("{not json", "\"{not json\""),
        ("1e400", "\"1e400\""),
        ("NaN", "\"NaN\""),
    ] {
        let (_, _, calls, _) = parse(&format!(
            "r</think>{}",
            BLOCK.replace(">2<", &format!(">{value}<"))
        ));
        assert_eq!(
            calls[0].1,
            format!(r#"{{"query": "value", "limit": {expected}}}"#)
        );
    }
    // No tools offered: the markup stays in the answer.
    let (delta, reason) = run(&format!("r</think>a{BLOCK}"), &[], true, None);
    assert_eq!(delta.content, format!("a{BLOCK}"));
    assert_eq!(reason, FinishReason::Stop);
    // Chat mode has no reasoning to split.
    let (delta, _) = run("a</think>b", &[], false, tools());
    assert_eq!(
        (delta.reasoning_content.as_str(), delta.content.as_str()),
        ("", "a</think>b")
    );
    // Cut off by the length limit, completed calls stay.
    let mut cut = parser(true, tools());
    cut.push_text(&format!("r</think>a{BLOCK}"));
    assert_eq!(cut.finish(StopCause::MaxTokens).1, FinishReason::Length);
}

/// Pieces of the format, of malformed variants of it, and of ordinary text.
const FRAGMENTS: &[&str] = &[
    "</think>",
    "<think>",
    "\n\n",
    "\n",
    " ",
    "text",
    "é😀",
    "<",
    "</",
    "｜",
    "DSML",
    "｜DSML｜",
    "<｜DSML｜",
    " calls>",
    "<｜DSML｜ calls>",
    "</｜DSML｜ calls>",
    "<｜DSML｜ invoke",
    " name=\"f\">",
    " name=\"g\">",
    " name=\"nope\">",
    "</｜DSML｜ invoke>",
    "<｜DSML｜ parameter",
    " name=\"a\" string=\"true\">",
    " name=\"b\" string=\"false\">",
    " name=\"a",
    "\" string=\"",
    "true\">",
    "</｜DSML｜ parameter>",
    "1",
    "1e400",
    "{\"k\": [1, \"s\"]}",
    "{\"k\": [1e400]}",
    "\"s\"",
    BLOCK,
    "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n</｜DSML｜ invoke>\n<｜DSML｜ invoke name=\"g\">\n<｜DSML｜ parameter name=\"b\" string=\"false\">[1]</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
];

fn schemas() -> ToolSchemas {
    declare(["f", "g", "lookup"])
}

fn fragments() -> impl Strategy<Value = String> {
    proptest::collection::vec(0..FRAGMENTS.len(), 0..40)
        .prop_map(|pieces| pieces.iter().map(|&i| FRAGMENTS[i]).collect())
}

fn cuts() -> impl Strategy<Value = Vec<usize>> {
    proptest::collection::vec(0usize..600, 0..24)
}

/// Text that holds none of the format's delimiters.
fn plain() -> impl Strategy<Value = String> {
    proptest::collection::vec(
        proptest::sample::select(vec![
            "a",
            "word ",
            "\n",
            "\n\n",
            " ",
            "é",
            "😀",
            "你好",
            "<",
            ">",
            "\"",
            "\\",
            "{",
            "}",
            "name=",
            "string=",
            "true",
            "DSML",
            "calls",
            "</",
            "&amp;",
            "\t",
            "1",
            "[",
            "]",
            ":",
            ",",
            "<｜",
            "｜>",
            "e\u{301}",
            "مرحبا",
        ]),
        0..8,
    )
    .prop_map(|pieces| pieces.concat())
    .prop_filter("holds a delimiter", |text| {
        !["</think>", "｜DSML｜", "\" string=\""]
            .iter()
            .any(|delimiter| text.contains(delimiter))
    })
}

fn argument() -> impl Strategy<Value = Json> {
    let leaf = prop_oneof![
        plain().prop_map(Json::Str),
        any::<i64>().prop_map(|n| json::parse(&n.to_string()).unwrap()),
        any::<bool>().prop_map(Json::Bool),
        Just(Json::Null),
        (-1.0e12f64..1.0e12).prop_map(Json::Float),
        Just(json::parse("123456789012345678901234567890").unwrap()),
    ];
    leaf.prop_recursive(3, 12, 3, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..3).prop_map(Json::Array),
            proptest::collection::vec((plain(), inner), 0..3).prop_map(|fields| {
                let mut object: Vec<(String, Json)> = Vec::new();
                for (key, value) in fields {
                    if !object.iter().any(|(k, _)| *k == key) {
                        object.push((key, value));
                    }
                }
                Json::Object(object)
            }),
        ]
    })
}

fn call() -> impl Strategy<Value = (String, Vec<(String, Json)>)> {
    (
        proptest::sample::select(vec!["f", "g", "lookup", "files::read", "名前"]),
        proptest::collection::vec((plain(), argument()), 0..4),
    )
        .prop_map(|(name, fields)| {
            let mut arguments: Vec<(String, Json)> = Vec::new();
            for (key, value) in fields {
                if !arguments.iter().any(|(k, _)| *k == key) {
                    arguments.push((key, value));
                }
            }
            (name.to_string(), arguments)
        })
}

proptest! {
    #[test]
    fn reasoning_streams_as_it_splits(text in fragments(), cuts in cuts(), thinking: bool) {
        let whole = split_reasoning(&text, thinking);
        prop_assert_eq!(stream_reasoning(&text, &cuts, thinking), (whole.reasoning.clone(), whole.content.clone()));
        // Nothing but one `</think>` is lost.
        let rejoined = [whole.reasoning.as_str(), whole.content.as_str()];
        prop_assert!(text == rejoined.concat() || text == rejoined.join("</think>"));
    }

    #[test]
    fn tool_calls_stream_as_they_parse(text in fragments(), cuts in cuts()) {
        let whole = parse_complete(&text, &schemas());
        prop_assert_eq!(&stream_calls(&text, &cuts, &schemas()), &whole);
        for (name, arguments) in &whole.calls {
            prop_assert!(schemas().contains(name));
            prop_assert!(matches!(json::parse_strict(arguments), Ok(Json::Object(_))), "{}", arguments);
        }
        // An answer without a closed block is the text itself.
        if !text.contains("</｜DSML｜ calls>") {
            prop_assert!(whole.calls.is_empty());
            prop_assert_eq!(&whole.content, &text);
        }
        // With no tool declared nothing is a call.
        prop_assert!(parse_complete(&text, &declare([])).calls.is_empty());
    }

    #[test]
    fn the_pipeline_streams_as_it_parses(text in fragments(), cuts in cuts(), thinking: bool, tools: bool) {
        let tools = || tools.then(schemas);
        let (streamed, reason) = run(&text, &cuts, thinking, tools());
        let split = split_reasoning(&text, thinking);
        let calls = match tools() {
            Some(schemas) => parse_complete(&split.content, &schemas),
            None => ParsedToolCalls { content: split.content.clone(), calls: Vec::new() },
        };
        prop_assert_eq!(&streamed.reasoning_content, &split.reasoning);
        prop_assert_eq!(&streamed.content, &calls.content);
        let given: Vec<(String, String)> = streamed.tool_calls.iter().map(|c| (c.name.clone(), c.arguments.clone())).collect();
        prop_assert_eq!(&given, &calls.calls);
        prop_assert_eq!(reason == FinishReason::ToolCalls, !calls.calls.is_empty());
    }

    #[test]
    fn arbitrary_text_never_panics(text in any::<String>(), cuts in cuts(), thinking: bool) {
        let (streamed, _) = run(&text, &cuts, thinking, Some(schemas()));
        let (whole, _) = run(&text, &[], thinking, Some(schemas()));
        prop_assert_eq!(streamed, whole);
    }

    /// Parsing inverts the encoder: what an assistant message renders to is
    /// read back as its reasoning, content and calls.
    #[test]
    fn parsing_inverts_the_encoder(
        reasoning in plain(),
        content in plain(),
        calls in proptest::collection::vec(call(), 0..4),
        thinking: bool,
        cuts in cuts(),
    ) {
        let tool_calls: Vec<Json> = calls.iter().map(|(name, arguments)| {
            Json::Object(vec![("function".into(), Json::Object(vec![
                ("name".into(), Json::Str(name.clone())),
                ("arguments".into(), Json::Object(arguments.clone())),
            ]))])
        }).collect();
        let message = |role: &str, fields: Vec<(&str, Json)>| {
            let mut object = vec![("role".to_string(), Json::Str(role.into()))];
            object.extend(fields.into_iter().map(|(k, v)| (k.to_string(), v)));
            Json::Object(object)
        };
        let input = ChatInput {
            messages: Json::Array(vec![
                message("user", vec![("content", Json::Str("q".into()))]),
                message("assistant", vec![
                    ("reasoning_content", Json::Str(reasoning.clone())),
                    ("content", Json::Str(content.clone())),
                    ("tool_calls", Json::Array(tool_calls)),
                    ("wo_eos", Json::Bool(true)),
                ]),
            ]),
            tools: None,
        };
        let prompt = deepseek_v4_1::encode(&input, EncodeOptions::new(thinking)).unwrap();
        let header = if thinking { "<｜Assistant｜><think>" } else { "<｜Assistant｜></think>" };
        let output = prompt.text().split_once(header).unwrap().1;
        let declared = declare(calls.iter().map(|(name, _)| name.as_str()));
        let (parsed, _) = run(output, &cuts, thinking, Some(declared));
        prop_assert_eq!(&parsed.reasoning_content, if thinking { &reasoning } else { "" });
        prop_assert_eq!(&parsed.content, &content);
        prop_assert_eq!(parsed.tool_calls.len(), calls.len());
        for (parsed, (name, arguments)) in parsed.tool_calls.iter().zip(&calls) {
            prop_assert_eq!(&parsed.name, name);
            // Through `dumps` both sides, which is how a float compares.
            let read = json::parse_strict(&parsed.arguments).unwrap();
            prop_assert_eq!(json::dumps_default(&read), json::dumps_default(&Json::Object(arguments.clone())));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Token by token through the real tokenizer: an id it does not define is
    /// refused and changes nothing, and the rest parse as their text does.
    #[test]
    fn arbitrary_tokens_never_panic(
        ids in proptest::collection::vec(prop_oneof![0u32..129_280, 128_790u32..128_850, 129_270u32..129_400], 0..60),
        thinking: bool,
    ) {
        let tokenizer = &*TOKENIZER;
        let mut by_token = parser(thinking, Some(schemas()));
        let mut all = ChatDelta::default();
        let mut defined = Vec::new();
        for &id in &ids {
            match by_token.push_token(tokenizer, id) {
                Ok(delta) => {
                    defined.push(id);
                    all.reasoning_content.push_str(&delta.reasoning_content);
                    all.content.push_str(&delta.content);
                    all.tool_calls.extend(delta.tool_calls);
                }
                Err(error) => {
                    prop_assert!(id as usize >= tokenizer.vocab_size());
                    prop_assert_eq!(error, ChatError::UnknownToken(id));
                }
            }
        }
        let (rest, reason) = by_token.finish(StopCause::EndOfSequence);
        all.reasoning_content.push_str(&rest.reasoning_content);
        all.content.push_str(&rest.content);
        all.tool_calls.extend(rest.tool_calls);
        let text = tokenizer.decode(&defined, true).unwrap();
        prop_assert_eq!((all, reason), run(&text, &[], thinking, Some(schemas())));
    }
}
