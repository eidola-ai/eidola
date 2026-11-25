//! Qwen3.8 output parsing: the rule on the cases it decides, and property
//! tests over arbitrary text, splits and token ids (no panic, streaming equals
//! whole-output parsing, strict-JSON arguments), and over turns the real
//! template renders (parsing inverts the template). Qwen3.5's output is
//! parsed by the same parsers; its template's renders invert too, and its
//! output starts in the answer unless thinking was asked for.

use std::io::Read;
use std::path::PathBuf;

use eidola_engine_chat::json::{self, Json};
use eidola_engine_chat::pysem::py_strip;
use eidola_engine_chat::qwen::{ChatTemplate, RenderOptions};
use eidola_engine_chat::{
    ChatDelta, ChatError, ChatFormat, ChatInput, FinishReason, OutputConfig, OutputParser,
    PrefixedCallIds, StopCause, Tokenizer, ToolSchemas,
};
use proptest::prelude::*;

const TOOLS: &str = r#"[
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "typed", "parameters": {"type": "object", "properties": {
        "s": {"type": "string"}, "i": {"type": "integer"}, "n": {"type": "number"},
        "b": {"type": "boolean"}, "o": {"type": "object"}, "a": {"type": "array"}}}}}
]"#;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen")
}

fn tokenizer() -> Tokenizer {
    let compressed = std::fs::read(fixtures().join("tokenizer.json.gz")).unwrap();
    let mut tokenizer_json = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut tokenizer_json)
        .unwrap();
    let config = std::fs::read(fixtures().join("generation_config.json")).unwrap();
    Tokenizer::for_format(ChatFormat::Qwen3_8, &tokenizer_json, &config).unwrap()
}

fn parser(thinking: Option<bool>, tools: bool) -> OutputParser {
    parser_for(ChatFormat::Qwen3_8, thinking, tools)
}

fn parser_for(format: ChatFormat, thinking: Option<bool>, tools: bool) -> OutputParser {
    let tools = tools.then(|| ToolSchemas::from_tools(&json::parse(TOOLS).unwrap()));
    let config = match format {
        ChatFormat::Qwen3_8 => OutputConfig::qwen3_8(thinking, tools),
        ChatFormat::Qwen3_5 => OutputConfig::qwen3_5(thinking, tools),
        other => panic!("{other:?} is not a Qwen format"),
    };
    OutputParser::new(config, PrefixedCallIds("call_".into()))
}

fn append(all: &mut ChatDelta, delta: ChatDelta) {
    all.reasoning_content.push_str(&delta.reasoning_content);
    all.content.push_str(&delta.content);
    all.tool_calls.extend(delta.tool_calls);
}

fn run(
    thinking: Option<bool>,
    tools: bool,
    chunks: &[&str],
    cause: StopCause,
) -> (ChatDelta, FinishReason) {
    run_for(ChatFormat::Qwen3_8, thinking, tools, chunks, cause)
}

fn run_for(
    format: ChatFormat,
    thinking: Option<bool>,
    tools: bool,
    chunks: &[&str],
    cause: StopCause,
) -> (ChatDelta, FinishReason) {
    let mut p = parser_for(format, thinking, tools);
    let mut all = ChatDelta::default();
    for c in chunks {
        append(&mut all, p.push_text(c));
    }
    let (rest, reason) = p.finish(cause);
    append(&mut all, rest);
    (all, reason)
}

fn calls(delta: &ChatDelta) -> Vec<(&str, &str)> {
    delta
        .tool_calls
        .iter()
        .map(|c| (c.name.as_str(), c.arguments.as_str()))
        .collect()
}

#[test]
fn reasoning_answer_and_calls() {
    let (d, reason) = run(
        None,
        true,
        &[
            "Need the weather.\n</think>\n\nChecking.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
        ],
        StopCause::EndOfSequence,
    );
    assert_eq!(d.reasoning_content, "Need the weather.\n");
    assert_eq!(d.content, "\n\nChecking.\n\n");
    assert_eq!(
        calls(&d),
        [("get_weather", r#"{"city": "Oslo", "days": 2}"#)]
    );
    assert_eq!(reason, FinishReason::ToolCalls);

    // Thinking off: the prompt closed the reasoning, so all of it is answer.
    let (d, _) = run(
        Some(false),
        false,
        &["Plain </think> answer"],
        StopCause::EndOfSequence,
    );
    assert_eq!(
        (d.reasoning_content.as_str(), d.content.as_str()),
        ("", "Plain </think> answer")
    );

    // A call opened inside the reasoning ends it.
    let (d, _) = run(
        None,
        true,
        &[
            "Plan.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nX\n</parameter>\n</function>\n</tool_call>",
        ],
        StopCause::EndOfSequence,
    );
    assert_eq!(d.reasoning_content, "Plan.\n\n");
    assert_eq!(calls(&d), [("get_weather", r#"{"city": "X"}"#)]);
}

/// The answer loses only its calls: an unclosed block, text after a call, an
/// undeclared function's block and prose mentioning the tag are all content.
#[test]
fn text_that_is_not_a_call_is_content() {
    let call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nP\n</parameter>\n</function>\n</tool_call>";
    let cases: &[(&str, &str, usize)] = &[
        (
            "R</think>A <tool_call>\n<function=get_weather>\n<parameter=city>\nPa",
            "A <tool_call>\n<function=get_weather>\n<parameter=city>\nPa",
            0,
        ),
        (
            "R</think>Write <tool_call> to call.",
            "Write <tool_call> to call.",
            0,
        ),
        (
            "R</think><tool_call>\n<function=nope>\n</function>\n</tool_call>",
            "<tool_call>\n<function=nope>\n</function>\n</tool_call>",
            0,
        ),
    ];
    for &(text, content, n) in cases {
        let (d, _) = run(None, true, &[text], StopCause::MaxTokens);
        assert_eq!(
            (d.content.as_str(), d.tool_calls.len()),
            (content, n),
            "{text:?}"
        );
    }
    let text = format!("R</think>{call}\nAfter.");
    let (d, _) = run(None, true, &[&text], StopCause::EndOfSequence);
    assert_eq!((d.content.as_str(), d.tool_calls.len()), ("\nAfter.", 1));
    // Whitespace between calls is the markup's, not the answer's.
    let text = format!("R</think>{call}\n{call}");
    let (d, _) = run(None, true, &[&text], StopCause::EndOfSequence);
    assert_eq!((d.content.as_str(), d.tool_calls.len()), ("", 2));
}

/// One newline at each end of a value is the markup's; the rest is the value.
#[test]
fn value_framing() {
    let block = |v: &str| {
        format!(
            "<tool_call>\n<function=typed>\n<parameter=s>{v}</parameter>\n<parameter=i>{v}</parameter>\n</function>\n</tool_call>"
        )
    };
    for (raw, s) in [
        ("\nx\n", "x"),
        ("x", "x"),
        ("\n\nx\n\n", "\nx\n"),
        ("\n", ""),
        ("\n\n", ""),
        (" x ", " x "),
    ] {
        let (d, _) = run(Some(false), true, &[&block(raw)], StopCause::EndOfSequence);
        let args = json::parse(&d.tool_calls[0].arguments).unwrap();
        assert_eq!(args.get("s"), Some(&Json::Str(s.into())), "{raw:?}");
    }
    let (d, _) = run(
        Some(false),
        true,
        &[&block("\n7\n")],
        StopCause::EndOfSequence,
    );
    assert_eq!(d.tool_calls[0].arguments, r#"{"s": "7", "i": 7}"#);
}

/// Qwen3.5 thinks only when asked: its output is the answer unless
/// `enable_thinking` was `true`, and then it starts in the reasoning.
#[test]
fn qwen35_thinks_only_when_asked() {
    let output = ["R\n</think>\n\nThe answer"];
    for thinking in [None, Some(false)] {
        let (d, _) = run_for(
            ChatFormat::Qwen3_5,
            thinking,
            false,
            &output,
            StopCause::EndOfSequence,
        );
        assert_eq!(d.reasoning_content, "", "{thinking:?}");
        assert_eq!(d.content, output[0]);
    }
    let (d, _) = run_for(
        ChatFormat::Qwen3_5,
        Some(true),
        false,
        &output,
        StopCause::EndOfSequence,
    );
    assert_eq!(
        (d.reasoning_content.as_str(), d.content.as_str()),
        ("R\n", "\n\nThe answer")
    );
    let mut p = parser_for(ChatFormat::Qwen3_5, None, false);
    p.push_text("The answer");
    p.finish(StopCause::MaxTokens);
    assert!(!p.resumes_exactly());
}

/// A Qwen3.8 response is never continued from what it gave out.
#[test]
fn never_resumes_exactly() {
    for chunks in [&["thinking about"][..], &["R\n</think>\n\nThe answer"]] {
        let mut p = parser(None, false);
        for c in chunks {
            p.push_text(c);
        }
        p.finish(StopCause::MaxTokens);
        assert!(!p.resumes_exactly());
        assert!(!p.resumes_exactly_as_tokens(&[]));
    }
}

const FRAGMENTS: &[&str] = &[
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "\n<function=get_weather>\n",
    "<function=typed>",
    "<function=nope>",
    "<function=",
    "</function>",
    "<parameter=city>\n",
    "<parameter=days>",
    "<parameter=o>\n",
    "\n</parameter>\n",
    "</parameter>",
    ">",
    "<",
    "\n",
    "\n\n",
    " ",
    "Paris",
    "3",
    "4.0",
    "{\"a\": [1, 1e400]}",
    "null",
    "中文",
    "😀",
    "</th",
    "ink>",
    "<tool_",
    "call>",
    "</tool",
    "<|im_end|>",
];

fn split(text: &str, cuts: &[usize]) -> Vec<String> {
    let mut cuts: Vec<usize> = cuts
        .iter()
        .map(|c| c % (text.len() + 1))
        .filter(|&c| text.is_char_boundary(c))
        .collect();
    cuts.sort_unstable();
    let mut out = Vec::new();
    let mut last = 0;
    for cut in cuts.into_iter().chain([text.len()]) {
        if cut >= last {
            out.push(text[last..cut].to_string());
            last = cut;
        }
    }
    out
}

proptest! {
    /// Format fragments, so markup is unclosed, interleaved with reasoning and
    /// split at every kind of boundary: any split parses as the whole text does,
    /// and every `arguments` is a strict JSON object.
    #[test]
    fn any_split_of_markup_is_the_whole_parse(
        pieces in proptest::collection::vec(0..FRAGMENTS.len(), 0..48),
        cuts in proptest::collection::vec(any::<usize>(), 0..24),
        thinking: Option<bool>,
        tools: bool,
        cause in prop_oneof![Just(StopCause::EndOfSequence), Just(StopCause::MaxTokens)],
    ) {
        let text: String = pieces.iter().map(|&i| FRAGMENTS[i]).collect();
        let chunks = split(&text, &cuts);
        let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let streamed = run(thinking, tools, &chunks, cause);
        prop_assert_eq!(&streamed, &run(thinking, tools, &[&text], cause));
        for call in &streamed.0.tool_calls {
            prop_assert!(matches!(json::parse_strict(&call.arguments), Ok(Json::Object(_))), "{}", call.arguments);
        }
        // Nothing is lost but calls: without tools, reasoning and content are
        // the text less only the tags the splitter consumed.
        if !tools {
            let given = format!("{}{}", streamed.0.reasoning_content, streamed.0.content);
            prop_assert!(given.len() <= text.len());
        }
    }

    #[test]
    fn arbitrary_text_never_panics(
        text in "\\PC{0,300}",
        cuts in proptest::collection::vec(any::<usize>(), 0..10),
        thinking: Option<bool>,
    ) {
        let chunks = split(&text, &cuts);
        let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
        prop_assert_eq!(
            run(thinking, true, &chunks, StopCause::MaxTokens),
            run(thinking, true, &[&text], StopCause::MaxTokens)
        );
    }
}

/// Arbitrary ids of the real tokenizer, past its vocabulary too: decoded one at
/// a time they parse as their whole decoded text does; an undefined id is
/// refused and changes nothing.
#[test]
fn any_token_stream_is_safe_and_split_invariant() {
    let tokenizer = tokenizer();
    let vocab = tokenizer.vocab_size() as u32;
    // Ids around the markup tokens, and anywhere.
    let near: Vec<u32> = (248_040..vocab + 3).collect();
    let mut runner = proptest::test_runner::TestRunner::new(ProptestConfig::with_cases(256));
    let strategy = (
        proptest::collection::vec(
            prop_oneof![proptest::sample::select(near), 0u32..vocab + 1000],
            0..96,
        ),
        any::<Option<bool>>(),
        any::<bool>(),
    );
    runner
        .run(&strategy, |(ids, thinking, tools)| {
            let mut one_by_one = parser(thinking, tools);
            let mut streamed = ChatDelta::default();
            for &id in &ids {
                match one_by_one.push_token(&tokenizer, id) {
                    Ok(delta) => append(&mut streamed, delta),
                    Err(e) => {
                        prop_assert!(id >= vocab);
                        prop_assert_eq!(e, ChatError::UnknownToken(id));
                    }
                }
            }
            let (rest, reason) = one_by_one.finish(StopCause::EndOfSequence);
            append(&mut streamed, rest);
            let known: Vec<u32> = ids.iter().copied().filter(|&id| id < vocab).collect();
            let mut whole = parser(thinking, tools);
            let mut at_once = whole.push_text(&tokenizer.decode(&known, true).unwrap());
            let (rest, whole_reason) = whole.finish(StopCause::EndOfSequence);
            append(&mut at_once, rest);
            prop_assert_eq!(&streamed, &at_once);
            prop_assert_eq!(reason, whole_reason);
            for call in &streamed.tool_calls {
                prop_assert!(matches!(
                    json::parse_strict(&call.arguments),
                    Ok(Json::Object(_))
                ));
            }
            Ok(())
        })
        .unwrap();
}

/// Text with no markup in it, its ends possibly whitespace.
fn prose() -> impl Strategy<Value = String> {
    "[^<]{0,24}"
}

fn string_value() -> impl Strategy<Value = Json> {
    prop_oneof![
        "\\PC{0,16}".prop_map(Json::Str),
        proptest::sample::select(vec![
            "",
            "\n",
            "\nlead",
            "trail\n",
            "a\nb",
            "null",
            "42",
            "{\"a\": 1}"
        ])
        .prop_map(|s| Json::Str(s.into())),
    ]
    .prop_filter("format limits", |v| {
        !["</parameter>", "</function>", "</tool_call>"]
            .iter()
            .any(|t| v.as_str().unwrap().contains(t))
    })
}

fn json_value() -> impl Strategy<Value = Json> {
    let leaf = prop_oneof![
        Just(Json::Null),
        any::<bool>().prop_map(Json::Bool),
        any::<i64>().prop_map(|i| json::parse(&i.to_string()).unwrap()),
        (-1e6f64..1e6).prop_map(Json::Float),
        "[a-z]{0,6}".prop_map(Json::Str),
    ];
    leaf.prop_recursive(3, 16, 4, |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..4).prop_map(Json::Array),
            proptest::collection::vec(("[a-z]{1,4}", inner), 0..4).prop_map(|members| {
                let mut seen = std::collections::HashSet::new();
                Json::Object(
                    members
                        .into_iter()
                        .filter(|(k, _)| seen.insert(k.clone()))
                        .collect(),
                )
            }),
        ]
    })
}

/// Arguments that conform to the `typed` tool's schema, in any order.
fn typed_arguments() -> impl Strategy<Value = Json> {
    let member = prop_oneof![
        string_value().prop_map(|v| ("s".to_string(), v)),
        any::<i64>().prop_map(|i| ("i".to_string(), json::parse(&i.to_string()).unwrap())),
        prop_oneof![
            (-1e9f64..1e9),
            any::<f64>().prop_filter("finite", |f| f.is_finite())
        ]
        .prop_map(|f| ("n".to_string(), Json::Float(f))),
        any::<bool>().prop_map(|b| ("b".to_string(), Json::Bool(b))),
        proptest::collection::vec(("[a-z]{1,3}", json_value()), 0..3).prop_map(|m| {
            let mut seen = std::collections::HashSet::new();
            (
                "o".to_string(),
                Json::Object(
                    m.into_iter()
                        .filter(|(k, _)| seen.insert(k.clone()))
                        .collect(),
                ),
            )
        }),
        proptest::collection::vec(json_value(), 0..3)
            .prop_map(|a| ("a".to_string(), Json::Array(a))),
    ];
    proptest::collection::vec(member, 0..6).prop_map(|members| {
        let mut seen = std::collections::HashSet::new();
        Json::Object(
            members
                .into_iter()
                .filter(|(k, _)| seen.insert(k.clone()))
                .collect(),
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// **Parsing inverts the template**: an assistant turn the template
    /// renders, cut after the generation prompt, parses back into its
    /// reasoning and answer (as the template trims them) and its calls'
    /// arguments exactly.
    #[test]
    fn parsing_inverts_the_template(
        reasoning in prose(),
        content in prose(),
        arguments in proptest::collection::vec(typed_arguments(), 0..3),
        thinking: bool,
    ) {
        inverts(ChatFormat::Qwen3_8, &reasoning, &content, &arguments, thinking)?;
    }

    /// The same for Qwen3.5's template, which thinks only when asked and
    /// writes a boolean or a number argument with Python's `str()`.
    #[test]
    fn parsing_inverts_the_qwen35_template(
        reasoning in prose(),
        content in prose(),
        arguments in proptest::collection::vec(typed_arguments(), 0..3),
        thinking: bool,
    ) {
        inverts(ChatFormat::Qwen3_5, &reasoning, &content, &arguments, thinking)?;
    }
}

/// [`parsing_inverts_the_template`]'s property for the template of `format`.
fn inverts(
    format: ChatFormat,
    reasoning: &str,
    content: &str,
    arguments: &[Json],
    thinking: bool,
) -> Result<(), TestCaseError> {
    {
        let dir = match format {
            ChatFormat::Qwen3_8 => "qwen",
            _ => "qwen35",
        };
        let template = ChatTemplate::from_model_dir(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures")
                .join(dir),
        )
        .unwrap();
        // The switch that turns thinking `on` or off in this format's template.
        let switch = match format {
            ChatFormat::Qwen3_8 => (!thinking).then_some(false),
            _ => thinking.then_some(true),
        };
        let calls: Vec<serde_json::Value> = arguments
            .iter()
            .map(|a| serde_json::json!({"type": "function", "function": {"name": "typed", "arguments": json::dumps_default(a)}}))
            .collect();
        let mut assistant = serde_json::json!({"role": "assistant", "content": content});
        if thinking {
            assistant["reasoning_content"] = reasoning.into();
        }
        if !calls.is_empty() {
            assistant["tool_calls"] = calls.into();
        }
        let question = serde_json::json!([{"role": "user", "content": "q"}]);
        let options = RenderOptions {
            enable_thinking: switch,
            ..RenderOptions::default()
        };
        let render = |messages: &serde_json::Value, add_generation_prompt| {
            let input = ChatInput::from_json(&messages.to_string(), Some(TOOLS)).unwrap();
            template
                .render(
                    &input,
                    &RenderOptions {
                        add_generation_prompt,
                        ..options.clone()
                    },
                )
                .unwrap()
        };
        let prompt = render(&question, true);
        let mut turn = question.clone();
        turn.as_array_mut().unwrap().push(assistant);
        let full = render(&turn, false);
        prop_assert!(full.starts_with(&prompt));
        let output = full[prompt.len()..].strip_suffix("<|im_end|>\n").unwrap();

        let (d, _) = run_for(format, switch, true, &[output], StopCause::EndOfSequence);
        if thinking {
            prop_assert_eq!(py_strip(&d.reasoning_content), py_strip(reasoning));
        } else {
            prop_assert_eq!(d.reasoning_content.as_str(), "");
        }
        prop_assert_eq!(py_strip(&d.content), py_strip(content));
        let parsed: Vec<String> = d.tool_calls.iter().map(|c| c.arguments.clone()).collect();
        let sent: Vec<String> = arguments.iter().map(json::dumps_default).collect();
        prop_assert_eq!(parsed, sent);
    }
    Ok(())
}
