//! A continued message resumes exactly where its generation stopped.
//!
//! `template_fixtures.rs` holds the continued render byte-exact to
//! `transformers`. This holds it to what continuation is for: the continued
//! prompt is the original request's prompt followed by what the model
//! generated, as text and as tokens, so the engine's prefix cache can serve it
//! whole and the model reads exactly the sequence it would have kept reading;
//! and the output of the continuation, parsed from where the prompt ended,
//! joins the partial output into the same reasoning and answer an
//! uninterrupted generation would have produced.

use std::io::Read;
use std::path::PathBuf;

use eidola_engine_chat::json::Json;

use eidola_engine_chat::{
    ChatInput, ChatTemplate, MimoTokenizer, OutputConfig, OutputParser, PrefixedCallIds,
    RenderOptions, StopCause,
};

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn template() -> ChatTemplate {
    ChatTemplate::from_model_dir(&fixtures()).expect("pinned template")
}

fn tokenizer() -> MimoTokenizer {
    let compressed = std::fs::read(fixtures().join("tokenizer.json.gz")).unwrap();
    let mut tokenizer_json = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut tokenizer_json)
        .unwrap();
    let generation_config = std::fs::read(fixtures().join("generation_config.json")).unwrap();
    MimoTokenizer::from_bytes(&tokenizer_json, &generation_config).expect("pinned tokenizer")
}

/// Parses model output: from the start of a new assistant turn (`None`), or
/// from inside the field a continued prompt ended in.
fn parse(output: &str, continuing: Option<&'static str>) -> (String, String) {
    let config = OutputConfig::new(None, None).continuing_in(continuing);
    let mut parser = OutputParser::new(config, PrefixedCallIds("call_".into()));
    let mut delta = parser.push_text(output);
    let (tail, _) = parser.finish(StopCause::MaxTokens);
    delta.reasoning_content.push_str(&tail.reasoning_content);
    delta.content.push_str(&tail.content);
    (delta.reasoning_content, delta.content)
}

/// The prompt of a request continuing `partial` (reasoning and answer so far),
/// and the field it continues: the answer once there is any, the reasoning
/// before — the rule a client follows.
fn continued_prompt(
    template: &ChatTemplate,
    question: &str,
    reasoning: &str,
    content: &str,
) -> (String, &'static str) {
    let field = if content.is_empty() {
        "reasoning_content"
    } else {
        "content"
    };
    let messages = serde_json::json!([
        {"role": "user", "content": question},
        {"role": "assistant", "reasoning_content": reasoning, "content": content},
    ]);
    let input = ChatInput::from_json(&messages.to_string(), None).unwrap();
    let prompt = template
        .render(
            &input,
            RenderOptions {
                add_generation_prompt: false,
                enable_thinking: None,
                continue_final_message: Some(field),
            },
        )
        .unwrap();
    (prompt, field)
}

fn original_prompt(template: &ChatTemplate, question: &str) -> String {
    let messages = serde_json::json!([{"role": "user", "content": question}]);
    let input = ChatInput::from_json(&messages.to_string(), None).unwrap();
    template
        .render(
            &input,
            RenderOptions {
                add_generation_prompt: true,
                ..RenderOptions::default()
            },
        )
        .unwrap()
}

/// Cut a generation after every one of its tokens and continue it from there:
/// the continued prompt is a prefix of the uninterrupted sequence, in text and
/// in tokens, and the partial output joined to the continuation's output is the
/// uninterrupted output.
///
/// The one cut that does not continue the cut text verbatim is right after
/// `</think>` with no answer yet: the partial reads as reasoning only, so the
/// continuation reopens the reasoning and the model writes `</think>` again —
/// the continued prompt is then one token shorter than the cut, still a prefix
/// of the same sequence, and the joined output is still the same.
#[test]
fn a_generation_cut_after_any_token_continues_into_the_same_output() {
    let template = template();
    let tokenizer = tokenizer();
    let question = "Explain the tides.";
    let original = original_prompt(&template, question);
    let original_ids = tokenizer.encode(&original).unwrap();
    let generations = [
        "<think>The moon's gravity pulls the oceans; the sun adds a smaller pull.\n\nSo: two bulges.</think>The tides are driven mainly by the Moon. 潮汐主要由月球引起。",
        "<think>Short.</think>\n\nAnswer with a paragraph break.\n\n- and a list",
        "<think></think>No reasoning at all, straight to the answer.",
        "<think>Only reasoning, and the ceiling arrives before any answer",
    ];
    let mut cuts = 0;
    for generation in generations {
        let whole = format!("{original}{generation}");
        let whole_ids = tokenizer.encode(&whole).unwrap();
        assert!(
            whole_ids.starts_with(&original_ids),
            "the original prompt's tokens are a prefix of the whole sequence"
        );
        let generated = &whole_ids[original_ids.len()..];
        let (reasoning, content) = parse(generation, None);
        for k in 1..generated.len() {
            let cut_text = tokenizer.decode(&generated[..k], false).unwrap();
            if !generation.starts_with(&cut_text) {
                // A cut inside a multi-byte character; the model's output is
                // whole characters.
                continue;
            }
            let (r, c) = parse(&cut_text, None);
            let (prompt, field) = continued_prompt(&template, question, &r, &c);
            assert!(
                whole.starts_with(&prompt),
                "cut after {k} tokens of {generation:?}: {prompt:?} is not a prefix"
            );
            let prompt_ids = tokenizer.encode(&prompt).unwrap();
            assert!(
                whole_ids.starts_with(&prompt_ids),
                "cut after {k} tokens of {generation:?}: its tokens are not a prefix"
            );
            assert!(
                prompt_ids.len() >= original_ids.len() + k - 1,
                "cut after {k} tokens: the continued prompt keeps all but at most one"
            );
            let rest = &whole[prompt.len()..];
            let (r2, c2) = parse(rest, Some(field));
            assert_eq!(format!("{r}{r2}"), reasoning, "cut after {k}: reasoning");
            assert_eq!(format!("{c}{c2}"), content, "cut after {k}: content");
            cuts += 1;
        }
    }
    assert!(cuts > 60, "only {cuts} cuts checked");
}

/// Continuing the answer ends the prompt after it, and continuing the
/// reasoning ends it inside an open reasoning block: the two shapes the rule
/// above relies on, spelled out.
#[test]
fn a_continued_prompt_ends_inside_the_field_it_continues() {
    let template = template();
    let question = "q";
    let original = original_prompt(&template, question);
    let (prompt, _) = continued_prompt(&template, question, "pulls", "The tides");
    assert_eq!(prompt, format!("{original}<think>pulls</think>The tides"));
    let (prompt, _) = continued_prompt(&template, question, "pulls", "");
    assert_eq!(prompt, format!("{original}<think>pulls"));
}

/// A continuation's output starts inside the field: reasoning until
/// `</think>`, with no leading tag to strip, or answer throughout.
#[test]
fn continued_output_is_split_from_where_the_prompt_ended() {
    assert_eq!(
        parse(
            " and the sun.</think>The answer.",
            Some("reasoning_content")
        ),
        (" and the sun.".into(), "The answer.".into())
    );
    // Still reasoning when the ceiling arrives again.
    assert_eq!(
        parse(" still thinking", Some("reasoning_content")),
        (" still thinking".into(), String::new())
    );
    // A `<think>` the reasoning writes is its text, not a tag to drop.
    assert_eq!(
        parse("<think> is a tag</think>ok", Some("reasoning_content")),
        ("<think> is a tag".into(), "ok".into())
    );
    // In the answer, reasoning tags are text.
    assert_eq!(
        parse(" more.</think> <think>", Some("content")),
        (String::new(), " more.</think> <think>".into())
    );
}

/// The renderer does not mutate what it was given: the request's own messages
/// are unchanged after a continued render, so nothing downstream ever sees the
/// sentinel.
#[test]
fn a_continued_render_leaves_its_input_alone() {
    let template = template();
    let messages = r#"[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]"#;
    let input = ChatInput::from_json(messages, None).unwrap();
    let before = input.messages.clone();
    template
        .render(
            &input,
            RenderOptions {
                continue_final_message: Some("content"),
                ..RenderOptions::default()
            },
        )
        .unwrap();
    assert_eq!(input.messages, before);
    assert!(matches!(input.messages, Json::Array(_)));
}
