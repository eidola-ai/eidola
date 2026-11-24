//! When the next turn's prompt extends this one's: the relation a hybrid
//! model's prefix cache at turn ends depends on.
//!
//! A turn is simulated end to end: the prompt is rendered, the model's output
//! is the text the template renders for the assistant turn after it (what a
//! model that follows its trained format writes), the output parser splits it,
//! and the client sends back what was given out (reasoning included, and calls
//! with JSON-string arguments) with the next message. The rule:
//!
//! - **The next prompt is this prompt, the output, `<|im_end|>`, and more**, in
//!   text and in tokens (the output encoded as the model sampled it, its
//!   markup as added tokens), when the output is in the trained format, it
//!   ended with `<|im_end|>`, the client returned the reasoning, and the
//!   template keeps it: `preserve_thinking` is not `false`, or nothing but
//!   tool results came after it. With thinking on, an empty reasoning is the
//!   one exception in tokens: `<think>\n\n</think>` encodes its two newlines
//!   as one token, where the model wrote the second after the prompt's first,
//!   so the tokens extend the prompt less its last.
//! - Otherwise the next prompt still extends this prompt **up to the end of
//!   the generation header** `<|im_start|>assistant\n` of the turn's first
//!   request after the last user message: the generation prompt's last two
//!   tokens (`<think>`, `\n`) with thinking on, four with it off, are the only
//!   part a later render may change. A client that drops the reasoning keeps
//!   the prompt less its last token.
//! - A changed thinking switch or reasoning effort changes the system turn the
//!   prompt starts with.
//!
//! Qwen3.5's template renders the reasoning only of the turns after the last
//! user query, so it behaves as Qwen3.8's with `preserve_thinking` off: within
//! a user turn (a tool loop) the next prompt extends this one and its output,
//! and a new user message's prompt extends the turn's first request only to its
//! generation header (two tokens before its end with thinking on, four with it
//! off, its default).

use std::io::Read;
use std::path::PathBuf;

use eidola_engine_chat::json;
use eidola_engine_chat::qwen::{ChatTemplate, RenderOptions, TemplateKind};
use eidola_engine_chat::{
    ChatDelta, ChatFormat, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, StopCause,
    Tokenizer, ToolSchemas,
};
use serde_json::{Value, json};

const TOOLS: &str = r#"[{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}}}}]"#;

fn fixtures(format: ChatFormat) -> PathBuf {
    let dir = match format {
        ChatFormat::Qwen3_8 => "qwen",
        _ => "qwen35",
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
}

fn tokenizer(format: ChatFormat) -> Tokenizer {
    let compressed = std::fs::read(fixtures(format).join("tokenizer.json.gz")).unwrap();
    let mut tokenizer_json = Vec::new();
    flate2::read::GzDecoder::new(compressed.as_slice())
        .read_to_end(&mut tokenizer_json)
        .unwrap();
    let config = std::fs::read(fixtures(format).join(format.eos_config_file())).unwrap();
    Tokenizer::for_format(format, &tokenizer_json, &config).unwrap()
}

struct Session {
    template: ChatTemplate,
    tokenizer: Tokenizer,
    options: RenderOptions,
    format: ChatFormat,
}

/// What the model writes in a turn.
struct Reply {
    reasoning: &'static str,
    content: &'static str,
    calls: Vec<Value>,
}

fn reply(reasoning: &'static str, content: &'static str, calls: &[Value]) -> Reply {
    Reply {
        reasoning,
        content,
        calls: calls.to_vec(),
    }
}

impl Session {
    fn new(thinking: Option<bool>, preserve: Option<bool>) -> Session {
        Session::of(ChatFormat::Qwen3_8, thinking, preserve)
    }

    fn qwen35(thinking: Option<bool>) -> Session {
        Session::of(ChatFormat::Qwen3_5, thinking, None)
    }

    fn of(format: ChatFormat, thinking: Option<bool>, preserve: Option<bool>) -> Session {
        Session {
            template: ChatTemplate::from_model_dir(&fixtures(format)).unwrap(),
            tokenizer: tokenizer(format),
            options: RenderOptions {
                add_generation_prompt: true,
                enable_thinking: thinking,
                preserve_thinking: preserve,
                reasoning_effort: None,
            },
            format,
        }
    }

    fn thinks(&self) -> bool {
        self.template.kind().thinks(self.options.enable_thinking)
    }

    fn input(messages: &[Value]) -> ChatInput {
        ChatInput::from_json(&Value::from(messages.to_vec()).to_string(), Some(TOOLS)).unwrap()
    }

    fn prompt(&self, messages: &[Value]) -> (String, Vec<u32>) {
        let prompt = self
            .template
            .render_prompt(&Self::input(messages), &self.options)
            .unwrap();
        let ids = self.tokenizer.encode_prompt(&prompt).unwrap();
        (prompt.into_text(), ids)
    }

    /// The text the model writes after `messages`' generation prompt: the
    /// template's rendering of `reply`, without the end-of-turn token.
    fn output(&self, messages: &[Value], reply: &Reply) -> String {
        let thinking = self.thinks();
        let mut assistant = json!({"role": "assistant", "content": reply.content});
        if thinking {
            assistant["reasoning_content"] = reply.reasoning.into();
        }
        if !reply.calls.is_empty() {
            assistant["tool_calls"] = reply.calls.clone().into();
        }
        let mut turn = messages.to_vec();
        turn.push(assistant);
        let options = RenderOptions {
            add_generation_prompt: false,
            ..self.options.clone()
        };
        let full = self.template.render(&Self::input(&turn), &options).unwrap();
        let (prompt, _) = self.prompt(messages);
        full[prompt.len()..]
            .strip_suffix("<|im_end|>\n")
            .unwrap()
            .to_string()
    }

    /// The assistant message a client sends back: what the parser gave out.
    fn history(&self, output: &str, with_reasoning: bool) -> Value {
        let tools = Some(ToolSchemas::from_tools(&json::parse(TOOLS).unwrap()));
        let config = match self.template.kind() {
            TemplateKind::Qwen3_8 => OutputConfig::qwen3_8(self.options.enable_thinking, tools),
            TemplateKind::Qwen3_5 => OutputConfig::qwen3_5(self.options.enable_thinking, tools),
        };
        let mut parser = OutputParser::new(config, PrefixedCallIds("call_".into()));
        let mut d = parser.push_text(output);
        let rest = parser.finish(StopCause::EndOfSequence).0;
        d.reasoning_content.push_str(&rest.reasoning_content);
        d.content.push_str(&rest.content);
        d.tool_calls.extend(rest.tool_calls);
        let ChatDelta {
            reasoning_content,
            content,
            tool_calls,
        } = d;
        let mut message = json!({"role": "assistant", "content": content});
        if with_reasoning {
            message["reasoning_content"] = reasoning_content.into();
        }
        if !tool_calls.is_empty() {
            message["tool_calls"] = tool_calls.iter().map(|c| c.to_openai()).collect();
        }
        message
    }

    /// The generation prompt's tokens a later render may change.
    fn header_tail(&self) -> usize {
        if self.thinks() { 2 } else { 4 }
    }
}

fn bash(command: &str) -> Value {
    json!({"type": "function", "function": {"name": "bash", "arguments": {"command": command}}})
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

fn tool(text: &str) -> Value {
    json!({"role": "tool", "content": text})
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// How far the next prompt extends this one, in tokens.
enum Extends {
    /// The prompt, the output and `<|im_end|>`.
    Whole,
    /// In text, the prompt, the output and `<|im_end|>`; in tokens, the prompt
    /// less its last: an empty reasoning renders `<think>\n\n</think>`, whose
    /// two newlines are one token, where the model wrote them after the
    /// prompt's.
    EmptyReasoning,
    /// The prompt up to the generation header of the user turn's first
    /// request.
    ToHeader,
}

/// Runs a conversation: after each reply, the next messages; checks how far
/// each next prompt extends the one before.
fn converse(
    session: &Session,
    start: Vec<Value>,
    turns: Vec<(Reply, Vec<Value>, Extends)>,
    with_reasoning: bool,
) {
    let im_end = session.tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut messages = start;
    let mut first_of_user_turn: Option<usize> = None;
    for (i, (reply, next, expect)) in turns.into_iter().enumerate() {
        let (prompt, ids) = session.prompt(&messages);
        let header = ids.len() - session.header_tail();
        let header_text =
            prompt.rfind("<|im_start|>assistant\n").unwrap() + "<|im_start|>assistant\n".len();
        if messages.last().unwrap()["role"] == "user" {
            first_of_user_turn = Some(header);
        }
        assert_eq!(
            session.tokenizer.decode(&ids[header..], false).unwrap(),
            &prompt[header_text..],
            "the generation header ends {} tokens before the prompt does",
            session.header_tail()
        );
        let output = session.output(&messages, &reply);
        let mut generated = session.tokenizer.encode(&output).unwrap();
        generated.push(im_end);

        messages.push(session.history(&output, with_reasoning));
        messages.extend(next);
        let (next_prompt, next_ids) = session.prompt(&messages);
        let whole = [ids.clone(), generated].concat();
        let shared = common_prefix(&whole, &next_ids);
        match expect {
            Extends::Whole => {
                assert!(
                    next_prompt.starts_with(&format!("{prompt}{output}<|im_end|>")),
                    "turn {i}"
                );
                assert_eq!(shared, whole.len(), "turn {i}");
            }
            Extends::EmptyReasoning => {
                assert!(
                    next_prompt.starts_with(&format!("{prompt}{output}<|im_end|>")),
                    "turn {i}"
                );
                assert_eq!(shared, ids.len() - 1, "turn {i}");
            }
            Extends::ToHeader => {
                assert!(shared < whole.len(), "turn {i}");
                assert_eq!(
                    shared,
                    first_of_user_turn.unwrap(),
                    "turn {i}: {next_prompt:?}"
                );
            }
        }
    }
}

fn agentic_conversation(
    preserve_off: bool,
    thinking_off: bool,
) -> Vec<(Reply, Vec<Value>, Extends)> {
    // A new user message drops the reasoning of the turns before it.
    let to_header = || {
        if preserve_off {
            Extends::ToHeader
        } else {
            Extends::Whole
        }
    };
    vec![
        (
            reply("List first.", "Let me look.", &[bash("ls")]),
            vec![tool("a.txt\nb.txt")],
            Extends::Whole,
        ),
        (
            reply("Read both.", "", &[bash("cat a.txt"), bash("cat b.txt")]),
            vec![tool("A"), tool("B")],
            Extends::Whole,
        ),
        (
            reply("Done.", "Both files hold one letter.", &[]),
            vec![user("Thanks. And c.txt?")],
            to_header(),
        ),
        (
            reply("Check c.", "", &[bash("cat c.txt")]),
            vec![tool("C")],
            Extends::Whole,
        ),
        (
            reply("", "It holds C.", &[]),
            vec![user("Bye")],
            match to_header() {
                Extends::Whole if !thinking_off => Extends::EmptyReasoning,
                other => other,
            },
        ),
    ]
}

#[test]
fn the_next_prompt_extends_this_one_and_its_output() {
    for thinking in [None, Some(false)] {
        for preserve in [None, Some(true), Some(false)] {
            let session = Session::new(thinking, preserve);
            let start = vec![
                json!({"role": "system", "content": "You are terse."}),
                user("What is in this directory?"),
            ];
            converse(
                &session,
                start,
                agentic_conversation(preserve == Some(false), thinking == Some(false)),
                true,
            );
        }
    }
}

/// With thinking off the reasoning is empty, so a client that drops it loses
/// nothing; with thinking on, the next prompt extends this one's text only to
/// its end, where the reasoning starts, and its tokens to the last but one.
#[test]
fn a_client_that_drops_the_reasoning_keeps_only_the_prompt() {
    let session = Session::new(None, None);
    let (prompt, ids) = session.prompt(&[user("Hi")]);
    let output = session.output(&[user("Hi")], &reply("Greet back.", "Hello!", &[]));
    let messages = [
        user("Hi"),
        session.history(&output, false),
        user("How are you?"),
    ];
    let (next, next_ids) = session.prompt(&messages);
    assert!(next.starts_with(&prompt) && !next.starts_with(&format!("{prompt}{output}")));
    assert_eq!(common_prefix(&ids, &next_ids), ids.len() - 1);

    let session = Session::new(Some(false), None);
    converse(
        &session,
        vec![user("Hi")],
        vec![(
            reply("", "Hello!", &[]),
            vec![user("How are you?")],
            Extends::Whole,
        )],
        false,
    );
}

/// Output outside the trained format comes back as the template writes it:
/// here reasoning with a trailing space, which the template trims.
#[test]
fn output_outside_the_trained_format_is_rewritten() {
    let session = Session::new(None, None);
    let (prompt, _) = session.prompt(&[user("Hi")]);
    let output = "Greet back. \n</think>\n\nHello!";
    let messages = [user("Hi"), session.history(output, true), user("Again")];
    let (next, _) = session.prompt(&messages);
    assert!(next.starts_with(&format!("{prompt}Greet back.")));
    assert!(!next.starts_with(&format!("{prompt}{output}")));
}

/// A reasoning effort or thinking switch that changes between turns changes
/// the system turn, so the next prompt shares only what comes before it.
#[test]
fn changing_the_thinking_switch_changes_the_system_turn() {
    let messages = [user("Hi")];
    let (on, _) = Session::new(None, None).prompt(&messages);
    let (off, _) = Session::new(Some(false), None).prompt(&messages);
    let shared = on
        .bytes()
        .zip(off.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    assert_eq!(&on[..shared], "<|im_start|>system\n");
    assert!(on[shared..].starts_with("Reasoning effort is set to xhigh."));
}

/// Qwen3.5, thinking off (its default) and on: a tool loop's next prompt
/// extends this one and its output; a new user message's extends the user
/// turn's first request to its generation header.
#[test]
fn qwen35_the_next_prompt_extends_this_one_only_within_a_user_turn() {
    for thinking in [None, Some(true)] {
        let session = Session::qwen35(thinking);
        assert_eq!(session.format, ChatFormat::Qwen3_5);
        let start = vec![
            json!({"role": "system", "content": "You are terse."}),
            user("What is in this directory?"),
        ];
        converse(
            &session,
            start,
            agentic_conversation(true, thinking != Some(true)),
            true,
        );
    }
}

/// Qwen3.5 with thinking on: a client that drops the reasoning in a tool loop
/// keeps the prompt less its last token; with thinking off there is none to
/// drop.
#[test]
fn qwen35_a_client_that_drops_the_reasoning_keeps_only_the_prompt() {
    let session = Session::qwen35(Some(true));
    let start = [
        user("List it."),
        json!({"role": "assistant", "content": "", "tool_calls": [bash("ls")]}),
        tool("a"),
    ];
    let (prompt, ids) = session.prompt(&start);
    let output = session.output(&start, &reply("Read a.", "", &[bash("cat a")]));
    let mut messages = start.to_vec();
    messages.push(session.history(&output, false));
    messages.push(tool("A"));
    let (next, next_ids) = session.prompt(&messages);
    assert!(next.starts_with(&prompt) && !next.starts_with(&format!("{prompt}{output}")));
    assert_eq!(common_prefix(&ids, &next_ids), ids.len() - 1);

    converse(
        &Session::qwen35(None),
        vec![user("Hi")],
        vec![(
            reply("", "Let me look.", &[bash("ls")]),
            vec![tool("a")],
            Extends::Whole,
        )],
        false,
    );
}
