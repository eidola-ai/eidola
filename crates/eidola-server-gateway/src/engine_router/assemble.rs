//! A blocking answer built from an engine's stream.
//!
//! The gateway asks a node for a stream whatever its client asked for, so a
//! client's blocking request is answered from the chunks the node sent: each
//! choice's deltas joined in order (`content`, `reasoning_content`,
//! `reasoning`, and each tool call by its `index`), its finish reason and
//! `continuable` from the chunk that carries them, and the last `usage` the
//! node reported. The result is the completion the node's own blocking answer
//! was: `content` null when nothing was given out, the reasoning fields only
//! when something was, `tool_calls` only when a call was made, each call
//! without the stream's `index` framing key. Held to the node's own blocking
//! answers by `tests::the_assembled_answer_is_the_nodes_blocking_answer`.

use std::collections::BTreeMap;

use crate::backend::{BackendMeta, BackendResponse};
use crate::types::{
    AssistantMessage, ChatCompletionChunk, ChatCompletionResponse, Choice, FinishReason, Role,
    Usage,
};

/// The most text one assembled answer holds: the bytes of its content,
/// reasoning and tool calls (their names, ids and arguments), as decoded, not
/// as their JSON spelling.
///
/// A blocking answer is held whole until it is done, so this is what one such
/// request can make the gateway buffer. The node stops at the request's
/// completion ceiling, at most the model's declared `max_output_tokens`, so a
/// real answer is that many tokens of a few bytes each, far below this; it is
/// also the most Eidola's client reads of a blocking answer
/// (`eidola-app-core`'s `MAX_RESPONSE_BYTES`), so an answer past it is one
/// that client would refuse. An answer that would exceed it ends the request
/// early, settled on what was generated.
pub const MAX_ASSEMBLED_BYTES: usize = 8 << 20;

/// The answer would exceed [`MAX_ASSEMBLED_BYTES`].
#[derive(Debug)]
pub struct TooLarge;

/// One choice as its deltas arrive.
#[derive(Default)]
struct ChoiceParts {
    content: String,
    reasoning_content: Option<String>,
    reasoning: Option<String>,
    /// Each tool call by its `index`, built from its fragments.
    tool_calls: BTreeMap<u64, serde_json::Value>,
    finish_reason: Option<FinishReason>,
    continuable: Option<bool>,
}

/// A blocking answer being built from a stream's chunks.
#[derive(Default)]
pub struct Assembler {
    id: Option<String>,
    created: u64,
    model: String,
    choices: BTreeMap<u32, ChoiceParts>,
    usage: Option<Usage>,
    held: usize,
}

impl Assembler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Take one chunk in. The first chunk names the answer (`id`, `created`,
    /// `model`); a chunk's `usage` replaces any earlier one.
    pub fn push(&mut self, chunk: &ChatCompletionChunk) -> Result<(), TooLarge> {
        if self.id.is_none() {
            self.id = Some(chunk.id.clone());
            self.created = chunk.created;
            self.model.clone_from(&chunk.model);
        }
        if chunk.usage.is_some() {
            self.usage.clone_from(&chunk.usage);
        }
        for choice in &chunk.choices {
            let delta = &choice.delta;
            let grows = [&delta.content, &delta.reasoning_content, &delta.reasoning]
                .into_iter()
                .flatten()
                .map(String::len)
                .sum::<usize>()
                + delta
                    .tool_calls
                    .iter()
                    .flatten()
                    .map(text_bytes)
                    .sum::<usize>();
            self.held = self.held.saturating_add(grows);
            if self.held > MAX_ASSEMBLED_BYTES {
                return Err(TooLarge);
            }
            let parts = self.choices.entry(choice.index).or_default();
            if let Some(text) = &delta.content {
                parts.content.push_str(text);
            }
            if let Some(text) = &delta.reasoning_content {
                parts
                    .reasoning_content
                    .get_or_insert_default()
                    .push_str(text);
            }
            if let Some(text) = &delta.reasoning {
                parts.reasoning.get_or_insert_default().push_str(text);
            }
            for (position, call) in delta.tool_calls.iter().flatten().enumerate() {
                let index = call
                    .get("index")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(position as u64);
                match parts.tool_calls.get_mut(&index) {
                    Some(built) => merge_fragment(built, call),
                    None => {
                        let mut call = call.clone();
                        if let Some(fields) = call.as_object_mut() {
                            fields.remove("index");
                        }
                        parts.tool_calls.insert(index, call);
                    }
                }
            }
            if choice.finish_reason.is_some() {
                parts.finish_reason = choice.finish_reason;
            }
            if choice.continuable.is_some() {
                parts.continuable = choice.continuable;
            }
        }
        Ok(())
    }

    /// The last usage the stream reported: the node's running count until its
    /// final one arrives.
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// The completion, for a stream that ended with its `[DONE]`.
    pub fn finish(self, meta: BackendMeta) -> BackendResponse {
        let choices = self
            .choices
            .into_iter()
            .map(|(index, parts)| Choice {
                index,
                message: AssistantMessage {
                    role: Role::Assistant,
                    content: (!parts.content.is_empty()).then_some(parts.content),
                    reasoning_content: parts.reasoning_content.filter(|s| !s.is_empty()),
                    reasoning: parts.reasoning.filter(|s| !s.is_empty()),
                    tool_calls: (!parts.tool_calls.is_empty())
                        .then(|| parts.tool_calls.into_values().collect()),
                },
                finish_reason: parts.finish_reason,
                continuable: parts.continuable,
            })
            .collect();
        BackendResponse {
            response: ChatCompletionResponse {
                id: self.id.unwrap_or_default(),
                object: "chat.completion".to_string(),
                created: self.created,
                model: self.model,
                choices,
                usage: self.usage,
            },
            meta,
        }
    }
}

/// The bytes of every string in a tool-call fragment.
fn text_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::String(s) => s.len(),
        serde_json::Value::Array(items) => items.iter().map(text_bytes).sum(),
        serde_json::Value::Object(fields) => {
            fields.iter().map(|(k, v)| k.len() + text_bytes(v)).sum()
        }
        _ => 0,
    }
}

/// Join a later fragment of a tool call into what has been built of it: its
/// strings appended (`function.name`, `function.arguments`), anything else
/// set where it is still absent.
fn merge_fragment(built: &mut serde_json::Value, fragment: &serde_json::Value) {
    let (Some(built), Some(fragment)) = (built.as_object_mut(), fragment.as_object()) else {
        return;
    };
    for (key, value) in fragment {
        if key == "index" {
            continue;
        }
        match (built.get_mut(key), value) {
            (Some(serde_json::Value::String(have)), serde_json::Value::String(more))
                if key != "id" && key != "type" =>
            {
                have.push_str(more)
            }
            (Some(have @ serde_json::Value::Object(_)), serde_json::Value::Object(_)) => {
                merge_fragment(have, value)
            }
            (Some(_), _) => {}
            (None, _) => {
                built.insert(key.clone(), value.clone());
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::backend::TeeType;

    /// The node's blocking answers to a varied set of requests, each beside
    /// the stream the node sent for the same request: scripted engine steps
    /// through the node's own response code (content, reasoning, one and
    /// several tool calls, `length` endings continuable and not, a stop
    /// sequence, an empty answer, escapes and multi-byte text), and requests
    /// to the node's dev model on its CPU executor.
    pub(crate) const PARITY: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/blocking-parity.json"
    ));

    pub(crate) struct ParityCase {
        pub name: String,
        pub blocking: serde_json::Value,
        /// The stream's events, `[DONE]` last.
        pub stream: Vec<String>,
    }

    pub(crate) fn parity_cases() -> Vec<ParityCase> {
        let cases: Vec<serde_json::Value> = serde_json::from_str(PARITY).unwrap();
        cases
            .into_iter()
            .map(|c| ParityCase {
                name: c["name"].as_str().unwrap().to_string(),
                blocking: c["blocking"].clone(),
                stream: c["stream"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| e.as_str().unwrap().to_string())
                    .collect(),
            })
            .collect()
    }

    /// A completion as the gateway hands it on, with what differs between two
    /// requests set to fixed values: the completion's `id` and `created`,
    /// and each tool call's `id` (a per-response random prefix), by position.
    pub(crate) fn normalised(completion: &ChatCompletionResponse) -> serde_json::Value {
        let mut v = serde_json::to_value(completion).unwrap();
        v["id"] = "chatcmpl-normalised".into();
        v["created"] = 0.into();
        for choice in v["choices"].as_array_mut().unwrap() {
            if let Some(calls) = choice["message"]["tool_calls"].as_array_mut() {
                for (i, call) in calls.iter_mut().enumerate() {
                    call["id"] = format!("call_{i}").into();
                }
            }
        }
        v
    }

    pub(crate) fn meta() -> BackendMeta {
        BackendMeta {
            provider: "test".to_string(),
            chat_id: None,
            backend_model: String::new(),
            tee_type: Some(TeeType::EidolaEngine),
            usage: None,
        }
    }

    pub(crate) fn assembled(stream: &[String]) -> ChatCompletionResponse {
        assert_eq!(stream.last().map(String::as_str), Some("[DONE]"));
        let mut assembler = Assembler::new();
        for event in &stream[..stream.len() - 1] {
            let chunk: ChatCompletionChunk = serde_json::from_str(event).unwrap();
            assembler.push(&chunk).unwrap();
        }
        assembler.finish(meta()).response
    }

    #[test]
    fn the_assembled_answer_is_the_nodes_blocking_answer() {
        let cases = parity_cases();
        assert!(cases.len() >= 10);
        for case in cases {
            let node: ChatCompletionResponse =
                serde_json::from_value(case.blocking.clone()).unwrap();
            assert_eq!(
                normalised(&assembled(&case.stream)),
                normalised(&node),
                "{}",
                case.name
            );
        }
    }

    /// The parity cases cover what they say: every message field, each
    /// finish reason and both `continuable` verdicts.
    #[test]
    fn the_parity_cases_cover_every_shape() {
        let blocking: Vec<serde_json::Value> =
            parity_cases().into_iter().map(|c| c.blocking).collect();
        let choice = |b: &serde_json::Value| b["choices"][0].clone();
        let any = |f: &dyn Fn(&serde_json::Value) -> bool| blocking.iter().any(|b| f(&choice(b)));
        assert!(any(&|c| c["message"]["content"].is_string()));
        assert!(any(&|c| c["message"]["content"].is_null()));
        assert!(any(&|c| c["message"]["reasoning_content"].is_string()
            && c["message"]["content"].is_string()));
        assert!(any(&|c| c["message"]["tool_calls"]
            .as_array()
            .map(Vec::len)
            == Some(1)));
        assert!(any(&|c| c["message"]["tool_calls"]
            .as_array()
            .map(Vec::len)
            > Some(1)));
        for reason in ["stop", "length", "tool_calls"] {
            assert!(any(&|c| c["finish_reason"] == reason), "{reason}");
        }
        assert!(any(&|c| c["continuable"] == true));
        assert!(any(&|c| c["continuable"] == false));
    }

    /// A tool call that arrives in fragments, as OpenAI-compatible streams
    /// send them, is joined by its index.
    #[test]
    fn a_tool_call_in_fragments_is_joined_by_its_index() {
        let chunk = |delta: serde_json::Value| -> ChatCompletionChunk {
            serde_json::from_value(serde_json::json!({
                "id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
            }))
            .unwrap()
        };
        let mut a = Assembler::new();
        for delta in [
            serde_json::json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_", "arguments": "{\"ci"}}]}),
            serde_json::json!({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}),
            serde_json::json!({"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": "ty\": 1}"}}]}),
        ] {
            a.push(&chunk(delta)).unwrap();
        }
        let calls = a.finish(meta()).response.choices[0]
            .message
            .tool_calls
            .clone()
            .unwrap();
        assert_eq!(
            serde_json::Value::Array(calls),
            serde_json::json!([
                {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": 1}"}},
                {"id": "call_b", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ])
        );
    }

    /// The text held is bounded: the delta that takes it past
    /// [`MAX_ASSEMBLED_BYTES`] is refused.
    #[test]
    fn an_answer_past_the_bound_is_refused() {
        let chunk = |text: String| -> ChatCompletionChunk {
            serde_json::from_value(serde_json::json!({
                "id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {"reasoning_content": text}, "finish_reason": null}],
            }))
            .unwrap()
        };
        let mut a = Assembler::new();
        a.push(&chunk("x".repeat(MAX_ASSEMBLED_BYTES - 1))).unwrap();
        a.push(&chunk("x".to_string())).unwrap();
        assert!(a.push(&chunk("x".to_string())).is_err());
    }
}
