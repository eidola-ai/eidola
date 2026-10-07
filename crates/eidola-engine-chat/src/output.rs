//! The streaming output pipeline: token ids → text → reasoning / content /
//! tool calls → OpenAI deltas, and the finish-reason mapping.
//!
//! ```text
//! ids ─ Detokenizer ─ text ─ ReasoningParser ─┬─ reasoning_content
//!                                             └─ content ─ ToolCallParser ─┬─ content
//!                                                                          └─ tool_calls
//! ```
//!
//! # Finish reasons
//!
//! The engine reports why generation stopped ([`StopCause`]); the OpenAI
//! `finish_reason` follows from it and from whether any tool call was
//! emitted ([`finish_reason`]):
//!
//! | Stop cause | No tool call | ≥ 1 tool call |
//! |---|---|---|
//! | end-of-sequence token (`generation_config.json` `eos_token_id`) | `stop` | `tool_calls` |
//! | a request `stop` sequence matched | `stop` | `tool_calls` |
//! | `max_tokens` or the context window reached | `length` | `length` |
//!
//! A response cut off by `length` keeps the calls already completed; a call
//! whose closing tag never arrived is dropped (see [`crate::tool_call`]).

use crate::args::ToolSchemas;
use crate::error::ChatError;
use crate::reasoning::ReasoningParser;
use crate::tokenizer::{Detokenizer, MimoTokenizer};
use crate::tool_call::{CallIdSource, ToolCall, ToolCallParser};

/// Why the engine stopped generating.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopCause {
    /// An end-of-sequence token was sampled.
    EndOfSequence,
    /// A request `stop` sequence matched.
    StopSequence,
    /// `max_tokens` (or the context window) was exhausted.
    MaxTokens,
}

/// OpenAI's `finish_reason`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

/// The finish-reason mapping (module docs).
pub fn finish_reason(cause: StopCause, tool_calls_emitted: bool) -> FinishReason {
    match cause {
        StopCause::MaxTokens => FinishReason::Length,
        StopCause::EndOfSequence | StopCause::StopSequence if tool_calls_emitted => {
            FinishReason::ToolCalls
        }
        StopCause::EndOfSequence | StopCause::StopSequence => FinishReason::Stop,
    }
}

/// How to interpret one response's output.
#[derive(Clone, Debug)]
pub struct OutputConfig {
    /// Whether thinking is on (`enable_thinking` was not `false`).
    pub thinking: bool,
    /// The request's tools; `None` (or no tools, or `tool_choice: "none"`)
    /// disables tool-call parsing and leaves the markup in the content.
    pub tools: Option<ToolSchemas>,
    /// Drop special tokens from the text (the OpenAI-compatible default).
    pub skip_special_tokens: bool,
}

impl OutputConfig {
    pub fn new(enable_thinking: Option<bool>, tools: Option<ToolSchemas>) -> OutputConfig {
        OutputConfig {
            thinking: enable_thinking != Some(false),
            tools: tools.filter(|t| !t.is_empty()),
            skip_special_tokens: true,
        }
    }
}

/// One streaming step's worth of output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChatDelta {
    pub reasoning_content: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ChatDelta {
    pub fn is_empty(&self) -> bool {
        self.reasoning_content.is_empty() && self.content.is_empty() && self.tool_calls.is_empty()
    }

    fn append(&mut self, other: ChatDelta) {
        self.reasoning_content.push_str(&other.reasoning_content);
        self.content.push_str(&other.content);
        self.tool_calls.extend(other.tool_calls);
    }

    /// The `choices[].delta` object of a streaming chunk (empty fields
    /// omitted).
    pub fn to_openai_delta(&self) -> serde_json::Value {
        let mut delta = serde_json::Map::new();
        if !self.reasoning_content.is_empty() {
            delta.insert(
                "reasoning_content".into(),
                self.reasoning_content.clone().into(),
            );
        }
        if !self.content.is_empty() {
            delta.insert("content".into(), self.content.clone().into());
        }
        if !self.tool_calls.is_empty() {
            delta.insert(
                "tool_calls".into(),
                self.tool_calls.iter().map(ToolCall::to_openai).collect(),
            );
        }
        delta.into()
    }
}

/// The whole output pipeline for one response.
pub struct OutputParser {
    detok: Detokenizer,
    reasoning: ReasoningParser,
    tools: Option<ToolCallParser>,
    tool_calls_emitted: bool,
}

impl std::fmt::Debug for OutputParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputParser")
            .field("tool_calls_emitted", &self.tool_calls_emitted)
            .finish_non_exhaustive()
    }
}

impl OutputParser {
    pub fn new(config: OutputConfig, ids: impl CallIdSource + 'static) -> OutputParser {
        OutputParser {
            detok: Detokenizer::new(config.skip_special_tokens),
            reasoning: ReasoningParser::new(config.thinking),
            tools: config
                .tools
                .map(|schemas| ToolCallParser::new(schemas, ids)),
            tool_calls_emitted: false,
        }
    }

    /// Feeds one generated token. An id the tokenizer does not define is refused
    /// ([`ChatError::UnknownToken`]) and changes nothing.
    pub fn push_token(
        &mut self,
        tokenizer: &MimoTokenizer,
        id: u32,
    ) -> Result<ChatDelta, ChatError> {
        let text = self.detok.push(tokenizer, id)?;
        Ok(self.push_text(&text))
    }

    /// Feeds already-decoded text.
    pub fn push_text(&mut self, text: &str) -> ChatDelta {
        if text.is_empty() {
            return ChatDelta::default();
        }
        let split = self.reasoning.push(text);
        let mut delta = ChatDelta {
            reasoning_content: split.reasoning,
            ..ChatDelta::default()
        };
        delta.append(self.route_content(&split.content));
        delta
    }

    fn route_content(&mut self, content: &str) -> ChatDelta {
        match &mut self.tools {
            None => ChatDelta {
                content: content.to_string(),
                ..ChatDelta::default()
            },
            Some(parser) => {
                let parsed = parser.push(content);
                self.tool_calls_emitted |= !parsed.calls.is_empty();
                ChatDelta {
                    content: parsed.content,
                    tool_calls: parsed.calls,
                    ..ChatDelta::default()
                }
            }
        }
    }

    /// Ends the response: flushes held-back text and maps the finish reason.
    pub fn finish(&mut self, cause: StopCause) -> (ChatDelta, FinishReason) {
        let tail = self.detok.finish();
        let mut delta = self.push_text(&tail);
        delta
            .reasoning_content
            .push_str(&self.reasoning.finish().reasoning);
        if let Some(parser) = &mut self.tools {
            let rest = parser.finish();
            self.tool_calls_emitted |= !rest.calls.is_empty();
            delta.content.push_str(&rest.content);
            delta.tool_calls.extend(rest.calls);
        }
        (delta, finish_reason(cause, self.tool_calls_emitted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;
    use crate::tool_call::PrefixedCallIds;

    fn parser(thinking: Option<bool>, with_tools: bool) -> OutputParser {
        let tools = with_tools.then(|| {
            ToolSchemas::from_tools(
                &json::parse(
                    r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#,
                )
                .unwrap(),
            )
        });
        OutputParser::new(
            OutputConfig::new(thinking, tools),
            PrefixedCallIds("call_".into()),
        )
    }

    fn run(p: &mut OutputParser, chunks: &[&str], cause: StopCause) -> (ChatDelta, FinishReason) {
        let mut all = ChatDelta::default();
        for c in chunks {
            all.append(p.push_text(c));
        }
        let (rest, reason) = p.finish(cause);
        all.append(rest);
        (all, reason)
    }

    #[test]
    fn reasoning_content_and_tool_call() {
        let mut p = parser(None, true);
        let (d, reason) = run(
            &mut p,
            &[
                "<think>need wea",
                "ther</think>Checking.",
                "<tool_call><function=get_weather><parameter=city>Oslo</parameter></function></tool_call>",
            ],
            StopCause::EndOfSequence,
        );
        assert_eq!(d.reasoning_content, "need weather");
        assert_eq!(d.content, "Checking.");
        assert_eq!(d.tool_calls.len(), 1);
        assert_eq!(d.tool_calls[0].arguments, r#"{"city": "Oslo"}"#);
        assert_eq!(reason, FinishReason::ToolCalls);
        assert_eq!(d.to_openai_delta()["tool_calls"][0]["id"], "call_0");
    }

    #[test]
    fn tool_call_opened_inside_reasoning() {
        let mut p = parser(None, true);
        let (d, reason) = run(
            &mut p,
            &["<think>plan<tool_call><function=get_weather></function></tool_call>"],
            StopCause::EndOfSequence,
        );
        assert_eq!(d.reasoning_content, "plan");
        assert_eq!(d.tool_calls[0].arguments, "{}");
        assert_eq!(reason, FinishReason::ToolCalls);
    }

    #[test]
    fn finish_reasons() {
        assert_eq!(
            finish_reason(StopCause::EndOfSequence, false),
            FinishReason::Stop
        );
        assert_eq!(
            finish_reason(StopCause::StopSequence, true),
            FinishReason::ToolCalls
        );
        assert_eq!(
            finish_reason(StopCause::MaxTokens, true),
            FinishReason::Length
        );
        let mut p = parser(Some(false), true);
        let (d, reason) = run(
            &mut p,
            &["Hi <tool_call><function=get_weather>"],
            StopCause::MaxTokens,
        );
        assert_eq!((d.content.as_str(), reason), ("Hi ", FinishReason::Length));
        assert!(d.reasoning_content.is_empty());
    }

    #[test]
    fn without_tools_markup_stays_content() {
        let mut p = parser(Some(false), false);
        let text = "<tool_call><function=x></function></tool_call>";
        let (d, reason) = run(&mut p, &[text], StopCause::EndOfSequence);
        assert_eq!((d.content.as_str(), reason), (text, FinishReason::Stop));
    }

    proptest::proptest! {
        #[test]
        fn any_token_stream_is_safe_and_split_invariant(
            ids in proptest::collection::vec(0u32..270, 0..80),
            thinking: Option<bool>,
            with_tools: bool,
        ) {
            let tokenizer = crate::tokenizer::tests::synthetic();
            let mut one_by_one = parser(thinking, with_tools);
            let mut streamed = ChatDelta::default();
            for &id in &ids {
                // Ids past the vocabulary are refused and change nothing.
                match one_by_one.push_token(&tokenizer, id) {
                    Ok(delta) => streamed.append(delta),
                    Err(e) => {
                        proptest::prop_assert!((id as usize) >= tokenizer.vocab_size());
                        proptest::prop_assert_eq!(e, ChatError::UnknownToken(id));
                    }
                }
            }
            let (rest, reason) = one_by_one.finish(StopCause::EndOfSequence);
            streamed.append(rest);

            let known: Vec<u32> = ids
                .iter()
                .copied()
                .filter(|&id| (id as usize) < tokenizer.vocab_size())
                .collect();
            let mut whole = parser(thinking, with_tools);
            let mut at_once = whole.push_text(&tokenizer.decode(&known, true).unwrap());
            let (rest, reason_whole) = whole.finish(StopCause::EndOfSequence);
            at_once.append(rest);
            proptest::prop_assert_eq!(&streamed, &at_once);
            proptest::prop_assert_eq!(reason, reason_whole);
            for call in &streamed.tool_calls {
                proptest::prop_assert!(matches!(json::parse(&call.arguments), Ok(json::Json::Object(_))));
            }
        }

        #[test]
        fn any_text_chunking_is_safe_and_split_invariant(
            chunks in proptest::collection::vec(
                proptest::prop_oneof![
                    "\\PC{0,8}",
                    proptest::strategy::Just("<think>".to_string()),
                    proptest::strategy::Just("</think>".to_string()),
                    proptest::strategy::Just("<tool_call><function=get_weather>".to_string()),
                    proptest::strategy::Just("<parameter=city>".to_string()),
                    proptest::strategy::Just("</parameter></function></tool_call>".to_string()),
                ],
                0..24,
            ),
            thinking: Option<bool>,
        ) {
            let mut streaming = parser(thinking, true);
            let mut streamed = ChatDelta::default();
            for c in &chunks {
                streamed.append(streaming.push_text(c));
            }
            streamed.append(streaming.finish(StopCause::MaxTokens).0);
            let mut whole = parser(thinking, true);
            let mut at_once = whole.push_text(&chunks.concat());
            at_once.append(whole.finish(StopCause::MaxTokens).0);
            proptest::prop_assert_eq!(streamed, at_once);
        }
    }
}
