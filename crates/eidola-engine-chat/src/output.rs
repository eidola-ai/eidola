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
//! whose closing tag never arrived is given out as content (see
//! [`crate::tool_call`]).

use std::ops::Range;

use crate::ChatFormat;
use crate::args::ToolSchemas;
use crate::deepseek_v4_1::output::{DsmlParser, ReasoningSplitter};
use crate::error::ChatError;
use crate::reasoning::{ReasoningDelta, ReasoningParser};
use crate::tokenizer::{Detokenizer, Tokenizer};
use crate::tool_call::{CallIdSource, ToolCall, ToolCallDelta, ToolCallParser};

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
    /// The final message's field the prompt ended inside
    /// ([`crate::RenderOptions::continue_final_message`]), so the output
    /// continues it: `reasoning_content` starts inside the reasoning, any other
    /// field in the answer. `None` for a new assistant turn.
    pub continuing: Option<&'static str>,
    /// The format the output is written in.
    pub format: ChatFormat,
}

impl OutputConfig {
    /// The configuration for a MiMo response.
    pub fn new(enable_thinking: Option<bool>, tools: Option<ToolSchemas>) -> OutputConfig {
        OutputConfig {
            thinking: enable_thinking != Some(false),
            tools: tools.filter(|t| !t.is_empty()),
            skip_special_tokens: true,
            continuing: None,
            format: ChatFormat::MimoV2_6,
        }
    }

    /// The configuration for a DeepSeek-V4.1 response to a prompt encoded
    /// with [`crate::deepseek_v4_1::EncodeOptions::thinking`] as `thinking`;
    /// `tools` are the request's, from [`crate::deepseek_v4_1::tool_schemas`].
    pub fn deepseek_v4_1(thinking: bool, tools: Option<ToolSchemas>) -> OutputConfig {
        OutputConfig {
            thinking,
            tools: tools.filter(|t| !t.is_empty()),
            skip_special_tokens: true,
            continuing: None,
            format: ChatFormat::DeepSeekV4_1,
        }
    }

    /// The same configuration for the output of a prompt rendered with
    /// `continue_final_message` set to `field`.
    pub fn continuing_in(mut self, field: Option<&'static str>) -> OutputConfig {
        self.continuing = field;
        self
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

/// A format's reasoning splitter.
enum Reasoning {
    Mimo(ReasoningParser),
    DeepSeek(ReasoningSplitter),
}

impl Reasoning {
    fn push(&mut self, text: &str) -> ReasoningDelta {
        match self {
            Reasoning::Mimo(parser) => parser.push(text),
            Reasoning::DeepSeek(splitter) => splitter.push(text),
        }
    }

    fn finish(&mut self) -> ReasoningDelta {
        match self {
            Reasoning::Mimo(parser) => parser.finish(),
            Reasoning::DeepSeek(splitter) => splitter.finish(),
        }
    }
}

/// A format's tool-call parser.
enum Tools {
    Mimo(ToolCallParser),
    DeepSeek(DsmlParser),
}

impl Tools {
    fn push(&mut self, text: &str) -> ToolCallDelta {
        match self {
            Tools::Mimo(parser) => parser.push(text),
            Tools::DeepSeek(parser) => parser.push(text),
        }
    }

    fn before_any_block(&self) -> bool {
        match self {
            Tools::Mimo(parser) => parser.before_any_block(),
            Tools::DeepSeek(parser) => parser.before_any_block(),
        }
    }

    fn finish(&mut self) -> ToolCallDelta {
        match self {
            Tools::Mimo(parser) => parser.finish(),
            Tools::DeepSeek(parser) => parser.finish(),
        }
    }
}

/// The whole output pipeline for one response.
pub struct OutputParser {
    detok: Detokenizer,
    reasoning: Reasoning,
    tools: Option<Tools>,
    tool_calls_emitted: bool,
    /// What [`OutputParser::resumes_exactly`] compares: the text fed in, the
    /// reasoning and content given out, whether the text lost bytes on the
    /// way, and where the prompt ended.
    raw: String,
    emitted_reasoning: String,
    emitted_content: String,
    lossy: bool,
    /// The output ended inside a possible delimiter a splitter was holding back.
    cut_in_delimiter: bool,
    thinking: bool,
    continuing: Option<&'static str>,
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
            reasoning: match (config.format, config.continuing) {
                (ChatFormat::MimoV2_6, None) => {
                    Reasoning::Mimo(ReasoningParser::new(config.thinking))
                }
                (ChatFormat::MimoV2_6, Some(field)) => {
                    Reasoning::Mimo(ReasoningParser::resuming(field == "reasoning_content"))
                }
                (ChatFormat::DeepSeekV4_1, None) => {
                    Reasoning::DeepSeek(ReasoningSplitter::new(config.thinking))
                }
                (ChatFormat::DeepSeekV4_1, Some(field)) => {
                    Reasoning::DeepSeek(ReasoningSplitter::new(field == "reasoning_content"))
                }
            },
            tools: config.tools.map(|schemas| match config.format {
                ChatFormat::MimoV2_6 => Tools::Mimo(ToolCallParser::new(schemas, ids)),
                ChatFormat::DeepSeekV4_1 => Tools::DeepSeek(DsmlParser::new(schemas, ids)),
            }),
            tool_calls_emitted: false,
            raw: String::new(),
            emitted_reasoning: String::new(),
            emitted_content: String::new(),
            lossy: false,
            cut_in_delimiter: false,
            thinking: config.thinking,
            continuing: config.continuing,
        }
    }

    /// The text fed in was decoded lossily (an incomplete UTF-8 sequence
    /// flushed at the end), so it is not the bytes the model generated. Called
    /// by a caller that runs its own detokenizer and feeds text.
    pub fn mark_lossy(&mut self) {
        self.lossy = true;
    }

    /// **Whether this response can be continued from what it gave out**, read
    /// after [`Self::finish`]: true exactly when the reasoning and content it
    /// emitted, sent back as a continued final message (the answer once there
    /// is any, the reasoning before), make the template render precisely the
    /// text the model generated after the prompt — nothing dropped, nothing
    /// added. A tool call, a tool-call block begun (past one the parser no
    /// longer reads as a resumed one would), a lossy final character, a repeated or missing `<think>`, reasoning ended
    /// by `<tool_call>` instead of `</think>`, a `</think>` with no answer after
    /// it, an ending inside a possible delimiter (`</thi`, `<tool_`: text
    /// byte-exact, but a resumed parser would not know it is mid-delimiter), or
    /// a splitter in another mode than a resumed one starts in (still at the
    /// start, right after a lone `<think>`) each make it false — so a resumed parse of the rest splits exactly as an
    /// uninterrupted one would.
    pub fn resumes_exactly(&self) -> bool {
        self.continued_text()
            .is_some_and(|(expected, _)| expected == self.raw)
    }

    /// [`Self::resumes_exactly`], **in tokens as well as in text**. `added` is
    /// where in the text fed in the model sampled an added token (`<think>`,
    /// `</think>`, `<tool_call>`, …): the byte range of each one's text, in
    /// order. A continued prompt encodes an added token only where the template
    /// writes one, the `<think>` and `</think>` around the reasoning, and
    /// everything given out as reasoning or content is request data there, which
    /// spells none ([`crate::MimoTokenizer::encode_prompt`]). So the verdict is
    /// true only when the model sampled exactly those tags as tokens and no
    /// other: a sampled added token that reached the text, or a tag the model
    /// wrote out of ordinary tokens, would come back as different tokens.
    pub fn resumes_exactly_as_tokens(&self, added: &[Range<usize>]) -> bool {
        self.continued_text()
            .is_some_and(|(expected, tags)| expected == self.raw && added == tags)
    }

    /// What the template renders after the prompt when the reasoning and
    /// content given out come back as a continued final message, and where in it
    /// the template's own `<think>` and `</think>` are. `None` where no
    /// continuation would resume this parse.
    fn continued_text(&self) -> Option<(String, Vec<Range<usize>>)> {
        // The DeepSeek-V4.1 encoder has no continued final message to resume
        // from.
        let Reasoning::Mimo(reasoning) = &self.reasoning else {
            return None;
        };
        if self.tool_calls_emitted || self.lossy || self.cut_in_delimiter {
            return None;
        }
        let (r, c) = (&self.emitted_reasoning, &self.emitted_content);
        // **The splitter must be in the mode a resumed one starts in**: the
        // answer once there is any (a resumed parser in content), the reasoning
        // before (a resumed parser inside the reasoning). Right after a lone
        // `<think>` the splitter is still at the start, where a further `<think>`
        // is a tag to strip, and with thinking off it is in the answer although
        // nothing has been said: neither is where a continuation would resume.
        let resumed_mode = if c.is_empty() {
            reasoning.in_reasoning()
        } else {
            reasoning.in_content()
        };
        if !resumed_mode {
            return None;
        }
        let after_reasoning = |open: &str| {
            let mut text = String::new();
            let mut tags = Vec::new();
            for (tag, written) in [(open, true), (r, false), ("</think>", true), (c, false)] {
                if tag == "</think>" && written && c.is_empty() {
                    break;
                }
                if written && !tag.is_empty() {
                    tags.push(text.len()..text.len() + tag.len());
                }
                text.push_str(tag);
            }
            (text, tags)
        };
        match self.continuing {
            None if self.thinking => Some(after_reasoning("<think>")),
            Some("reasoning_content") => Some(after_reasoning("")),
            // Thinking disabled, or the answer continued: the prompt already
            // holds any reasoning, so all the output is answer.
            _ if r.is_empty() => Some((c.clone(), Vec::new())),
            _ => None,
        }
    }

    /// Feeds one generated token. An id the tokenizer does not define is refused
    /// ([`ChatError::UnknownToken`]) and changes nothing.
    pub fn push_token(&mut self, tokenizer: &Tokenizer, id: u32) -> Result<ChatDelta, ChatError> {
        let text = self.detok.push(tokenizer, id)?;
        Ok(self.push_text(&text))
    }

    /// Feeds already-decoded text.
    pub fn push_text(&mut self, text: &str) -> ChatDelta {
        if text.is_empty() {
            return ChatDelta::default();
        }
        self.raw.push_str(text);
        let split = self.reasoning.push(text);
        let mut delta = ChatDelta {
            reasoning_content: split.reasoning,
            ..ChatDelta::default()
        };
        delta.append(self.route_content(&split.content));
        self.emitted_reasoning.push_str(&delta.reasoning_content);
        self.emitted_content.push_str(&delta.content);
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
        self.lossy |= self.detok.pending_len() > 0;
        let tail = self.detok.finish();
        let mut delta = self.push_text(&tail);
        let held = self.reasoning.finish().reasoning;
        // What a splitter held back is a possible partial delimiter (`</thi`,
        // `<tool_`): given out as text here, but a parser resumed from it starts
        // with nothing held and would read the rest of the delimiter as text.
        self.cut_in_delimiter |= !held.is_empty();
        self.emitted_reasoning.push_str(&held);
        delta.reasoning_content.push_str(&held);
        if let Some(parser) = &mut self.tools {
            // Past a block, the parser no longer reads text as it does before one,
            // and a resumed parser would start before any.
            self.cut_in_delimiter |= !parser.before_any_block();
            let rest = parser.finish();
            self.tool_calls_emitted |= !rest.calls.is_empty();
            self.cut_in_delimiter |= !rest.content.is_empty();
            self.emitted_content.push_str(&rest.content);
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
        assert_eq!(
            (d.content.as_str(), reason),
            ("Hi <tool_call><function=get_weather>", FinishReason::Length)
        );
        assert!(d.reasoning_content.is_empty());
    }

    /// With tools offered, an answer that mentions `<tool_call>` and never
    /// closes it keeps every word, streamed or not.
    #[test]
    fn prose_mentioning_the_tag_reaches_the_answer() {
        let text = "Emit <tool_call> followed by the function, and the rest of the answer.";
        let mut p = parser(Some(false), true);
        let chunks: Vec<String> = text.chars().map(String::from).collect();
        let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let (d, reason) = run(&mut p, &chunks, StopCause::EndOfSequence);
        assert_eq!((d.content.as_str(), reason), (text, FinishReason::Stop));
        assert!(d.tool_calls.is_empty());
        // Past a block, a cut is not resumed exactly.
        let mut p = parser(Some(false), true);
        run(&mut p, &[text], StopCause::MaxTokens);
        assert!(!p.resumes_exactly());
    }

    /// Which length endings can be continued from what the parser gave out —
    /// exactly those where the given-out text, rendered back as the continued
    /// message, is the text the model generated.
    #[test]
    fn a_cut_resumes_exactly_only_where_nothing_was_dropped() {
        let ends = |thinking: Option<bool>, tools: bool, chunks: &[&str]| {
            let mut p = parser(thinking, tools);
            run(&mut p, chunks, StopCause::MaxTokens);
            p.resumes_exactly()
        };
        // Inside the reasoning, inside the answer, and an answer with thinking off.
        assert!(ends(None, true, &["<think>weighing the"]));
        assert!(ends(None, true, &["<think>weighed</think>The sky is"]));
        assert!(ends(Some(false), true, &["The sky is"]));
        // Markup the parser keeps as text is still exact.
        assert!(ends(Some(false), false, &["Hi <tool_call><function=x>"]));
        // An unfinished tool call is given out as text, but the parser is past a
        // block.
        assert!(!ends(
            Some(false),
            true,
            &["Hi <tool_call><function=get_weather>"]
        ));
        assert!(!ends(
            None,
            true,
            &["<think>plan<tool_call><function=get_weather>"]
        ));
        // A completed call is a tool round, never a continued answer.
        assert!(!ends(
            Some(false),
            true,
            &["<tool_call><function=get_weather></function></tool_call>"]
        ));
        // A repeated `<think>` is stripped; a `</think>` with no answer after it
        // would be dropped by a reasoning continuation.
        assert!(!ends(None, true, &["<think><think>twice"]));
        assert!(!ends(None, true, &["<think>done</think>"]));
        // No opening `<think>`: the continued prompt would add one.
        assert!(!ends(None, true, &["straight into it"]));
        // Right after a lone `<think>` the splitter is still at the start, where
        // a further `<think>` is stripped; a resumed reasoning would keep it. And
        // with thinking off and nothing said, the splitter is in the answer while
        // a client would continue the (empty) reasoning.
        assert!(!ends(None, true, &["<think>"]));
        assert!(!ends(None, true, &["<think><think>"]));
        assert!(!ends(Some(false), true, &[""]));
        assert!(ends(None, true, &["<think>\n"]));
        // Inside a delimiter written out of ordinary tokens: the text is
        // byte-exact, but a resumed parser would not know it is mid-delimiter.
        assert!(!ends(None, true, &["<think>plan</thi"]));
        assert!(!ends(Some(false), true, &["Hi <tool_"]));
        assert!(!ends(None, false, &["<thi"]));
        // Past an undeclared function's block, returned to the answer as text:
        // the parser no longer reads as it does before any block.
        assert!(!ends(
            Some(false),
            true,
            &["<tool_call><function=unknown></function></tool_call>"]
        ));
        // An incomplete final character is flushed lossily.
        let mut p = parser(Some(false), false);
        run(&mut p, &["ok"], StopCause::MaxTokens);
        assert!(p.resumes_exactly());
        p.mark_lossy();
        assert!(!p.resumes_exactly());

        // Continuations, judged from where their prompt ended.
        let resumed = |field: &'static str, chunks: &[&str]| {
            let mut p = OutputParser::new(
                OutputConfig::new(None, None).continuing_in(Some(field)),
                PrefixedCallIds("call_".into()),
            );
            run(&mut p, chunks, StopCause::MaxTokens);
            p.resumes_exactly()
        };
        assert!(resumed("reasoning_content", &[" more thought"]));
        assert!(resumed("reasoning_content", &[" more</think>An answer"]));
        assert!(!resumed("reasoning_content", &[" more</think>"]));
        assert!(resumed("content", &[" and the rest"]));
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
                proptest::prop_assert!(matches!(json::parse_strict(&call.arguments), Ok(json::Json::Object(_))));
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
