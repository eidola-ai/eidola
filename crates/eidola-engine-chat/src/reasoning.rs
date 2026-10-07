//! Separates MiMo's `<think>…</think>` reasoning from visible content.
//!
//! With thinking enabled the prompt ends at `<|im_start|>assistant\n` and the
//! model opens its turn with `<think>`. The parser starts in reasoning mode:
//!
//! - Leading `<think>` tags are removed (repeated ones too).
//! - Reasoning ends at the first `</think>` (dropped) or the first
//!   `<tool_call>` (kept, as the start of the content), whichever comes
//!   first. A `<tool_call>` opening before `</think>` is how the model
//!   sometimes starts calling a tool without closing its reasoning.
//! - Everything after that is content, verbatim (including any later
//!   `<think>` or `</think>` text).
//! - If the output ends inside reasoning, all of it is reasoning.
//!
//! With thinking disabled (`enable_thinking=false`, where the prompt already
//! ends with an empty `<think></think>`), everything is content.
//!
//! Streaming yields the same split for any chunking: only a suffix that could
//! still become `</think>` or `<tool_call>` (or, at the start, `<think>`) is
//! held back.

use crate::tool_call::partial_suffix;

const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL_START: &str = "<tool_call>";

/// Reasoning and content produced by one step.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReasoningDelta {
    pub reasoning: String,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Before the first non-`<think>` text.
    Start,
    Reasoning,
    Content,
}

/// Streaming reasoning splitter.
#[derive(Clone, Debug)]
pub struct ReasoningParser {
    mode: Mode,
    held: String,
}

impl ReasoningParser {
    pub fn new(thinking_enabled: bool) -> ReasoningParser {
        ReasoningParser {
            mode: if thinking_enabled {
                Mode::Start
            } else {
                Mode::Content
            },
            held: String::new(),
        }
    }

    /// Whether reasoning has ended (or never applied).
    pub fn in_content(&self) -> bool {
        self.mode == Mode::Content
    }

    pub fn push(&mut self, text: &str) -> ReasoningDelta {
        let mut out = ReasoningDelta::default();
        if self.mode == Mode::Content {
            out.content.push_str(text);
            return out;
        }
        self.held.push_str(text);
        if self.mode == Mode::Start {
            loop {
                if self.held.starts_with(THINK_START) {
                    self.held.drain(..THINK_START.len());
                } else if THINK_START.starts_with(self.held.as_str()) {
                    // Empty, or a partial `<think>`: wait.
                    return out;
                } else {
                    self.mode = Mode::Reasoning;
                    break;
                }
            }
        }
        let end = self.held.find(THINK_END);
        let tool = self.held.find(TOOL_CALL_START);
        let split = match (end, tool) {
            (Some(e), Some(t)) if t < e => Some((t, t)),
            (Some(e), _) => Some((e, e + THINK_END.len())),
            (None, Some(t)) => Some((t, t)),
            (None, None) => None,
        };
        match split {
            Some((reasoning_end, content_start)) => {
                out.reasoning.push_str(&self.held[..reasoning_end]);
                out.content.push_str(&self.held[content_start..]);
                self.held.clear();
                self.mode = Mode::Content;
            }
            None => {
                let keep = partial_suffix(&self.held, THINK_END)
                    .max(partial_suffix(&self.held, TOOL_CALL_START));
                let emit = self.held.len() - keep;
                out.reasoning.push_str(&self.held[..emit]);
                self.held.drain(..emit);
            }
        }
        out
    }

    /// Ends the stream; held-back text is reasoning.
    pub fn finish(&mut self) -> ReasoningDelta {
        ReasoningDelta {
            reasoning: std::mem::take(&mut self.held),
            content: String::new(),
        }
    }
}

/// Splits a complete output (the reference the streaming parser must equal).
pub fn split_reasoning(text: &str, thinking_enabled: bool) -> ReasoningDelta {
    if !thinking_enabled {
        return ReasoningDelta {
            reasoning: String::new(),
            content: text.to_string(),
        };
    }
    let mut rest = text;
    while let Some(stripped) = rest.strip_prefix(THINK_START) {
        rest = stripped;
    }
    let end = rest.find(THINK_END);
    let tool = rest.find(TOOL_CALL_START);
    let (reasoning, content) = match (end, tool) {
        (Some(e), Some(t)) if t < e => (&rest[..t], &rest[t..]),
        (Some(e), _) => (&rest[..e], &rest[e + THINK_END.len()..]),
        (None, Some(t)) => (&rest[..t], &rest[t..]),
        (None, None) => (rest, ""),
    };
    ReasoningDelta {
        reasoning: reasoning.to_string(),
        content: content.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn split(text: &str) -> (String, String) {
        let d = split_reasoning(text, true);
        (d.reasoning, d.content)
    }

    #[test]
    fn splits_reasoning() {
        assert_eq!(split("<think>a</think>b"), ("a".into(), "b".into()));
        assert_eq!(
            split("<think><think>a</think>b</think>"),
            ("a".into(), "b</think>".into())
        );
        assert_eq!(split("no tag</think>b"), ("no tag".into(), "b".into()));
        assert_eq!(
            split("<think>r<tool_call>x</think>"),
            ("r".into(), "<tool_call>x</think>".into())
        );
        assert_eq!(split("<think>unfinished"), ("unfinished".into(), "".into()));
        assert_eq!(split("<thi"), ("<thi".into(), "".into()));
        assert_eq!(
            split("r <think> r</think>"),
            ("r <think> r".into(), "".into())
        );
        let off = split_reasoning("<think>a</think>b", false);
        assert_eq!(
            (off.reasoning.as_str(), off.content.as_str()),
            ("", "<think>a</think>b")
        );
    }

    const FRAGMENTS: &[&str] = &[
        "<think>",
        "</think>",
        "<tool_call>",
        "<",
        "/",
        "think",
        ">",
        "a",
        " ",
        "中",
        "😀",
        "</th",
        "ink>",
        "<tool",
        "_call>",
        "\n",
    ];

    proptest! {
        #[test]
        fn streaming_equals_complete(
            pieces in proptest::collection::vec(0..FRAGMENTS.len(), 0..30),
            cuts in proptest::collection::vec(0usize..200, 0..12),
            enabled: bool,
        ) {
            let text: String = pieces.iter().map(|&i| FRAGMENTS[i]).collect();
            let mut cuts = cuts;
            cuts.sort();
            let mut parser = ReasoningParser::new(enabled);
            let mut got = ReasoningDelta::default();
            let mut last = 0;
            for cut in cuts.into_iter().chain([text.len()]) {
                if cut >= last && cut <= text.len() && text.is_char_boundary(cut) {
                    let d = parser.push(&text[last..cut]);
                    got.reasoning.push_str(&d.reasoning);
                    got.content.push_str(&d.content);
                    last = cut;
                }
            }
            let d = parser.finish();
            got.reasoning.push_str(&d.reasoning);
            prop_assert_eq!(got, split_reasoning(&text, enabled));
        }
    }
}
