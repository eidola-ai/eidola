//! Splits DeepSeek-V4.1 output into reasoning, content and tool calls.
//!
//! A thinking-mode prompt ends with `<think>` and a chat-mode prompt with
//! `</think>`, so the model writes
//!
//! ```text
//! {reasoning}</think>{content}
//!
//! <｜DSML｜ calls>
//! <｜DSML｜ invoke name="NAME">
//! <｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>
//! </｜DSML｜ invoke>
//! </｜DSML｜ calls>
//! ```
//!
//! with the reasoning and its `</think>` only in thinking mode. The
//! reference's `parse_message_from_completion_text` reads exactly that and
//! raises on anything else; these parsers give the same reasoning, content and
//! calls for output it accepts, and for the rest follow the rules MiMo's
//! parsers follow, so that **the answer loses only its calls**.
//!
//! # Reasoning ([`ReasoningSplitter`])
//!
//! In thinking mode the output starts in reasoning, which ends at the first
//! `</think>` (dropped) or the first `<｜DSML｜ calls>` (kept as content, with
//! the blank line before it), whichever comes first; output that ends inside
//! reasoning is all reasoning. In chat mode everything is content.
//!
//! # Tool calls ([`DsmlParser`])
//!
//! - **Content** is the text before the first `<｜DSML｜ calls>`.
//! - Each `<｜DSML｜ calls>…</｜DSML｜ calls>` block (to the first closing tag)
//!   holds the calls of one step. It is read as the reference writes it:
//!   `invoke` elements, each a `name="NAME">` header and `parameter` elements
//!   ` name="K" string="true|false">V`, with only whitespace between
//!   elements. A block is its calls when every element is a well-formed
//!   call of a declared tool; the blank line the format puts before the block
//!   goes with it, and a block of no elements is no calls, dropped as the
//!   reference drops it.
//! - Any other block is content, as are an unclosed block and text between or
//!   after blocks, except text that is only whitespace next to a block of
//!   calls.
//!
//! **Arguments** are a JSON object in the reference's layout
//! (`{"k": v, "k2": v2}`): a `string="true"` value is the string it spells, and
//! a `string="false"` value is spliced in as written when it is strict JSON
//! and degrades to a string otherwise, so `arguments` is always valid JSON. A
//! block with a repeated parameter name is not calls.
//!
//! Both parsers give the same result for any split of the text into chunks,
//! scan linearly, and cannot panic.
//!
//! **Format limits**: a string value containing `</｜DSML｜ parameter>`, or a
//! parameter name containing `" string="true">`, cannot be represented.

use std::collections::HashSet;

use super::{
    CALLS_CLOSE, CALLS_OPEN, CALLS_SEPARATOR, INVOKE_CLOSE, INVOKE_OPEN, PARAMETER_CLOSE,
    PARAMETER_OPEN, THINK_END,
};
use crate::args::ToolSchemas;
use crate::json;
use crate::pysem::{is_py_space, py_strip};
use crate::reasoning::ReasoningDelta;
use crate::tool_call::{CallIdSource, ParsedToolCalls, ToolCall, ToolCallDelta, partial_suffix};

/// The longest suffix of `text` that may yet become the opening of a calls
/// block, the blank line before it included.
fn partial_calls_open(text: &str) -> usize {
    const SEPARATED_OPEN: &str = "\n\n<｜DSML｜ calls>";
    partial_suffix(text, SEPARATED_OPEN).max(partial_suffix(text, CALLS_OPEN))
}

/// Where the region of a calls block opening at `open` starts in
/// `text[from..]`: at the blank line before it, when there is one.
fn region_start(text: &str, from: usize, open: usize) -> usize {
    if text[from..open].ends_with(CALLS_SEPARATOR) {
        open - CALLS_SEPARATOR.len()
    } else {
        open
    }
}

/// Streaming reasoning splitter (module docs).
#[derive(Clone, Debug)]
pub struct ReasoningSplitter {
    in_reasoning: bool,
    held: String,
}

impl ReasoningSplitter {
    /// A splitter for the output of a prompt in thinking or chat mode.
    pub fn new(thinking: bool) -> ReasoningSplitter {
        ReasoningSplitter {
            in_reasoning: thinking,
            held: String::new(),
        }
    }

    pub fn in_reasoning(&self) -> bool {
        self.in_reasoning
    }

    pub fn push(&mut self, text: &str) -> ReasoningDelta {
        let mut out = ReasoningDelta::default();
        if !self.in_reasoning {
            out.content.push_str(text);
            return out;
        }
        self.held.push_str(text);
        match reasoning_end(&self.held) {
            Some((reasoning_end, content_start)) => {
                out.reasoning.push_str(&self.held[..reasoning_end]);
                out.content.push_str(&self.held[content_start..]);
                self.held.clear();
                self.in_reasoning = false;
            }
            None => {
                let keep =
                    partial_suffix(&self.held, THINK_END).max(partial_calls_open(&self.held));
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

/// Where the reasoning at the start of `text` ends and the content begins.
fn reasoning_end(text: &str) -> Option<(usize, usize)> {
    let end = text.find(THINK_END);
    let calls = text
        .find(CALLS_OPEN)
        .map(|open| region_start(text, 0, open));
    match (end, calls) {
        (Some(e), Some(c)) if c < e => Some((c, c)),
        (Some(e), _) => Some((e, e + THINK_END.len())),
        (None, Some(c)) => Some((c, c)),
        (None, None) => None,
    }
}

/// Splits a complete output (the reference the streaming splitter must equal).
pub fn split_reasoning(text: &str, thinking: bool) -> ReasoningDelta {
    let (reasoning, content) = match reasoning_end(text) {
        _ if !thinking => ("", text),
        Some((reasoning_end, content_start)) => (&text[..reasoning_end], &text[content_start..]),
        None => (text, ""),
    };
    ReasoningDelta {
        reasoning: reasoning.to_string(),
        content: content.to_string(),
    }
}

/// The first of `needles` in `text`, all of which start with `<`: its position
/// and index. One pass over the text, whichever needle is absent.
fn find_first(text: &str, needles: &[&str]) -> Option<(usize, usize)> {
    text.match_indices('<').find_map(|(at, _)| {
        needles
            .iter()
            .position(|needle| text[at..].starts_with(needle))
            .map(|which| (at, which))
    })
}

fn skip_space(text: &str) -> &str {
    text.trim_start_matches(is_py_space)
}

/// The calls of one block's body, `(name, arguments)` each; `None` unless it
/// is nothing but well-formed calls of declared tools.
fn parse_block(body: &str, schemas: &ToolSchemas) -> Option<Vec<(String, String)>> {
    let mut calls = Vec::new();
    let mut rest = skip_space(body);
    while !rest.is_empty() {
        rest = rest.strip_prefix(INVOKE_OPEN)?;
        let (stop, _) = find_first(rest, &[PARAMETER_OPEN, INVOKE_CLOSE])?;
        let name = py_strip(&rest[..stop])
            .strip_prefix("name=\"")?
            .strip_suffix("\">")?;
        if !schemas.contains(name) {
            return None;
        }
        rest = &rest[stop..];
        let mut arguments = String::from("{");
        let mut names: HashSet<&str> = HashSet::new();
        while let Some(parameter) = rest.strip_prefix(PARAMETER_OPEN) {
            let end = parameter.find(PARAMETER_CLOSE)?;
            let after = &parameter[end + PARAMETER_CLOSE.len()..];
            let parameter = parameter[..end].strip_prefix(" name=\"")?;
            let (key, is_string, value) =
                parameter
                    .match_indices("\" string=\"")
                    .find_map(|(at, marker)| {
                        let after = &parameter[at + marker.len()..];
                        let flag = [("true\">", true), ("false\">", false)]
                            .into_iter()
                            .find(|(flag, _)| after.starts_with(flag))?;
                        Some((&parameter[..at], flag.1, &after[flag.0.len()..]))
                    })?;
            if !names.insert(key) {
                return None;
            }
            if names.len() > 1 {
                arguments.push_str(", ");
            }
            json::write_string(&mut arguments, key, false);
            arguments.push_str(": ");
            if !is_string && json::parse_strict(value).is_ok() {
                arguments.push_str(value);
            } else {
                json::write_string(&mut arguments, value, false);
            }
            rest = skip_space(after);
        }
        arguments.push('}');
        rest = skip_space(rest.strip_prefix(INVOKE_CLOSE)?);
        calls.push((name.to_string(), arguments));
    }
    Some(calls)
}

/// Whether the text between two blocks (or between a block and the end) is
/// content: all of it, unless it is only whitespace next to a block of calls.
fn gap_is_content(gap: &str, after_calls: bool, before_calls: bool) -> bool {
    !(py_strip(gap).is_empty() && (after_calls || before_calls))
}

/// Parses a complete output in one go.
pub fn parse_complete(text: &str, schemas: &ToolSchemas) -> ParsedToolCalls {
    let mut out = ParsedToolCalls::default();
    // `text[decided..]` has not been given out or dropped.
    let (mut decided, mut scan) = (0, 0);
    let mut after_calls = false;
    let mut first = true;
    while let Some(p) = text[scan..].find(CALLS_OPEN) {
        let open = scan + p;
        let start = region_start(text, decided, open);
        if first {
            out.content.push_str(&text[..start]);
            decided = start;
            first = false;
        }
        let body_start = open + CALLS_OPEN.len();
        let Some(e) = text[body_start..].find(CALLS_CLOSE) else {
            let gap = &text[decided..start];
            if gap_is_content(gap, after_calls, false) {
                out.content.push_str(gap);
            }
            out.content.push_str(&text[start..]);
            return out;
        };
        let end = body_start + e + CALLS_CLOSE.len();
        let gap = &text[decided..start];
        match parse_block(&text[body_start..body_start + e], schemas) {
            Some(calls) => {
                if gap_is_content(gap, after_calls, true) {
                    out.content.push_str(gap);
                }
                out.calls.extend(calls);
                after_calls = true;
            }
            None => {
                if gap_is_content(gap, after_calls, false) {
                    out.content.push_str(gap);
                }
                out.content.push_str(&text[start..end]);
                after_calls = false;
            }
        }
        (decided, scan) = (end, end);
    }
    let gap = &text[decided..];
    if first || gap_is_content(gap, after_calls, false) {
        out.content.push_str(gap);
    }
    out
}

/// Where the parser is. Positions index [`DsmlParser::buf`], the text not yet
/// given out or dropped.
enum State {
    /// No calls block seen yet; `buf` is a possible partial opening.
    Content,
    /// Inside a block whose region starts at `start` and whose opening tag is
    /// at `open`; `buf[..start]` is the text since the previous block. No
    /// closing tag ends before `scanned`.
    Block {
        start: usize,
        open: usize,
        scanned: usize,
    },
    /// After a block, waiting for another; `buf` is the text since it. No
    /// opening tag starts before `scanned`.
    Between { scanned: usize },
}

/// Streaming tool-call parser (module docs).
pub struct DsmlParser {
    schemas: ToolSchemas,
    ids: Box<dyn CallIdSource>,
    state: State,
    buf: String,
    /// The last closed block was calls.
    after_calls: bool,
    emitted: usize,
}

impl std::fmt::Debug for DsmlParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DsmlParser")
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

impl DsmlParser {
    /// A parser recognizing calls to the tools in `schemas`
    /// ([`super::tool_schemas`]).
    pub fn new(schemas: ToolSchemas, ids: impl CallIdSource + 'static) -> DsmlParser {
        DsmlParser {
            schemas,
            ids: Box::new(ids),
            state: State::Content,
            buf: String::new(),
            after_calls: false,
            emitted: 0,
        }
    }

    /// Number of calls emitted so far.
    pub fn calls_emitted(&self) -> usize {
        self.emitted
    }

    /// Whether no calls block has begun.
    pub fn before_any_block(&self) -> bool {
        matches!(self.state, State::Content)
    }

    /// Feeds a chunk of content text.
    pub fn push(&mut self, text: &str) -> ToolCallDelta {
        let mut delta = ToolCallDelta::default();
        self.buf.push_str(text);
        let buf = self.buf.as_str();
        // `buf[pos..]` is the text still undecided.
        let mut pos = 0;
        loop {
            match self.state {
                State::Content => {
                    if let Some(p) = buf[pos..].find(CALLS_OPEN) {
                        let open = pos + p;
                        let start = region_start(buf, pos, open);
                        delta.content.push_str(&buf[pos..start]);
                        pos = start;
                        self.state = State::Block {
                            start,
                            open,
                            scanned: open + CALLS_OPEN.len(),
                        };
                        continue;
                    }
                    let keep = partial_calls_open(&buf[pos..]);
                    delta.content.push_str(&buf[pos..buf.len() - keep]);
                    pos = buf.len() - keep;
                    break;
                }
                State::Block {
                    start,
                    open,
                    scanned,
                } => {
                    let Some(e) = buf[scanned..].find(CALLS_CLOSE) else {
                        // Resume scanning where a closing tag could still start.
                        let mut resume = buf
                            .len()
                            .saturating_sub(CALLS_CLOSE.len() - 1)
                            .max(open + CALLS_OPEN.len());
                        while !buf.is_char_boundary(resume) {
                            resume -= 1;
                        }
                        self.state = State::Block {
                            start,
                            open,
                            scanned: resume.max(scanned),
                        };
                        break;
                    };
                    let body_end = scanned + e;
                    let end = body_end + CALLS_CLOSE.len();
                    let gap = &buf[pos..start];
                    match parse_block(&buf[open + CALLS_OPEN.len()..body_end], &self.schemas) {
                        Some(calls) => {
                            if gap_is_content(gap, self.after_calls, true) {
                                delta.content.push_str(gap);
                            }
                            for (name, arguments) in calls {
                                let index = self.emitted;
                                self.emitted += 1;
                                delta.calls.push(ToolCall {
                                    index,
                                    id: self.ids.call_id(index),
                                    name,
                                    arguments,
                                });
                            }
                            self.after_calls = true;
                        }
                        None => {
                            if gap_is_content(gap, self.after_calls, false) {
                                delta.content.push_str(gap);
                            }
                            delta.content.push_str(&buf[start..end]);
                            self.after_calls = false;
                        }
                    }
                    pos = end;
                    self.state = State::Between { scanned: end };
                }
                State::Between { scanned } => {
                    if let Some(p) = buf[scanned..].find(CALLS_OPEN) {
                        let open = scanned + p;
                        self.state = State::Block {
                            start: region_start(buf, pos, open),
                            open,
                            scanned: open + CALLS_OPEN.len(),
                        };
                        continue;
                    }
                    self.state = State::Between {
                        scanned: buf.len() - partial_suffix(&buf[scanned..], CALLS_OPEN),
                    };
                    break;
                }
            }
        }
        // Positions become relative to the text kept.
        self.buf.drain(..pos);
        match &mut self.state {
            State::Content => {}
            State::Block {
                start,
                open,
                scanned,
            } => {
                *start -= pos;
                *open -= pos;
                *scanned -= pos;
            }
            State::Between { scanned } => *scanned -= pos,
        }
        delta
    }

    /// Ends the stream, giving out the text held back: a possible partial
    /// opening before any block, and an unfinished block or the text after
    /// the last block.
    pub fn finish(&mut self) -> ToolCallDelta {
        let state = std::mem::replace(&mut self.state, State::Content);
        let buf = std::mem::take(&mut self.buf);
        let content = match state {
            State::Content => buf,
            State::Block { start, .. } => {
                if gap_is_content(&buf[..start], self.after_calls, false) {
                    buf
                } else {
                    buf[start..].to_string()
                }
            }
            State::Between { .. } if gap_is_content(&buf, self.after_calls, false) => buf,
            State::Between { .. } => String::new(),
        };
        ToolCallDelta {
            content,
            calls: Vec::new(),
        }
    }
}
