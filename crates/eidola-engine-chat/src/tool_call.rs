//! Tool-call extraction from MiMo output.
//!
//! MiMo emits calls as compact XML after any visible content:
//! `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`.
//!
//! The meaning of a complete output follows SGLang's MiMo detector
//! (`detect_and_parse`):
//!
//! - **Content** is the text before the first `<tool_call>`.
//! - Each `<tool_call>…</tool_call>` block (the first closing tag after the
//!   opening one) is parsed: the first `<function=NAME>…</function>` inside
//!   it names the call (`NAME` is everything up to the next `>`, trimmed of
//!   whitespace), and each `<parameter=K>V</parameter>` inside that function
//!   body is a parameter (`K` trimmed, `V` verbatim). A repeated parameter
//!   keeps its first position and takes its last value. Values are typed by
//!   the tool schema ([`crate::args`]).
//! - A block naming an undeclared function is not a call: the block, and any
//!   text between it and the previous block, is appended to the content.
//! - A block with no complete `<function=…>…</function>` is dropped.
//! - Text between or after blocks is otherwise dropped, as is an unclosed
//!   trailing block (e.g. output cut off by `max_tokens`).
//!
//! [`ToolCallParser`] produces exactly that result from any split of the text
//! into chunks: content streams as it arrives (holding back only a possible
//! partial `<tool_call>`), and each call is emitted whole, with its final
//! typed arguments, when its closing tag arrives. Calls are emitted whole
//! because typing a parameter needs its complete value and deciding whether
//! a block is a call needs its complete structure.
//!
//! All scanning is linear in the input; no input can make the parser panic.

use crate::args::{ToolSchemas, arguments_object};
use crate::json;
use crate::pysem::py_strip;

const CALL_START: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAM_START: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";

/// A parsed call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    /// Position among this response's calls (OpenAI's `index`).
    pub index: usize,
    /// OpenAI's `id`.
    pub id: String,
    pub name: String,
    /// The arguments object as JSON text (Python `json.dumps` layout).
    pub arguments: String,
}

impl ToolCall {
    /// The OpenAI `tool_calls[]` entry (also valid as a streaming delta).
    pub fn to_openai(&self) -> serde_json::Value {
        serde_json::json!({
            "index": self.index,
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        })
    }
}

/// Source of tool-call ids.
pub trait CallIdSource: Send {
    fn call_id(&mut self, index: usize) -> String;
}

/// Ids of the form `{prefix}{index}`. The caller supplies a prefix that is
/// unique per response (e.g. `call_` plus random characters), so ids never
/// repeat across a conversation.
#[derive(Clone, Debug)]
pub struct PrefixedCallIds(pub String);

impl CallIdSource for PrefixedCallIds {
    fn call_id(&mut self, index: usize) -> String {
        format!("{}{index}", self.0)
    }
}

impl<F: FnMut(usize) -> String + Send> CallIdSource for F {
    fn call_id(&mut self, index: usize) -> String {
        self(index)
    }
}

/// What one block of output means.
enum Block {
    Call { name: String, arguments: String },
    Unknown,
    Malformed,
}

/// Parses the body between `<tool_call>` and `</tool_call>`.
fn parse_block(body: &str, schemas: &ToolSchemas) -> Block {
    let Some((name, function_body)) = find_function(body) else {
        return Block::Malformed;
    };
    let name = py_strip(name);
    if name.is_empty() || !schemas.contains(name) {
        return Block::Unknown;
    }
    let params = find_parameters(function_body).into_iter().map(|(k, v)| {
        let k = py_strip(k).to_string();
        let value = schemas.convert(name, &k, v);
        (k, value)
    });
    Block::Call {
        name: name.to_string(),
        arguments: json::dumps_default(&arguments_object(params)),
    }
}

/// `re.search(r"<function=([^>]+)>(.*?)</function>", body, re.DOTALL)`.
fn find_function(body: &str) -> Option<(&str, &str)> {
    let mut from = 0;
    while let Some(p) = body[from..].find(FUNCTION_START) {
        let name_start = from + p + FUNCTION_START.len();
        let gt = body[name_start..].find('>')?;
        if gt == 0 {
            // `[^>]+` needs a character; retry at the next occurrence.
            from = from + p + 1;
            continue;
        }
        let value_start = name_start + gt + 1;
        let end = body[value_start..].find(FUNCTION_END)?;
        return Some((
            &body[name_start..name_start + gt],
            &body[value_start..value_start + end],
        ));
    }
    None
}

/// `re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", body, re.DOTALL)`.
fn find_parameters(body: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(p) = body[from..].find(PARAM_START) {
        let start = from + p;
        let name_start = start + PARAM_START.len();
        let Some(gt) = body[name_start..].find('>') else {
            break;
        };
        if gt == 0 {
            from = start + 1;
            continue;
        }
        let value_start = name_start + gt + 1;
        let Some(end) = body[value_start..].find(PARAM_END) else {
            break;
        };
        out.push((
            &body[name_start..name_start + gt],
            &body[value_start..value_start + end],
        ));
        from = value_start + end + PARAM_END.len();
    }
    out
}

/// The complete-output result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedToolCalls {
    pub content: String,
    /// `(name, arguments)` pairs in order.
    pub calls: Vec<(String, String)>,
}

/// Parses a complete output in one go (SGLang's `detect_and_parse`).
pub fn parse_complete(text: &str, schemas: &ToolSchemas) -> ParsedToolCalls {
    let Some(first) = text.find(CALL_START) else {
        return ParsedToolCalls {
            content: text.to_string(),
            calls: Vec::new(),
        };
    };
    let mut out = ParsedToolCalls {
        content: text[..first].to_string(),
        calls: Vec::new(),
    };
    let mut last_end = first;
    let mut search = first;
    while let Some(p) = text[search..].find(CALL_START) {
        let start = search + p;
        let body_start = start + CALL_START.len();
        let Some(e) = text[body_start..].find(CALL_END) else {
            break;
        };
        let end = body_start + e + CALL_END.len();
        match parse_block(&text[body_start..body_start + e], schemas) {
            Block::Call { name, arguments } => out.calls.push((name, arguments)),
            Block::Unknown => out.content.push_str(&text[last_end..end]),
            Block::Malformed => {}
        }
        last_end = end;
        search = end;
    }
    out
}

/// One step of streaming output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

enum State {
    /// No `<tool_call>` seen yet; `held` is a possible partial opening tag.
    Content { held: String },
    /// Inside a block; `buf` starts with `<tool_call>` and `gap` is the text
    /// between the previous block and this one.
    Block {
        buf: String,
        gap: String,
        scanned: usize,
    },
    /// After a block, waiting for another.
    Between { buf: String, scanned: usize },
}

/// Streaming tool-call parser.
pub struct ToolCallParser {
    schemas: ToolSchemas,
    ids: Box<dyn CallIdSource>,
    state: State,
    emitted: usize,
}

impl std::fmt::Debug for ToolCallParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallParser")
            .field("emitted", &self.emitted)
            .finish_non_exhaustive()
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`.
pub(crate) fn partial_suffix(text: &str, tag: &str) -> usize {
    let max = text.len().min(tag.len() - 1);
    (1..=max)
        .rev()
        .find(|&n| {
            text.is_char_boundary(text.len() - n) && tag.starts_with(&text[text.len() - n..])
        })
        .unwrap_or(0)
}

impl ToolCallParser {
    pub fn new(schemas: ToolSchemas, ids: impl CallIdSource + 'static) -> ToolCallParser {
        ToolCallParser {
            schemas,
            ids: Box::new(ids),
            state: State::Content {
                held: String::new(),
            },
            emitted: 0,
        }
    }

    /// Number of calls emitted so far.
    pub fn calls_emitted(&self) -> usize {
        self.emitted
    }

    /// Feeds a chunk of content text.
    pub fn push(&mut self, text: &str) -> ToolCallDelta {
        let mut delta = ToolCallDelta::default();
        let mut input = text.to_string();
        loop {
            match &mut self.state {
                State::Content { held } => {
                    held.push_str(&input);
                    if let Some(p) = held.find(CALL_START) {
                        delta.content.push_str(&held[..p]);
                        let buf = held[p..].to_string();
                        self.state = State::Block {
                            buf,
                            gap: String::new(),
                            scanned: CALL_START.len(),
                        };
                        input = String::new();
                        continue;
                    }
                    let keep = partial_suffix(held, CALL_START);
                    delta.content.push_str(&held[..held.len() - keep]);
                    *held = held[held.len() - keep..].to_string();
                    return delta;
                }
                State::Block { buf, gap, scanned } => {
                    buf.push_str(&input);
                    input = String::new();
                    let from = (*scanned).max(CALL_START.len());
                    let Some(e) = buf[from..].find(CALL_END) else {
                        // Resume scanning where a closing tag could still start.
                        *scanned = buf
                            .len()
                            .saturating_sub(CALL_END.len() - 1)
                            .max(CALL_START.len());
                        while !buf.is_char_boundary(*scanned) {
                            *scanned -= 1;
                        }
                        return delta;
                    };
                    let body_end = from + e;
                    let end = body_end + CALL_END.len();
                    match parse_block(&buf[CALL_START.len()..body_end], &self.schemas) {
                        Block::Call { name, arguments } => {
                            let index = self.emitted;
                            self.emitted += 1;
                            delta.calls.push(ToolCall {
                                index,
                                id: self.ids.call_id(index),
                                name,
                                arguments,
                            });
                        }
                        Block::Unknown => {
                            delta.content.push_str(gap);
                            delta.content.push_str(&buf[..end]);
                        }
                        Block::Malformed => {}
                    }
                    let rest = buf[end..].to_string();
                    self.state = State::Between {
                        buf: rest,
                        scanned: 0,
                    };
                }
                State::Between { buf, scanned } => {
                    buf.push_str(&input);
                    input = String::new();
                    if let Some(p) = buf[*scanned..].find(CALL_START) {
                        let start = *scanned + p;
                        let gap = buf[..start].to_string();
                        let block = buf[start..].to_string();
                        self.state = State::Block {
                            buf: block,
                            gap,
                            scanned: CALL_START.len(),
                        };
                        continue;
                    }
                    *scanned = buf.len() - partial_suffix(buf, CALL_START);
                    return delta;
                }
            }
        }
    }

    /// Ends the stream. A held-back partial tag before any block is content;
    /// an unfinished block and trailing text after blocks are dropped.
    pub fn finish(&mut self) -> ToolCallDelta {
        let state = std::mem::replace(
            &mut self.state,
            State::Content {
                held: String::new(),
            },
        );
        match state {
            State::Content { held } => ToolCallDelta {
                content: held,
                calls: Vec::new(),
            },
            State::Block { .. } | State::Between { .. } => ToolCallDelta::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn schemas() -> ToolSchemas {
        ToolSchemas::from_tools(
            &json::parse(
                r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}},
                    {"type":"function","function":{"name":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"},"opts":{"type":"object"}}}}}]"#,
            )
            .unwrap(),
        )
    }

    fn stream(text: &str, cuts: &[usize]) -> ParsedToolCalls {
        let mut parser = ToolCallParser::new(schemas(), PrefixedCallIds("call_".into()));
        let mut out = ParsedToolCalls::default();
        let mut last = 0;
        let collect = |d: ToolCallDelta, out: &mut ParsedToolCalls| {
            out.content.push_str(&d.content);
            out.calls
                .extend(d.calls.into_iter().map(|c| (c.name, c.arguments)));
        };
        for &cut in cuts {
            if cut > last && cut < text.len() && text.is_char_boundary(cut) {
                collect(parser.push(&text[last..cut]), &mut out);
                last = cut;
            }
        }
        collect(parser.push(&text[last..]), &mut out);
        collect(parser.finish(), &mut out);
        out
    }

    #[test]
    fn parses_calls_and_content() {
        let text = "Sure.<tool_call><function=get_weather><parameter=city>Paris</parameter><parameter=days>3</parameter></function></tool_call>\n<tool_call><function=run><parameter=cmd>ls</parameter><parameter=opts>{\"a\": true}</parameter></function></tool_call>";
        let parsed = parse_complete(text, &schemas());
        assert_eq!(parsed.content, "Sure.");
        assert_eq!(
            parsed.calls,
            vec![
                (
                    "get_weather".into(),
                    r#"{"city": "Paris", "days": 3}"#.into()
                ),
                ("run".into(), r#"{"cmd": "ls", "opts": {"a": true}}"#.into()),
            ]
        );
    }

    #[test]
    fn unknown_and_malformed_blocks() {
        let text = "a<tool_call><function=get_weather></function></tool_call>gap<tool_call><function=nope><parameter=x>1</parameter></function></tool_call>tail<tool_call>no function</tool_call>end<tool_call><function=run>";
        let parsed = parse_complete(text, &schemas());
        assert_eq!(parsed.calls, vec![("get_weather".into(), "{}".into())]);
        assert_eq!(
            parsed.content,
            "agap<tool_call><function=nope><parameter=x>1</parameter></function></tool_call>"
        );
    }

    #[test]
    fn ids_and_indices_are_stable() {
        let mut parser = ToolCallParser::new(schemas(), PrefixedCallIds("call_x".into()));
        let block = "<tool_call><function=run><parameter=cmd>a</parameter></function></tool_call>";
        let d1 = parser.push(block);
        let d2 = parser.push(block);
        assert_eq!((d1.calls[0].index, d1.calls[0].id.as_str()), (0, "call_x0"));
        assert_eq!((d2.calls[0].index, d2.calls[0].id.as_str()), (1, "call_x1"));
        assert_eq!(d2.calls[0].to_openai()["function"]["name"], "run");
    }

    const FRAGMENTS: &[&str] = &[
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
        "get_weather",
        "run",
        "nope",
        ">",
        "<",
        "city",
        "cmd",
        "days",
        "opts",
        "Paris",
        "3",
        " ",
        "\n",
        "{\"a\": 1}",
        "[1, 2",
        "&amp;",
        "null",
        "中文",
        "😀",
        "<tool_",
        "call>",
        "</tool",
        "=",
        "1e400",
        "[-1e400]",
        "{\"x\": 1e999}",
    ];

    proptest! {
        #[test]
        fn streaming_equals_complete_for_any_split(
            pieces in proptest::collection::vec(0..FRAGMENTS.len(), 0..40),
            cuts in proptest::collection::vec(0usize..400, 0..20),
        ) {
            let text: String = pieces.iter().map(|&i| FRAGMENTS[i]).collect();
            let mut cuts = cuts;
            cuts.sort();
            let streamed = stream(&text, &cuts);
            prop_assert_eq!(&streamed, &parse_complete(&text, &schemas()));
            // Every `arguments` is strict JSON (no `Infinity` from an overflowing
            // number) and an object.
            for (_, arguments) in streamed.calls {
                prop_assert!(matches!(json::parse_strict(&arguments), Ok(json::Json::Object(_))), "{arguments}");
            }
        }

        /// Overflowing numbers in every non-string parameter, streamed whole.
        #[test]
        fn overflowing_parameters_stay_strict_json(
            days in proptest::sample::select(vec!["1e400", "-1e400", "[1e400]", "{\"a\": [1e999]}"]),
            opts in proptest::sample::select(vec!["1e400", "{\"a\": -1e400}", "[1, 1e400]"]),
        ) {
            let text = format!(
                "<tool_call><function=get_weather><parameter=days>{days}</parameter></function></tool_call>\
                 <tool_call><function=run><parameter=opts>{opts}</parameter></function></tool_call>"
            );
            let parsed = parse_complete(&text, &schemas());
            prop_assert_eq!(parsed.calls.len(), 2);
            for (_, arguments) in parsed.calls {
                prop_assert!(matches!(json::parse_strict(&arguments), Ok(json::Json::Object(_))), "{arguments}");
                prop_assert!(serde_json::from_str::<serde_json::Value>(&arguments).is_ok(), "{arguments}");
            }
        }

        #[test]
        fn arbitrary_text_never_panics_and_yields_json_objects(text in ".{0,300}", cuts in proptest::collection::vec(0usize..300, 0..10)) {
            let mut cuts = cuts;
            cuts.sort();
            let streamed = stream(&text, &cuts);
            prop_assert_eq!(&streamed, &parse_complete(&text, &schemas()));
            for (_, arguments) in streamed.calls {
                prop_assert!(matches!(json::parse_strict(&arguments), Ok(json::Json::Object(_))));
            }
        }
    }
}
