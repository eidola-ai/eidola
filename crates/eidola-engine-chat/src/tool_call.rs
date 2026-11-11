//! Tool-call extraction from MiMo output.
//!
//! MiMo emits calls as compact XML after any visible content:
//! `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>`.
//!
//! The block structure follows SGLang's MiMo detector (`detect_and_parse`):
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
//!
//! What becomes of the rest of the text is [`UnparsedText`]. **Served output
//! loses only its calls** ([`UnparsedText::Content`]): a block with no
//! complete `<function=…>…</function>`, an unclosed trailing block (prose that
//! mentions `<tool_call>`, or output cut off by `max_tokens`) and text between
//! or after blocks are content, except text that is only whitespace next to a
//! call. SGLang drops all three ([`UnparsedText::Sglang`], kept for the
//! differential test).
//!
//! [`ToolCallParser`] produces exactly that result from any split of the text
//! into chunks: content before the first block streams as it arrives (holding
//! back only a possible partial `<tool_call>`), text after a block is held
//! until the next block closes or the stream ends, and each call is emitted
//! whole, with its final typed arguments, when its closing tag arrives. Calls
//! are emitted whole because typing a parameter needs its complete value and
//! deciding whether a block is a call needs its complete structure.
//!
//! All scanning is linear in the input, however many blocks one chunk holds;
//! no input can make the parser panic.

use crate::args::{ToolSchemas, arguments_object};
use crate::json;
use crate::pysem::py_strip;

const CALL_START: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAM_START: &str = "<parameter=";
const PARAM_END: &str = "</parameter>";

/// What becomes of output text that is not a call (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UnparsedText {
    /// Content, except whitespace-only text next to a call: the serving rule.
    #[default]
    Content,
    /// SGLang's: dropped, but for undeclared functions' blocks and the text
    /// before them. For differential tests against SGLang only.
    Sglang,
}

/// Whether the text between two blocks (or between a block and the end) is
/// content under [`UnparsedText::Content`].
fn gap_is_content(gap: &str, after_call: bool, before_call: bool) -> bool {
    !(py_strip(gap).is_empty() && (after_call || before_call))
}

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

/// Parses a complete output in one go.
pub fn parse_complete(text: &str, schemas: &ToolSchemas) -> ParsedToolCalls {
    let Some(first) = text.find(CALL_START) else {
        return ParsedToolCalls {
            content: text.to_string(),
            calls: Vec::new(),
        };
    };
    let sglang = schemas.unparsed_text() == UnparsedText::Sglang;
    let mut out = ParsedToolCalls {
        content: text[..first].to_string(),
        calls: Vec::new(),
    };
    let mut last_end = first;
    let mut after_call = false;
    let mut unclosed = None;
    while let Some(p) = text[last_end..].find(CALL_START) {
        let start = last_end + p;
        let body_start = start + CALL_START.len();
        let Some(e) = text[body_start..].find(CALL_END) else {
            unclosed = Some(start);
            break;
        };
        let end = body_start + e + CALL_END.len();
        let gap = &text[last_end..start];
        match parse_block(&text[body_start..body_start + e], schemas) {
            Block::Call { name, arguments } => {
                if !sglang && gap_is_content(gap, after_call, true) {
                    out.content.push_str(gap);
                }
                out.calls.push((name, arguments));
                after_call = true;
            }
            Block::Malformed if sglang => {}
            Block::Unknown | Block::Malformed => {
                if sglang || gap_is_content(gap, after_call, false) {
                    out.content.push_str(gap);
                }
                out.content.push_str(&text[start..end]);
                after_call = false;
            }
        }
        last_end = end;
    }
    if !sglang {
        let gap_end = unclosed.unwrap_or(text.len());
        let gap = &text[last_end..gap_end];
        if gap_is_content(gap, after_call, false) {
            out.content.push_str(gap);
        }
        out.content.push_str(&text[gap_end..]);
    }
    out
}

/// One step of streaming output.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolCallDelta {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

/// Where the parser is. Positions index [`ToolCallParser::buf`], the text
/// not yet given out or dropped.
enum State {
    /// No `<tool_call>` seen yet; `buf` is a possible partial opening tag.
    Content,
    /// Inside a block that starts at `start`; `buf[..start]` is the text
    /// between the previous block and this one. No closing tag ends before
    /// `scanned`.
    Block { start: usize, scanned: usize },
    /// After a block, waiting for another; `buf` is the text since it. No
    /// opening tag ends before `scanned`.
    Between { scanned: usize },
}

/// Streaming tool-call parser.
pub struct ToolCallParser {
    schemas: ToolSchemas,
    ids: Box<dyn CallIdSource>,
    state: State,
    buf: String,
    /// The last closed block was a call.
    after_call: bool,
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
            state: State::Content,
            buf: String::new(),
            after_call: false,
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
        let sglang = self.schemas.unparsed_text() == UnparsedText::Sglang;
        self.buf.push_str(text);
        let buf = self.buf.as_str();
        // `buf[pos..]` is the text still undecided.
        let mut pos = 0;
        loop {
            match self.state {
                State::Content => {
                    if let Some(p) = buf[pos..].find(CALL_START) {
                        delta.content.push_str(&buf[pos..pos + p]);
                        pos += p;
                        self.state = State::Block {
                            start: pos,
                            scanned: pos + CALL_START.len(),
                        };
                        continue;
                    }
                    let keep = partial_suffix(&buf[pos..], CALL_START);
                    delta.content.push_str(&buf[pos..buf.len() - keep]);
                    pos = buf.len() - keep;
                    break;
                }
                State::Block { start, scanned } => {
                    let Some(e) = buf[scanned..].find(CALL_END) else {
                        // Resume scanning where a closing tag could still start.
                        let mut resume = buf
                            .len()
                            .saturating_sub(CALL_END.len() - 1)
                            .max(start + CALL_START.len());
                        while !buf.is_char_boundary(resume) {
                            resume -= 1;
                        }
                        self.state = State::Block {
                            start,
                            scanned: resume,
                        };
                        break;
                    };
                    let body_end = scanned + e;
                    let end = body_end + CALL_END.len();
                    let gap = &buf[pos..start];
                    match parse_block(&buf[start + CALL_START.len()..body_end], &self.schemas) {
                        Block::Call { name, arguments } => {
                            if !sglang && gap_is_content(gap, self.after_call, true) {
                                delta.content.push_str(gap);
                            }
                            let index = self.emitted;
                            self.emitted += 1;
                            delta.calls.push(ToolCall {
                                index,
                                id: self.ids.call_id(index),
                                name,
                                arguments,
                            });
                            self.after_call = true;
                        }
                        Block::Malformed if sglang => {}
                        Block::Unknown | Block::Malformed => {
                            if sglang || gap_is_content(gap, self.after_call, false) {
                                delta.content.push_str(gap);
                            }
                            delta.content.push_str(&buf[start..end]);
                            self.after_call = false;
                        }
                    }
                    pos = end;
                    self.state = State::Between { scanned: end };
                }
                State::Between { scanned } => {
                    if let Some(p) = buf[scanned..].find(CALL_START) {
                        let start = scanned + p;
                        self.state = State::Block {
                            start,
                            scanned: start + CALL_START.len(),
                        };
                        continue;
                    }
                    self.state = State::Between {
                        scanned: buf.len() - partial_suffix(&buf[pos..], CALL_START),
                    };
                    break;
                }
            }
        }
        // Positions become relative to the text kept.
        self.buf.drain(..pos);
        match &mut self.state {
            State::Content => {}
            State::Block { start, scanned } => {
                *start -= pos;
                *scanned -= pos;
            }
            State::Between { scanned } => *scanned -= pos,
        }
        delta
    }

    /// Whether no `<tool_call>` block has begun: the state a parser starts in,
    /// and so the only one a parser resumed from the content given out so far
    /// would be in.
    pub fn before_any_block(&self) -> bool {
        matches!(self.state, State::Content)
    }

    /// Ends the stream, giving out the text held back ([`UnparsedText`]): a
    /// possible partial tag before any block, and an unfinished block or the
    /// text after the last block.
    pub fn finish(&mut self) -> ToolCallDelta {
        let state = std::mem::replace(&mut self.state, State::Content);
        let buf = std::mem::take(&mut self.buf);
        let sglang = self.schemas.unparsed_text() == UnparsedText::Sglang;
        let content = match state {
            State::Content => buf,
            State::Block { .. } | State::Between { .. } if sglang => String::new(),
            State::Block { start, .. } => {
                let gap = &buf[..start];
                if gap_is_content(gap, self.after_call, false) {
                    buf
                } else {
                    buf[start..].to_string()
                }
            }
            State::Between { .. } if gap_is_content(&buf, self.after_call, false) => buf,
            State::Between { .. } => String::new(),
        };
        ToolCallDelta {
            content,
            calls: Vec::new(),
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
            "agap<tool_call><function=nope><parameter=x>1</parameter></function></tool_call>tail<tool_call>no function</tool_call>end<tool_call><function=run>"
        );
        let sglang = parse_complete(text, &schemas().with_unparsed_text(UnparsedText::Sglang));
        assert_eq!(sglang.calls, parsed.calls);
        assert_eq!(
            sglang.content,
            "agap<tool_call><function=nope><parameter=x>1</parameter></function></tool_call>"
        );
    }

    /// Prose that mentions the tag, a block that never closes, and text after
    /// the last call all reach the content; whitespace next to a call does not.
    #[test]
    fn text_that_is_not_a_call_is_content() {
        let cases: &[(&str, &str, usize)] = &[
            (
                "Write <tool_call> to call a tool, then the rest of the answer.",
                "Write <tool_call> to call a tool, then the rest of the answer.",
                0,
            ),
            (
                "x<tool_call>no function</tool_call> and more",
                "x<tool_call>no function</tool_call> and more",
                0,
            ),
            (
                "<tool_call><function=run></function></tool_call>\nDone; next <tool_call><function=run>",
                "\nDone; next <tool_call><function=run>",
                1,
            ),
            (
                "<tool_call><function=run></function></tool_call> then <tool_call><function=run></function></tool_call>",
                " then ",
                2,
            ),
            (
                "Ok.\n<tool_call><function=run></function></tool_call>\n<tool_call><function=run></function></tool_call>\n",
                "Ok.\n",
                2,
            ),
            (
                "<tool_call><function=run></function></tool_call>\n<tool_call>bad</tool_call>",
                "<tool_call>bad</tool_call>",
                1,
            ),
        ];
        for &(text, content, calls) in cases {
            let parsed = parse_complete(text, &schemas());
            assert_eq!(
                (parsed.content.as_str(), parsed.calls.len()),
                (content, calls),
                "{text:?}"
            );
            // Streamed one character at a time, too.
            let cuts: Vec<usize> = (0..text.len()).collect();
            assert_eq!(stream(text, &cuts), parsed, "{text:?}");
        }
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
