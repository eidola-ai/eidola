//! The DeepSeek-V4.1 prompt encoder: `encode_messages` of the model
//! repository's `encoding/encoding.py`, byte for byte.
//!
//! # What the reference does
//!
//! - **Tool messages have no turn of their own.** Each becomes a
//!   `<tool_result>` block of a user turn, joining the user turn before it or
//!   starting one; consecutive user messages merge into one turn too (unless
//!   the earlier one carries a `task`). A turn's results are put in the order
//!   of the calls in the assistant message before it.
//! - **Request-level `tools` belong to a system message.** The reference reads
//!   tools off messages; the request's `tools` are attached as the model's
//!   `chat_template.jinja` attaches them: to a leading system message that has
//!   none of its own, else to a new empty system message put first.
//! - **The reasoning effort** is written once, at the start of a thinking-mode
//!   prompt, as a budget from 1 to 100.
//! - **Earlier reasoning is dropped** up to the last user turn (a system
//!   message after the first counts as one, a turn of tool results alone does
//!   not), unless any message declares tools, which keeps all of it.
//! - **The prompt ends** with `<｜Assistant｜><think>` in thinking mode and
//!   `<｜Assistant｜></think>` in chat mode when the last message is a user or
//!   mid-conversation system message.
//!
//! # Accepted input
//!
//! The reference is Python over untyped dicts, and prints Python's own
//! spellings (`None`, a list's `repr`) or raises for input it was not written
//! for. [`encode`] renders the domain where the reference's output is a
//! function of the JSON alone and refuses the rest: roles `system`, `user`,
//! `assistant`, `tool` and `latest_reminder`; `content` a string, absent,
//! `null` (not for `user`, `tool` or `latest_reminder`), or an array of
//! objects with a string `type`; text parts with a string `text`; tool calls
//! with a `function` object naming a string `name` and carrying `arguments`.
//! The reference's own fields are accepted where it reads them (`tools` and
//! `response_format` on a message, `task`, `wo_eos`, tool namespaces) except
//! `content_blocks`, its internal form of `content`.
//!
//! **Images are refused**, never dropped: an `image` or `image_url` part
//! anywhere is [`ChatError::UnsupportedContent`], and so is text that spells
//! the image placeholder where the reference refuses it.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;

use super::{
    ASSISTANT_TOKEN, BOS_TOKEN, CALLS_CLOSE, CALLS_OPEN, CALLS_SEPARATOR, EOS_TOKEN,
    IMAGE_PLACEHOLDER, INVOKE_CLOSE, INVOKE_OPEN, LATEST_REMINDER_TOKEN, PARAMETER_CLOSE,
    PARAMETER_OPEN, SYSTEM_TOKEN, THINK_END, THINK_START, USER_TOKEN,
};
use crate::args::ToolSchemas;
use crate::error::ChatError;
use crate::json::{self, Json};
use crate::template::{ChatInput, Prompt};

/// The reasoning budget written at the start of a thinking-mode prompt: an
/// integer from 1 to 100, which the reference's named levels are aliases of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReasoningEffort(u8);

impl ReasoningEffort {
    pub const LOW: ReasoningEffort = ReasoningEffort(50);
    pub const HIGH: ReasoningEffort = ReasoningEffort(75);
    pub const MAX: ReasoningEffort = ReasoningEffort(100);

    /// A numeric budget; `None` outside `1..=100`.
    pub fn budget(budget: u8) -> Option<ReasoningEffort> {
        (1..=100)
            .contains(&budget)
            .then_some(ReasoningEffort(budget))
    }

    /// One of the encoder's named levels: `low`, `high` or `max`.
    pub fn level(name: &str) -> Option<ReasoningEffort> {
        match name {
            "low" => Some(ReasoningEffort::LOW),
            "high" => Some(ReasoningEffort::HIGH),
            "max" => Some(ReasoningEffort::MAX),
            _ => None,
        }
    }

    /// The budget, from 1 to 100.
    pub fn get(self) -> u8 {
        self.0
    }
}

impl Default for ReasoningEffort {
    /// The reference's default, `high`.
    fn default() -> Self {
        ReasoningEffort::HIGH
    }
}

/// The reference's `encode_messages` keyword arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncodeOptions {
    /// `thinking_mode`: `"thinking"` when true, `"chat"` when false.
    pub thinking: bool,
    /// `reasoning_effort`; written only in thinking mode.
    pub reasoning_effort: ReasoningEffort,
    /// `drop_thinking`: leave out the reasoning of turns before the last user
    /// turn. Tools declared on any message override it to false.
    pub drop_thinking: bool,
}

impl EncodeOptions {
    /// The reference's defaults for a thinking mode: effort `high`, earlier
    /// reasoning dropped.
    pub fn new(thinking: bool) -> EncodeOptions {
        EncodeOptions {
            thinking,
            reasoning_effort: ReasoningEffort::default(),
            drop_thinking: true,
        }
    }

    /// The options a chat-completions request selects, resolved as the model's
    /// `chat_template.jinja` resolves them: `thinking` as given, else on
    /// unless the effort is `"none"`; `reasoning_effort` an integer from 1 to
    /// 100 or one of `low`, `high`, `max`, or OpenAI's `none`, `minimal`,
    /// `medium`, `xhigh` (mapped to 75, 50, 75, 75), `high` when absent or
    /// `null`. Any other effort is refused.
    pub fn from_request(
        thinking: Option<bool>,
        reasoning_effort: Option<&Json>,
    ) -> Result<EncodeOptions, ChatError> {
        let invalid = || {
            ChatError::InvalidInput(
                "reasoning_effort must be an integer from 1 to 100 or one of \"low\", \"high\", \
                 \"max\""
                    .into(),
            )
        };
        let effort = match reasoning_effort {
            None | Some(Json::Null) => ReasoningEffort::default(),
            Some(Json::Str(name)) => match name.as_str() {
                "minimal" => ReasoningEffort::LOW,
                "none" | "medium" | "xhigh" => ReasoningEffort::HIGH,
                name => ReasoningEffort::level(name).ok_or_else(invalid)?,
            },
            Some(Json::Int(n)) => n
                .to_i128()
                .and_then(|n| u8::try_from(n).ok())
                .and_then(ReasoningEffort::budget)
                .ok_or_else(invalid)?,
            Some(_) => return Err(invalid()),
        };
        let effort_is_none = reasoning_effort.and_then(Json::as_str) == Some("none");
        Ok(EncodeOptions {
            thinking: thinking.unwrap_or(!effort_is_none),
            reasoning_effort: effort,
            drop_thinking: true,
        })
    }
}

/// Renders the prompt for a request: its text, byte-identical to the
/// reference's `encode_messages` on the same messages (the request's `tools`
/// attached as the module docs say), and the parts of it that are request
/// data, which [`crate::Tokenizer::encode_prompt`] keeps from spelling an
/// added token.
pub fn encode(input: &ChatInput, options: EncodeOptions) -> Result<Prompt, ChatError> {
    let messages = conversation(input)?;
    let last_user = last_user_index(&messages);
    let drop_thinking = options.drop_thinking && !messages.iter().any(|m| !m.tools.is_empty());
    let mut out = Writer::default();
    out.own(BOS_TOKEN);
    for index in 0..messages.len() {
        render_message(
            &mut out,
            &messages,
            index,
            last_user,
            drop_thinking,
            options,
        );
    }
    Ok(Prompt::from_parts(out.text, out.data))
}

/// The tools a request declares, under the names the model calls them by
/// (`namespace::name` for a namespaced tool): what
/// [`crate::OutputConfig::deepseek_v4_1`] recognizes calls against.
pub fn tool_schemas(tools: &Json) -> Result<ToolSchemas, ChatError> {
    let functions = encoded_tools(tools, "tools")?
        .into_iter()
        .map(|function| Json::Object(vec![("function".to_string(), function)]))
        .collect();
    Ok(ToolSchemas::from_tools(&Json::Array(functions)))
}

/// The prompt under construction, with the ranges written from the request.
#[derive(Default)]
struct Writer {
    text: String,
    data: Vec<Range<usize>>,
}

impl Writer {
    /// Appends the format's own text.
    fn own(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// Appends text the request supplied.
    fn data(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        let start = self.text.len();
        self.text.push_str(text);
        match self.data.last_mut() {
            Some(last) if last.end == start => last.end = self.text.len(),
            _ => self.data.push(start..self.text.len()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    System,
    User,
    Assistant,
    LatestReminder,
}

/// A content part.
#[derive(Clone, Debug)]
enum Block<'a> {
    Text(&'a str),
    ToolResult {
        /// `tool_use_id`, when a non-empty string.
        id: Option<&'a str>,
        content: ToolContent<'a>,
    },
    /// Any other type, which a user turn prints as `[Unsupported {type}]`.
    Other(&'a str),
}

#[derive(Clone, Debug)]
enum ToolContent<'a> {
    Text(&'a str),
    Parts(Vec<Block<'a>>),
}

/// The quick-instruction tasks and their tokens.
const TASKS: &[(&str, &str)] = &[
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

/// A tool call as the prompt writes it.
#[derive(Clone, Debug)]
struct Call<'a> {
    /// `namespace::name`, or the bare name.
    name: String,
    /// The call's `id`, else its function's: what tool results are ordered by.
    id: Option<&'a str>,
    /// Parameter names with their values, a string written as is and anything
    /// else as JSON.
    parameters: Vec<(Cow<'a, str>, Parameter<'a>)>,
}

#[derive(Clone, Debug)]
enum Parameter<'a> {
    Text(Cow<'a, str>),
    Json(String),
}

/// One turn of the conversation after the reference's preprocessing.
#[derive(Clone, Debug)]
struct Message<'a> {
    role: Role,
    /// The reference's `content_blocks`: a user turn's parts, or any other
    /// message's array content. `None` when the content was not an array.
    blocks: Option<Vec<Block<'a>>>,
    /// The content as text: a string content, or the text parts of an array
    /// joined by blank lines. Empty when absent.
    content: Cow<'a, str>,
    /// The tool schemas this message declares, encoded.
    tools: Vec<Json>,
    response_format: Option<&'a Json>,
    calls: Vec<Call<'a>>,
    reasoning: &'a str,
    without_eos: bool,
    /// The task token to end the turn with, and whether it is `action`.
    task: Option<(&'static str, bool)>,
}

impl Message<'_> {
    fn tool_results(block: Block<'_>) -> Message<'_> {
        Message {
            role: Role::User,
            blocks: Some(vec![block]),
            content: Cow::Borrowed(""),
            tools: Vec::new(),
            response_format: None,
            calls: Vec::new(),
            reasoning: "",
            without_eos: false,
            task: None,
        }
    }
}

fn invalid(message: String) -> ChatError {
    ChatError::InvalidInput(message)
}

/// Python's truth value of a JSON value.
fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Int(n) => n.digits != "0",
        Json::Float(x) => *x != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Array(items) => !items.is_empty(),
        Json::Object(fields) => !fields.is_empty(),
    }
}

/// A member that must be a string when present and not `null`.
fn optional_str<'a>(object: &'a Json, key: &str, at: &str) -> Result<Option<&'a str>, ChatError> {
    match object.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s)),
        Some(_) => Err(invalid(format!("{at}.{key} must be a string"))),
    }
}

/// The reference's preprocessing: validates the request, merges tool and user
/// messages into user turns, attaches the request's tools, and orders each
/// turn's tool results by the calls before it.
fn conversation(input: &ChatInput) -> Result<Vec<Message<'_>>, ChatError> {
    let Some(raw) = input.messages.as_array().filter(|m| !m.is_empty()) else {
        return Err(invalid("messages must be a non-empty array".into()));
    };
    let mut messages: Vec<Message<'_>> = Vec::with_capacity(raw.len());
    for (m, message) in raw.iter().enumerate() {
        let at = format!("messages[{m}]");
        if message.as_object().is_none() {
            return Err(invalid(format!("{at} must be an object")));
        }
        if message.get("content_blocks").is_some() {
            return Err(invalid(format!("{at}.content_blocks is not accepted")));
        }
        let role = match message.get("role").and_then(Json::as_str) {
            Some("system") => Some(Role::System),
            Some("user") => Some(Role::User),
            Some("assistant") => Some(Role::Assistant),
            Some("latest_reminder") => Some(Role::LatestReminder),
            Some("tool") => None,
            Some(other) => return Err(invalid(format!("{at}.role {other:?} is not supported"))),
            None => return Err(invalid(format!("{at}.role must be a string"))),
        };
        let parsed = parse_message(message, role, &at)?;
        let previous_user = messages.last_mut().filter(|last| last.role == Role::User);
        match (role, previous_user) {
            (None, previous) => {
                let block = Block::ToolResult {
                    id: optional_str(message, "tool_call_id", &at)?.filter(|id| !id.is_empty()),
                    content: match parsed.blocks {
                        Some(parts) => ToolContent::Parts(parts),
                        None => match message.get("content") {
                            Some(Json::Str(text)) => ToolContent::Text(text),
                            None => ToolContent::Text(""),
                            Some(_) => {
                                return Err(invalid(format!(
                                    "{at}.content must be a string or an array"
                                )));
                            }
                        },
                    },
                };
                match previous {
                    Some(last) => last.blocks.get_or_insert_default().push(block),
                    None => messages.push(Message::tool_results(block)),
                }
            }
            (Some(Role::User), previous) => {
                let blocks = match parsed.blocks.clone() {
                    Some(parts) => parts,
                    None => match message.get("content") {
                        Some(Json::Str(text)) => vec![Block::Text(text)],
                        None => vec![Block::Text("")],
                        Some(_) => {
                            return Err(invalid(format!(
                                "{at}.content must be a string or an array"
                            )));
                        }
                    },
                };
                match previous {
                    Some(last) if last.task.is_none() => {
                        last.blocks.get_or_insert_default().extend(blocks)
                    }
                    _ => messages.push(Message {
                        blocks: Some(blocks),
                        ..parsed
                    }),
                }
            }
            (Some(_), _) => messages.push(parsed),
        }
    }

    if let Some(tools) = input.tools.as_ref().filter(|tools| truthy(tools)) {
        let tools = encoded_tools(tools, "tools")?;
        match messages.first_mut() {
            Some(first) if first.role == Role::System => {
                if first.tools.is_empty() {
                    first.tools = tools;
                }
            }
            _ => messages.insert(
                0,
                Message {
                    role: Role::System,
                    blocks: None,
                    tools,
                    ..Message::tool_results(Block::Text(""))
                },
            ),
        }
    }

    // A turn's tool results follow the order of the latest calls before it;
    // a result for none of them sorts first.
    let mut call_order: HashMap<&str, usize> = HashMap::new();
    for message in &mut messages {
        match message.role {
            Role::Assistant if !message.calls.is_empty() => {
                call_order.clear();
                for (position, call) in message.calls.iter().enumerate() {
                    if let Some(id) = call.id {
                        call_order.insert(id, position);
                    }
                }
            }
            Role::User if !call_order.is_empty() => {
                let Some(blocks) = &mut message.blocks else {
                    continue;
                };
                let position = |block: &Block<'_>| match block {
                    Block::ToolResult { id, .. } => {
                        Some(id.and_then(|id| call_order.get(id)).copied().unwrap_or(0))
                    }
                    _ => None,
                };
                let mut results: Vec<Block<'_>> = blocks
                    .iter()
                    .filter(|block| position(block).is_some())
                    .cloned()
                    .collect();
                if results.len() > 1 {
                    results.sort_by_key(|block| position(block));
                    let mut results = results.into_iter();
                    for block in blocks.iter_mut() {
                        if position(block).is_some() {
                            *block = results.next().expect("one per tool result");
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(messages)
}

/// Validates one message and reads the fields the reference reads from it.
/// `role` is `None` for a tool message, of which only the content is used.
fn parse_message<'a>(
    message: &'a Json,
    role: Option<Role>,
    at: &str,
) -> Result<Message<'a>, ChatError> {
    let content = message.get("content");
    let spells_image = |text: &str| text.contains(IMAGE_PLACEHOLDER);
    let (blocks, text) = match content {
        None | Some(Json::Null) => (None, None),
        Some(Json::Str(text)) => {
            if spells_image(text) {
                return Err(ChatError::UnsupportedContent(format!(
                    "{at}.content spells the image placeholder"
                )));
            }
            (None, Some(Cow::Borrowed(text.as_str())))
        }
        Some(Json::Array(parts)) => {
            let blocks = parse_blocks(parts, &format!("{at}.content"))?;
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Text(text) => Some(*text),
                    _ => None,
                })
                .collect();
            // An empty array is no content at all.
            let text = (!blocks.is_empty()).then(|| Cow::Owned(texts.join("\n\n")));
            (Some(blocks), text)
        }
        Some(_) => {
            return Err(invalid(format!(
                "{at}.content must be a string or an array"
            )));
        }
    };
    let needs_content = matches!(role, None | Some(Role::User | Role::LatestReminder));
    if needs_content && matches!(content, Some(Json::Null)) {
        return Err(invalid(format!("{at}.content must not be null")));
    }
    if role == Some(Role::LatestReminder) && text.is_none() {
        return Err(invalid(format!("{at}.content must be text")));
    }

    let reasoning = optional_str(message, "reasoning_content", at)?.unwrap_or("");
    if spells_image(reasoning) {
        return Err(ChatError::UnsupportedContent(format!(
            "{at}.reasoning_content spells the image placeholder"
        )));
    }
    let tools = match message.get("tools") {
        None | Some(Json::Null) => Vec::new(),
        Some(tools) => encoded_tools(tools, &format!("{at}.tools"))?,
    };
    let calls = match message.get("tool_calls") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::Array(calls)) => calls
            .iter()
            .enumerate()
            .map(|(c, call)| parse_call(call, &format!("{at}.tool_calls[{c}]")))
            .collect::<Result<_, _>>()?,
        Some(_) => return Err(invalid(format!("{at}.tool_calls must be an array"))),
    };
    let without_eos = match message.get("wo_eos") {
        None => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => return Err(invalid(format!("{at}.wo_eos must be a boolean"))),
    };
    let task = match optional_str(message, "task", at)? {
        None => None,
        Some(name) => match TASKS.iter().find(|(task, _)| *task == name) {
            Some((task, token)) => Some((*token, *task == "action")),
            None => return Err(invalid(format!("{at}.task {name:?} is not a task"))),
        },
    };
    optional_str(message, "tool_call_id", at)?;
    Ok(Message {
        role: role.unwrap_or(Role::User),
        blocks,
        content: text.unwrap_or_default(),
        tools,
        response_format: message.get("response_format").filter(|v| truthy(v)),
        calls,
        reasoning,
        without_eos,
        task,
    })
}

/// Validates an array content and reads its parts. Images are refused.
fn parse_blocks<'a>(parts: &'a [Json], at: &str) -> Result<Vec<Block<'a>>, ChatError> {
    let mut blocks = Vec::with_capacity(parts.len());
    for (p, part) in parts.iter().enumerate() {
        let at = format!("{at}[{p}]");
        if part.as_object().is_none() {
            return Err(invalid(format!("{at} must be an object")));
        }
        let Some(kind) = part.get("type").and_then(Json::as_str) else {
            return Err(invalid(format!("{at}.type must be a string")));
        };
        blocks.push(match kind {
            "image" | "image_url" => {
                return Err(ChatError::UnsupportedContent(format!(
                    "{at} is an image, and this model is served text only"
                )));
            }
            "text" => {
                let text = match part.get("text") {
                    None => "",
                    Some(Json::Str(text)) => text.as_str(),
                    Some(_) => return Err(invalid(format!("{at}.text must be a string"))),
                };
                if text.contains(IMAGE_PLACEHOLDER) {
                    return Err(ChatError::UnsupportedContent(format!(
                        "{at}.text spells the image placeholder"
                    )));
                }
                Block::Text(text)
            }
            "tool_result" => Block::ToolResult {
                id: optional_str(part, "tool_use_id", &at)?.filter(|id| !id.is_empty()),
                content: match part.get("content") {
                    None => ToolContent::Text(""),
                    Some(Json::Str(text)) => ToolContent::Text(text),
                    Some(Json::Array(parts)) => {
                        ToolContent::Parts(parse_blocks(parts, &format!("{at}.content"))?)
                    }
                    Some(_) => {
                        return Err(invalid(format!(
                            "{at}.content must be a string or an array"
                        )));
                    }
                },
            },
            other => Block::Other(other),
        });
    }
    Ok(blocks)
}

/// The reference's `_split_tool_name`: a `namespace::name` is split, and must
/// agree with a namespace given beside it; neither part may hold a further
/// `::`.
fn split_tool_name<'a>(
    name: &'a str,
    namespace: Option<&'a str>,
    at: &str,
) -> Result<(Option<&'a str>, &'a str), ChatError> {
    let (namespace, name) = match name.split_once("::") {
        Some((prefix, bare)) => {
            if namespace.is_some_and(|namespace| namespace != prefix) {
                return Err(invalid(format!("{at}: conflicting tool namespaces")));
            }
            (Some(prefix), bare)
        }
        None => (namespace, name),
    };
    if name.contains("::") || namespace.is_some_and(|namespace| namespace.contains("::")) {
        return Err(invalid(format!(
            "{at}: a tool name holds at most one \"::\""
        )));
    }
    Ok((namespace, name))
}

fn qualified_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(namespace) => format!("{namespace}::{name}"),
        None => name.to_string(),
    }
}

/// The reference's `tools_from_openai_format`: each tool's `function` object
/// under its qualified name, its namespace removed and the namespace's
/// description put before its own.
fn encoded_tools(tools: &Json, at: &str) -> Result<Vec<Json>, ChatError> {
    let Some(tools) = tools.as_array() else {
        return Err(invalid(format!("{at} must be an array")));
    };
    let mut out = Vec::with_capacity(tools.len());
    for (t, tool) in tools.iter().enumerate() {
        let at = format!("{at}[{t}]");
        let Some(function) = tool.as_object().and(tool.get("function")) else {
            return Err(invalid(format!("{at}.function must be an object")));
        };
        let Some(fields) = function.as_object() else {
            return Err(invalid(format!("{at}.function must be an object")));
        };
        let Some(name) = function.get("name").and_then(Json::as_str) else {
            return Err(invalid(format!("{at}.function.name must be a string")));
        };
        // The tool's own namespace wins over its function's.
        let namespace = match tool.get("namespace") {
            None | Some(Json::Null) => function.get("namespace"),
            given => given,
        };
        let (namespace_name, namespace_description) = match namespace {
            None | Some(Json::Null) => (None, None),
            Some(Json::Str(name)) => (Some(name.as_str()), None),
            Some(namespace @ Json::Object(_)) => {
                if namespace.get("name").is_none() {
                    return Err(invalid(format!("{at}.namespace.name is missing")));
                }
                (
                    optional_str(namespace, "name", &format!("{at}.namespace"))?,
                    optional_str(namespace, "description", &format!("{at}.namespace"))?
                        .filter(|description| !description.is_empty()),
                )
            }
            Some(_) => {
                return Err(invalid(format!(
                    "{at}.namespace must be a string or an object"
                )));
            }
        };
        let (namespace_name, bare) = split_tool_name(name, namespace_name, &at)?;
        let mut encoded: Vec<(String, Json)> = fields
            .iter()
            .filter(|(key, _)| key != "namespace")
            .cloned()
            .collect();
        let set = |fields: &mut Vec<(String, Json)>, key: &str, value: String| match fields
            .iter_mut()
            .find(|(k, _)| k == key)
        {
            Some((_, slot)) => *slot = Json::Str(value),
            None => fields.push((key.to_string(), Json::Str(value))),
        };
        set(&mut encoded, "name", qualified_name(namespace_name, bare));
        if let Some(description) = namespace_description {
            let own = match function.get("description") {
                Some(own) if truthy(own) => match own {
                    Json::Str(own) => own.as_str(),
                    _ => {
                        return Err(invalid(format!(
                            "{at}.function.description must be a string"
                        )));
                    }
                },
                _ => "",
            };
            set(&mut encoded, "description", format!("{description}\n{own}"));
        }
        out.push(Json::Object(encoded));
    }
    Ok(out)
}

/// Whether a failed parse is of text Python's `json.loads` would have read:
/// there the reference goes on with a value this crate cannot hold.
fn python_would_parse(error: &json::ParseError) -> bool {
    matches!(error.message, "lone surrogate" | "nesting too deep")
}

/// The reference's `tool_calls_from_openai_format` and
/// `encode_arguments_to_dsml` for one call.
fn parse_call<'a>(call: &'a Json, at: &str) -> Result<Call<'a>, ChatError> {
    let Some(function) = call
        .as_object()
        .and(call.get("function"))
        .filter(|function| function.as_object().is_some())
    else {
        return Err(invalid(format!("{at}.function must be an object")));
    };
    let function_at = format!("{at}.function");
    let Some(name) = function.get("name").and_then(Json::as_str) else {
        return Err(invalid(format!("{function_at}.name must be a string")));
    };
    // A namespace on the call wins over the function's unless it is empty.
    let namespace = match optional_str(call, "namespace", at)? {
        Some(namespace) if !namespace.is_empty() => Some(namespace),
        _ => optional_str(function, "namespace", &function_at)?,
    };
    let (namespace, bare) = split_tool_name(name, namespace, at)?;
    let id = match optional_str(call, "id", at)? {
        Some(id) if !id.is_empty() => Some(id),
        _ => optional_str(function, "id", &function_at)?.filter(|id| !id.is_empty()),
    };

    let Some(given) = function.get("arguments") else {
        return Err(invalid(format!("{function_at}.arguments is missing")));
    };
    // A JSON string is decoded, twice if it encodes another; anything that
    // does not come to an object is the single parameter `arguments`.
    let mut arguments = Cow::Borrowed(given);
    for _ in 0..2 {
        let Json::Str(text) = arguments.as_ref() else {
            break;
        };
        match json::parse_python_lenient(text) {
            Ok(decoded) => arguments = Cow::Owned(decoded),
            Err(error) if python_would_parse(&error) => {
                return Err(invalid(format!(
                    "{function_at}.arguments cannot be decoded: {error}"
                )));
            }
            Err(_) => break,
        }
    }
    let parameter = |value: Cow<'a, Json>| match value {
        Cow::Borrowed(Json::Str(text)) => Parameter::Text(Cow::Borrowed(text.as_str())),
        Cow::Owned(Json::Str(text)) => Parameter::Text(Cow::Owned(text)),
        other => Parameter::Json(json::dumps_default(&other)),
    };
    let parameters = match arguments {
        Cow::Borrowed(Json::Object(fields)) => fields
            .iter()
            .map(|(key, value)| (Cow::Borrowed(key.as_str()), parameter(Cow::Borrowed(value))))
            .collect(),
        Cow::Owned(Json::Object(fields)) => fields
            .into_iter()
            .map(|(key, value)| (Cow::Owned(key), parameter(Cow::Owned(value))))
            .collect(),
        _ => vec![(Cow::Borrowed("arguments"), parameter(Cow::Borrowed(given)))],
    };
    Ok(Call {
        name: qualified_name(namespace, bare),
        id,
        parameters,
    })
}

/// The reference's `find_last_user_index`: the last user turn or
/// mid-conversation system message, passing over those whose array content
/// holds no text part (a turn of tool results alone).
fn last_user_index(messages: &[Message<'_>]) -> Option<usize> {
    (0..messages.len()).rev().find(|&index| {
        let message = &messages[index];
        let starts_turn = match message.role {
            Role::User => true,
            Role::System => index > 0,
            _ => false,
        };
        starts_turn
            && message.blocks.as_ref().is_none_or(|blocks| {
                blocks.is_empty() || blocks.iter().any(|b| matches!(b, Block::Text(_)))
            })
    })
}

/// The reference's `render_message`.
fn render_message(
    out: &mut Writer,
    messages: &[Message<'_>],
    index: usize,
    last_user: Option<usize>,
    drop_thinking: bool,
    options: EncodeOptions,
) {
    let message = &messages[index];
    let thinking = options.thinking;
    // "After the last user turn", and "at or after it", with no user turn at
    // all counting as before every message.
    let after_last_user = last_user.is_none_or(|last| index > last);
    let from_last_user = last_user.is_none_or(|last| index >= last);

    if index == 0 && (thinking || message.role == Role::System) {
        out.own(SYSTEM_TOKEN);
    }
    if index == 0 && thinking {
        out.own(&format!(
            "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the \
             reasoning)\n\n",
            options.reasoning_effort.get()
        ));
    }

    match message.role {
        Role::System => {
            if index > 0 {
                out.own(SYSTEM_TOKEN);
            }
            out.data(&message.content);
            if !message.tools.is_empty() {
                out.own("\n\n");
                render_tools(out, &message.tools);
            }
            if let Some(schema) = message.response_format {
                out.own(
                    "\n\n## Response Format:\n\nYou MUST strictly adhere to the following \
                     schema to reply:\n",
                );
                out.data(&json::dumps_default(schema));
            }
        }
        Role::User => {
            out.own(USER_TOKEN);
            let blocks = message.blocks.as_deref().unwrap_or_default();
            for (b, block) in blocks.iter().enumerate() {
                if b > 0 {
                    out.own("\n\n");
                }
                match block {
                    Block::Text(text) => out.data(text),
                    Block::ToolResult { content, .. } => {
                        out.own("<tool_result>");
                        match content {
                            ToolContent::Text(text) => out.data(text),
                            ToolContent::Parts(parts) => {
                                for (p, part) in parts.iter().enumerate() {
                                    if p > 0 {
                                        out.own("\n\n");
                                    }
                                    match part {
                                        Block::Text(text) => out.data(text),
                                        Block::ToolResult { .. } => {
                                            out.own("[Unsupported tool_result]")
                                        }
                                        Block::Other(kind) => render_unsupported(out, kind),
                                    }
                                }
                            }
                        }
                        out.own("</tool_result>");
                    }
                    Block::Other(kind) => render_unsupported(out, kind),
                }
            }
        }
        Role::LatestReminder => {
            out.own(LATEST_REMINDER_TOKEN);
            out.data(&message.content);
        }
        Role::Assistant => {
            let answers_task = index > 0 && messages[index - 1].task.is_some();
            if thinking && !answers_task && (!drop_thinking || after_last_user) {
                out.data(message.reasoning);
                out.own(THINK_END);
            }
            out.data(&message.content);
            if !message.calls.is_empty() {
                out.own(CALLS_SEPARATOR);
                out.own(CALLS_OPEN);
                for call in &message.calls {
                    out.own("\n");
                    render_call(out, call);
                }
                out.own("\n");
                out.own(CALLS_CLOSE);
            }
            if !message.without_eos {
                out.own(EOS_TOKEN);
            }
        }
    }

    // What follows the message: nothing unless it is the last or an assistant
    // (or reminder) comes next.
    let next_role = messages.get(index + 1).map(|next| next.role);
    if next_role.is_some_and(|role| !matches!(role, Role::Assistant | Role::LatestReminder)) {
        return;
    }
    let starts_turn = match message.role {
        Role::User => true,
        Role::System => index > 0,
        _ => false,
    };
    match message.task {
        Some((token, true)) => {
            out.own(ASSISTANT_TOKEN);
            out.own(if thinking { THINK_START } else { THINK_END });
            out.own(token);
        }
        Some((token, false)) => out.own(token),
        None if starts_turn => {
            out.own(ASSISTANT_TOKEN);
            let thinks = thinking && (!drop_thinking || from_last_user);
            out.own(if thinks { THINK_START } else { THINK_END });
        }
        None => {}
    }
}

fn render_unsupported(out: &mut Writer, kind: &str) {
    out.own("[Unsupported ");
    out.data(kind);
    out.own("]");
}

/// The reference's `render_tools`.
fn render_tools(out: &mut Writer, tools: &[Json]) {
    out.own(TOOLS_HEAD);
    for (t, tool) in tools.iter().enumerate() {
        if t > 0 {
            out.own("\n");
        }
        out.data(&json::dumps_default(tool));
    }
    out.own(TOOLS_TAIL);
}

/// One `<｜DSML｜ invoke>` element.
fn render_call(out: &mut Writer, call: &Call<'_>) {
    out.own(INVOKE_OPEN);
    out.own(" name=\"");
    out.data(&call.name);
    out.own("\">\n");
    for (p, (name, value)) in call.parameters.iter().enumerate() {
        if p > 0 {
            out.own("\n");
        }
        out.own(PARAMETER_OPEN);
        out.own(" name=\"");
        out.data(name);
        match value {
            Parameter::Text(text) => {
                out.own("\" string=\"true\">");
                out.data(text);
            }
            Parameter::Json(text) => {
                out.own("\" string=\"false\">");
                out.data(text);
            }
        }
        out.own(PARAMETER_CLOSE);
    }
    out.own("\n");
    out.own(INVOKE_CLOSE);
}

/// The reference's `TOOLS_TEMPLATE` up to the tool schemas.
const TOOLS_HEAD: &str = "## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜ calls>\" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name=\"$TOOL_NAME\">
<｜DSML｜ parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name=\"$TOOL_NAME2\">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

";

/// The reference's `TOOLS_TEMPLATE` after the tool schemas.
const TOOLS_TAIL: &str = "

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
";
