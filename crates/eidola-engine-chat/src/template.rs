//! Chat-template rendering, byte-identical to Python `transformers`.
//!
//! The template is the model's own `chat_template.jinja`, executed by
//! minijinja in an environment configured like the one `transformers` builds
//! for `apply_chat_template` (an immutable sandbox with `trim_blocks`,
//! `lstrip_blocks`, loop controls, `raise_exception`, and a `tojson` that is
//! Python's `json.dumps` rather than Jinja's HTML-escaping filter). Only
//! templates whose SHA-256 is pinned here are accepted: byte-exactness is a
//! property proven for a specific template, not for Jinja in general.
//!
//! Inputs are validated against the OpenAI message shape before rendering
//! (see [`validate`]). The validated domain is exactly the one in which
//! minijinja and Jinja2 agree for the pinned template; requests outside it
//! (a non-string `role`, a content part that is neither a string nor an
//! object, …) are rejected rather than rendered differently from Python.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use minijinja::value::{Kwargs, Object, ObjectRepr, Value, ValueKind};
use minijinja::{
    AutoEscape, Environment, Error as JinjaError, ErrorKind, State, UndefinedBehavior,
};
use sha2::{Digest, Sha256};

use crate::error::ChatError;
use crate::json::{self, BigInt, DumpOptions, Json};

/// File name of the template inside a model directory.
pub const CHAT_TEMPLATE_FILE: &str = "chat_template.jinja";

/// SHA-256 of the MiMo-V2.6 chat template (identical in the Flash and Pro
/// releases).
pub const MIMO_V2_6_CHAT_TEMPLATE_SHA256: &str =
    "11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059";

/// Every template this crate will execute.
pub const PINNED_TEMPLATE_SHA256: &[&str] = &[MIMO_V2_6_CHAT_TEMPLATE_SHA256];

const TEMPLATE_NAME: &str = "chat_template";

/// Rendering switches, mirroring the keyword arguments of
/// `apply_chat_template`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderOptions {
    /// Append the assistant header that starts a new turn.
    pub add_generation_prompt: bool,
    /// `enable_thinking`; `None` leaves it undefined in the template, which
    /// for MiMo means thinking is on. Only `Some(false)` changes the prompt.
    pub enable_thinking: Option<bool>,
    /// `continue_final_message`, normalized the way `transformers` normalizes
    /// it: the name of the final message's field the prompt ends inside
    /// (`True` there is `"content"`), or `None` to end after the conversation.
    /// See [`ChatTemplate::render`].
    pub continue_final_message: Option<&'static str>,
}

/// The sentinel `transformers` appends to the field being continued and then
/// cuts the rendered prompt at (`render_jinja_template`'s
/// `continue_final_message_tag`).
const CONTINUE_FINAL_MESSAGE_TAG: &str = "CONTINUE_FINAL_MESSAGE_TAG ";

/// Context variable through which a render tells `tojson` the pair of marks to
/// put around its output. The template never reads it.
const DATA_MARKS_VAR: &str = "eidola_data_marks";

/// A rendered prompt: its text, and which parts of it the request supplied.
///
/// Everything the template writes from the request (a role, message content,
/// reasoning, a tool call's name, argument names and values, tool definitions)
/// is data; the rest is the template's own text, which alone may spell the
/// model's added tokens ([`crate::MimoTokenizer::encode_prompt`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    text: String,
    data: Vec<Range<usize>>,
}

impl Prompt {
    /// The prompt text, byte-identical to `apply_chat_template`'s.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The byte ranges of [`Self::text`] that are request data: ascending,
    /// disjoint, non-empty, on character boundaries.
    pub fn data_spans(&self) -> &[Range<usize>] {
        &self.data
    }

    pub fn into_text(self) -> String {
        self.text
    }

    /// Cuts the prompt to its first `len` bytes.
    fn truncate(&mut self, len: usize) {
        self.text.truncate(len);
        self.data.retain_mut(|span| {
            span.end = span.end.min(len);
            span.start < span.end
        });
    }
}

/// OpenAI-shaped chat input: the `messages` array and the optional `tools`
/// array of a chat-completions request.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatInput {
    pub messages: Json,
    pub tools: Option<Json>,
}

impl ChatInput {
    /// Parses the request's `messages` and `tools` JSON texts, preserving key
    /// order (which `tojson` reproduces in the prompt).
    pub fn from_json(messages: &str, tools: Option<&str>) -> Result<ChatInput, ChatError> {
        let messages =
            json::parse(messages).map_err(|e| ChatError::InvalidInput(format!("messages: {e}")))?;
        let tools = match tools {
            None => None,
            Some(text) => Some(
                json::parse(text).map_err(|e| ChatError::InvalidInput(format!("tools: {e}")))?,
            ),
        };
        Ok(ChatInput { messages, tools })
    }

    /// Replaces every assistant tool call's string `arguments` with the JSON
    /// object it encodes.
    ///
    /// OpenAI clients send `arguments` as a JSON string, but the template
    /// renders a string verbatim (inside `<function=…>`), which is not the
    /// parameter format the model was trained on. Serving must therefore
    /// decode it first; a string that is not a JSON object is an error.
    ///
    /// The call body is selected exactly as validation and the template select
    /// it ([`tool_call_body_key`]): `function`, else `custom`, else the call
    /// object itself. A body whose `input` is a string is left alone, because
    /// the template renders that `input` instead of `arguments`.
    pub fn normalize_tool_call_arguments(&mut self) -> Result<(), ChatError> {
        let Json::Array(messages) = &mut self.messages else {
            return Ok(());
        };
        for (m, message) in messages.iter_mut().enumerate() {
            let Json::Object(fields) = message else {
                continue;
            };
            if !fields
                .iter()
                .any(|(k, v)| k == "role" && v.as_str() == Some("assistant"))
            {
                continue;
            }
            let Some((_, Json::Array(calls))) = fields.iter_mut().find(|(k, _)| k == "tool_calls")
            else {
                continue;
            };
            for (c, call) in calls.iter_mut().enumerate() {
                let key = tool_call_body_key(call);
                let Json::Object(call_fields) = call else {
                    continue;
                };
                let body = match key {
                    Some(key) => match call_fields.iter_mut().find(|(k, _)| k == key) {
                        Some((_, Json::Object(body))) => body,
                        _ => continue,
                    },
                    None => call_fields,
                };
                if body
                    .iter()
                    .any(|(k, v)| k == "input" && matches!(v, Json::Str(_)))
                {
                    continue;
                }
                let at = key.map_or(String::new(), |k| format!(".{k}"));
                let Some((_, arguments)) = body.iter_mut().find(|(k, _)| k == "arguments") else {
                    continue;
                };
                if let Json::Str(text) = arguments {
                    let parsed = json::parse(text).map_err(|e| {
                        ChatError::InvalidInput(format!(
                            "messages[{m}].tool_calls[{c}]{at}.arguments is not valid JSON: {e}"
                        ))
                    })?;
                    if !matches!(parsed, Json::Object(_)) {
                        return Err(ChatError::InvalidInput(format!(
                            "messages[{m}].tool_calls[{c}]{at}.arguments must be a JSON object"
                        )));
                    }
                    *arguments = parsed;
                }
            }
        }
        Ok(())
    }
}

/// A pinned chat template, ready to render.
pub struct ChatTemplate {
    env: Environment<'static>,
    sha256: String,
    /// The template's source, which `transformers` searches for the name of
    /// the field a continuation extends.
    source: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("sha256", &self.sha256)
            .finish_non_exhaustive()
    }
}

impl ChatTemplate {
    /// Loads `chat_template.jinja` from a model directory.
    pub fn from_model_dir(dir: &Path) -> Result<ChatTemplate, ChatError> {
        let path = dir.join(CHAT_TEMPLATE_FILE);
        let source = std::fs::read_to_string(&path).map_err(|e| ChatError::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        ChatTemplate::from_source(source)
    }

    /// Compiles template source, refusing anything whose hash is not pinned.
    pub fn from_source(source: String) -> Result<ChatTemplate, ChatError> {
        let sha256 = sha256_hex(source.as_bytes());
        if !PINNED_TEMPLATE_SHA256.contains(&sha256.as_str()) {
            return Err(ChatError::UnpinnedArtifact {
                what: "chat template",
                sha256,
            });
        }
        let mut env = Environment::new();
        // `transformers` uses jinja2's defaults here; `keep_trailing_newline`
        // is false in both engines.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_undefined_behavior(UndefinedBehavior::Lenient);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.add_filter("tojson", tojson_filter);
        env.add_test("iterable", is_iterable);
        env.add_function("raise_exception", raise_exception);
        env.add_template_owned(TEMPLATE_NAME, source.clone())
            .map_err(|e| ChatError::Template(e.to_string()))?;
        Ok(ChatTemplate {
            env,
            sha256,
            source,
        })
    }

    /// Hex SHA-256 of the template source.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Renders the prompt text ([`Self::render_prompt`] without the data spans).
    pub fn render(&self, input: &ChatInput, options: RenderOptions) -> Result<String, ChatError> {
        self.render_prompt(input, options).map(Prompt::into_text)
    }

    /// Renders the prompt, with the parts of it that are request data marked.
    ///
    /// With [`RenderOptions::continue_final_message`] set, the prompt ends
    /// inside the final message's field rather than after the conversation,
    /// exactly where `transformers` 5.19's `render_jinja_template` ends it: the
    /// sentinel [`CONTINUE_FINAL_MESSAGE_TAG`] is appended to the field (to the
    /// last content part with a `text` when the field is a list), the
    /// conversation is rendered, and the result is cut at the last occurrence
    /// of the sentinel, keeping trailing whitespace when the template wrote the
    /// sentinel whole and `rstrip`ping it otherwise. The reference's refusals
    /// stay refusals: a generation prompt asked for as well (refused by
    /// `apply_chat_template` itself), a field that is absent or `null`, a field name the
    /// template never mentions, a list with no `text` part, a value that
    /// cannot be concatenated with a string, and a render in which the field's
    /// text or the sentinel does not appear.
    ///
    /// For the pinned template, continuing `content` ends the prompt after the
    /// partial answer (`…<think>R</think>C`), and continuing
    /// `reasoning_content` ends it inside an open reasoning block
    /// (`…<think>R`). Either is the original request's prompt followed by the
    /// text the model generated for it, so a continuation resumes exactly
    /// where generation stopped.
    pub fn render_prompt(
        &self,
        input: &ChatInput,
        options: RenderOptions,
    ) -> Result<Prompt, ChatError> {
        validate(input)?;
        let Some(field) = options.continue_final_message else {
            return self.render_marked(&input.messages, input, options);
        };
        if options.add_generation_prompt {
            return Err(ChatError::InvalidInput(
                "continue_final_message and add_generation_prompt are not compatible".into(),
            ));
        }
        let (messages, final_text) = continued_messages(&input.messages, field, &self.source)?;
        let mut prompt = self.render_marked(&messages, input, options)?;
        let rendered = prompt.text();
        let tag = CONTINUE_FINAL_MESSAGE_TAG.trim_end();
        if !rendered.contains(crate::pysem::py_strip(&final_text)) || !rendered.contains(tag) {
            return Err(ChatError::InvalidInput(
                "continue_final_message is set but the final message does not appear in the \
                 chat after applying the chat template"
                    .into(),
            ));
        }
        let at = rendered.rfind(tag).expect("checked above");
        let len = if rendered[at..].starts_with(CONTINUE_FINAL_MESSAGE_TAG) {
            at
        } else {
            rendered[..at]
                .trim_end_matches(crate::pysem::is_py_space)
                .len()
        };
        prompt.truncate(len);
        Ok(prompt)
    }

    /// Renders `messages` twice: as they are, which is the prompt text, and
    /// with every string the template prints from the request wrapped in a pair
    /// of marks, which shows where in that text the data lies. The marks are
    /// two characters the text does not contain, and the marked render with its
    /// marks removed must be the text itself, so marking can never change what
    /// the prompt says.
    fn render_marked(
        &self,
        messages: &Json,
        input: &ChatInput,
        options: RenderOptions,
    ) -> Result<Prompt, ChatError> {
        let text = self.render_messages(messages, input, options, None)?;
        let marks = unused_marks(&text)?;
        let marked = self.render_messages(
            &marked_messages(messages, marks),
            input,
            options,
            Some(marks),
        )?;
        match strip_marks(&marked, marks) {
            Some((stripped, data)) if stripped == text => Ok(Prompt { text, data }),
            _ => Err(ChatError::Template(
                "the prompt's request data could not be told from the template's text".into(),
            )),
        }
    }

    fn render_messages(
        &self,
        messages: &Json,
        input: &ChatInput,
        options: RenderOptions,
        marks: Option<(char, char)>,
    ) -> Result<String, ChatError> {
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| ChatError::Template(e.to_string()))?;
        let mut context: Vec<(&str, Value)> = vec![
            ("messages", to_value(messages)),
            (
                "tools",
                input
                    .tools
                    .as_ref()
                    .map(to_value)
                    .unwrap_or(Value::from(())),
            ),
            // `transformers` always passes `documents`, `None` unless given.
            ("documents", Value::from(())),
            (
                "add_generation_prompt",
                Value::from(options.add_generation_prompt),
            ),
        ];
        if let Some(enable_thinking) = options.enable_thinking {
            context.push(("enable_thinking", Value::from(enable_thinking)));
        }
        if let Some((open, close)) = marks {
            context.push((
                DATA_MARKS_VAR,
                Value::from(String::from_iter([open, close])),
            ));
        }
        let context: Value = context.into_iter().collect();
        template
            .render(context)
            .map_err(|e| ChatError::Template(format!("{e:#}")))
    }
}

/// Two private-use characters `text` does not contain.
fn unused_marks(text: &str) -> Result<(char, char), ChatError> {
    let private_use = |c: &char| matches!(*c as u32, 0xE000..=0xF8FF | 0xF0000..=0xFFFFD);
    let used: std::collections::HashSet<char> = text.chars().filter(private_use).collect();
    let mut free = ('\u{E000}'..='\u{F8FF}')
        .chain('\u{F0000}'..='\u{FFFFD}')
        .filter(|c| !used.contains(c));
    match (free.next(), free.next()) {
        (Some(open), Some(close)) => Ok((open, close)),
        _ => Err(ChatError::InvalidInput(
            "the conversation uses every private-use character".into(),
        )),
    }
}

/// `text` between the marks, or unchanged when empty (an empty string must stay
/// false to the template).
fn mark(text: &str, (open, close): (char, char)) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(text.len() + open.len_utf8() + close.len_utf8());
    out.push(open);
    out.push_str(text);
    out.push(close);
    out
}

fn mark_in_place(value: &mut Json, marks: (char, char)) {
    if let Json::Str(text) = value {
        *text = mark(text, marks);
    }
}

/// The conversation with every string the template prints verbatim marked: a
/// role other than `assistant` (which the template only compares), content
/// strings and text parts, `reasoning_content`, and a tool call's name, string
/// `input` or `arguments`, argument names and string argument values. What the
/// template prints through `tojson` (tool definitions, other argument values)
/// is marked by the filter.
fn marked_messages(messages: &Json, marks: (char, char)) -> Json {
    let mut messages = messages.clone();
    let Json::Array(items) = &mut messages else {
        return messages;
    };
    for message in items {
        let Json::Object(fields) = message else {
            continue;
        };
        for (key, value) in fields {
            match (key.as_str(), value) {
                ("role", Json::Str(role)) if role != "assistant" => *role = mark(role, marks),
                ("reasoning_content", value) => mark_in_place(value, marks),
                ("content", Json::Array(parts)) => {
                    for part in parts {
                        match part {
                            Json::Object(members) => {
                                for (key, value) in members {
                                    if key == "text" {
                                        mark_in_place(value, marks);
                                    }
                                }
                            }
                            part => mark_in_place(part, marks),
                        }
                    }
                }
                ("content", value) => mark_in_place(value, marks),
                ("tool_calls", Json::Array(calls)) => {
                    for call in calls {
                        mark_tool_call(call, marks);
                    }
                }
                _ => {}
            }
        }
    }
    messages
}

fn mark_tool_call(call: &mut Json, marks: (char, char)) {
    let key = tool_call_body_key(call);
    let Json::Object(call_fields) = call else {
        return;
    };
    let body = match key {
        Some(key) => match call_fields.iter_mut().find(|(k, _)| k == key) {
            Some((_, Json::Object(body))) => body,
            _ => return,
        },
        None => call_fields,
    };
    for (key, value) in body {
        match (key.as_str(), value) {
            ("name" | "input", value) => mark_in_place(value, marks),
            ("arguments", Json::Object(arguments)) => {
                for (name, value) in arguments {
                    *name = mark(name, marks);
                    mark_in_place(value, marks);
                }
            }
            ("arguments", value) => mark_in_place(value, marks),
            _ => {}
        }
    }
}

/// A marked render without its marks, and the byte ranges of the result that
/// lay between them. `None` if the marks do not pair up.
fn strip_marks(marked: &str, (open, close): (char, char)) -> Option<(String, Vec<Range<usize>>)> {
    let mut text = String::with_capacity(marked.len());
    let mut data = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for c in marked.chars() {
        if c == open {
            if depth == 0 {
                start = text.len();
            }
            depth += 1;
        } else if c == close {
            depth = depth.checked_sub(1)?;
            if depth == 0 && text.len() > start {
                data.push(start..text.len());
            }
        } else {
            text.push(c);
        }
    }
    (depth == 0).then_some((text, data))
}

/// The conversation with the sentinel appended to the final message's
/// `field`, and the text the reference looks for afterwards (the field's value
/// before the sentinel, or the chosen part's `text`).
fn continued_messages(
    messages: &Json,
    field: &str,
    source: &str,
) -> Result<(Json, String), ChatError> {
    let invalid = |m: String| Err(ChatError::InvalidInput(m));
    let mut messages = messages.clone();
    let Json::Array(items) = &mut messages else {
        return invalid("messages must be an array".into());
    };
    let Some(Json::Object(last)) = items.last_mut() else {
        return invalid("messages must not be empty".into());
    };
    let value = match last.iter_mut().find(|(k, _)| k == field) {
        Some((_, v)) if !matches!(v, Json::Null) => v,
        _ => {
            return invalid(format!(
                "continue_final_message is set but the final message has no \"{field}\" to \
                 continue"
            ));
        }
    };
    if !source.contains(field) {
        return invalid(format!(
            "continue_final_message is set to \"{field}\" but this is not an accepted field in \
             the chat_template"
        ));
    }
    let text = match value {
        Json::Str(text) => text,
        Json::Array(parts) => {
            // Python's `"text" in block` over the parts in reverse: a key test
            // on an object, a substring test on a string (whose `block["text"]`
            // then raises).
            let mut chosen = None;
            for part in parts.iter_mut().rev() {
                match part {
                    Json::Object(members) => {
                        if let Some((_, text)) = members.iter_mut().find(|(k, _)| k == "text") {
                            chosen = Some(text);
                            break;
                        }
                    }
                    Json::Str(s) if s.contains("text") => {
                        return invalid("a string content part cannot be continued".into());
                    }
                    _ => {}
                }
            }
            match chosen {
                Some(Json::Str(text)) => text,
                Some(_) => return invalid("a content part's text must be a string".into()),
                None => {
                    return invalid(
                        "continue_final_message is set but we could not find any text to \
                         continue in the final message"
                            .into(),
                    );
                }
            }
        }
        _ => return invalid(format!("the final message's \"{field}\" is not a string")),
    };
    let original = text.clone();
    text.push_str(CONTINUE_FINAL_MESSAGE_TAG);
    Ok((messages, original))
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// Checks that the input lies in the domain where minijinja's rendering of the
/// pinned template is proven identical to Jinja2's.
///
/// - `messages` is a non-empty array of objects, each with a string `role`.
/// - `content` (any role) is absent, `null`, a string, or an array whose items
///   are strings or objects; an object's `text`, when present, is a string.
/// - An assistant's `tool_calls` is absent, `null`, or an array of objects;
///   each call's `function` (or `custom`) is an object whose `name`, when
///   present, is a string and whose `arguments` is absent, `null`, a string,
///   or an object.
/// - `tools` (top-level, and per message) is absent, `null`, or an array.
///
/// Everything else (`reasoning_content`, tool schemas, argument values, extra
/// keys) is free-form: it either reaches the prompt through `tojson`, which is
/// exact for all JSON, or is ignored by the template.
pub fn validate(input: &ChatInput) -> Result<(), ChatError> {
    let invalid = |msg: String| Err(ChatError::InvalidInput(msg));
    if let Some(tools) = &input.tools
        && !matches!(tools, Json::Null | Json::Array(_))
    {
        return invalid("tools must be an array".into());
    }
    let Json::Array(messages) = &input.messages else {
        return invalid("messages must be an array".into());
    };
    if messages.is_empty() {
        return invalid("messages must not be empty".into());
    }
    for (i, message) in messages.iter().enumerate() {
        let Json::Object(_) = message else {
            return invalid(format!("messages[{i}] must be an object"));
        };
        let Some(Json::Str(role)) = message.get("role") else {
            return invalid(format!("messages[{i}].role must be a string"));
        };
        if let Some(content) = message.get("content") {
            validate_content(content, &format!("messages[{i}].content"))?;
        }
        if role == "assistant" {
            match message.get("tool_calls") {
                None | Some(Json::Null) => {}
                Some(Json::Array(calls)) => {
                    for (c, call) in calls.iter().enumerate() {
                        validate_tool_call(call, &format!("messages[{i}].tool_calls[{c}]"))?;
                    }
                }
                Some(_) => return invalid(format!("messages[{i}].tool_calls must be an array")),
            }
        } else if let Some(tools) = message.get("tools")
            && !matches!(tools, Json::Null | Json::Array(_))
        {
            return invalid(format!("messages[{i}].tools must be an array"));
        }
    }
    Ok(())
}

fn validate_content(content: &Json, path: &str) -> Result<(), ChatError> {
    match content {
        Json::Null | Json::Str(_) => Ok(()),
        Json::Array(parts) => {
            for (p, part) in parts.iter().enumerate() {
                match part {
                    Json::Str(_) => {}
                    Json::Object(_) => {
                        if let Some(text) = part.get("text")
                            && !matches!(text, Json::Str(_))
                        {
                            return Err(ChatError::InvalidInput(format!(
                                "{path}[{p}].text must be a string"
                            )));
                        }
                    }
                    _ => {
                        return Err(ChatError::InvalidInput(format!(
                            "{path}[{p}] must be a string or an object"
                        )));
                    }
                }
            }
            Ok(())
        }
        _ => Err(ChatError::InvalidInput(format!(
            "{path} must be a string, an array, or null"
        ))),
    }
}

/// Which member of a tool call holds its body, as the template selects it:
/// `function` if present, else `custom`, else `None` for the call object itself.
/// Validation and argument normalization both go through this, so they can
/// never disagree with each other or with rendering.
fn tool_call_body_key(call: &Json) -> Option<&'static str> {
    ["function", "custom"]
        .into_iter()
        .find(|key| call.get(key).is_some())
}

fn validate_tool_call(call: &Json, path: &str) -> Result<(), ChatError> {
    let invalid = |msg: String| Err(ChatError::InvalidInput(msg));
    let Json::Object(_) = call else {
        return invalid(format!("{path} must be an object"));
    };
    let (inner, inner_path) = match tool_call_body_key(call) {
        Some(key) => (
            call.get(key).expect("selected key"),
            format!("{path}.{key}"),
        ),
        None => (call, path.to_string()),
    };
    let Json::Object(_) = inner else {
        return invalid(format!("{inner_path} must be an object"));
    };
    if let Some(name) = inner.get("name")
        && !matches!(name, Json::Str(_))
    {
        return invalid(format!("{inner_path}.name must be a string"));
    }
    match inner.get("arguments") {
        None | Some(Json::Null | Json::Str(_) | Json::Object(_)) => Ok(()),
        Some(_) => invalid(format!(
            "{inner_path}.arguments must be a JSON object or a string"
        )),
    }
}

/// An integer too large for minijinja's native integers.
#[derive(Debug)]
struct BigIntValue(BigInt);

impl Object for BigIntValue {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Plain
    }

    fn render(self: &Arc<Self>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

fn to_value(json: &Json) -> Value {
    match json {
        Json::Null => Value::from(()),
        Json::Bool(b) => Value::from(*b),
        Json::Int(i) => match i.to_i128() {
            Some(v) => match i64::try_from(v) {
                Ok(small) => Value::from(small),
                Err(_) => Value::from(v),
            },
            None => Value::from_object(BigIntValue(i.clone())),
        },
        Json::Float(f) => Value::from(*f),
        Json::Str(s) => Value::from(s.as_str()),
        Json::Array(items) => items.iter().map(to_value).collect(),
        Json::Object(members) => members
            .iter()
            .map(|(k, v)| (Value::from(k.as_str()), to_value(v)))
            .collect(),
    }
}

fn from_value(value: &Value) -> Result<Json, JinjaError> {
    let unsupported = |what: &str| {
        JinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Object of type {what} is not JSON serializable"),
        )
    };
    if let Some(big) = value.downcast_object_ref::<BigIntValue>() {
        return Ok(Json::Int(big.0.clone()));
    }
    Ok(match value.kind() {
        ValueKind::Undefined => return Err(unsupported("Undefined")),
        ValueKind::None => Json::Null,
        ValueKind::Bool => Json::Bool(value.is_true()),
        ValueKind::Number => {
            if value.is_integer() {
                let v = i128::try_from(value.clone())?;
                BigInt::from_digits(v < 0, &v.unsigned_abs().to_string()).into()
            } else {
                Json::Float(f64::try_from(value.clone())?)
            }
        }
        ValueKind::String => Json::Str(value.as_str().unwrap_or_default().to_string()),
        ValueKind::Seq | ValueKind::Iterable => Json::Array(
            value
                .try_iter()?
                .map(|item| from_value(&item))
                .collect::<Result<_, _>>()?,
        ),
        ValueKind::Map => {
            let mut members = Vec::new();
            for key in value.try_iter()? {
                let Some(name) = key.as_str() else {
                    return Err(unsupported("non-string key"));
                };
                members.push((name.to_string(), from_value(&value.get_item(&key)?)?));
            }
            Json::Object(members)
        }
        _ => return Err(unsupported(&value.kind().to_string())),
    })
}

/// `tojson`, as `transformers` defines it:
/// `json.dumps(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)`.
///
/// In a marked render the output is request data, and goes between the marks.
fn tojson_filter(state: &State, value: &Value, kwargs: Kwargs) -> Result<String, JinjaError> {
    let mut options = DumpOptions::default();
    if let Some(v) = kwargs.get::<Option<Value>>("ensure_ascii")? {
        options.ensure_ascii = v.is_true();
    }
    if let Some(v) = kwargs.get::<Option<Value>>("sort_keys")? {
        options.sort_keys = v.is_true();
    }
    if let Some(v) = kwargs.get::<Option<Value>>("indent")? {
        options.indent = match v.kind() {
            ValueKind::None | ValueKind::Undefined => None,
            ValueKind::String => Some(v.as_str().unwrap_or_default().to_string()),
            ValueKind::Number if v.is_integer() => {
                let n = i64::try_from(v.clone())?;
                Some(" ".repeat(n.max(0) as usize))
            }
            _ => {
                return Err(JinjaError::new(
                    ErrorKind::InvalidOperation,
                    "indent must be None, an integer, or a string",
                ));
            }
        };
        if options.indent.is_some() {
            // json.dumps: with an indent, the default item separator drops
            // its trailing space.
            options.item_separator = ",".to_string();
        }
    }
    if let Some(v) = kwargs.get::<Option<Value>>("separators")?
        && !v.is_none()
    {
        let parts: Vec<Value> = v.try_iter()?.collect();
        match parts.as_slice() {
            [item, key] if item.as_str().is_some() && key.as_str().is_some() => {
                options.item_separator = item.as_str().unwrap_or_default().to_string();
                options.key_separator = key.as_str().unwrap_or_default().to_string();
            }
            _ => {
                return Err(JinjaError::new(
                    ErrorKind::InvalidOperation,
                    "separators must be a pair of strings",
                ));
            }
        }
    }
    kwargs.assert_all_used()?;
    let dumped = json::dumps(&from_value(value)?, &options);
    let marks = state.lookup(DATA_MARKS_VAR);
    let mut marks = marks.as_ref().and_then(Value::as_str).unwrap_or("").chars();
    Ok(match (marks.next(), marks.next()) {
        (Some(open), Some(close)) => mark(&dumped, (open, close)),
        _ => dumped,
    })
}

/// Jinja2's `iterable` test is Python's `iter(value)`: `None`, booleans and
/// numbers are not iterable, and `Undefined` is (it iterates as empty).
/// minijinja's built-in test also accepts `none`, which would let the
/// template call `length` on a missing `tools` list.
fn is_iterable(value: &Value) -> bool {
    match value.kind() {
        ValueKind::None | ValueKind::Bool | ValueKind::Number => false,
        ValueKind::Undefined => true,
        _ => value.try_iter().is_ok(),
    }
}

fn raise_exception(message: String) -> Result<Value, JinjaError> {
    Err(JinjaError::new(ErrorKind::InvalidOperation, message))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unpinned_templates() {
        let err = ChatTemplate::from_source("{{ messages }}".to_string()).unwrap_err();
        assert!(matches!(err, ChatError::UnpinnedArtifact { .. }));
    }
}
