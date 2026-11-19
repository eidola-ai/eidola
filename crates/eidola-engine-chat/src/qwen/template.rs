//! Qwen3.8's chat-template rendering, byte-identical to Python
//! `transformers`.
//!
//! The template is the model's own `chat_template.jinja`, executed by
//! minijinja in the environment the crate's `jinja` module builds, plus what this
//! template needs of Python's semantics:
//!
//! - **`{% generation %}`** is `transformers`' own tag (it records which
//!   tokens are the assistant's, and otherwise renders its body as is). It is
//!   not Jinja, so its two tags are rewritten to `{% if true %}` /
//!   `{% endif %}` before compiling, which renders the body the same way.
//! - **`trim`** is Python's `str.strip()`, which strips more characters than
//!   Rust's `trim` (U+001C to U+001F).
//! - **`startswith` / `endswith`** are Python's string methods.
//!
//! Inputs are validated against the domain where the two engines agree
//! ([`validate`]); images and videos are a typed refusal
//! ([`ChatError::UnsupportedContent`]), since the model is served text only.
//!
//! The template trims every message's content and reasoning, and branches on
//! what a user message's trimmed content starts with. A marked render (which
//! finds the request data) therefore treats
//! the marks as absent in those three operations, so that it takes the same
//! branches as the plain render.

use std::path::Path;

use minijinja::value::{Value, ValueKind};
use minijinja::{Environment, Error as JinjaError, ErrorKind, State};

use crate::error::ChatError;
use crate::jinja::{self, data_marks, mark, mark_in_place, to_value};
use crate::json::{self, Json};
use crate::prompt::{ChatInput, Prompt, sha256_hex};
use crate::pysem::is_py_space;

pub use crate::mimo::template::CHAT_TEMPLATE_FILE;

/// SHA-256 of the Qwen3.8-27B chat template.
pub const QWEN3_8_CHAT_TEMPLATE_SHA256: &str =
    "af16f654968d47d09a6c604e00370b12dbc21f76456f59c9e98b82e84d9f6375";

/// Every Qwen template this crate will execute.
pub const PINNED_TEMPLATE_SHA256: &[&str] = &[QWEN3_8_CHAT_TEMPLATE_SHA256];

const TEMPLATE_NAME: &str = "chat_template";

/// `transformers`' `{% generation %}` tags, and what they become.
const GENERATION_TAGS: [(&str, &str); 2] = [
    ("{%- generation %}", "{%- if true %}"),
    ("{%- endgeneration %}", "{%- endif %}"),
];

/// Rendering switches, mirroring the keyword arguments of
/// `apply_chat_template` the template reads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderOptions {
    /// Append the assistant header that starts a new turn
    /// (`<|im_start|>assistant\n<think>\n`, or with thinking off
    /// `<|im_start|>assistant\n<think>\n\n</think>\n\n`).
    pub add_generation_prompt: bool,
    /// `enable_thinking`; `None` leaves it undefined, which means on. Thinking
    /// on also writes the reasoning effort's instructions into a system turn.
    pub enable_thinking: Option<bool>,
    /// `preserve_thinking`; `None` leaves it undefined, which means on: every
    /// past assistant turn keeps its reasoning. `Some(false)` drops the
    /// reasoning of turns before the last user query.
    pub preserve_thinking: Option<bool>,
    /// `reasoning_effort`, passed as given: the template accepts `xhigh` (its
    /// default), `medium` and `low`, and raises on anything else.
    pub reasoning_effort: Option<String>,
}

/// A pinned Qwen3.8 chat template, ready to render.
pub struct ChatTemplate {
    env: Environment<'static>,
    sha256: String,
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
        let mut source = source;
        for (tag, replacement) in GENERATION_TAGS {
            if source.matches(tag).count() != 1 {
                return Err(ChatError::InvalidArtifact(format!(
                    "chat template: expected one {tag}"
                )));
            }
            source = source.replace(tag, replacement);
        }
        let mut env = jinja::transformers_environment()?;
        env.add_filter("trim", py_trim);
        env.set_unknown_method_callback(py_string_method);
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(|e| ChatError::Template(e.to_string()))?;
        Ok(ChatTemplate { env, sha256 })
    }

    /// Hex SHA-256 of the template source.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Renders the prompt text ([`Self::render_prompt`] without the data spans).
    pub fn render(&self, input: &ChatInput, options: &RenderOptions) -> Result<String, ChatError> {
        self.render_prompt(input, options).map(Prompt::into_text)
    }

    /// Renders the prompt, with the parts of it that are request data marked.
    ///
    /// History tool-call `arguments` sent as a JSON string (as OpenAI clients
    /// send them) are decoded first, and must decode to an object; an empty
    /// string stays, and renders no parameters, as `{}` does. The template
    /// renders no string arguments (Python raises on one), so decoding never
    /// changes a prompt the reference renders.
    pub fn render_prompt(
        &self,
        input: &ChatInput,
        options: &RenderOptions,
    ) -> Result<Prompt, ChatError> {
        let mut messages = input.messages.clone();
        decode_tool_call_arguments(&mut messages)?;
        validate(&messages, input.tools.as_ref())?;
        jinja::render_prompt(|marks| match marks {
            None => self.render_messages(&messages, input.tools.as_ref(), options, None),
            Some(marks) => self.render_messages(
                &marked_messages(&messages, marks),
                input.tools.as_ref(),
                options,
                Some(marks),
            ),
        })
    }

    fn render_messages(
        &self,
        messages: &Json,
        tools: Option<&Json>,
        options: &RenderOptions,
        marks: Option<(char, char)>,
    ) -> Result<String, ChatError> {
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| ChatError::Template(e.to_string()))?;
        let mut context: Vec<(&str, Value)> = vec![
            ("messages", to_value(messages)),
            ("tools", tools.map(to_value).unwrap_or(Value::from(()))),
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
        if let Some(preserve_thinking) = options.preserve_thinking {
            context.push(("preserve_thinking", Value::from(preserve_thinking)));
        }
        if let Some(effort) = &options.reasoning_effort {
            context.push(("reasoning_effort", Value::from(effort.as_str())));
        }
        if let Some(marks) = marks {
            context.push(jinja::data_marks_entry(marks));
        }
        template
            .render(Value::from_pairs(context))
            .map_err(|e| ChatError::Template(format!("{e:#}")))
    }
}

/// Which member of a tool call holds its body, as the template selects it:
/// `function` if present, else the call object itself.
fn tool_call_body(call: &mut Json) -> Option<&mut Vec<(String, Json)>> {
    let Json::Object(fields) = call else {
        return None;
    };
    if fields.iter().any(|(k, _)| k == "function") {
        match fields.iter_mut().find(|(k, _)| k == "function") {
            Some((_, Json::Object(body))) => Some(body),
            _ => None,
        }
    } else {
        Some(fields)
    }
}

/// Decodes every assistant tool call's non-empty string `arguments` into the
/// object it encodes.
fn decode_tool_call_arguments(messages: &mut Json) -> Result<(), ChatError> {
    let Json::Array(messages) = messages else {
        return Ok(());
    };
    for (m, message) in messages.iter_mut().enumerate() {
        if message.get("role").and_then(Json::as_str) != Some("assistant") {
            continue;
        }
        let Json::Object(fields) = message else {
            continue;
        };
        let Some((_, Json::Array(calls))) = fields.iter_mut().find(|(k, _)| k == "tool_calls")
        else {
            continue;
        };
        for (c, call) in calls.iter_mut().enumerate() {
            let Some(body) = tool_call_body(call) else {
                continue;
            };
            let Some((_, arguments)) = body.iter_mut().find(|(k, _)| k == "arguments") else {
                continue;
            };
            let Json::Str(text) = arguments else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            let parsed = json::parse(text).map_err(|e| {
                ChatError::InvalidInput(format!(
                    "messages[{m}].tool_calls[{c}] arguments are not valid JSON: {e}"
                ))
            })?;
            if !matches!(parsed, Json::Object(_)) {
                return Err(ChatError::InvalidInput(format!(
                    "messages[{m}].tool_calls[{c}] arguments must be a JSON object"
                )));
            }
            *arguments = parsed;
        }
    }
    Ok(())
}

/// The conversation with every string the template prints verbatim marked:
/// content strings and text parts, `reasoning_content`, and a tool call's
/// name, argument names and string argument values. What it prints through
/// `tojson` (tool definitions, other argument values) is marked by the filter.
/// Roles are not data: validation holds them to the four the template knows,
/// and it prints only `user` and `assistant`.
fn marked_messages(messages: &Json, marks: (char, char)) -> Json {
    let mut messages = messages.clone();
    let Json::Array(items) = &mut messages else {
        return messages;
    };
    for message in items {
        let Json::Object(fields) = message else {
            continue;
        };
        for (key, value) in fields.iter_mut() {
            match (key.as_str(), value) {
                ("reasoning_content", value) => mark_in_place(value, marks),
                ("content", Json::Array(parts)) => {
                    for part in parts {
                        if let Json::Object(members) = part {
                            for (key, value) in members {
                                if key == "text" {
                                    mark_in_place(value, marks);
                                }
                            }
                        }
                    }
                }
                ("content", value) => mark_in_place(value, marks),
                ("tool_calls", Json::Array(calls)) => {
                    for call in calls {
                        let Some(body) = tool_call_body(call) else {
                            continue;
                        };
                        for (key, value) in body {
                            match (key.as_str(), value) {
                                ("name", value) => mark_in_place(value, marks),
                                ("arguments", Json::Object(arguments)) => {
                                    for (name, value) in arguments {
                                        *name = mark(name, marks);
                                        mark_in_place(value, marks);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    messages
}

/// Checks that the input lies in the domain where minijinja's rendering of the
/// pinned template is proven identical to Jinja2's (after string `arguments`
/// are decoded).
///
/// - `messages` is a non-empty array of objects, each with a `role` of
///   `system`, `user`, `assistant` or `tool` (the template raises on any
///   other).
/// - `content` (any role) is absent, `null`, a string, or an array of
///   objects with a string `text`. A part that is an image or a video (an
///   `image`, `image_url` or `video` member, or that `type`) is
///   [`ChatError::UnsupportedContent`]; Python would render it as a vision
///   placeholder.
/// - An assistant's `tool_calls` is absent, `null`, or an array of objects;
///   each call's body (its `function`, which must be an object, or else the
///   call itself) has a string `name` and `arguments` that are absent, an
///   object, or the empty string. (Python raises on a missing name and on
///   other arguments.)
/// - `tools` is absent, `null`, or an array of objects (`apply_chat_template`
///   refuses any other tool).
///
/// Everything else (`reasoning_content`, tool schemas, argument values, extra
/// keys) is free-form: it either reaches the prompt through `tojson`, which is
/// exact for all JSON, or is ignored by the template. Where the conversation
/// breaks the template's own rules (a system message after the first, no user
/// query, a reasoning effort it does not know), both engines raise.
pub fn validate(messages: &Json, tools: Option<&Json>) -> Result<(), ChatError> {
    let invalid = |msg: String| Err(ChatError::InvalidInput(msg));
    match tools {
        None | Some(Json::Null) => {}
        Some(Json::Array(tools)) => {
            if tools.iter().any(|tool| !matches!(tool, Json::Object(_))) {
                return invalid("tools must be objects".into());
            }
        }
        Some(_) => return invalid("tools must be an array".into()),
    }
    let Json::Array(messages) = messages else {
        return invalid("messages must be an array".into());
    };
    if messages.is_empty() {
        return invalid("messages must not be empty".into());
    }
    for (i, message) in messages.iter().enumerate() {
        let Json::Object(_) = message else {
            return invalid(format!("messages[{i}] must be an object"));
        };
        let role = match message.get("role") {
            Some(Json::Str(role)) if ["system", "user", "assistant", "tool"].contains(&&**role) => {
                role
            }
            _ => {
                return invalid(format!(
                    "messages[{i}].role must be system, user, assistant or tool"
                ));
            }
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
        }
    }
    Ok(())
}

fn validate_content(content: &Json, path: &str) -> Result<(), ChatError> {
    let invalid = |msg: String| Err(ChatError::InvalidInput(msg));
    match content {
        Json::Null | Json::Str(_) => Ok(()),
        Json::Array(parts) => {
            for (p, part) in parts.iter().enumerate() {
                let Json::Object(_) = part else {
                    return invalid(format!("{path}[{p}] must be an object"));
                };
                let typed = |kind: &str| part.get("type").and_then(Json::as_str) == Some(kind);
                if ["image", "image_url", "video"]
                    .iter()
                    .any(|key| part.get(key).is_some())
                    || typed("image")
                    || typed("video")
                {
                    return Err(ChatError::UnsupportedContent(format!(
                        "{path}[{p}] is an image or a video; the model is served text only"
                    )));
                }
                match part.get("text") {
                    Some(Json::Str(_)) => {}
                    Some(_) => return invalid(format!("{path}[{p}].text must be a string")),
                    None => return invalid(format!("{path}[{p}] has no text")),
                }
            }
            Ok(())
        }
        _ => invalid(format!("{path} must be a string, an array, or null")),
    }
}

fn validate_tool_call(call: &Json, path: &str) -> Result<(), ChatError> {
    let invalid = |msg: String| Err(ChatError::InvalidInput(msg));
    let Json::Object(_) = call else {
        return invalid(format!("{path} must be an object"));
    };
    let (body, path) = match call.get("function") {
        Some(function) => (function, format!("{path}.function")),
        None => (call, path.to_string()),
    };
    let Json::Object(_) = body else {
        return invalid(format!("{path} must be an object"));
    };
    let Some(Json::Str(_)) = body.get("name") else {
        return invalid(format!("{path}.name must be a string"));
    };
    match body.get("arguments") {
        None | Some(Json::Object(_)) => Ok(()),
        Some(Json::Str(s)) if s.is_empty() => Ok(()),
        Some(_) => invalid(format!("{path}.arguments must be a JSON object")),
    }
}

/// `text` without the marks of a marked render.
fn unmarked(text: &str, marks: Option<(char, char)>) -> String {
    match marks {
        Some((open, close)) => text.chars().filter(|&c| c != open && c != close).collect(),
        None => text.to_string(),
    }
}

/// Jinja2's `trim`: Python's `str.strip()`. In a marked render the marks are
/// kept where they are and stripping sees through them; a string that strips
/// to nothing is empty, marks and all, so it stays false.
fn py_trim(state: &State, value: &Value) -> Result<Value, JinjaError> {
    let Some(text) = value.as_str() else {
        return Err(JinjaError::new(
            ErrorKind::InvalidOperation,
            format!("trim of a {}", value.kind()),
        ));
    };
    let marks = data_marks(state);
    let is_mark = |c: char| marks.is_some_and(|(open, close)| c == open || c == close);
    let keep = |c: char| !is_mark(c) && !is_py_space(c);
    let (Some(first), Some(last)) = (
        text.char_indices().find(|&(_, c)| keep(c)).map(|(i, _)| i),
        text.char_indices().rev().find(|&(_, c)| keep(c)),
    ) else {
        return Ok(Value::from(""));
    };
    let end = last.0 + last.1.len_utf8();
    let mut out: String = text[..first].chars().filter(|&c| is_mark(c)).collect();
    out.push_str(&text[first..end]);
    out.extend(text[end..].chars().filter(|&c| is_mark(c)));
    Ok(Value::from(out))
}

/// Python's `str.startswith` and `str.endswith` with one string argument,
/// seeing through the marks of a marked render.
fn py_string_method(
    state: &mut State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, JinjaError> {
    let (Some(text), [affix]) = (value.as_str(), args) else {
        return Err(JinjaError::from(ErrorKind::UnknownMethod));
    };
    let (Some(affix), true) = (affix.as_str(), affix.kind() == ValueKind::String) else {
        return Err(JinjaError::from(ErrorKind::UnknownMethod));
    };
    let text = unmarked(text, data_marks(state));
    match method {
        "startswith" => Ok(Value::from(text.starts_with(affix))),
        "endswith" => Ok(Value::from(text.ends_with(affix))),
        _ => Err(JinjaError::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unpinned_templates() {
        let err = ChatTemplate::from_source("{{ messages }}".to_string()).unwrap_err();
        assert!(matches!(err, ChatError::UnpinnedArtifact { .. }));
    }

    #[test]
    fn python_strip_and_affix_tests() {
        let env = {
            let mut env = jinja::transformers_environment().unwrap();
            env.add_filter("trim", py_trim);
            env.set_unknown_method_callback(py_string_method);
            env
        };
        let render = |source: &str, s: &str, marks: Option<(char, char)>| {
            let mut context = vec![("s", Value::from(s))];
            if let Some(marks) = marks {
                context.push(jinja::data_marks_entry(marks));
            }
            env.render_str(source, Value::from_pairs(context)).unwrap()
        };
        // U+001C..U+001F are Python whitespace, not Rust's.
        assert_eq!(render("{{ s|trim }}", "\u{1c} a b\u{1f}\n", None), "a b");
        let marks = Some(('\u{E000}', '\u{E001}'));
        assert_eq!(
            render(
                "{{ s|trim }}",
                " \u{E000} a \u{E001}\u{E000}b \u{E001} ",
                marks
            ),
            "\u{E000}a \u{E001}\u{E000}b\u{E001}"
        );
        assert_eq!(render("{{ s|trim }}|", "\u{E000} \n\u{E001}", marks), "|");
        assert_eq!(
            render(
                "{{ s.startswith('<a>') }} {{ s.endswith('</a>') }}",
                "\u{E000}<a>x</a>\u{E001}",
                marks
            ),
            "True True"
        );
        assert_eq!(render("{{ s.startswith('<a>') }}", "<b><a>", None), "False");
    }
}
