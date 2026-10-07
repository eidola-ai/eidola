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

use std::path::Path;
use std::sync::Arc;

use minijinja::value::{Kwargs, Object, ObjectRepr, Value, ValueKind};
use minijinja::{AutoEscape, Environment, Error as JinjaError, ErrorKind, UndefinedBehavior};
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
        env.add_template_owned(TEMPLATE_NAME, source)
            .map_err(|e| ChatError::Template(e.to_string()))?;
        Ok(ChatTemplate { env, sha256 })
    }

    /// Hex SHA-256 of the template source.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Renders the prompt text.
    pub fn render(&self, input: &ChatInput, options: RenderOptions) -> Result<String, ChatError> {
        validate(input)?;
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| ChatError::Template(e.to_string()))?;
        let mut context: Vec<(&str, Value)> = vec![
            ("messages", to_value(&input.messages)),
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
        let context: Value = context.into_iter().collect();
        template
            .render(context)
            .map_err(|e| ChatError::Template(format!("{e:#}")))
    }
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
fn tojson_filter(value: &Value, kwargs: Kwargs) -> Result<String, JinjaError> {
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
    Ok(json::dumps(&from_value(value)?, &options))
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
