//! The Jinja execution environment of a model's own chat template, shared by
//! the formats that render one ([`crate::mimo`], [`crate::qwen`]).
//!
//! minijinja is configured like the environment `transformers` builds for
//! `apply_chat_template` (an immutable sandbox with `trim_blocks`,
//! `lstrip_blocks`, loop controls, `raise_exception`, and a `tojson` that is
//! Python's `json.dumps` rather than Jinja's HTML-escaping filter). That is
//! enough only for a pinned template whose fixture corpus proves it: each
//! format adds what its template needs and validates its input to the domain
//! where the two engines agree.
//!
//! A prompt's request data is found by rendering twice ([`render_prompt`]):
//! once as is, and once with every string the template prints from the
//! request between a pair of marks, which `tojson` also puts around its
//! output.

use std::ops::Range;
use std::sync::Arc;

use minijinja::value::{Kwargs, Object, ObjectRepr, Value, ValueKind};
use minijinja::{
    AutoEscape, Environment, Error as JinjaError, ErrorKind, State, UndefinedBehavior,
};

use crate::error::ChatError;
use crate::json::{self, BigInt, DumpOptions, Json};
use crate::prompt::Prompt;

/// Context variable through which a render tells the filters the pair of marks
/// to put around request data. Templates never read it.
pub(crate) const DATA_MARKS_VAR: &str = "eidola_data_marks";

/// An environment configured as `transformers` configures one, before any
/// template is added.
pub(crate) fn transformers_environment() -> Result<Environment<'static>, ChatError> {
    let mut env = Environment::new();
    // `transformers` turns on `trim_blocks` and `lstrip_blocks` and keeps
    // jinja2's other defaults: `{{ }}` / `{% %}` / `{# #}` delimiters, no
    // line statements, and `keep_trailing_newline` off.
    let syntax = env
        .syntax()
        .to_builder()
        .trim_blocks(true)
        .lstrip_blocks(true)
        .keep_trailing_newline(false)
        .build()
        .map_err(|e| ChatError::Template(e.to_string()))?;
    env.set_syntax(syntax);
    env.set_undefined_behavior(UndefinedBehavior::Lenient);
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.add_filter("tojson", tojson_filter);
    env.add_test("iterable", is_iterable);
    env.add_function("raise_exception", raise_exception);
    Ok(env)
}

/// Renders a prompt and finds its request data: `render(None)` is the text,
/// and `render(Some(marks))` must render the same conversation with every
/// string the template prints from the request between the marks (two
/// characters the text does not contain). The marked render with its marks
/// removed must be the text itself, or the prompt is refused, so marking can
/// never change what the prompt says.
pub(crate) fn render_prompt(
    render: impl Fn(Option<(char, char)>) -> Result<String, ChatError>,
) -> Result<Prompt, ChatError> {
    let text = render(None)?;
    let marks = unused_marks(&text)?;
    let marked = render(Some(marks))?;
    match strip_marks(&marked, marks) {
        Some((stripped, data)) if stripped == text => Ok(Prompt::from_parts(text, data)),
        _ => Err(ChatError::Template(
            "the prompt's request data could not be told from the template's text".into(),
        )),
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
pub(crate) fn mark(text: &str, (open, close): (char, char)) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(text.len() + open.len_utf8() + close.len_utf8());
    out.push(open);
    out.push_str(text);
    out.push(close);
    out
}

pub(crate) fn mark_in_place(value: &mut Json, marks: (char, char)) {
    if let Json::Str(text) = value {
        *text = mark(text, marks);
    }
}

/// The marks of the render `state` belongs to, if it is a marked one.
pub(crate) fn data_marks(state: &State) -> Option<(char, char)> {
    let marks = state.lookup(DATA_MARKS_VAR)?;
    let mut marks = marks.as_str()?.chars();
    Some((marks.next()?, marks.next()?))
}

/// The context entry that makes a render a marked one.
pub(crate) fn data_marks_entry((open, close): (char, char)) -> (&'static str, Value) {
    (
        DATA_MARKS_VAR,
        Value::from(String::from_iter([open, close])),
    )
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

/// A JSON value as a template sees it. Integers beyond `i128` travel as an
/// opaque object that only `tojson` reads.
pub(crate) fn to_value(json: &Json) -> Value {
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
        Json::Object(members) => Value::from_pairs(
            members
                .iter()
                .map(|(k, v)| (Value::from(k.as_str()), to_value(v))),
        ),
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
    Ok(match data_marks(state) {
        Some(marks) => mark(&dumped, marks),
        None => dumped,
    })
}

/// Jinja2's `iterable` test is Python's `iter(value)`: `None`, booleans and
/// numbers are not iterable, and `Undefined` is (it iterates as empty).
/// minijinja's built-in test also accepts `none`, which would let a template
/// call `length` on a missing `tools` list.
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
