//! Schema-typed conversion of tool-call parameter values.
//!
//! MiMo writes every parameter as text (`<parameter=K>V</parameter>`); the
//! JSON type is recovered from the tool's schema.
//!
//! # The rule: conversion inverts the chat template
//!
//! The template renders a string argument verbatim and every other value as
//! `json.dumps(value)`. The parser applies the exact inverse
//! ([`convert_parameter_round_trip`]), so for arguments that conform to the
//! tool schema:
//!
//! - `parse(render(arguments)) == arguments`: the tool receives exactly what
//!   the model wrote; and
//! - when the client sends the call back as history, the next turn's prompt
//!   renders byte for byte what the model generated, so the conversation
//!   stays on the trained distribution and its KV prefix stays reusable.
//!
//! Concretely:
//!
//! 1. The declared type is the `type` of the parameter's property in the
//!    tool's `parameters.properties` (or, when the schema has no top-level
//!    `properties`, the first match among its `anyOf`/`oneOf`/`allOf`
//!    branches). Type names are normalised: case-folded, `varchar(255)`
//!    style parameters dropped, database aliases mapped (`text`, `uuid`,
//!    `date`, … → `string`; `bigint`, `int32`, … → `integer`; `double`,
//!    `float64`, … → `number`; `bool` → `boolean`; `list[...]`, `tuple`,
//!    `set` → `array`; `dict[...]`, `map` → `object`). An array of types is
//!    `string` if it contains `string`, else its first non-`null` entry.
//!    Undeclared parameters, properties without a usable `type`, and
//!    property schemas that are not objects are `string`.
//! 2. A string-typed value is the text, verbatim — no unescaping, no
//!    trimming; the words `null` and `None` stay strings.
//! 3. Any other value is the text parsed as strict JSON when it is valid
//!    JSON (whatever JSON type results; validating it against the schema is
//!    the tool's job).
//! 4. Otherwise a lenient spelling for the declared type is accepted, as
//!    Python reads it: `int(text)` for integer types, `float(text)` for
//!    number types (finite only; integral values become integers), `true` /
//!    `false` in any case for booleans, and `ast.literal_eval(text)` (tuples
//!    become arrays) for anything else. These never apply to text the
//!    template could have produced, so they cannot break the round trip.
//! 5. Otherwise the value is the text.
//!
//! The output is always a JSON object of finite values. JSON nesting deeper
//! than 256 and literal nesting deeper than 200 are not decoded.
//!
//! # Why not SGLang's rule
//!
//! SGLang's MiMo detector (`function_call/mimo_detector.py`) also types by
//! schema, but its conversion is lossy and does not invert the template:
//!
//! - it runs `html.unescape` on text the template wrote verbatim, so a string
//!   `&amp;` arrives as `&`, a URL's `?a=1&copy=2` as `?a=1©=2` (HTML5 legacy
//!   entities need no `;`), and `&lt;div&gt;` in a file being written as
//!   `<div>`;
//! - `null` in any case becomes JSON `null`, even for a string parameter;
//! - any word other than `true` for a boolean becomes `false` (`yes` →
//!   `false`);
//! - a number written `4.0` becomes `4`;
//! - it fails the whole request on valid schemas and outputs (a list-valued
//!   `type`, a boolean property schema, `1e400`), and can emit the non-JSON
//!   tokens `NaN` / `Infinity`.
//!
//! Every coercion above also makes the re-rendered history differ from what
//! the model generated. That rule is kept as [`convert_parameter_sglang`]
//! for one purpose: it is differentially tested against SGLang's own code,
//! which verifies the Python-semantics pieces (`int`, `float`,
//! `ast.literal_eval`, type normalisation) that rule 4 shares with it. The
//! parser itself uses it only when a test selects [`ArgumentTyping::Sglang`].

use crate::json::{self, BigInt, Json, ObjectBuilder, PY_MAX_STR_DIGITS};
use crate::pysem::{self, PyLit};

/// The tools of a request, indexed for parameter typing.
#[derive(Clone, Debug, Default)]
pub struct ToolSchemas {
    tools: Vec<ToolEntry>,
    typing: ArgumentTyping,
}

/// How parameter text becomes a JSON value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArgumentTyping {
    /// The rule (module docs): the exact inverse of the template.
    #[default]
    RoundTrip,
    /// SGLang's lenient rule, for differential tests against SGLang only.
    Sglang,
}

#[derive(Clone, Debug)]
struct ToolEntry {
    name: String,
    /// `get_schema_properties(parameters)`.
    properties: Vec<(String, Json)>,
}

impl ToolSchemas {
    /// Indexes the request's `tools` array. Entries that are not
    /// `{"function": {"name": "...", ...}}` objects are ignored.
    pub fn from_tools(tools: &Json) -> ToolSchemas {
        let mut out = Vec::new();
        for tool in tools.as_array().unwrap_or_default() {
            let Some(function) = tool.get("function") else {
                continue;
            };
            let Some(name) = function.get("name").and_then(Json::as_str) else {
                continue;
            };
            let properties = function
                .get("parameters")
                .map(schema_properties)
                .unwrap_or_default();
            out.push(ToolEntry {
                name: name.to_string(),
                properties,
            });
        }
        ToolSchemas {
            tools: out,
            typing: ArgumentTyping::default(),
        }
    }

    /// Selects the conversion rule.
    pub fn with_typing(mut self, typing: ArgumentTyping) -> ToolSchemas {
        self.typing = typing;
        self
    }

    pub fn typing(&self) -> ArgumentTyping {
        self.typing
    }

    /// Converts the raw text of `param` in a call to `function`.
    pub fn convert(&self, function: &str, param: &str, raw: &str) -> Json {
        let t = self.param_type(function, param);
        match self.typing {
            ArgumentTyping::Sglang => convert_parameter_sglang(raw, &t),
            ArgumentTyping::RoundTrip => convert_parameter_round_trip(raw, &t),
        }
    }

    /// Whether a tool of this name was declared.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name)
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The declared type of a parameter, resolved to one type name.
    pub fn param_type(&self, function: &str, param: &str) -> String {
        let Some(tool) = self.tools.iter().find(|t| t.name == function) else {
            return "string".to_string();
        };
        let Some((_, schema)) = tool.properties.iter().find(|(k, _)| k == param) else {
            return "string".to_string();
        };
        let Json::Object(_) = schema else {
            return "string".to_string();
        };
        match schema.get("type") {
            None => "string".to_string(),
            Some(Json::Str(t)) => normalize_type_name(t),
            Some(Json::Array(types)) => {
                let mut normalized: Vec<String> = Vec::new();
                for t in types.iter().filter_map(Json::as_str) {
                    let n = normalize_type_name(t);
                    if !normalized.contains(&n) {
                        normalized.push(n);
                    }
                }
                if normalized.iter().any(|t| t == "string") {
                    "string".to_string()
                } else {
                    normalized
                        .into_iter()
                        .find(|t| t != "null")
                        .unwrap_or_else(|| "string".to_string())
                }
            }
            Some(_) => "string".to_string(),
        }
    }
}

/// SGLang's `get_schema_properties`: top-level `properties`, else the merged
/// properties of `anyOf`/`oneOf`/`allOf` branches (first definition wins).
fn schema_properties(schema: &Json) -> Vec<(String, Json)> {
    let Json::Object(_) = schema else {
        return Vec::new();
    };
    if let Some(Json::Object(props)) = schema.get("properties") {
        return props.clone();
    }
    let mut merged: Vec<(String, Json)> = Vec::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(Json::Array(branches)) = schema.get(keyword) {
            for branch in branches {
                for (k, v) in schema_properties(branch) {
                    if !merged.iter().any(|(m, _)| *m == k) {
                        merged.push((k, v));
                    }
                }
            }
        }
    }
    merged
}

const STANDARD_TYPES: &[&str] = &[
    "null", "boolean", "object", "array", "number", "string", "integer",
];

const TYPE_ALIASES: &[(&str, &str)] = &[
    ("str", "string"),
    ("text", "string"),
    ("varchar", "string"),
    ("char", "string"),
    ("enum", "string"),
    ("uuid", "string"),
    ("date", "string"),
    ("datetime", "string"),
    ("time", "string"),
    ("timestamp", "string"),
    ("binary", "string"),
    ("blob", "string"),
    ("bytea", "string"),
    ("bytes", "string"),
    ("varbinary", "string"),
    ("bool", "boolean"),
    ("bigint", "integer"),
    ("smallint", "integer"),
    ("tinyint", "integer"),
    ("double", "number"),
    ("decimal", "number"),
    ("real", "number"),
    ("numeric", "number"),
    ("arr", "array"),
    ("tuple", "array"),
    ("set", "array"),
    ("map", "object"),
];

const PREFIX_RULES: &[(&[&str], &str)] = &[
    (&["int", "uint", "long", "short", "unsigned"], "integer"),
    (&["num", "float"], "number"),
    (&["list"], "array"),
    (&["dict"], "object"),
];

/// SGLang's `_normalize_single_type`.
pub fn normalize_type_name(raw: &str) -> String {
    if STANDARD_TYPES.contains(&raw) {
        return raw.to_string();
    }
    let base = pysem::py_strip(raw.split('(').next().unwrap_or_default()).to_lowercase();
    if STANDARD_TYPES.contains(&base.as_str()) {
        return base;
    }
    if let Some((_, mapped)) = TYPE_ALIASES.iter().find(|(alias, _)| *alias == base) {
        return (*mapped).to_string();
    }
    for (prefixes, target) in PREFIX_RULES {
        for p in *prefixes {
            if base == *p
                || (base.len() > p.len()
                    && base.starts_with(p)
                    && base[p.len()..].starts_with([
                        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '[', '<', '(', ' ', '\t',
                    ]))
            {
                return (*target).to_string();
            }
        }
    }
    raw.to_string()
}

fn starts_with_any(t: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| t.starts_with(p))
}

const STRING_TYPES: &[&str] = &["string", "str", "text", "varchar", "char", "enum"];
const INT_PREFIXES: &[&str] = &["int", "integer", "uint", "long", "short", "unsigned"];
const NUMBER_PREFIXES: &[&str] = &["num", "float"];
const BOOL_TYPES: &[&str] = &["boolean", "bool", "binary"];

fn is_json_container_type(t: &str) -> bool {
    ["object", "array", "arr"].contains(&t) || starts_with_any(t, &["dict", "list"])
}

/// Python `float(text)`, as SGLang emits it: integral values become
/// integers. `None` where SGLang falls back (or, for infinities, fails).
fn sglang_number(text: &str) -> Option<Json> {
    match pysem::py_float(text) {
        Ok(f) if f.is_finite() => Some(if f.fract() != 0.0 {
            Json::Float(f)
        } else {
            Json::Int(BigInt::from_integral_f64(f))
        }),
        // NaN: `f - int(f)` raises ValueError, which SGLang catches.
        // Infinity: `int(f)` raises OverflowError, which it does not.
        _ => None,
    }
}

/// `ast.literal_eval(text)` serialised by `json.dumps`, when both succeed
/// and the result is finite.
fn literal_value(text: &str) -> Option<Json> {
    pysem::literal_eval(text)
        .ok()
        .and_then(|lit| literal_to_json(&lit))
}

/// SGLang's conversion (see "Why not SGLang's rule"), with each of its
/// failures contained to the one parameter, which degrades to its unescaped
/// text (or, if unescaping itself fails, the raw text). Not used by the
/// parser; kept as the bridge to SGLang's differential corpus.
pub fn convert_parameter_sglang(raw: &str, param_type: &str) -> Json {
    let Ok(value) = pysem::html_unescape(raw) else {
        return Json::Str(raw.to_string());
    };
    if value.to_lowercase() == "null" {
        return Json::Null;
    }
    let text = || Json::Str(value.clone());
    let t = param_type;
    if STRING_TYPES.contains(&t) {
        return text();
    }
    if starts_with_any(t, INT_PREFIXES) {
        return pysem::py_int(&value)
            .map(Json::Int)
            .unwrap_or_else(|_| text());
    }
    if starts_with_any(t, NUMBER_PREFIXES) {
        return sglang_number(&value).unwrap_or_else(text);
    }
    if BOOL_TYPES.contains(&t) {
        return Json::Bool(value.to_lowercase() == "true");
    }
    if is_json_container_type(t)
        && let Ok(parsed) = json::parse_python_lenient(&value)
    {
        return if is_finite_json(&parsed) {
            parsed
        } else {
            text()
        };
    }
    literal_value(&value).unwrap_or_else(text)
}

/// Converts one parameter's raw text by its declared type: the rule (module
/// docs).
pub fn convert_parameter_round_trip(raw: &str, param_type: &str) -> Json {
    let text = || Json::Str(raw.to_string());
    let t = param_type;
    if STRING_TYPES.contains(&t) {
        return text();
    }
    if let Ok(parsed) = json::parse(raw) {
        return parsed;
    }
    if starts_with_any(t, INT_PREFIXES) {
        return pysem::py_int(raw).map(Json::Int).unwrap_or_else(|_| text());
    }
    if starts_with_any(t, NUMBER_PREFIXES) {
        return sglang_number(raw).unwrap_or_else(text);
    }
    if BOOL_TYPES.contains(&t) {
        return match raw.to_lowercase().as_str() {
            "true" => Json::Bool(true),
            "false" => Json::Bool(false),
            _ => text(),
        };
    }
    literal_value(raw).unwrap_or_else(text)
}

fn is_finite_json(value: &Json) -> bool {
    match value {
        Json::Float(f) => f.is_finite(),
        Json::Array(items) => items.iter().all(is_finite_json),
        Json::Object(members) => members.iter().all(|(_, v)| is_finite_json(v)),
        _ => true,
    }
}

/// `json.dumps` of a literal value, as a [`Json`]; `None` where `json.dumps`
/// raises or would emit a non-finite float.
fn literal_to_json(lit: &PyLit) -> Option<Json> {
    Some(match lit {
        PyLit::None => Json::Null,
        PyLit::Bool(b) => Json::Bool(*b),
        PyLit::Int(i) => {
            if i.digits.len() > PY_MAX_STR_DIGITS {
                return None;
            }
            Json::Int(i.clone())
        }
        PyLit::Float(f) => {
            if !f.is_finite() {
                return None;
            }
            Json::Float(*f)
        }
        PyLit::Str(s) => Json::Str(s.clone()),
        PyLit::List(items) | PyLit::Tuple(items) => {
            Json::Array(items.iter().map(literal_to_json).collect::<Option<_>>()?)
        }
        PyLit::Dict(entries) => {
            // Keys are already unique under Python equality; their JSON
            // spellings may still collide (`1` and `"1"`), and json.dumps
            // writes both, so no deduplication here.
            let mut members = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                let key = match k {
                    PyLit::Str(s) => s.clone(),
                    PyLit::Int(i) => {
                        if i.digits.len() > PY_MAX_STR_DIGITS {
                            return None;
                        }
                        i.to_string()
                    }
                    PyLit::Float(f) => json::json_float(*f),
                    PyLit::Bool(b) => b.to_string(),
                    PyLit::None => "null".to_string(),
                    _ => return None,
                };
                members.push((key, literal_to_json(v)?));
            }
            Json::Object(members)
        }
    })
}

/// Builds the arguments object of one call: Python dict semantics for
/// repeated parameter names (first position, last value).
pub fn arguments_object(params: impl IntoIterator<Item = (String, Json)>) -> Json {
    let mut object = ObjectBuilder::default();
    for (name, value) in params {
        object.insert(name, value);
    }
    object.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(raw: &str, t: &str) -> String {
        json::dumps_default(&convert_parameter_sglang(raw, t))
    }

    #[test]
    fn sglang_rule_matches_sglang() {
        assert_eq!(conv("hello &amp; bye", "string"), "\"hello & bye\"");
        assert_eq!(conv("NULL", "string"), "null");
        assert_eq!(conv(" 42 ", "integer"), "42");
        assert_eq!(conv("4.0", "integer"), "\"4.0\"");
        assert_eq!(conv("4.0", "number"), "4");
        assert_eq!(conv("4.5", "number"), "4.5");
        assert_eq!(conv("1e400", "number"), "\"1e400\"");
        assert_eq!(conv("nan", "number"), "\"nan\"");
        assert_eq!(conv("TRUE", "boolean"), "true");
        assert_eq!(conv("yes", "boolean"), "false");
        assert_eq!(conv("{\"a\": [1, 2.5]}", "object"), "{\"a\": [1, 2.5]}");
        assert_eq!(conv("5", "object"), "5");
        assert_eq!(
            conv("{'a': True, 'b': (1, 2)}", "object"),
            "{\"a\": true, \"b\": [1, 2]}"
        );
        assert_eq!(conv("[NaN]", "array"), "\"[NaN]\"");
        assert_eq!(conv("{1, 2}", "array"), "\"{1, 2}\"");
        assert_eq!(
            conv("{1: 'a', '1': 'b'}", "object"),
            "{\"1\": \"a\", \"1\": \"b\"}"
        );
        assert_eq!(conv("plain words", "object"), "\"plain words\"");
        assert_eq!(conv("12", "null"), "12");
        assert_eq!(conv("'x'", "custom_type"), "\"x\"");
    }

    #[test]
    fn normalizes_type_names() {
        for (raw, expected) in [
            ("VARCHAR(255)", "string"),
            ("int32", "integer"),
            ("internal", "internal"),
            ("list[str]", "array"),
            ("Dict[str, int]", "object"),
            ("float64", "number"),
            ("double", "number"),
            ("uuid", "string"),
            ("Integer", "integer"),
        ] {
            assert_eq!(normalize_type_name(raw), expected, "{raw}");
        }
    }

    #[test]
    fn resolves_declared_types() {
        let tools = json::parse(
            r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{
                "a":{"type":"integer"},"b":{"type":["null","number"]},"c":{"type":["integer","string"]},
                "d":true,"e":{"anyOf":[{"type":"integer"}]},"g":{"type":"bigint"}}}}},
               {"type":"function","function":{"name":"u","parameters":{"anyOf":[{"properties":{"x":{"type":"boolean"}}}]}}}]"#,
        )
        .unwrap();
        let schemas = ToolSchemas::from_tools(&tools);
        assert_eq!(schemas.param_type("f", "a"), "integer");
        assert_eq!(schemas.param_type("f", "b"), "number");
        assert_eq!(schemas.param_type("f", "c"), "string");
        assert_eq!(schemas.param_type("f", "d"), "string");
        assert_eq!(schemas.param_type("f", "e"), "string");
        assert_eq!(schemas.param_type("f", "g"), "integer");
        assert_eq!(schemas.param_type("f", "zzz"), "string");
        assert_eq!(schemas.param_type("u", "x"), "boolean");
        assert_eq!(schemas.param_type("nope", "x"), "string");
    }

    proptest::proptest! {
        #[test]
        fn any_text_converts_to_valid_json(
            raw in proptest::prop_oneof![
                "\\PC{0,40}",
                "[\\[\\]{}(),:'\"\\\\#\n +\\-.0-9eEjxXbBoO_a-zA-Z&;]{0,40}",
            ],
            t in proptest::sample::select(vec![
                "string", "integer", "number", "boolean", "object", "array", "null", "custom",
            ]),
        ) {
            for value in [convert_parameter_sglang(&raw, t), convert_parameter_round_trip(&raw, t)] {
                let text = json::dumps_default(&value);
                proptest::prop_assert!(json::parse(&text).is_ok(), "{text}");
            }
        }
    }
}
