//! An order-preserving JSON value with Python `json` semantics.
//!
//! The chat template and the tool-argument converter must agree byte for byte
//! with Python, so this module reproduces the parts of CPython's `json` module
//! that reach the prompt or a tool call:
//!
//! - **Parsing** follows `json.loads`: object keys keep insertion order, a
//!   duplicated key keeps its first position and takes its last value, integers
//!   keep every digit (Python ints are unbounded; the 4300-digit
//!   string-conversion limit is enforced the way CPython enforces it), and
//!   floats are correctly rounded. Strict mode is RFC 8259; lenient mode also
//!   accepts Python's `NaN`, `Infinity` and `-Infinity` constants.
//! - **Serialising** follows `json.dumps` with its defaults (`", "` and `": "`
//!   separators, insertion order, `float.__repr__` for floats), plus the
//!   `ensure_ascii`, `indent`, `separators` and `sort_keys` options.
//!
//! `serde_json::Value` cannot stand in for this: without workspace-wide
//! features it sorts object keys and rounds large integers.

use std::collections::HashMap;
use std::fmt::Write as _;

/// CPython's default `sys.int_info.default_max_str_digits`.
pub const PY_MAX_STR_DIGITS: usize = 4300;

/// Maximum nesting depth accepted by [`parse`]. Deeper input is rejected
/// rather than risking the stack.
pub const MAX_DEPTH: usize = 256;

/// A Python integer, kept as its canonical decimal digits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BigInt {
    /// True for values below zero. Zero is never negative.
    pub negative: bool,
    /// Decimal digits without leading zeros (`"0"` for zero).
    pub digits: String,
}

impl BigInt {
    /// Builds a value from ASCII decimal digits, stripping leading zeros.
    pub fn from_digits(negative: bool, digits: &str) -> BigInt {
        debug_assert!(digits.bytes().all(|b| b.is_ascii_digit()));
        let trimmed = digits.trim_start_matches('0');
        if trimmed.is_empty() {
            BigInt {
                negative: false,
                digits: "0".to_string(),
            }
        } else {
            BigInt {
                negative,
                digits: trimmed.to_string(),
            }
        }
    }

    /// The value as an `i128`, when it fits.
    pub fn to_i128(&self) -> Option<i128> {
        let magnitude: i128 = if self.digits.len() <= 38 {
            self.digits.parse().ok()?
        } else {
            return None;
        };
        Some(if self.negative { -magnitude } else { magnitude })
    }

    /// Builds a value from base-2^k or arbitrary-radix digits (each `< radix`).
    pub fn from_radix_digits(negative: bool, radix: u32, digits: &[u32]) -> BigInt {
        // Little-endian limbs in base 1e9.
        const BASE: u64 = 1_000_000_000;
        let mut limbs: Vec<u64> = vec![0];
        for &d in digits {
            let mut carry = d as u64;
            for limb in limbs.iter_mut() {
                let v = *limb * radix as u64 + carry;
                *limb = v % BASE;
                carry = v / BASE;
            }
            while carry > 0 {
                limbs.push(carry % BASE);
                carry /= BASE;
            }
        }
        let mut out = String::new();
        let mut iter = limbs.iter().rev();
        if let Some(first) = iter.next() {
            write!(out, "{first}").expect("write to String");
        }
        for limb in iter {
            write!(out, "{limb:09}").expect("write to String");
        }
        BigInt::from_digits(negative, &out)
    }

    /// The exact integer value of a finite, integral `f64` (`int(x)` in Python).
    pub fn from_integral_f64(x: f64) -> BigInt {
        debug_assert!(x.is_finite() && x.trunc() == x);
        let negative = x < 0.0;
        let bits = x.abs().to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i64;
        let fraction = bits & ((1u64 << 52) - 1);
        if exponent == 0 {
            // Zero or subnormal; integral subnormals are zero.
            return BigInt::from_digits(false, "0");
        }
        let mantissa = fraction | (1u64 << 52);
        let shift = exponent - 1075;
        if shift <= 0 {
            let value = mantissa >> (-shift) as u32;
            return BigInt::from_digits(negative, &value.to_string());
        }
        // mantissa * 2^shift: feed the binary expansion as base-2 digits.
        let mut binary: Vec<u32> = (0..53)
            .rev()
            .map(|i| ((mantissa >> i) & 1) as u32)
            .collect();
        binary.extend(std::iter::repeat_n(0, shift as usize));
        BigInt::from_radix_digits(negative, 2, &binary)
    }
}

impl std::fmt::Display for BigInt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.negative {
            f.write_str("-")?;
        }
        f.write_str(&self.digits)
    }
}

/// A JSON value as Python's `json` module sees it.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(BigInt),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    /// Keys in insertion order, without duplicates.
    Object(Vec<(String, Json)>),
}

impl Json {
    /// Looks up an object member.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(members) => Some(members),
            _ => None,
        }
    }

    /// The Python type name, for error messages.
    pub fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Int(_) => "integer",
            Json::Float(_) => "number",
            Json::Str(_) => "string",
            Json::Array(_) => "array",
            Json::Object(_) => "object",
        }
    }

    /// Converts a `serde_json::Value`. Key order is whatever that map yields
    /// (sorted unless serde_json's `preserve_order` feature is on), so callers
    /// that need byte-exact prompts should parse the original text instead.
    pub fn from_serde(value: &serde_json::Value) -> Json {
        match value {
            serde_json::Value::Null => Json::Null,
            serde_json::Value::Bool(b) => Json::Bool(*b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    BigInt::from_digits(i < 0, &i.unsigned_abs().to_string()).into()
                } else if let Some(u) = n.as_u64() {
                    BigInt::from_digits(false, &u.to_string()).into()
                } else {
                    Json::Float(n.as_f64().unwrap_or(f64::NAN))
                }
            }
            serde_json::Value::String(s) => Json::Str(s.clone()),
            serde_json::Value::Array(items) => {
                Json::Array(items.iter().map(Json::from_serde).collect())
            }
            serde_json::Value::Object(map) => Json::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), Json::from_serde(v)))
                    .collect(),
            ),
        }
    }

    /// Converts to a `serde_json::Value`. Integers outside `i64`/`u64` and
    /// non-finite floats have no serde_json representation and yield `None`.
    pub fn to_serde(&self) -> Option<serde_json::Value> {
        Some(match self {
            Json::Null => serde_json::Value::Null,
            Json::Bool(b) => serde_json::Value::Bool(*b),
            Json::Int(i) => {
                let v = i.to_i128()?;
                if let Ok(v) = i64::try_from(v) {
                    v.into()
                } else {
                    u64::try_from(v).ok()?.into()
                }
            }
            Json::Float(f) => serde_json::Number::from_f64(*f)?.into(),
            Json::Str(s) => serde_json::Value::String(s.clone()),
            Json::Array(items) => {
                serde_json::Value::Array(items.iter().map(Json::to_serde).collect::<Option<_>>()?)
            }
            Json::Object(members) => serde_json::Value::Object(
                members
                    .iter()
                    .map(|(k, v)| Some((k.clone(), v.to_serde()?)))
                    .collect::<Option<_>>()?,
            ),
        })
    }
}

impl From<BigInt> for Json {
    fn from(value: BigInt) -> Json {
        Json::Int(value)
    }
}

/// Builds an object, applying Python's duplicate-key rule.
#[derive(Default)]
pub struct ObjectBuilder {
    members: Vec<(String, Json)>,
    index: HashMap<String, usize>,
}

impl ObjectBuilder {
    pub fn insert(&mut self, key: String, value: Json) {
        if let Some(&i) = self.index.get(&key) {
            self.members[i].1 = value;
        } else {
            self.index.insert(key.clone(), self.members.len());
            self.members.push((key, value));
        }
    }

    pub fn finish(self) -> Json {
        Json::Object(self.members)
    }
}

/// Why a JSON text was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub offset: usize,
    pub message: &'static str,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for ParseError {}

/// Parses RFC 8259 JSON with Python value semantics.
pub fn parse(text: &str) -> Result<Json, ParseError> {
    Parser::new(text, false).parse_document()
}

/// Parses like Python's default `json.loads`, which also accepts the `NaN`,
/// `Infinity` and `-Infinity` constants.
pub fn parse_python_lenient(text: &str) -> Result<Json, ParseError> {
    Parser::new(text, true).parse_document()
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    allow_constants: bool,
}

impl<'a> Parser<'a> {
    fn new(text: &'a str, allow_constants: bool) -> Self {
        Parser {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            allow_constants,
        }
    }

    fn err<T>(&self, message: &'static str) -> Result<T, ParseError> {
        Err(ParseError {
            offset: self.pos,
            message,
        })
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_document(mut self) -> Result<Json, ParseError> {
        self.skip_ws();
        let value = self.parse_value(0)?;
        self.skip_ws();
        if self.pos != self.bytes.len() {
            return self.err("extra data");
        }
        Ok(value)
    }

    fn eat_literal(&mut self, literal: &str) -> bool {
        if self.text[self.pos..].starts_with(literal) {
            self.pos += literal.len();
            true
        } else {
            false
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<Json, ParseError> {
        if depth >= MAX_DEPTH {
            return self.err("nesting too deep");
        }
        match self.bytes.get(self.pos) {
            None => self.err("expecting value"),
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b'n') if self.eat_literal("null") => Ok(Json::Null),
            Some(b't') if self.eat_literal("true") => Ok(Json::Bool(true)),
            Some(b'f') if self.eat_literal("false") => Ok(Json::Bool(false)),
            Some(b'N') if self.allow_constants && self.eat_literal("NaN") => {
                Ok(Json::Float(f64::NAN))
            }
            Some(b'I') if self.allow_constants && self.eat_literal("Infinity") => {
                Ok(Json::Float(f64::INFINITY))
            }
            Some(b'-') if self.allow_constants && self.eat_literal("-Infinity") => {
                Ok(Json::Float(f64::NEG_INFINITY))
            }
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => self.err("expecting value"),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.pos += 1;
        let mut object = ObjectBuilder::default();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(object.finish());
        }
        loop {
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b'"') {
                return self.err("expecting property name enclosed in double quotes");
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return self.err("expecting ':' delimiter");
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.parse_value(depth + 1)?;
            object.insert(key, value);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(object.finish());
                }
                _ => return self.err("expecting ',' delimiter"),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return self.err("expecting ',' delimiter"),
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, ParseError> {
        let Some(hex) = self.text.get(self.pos..self.pos + 4) else {
            return self.err("invalid \\uXXXX escape");
        };
        if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return self.err("invalid \\uXXXX escape");
        }
        self.pos += 4;
        Ok(u32::from_str_radix(hex, 16).expect("validated hex"))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&self.text[start..self.pos]);
            match self.bytes.get(self.pos) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let Some(&escape) = self.bytes.get(self.pos) else {
                        return self.err("unterminated string");
                    };
                    self.pos += 1;
                    match escape {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let first = self.parse_hex4()?;
                            let code = if (0xd800..0xdc00).contains(&first) {
                                // A high surrogate must pair with a low one;
                                // Rust strings cannot hold a lone surrogate.
                                if !self.text[self.pos..].starts_with("\\u") {
                                    return self.err("lone surrogate");
                                }
                                self.pos += 2;
                                let second = self.parse_hex4()?;
                                if !(0xdc00..0xe000).contains(&second) {
                                    return self.err("lone surrogate");
                                }
                                0x10000 + ((first - 0xd800) << 10) + (second - 0xdc00)
                            } else if (0xdc00..0xe000).contains(&first) {
                                return self.err("lone surrogate");
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).expect("valid scalar"));
                        }
                        _ => {
                            self.pos -= 1;
                            return self.err("invalid escape");
                        }
                    }
                }
                Some(_) => return self.err("invalid control character"),
            }
        }
    }

    fn parse_number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        let negative = self.bytes[self.pos] == b'-';
        if negative {
            self.pos += 1;
        }
        let int_start = self.pos;
        match self.bytes.get(self.pos) {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
            }
            _ => return self.err("expecting value"),
        }
        let int_end = self.pos;
        let mut is_float = false;
        // A fraction or exponent only counts when complete; Python's scanner
        // otherwise stops before it and reports "extra data".
        if self.bytes.get(self.pos) == Some(&b'.')
            && self.bytes.get(self.pos + 1).is_some_and(u8::is_ascii_digit)
        {
            is_float = true;
            self.pos += 1;
            while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            let mut look = self.pos + 1;
            if matches!(self.bytes.get(look), Some(b'+' | b'-')) {
                look += 1;
            }
            if self.bytes.get(look).is_some_and(u8::is_ascii_digit) {
                is_float = true;
                self.pos = look;
                while self.bytes.get(self.pos).is_some_and(u8::is_ascii_digit) {
                    self.pos += 1;
                }
            }
        }
        if is_float {
            let value: f64 = self.text[start..self.pos].parse().expect("validated float");
            Ok(Json::Float(value))
        } else {
            let digits = &self.text[int_start..int_end];
            if digits.len() > PY_MAX_STR_DIGITS {
                return self.err("integer exceeds the 4300-digit conversion limit");
            }
            Ok(Json::Int(BigInt::from_digits(negative, digits)))
        }
    }
}

/// Python's `float.__repr__`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    let (digits, exponent) = if x == 0.0 {
        ("0".to_string(), 0)
    } else {
        shortest_digits(x.abs())
    };
    let mut out = String::new();
    if x.is_sign_negative() {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let int_len = exponent as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        write!(out, "{:02}", exponent.abs()).expect("write to String");
    }
    out
}

/// The digits and decimal exponent Python's `repr` chooses for a positive finite
/// `x` (`d.ddd × 10^exponent`): CPython's `float_repr_style = 'short'`, David Gay's
/// `dtoa` mode 0. That is the fewest significant digits that read back as `x`, and
/// among the (at most two) candidates of that length, the one nearest `x`, an exact
/// tie going to the even last digit.
///
/// Rust's `{:e}` also prints the fewest round-tripping digits, but on a tie between
/// two shortest candidates it does not pick the same one: `181703637716804.12` parses
/// to exactly `181703637716804.125`, which Python prints as `…804.12` and `{:e}` as
/// `…804.13`. So this does not use it. Instead, for each length `n`, it takes the
/// correctly rounded `n`-digit decimal (`{:.(n-1)e}`, exact with ties to even, which is
/// the nearest candidate) and, if that does not read back as `x`, the `n`-digit decimal
/// on the other side of `x`; the first length with a round-tripping candidate wins.
/// Only the two decimals bracketing `x` can round-trip, because the set of decimals
/// that read back as `x` is an interval containing `x`. Differentially tested against
/// CPython (`tests/fixtures/float_repr_cases.json`).
fn shortest_digits(x: f64) -> (String, i32) {
    for n in 1..=17usize {
        let sci = format!("{:.*e}", n - 1, x);
        let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
        let exp: i32 = exp.parse().expect("integer exponent");
        let int: u64 = mantissa
            .chars()
            .filter(|c| *c != '.')
            .collect::<String>()
            .parse()
            .expect("decimal digits");
        // `value = int × 10^scale`, `int` having exactly `n` digits.
        let scale = exp - (n as i32 - 1);
        let reads_back = |int: u64, scale: i32| {
            format!("{int}e{scale}")
                .parse::<f64>()
                .expect("decimal parses")
                == x
        };
        let candidate = if reads_back(int, scale) {
            Some((int, scale))
        } else {
            let low = 10u64.pow(n as u32 - 1);
            let other = if format!("{int}e{scale}").parse::<f64>().expect("parses") < x {
                // `int` is below `x`: the next `n`-digit decimal up.
                (int + 1, scale)
            } else if int == low {
                // The next one down is in the decade below: `99…9 × 10^(scale-1)`.
                (low * 10 - 1, scale - 1)
            } else {
                (int - 1, scale)
            };
            reads_back(other.0, other.1).then_some(other)
        };
        if let Some((int, scale)) = candidate {
            let mut digits = int.to_string();
            let exponent = scale + digits.len() as i32 - 1;
            while digits.len() > 1 && digits.ends_with('0') {
                digits.pop();
            }
            return (digits, exponent);
        }
    }
    unreachable!("17 significant digits always round-trip a binary64")
}

/// `json.dumps` options.
#[derive(Clone, Debug)]
pub struct DumpOptions {
    pub ensure_ascii: bool,
    /// `None` for single-line output; otherwise the per-level indent string
    /// (Python's integer indent `n` is `" " * n`).
    pub indent: Option<String>,
    pub item_separator: String,
    pub key_separator: String,
    pub sort_keys: bool,
}

impl Default for DumpOptions {
    /// `json.dumps(x, ensure_ascii=False)`, the form chat templates use.
    fn default() -> Self {
        DumpOptions {
            ensure_ascii: false,
            indent: None,
            item_separator: ", ".to_string(),
            key_separator: ": ".to_string(),
            sort_keys: false,
        }
    }
}

/// Error for values `json.dumps` refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpError(pub String);

impl std::fmt::Display for DumpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DumpError {}

/// Python's `json.dumps(value, ...)`.
pub fn dumps(value: &Json, options: &DumpOptions) -> String {
    let mut out = String::new();
    write_value(&mut out, value, options, 0);
    out
}

/// Python's `json.dumps(value)` with all defaults except `ensure_ascii=False`.
pub fn dumps_default(value: &Json) -> String {
    dumps(value, &DumpOptions::default())
}

fn write_newline_indent(out: &mut String, options: &DumpOptions, level: usize) {
    if let Some(indent) = &options.indent {
        out.push('\n');
        for _ in 0..level {
            out.push_str(indent);
        }
    }
}

fn write_value(out: &mut String, value: &Json, options: &DumpOptions, level: usize) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Int(i) => write!(out, "{i}").expect("write to String"),
        Json::Float(f) => out.push_str(&json_float(*f)),
        Json::Str(s) => write_string(out, s, options.ensure_ascii),
        Json::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(&options.item_separator);
                }
                write_newline_indent(out, options, level + 1);
                write_value(out, item, options, level + 1);
            }
            write_newline_indent(out, options, level);
            out.push(']');
        }
        Json::Object(members) => {
            if members.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut ordered: Vec<&(String, Json)> = members.iter().collect();
            if options.sort_keys {
                ordered.sort_by(|a, b| a.0.cmp(&b.0));
            }
            out.push('{');
            for (i, (key, item)) in ordered.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(&options.item_separator);
                }
                write_newline_indent(out, options, level + 1);
                write_string(out, key, options.ensure_ascii);
                out.push_str(&options.key_separator);
                write_value(out, item, options, level + 1);
            }
            write_newline_indent(out, options, level);
            out.push('}');
        }
    }
}

/// `json.dumps` of a float: `float.__repr__` with Python's spellings for the
/// non-finite values (which are not valid JSON).
pub fn json_float(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_string()
    } else if x == f64::INFINITY {
        "Infinity".to_string()
    } else if x == f64::NEG_INFINITY {
        "-Infinity".to_string()
    } else {
        py_float_repr(x)
    }
}

/// Writes a JSON string literal exactly as `json.dumps` does.
pub fn write_string(out: &mut String, s: &str, ensure_ascii: bool) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                write!(out, "\\u{:04x}", c as u32).expect("write to String");
            }
            c if ensure_ascii && !(' '..='~').contains(&c) => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    write!(out, "\\u{unit:04x}").expect("write to String");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789.125, "123456789.125"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e100, "1e+100"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            (2.5, "2.5"),
            (1.2345678901234567e19, "1.2345678901234567e+19"),
        ];
        for (x, expected) in cases {
            assert_eq!(py_float_repr(x), expected, "{x:e}");
        }
    }

    /// Every value in the CPython-generated fixture (`dev/gen_float_repr_cases.py`:
    /// exact ties between shortest candidates, powers of two, the fixed/exponent
    /// switch, subnormals, random bit patterns and decimals) prints exactly as
    /// `float.__repr__` does.
    #[test]
    fn float_repr_matches_cpython() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/float_repr_cases.json"
        );
        let cases: Vec<(String, String)> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(cases.len() > 10_000);
        let mut failures = Vec::new();
        for (bits, expected) in &cases {
            let x = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
            let got = py_float_repr(x);
            if &got != expected {
                failures.push(format!("{bits}: {got} != {expected}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} differ, e.g. {:?}",
            failures.len(),
            cases.len(),
            &failures[..failures.len().min(5)]
        );
        // The tie that `{:e}` breaks the other way.
        assert_eq!(py_float_repr(181703637716804.12), "181703637716804.12");
    }

    #[test]
    fn integral_floats_expand_exactly() {
        assert_eq!(
            BigInt::from_integral_f64(1e20).to_string(),
            "100000000000000000000"
        );
        assert_eq!(BigInt::from_integral_f64(-3.0).to_string(), "-3");
        assert_eq!(BigInt::from_integral_f64(-0.0).to_string(), "0");
        assert_eq!(
            BigInt::from_integral_f64(2f64.powi(80)).to_string(),
            "1208925819614629174706176"
        );
    }

    #[test]
    fn parse_keeps_order_and_python_duplicate_rule() {
        let v = parse(r#"{"b": 1, "a": 2, "b": 3}"#).unwrap();
        assert_eq!(dumps_default(&v), r#"{"b": 3, "a": 2}"#);
    }

    #[test]
    fn parse_keeps_big_integers() {
        let v = parse("[123456789012345678901234567890, -0, 1.0, 1e400]").unwrap();
        assert_eq!(
            dumps_default(&v),
            "[123456789012345678901234567890, 0, 1.0, Infinity]"
        );
    }

    #[test]
    fn dumps_escapes_like_python() {
        let v = Json::Str("a\"\\\n\u{1}\u{7f}é😀<>&'/".to_string());
        assert_eq!(dumps_default(&v), "\"a\\\"\\\\\\n\\u0001\u{7f}é😀<>&'/\"");
        let ascii = DumpOptions {
            ensure_ascii: true,
            ..DumpOptions::default()
        };
        assert_eq!(
            dumps(&v, &ascii),
            "\"a\\\"\\\\\\n\\u0001\\u007f\\u00e9\\ud83d\\ude00<>&'/\""
        );
    }

    #[test]
    fn rejects_lone_surrogates_and_constants_in_strict_mode() {
        assert!(parse(r#""\ud800""#).is_err());
        assert!(parse("NaN").is_err());
        assert!(parse_python_lenient("[NaN, -Infinity]").is_ok());
    }
}
