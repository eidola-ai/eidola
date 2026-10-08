//! The wire contract between Eidola's clients, its gateway, and its inference
//! nodes (`eidola-server-engine`) that every side must spell identically.
//!
//! Five things live here:
//!
//! - **The prefix-cache key** a client may send as the request body's
//!   `cache_key` member: [`CACHE_KEY_BYTES`] bytes, encoded as unpadded
//!   base64url ([`CACHE_KEY_TEXT_LEN`] characters). [`is_cache_key_text`] is
//!   the shape rule: the gateway refuses anything else before a request is
//!   paid for, and the node's decoder accepts exactly what it accepts.
//!   [`encode_cache_key`] is the one encoder, used by the client that mints
//!   keys, so every key it sends has the spelling the rule accepts.
//! - **The weights header** ([`WEIGHTS_HEADER`]) the gateway sends on every
//!   chat request to a node, carrying the lowercase-hex weights hash it
//!   expects that node to serve. The node refuses a request without it, or
//!   with a different hash, before reading the body.
//! - **The request body's JSON shape**: at most [`MAX_REQUEST_JSON_VALUES`]
//!   values nested at most [`MAX_REQUEST_JSON_DEPTH`] deep, checked by
//!   [`check_request_json`] before anything is parsed into memory. The node's
//!   parse trees cost memory per value, not only per byte, so the value count
//!   is what bounds them; the gateway refuses the same bodies first.
//! - **The rules of the node's accepted subset** that are not a matter of the
//!   request type ([`check_tool_choice`], [`check_stop`],
//!   [`check_max_completion_tokens`], [`SubsetError`]): the gateway refuses a
//!   request bound for a node by the same functions the node applies, so a
//!   request the gateway routes is never one the node refuses for its shape.
//! - **The node's error types** ([`error_type`]) and which refusals precede
//!   admission ([`refused_before_admission`]): the only refusals a gateway
//!   may send to another node.
//!
//! The key is secret material on every side: a holder never logs it, prints
//! it redacted, and scrubs it when done. Nothing here holds one; this module
//! only states the shape, so it needs no secret-handling dependency.

/// Length of a decoded prefix-cache key, in bytes.
pub const CACHE_KEY_BYTES: usize = 32;

/// Length of a prefix-cache key's text: [`CACHE_KEY_BYTES`] encoded as
/// base64url without padding.
pub const CACHE_KEY_TEXT_LEN: usize = (CACHE_KEY_BYTES * 8).div_ceil(6);

/// Largest chat request body a node reads, in bytes. A node holds at most one
/// such body per admission slot (`EIDOLA_ENGINE_MAX_REQUESTS`), so a
/// deployment's host-memory check counts it.
pub const MAX_REQUEST_BODY_BYTES: usize = 32 * 1024 * 1024;

/// The request header carrying the weights hash a gateway expects a node to
/// serve: the node's weights hash, 64 hex digits. Header names are
/// case-insensitive; this is the lowercase spelling both sides use.
pub const WEIGHTS_HEADER: &str = "x-eidola-weights-sha256";

/// Most JSON values a chat request body may hold, counting every scalar,
/// array and object, and every object key (each is its own allocation in a
/// parse tree). A node's host-memory budget per request is linear in this
/// count (`engine_deployment::host_memory_bytes`), so it is what bounds the
/// parse trees of a small, dense body.
///
/// 2^18 = 262,144. Realistic requests use far fewer: a long agentic history
/// of 2,000 messages with tool calls is about 20,000 values, and 100 tools
/// with detailed JSON Schemas of about 500 values each another 50,000, so the
/// cap leaves several times that heaviest case. Long text costs bytes, not
/// values: a million-token conversation in a few hundred messages is a few
/// thousand values.
pub const MAX_REQUEST_JSON_VALUES: usize = 1 << 18;

/// Deepest nesting a chat request body may have (a scalar at the top level
/// is depth 0; every array or object adds one). A request's own structure is
/// about five deep, and real tool schemas rarely reach twenty; 64 keeps every
/// parser's recursion short, below `serde_json`'s own limit of 128.
pub const MAX_REQUEST_JSON_DEPTH: usize = 64;

/// What [`check_request_json`] found in a body it accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonShape {
    /// Values, object keys included.
    pub values: usize,
    /// Deepest nesting.
    pub depth: usize,
}

/// Why [`check_request_json`] refused a body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonShapeError {
    /// Not one well-formed JSON text (RFC 8259) in UTF-8.
    Malformed,
    /// More than [`MAX_REQUEST_JSON_VALUES`] values.
    TooManyValues,
    /// Nested deeper than [`MAX_REQUEST_JSON_DEPTH`].
    TooDeep,
}

impl core::fmt::Display for JsonShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            JsonShapeError::Malformed => f.write_str("the body is not valid JSON"),
            JsonShapeError::TooManyValues => write!(
                f,
                "the body holds more than {MAX_REQUEST_JSON_VALUES} JSON values"
            ),
            JsonShapeError::TooDeep => write!(
                f,
                "the body nests JSON deeper than {MAX_REQUEST_JSON_DEPTH} levels"
            ),
        }
    }
}

/// Validate `body` as one JSON text and count its values and depth, refusing
/// as soon as either limit is passed. It allocates nothing and does not
/// recurse, so a hostile body costs only the scan: every parse that follows
/// (and allocates per value) sees a body within both limits.
///
/// Strict RFC 8259, as `serde_json` is: no comments, no trailing commas, no
/// leading zeros, whitespace only between tokens. A `\uXXXX` escape is
/// checked for its shape only; what it decodes to (a lone surrogate, say) is
/// for the parsers that follow to refuse.
pub fn check_request_json(body: &[u8]) -> Result<JsonShape, JsonShapeError> {
    use JsonShapeError::{Malformed, TooDeep, TooManyValues};
    if core::str::from_utf8(body).is_err() {
        return Err(Malformed);
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Need {
        Value,
        ArrayFirst,
        ObjectFirst,
        Key,
        AfterValue,
    }
    // `true` for an object, per open container; fixed-size, so nothing is
    // allocated however deep the body tries to go.
    let mut stack = [false; MAX_REQUEST_JSON_DEPTH];
    let mut depth = 0usize;
    let mut deepest = 0usize;
    let mut values = 0usize;
    let mut i = 0usize;
    let mut need = Need::Value;
    let count = |values: &mut usize| -> Result<(), JsonShapeError> {
        *values += 1;
        if *values > MAX_REQUEST_JSON_VALUES {
            Err(TooManyValues)
        } else {
            Ok(())
        }
    };
    loop {
        while i < body.len() && matches!(body[i], b' ' | b'\t' | b'\n' | b'\r') {
            i += 1;
        }
        let Some(&b) = body.get(i) else {
            return if need == Need::AfterValue && depth == 0 {
                Ok(JsonShape {
                    values,
                    depth: deepest,
                })
            } else {
                Err(Malformed)
            };
        };
        match need {
            Need::Value | Need::ArrayFirst => {
                if need == Need::ArrayFirst && b == b']' {
                    i += 1;
                    depth -= 1;
                    need = Need::AfterValue;
                    continue;
                }
                count(&mut values)?;
                match b {
                    b'{' | b'[' => {
                        if depth == MAX_REQUEST_JSON_DEPTH {
                            return Err(TooDeep);
                        }
                        stack[depth] = b == b'{';
                        depth += 1;
                        deepest = deepest.max(depth);
                        i += 1;
                        need = if b == b'{' {
                            Need::ObjectFirst
                        } else {
                            Need::ArrayFirst
                        };
                    }
                    b'"' => {
                        i = scan_string(body, i)?;
                        need = Need::AfterValue;
                    }
                    b't' | b'f' | b'n' => {
                        let word: &[u8] = match b {
                            b't' => b"true",
                            b'f' => b"false",
                            _ => b"null",
                        };
                        if !body[i..].starts_with(word) {
                            return Err(Malformed);
                        }
                        i += word.len();
                        need = Need::AfterValue;
                    }
                    b'-' | b'0'..=b'9' => {
                        i = scan_number(body, i)?;
                        need = Need::AfterValue;
                    }
                    _ => return Err(Malformed),
                }
            }
            Need::ObjectFirst | Need::Key => {
                if need == Need::ObjectFirst && b == b'}' {
                    i += 1;
                    depth -= 1;
                    need = Need::AfterValue;
                    continue;
                }
                if b != b'"' {
                    return Err(Malformed);
                }
                count(&mut values)?;
                i = scan_string(body, i)?;
                while i < body.len() && matches!(body[i], b' ' | b'\t' | b'\n' | b'\r') {
                    i += 1;
                }
                if body.get(i) != Some(&b':') {
                    return Err(Malformed);
                }
                i += 1;
                need = Need::Value;
            }
            Need::AfterValue => {
                if depth == 0 {
                    return Err(Malformed);
                }
                let object = stack[depth - 1];
                match (b, object) {
                    (b',', true) => need = Need::Key,
                    (b',', false) => need = Need::Value,
                    (b'}', true) | (b']', false) => {
                        depth -= 1;
                        need = Need::AfterValue;
                    }
                    _ => return Err(Malformed),
                }
                i += 1;
            }
        }
    }
}

/// The index just past the string starting at `body[start]` (a `"`).
fn scan_string(body: &[u8], start: usize) -> Result<usize, JsonShapeError> {
    let mut i = start + 1;
    loop {
        match body.get(i) {
            None => return Err(JsonShapeError::Malformed),
            Some(b'"') => return Ok(i + 1),
            Some(b'\\') => match body.get(i + 1) {
                Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => i += 2,
                Some(b'u') => {
                    let hex = body.get(i + 2..i + 6).ok_or(JsonShapeError::Malformed)?;
                    if !hex.iter().all(u8::is_ascii_hexdigit) {
                        return Err(JsonShapeError::Malformed);
                    }
                    i += 6;
                }
                _ => return Err(JsonShapeError::Malformed),
            },
            Some(&c) if c < 0x20 => return Err(JsonShapeError::Malformed),
            Some(_) => i += 1,
        }
    }
}

/// The index just past the number starting at `body[start]`.
fn scan_number(body: &[u8], start: usize) -> Result<usize, JsonShapeError> {
    let digits = |mut i: usize| {
        let from = i;
        while body.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        (i, i > from)
    };
    let mut i = start;
    if body.get(i) == Some(&b'-') {
        i += 1;
    }
    match body.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits(i).0,
        _ => return Err(JsonShapeError::Malformed),
    }
    if body.get(i) == Some(&b'.') {
        let (next, any) = digits(i + 1);
        if !any {
            return Err(JsonShapeError::Malformed);
        }
        i = next;
    }
    if matches!(body.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(body.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let (next, any) = digits(i);
        if !any {
            return Err(JsonShapeError::Malformed);
        }
        i = next;
    }
    Ok(i)
}

/// Whether `text` is a well-formed prefix-cache key: exactly
/// [`CACHE_KEY_TEXT_LEN`] characters of the URL-safe base64 alphabet
/// (`A–Z a–z 0–9 - _`), no padding, and canonical, so it decodes to exactly
/// [`CACHE_KEY_BYTES`] bytes and has exactly one spelling.
///
/// Canonical means the bits the last character carries beyond the key's
/// 256 are zero: 43 characters carry 258 bits, so the last one must have its
/// two low bits clear. A strict base64 decoder (the `base64` crate's default,
/// which the node uses) refuses the other spellings, and accepting them here
/// would let a key with four spellings pass the gateway and fail at the node.
pub fn is_cache_key_text(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != CACHE_KEY_TEXT_LEN {
        return false;
    }
    let mut last = 0;
    for &b in bytes {
        match base64url_value(b) {
            Some(v) => last = v,
            None => return false,
        }
    }
    let spare_bits = CACHE_KEY_TEXT_LEN * 6 - CACHE_KEY_BYTES * 8;
    last & ((1 << spare_bits) - 1) == 0
}

/// The one spelling of a prefix-cache key: [`CACHE_KEY_BYTES`] bytes as
/// unpadded base64url, the text [`is_cache_key_text`] accepts.
///
/// Returned as ASCII bytes in a fixed array rather than a `String` so a
/// caller holding secret key material decides where the text lives and when
/// it is scrubbed; nothing here keeps a copy. The final character carries the
/// key's last four bits and two zero bits, which is what makes the spelling
/// canonical.
pub fn encode_cache_key(key: &[u8; CACHE_KEY_BYTES]) -> [u8; CACHE_KEY_TEXT_LEN] {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = [0u8; CACHE_KEY_TEXT_LEN];
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    let mut written = 0;
    for &byte in key {
        acc = (acc << 8) | u32::from(byte);
        bits += 8;
        while bits >= 6 {
            bits -= 6;
            out[written] = ALPHABET[((acc >> bits) & 0x3f) as usize];
            written += 1;
        }
    }
    if bits > 0 {
        out[written] = ALPHABET[((acc << (6 - bits)) & 0x3f) as usize];
        written += 1;
    }
    debug_assert_eq!(written, CACHE_KEY_TEXT_LEN);
    out
}

/// The six-bit value of one URL-safe base64 character, or `None` for any
/// other byte (padding included).
const fn base64url_value(b: u8) -> Option<u8> {
    match b {
        b'A'..=b'Z' => Some(b - b'A'),
        b'a'..=b'z' => Some(b - b'a' + 26),
        b'0'..=b'9' => Some(b - b'0' + 52),
        b'-' => Some(62),
        b'_' => Some(63),
        _ => None,
    }
}

/// The node's `error.type` values: what a node answers in an OpenAI-shaped
/// error body (`{"error": {"message", "type", "code"}}`). The node's
/// `ApiError::error_type` returns these, and the gateway reads a node's
/// refusal by them, so the two cannot spell one differently.
pub mod error_type {
    pub const AUTHENTICATION_ERROR: &str = "authentication_error";
    pub const WEIGHTS_HASH_REQUIRED: &str = "weights_hash_required";
    pub const WEIGHTS_HASH_MISMATCH: &str = "weights_hash_mismatch";
    pub const MODEL_NOT_FOUND: &str = "model_not_found";
    pub const INVALID_REQUEST: &str = "invalid_request_error";
    pub const CONTEXT_LENGTH_EXCEEDED: &str = "context_length_exceeded";
    pub const REQUEST_TOO_LARGE: &str = "request_too_large";
    pub const OVERLOADED: &str = "overloaded";
    pub const ENGINE_UNAVAILABLE: &str = "engine_unavailable";
    pub const INTERNAL_ERROR: &str = "internal_error";
}

/// A node refusal made before the request was admitted: nothing of it was
/// rendered, tokenized or scheduled, so another node may run it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreAdmission {
    /// The node does not serve what the gateway asked for: the gateway's
    /// token, the pinned weights, or the model.
    Misconfigured,
    /// Every read or admission slot is taken.
    Overloaded,
}

/// Whether a node's refusal, by its status and `error.type` exactly, was made
/// before admission. Only these pairs are: every other refusal (an engine that
/// stopped, `engine_unavailable`, among them) may come after the request ran,
/// so a gateway returns it rather than sending the request elsewhere. The node
/// holds itself to this list (`eidola-server-engine`'s
/// `only_pre_admission_refusals_say_so`).
pub fn refused_before_admission(status: u16, kind: &str) -> Option<PreAdmission> {
    use self::error_type as t;
    match (status, kind) {
        (401, t::AUTHENTICATION_ERROR)
        | (428, t::WEIGHTS_HASH_REQUIRED)
        | (412, t::WEIGHTS_HASH_MISMATCH)
        | (404, t::MODEL_NOT_FOUND) => Some(PreAdmission::Misconfigured),
        (503, t::OVERLOADED) => Some(PreAdmission::Overloaded),
        _ => None,
    }
}

/// Most stop sequences a node accepts on one request (OpenAI's limit).
pub const MAX_STOP_SEQUENCES: usize = 4;

/// Longest stop sequence a node accepts, in UTF-8 bytes. Stop sequences are
/// delimiters; the cap bounds the node's matcher state per request.
pub const MAX_STOP_BYTES: usize = 256;

/// How a node treats a request's tools: offered and parsed (`auto`, also the
/// meaning of an absent `tool_choice`), or withheld from the prompt (`none`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
}

/// A request a node refuses although the gateway's public request type
/// accepts it: the part of the node's accepted subset that is a rule over a
/// parsed request rather than over the type. The gateway checks these before
/// routing a request to a node, the node checks them again on arrival, and
/// both call the functions below, so the two cannot disagree. `Display` is the
/// node's refusal text; it is fixed, never quoting the request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubsetError {
    /// `messages` is empty.
    EmptyMessages,
    /// A content part is an image (a node serves a text-only model).
    ImageContent,
    /// `max_completion_tokens` is 0.
    ZeroMaxCompletionTokens,
    /// More than [`MAX_STOP_SEQUENCES`] stop sequences.
    TooManyStops,
    /// An empty stop sequence.
    EmptyStop,
    /// A stop sequence longer than [`MAX_STOP_BYTES`].
    StopTooLong,
    /// `tool_choice` other than `"auto"` or `"none"` (a node has no
    /// constrained decoding, so `"required"` and named functions are refused).
    UnsupportedToolChoice,
}

impl core::fmt::Display for SubsetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SubsetError::EmptyMessages => f.write_str("messages must not be empty"),
            SubsetError::ImageContent => f.write_str("this model accepts text only"),
            SubsetError::ZeroMaxCompletionTokens => {
                f.write_str("max_completion_tokens must be at least 1")
            }
            SubsetError::TooManyStops => {
                write!(f, "stop accepts at most {MAX_STOP_SEQUENCES} sequences")
            }
            SubsetError::EmptyStop => f.write_str("stop sequences must not be empty"),
            SubsetError::StopTooLong => write!(
                f,
                "each stop sequence may be at most {MAX_STOP_BYTES} bytes"
            ),
            SubsetError::UnsupportedToolChoice => {
                f.write_str("tool_choice supports only \"auto\" and \"none\"")
            }
        }
    }
}

/// The node's reading of `tool_choice`: absent or `"auto"` is
/// [`ToolChoice::Auto`], `"none"` is [`ToolChoice::None`], anything else is
/// refused.
pub fn check_tool_choice(choice: Option<&serde_json::Value>) -> Result<ToolChoice, SubsetError> {
    match choice {
        None => Ok(ToolChoice::Auto),
        Some(serde_json::Value::String(s)) if s == "auto" => Ok(ToolChoice::Auto),
        Some(serde_json::Value::String(s)) if s == "none" => Ok(ToolChoice::None),
        Some(_) => Err(SubsetError::UnsupportedToolChoice),
    }
}

/// The node's stop-sequence rule: at most [`MAX_STOP_SEQUENCES`], each
/// non-empty and at most [`MAX_STOP_BYTES`] bytes.
pub fn check_stop<S: AsRef<str>>(stops: &[S]) -> Result<(), SubsetError> {
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(SubsetError::TooManyStops);
    }
    if stops.iter().any(|s| s.as_ref().is_empty()) {
        return Err(SubsetError::EmptyStop);
    }
    if stops.iter().any(|s| s.as_ref().len() > MAX_STOP_BYTES) {
        return Err(SubsetError::StopTooLong);
    }
    Ok(())
}

/// The node's completion-limit rule: absent, or at least 1.
pub fn check_max_completion_tokens(limit: Option<u32>) -> Result<(), SubsetError> {
    if limit == Some(0) {
        return Err(SubsetError::ZeroMaxCompletionTokens);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_pre_admission_pairs_are_retryable() {
        use error_type as t;
        assert_eq!(
            refused_before_admission(503, t::OVERLOADED),
            Some(PreAdmission::Overloaded)
        );
        for (status, ty) in [
            (401, t::AUTHENTICATION_ERROR),
            (428, t::WEIGHTS_HASH_REQUIRED),
            (412, t::WEIGHTS_HASH_MISMATCH),
            (404, t::MODEL_NOT_FOUND),
        ] {
            assert_eq!(
                refused_before_admission(status, ty),
                Some(PreAdmission::Misconfigured)
            );
        }
        for (status, ty) in [
            (503, t::ENGINE_UNAVAILABLE),
            (503, "unknown"),
            (500, t::INTERNAL_ERROR),
            (400, t::INVALID_REQUEST),
            (404, "not_found"),
            (429, t::OVERLOADED),
        ] {
            assert_eq!(refused_before_admission(status, ty), None, "{status} {ty}");
        }
    }

    #[test]
    fn the_subset_rules_accept_and_refuse_at_their_bounds() {
        use serde_json::json;
        assert_eq!(check_tool_choice(None), Ok(ToolChoice::Auto));
        assert_eq!(
            check_tool_choice(Some(&json!("auto"))),
            Ok(ToolChoice::Auto)
        );
        assert_eq!(
            check_tool_choice(Some(&json!("none"))),
            Ok(ToolChoice::None)
        );
        for refused in [
            json!("required"),
            json!("AUTO"),
            json!(null),
            json!({"type": "function", "function": {"name": "f"}}),
        ] {
            assert_eq!(
                check_tool_choice(Some(&refused)),
                Err(SubsetError::UnsupportedToolChoice),
                "{refused}"
            );
        }

        let at_cap = "é".repeat(MAX_STOP_BYTES / 2);
        assert_eq!(check_stop::<&str>(&[]), Ok(()));
        assert_eq!(check_stop(&[at_cap.as_str(); MAX_STOP_SEQUENCES]), Ok(()));
        assert_eq!(
            check_stop(&["a"; MAX_STOP_SEQUENCES + 1]),
            Err(SubsetError::TooManyStops)
        );
        assert_eq!(check_stop(&["a", ""]), Err(SubsetError::EmptyStop));
        let over = "x".repeat(MAX_STOP_BYTES + 1);
        assert_eq!(check_stop(&[over.as_str()]), Err(SubsetError::StopTooLong));

        assert_eq!(check_max_completion_tokens(None), Ok(()));
        assert_eq!(check_max_completion_tokens(Some(1)), Ok(()));
        assert_eq!(
            check_max_completion_tokens(Some(0)),
            Err(SubsetError::ZeroMaxCompletionTokens)
        );
        assert_eq!(
            SubsetError::StopTooLong.to_string(),
            "each stop sequence may be at most 256 bytes"
        );
        assert_eq!(
            SubsetError::UnsupportedToolChoice.to_string(),
            "tool_choice supports only \"auto\" and \"none\""
        );
    }

    #[test]
    fn a_body_within_the_limits_is_counted() {
        assert_eq!(
            check_request_json(br#" {"a": [1, -2.5e3, "x\u00e9\n", true, null], "b": {}} "#),
            Ok(JsonShape {
                values: 10,
                depth: 2
            })
        );
        assert_eq!(
            check_request_json(b"0"),
            Ok(JsonShape {
                values: 1,
                depth: 0
            })
        );
        // Every valid text serde_json accepts, the scan accepts.
        for text in [
            "[]",
            "{}",
            "[[[]]]",
            r#"{"":0}"#,
            "-0",
            "1E+2",
            "0.5e-1",
            r#""\ud83d\ude00""#,
            "\"é\"",
        ] {
            assert!(
                serde_json::from_str::<serde_json::Value>(text).is_ok(),
                "{text}"
            );
            assert!(check_request_json(text.as_bytes()).is_ok(), "{text}");
        }
    }

    #[test]
    fn a_malformed_body_is_refused() {
        for text in [
            "",
            " ",
            "[",
            "[1,]",
            "{\"a\":1,}",
            "{\"a\"}",
            "{1:2}",
            "01",
            "1.",
            "1e",
            "+1",
            "tru",
            "nul",
            "[1] [2]",
            "\"a",
            "\"\\x\"",
            "\"\\u12\"",
            "\"\t\"",
            "[1}",
            "{\"a\":1]",
            "]",
            "// c\n1",
        ] {
            assert!(
                serde_json::from_str::<serde_json::Value>(text).is_err(),
                "{text}"
            );
            assert_eq!(
                check_request_json(text.as_bytes()),
                Err(JsonShapeError::Malformed),
                "{text:?}"
            );
        }
        assert_eq!(
            check_request_json(b"\"\xff\""),
            Err(JsonShapeError::Malformed)
        );
    }

    #[test]
    fn the_value_count_and_depth_are_capped() {
        let at_cap = format!("[{}0]", "0,".repeat(MAX_REQUEST_JSON_VALUES - 2));
        assert_eq!(
            check_request_json(at_cap.as_bytes()).map(|s| s.values),
            Ok(MAX_REQUEST_JSON_VALUES)
        );
        let over = format!("[{}0]", "0,".repeat(MAX_REQUEST_JSON_VALUES - 1));
        assert_eq!(
            check_request_json(over.as_bytes()),
            Err(JsonShapeError::TooManyValues)
        );
        // Keys count: an object of n members is 2n + 1 values.
        let n = MAX_REQUEST_JSON_VALUES / 2;
        let keys = format!("{{{}}}", vec!["\"\":0"; n].join(","));
        assert_eq!(
            check_request_json(keys.as_bytes()),
            Err(JsonShapeError::TooManyValues)
        );
        let deepest = format!(
            "{}{}",
            "[".repeat(MAX_REQUEST_JSON_DEPTH),
            "]".repeat(MAX_REQUEST_JSON_DEPTH)
        );
        assert_eq!(
            check_request_json(deepest.as_bytes()).map(|s| s.depth),
            Ok(MAX_REQUEST_JSON_DEPTH)
        );
        let deeper = format!(
            "{}{}",
            "[".repeat(MAX_REQUEST_JSON_DEPTH + 1),
            "]".repeat(MAX_REQUEST_JSON_DEPTH + 1)
        );
        assert_eq!(
            check_request_json(deeper.as_bytes()),
            Err(JsonShapeError::TooDeep)
        );
    }

    /// 32 zero bytes, encoded.
    const ZERO_KEY: &str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

    #[test]
    fn the_text_length_follows_from_the_key_length() {
        assert_eq!(CACHE_KEY_TEXT_LEN, 43);
        assert_eq!(ZERO_KEY.len(), CACHE_KEY_TEXT_LEN);
    }

    #[test]
    fn a_canonical_key_is_accepted() {
        assert!(is_cache_key_text(ZERO_KEY));
        // Every alphabet character in a non-final position.
        let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let key = format!("{}A", &alphabet[..42]);
        assert!(is_cache_key_text(&key));
        let key = format!("{}A", &alphabet[22..64]);
        assert!(is_cache_key_text(&key));
    }

    #[test]
    fn wrong_lengths_padding_and_other_alphabets_are_refused() {
        assert!(!is_cache_key_text(""));
        assert!(!is_cache_key_text(&ZERO_KEY[..42]));
        assert!(!is_cache_key_text(&format!("{ZERO_KEY}A")));
        assert!(!is_cache_key_text(&format!("{}=", &ZERO_KEY[..42])));
        // The standard alphabet's two extra characters.
        assert!(!is_cache_key_text(&format!("+{}", &ZERO_KEY[1..])));
        assert!(!is_cache_key_text(&format!("/{}", &ZERO_KEY[1..])));
        assert!(!is_cache_key_text(&format!(" {}", &ZERO_KEY[1..])));
        // A multi-byte character of the right *character* count is the wrong
        // byte count, and outside the alphabet regardless.
        assert!(!is_cache_key_text(&format!("é{}", &ZERO_KEY[2..])));
    }

    #[test]
    fn only_the_canonical_final_character_is_accepted() {
        let accepted: Vec<char> =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                .chars()
                .filter(|c| is_cache_key_text(&format!("{}{c}", &ZERO_KEY[..42])))
                .collect();
        // Values 0, 4, 8, …, 60: the sixteen characters whose two low bits
        // are zero.
        assert_eq!(accepted.iter().collect::<String>(), "AEIMQUYcgkosw048");
    }

    fn encoded(key: &[u8; CACHE_KEY_BYTES]) -> String {
        String::from_utf8(encode_cache_key(key).to_vec()).expect("ASCII")
    }

    #[test]
    fn the_encoder_spells_known_keys_as_base64url_without_padding() {
        // Vectors from an independent implementation (Python's
        // `base64.urlsafe_b64encode`, padding stripped).
        assert_eq!(encoded(&[0; CACHE_KEY_BYTES]), ZERO_KEY);
        assert_eq!(
            encoded(&[0xff; CACHE_KEY_BYTES]),
            "__________________________________________8"
        );
        let mut ascending = [0u8; CACHE_KEY_BYTES];
        for (i, b) in ascending.iter_mut().enumerate() {
            *b = i as u8;
        }
        assert_eq!(
            encoded(&ascending),
            "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
        );
    }

    #[test]
    fn every_encoded_key_is_one_the_shape_rule_accepts() {
        // Each byte value in each position, so every final-character case and
        // every alphabet entry is produced at least once.
        for position in 0..CACHE_KEY_BYTES {
            for value in 0..=255u8 {
                let mut key = [0x5a; CACHE_KEY_BYTES];
                key[position] = value;
                let text = encoded(&key);
                assert!(is_cache_key_text(&text), "{text}");
            }
        }
    }

    #[test]
    fn distinct_keys_have_distinct_spellings() {
        let mut a = [0u8; CACHE_KEY_BYTES];
        let mut b = [0u8; CACHE_KEY_BYTES];
        a[31] = 0x01;
        b[31] = 0x02;
        assert_ne!(encoded(&a), encoded(&b));
        assert_ne!(encoded(&a), ZERO_KEY);
    }
}
