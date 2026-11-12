//! The wire contract between Eidola's clients, its gateway, and its inference
//! nodes (`eidola-server-engine`) that every side must spell identically.
//!
//! Seven things live here:
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
//!   [`check_max_completion_tokens`], [`check_continuation`] with its
//!   [`ContinueField`], [`SubsetError`]): the gateway refuses a
//!   request bound for a node by the same functions the node applies, so a
//!   request the gateway routes is never one the node refuses for its shape.
//! - **The node's error types** ([`error_type`]) and which refusals precede
//!   admission ([`refused_before_admission`]): the only refusals a gateway
//!   may send to another node.
//! - **The declared cached prefix** a client may send as the request body's
//!   `cached_prefix` member ([`CachedPrefix`], [`check_cached_prefix`]): how
//!   many of the prompt's leading tokens an engine holds in its prefix cache
//!   under the request's key, and how many more the prompt may run to. It is
//!   what the request is priced by (`crate::pricing`), so a node refuses a
//!   request whose declaration does not hold before computing anything
//!   ([`declared_prefix_refusal`]). [`declared_prompt`] is the rule a client
//!   derives a declaration from, and the node's tests hold its engine to
//!   accepting what that rule produces.
//! - **The shape of a node's response** ([`STREAM_CADENCE`]): a streamed one
//!   is written as frames of one size on a fixed tick, through
//!   `stream_shape`'s framer, and read back by its reader; every other one
//!   is padded to `stream_shape::body_bucket`.
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

/// The most a gateway's forwarded body can exceed the client's body it was
/// made from, in bytes. The gateway re-emits the body compactly with every
/// scalar re-spelled as `serde_json` writes it: a string never grows (an
/// escape `serde_json` writes is one the input had to spell at least as
/// long), and a number grows to at most [`RESPELLED_NUMBER_BYTES`], so the
/// growth is bounded by the value count; the members the gateway writes
/// itself (the model's hash, the completion ceiling, `stream_options`) fit in
/// the slack.
pub const FORWARDED_BODY_GROWTH_BYTES: usize = 8 * 1024 * 1024;

/// The longest text `serde_json` writes for a number: an `f64` in its
/// shortest round-trip form (`-1.2345678901234567e-300`).
pub const RESPELLED_NUMBER_BYTES: usize = 24;

const _: () = assert!(
    MAX_REQUEST_JSON_VALUES * RESPELLED_NUMBER_BYTES + 64 * 1024 <= FORWARDED_BODY_GROWTH_BYTES
);

/// Largest request body the gateway reads from a client, in bytes: the node's
/// [`MAX_REQUEST_BODY_BYTES`] less [`FORWARDED_BODY_GROWTH_BYTES`], so a body
/// the gateway accepts is never one a node refuses for its size. At 24 MiB it
/// holds a prompt that fills the largest context an architecture declares
/// (2^20 positions) at 24 bytes of JSON a token.
pub const GATEWAY_MAX_REQUEST_BODY_BYTES: usize =
    MAX_REQUEST_BODY_BYTES - FORWARDED_BODY_GROWTH_BYTES;

/// The cadence a node writes a streamed chat response at: one frame of
/// 4 KiB every 20 ms, from the moment the response opens until the frame
/// that carries its last event (`stream_shape`).
///
/// **What it must carry.** A node sends one chunk per engine step, not per
/// token: a step's accepted tokens (one, plus up to three accepted drafts at
/// the deployments' draft width) are one delta. A chunk on the wire is about
/// 400 bytes (the response id, the model's 64-digit hash, the choice
/// envelope and the running usage, around a few bytes of text), so the rate,
/// 200 KiB/s, carries about 500 chunks a second: a step every 2 ms, and up to
/// 2,000 tokens a second for one stream at four tokens a step. The fastest
/// single-request decode step measured for the served model on its GPU is
/// 7.3 ms (in another serving engine), so the cap does not bind; past it, output waits in
/// the node's backlog and is delivered at the cap. The engine's aggregate
/// rate across streams is not what matters: each stream has its own frames.
///
/// **What it costs.** Up to one tick (20 ms) between a step's output and its
/// frame, before the gateway re-frames it for its client; and 200 KiB/s on
/// the connection between the gateway and the node for as long as a response
/// runs, mostly padding.
pub const STREAM_CADENCE: crate::stream_shape::Cadence =
    crate::stream_shape::Cadence::new(4096, std::time::Duration::from_millis(20));

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
/// (and allocates per value) sees a body within both limits. That holds for a
/// body refused as [`JsonShapeError::Malformed`] too: the scan counts as it
/// goes, so what precedes the fault is within both limits.
///
/// Strict RFC 8259, as `serde_json` is: no comments, no trailing commas, no
/// leading zeros, whitespace only between tokens. A `\uXXXX` escape is
/// checked for its shape only; what it decodes to (a lone surrogate, say) is
/// for the parsers that follow to refuse.
pub fn check_request_json(body: &[u8]) -> Result<JsonShape, JsonShapeError> {
    use JsonShapeError::{Malformed, TooDeep, TooManyValues};
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
            // UTF-8 is checked last: the scan reads bytes, so a body past a
            // limit is refused for it whatever its strings hold, and a body
            // that is not UTF-8 goes on to the parsers within both limits.
            return if need == Need::AfterValue && depth == 0 && core::str::from_utf8(body).is_ok() {
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
    /// A declared cached prefix the engine's cache does not hold.
    pub const CACHED_PREFIX_MISSING: &str = "cached_prefix_missing";
    /// A prompt longer than its declaration allows.
    pub const PROMPT_BUDGET_EXCEEDED: &str = "prompt_budget_exceeded";
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

/// The request body member a declared cached prefix travels in.
pub const CACHED_PREFIX_MEMBER: &str = "cached_prefix";

/// The step a client rounds a declared prefix down to
/// ([`declarable_cached_tokens`]). An engine's cache resumes only at block
/// boundaries, so a deployment's KV block size must divide this
/// ([`quantum_covers_block_size`]) for the rule's result to be one.
pub const CACHED_PREFIX_QUANTUM: u32 = 256;

/// A request's declaration of its cached prefix: the `cached_prefix` member,
/// `{"cached_tokens": <n>, "new_tokens": <n>}`.
///
/// It says two things about the prompt as the node will tokenize it: its first
/// `cached_tokens` tokens are held in the prefix cache under the request's
/// `cache_key`, and it has at most `new_tokens` after them. The request is
/// priced by exactly these two numbers (`crate::pricing::Prompt::Declared`),
/// so a node runs it only if both hold and refuses it otherwise before
/// computing anything ([`DeclaredPrefixRefusal`]). The declaration is the
/// client's own statement about its own earlier requests; nothing a server
/// remembered sets the price.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CachedPrefix {
    /// Leading prompt tokens served from the prefix cache. At least 1.
    pub cached_tokens: u32,
    /// The most prompt tokens the request may have after them.
    pub new_tokens: u32,
}

impl CachedPrefix {
    /// The longest prompt, in tokens, the declaration covers.
    pub const fn prompt_limit(self) -> u64 {
        self.cached_tokens as u64 + self.new_tokens as u64
    }

    /// The member's value as a client writes it.
    pub fn to_value(self) -> serde_json::Value {
        serde_json::json!({
            "cached_tokens": self.cached_tokens,
            "new_tokens": self.new_tokens,
        })
    }

    /// Writes the member into a request body built by
    /// [`crate::chat_completion_request_body`].
    pub fn insert_into(self, body: &mut serde_json::Value) {
        body[CACHED_PREFIX_MEMBER] = self.to_value();
    }

    /// Reads the member's value: an object with exactly the two members, each
    /// an integer that fits `u32`. `None` for any other shape. Whether the
    /// declaration is admissible is [`check_cached_prefix`]'s to say.
    pub fn from_value(value: &serde_json::Value) -> Option<Self> {
        let object = value.as_object()?;
        if object.len() != 2 {
            return None;
        }
        let member = |name: &str| u32::try_from(object.get(name)?.as_u64()?).ok();
        Some(Self {
            cached_tokens: member("cached_tokens")?,
            new_tokens: member("new_tokens")?,
        })
    }
}

/// The node's rule for a declared cached prefix: it needs the cache key whose
/// entries it names, and it declares at least one cached token (a request with
/// nothing cached sends no declaration).
pub fn check_cached_prefix(
    prefix: Option<CachedPrefix>,
    has_cache_key: bool,
) -> Result<(), SubsetError> {
    match prefix {
        None => Ok(()),
        Some(_) if !has_cache_key => Err(SubsetError::CachedPrefixWithoutKey),
        Some(p) if p.cached_tokens == 0 => Err(SubsetError::EmptyCachedPrefix),
        Some(_) => Ok(()),
    }
}

/// Why a node refused a request for its declared cached prefix. Either way the
/// node rendered and tokenized the request and computed nothing: no prefill,
/// and no cached block mapped, promoted or decrypted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaredPrefixRefusal {
    /// The cache holds fewer of the prompt's leading tokens than declared: the
    /// entry expired, was evicted, never existed on this node, or belongs to a
    /// prompt this one does not extend.
    Missing,
    /// The prompt has more tokens than `cached_tokens + new_tokens`.
    OverBudget,
}

impl DeclaredPrefixRefusal {
    /// The status a node, and the gateway relaying it, answers with.
    pub const STATUS: u16 = 409;

    /// The `error.type` a node, and the gateway relaying it, answers with.
    pub const fn error_type(self) -> &'static str {
        match self {
            DeclaredPrefixRefusal::Missing => error_type::CACHED_PREFIX_MISSING,
            DeclaredPrefixRefusal::OverBudget => error_type::PROMPT_BUDGET_EXCEEDED,
        }
    }
}

/// Whether a refusal, by its status and `error.type` exactly, is one of the
/// two a declared cached prefix can draw. A gateway returns it to its client
/// without trying another node (the cache a declaration names is on the node
/// its key routes to), and a client reads it to send the request again
/// without a declaration.
pub fn declared_prefix_refusal(status: u16, kind: &str) -> Option<DeclaredPrefixRefusal> {
    if status != DeclaredPrefixRefusal::STATUS {
        return None;
    }
    [
        DeclaredPrefixRefusal::Missing,
        DeclaredPrefixRefusal::OverBudget,
    ]
    .into_iter()
    .find(|r| r.error_type() == kind)
}

/// The cached prefix a client may declare for a request whose prompt begins
/// with the whole prompt, `anchor_prompt_tokens` long, of an earlier request
/// under the same cache key: one that asked for a new assistant turn, so that
/// its prompt ended at the template's generation prompt.
///
/// An engine keeps a resumable point at the last block boundary before a
/// prompt's final token (the final token is recomputed to produce the first
/// output), so the rule is the largest multiple of [`CACHED_PREFIX_QUANTUM`]
/// strictly below `anchor_prompt_tokens`. `0` means there is nothing to
/// declare.
///
/// **Only such a prompt is an anchor.** A request that extends it (a new turn,
/// a tool round, a regeneration, a continuation of its answer) renders to a
/// token sequence that begins with the anchor's, because the template's
/// markers are single tokens that text cannot merge across. Nothing after the
/// anchor is declared, the answer it drew included: a continuation sends that
/// answer back as text, the cache is keyed by tokens, and text does not always
/// encode to the tokens a model sampled, nor does a longer text always begin
/// with the tokens of a shorter one. So a continuation's own prompt, which
/// ends inside a message, is never an anchor either.
pub const fn declarable_cached_tokens(anchor_prompt_tokens: u64) -> u32 {
    let quantum = CACHED_PREFIX_QUANTUM as u64;
    let declarable = (anchor_prompt_tokens.saturating_sub(1) / quantum) * quantum;
    if declarable > u32::MAX as u64 {
        // The largest multiple of the quantum a `u32` holds.
        u32::MAX - (u32::MAX % CACHED_PREFIX_QUANTUM)
    } else {
        declarable as u32
    }
}

/// The declaration for a request whose prompt is an anchor's
/// ([`declarable_cached_tokens`]) followed by `added_content_bytes` of priced
/// content in `added_messages` messages: the new turn's messages, a tool
/// round's call and results, or, for a continuation, the partial assistant
/// message holding the answer so far. Both are zero for a regeneration.
///
/// The cached part is the rule's. The budget for the rest is the anchor's
/// tokens past the declared boundary, which the client knows exactly, and the
/// byte contract's count for what was added ([`crate::chargeable_prompt_tokens`],
/// whose constants cover the added messages' template tokens). `None` when
/// there is nothing to declare, or the budget does not fit the member.
///
/// A prompt whose added part is denser than the byte contract assumes can
/// exceed the budget; the node then refuses it
/// ([`DeclaredPrefixRefusal::OverBudget`]) and the request is sent again
/// without a declaration.
pub fn declared_prompt(
    anchor_prompt_tokens: u64,
    added_content_bytes: u64,
    added_messages: u64,
) -> Option<CachedPrefix> {
    let cached_tokens = declarable_cached_tokens(anchor_prompt_tokens);
    if cached_tokens == 0 {
        return None;
    }
    let carried = anchor_prompt_tokens - cached_tokens as u64;
    let added = crate::chargeable_prompt_tokens(added_content_bytes, added_messages);
    let new_tokens = u32::try_from(carried.checked_add(added)?).ok()?;
    Some(CachedPrefix {
        cached_tokens,
        new_tokens,
    })
}

/// Whether [`declarable_cached_tokens`] lands on a block boundary of an engine
/// with this KV block size: the block size is non-zero and divides the
/// quantum. A node that serves a prefix cache refuses to boot otherwise.
pub const fn quantum_covers_block_size(block_size: u32) -> bool {
    block_size != 0 && CACHED_PREFIX_QUANTUM.is_multiple_of(block_size)
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
    /// `continue_final_message` other than a boolean, `"content"` or
    /// `"reasoning_content"`.
    UnsupportedContinuation,
    /// `add_generation_prompt` contradicting `continue_final_message`.
    GenerationPromptMismatch,
    /// A continuation whose final message cannot be continued
    /// ([`check_continuation`]).
    UncontinuableFinalMessage,
    /// A `cached_prefix` on a request with no `cache_key`.
    CachedPrefixWithoutKey,
    /// A `cached_prefix` declaring no cached token.
    EmptyCachedPrefix,
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
            SubsetError::UnsupportedContinuation => f.write_str(
                "continue_final_message must be a boolean, \"content\" or \"reasoning_content\"",
            ),
            SubsetError::GenerationPromptMismatch => f.write_str(
                "add_generation_prompt must be false with continue_final_message and true \
                 without it",
            ),
            SubsetError::UncontinuableFinalMessage => f.write_str(
                "continue_final_message needs a final assistant message without tool calls, \
                 with string content to continue the content, or with string \
                 reasoning_content and no content to continue the reasoning",
            ),
            SubsetError::CachedPrefixWithoutKey => {
                f.write_str("cached_prefix needs the cache_key its entries are under")
            }
            SubsetError::EmptyCachedPrefix => {
                f.write_str("cached_prefix.cached_tokens must be at least 1")
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

/// Which field of the final assistant message a continuation extends.
///
/// The request members are vLLM's and SGLang's (`continue_final_message`, with
/// `add_generation_prompt: false`), and their meaning is the one
/// `transformers.apply_chat_template` gives its own `continue_final_message`
/// argument: `true` continues the message's `content`, and a string names the
/// field to continue. [`ContinueField::wire_value`] is that argument's value,
/// so a node renders exactly what the reference renders for the same request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContinueField {
    /// The answer: the prompt ends inside the message's `content`.
    Content,
    /// The reasoning: the prompt ends inside the message's `reasoning_content`,
    /// with its reasoning block still open, for a message that stopped before
    /// any answer.
    ReasoningContent,
}

impl ContinueField {
    /// The message field the continuation extends.
    pub const fn field(self) -> &'static str {
        match self {
            ContinueField::Content => "content",
            ContinueField::ReasoningContent => "reasoning_content",
        }
    }

    /// The `continue_final_message` value a client sends: `true` for the
    /// content (the spelling every OpenAI-compatible server that has the member
    /// accepts), the field's name for the reasoning.
    pub fn wire_value(self) -> serde_json::Value {
        match self {
            ContinueField::Content => serde_json::Value::Bool(true),
            ContinueField::ReasoningContent => serde_json::Value::String(self.field().to_owned()),
        }
    }
}

/// The shape of a request's final message, as [`check_continuation`] reads it.
/// Both sides fill it from their own typed request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalMessage<'a> {
    /// `role` is `assistant`.
    pub is_assistant: bool,
    pub content: FinalContent<'a>,
    /// `tool_calls` is present and non-empty.
    pub has_tool_calls: bool,
    /// `reasoning_content`, when present and a string.
    pub reasoning_content: Option<&'a str>,
}

/// A final message's `content`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalContent<'a> {
    /// Absent or `null`.
    Absent,
    Text(&'a str),
    /// An array of content parts.
    Parts,
}

/// The node's continuation rule: what `continue_final_message` and
/// `add_generation_prompt` may say, and of which final message.
///
/// - `continue_final_message` absent or `false` asks for a new assistant turn;
///   `true` or `"content"` continues the content, and `"reasoning_content"` the
///   reasoning. Any other value is refused.
/// - `add_generation_prompt` may be omitted. When given it must agree:
///   `false` with a continuation (vLLM refuses the pair `true`/`true`, and
///   requires `false` because its default is `true`), `true` without one. A
///   node has no use for a prompt that ends after a closed final message, so
///   `false` alone is refused rather than rendered.
/// - The message continued is the last one, and it must be an assistant's,
///   without tool calls (their markup follows the content in the prompt, so
///   continuing the content would silently drop them). Continuing the content
///   needs a string content (`""` included); continuing the reasoning needs a
///   string `reasoning_content` and no content (`""` included), since an
///   answer after the reasoning would otherwise be dropped from the prompt.
///
/// Returns the field to continue, or `None` for an ordinary request.
pub fn check_continuation(
    continue_final_message: Option<&serde_json::Value>,
    add_generation_prompt: Option<bool>,
    final_message: Option<FinalMessage<'_>>,
) -> Result<Option<ContinueField>, SubsetError> {
    use serde_json::Value;
    let field = match continue_final_message {
        None | Some(Value::Bool(false)) => None,
        Some(Value::Bool(true)) => Some(ContinueField::Content),
        Some(Value::String(s)) if s == "content" => Some(ContinueField::Content),
        Some(Value::String(s)) if s == "reasoning_content" => Some(ContinueField::ReasoningContent),
        Some(_) => return Err(SubsetError::UnsupportedContinuation),
    };
    match (field, add_generation_prompt) {
        (Some(_), Some(true)) | (None, Some(false)) => {
            return Err(SubsetError::GenerationPromptMismatch);
        }
        _ => {}
    }
    let Some(field) = field else {
        return Ok(None);
    };
    let Some(last) = final_message else {
        return Err(SubsetError::UncontinuableFinalMessage);
    };
    let shaped = last.is_assistant
        && !last.has_tool_calls
        && match field {
            ContinueField::Content => matches!(last.content, FinalContent::Text(_)),
            ContinueField::ReasoningContent => {
                last.reasoning_content.is_some()
                    && matches!(last.content, FinalContent::Absent | FinalContent::Text(""))
            }
        };
    if !shaped {
        return Err(SubsetError::UncontinuableFinalMessage);
    }
    Ok(Some(field))
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
    fn the_continuation_rule_accepts_and_refuses_at_its_bounds() {
        use serde_json::json;
        let answer = |content| FinalMessage {
            is_assistant: true,
            content,
            has_tool_calls: false,
            reasoning_content: Some("thinking"),
        };
        let partial = Some(answer(FinalContent::Text("so far")));
        let reasoning_only = Some(answer(FinalContent::Absent));

        // An ordinary request, however the two members spell it.
        for (cont, agp) in [(None, None), (Some(json!(false)), None), (None, Some(true))] {
            assert_eq!(check_continuation(cont.as_ref(), agp, partial), Ok(None));
        }
        // Continuing the content, in both spellings, with the generation prompt
        // omitted or turned off.
        for cont in [json!(true), json!("content")] {
            for agp in [None, Some(false)] {
                assert_eq!(
                    check_continuation(Some(&cont), agp, partial),
                    Ok(Some(ContinueField::Content))
                );
            }
        }
        // An empty content is still a content to continue.
        assert_eq!(
            check_continuation(
                Some(&json!(true)),
                None,
                Some(answer(FinalContent::Text("")))
            ),
            Ok(Some(ContinueField::Content))
        );
        // Continuing the reasoning, with no answer after it.
        for content in [FinalContent::Absent, FinalContent::Text("")] {
            assert_eq!(
                check_continuation(
                    Some(&json!("reasoning_content")),
                    None,
                    Some(answer(content))
                ),
                Ok(Some(ContinueField::ReasoningContent))
            );
        }

        for cont in [json!("tool_calls"), json!(1), json!(null), json!("CONTENT")] {
            assert_eq!(
                check_continuation(Some(&cont), None, partial),
                Err(SubsetError::UnsupportedContinuation),
                "{cont}"
            );
        }
        assert_eq!(
            check_continuation(Some(&json!(true)), Some(true), partial),
            Err(SubsetError::GenerationPromptMismatch)
        );
        assert_eq!(
            check_continuation(None, Some(false), partial),
            Err(SubsetError::GenerationPromptMismatch)
        );

        let uncontinuable = [
            (json!(true), None),
            (json!(true), reasoning_only),
            (
                json!(true),
                Some(FinalMessage {
                    is_assistant: false,
                    ..answer(FinalContent::Text("a user's words"))
                }),
            ),
            (
                json!(true),
                Some(FinalMessage {
                    has_tool_calls: true,
                    ..answer(FinalContent::Text("calling"))
                }),
            ),
            (json!(true), Some(answer(FinalContent::Parts))),
            // The answer after it would be dropped from the prompt.
            (json!("reasoning_content"), partial),
            (
                json!("reasoning_content"),
                Some(FinalMessage {
                    reasoning_content: None,
                    ..answer(FinalContent::Absent)
                }),
            ),
        ];
        for (cont, last) in uncontinuable {
            assert_eq!(
                check_continuation(Some(&cont), None, last),
                Err(SubsetError::UncontinuableFinalMessage),
                "{cont} {last:?}"
            );
        }

        assert_eq!(ContinueField::Content.wire_value(), json!(true));
        assert_eq!(
            ContinueField::ReasoningContent.wire_value(),
            json!("reasoning_content")
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
        // The limits come before the encoding: a body past the cap is refused
        // for it even when a string in it is not UTF-8, so no parser that
        // allocates per value ever reads it.
        let mut hostile = over.clone().into_bytes();
        let end = hostile.len() - 1;
        hostile.splice(end..end, *b",\"\xff\"");
        assert_eq!(
            check_request_json(&hostile),
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
    #[test]
    fn a_cached_prefix_has_one_shape() {
        use serde_json::json;
        let prefix = CachedPrefix {
            cached_tokens: 4_096,
            new_tokens: 300,
        };
        assert_eq!(CachedPrefix::from_value(&prefix.to_value()), Some(prefix));
        assert_eq!(prefix.prompt_limit(), 4_396);
        assert_eq!(
            CachedPrefix {
                cached_tokens: u32::MAX,
                new_tokens: u32::MAX
            }
            .prompt_limit(),
            2 * u32::MAX as u64
        );
        for bad in [
            json!(null),
            json!(4096),
            json!([4096, 300]),
            json!({}),
            json!({"cached_tokens": 4096}),
            json!({"new_tokens": 300}),
            json!({"cached_tokens": 4096, "new_tokens": 300, "extra": 1}),
            json!({"cached_tokens": "4096", "new_tokens": 300}),
            json!({"cached_tokens": 4096.5, "new_tokens": 300}),
            json!({"cached_tokens": -1, "new_tokens": 300}),
            json!({"cached_tokens": 4_294_967_296_u64, "new_tokens": 300}),
            json!({"cached_tokens": 4096, "new_tokens": null}),
        ] {
            assert_eq!(CachedPrefix::from_value(&bad), None, "{bad}");
        }
    }

    #[test]
    fn a_cached_prefix_needs_a_key_and_a_cached_token() {
        let prefix = |cached_tokens| {
            Some(CachedPrefix {
                cached_tokens,
                new_tokens: 0,
            })
        };
        assert_eq!(check_cached_prefix(None, false), Ok(()));
        assert_eq!(check_cached_prefix(None, true), Ok(()));
        assert_eq!(check_cached_prefix(prefix(1), true), Ok(()));
        assert_eq!(
            check_cached_prefix(prefix(1), false),
            Err(SubsetError::CachedPrefixWithoutKey)
        );
        assert_eq!(
            check_cached_prefix(prefix(0), true),
            Err(SubsetError::EmptyCachedPrefix)
        );
    }

    #[test]
    fn only_the_two_declared_prefix_refusals_read_as_one() {
        use error_type as t;
        assert_eq!(
            declared_prefix_refusal(409, t::CACHED_PREFIX_MISSING),
            Some(DeclaredPrefixRefusal::Missing)
        );
        assert_eq!(
            declared_prefix_refusal(409, t::PROMPT_BUDGET_EXCEEDED),
            Some(DeclaredPrefixRefusal::OverBudget)
        );
        for (status, ty) in [
            (400, t::CACHED_PREFIX_MISSING),
            (503, t::PROMPT_BUDGET_EXCEEDED),
            (409, t::INVALID_REQUEST),
            (409, "conflict"),
            (409, ""),
        ] {
            assert_eq!(declared_prefix_refusal(status, ty), None, "{status} {ty}");
        }
        // Neither is a refusal another node may be sent.
        for refusal in [
            DeclaredPrefixRefusal::Missing,
            DeclaredPrefixRefusal::OverBudget,
        ] {
            assert_eq!(
                refused_before_admission(DeclaredPrefixRefusal::STATUS, refusal.error_type()),
                None
            );
        }
    }

    #[test]
    fn a_declarable_prefix_is_the_last_quantum_boundary_before_the_final_token() {
        let q = CACHED_PREFIX_QUANTUM as u64;
        assert_eq!(declarable_cached_tokens(0), 0);
        assert_eq!(declarable_cached_tokens(1), 0);
        assert_eq!(declarable_cached_tokens(q), 0);
        assert_eq!(declarable_cached_tokens(q + 1), q as u32);
        assert_eq!(declarable_cached_tokens(2 * q), q as u32);
        assert_eq!(declarable_cached_tokens(2 * q + 1), 2 * q as u32);
        for anchor in [1, 255, 256, 257, 1_000, 65_537, 1_048_576] {
            let declared = declarable_cached_tokens(anchor) as u64;
            assert_eq!(declared % q, 0);
            assert!(declared < anchor);
            assert!(anchor - declared <= q);
        }
        let most = declarable_cached_tokens(u64::MAX);
        assert_eq!(most % CACHED_PREFIX_QUANTUM, 0);
        assert!(u32::MAX - most < CACHED_PREFIX_QUANTUM);
    }

    #[test]
    fn a_declared_prompt_budgets_the_carried_tokens_and_the_added_bytes() {
        // Nothing to declare below one quantum.
        assert_eq!(declared_prompt(256, 10_000, 2), None);
        // A regeneration: an anchor of 10,000 tokens, nothing added. 9,984
        // cached, 16 carried, and the per-request allowance.
        assert_eq!(
            declared_prompt(10_000, 0, 0),
            Some(CachedPrefix {
                cached_tokens: 9_984,
                new_tokens: 16 + crate::PER_REQUEST_TOKENS as u32,
            })
        );
        // A new turn: 3,000 bytes in two messages added.
        let turn = declared_prompt(10_000, 3_000, 2).unwrap();
        assert_eq!(turn.cached_tokens, 9_984);
        assert_eq!(
            turn.new_tokens as u64,
            16 + crate::chargeable_prompt_tokens(3_000, 2)
        );
        // The budget always covers the anchor's own tokens.
        for anchor in [257, 1_000, 65_536, 1_048_576] {
            let p = declared_prompt(anchor, 0, 0).unwrap();
            assert!(p.prompt_limit() >= anchor);
        }
        // A budget past the member's range is not declared.
        assert_eq!(declared_prompt(10_000, u64::MAX, 0), None);
        assert_eq!(declared_prompt(10_000, 9_000_000_000, 0), None);
    }

    #[test]
    fn a_cached_prefix_is_written_into_a_built_body() {
        let mut body =
            crate::chat_completion_request_body("m", &[], 1, &[], false, false, None, None);
        let prefix = CachedPrefix {
            cached_tokens: 512,
            new_tokens: 40,
        };
        prefix.insert_into(&mut body);
        assert_eq!(
            CachedPrefix::from_value(&body[CACHED_PREFIX_MEMBER]),
            Some(prefix)
        );
    }

    #[test]
    fn the_quantum_covers_power_of_two_block_sizes_up_to_itself() {
        for block in [1, 2, 16, 32, 64, 128, 256] {
            assert!(quantum_covers_block_size(block), "{block}");
        }
        for block in [0, 3, 24, 48, 100, 512] {
            assert!(!quantum_covers_block_size(block), "{block}");
        }
    }
}
