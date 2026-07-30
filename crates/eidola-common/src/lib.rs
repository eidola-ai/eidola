//! Shared client/server contract logic for eidola.
//!
//! This crate holds the small pieces of pure logic that MUST be bit-identical
//! on both sides of a contract boundary — the anonymous-credit pricing
//! contract for chat-completion prompt holds ([`chargeable_prompt_tokens`]),
//! and the embed-marker recognition rule shared between the markdown
//! editor's embed plugin and app-core's upstream quote expansion
//! ([`embed`]). It is intentionally lib-only, pure Rust, and float-free so
//! `eidola-app-core`, `eidola-server`, and tests can depend on it and
//! compute identical results on any platform.
//!
//! # The dependency rule
//!
//! This crate was zero-dependency, which cost more fidelity than it bought:
//! unable to see a request, it could own the pricing *arithmetic* but not
//! the *walk* — which parts of a request count and how each is measured —
//! so each consumer reimplemented the walk and the two could drift. The
//! rule that replaced it:
//!
//! > **A dependency is admissible here only if it is already in every
//! > consumer's graph and is required for contract fidelity.** Today that
//! > set is `serde_json`. Any addition needs the same argument written
//! > down.
//!
//! `serde_json` qualifies on both counts: every consumer (app-core, server,
//! gui; cli via app-core) already carries it, so admitting it adds no code,
//! no trust surface, and no compile time to any binary — and it is what
//! lets [`prompt_charge`] be the single walk both sides call rather than
//! two implementations held together by a test.
//!
//! # The prompt-hold pricing contract
//!
//! Each inference turn is paid with an anonymous credit token (ACT): the
//! client *holds* (spends) a worst-case charge estimate up front, the server
//! computes the actual charge from real token usage, and refunds the
//! difference. Two defects motivated this contract:
//!
//! 1. **Bytes-as-tokens is not an upper bound.** The old prompt-side hold
//!    treated content bytes as tokens (`total_prompt_bytes × per-token
//!    rate`). That over-holds ~4× for prose, but chat templates add
//!    per-message tokens (role markers, BOS/EOS, scaffolding) beyond content
//!    bytes, so a context of many tiny messages can have *more* real prompt
//!    tokens than content bytes. The server's charge could then exceed the
//!    hold — a rough failure, and with unlinkable ACTs an under-allocation
//!    is an abuse vector.
//! 2. **No shared, enforced formula.** The go/no-go logic must be
//!    bit-identical on the client and the server; two independent
//!    implementations drift.
//!
//! [`chargeable_prompt_tokens`] is the single shared formula. It is
//! simultaneously:
//!
//! - **(a) the client's prompt-side hold**: the client sizes the prompt
//!   component of its ACT spend as
//!   `chargeable_prompt_tokens(bytes, count) × prompt_rate`;
//! - **(b) the server's pre-flight minimum**: before calling upstream, the
//!   server recomputes the same function of the same request and rejects any
//!   spend that presents less;
//! - **(c) the server's cap on charged prompt tokens**:
//!   `charged_prompt_tokens = min(actual_prompt_tokens,
//!   chargeable_prompt_tokens(...))`.
//!
//! # Safety invariants
//!
//! - **Hold ≥ charge by construction.** Both sides compute the same function
//!   of the same request (same `messages` array → same content bytes and
//!   message count), and the server clamps its charged prompt tokens to the
//!   formula's value. No under-allocation is possible.
//! - **Cost recovery.** For any BPE tokenizer, actual tokens ≤ content bytes.
//!   The byte term charges at least `bytes / N` where
//!   `N = SAFE_COST_FACTOR_NUM / SAFE_COST_FACTOR_DEN`, so the worst-case
//!   actual/charged token ratio on the byte term is exactly `N`. Break-even
//!   on dynamic costs therefore requires `N ≤ PRICING_MARKUP`; the server
//!   asserts `PRICING_MARKUP ≥ N` at startup and refuses to start otherwise
//!   (a markup below the factor would silently open a loss window). The
//!   factor is thus "the safe cost factor" — the floor of the markup.
//!
//! # Inputs
//!
//! - `total_content_bytes`: the sum of UTF-8 byte lengths of every part of
//!   the request the model actually reads — see [`PromptCharge`] for the
//!   exact enumeration.
//! - `message_count`: the number of entries in the `messages` array.
//!
//! Roles and JSON structure are deliberately excluded from the byte count;
//! their token cost is covered by [`PER_MESSAGE_TOKENS`] and
//! [`PER_REQUEST_TOKENS`].
//!
//! # Gathering the inputs: [`prompt_charge`]
//!
//! *What* counts is as much a part of the contract as the arithmetic, and it
//! is where the two sides could silently drift. [`prompt_charge`] is the one
//! walk: it reads a request's `messages` and `tools` arrays as
//! `serde_json::Value` and returns a [`PromptCharge`]. app-core already
//! holds its messages as `Value`s; the server converts its parsed
//! `ChatCompletionRequest` with `serde_json::to_value` at the pricing call
//! site (negligible against a request it already fully re-serializes to
//! forward upstream). [`PromptCharge`] itself stays pure arithmetic — serde
//! is for the walk, never for the accounting semantics.

pub mod embed;

use serde_json::Value;

/// Numerator of the safe cost factor `N = NUM/DEN = 1.5`.
///
/// Kept as an integer ratio so client and server compute identically —
/// floats are forbidden in the formula. `N` is the floor of the server's
/// `PRICING_MARKUP` (see the crate docs' cost-recovery invariant).
pub const SAFE_COST_FACTOR_NUM: u64 = 3;

/// Denominator of the safe cost factor `N = NUM/DEN = 1.5`.
pub const SAFE_COST_FACTOR_DEN: u64 = 2;

/// Per-message token allowance, covering chat-template per-message overhead
/// (role markers, message BOS/EOS, scaffolding).
pub const PER_MESSAGE_TOKENS: u64 = 8;

/// Per-request token allowance, covering BOS/system-preamble slack.
pub const PER_REQUEST_TOKENS: u64 = 32;

/// The shared prompt-token formula of the client/server pricing contract:
///
/// ```text
/// chargeable_prompt_tokens(total_content_bytes, message_count) =
///     ceil(total_content_bytes * SAFE_COST_FACTOR_DEN / SAFE_COST_FACTOR_NUM)
///     + PER_MESSAGE_TOKENS * message_count
///     + PER_REQUEST_TOKENS
/// ```
///
/// See the crate docs for the three roles this value plays (client hold,
/// server pre-flight minimum, server charge cap) and the safety invariants.
/// Integer-exact: computed in `u128` internally (no intermediate overflow for
/// any `u64` inputs) and saturated to `u64::MAX` on return — saturation can
/// only ever *raise* the hold/cap, never open an under-charge.
pub fn chargeable_prompt_tokens(total_content_bytes: u64, message_count: u64) -> u64 {
    let byte_term = (total_content_bytes as u128 * SAFE_COST_FACTOR_DEN as u128)
        .div_ceil(SAFE_COST_FACTOR_NUM as u128);
    let message_term = PER_MESSAGE_TOKENS as u128 * message_count as u128;
    let total = byte_term + message_term + PER_REQUEST_TOKENS as u128;
    u64::try_from(total).unwrap_or(u64::MAX)
}

/// The gathered inputs of the prompt-hold pricing contract for one
/// chat-completion request.
///
/// # What counts
///
/// Everything the model reads is charged, at the same safe cost factor:
///
/// | Part of the request | Method | Counted as |
/// | --- | --- | --- |
/// | a `messages` entry | [`add_message`](Self::add_message) | one message + its `content` string's bytes |
/// | an assistant `tool_calls` entry | [`add_tool_call`](Self::add_tool_call) | the entry's compact JSON serialization, in bytes |
/// | a request-level `tools` entry | [`add_tool_definition`](Self::add_tool_definition) | the entry's compact JSON serialization, in bytes |
///
/// # Why tool bytes are charged at all
///
/// Tool calling moves a large share of a turn's prompt out of `content`
/// strings: the advertised `tools` schemas are re-injected into the prompt by
/// the chat template on **every** round of an agentic loop, and a model's own
/// `arguments` are replayed verbatim in the follow-up request. Counting only
/// `content` would let all of it ride upstream uncharged — a cost-recovery
/// gap (never a safety one: the hold and the charge cap are the same
/// function, so hold ≥ charge held either way). It is the byte term, not a
/// new constant, because the cost-recovery invariant it rests on —
/// `actual_tokens ≤ bytes` for any BPE tokenizer — holds for JSON exactly as
/// it does for prose.
///
/// `role: "tool"` result messages need no special handling: their content is
/// an ordinary `content` string and was already counted by
/// [`add_message`](Self::add_message). An assistant message that only called
/// tools has no content string and contributes `0` there — its bytes arrive
/// through [`add_tool_call`](Self::add_tool_call).
///
/// # Both sides must walk their request the same way
///
/// The client sizes its hold and the server recomputes its pre-flight
/// minimum and charge cap from the *same* request, so every part above must
/// be reported by both or neither. Two rules make the walks comparable
/// across the two shapes:
///
/// * a `tools` entry is measured as its **compact JSON serialization**. The
///   server's value is parsed from the very bytes the client serialized, and
///   a serialization's *length* does not depend on key order, so the two
///   sides agree without needing a canonical form;
/// * a `tool_calls` entry is measured the same way, for the same reason.
///
/// The one thing charged per *message* rather than per byte is the
/// message-level scalar framing — `name`, and a tool result's
/// `tool_call_id` — which [`PER_MESSAGE_TOKENS`] already exists to cover.
/// That allowance is per message, so it cannot stretch over the *N* call
/// objects an assistant message may carry; those get the byte measure above.
///
/// All adds saturate: an absurd input can only ever raise the hold and the
/// cap, never open an under-charge.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptCharge {
    total_content_bytes: u64,
    message_count: u64,
}

impl PromptCharge {
    /// An empty request.
    pub const fn new() -> Self {
        Self {
            total_content_bytes: 0,
            message_count: 0,
        }
    }

    /// Account for one entry of the `messages` array: the per-message
    /// allowance plus its `content` string's UTF-8 byte length.
    ///
    /// Pass `0` for a message whose `content` is absent or `null` — the
    /// message itself still counts.
    pub fn add_message(&mut self, content_bytes: u64) -> &mut Self {
        self.total_content_bytes = self.total_content_bytes.saturating_add(content_bytes);
        self.message_count = self.message_count.saturating_add(1);
        self
    }

    /// Account for one `tool_calls` entry on an assistant message, measured
    /// as the UTF-8 byte length of its compact JSON serialization.
    ///
    /// **The whole entry, not just `function.{name,arguments}`.** A client
    /// replays the provider's call object verbatim and the proxy forwards it
    /// verbatim, so the `id`, the `type`, and any provider extension field
    /// all reach the upstream — and which of them the chat template renders
    /// into the prompt is a per-model decision the proxy cannot see (Mistral
    /// templates render the call id; Qwen and Llama ones do not). Charging
    /// the forwarded bytes is the measure that does not depend on knowing.
    ///
    /// The enclosing message is counted separately by
    /// [`add_message`](Self::add_message) — this only adds bytes.
    pub fn add_tool_call(&mut self, serialized_bytes: u64) -> &mut Self {
        self.total_content_bytes = self.total_content_bytes.saturating_add(serialized_bytes);
        self
    }

    /// Account for one entry of the request's `tools` array, measured as the
    /// UTF-8 byte length of its compact JSON serialization.
    pub fn add_tool_definition(&mut self, serialized_bytes: u64) -> &mut Self {
        self.total_content_bytes = self.total_content_bytes.saturating_add(serialized_bytes);
        self
    }

    /// The byte term's input: every counted byte of the request.
    pub const fn total_content_bytes(&self) -> u64 {
        self.total_content_bytes
    }

    /// The number of `messages` entries counted.
    pub const fn message_count(&self) -> u64 {
        self.message_count
    }

    /// The contract's chargeable prompt tokens for this request.
    pub fn chargeable_prompt_tokens(&self) -> u64 {
        chargeable_prompt_tokens(self.total_content_bytes, self.message_count)
    }
}

/// Byte length of a JSON value measured as text: its own bytes when it is a
/// string (the OpenAI shape for a tool call's `arguments`), otherwise its
/// compact JSON serialization.
///
/// # Two properties this measure rests on
///
/// **Key order cannot skew it.** The measure is a *length*, and a compact
/// serialization's length does not depend on the order its object keys are
/// emitted in. So even if `serde_json`'s `preserve_order` feature were ever
/// unified into the workspace graph — flipping `Map` from a sorted `BTreeMap`
/// to an insertion-ordered `IndexMap` — the client and server would still
/// agree. The dependency is nevertheless declared featureless on purpose:
/// relying on the invariance is a safety net, not a licence to perturb a
/// contract crate's dependency behaviour.
///
/// **Formatting is assumed stable across `serde_json` releases.** A byte
/// measure of a serializer's output is only cross-version-exact if that
/// serializer formats identically in both processes — which matters for
/// old-client/new-server pairs, since they are separate binaries built at
/// different times. The realistic exposure is float formatting inside a
/// `tools` JSON Schema (`{"multipleOf": 0.1}` and friends); integers,
/// strings, and structural bytes have one spelling. A divergence there would
/// shift the charge by a few bytes on a rare shape — bounded, and clamped on
/// the server to actual usage either way — so this is **documented, not
/// designed around**. It was equally true when the two sides each had their
/// own walk; centralizing the walk is what makes the assumption auditable in
/// one place instead of implicit in two.
pub fn json_text_bytes(value: &Value) -> u64 {
    match value.as_str() {
        Some(s) => s.len() as u64,
        None => serde_json::to_string(value).map(|s| s.len()).unwrap_or(0) as u64,
    }
}

/// **The** walk of the pricing contract: gather a chat-completion request's
/// chargeable parts into a [`PromptCharge`].
///
/// One implementation, called by both sides — app-core passes the `Value`
/// messages it already holds, the server passes the result of
/// `serde_json::to_value` on its parsed request. `tools` is `None` for a
/// request that advertises none (equivalent to an empty slice; the wire
/// distinguishes absent from empty, the charge does not).
///
/// # What is read
///
/// * **`content`** — a string contributes its UTF-8 bytes; an array of
///   content parts contributes the sum of its parts' `text` strings (a
///   multimodal message's image parts carry no prompt text of their own);
///   absent or `null` contributes zero. Every entry counts as one message
///   regardless.
/// * **`tool_calls[]`** — each entry contributes [`json_text_bytes`] of the
///   whole entry.
/// * **`tools[]`** — each entry contributes [`json_text_bytes`] of the whole
///   entry, once per request.
///
/// Anything not named above is deliberately not measured by byte: roles,
/// JSON structure, and the message-level scalars (`name`, `tool_call_id`)
/// are what [`PER_MESSAGE_TOKENS`] and [`PER_REQUEST_TOKENS`] cover.
pub fn prompt_charge(messages: &[Value], tools: Option<&[Value]>) -> PromptCharge {
    let mut charge = PromptCharge::new();

    for message in messages {
        charge.add_message(content_bytes(message.get("content")));
        if let Some(calls) = message.get("tool_calls").and_then(|c| c.as_array()) {
            for call in calls {
                charge.add_tool_call(json_text_bytes(call));
            }
        }
    }

    for tool in tools.unwrap_or(&[]) {
        charge.add_tool_definition(json_text_bytes(tool));
    }

    charge
}

/// Prompt-text bytes of a message's `content`: a plain string, or the summed
/// `text` of an array of content parts (the multimodal shape). Absent,
/// `null`, or any other shape contributes zero.
fn content_bytes(content: Option<&Value>) -> u64 {
    match content {
        Some(Value::String(s)) => s.len() as u64,
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| {
                p.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.len() as u64)
                    .unwrap_or(0)
            })
            .sum(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_messages_charges_only_the_per_request_constant() {
        assert_eq!(chargeable_prompt_tokens(0, 0), PER_REQUEST_TOKENS);
    }

    #[test]
    fn empty_messages_still_charge_per_message_overhead() {
        assert_eq!(
            chargeable_prompt_tokens(0, 3),
            3 * PER_MESSAGE_TOKENS + PER_REQUEST_TOKENS
        );
    }

    #[test]
    fn byte_term_uses_ceiling_division() {
        // 1 byte: ceil(1 * 2 / 3) = 1, not 0.
        assert_eq!(chargeable_prompt_tokens(1, 0), 1 + PER_REQUEST_TOKENS);
        // 2 bytes: ceil(4/3) = 2.
        assert_eq!(chargeable_prompt_tokens(2, 0), 2 + PER_REQUEST_TOKENS);
        // 3 bytes: 6/3 = 2 exactly (no rounding up past the true quotient).
        assert_eq!(chargeable_prompt_tokens(3, 0), 2 + PER_REQUEST_TOKENS);
        // 4 bytes: ceil(8/3) = 3.
        assert_eq!(chargeable_prompt_tokens(4, 0), 3 + PER_REQUEST_TOKENS);
    }

    #[test]
    fn typical_prompt() {
        // 300 bytes over 4 messages: ceil(600/3)=200 + 32 + 32 = 264.
        assert_eq!(chargeable_prompt_tokens(300, 4), 264);
    }

    #[test]
    fn charge_is_at_least_bytes_over_factor() {
        // Cost-recovery invariant: charged ≥ bytes / N for the byte term.
        for bytes in [0u64, 1, 2, 3, 100, 12345, 1_000_000] {
            let charged = chargeable_prompt_tokens(bytes, 0) - PER_REQUEST_TOKENS;
            assert!(
                charged as u128 * SAFE_COST_FACTOR_NUM as u128
                    >= bytes as u128 * SAFE_COST_FACTOR_DEN as u128,
                "byte term under-charges at {bytes} bytes"
            );
        }
    }

    #[test]
    fn many_tiny_messages_exceed_their_byte_count() {
        // The defect the constants fix: 100 one-byte messages have 100
        // content bytes but far more real prompt tokens once the chat
        // template adds per-message scaffolding. The formula must exceed
        // the raw byte count here.
        let v = chargeable_prompt_tokens(100, 100);
        assert!(v > 100, "per-message overhead must dominate: {v}");
    }

    // -----------------------------------------------------------------
    // PromptCharge: what counts
    // -----------------------------------------------------------------

    #[test]
    fn a_plain_conversation_charges_exactly_the_free_function() {
        let mut charge = PromptCharge::new();
        charge.add_message(100).add_message(200).add_message(0);
        assert_eq!(charge.total_content_bytes(), 300);
        assert_eq!(charge.message_count(), 3);
        assert_eq!(
            charge.chargeable_prompt_tokens(),
            chargeable_prompt_tokens(300, 3),
            "no tools ⇒ identical to the pre-tool-calling contract"
        );
    }

    #[test]
    fn tool_call_bytes_are_charged_on_a_content_less_message() {
        // An assistant message that only called tools: no content string,
        // but a function name and an arguments payload the model read.
        let mut with_call = PromptCharge::new();
        with_call.add_message(0).add_tool_call(93);

        let mut without = PromptCharge::new();
        without.add_message(0);

        assert_eq!(with_call.total_content_bytes(), 93);
        assert_eq!(with_call.message_count(), without.message_count());
        assert!(
            with_call.chargeable_prompt_tokens() > without.chargeable_prompt_tokens(),
            "tool-call bytes must not ride free"
        );
        // The bytes land in the byte term at the safe cost factor, like any
        // other bytes: ceil(93*2/3) = 62 tokens more than the empty message.
        assert_eq!(
            with_call.chargeable_prompt_tokens() - without.chargeable_prompt_tokens(),
            62
        );
    }

    #[test]
    fn several_tool_calls_on_one_message_count_once_as_a_message() {
        let mut charge = PromptCharge::new();
        charge.add_message(0).add_tool_call(40).add_tool_call(40);
        assert_eq!(charge.message_count(), 1, "one message, two calls");
        assert_eq!(charge.total_content_bytes(), 80);
    }

    #[test]
    fn tool_definitions_are_charged_once_per_request() {
        let mut charge = PromptCharge::new();
        charge.add_message(10).add_tool_definition(150);
        assert_eq!(charge.total_content_bytes(), 160);
        assert_eq!(
            charge.message_count(),
            1,
            "a tool definition is not a message"
        );
    }

    #[test]
    fn tool_result_content_counts_as_ordinary_content() {
        // A `role: "tool"` message is an ordinary message with a content
        // string — pinned so a later refactor can't start double-counting it
        // through `add_tool_call`.
        let mut tool_result = PromptCharge::new();
        tool_result.add_message(64);

        let mut user = PromptCharge::new();
        user.add_message(64);

        assert_eq!(tool_result, user);
    }

    #[test]
    fn accounting_saturates_rather_than_overflowing() {
        let mut charge = PromptCharge::new();
        charge
            .add_message(u64::MAX)
            .add_tool_call(u64::MAX)
            .add_tool_definition(u64::MAX);
        assert_eq!(charge.total_content_bytes(), u64::MAX);
        assert_eq!(
            charge.chargeable_prompt_tokens(),
            chargeable_prompt_tokens(u64::MAX, 1)
        );
    }

    // -----------------------------------------------------------------
    // prompt_charge: the walk
    // -----------------------------------------------------------------

    #[test]
    fn a_tool_less_request_walks_to_the_pre_tool_calling_value() {
        // The shape of nearly all traffic: no tools, no tool calls. Must be
        // exactly what the contract charged before tool calling existed.
        let messages = vec![
            serde_json::json!({"role": "system", "content": "be brief"}),
            serde_json::json!({"role": "user", "content": "héllo wörld"}),
        ];
        let bytes: u64 = messages
            .iter()
            .map(|m| m["content"].as_str().unwrap().len() as u64)
            .sum();
        assert_eq!(
            prompt_charge(&messages, None).chargeable_prompt_tokens(),
            chargeable_prompt_tokens(bytes, 2)
        );
    }

    #[test]
    fn absent_and_empty_tools_charge_alike() {
        let messages = vec![serde_json::json!({"role": "user", "content": "hi"})];
        assert_eq!(
            prompt_charge(&messages, None),
            prompt_charge(&messages, Some(&[]))
        );
    }

    #[test]
    fn multimodal_content_parts_contribute_their_text() {
        // A `content` array (the multimodal shape) contributes the sum of its
        // parts' `text`; an image part carries no prompt text of its own.
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": " this?"}
            ]
        })];
        let charge = prompt_charge(&messages, None);
        assert_eq!(charge.total_content_bytes(), 10 + 6);
        assert_eq!(charge.message_count(), 1);
    }

    #[test]
    fn absent_or_null_content_still_counts_as_a_message() {
        for content in [
            serde_json::json!({"role": "assistant", "content": null}),
            serde_json::json!({"role": "assistant"}),
        ] {
            let charge = prompt_charge(&[content], None);
            assert_eq!(charge.total_content_bytes(), 0);
            assert_eq!(charge.message_count(), 1);
        }
    }

    #[test]
    fn the_whole_tool_call_entry_is_charged_not_just_name_and_arguments() {
        // The client replays the provider's call object verbatim — id, `type`
        // and any provider extension included — and those bytes reach the
        // upstream, where a template like Mistral's renders the call id
        // straight into the prompt.
        let call = |extra: serde_json::Value| {
            let mut entry = serde_json::json!({
                "id": "c", "type": "function",
                "function": {"name": "calc", "arguments": "{}"}
            });
            if let Some(obj) = extra.as_object() {
                for (k, v) in obj {
                    entry[k] = v.clone();
                }
            }
            vec![serde_json::json!({
                "role": "assistant", "content": null, "tool_calls": [entry]
            })]
        };

        let plain = prompt_charge(&call(serde_json::json!({})), None);
        let long_id = prompt_charge(
            &call(serde_json::json!({"id": "call_9tK2mQz7Xb4LpR1vN8sYcE0d"})),
            None,
        );
        let extension = prompt_charge(
            &call(serde_json::json!({"provider_extra": {"trace": "0123456789abcdef"}})),
            None,
        );

        assert!(
            long_id.total_content_bytes() > plain.total_content_bytes(),
            "a longer call id is more forwarded bytes and must cost more"
        );
        assert!(
            extension.total_content_bytes() > plain.total_content_bytes(),
            "a forwarded provider extension must not ride free"
        );
    }

    #[test]
    fn non_string_tool_call_arguments_are_measured_as_json() {
        // Off-spec (the OpenAI shape mandates a string), but seen in the
        // wild. Measured as the entry's serialization like any other entry,
        // so both sides still agree.
        let messages = vec![serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id": "c", "function": {"name": "ca", "arguments": {"expr": "2+2"}}}]
        })];
        let charge = prompt_charge(&messages, None);
        assert_eq!(
            charge.total_content_bytes(),
            json_text_bytes(&messages[0]["tool_calls"][0])
        );
    }

    #[test]
    fn json_text_bytes_reads_a_string_as_itself_not_as_quoted_json() {
        // A quoted re-serialization would add the surrounding quotes and
        // escape the inner ones, over-counting an `arguments` payload.
        assert_eq!(json_text_bytes(&serde_json::json!("{\"a\":1}")), 7);
        assert_eq!(json_text_bytes(&serde_json::json!({"a": 1})), 7);
    }

    /// **The contract's canonical pin.** One logical request — a user post,
    /// an assistant tool call, a tool result, and one advertised tool —
    /// walked by the real [`prompt_charge`] and pinned to exact numbers.
    ///
    /// Each consumer keeps one agreement test that feeds its *own* request
    /// representation through the same walk and asserts the same 230
    /// (`eidola-server`'s `tool_round_fixture_charges_the_pinned_contract_value`
    /// over a parsed `ChatCompletionRequest`, `eidola-app-core`'s
    /// `prompt_charge_matches_the_shared_contract_fixture` over the `Value`
    /// messages a turn actually sends). Those still have teeth after the
    /// consolidation: they catch a consumer feeding the shared walk a
    /// malformed *view* of its request, which no in-crate test here can see.
    #[test]
    fn cross_crate_tool_round_fixture() {
        let (messages, tools) = tool_round_fixture();
        let charge = prompt_charge(&messages, Some(&tools));

        // The two composite measures, pinned individually so a change in
        // either is named rather than showing up only in the total.
        assert_eq!(
            json_text_bytes(&messages[1]["tool_calls"][0]),
            TOOL_CALL_BYTES,
            "the fixture's call entry must stay {TOOL_CALL_BYTES} bytes"
        );
        assert_eq!(
            json_text_bytes(&tools[0]),
            TOOL_DEFINITION_BYTES,
            "the fixture's schema must stay {TOOL_DEFINITION_BYTES} bytes"
        );

        assert_eq!(charge.message_count(), 3);
        // 12 ("what is 2+2?") + 0 (null) + 1 ("4") content bytes.
        assert_eq!(
            charge.total_content_bytes(),
            13 + TOOL_CALL_BYTES + TOOL_DEFINITION_BYTES
        );
        // ceil((13 + 93 + 154) * 2 / 3) + 8*3 + 32 = 174 + 24 + 32 = 230
        assert_eq!(charge.chargeable_prompt_tokens(), 230);
    }

    /// Compact JSON byte length of the fixture's single tool-call entry.
    const TOOL_CALL_BYTES: u64 = 93;

    /// Compact JSON byte length of the fixture's single tool definition.
    const TOOL_DEFINITION_BYTES: u64 = 154;

    /// The canonical fixture request, as `(messages, tools)`. The same
    /// logical request each consumer reconstructs in its own representation.
    fn tool_round_fixture() -> (Vec<Value>, Vec<Value>) {
        let messages = vec![
            serde_json::json!({"role": "user", "content": "what is 2+2?"}),
            serde_json::json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": "{\"expr\":\"2+2\"}"}
                }]
            }),
            serde_json::json!({"role": "tool", "tool_call_id": "call_1", "content": "4"}),
        ];
        let tools = vec![serde_json::json!({
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Evaluate arithmetic.",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            }
        })];
        (messages, tools)
    }

    #[test]
    fn overflow_safe_at_extreme_inputs() {
        // No panic, and saturation only ever raises the value.
        let v = chargeable_prompt_tokens(u64::MAX, u64::MAX);
        assert_eq!(v, u64::MAX);
        // Large-but-realistic inputs stay exact.
        let bytes = 1u64 << 40; // 1 TiB of content
        let v = chargeable_prompt_tokens(bytes, 1 << 20);
        assert_eq!(
            v as u128,
            (bytes as u128 * 2).div_ceil(3)
                + PER_MESSAGE_TOKENS as u128 * (1u128 << 20)
                + PER_REQUEST_TOKENS as u128
        );
    }
}
