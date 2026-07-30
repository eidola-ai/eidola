//! Shared client/server contract logic for eidola.
//!
//! This crate holds the small pieces of pure logic that MUST be bit-identical
//! on both sides of a contract boundary — the anonymous-credit pricing
//! contract for chat-completion prompt holds ([`chargeable_prompt_tokens`]),
//! and the embed-marker recognition rule shared between the markdown
//! editor's embed plugin and app-core's upstream quote expansion
//! ([`embed`]). It is intentionally lib-only, zero-dependency, pure Rust,
//! and float-free so `eidola-app-core`, `eidola-server`, and tests can
//! depend on it and compute identical results on any platform.
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
//! # Gathering the inputs: [`PromptCharge`]
//!
//! *What* counts is as much a part of the contract as the arithmetic, and it
//! is where the two sides can silently drift: the client holds
//! `serde_json::Value` messages, the server holds parsed
//! `ChatCompletionRequest` structs, and this crate is zero-dependency (it
//! cannot see either shape). [`PromptCharge`] is therefore the shared
//! *accounting* surface: each side walks its own request shape and reports
//! the parts through one API whose semantics — which counter each
//! contribution lands in — live here, in one place. Each side's ~10-line walk
//! is held to the other by a tripwire test
//! (`eidola-server`'s `client_and_server_prompt_terms_agree`).

pub mod embed;

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
/// | an assistant `tool_calls` entry | [`add_tool_call`](Self::add_tool_call) | the function name's bytes + the raw `arguments` string's bytes |
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
/// * an `arguments` value that is not a string (off-spec, but seen in the
///   wild) is measured the same way, as its compact serialization.
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

    /// Account for one `tool_calls` entry on an assistant message: the
    /// function name and the raw `arguments` payload.
    ///
    /// The enclosing message is counted separately by
    /// [`add_message`](Self::add_message) — this only adds bytes.
    pub fn add_tool_call(&mut self, name_bytes: u64, arguments_bytes: u64) -> &mut Self {
        self.total_content_bytes = self
            .total_content_bytes
            .saturating_add(name_bytes)
            .saturating_add(arguments_bytes);
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
        with_call.add_message(0).add_tool_call(4, 14);

        let mut without = PromptCharge::new();
        without.add_message(0);

        assert_eq!(with_call.total_content_bytes(), 18);
        assert_eq!(with_call.message_count(), without.message_count());
        assert!(
            with_call.chargeable_prompt_tokens() > without.chargeable_prompt_tokens(),
            "tool-call bytes must not ride free"
        );
        // The bytes land in the byte term at the safe cost factor, like any
        // other bytes: ceil(18*2/3) = 12 tokens more than the empty message.
        assert_eq!(
            with_call.chargeable_prompt_tokens() - without.chargeable_prompt_tokens(),
            12
        );
    }

    #[test]
    fn several_tool_calls_on_one_message_count_once_as_a_message() {
        let mut charge = PromptCharge::new();
        charge
            .add_message(0)
            .add_tool_call(3, 10)
            .add_tool_call(3, 10);
        assert_eq!(charge.message_count(), 1, "one message, two calls");
        assert_eq!(charge.total_content_bytes(), 26);
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
            .add_tool_call(u64::MAX, u64::MAX)
            .add_tool_definition(u64::MAX);
        assert_eq!(charge.total_content_bytes(), u64::MAX);
        assert_eq!(
            charge.chargeable_prompt_tokens(),
            chargeable_prompt_tokens(u64::MAX, 1)
        );
    }

    /// The cross-crate fixture. `eidola-server`'s
    /// `client_and_server_prompt_terms_agree` and `eidola-app-core`'s
    /// `prompt_charge_matches_the_shared_contract_fixture` both walk the same
    /// logical request — a user post, an assistant tool call, a tool result,
    /// and one advertised tool — and must arrive at exactly these numbers.
    /// Change them here and both sides fail until they agree again.
    #[test]
    fn cross_crate_tool_round_fixture() {
        let mut charge = PromptCharge::new();
        // {"role":"user","content":"what is 2+2?"}
        charge.add_message(12);
        // {"role":"assistant","content":null,"tool_calls":[…name "calc",
        //  arguments "{\"expr\":\"2+2\"}"…]}
        charge.add_message(0).add_tool_call(4, 14);
        // {"role":"tool","tool_call_id":"call_1","content":"4"}
        charge.add_message(1);
        // The advertised `calc` schema, compact-serialized (see the
        // consuming tests for the literal JSON).
        charge.add_tool_definition(TOOL_DEFINITION_BYTES);

        assert_eq!(charge.message_count(), 3);
        assert_eq!(charge.total_content_bytes(), 31 + TOOL_DEFINITION_BYTES);
        // ceil((31 + 154) * 2 / 3) + 8*3 + 32 = 124 + 24 + 32 = 180
        assert_eq!(charge.chargeable_prompt_tokens(), 180);
    }

    /// Compact JSON byte length of the fixture's single tool definition.
    const TOOL_DEFINITION_BYTES: u64 = 154;

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
