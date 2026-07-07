//! Shared client/server contract logic for eidola.
//!
//! This crate holds the small pieces of pure logic that MUST be bit-identical
//! on both sides of the wire — currently the anonymous-credit pricing
//! contract for chat-completion prompt holds. It is intentionally lib-only,
//! zero-dependency, pure Rust, and float-free so both `eidola-app-core` and
//! `eidola-server` can depend on it and compute identical results on any
//! platform.
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
//! - `total_content_bytes`: the sum of UTF-8 byte lengths of each message's
//!   `content` string in the OpenAI `messages` array.
//! - `message_count`: the number of entries in that array.
//!
//! Roles and JSON structure are deliberately excluded from the byte count;
//! their token cost is covered by [`PER_MESSAGE_TOKENS`] and
//! [`PER_REQUEST_TOKENS`].

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
