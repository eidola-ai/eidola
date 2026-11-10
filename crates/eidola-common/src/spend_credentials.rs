//! How a request presents its anonymous credits: the `credentials` member of
//! the JSON request body, and the `refunds` member of the answer.
//!
//! ```text
//! "credentials": [
//!   { "token": "<base64url(Token), padded>", "refund_key": "<64 lowercase hex digits>" }
//! ]
//! ```
//!
//! Each element is one *input*: one ACT Token exactly as
//! `draft-schlesinger-privacypass-act` defines it (`token_type`,
//! `challenge_digest`, `issuer_key_id`, the encoded spend proof) and that
//! spend's refund recovery key ([`crate::refund_key`]). The member is always an
//! array, whatever the count, so every count has one encoding: a request
//! carries one input, or at most [`MAX_CREDENTIALS_PER_REQUEST`].
//!
//! A credential travels in the body and nowhere else. A Token at a practical
//! `L` is larger than common header limits, and a body has no such limit along
//! the path; the gateway refuses a credential in a header, so there is one
//! transport and nothing to reconcile between two.
//!
//! The client writes the member with [`credentials_member`] and the gateway
//! reads it through [`CredentialWire`] and [`decode_credentials`], so the two
//! sides cannot disagree about the spelling, and there is exactly one: the
//! Token is base64url (RFC 4648, Section 5) *with* padding, and anything that
//! does not re-encode to itself is refused (missing or extra padding, non-zero
//! pad bits, whitespace, the standard alphabet's `+` and `/`); the refund key
//! is exactly the lowercase hex [`crate::refund_key`] writes. Each element has
//! exactly the members `token` and `refund_key`: an unknown, missing or
//! repeated member is refused by the deserializer.
//!
//! The answer carries one refund per input, in input order, as the
//! [`REFUNDS_MEMBER`] array of `{ "refund", "issuer_key_id" }` objects, in a
//! blocking response, an error body, the streamed final metadata event, and a
//! recovery answer alike.
//!
//! A request with several inputs is charged across them in input order
//! ([`partial_returns`]): the charge is consumed from the first input before
//! the second, and each input's change returns under its own issuer key, so
//! no value moves from one input to another.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE;

use crate::refund_key::{REFUND_KEY_BYTES, decode_refund_key, encode_refund_key};

/// The request body member carrying the inputs.
pub const CREDENTIALS_MEMBER: &str = "credentials";

/// The answer member carrying one refund per input.
pub const REFUNDS_MEMBER: &str = "refunds";

/// The most inputs one request carries: one drain and one fill. A request
/// with more is refused before any spend proof is decoded.
///
/// Every input beyond the first narrows the request's anonymity set to the
/// accounts issued at least that many credentials under each presented key,
/// which is why the client provisions two credentials per epoch and why this
/// is the smallest number that lets a residual be drained.
pub const MAX_CREDENTIALS_PER_REQUEST: usize = 2;

/// The ACT parameter `L`: a credential's balance lies in `[0, 2^L)` credits.
///
/// The ACT specification leaves `L` to the deployment and does not signal it
/// in keys, Tokens or challenges, so the client and the gateway fix it here,
/// at build time. At one credit per millionth of a US dollar, `2^32` credits
/// is about $4,294, far above any one credential's value. A spend proof's
/// size and its proving and verifying cost are linear in `L`; it bounds only
/// the largest representable balance and changes nothing a proof hides.
pub const ACT_L: usize = 32;

/// The wire form of one Token: padded base64url.
pub fn encode_token(token: &[u8]) -> String {
    URL_SAFE.encode(token)
}

/// A Token's bytes from its wire form, or `None` for anything
/// [`encode_token`] could not have written.
pub fn decode_token(text: &str) -> Option<Vec<u8>> {
    // `URL_SAFE` requires canonical padding and refuses non-zero pad bits,
    // which is what makes one spelling the only one; the round trip below
    // states that rule rather than trusting the engine's configuration.
    let bytes = URL_SAFE.decode(text).ok()?;
    (URL_SAFE.encode(&bytes) == text).then_some(bytes)
}

/// The `credentials` member's value for `inputs`, each a Token's bytes and its
/// refund recovery key, in input order.
///
/// The value holds the refund keys as text; it is the caller's to keep out of
/// anything durable (a request record, a log), as it kept the header values
/// before.
pub fn credentials_member(inputs: &[(&[u8], &[u8; REFUND_KEY_BYTES])]) -> serde_json::Value {
    serde_json::Value::Array(
        inputs
            .iter()
            .map(|(token, key)| {
                let key = encode_refund_key(key);
                serde_json::json!({
                    "token": encode_token(token),
                    "refund_key": std::str::from_utf8(&key).expect("lowercase hex is UTF-8"),
                })
            })
            .collect(),
    )
}

/// One element of the `credentials` member, as it arrived.
///
/// Strict: the derived deserializer refuses an unknown member
/// (`deny_unknown_fields`), a missing one, and a repeated one. Its values are
/// checked by [`decode_credentials`]. `Debug` prints neither value, and the
/// refund key's text is zeroed on drop.
#[derive(Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CredentialWire {
    token: String,
    refund_key: String,
}

impl CredentialWire {
    /// An element carrying `token` and `refund_key` exactly as given, for
    /// tests that need a malformed one.
    pub fn from_parts(token: String, refund_key: String) -> Self {
        Self { token, refund_key }
    }
}

impl std::fmt::Debug for CredentialWire {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CredentialWire(<redacted>)")
    }
}

impl Drop for CredentialWire {
    fn drop(&mut self) {
        // The refund key is a secret only the client holds. Its bytes are
        // overwritten in place (the capacity is kept, so nothing moves); what
        // the JSON parser buffered on the way in is out of reach here.
        let len = self.refund_key.len();
        self.refund_key.clear();
        self.refund_key.extend(std::iter::repeat_n('\0', len));
    }
}

/// One decoded input: a Token's bytes and its refund recovery key.
pub struct DecodedCredential {
    pub token: Vec<u8>,
    pub refund_key: [u8; REFUND_KEY_BYTES],
}

impl std::fmt::Debug for DecodedCredential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DecodedCredential(<redacted>)")
    }
}

impl Drop for DecodedCredential {
    fn drop(&mut self) {
        self.refund_key.fill(0);
    }
}

/// Why a `credentials` member is not one [`credentials_member`] could have
/// written. None of these quotes a presented value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialsError {
    /// No input at all: the member is absent or an empty array.
    Missing,
    /// More inputs than [`MAX_CREDENTIALS_PER_REQUEST`].
    TooMany,
    /// A `token` that is not canonical padded base64url.
    TokenEncoding,
    /// A `refund_key` that is not exactly 64 lowercase hex digits.
    RefundKeyEncoding,
}

impl std::fmt::Display for CredentialsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Missing => "a `credentials` array with at least one spend is required",
            Self::TooMany => "`credentials` carries more spends than a request may",
            Self::TokenEncoding => "a credential's `token` must be padded base64url",
            Self::RefundKeyEncoding => {
                "a credential's `refund_key` must be 64 lowercase hex digits"
            }
        })
    }
}

/// The hold of a request whose inputs present `shares`: their sum, or `None`
/// if it overflows (a malformed request).
pub fn hold(shares: &[u128]) -> Option<u128> {
    shares
        .iter()
        .try_fold(0u128, |sum, share| sum.checked_add(*share))
}

/// Each input's partial return `t` when a request whose inputs present
/// `shares`, in input order, is charged `charge`: the charge is consumed from
/// the first input, then the next, and the change lands on the inputs after
/// that.
///
/// ```text
/// remaining = charge
/// for each input i, in order:
///     a_i = min(s_i, remaining); remaining -= a_i; t_i = s_i - a_i
/// ```
///
/// So `Σ t = hold − charge`, `0 ≤ t_i ≤ s_i`, and for some index every input
/// before it is fully consumed and every input after it fully returned. The
/// result is a function of the shares, their order and the charge alone, so a
/// client can predict and check it, and the issuer has no choice of `t` with
/// which to mark a successor. `None` when the charge exceeds the hold or the
/// hold overflows: the caller clamps the charge to the hold first.
pub fn partial_returns(shares: &[u128], charge: u128) -> Option<Vec<u128>> {
    if charge > hold(shares)? {
        return None;
    }
    let mut remaining = charge;
    Some(
        shares
            .iter()
            .map(|share| {
                let consumed = (*share).min(remaining);
                remaining -= consumed;
                share - consumed
            })
            .collect(),
    )
}

/// The inputs of a `credentials` member as it was deserialized, in input
/// order, or why it is refused. The count is checked before any value is
/// decoded.
pub fn decode_credentials(
    wires: &[CredentialWire],
) -> Result<Vec<DecodedCredential>, CredentialsError> {
    if wires.is_empty() {
        return Err(CredentialsError::Missing);
    }
    if wires.len() > MAX_CREDENTIALS_PER_REQUEST {
        return Err(CredentialsError::TooMany);
    }
    wires
        .iter()
        .map(|wire| {
            let token = decode_token(&wire.token).ok_or(CredentialsError::TokenEncoding)?;
            let refund_key = decode_refund_key(wire.refund_key.as_bytes())
                .ok_or(CredentialsError::RefundKeyEncoding)?;
            Ok(DecodedCredential { token, refund_key })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    struct Body {
        #[serde(default)]
        credentials: Vec<CredentialWire>,
    }

    fn parse(json: &str) -> Result<Vec<DecodedCredential>, String> {
        let body: Body = serde_json::from_str(json).map_err(|e| e.to_string())?;
        decode_credentials(&body.credentials).map_err(|e| format!("{e:?}"))
    }

    const KEY: [u8; REFUND_KEY_BYTES] = [0xab; REFUND_KEY_BYTES];

    fn hex_key() -> String {
        String::from_utf8(encode_refund_key(&KEY).to_vec()).unwrap()
    }

    #[test]
    fn the_member_round_trips_for_every_padding_case() {
        for len in [1usize, 2, 3, 4, 5, 66, 67, 68, 4_985] {
            let token: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let member = credentials_member(&[(&token, &KEY)]);
            let encoded = member[0]["token"].as_str().unwrap();
            assert_eq!(encoded.len() % 4, 0, "padded to a multiple of four: {len}");
            let json = serde_json::json!({ "credentials": member }).to_string();
            let decoded = parse(&json).unwrap();
            assert_eq!(decoded.len(), 1);
            assert_eq!(decoded[0].token, token, "round trip: {len}");
            assert_eq!(decoded[0].refund_key, KEY);
        }
    }

    /// A fixed vector, computed independently: Python's
    /// `base64.urlsafe_b64encode(bytes([0xfb, 0xff, 0xbf, 0xe5, 0xad]))`.
    #[test]
    fn a_known_token_has_its_rfc_spelling() {
        let token = [0xfb, 0xff, 0xbf, 0xe5, 0xad];
        assert_eq!(encode_token(&token), "-_-_5a0=");
        assert_eq!(decode_token("-_-_5a0="), Some(token.to_vec()));
    }

    #[test]
    fn any_other_token_spelling_is_refused() {
        for text in [
            // Padding missing, short, or extra.
            "-_-_5a0",
            "-_-_5a0==",
            "AA=",
            "AAAA=",
            // Non-zero pad bits: `5a1=` decodes to the same bytes as `5a0=`.
            "-_-_5a1=",
            // The standard alphabet, and whitespace.
            "+/+/5a0=",
            "-_-_ 5a0=",
            "-_-_5a0=\n",
            " -_-_5a0=",
        ] {
            assert_eq!(decode_token(text), None, "{text:?}");
            let json = format!(
                r#"{{"credentials":[{{"token":"{}","refund_key":"{}"}}]}}"#,
                text.replace('\n', "\\n"),
                hex_key()
            );
            assert_eq!(parse(&json).unwrap_err(), "TokenEncoding", "{text:?}");
        }
    }

    #[test]
    fn a_refund_key_in_any_other_spelling_is_refused() {
        let upper = hex_key().to_ascii_uppercase();
        let short = &hex_key()[..63];
        let long = format!("{}0", hex_key());
        let spaced = format!(" {}", &hex_key()[1..]);
        for key in [upper.as_str(), short, long.as_str(), spaced.as_str(), ""] {
            let json =
                format!(r#"{{"credentials":[{{"token":"-_-_5a0=","refund_key":"{key}"}}]}}"#);
            assert_eq!(parse(&json).unwrap_err(), "RefundKeyEncoding", "{key:?}");
        }
    }

    #[test]
    fn the_count_is_checked_first() {
        assert_eq!(parse("{}").unwrap_err(), "Missing");
        assert_eq!(parse(r#"{"credentials":[]}"#).unwrap_err(), "Missing");
        // Three elements, the last malformed: refused for the count, before
        // any value is decoded.
        let json = format!(
            r#"{{"credentials":[{{"token":"-_-_5a0=","refund_key":"{k}"}},{{"token":"-_-_5a0=","refund_key":"{k}"}},{{"token":"bad","refund_key":"{k}"}}]}}"#,
            k = hex_key()
        );
        assert_eq!(parse(&json).unwrap_err(), "TooMany");
    }

    #[test]
    fn two_inputs_are_read_in_input_order() {
        let first = [1u8, 2, 3];
        let second = [4u8, 5, 6, 7];
        let other_key = [0xcd; REFUND_KEY_BYTES];
        let member = credentials_member(&[(&first, &KEY), (&second, &other_key)]);
        let json = serde_json::json!({ "credentials": member }).to_string();
        let decoded = parse(&json).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].token, first);
        assert_eq!(decoded[0].refund_key, KEY);
        assert_eq!(decoded[1].token, second);
        assert_eq!(decoded[1].refund_key, other_key);
    }

    /// The allocation vectors: one input, a charge of nothing, of
    /// everything, at the prefix boundary and one either side of it, and the
    /// worked example (a drain of 3,000 and a fill of 3,200).
    #[test]
    fn the_charge_is_consumed_in_input_order() {
        let cases: &[(&[u128], u128, &[u128])] = &[
            (&[6_200], 0, &[6_200]),
            (&[6_200], 4_000, &[2_200]),
            (&[6_200], 6_200, &[0]),
            (&[3_000, 3_200], 0, &[3_000, 3_200]),
            (&[3_000, 3_200], 500, &[2_500, 3_200]),
            (&[3_000, 3_200], 2_999, &[1, 3_200]),
            (&[3_000, 3_200], 3_000, &[0, 3_200]),
            (&[3_000, 3_200], 3_001, &[0, 3_199]),
            (&[3_000, 3_200], 4_000, &[0, 2_200]),
            (&[3_000, 3_200], 6_199, &[0, 1]),
            (&[3_000, 3_200], 6_200, &[0, 0]),
        ];
        for (shares, charge, expected) in cases {
            assert_eq!(
                partial_returns(shares, *charge).as_deref(),
                Some(*expected),
                "{shares:?} charged {charge}"
            );
        }
    }

    #[test]
    fn a_charge_beyond_the_hold_or_an_overflowing_hold_has_no_allocation() {
        assert_eq!(partial_returns(&[3_000, 3_200], 6_201), None);
        assert_eq!(hold(&[u128::MAX, 1]), None);
        assert_eq!(partial_returns(&[u128::MAX, 1], 0), None);
        assert_eq!(hold(&[1u128 << 63, 1u128 << 63]), Some(1u128 << 64));
    }

    /// The allocation's properties over every charge for a few share lists:
    /// exact, within each share, and a prefix of consumed inputs.
    #[test]
    fn the_allocation_is_exact_bounded_and_a_prefix() {
        for shares in [&[7u128][..], &[3, 5], &[1, 1], &[9, 2]] {
            let hold = hold(shares).unwrap();
            for charge in 0..=hold {
                let returns = partial_returns(shares, charge).unwrap();
                assert_eq!(returns.iter().sum::<u128>(), hold - charge);
                assert!(returns.iter().zip(shares).all(|(t, s)| t <= s));
                // Once an input is not fully consumed, every later one is
                // fully returned.
                let first_partial = returns.iter().position(|t| *t > 0);
                if let Some(j) = first_partial {
                    assert!(returns[..j].iter().all(|t| *t == 0));
                    assert!(
                        returns[j + 1..]
                            .iter()
                            .zip(&shares[j + 1..])
                            .all(|(t, s)| t == s)
                    );
                }
            }
        }
    }

    #[test]
    fn the_shape_is_strict() {
        let k = hex_key();
        for json in [
            // Not an array.
            format!(r#"{{"credentials":{{"token":"-_-_5a0=","refund_key":"{k}"}}}}"#),
            r#"{"credentials":null}"#.to_string(),
            r#"{"credentials":"-_-_5a0="}"#.to_string(),
            // A member missing, unknown, or repeated.
            r#"{"credentials":[{"token":"-_-_5a0="}]}"#.to_string(),
            format!(r#"{{"credentials":[{{"refund_key":"{k}"}}]}}"#),
            format!(
                r#"{{"credentials":[{{"token":"-_-_5a0=","refund_key":"{k}","issuer":"x"}}]}}"#
            ),
            format!(
                r#"{{"credentials":[{{"token":"-_-_5a0=","token":"-_-_5a0=","refund_key":"{k}"}}]}}"#
            ),
            // The member itself repeated at the top level.
            format!(
                r#"{{"credentials":[{{"token":"-_-_5a0=","refund_key":"{k}"}}],"credentials":[]}}"#
            ),
            // Not strings.
            format!(r#"{{"credentials":[{{"token":5,"refund_key":"{k}"}}]}}"#),
        ] {
            assert!(parse(&json).is_err(), "{json}");
        }
    }

    #[test]
    fn nothing_prints_a_credential() {
        let wire = CredentialWire::from_parts("-_-_5a0=".into(), hex_key());
        assert!(!format!("{wire:?}").contains("5a0"));
        let decoded = decode_credentials(std::slice::from_ref(&wire)).unwrap();
        assert!(!format!("{decoded:?}").contains("ab"));
        assert!(!CredentialsError::TokenEncoding.to_string().contains("5a0"));
    }
}
