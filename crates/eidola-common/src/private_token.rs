//! The `Authorization` value that carries an ACT spend: the `PrivateToken`
//! scheme of RFC 9577 (Section 2.2.2) with one `token` parameter.
//!
//! ```text
//! Authorization: PrivateToken token="<base64url(Token)>"
//! ```
//!
//! RFC 9577 makes the `token` value the base64url encoding (RFC 4648,
//! Section 5) of the Token *with* padding. The client builds the value with
//! [`authorization_value`] and the gateway reads it with [`token_bytes`], so
//! the two sides cannot disagree about the spelling, and there is exactly one:
//! the value is padded, and anything that does not re-encode to itself is
//! refused (missing or extra padding, non-zero pad bits, whitespace, the
//! standard alphabet's `+` and `/`). The scheme name and the parameter are
//! spelled exactly as [`authorization_value`] writes them, too.
//!
//! The Token's own layout (`token_type`, `challenge_digest`,
//! `issuer_key_id`, the spend proof) is the business of each side; this rule
//! carries its bytes.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE;

/// What precedes the encoded Token in the header value.
const PREFIX: &str = "PrivateToken token=\"";

/// What follows it.
const SUFFIX: &str = "\"";

/// The `Authorization` header value presenting `token` (the Token's bytes):
/// `PrivateToken token="…"`, the bytes as padded base64url.
pub fn authorization_value(token: &[u8]) -> String {
    format!("{PREFIX}{}{SUFFIX}", URL_SAFE.encode(token))
}

/// Why an `Authorization` value is not a Token presented as
/// [`authorization_value`] spells one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenAuthorizationError {
    /// Not `PrivateToken token="…"`, spelled exactly so.
    Scheme,
    /// The `token` value is not canonical padded base64url.
    Encoding,
}

/// The Token's bytes from an `Authorization` header value, or why the value
/// is not one [`authorization_value`] could have written.
pub fn token_bytes(value: &str) -> Result<Vec<u8>, TokenAuthorizationError> {
    let encoded = value
        .strip_prefix(PREFIX)
        .and_then(|rest| rest.strip_suffix(SUFFIX))
        .ok_or(TokenAuthorizationError::Scheme)?;
    // `URL_SAFE` requires canonical padding and refuses non-zero pad bits,
    // which is what makes one spelling the only one; the round trip below
    // states that rule rather than trusting the engine's configuration.
    let bytes = URL_SAFE
        .decode(encoded)
        .map_err(|_| TokenAuthorizationError::Encoding)?;
    if URL_SAFE.encode(&bytes) != encoded {
        return Err(TokenAuthorizationError::Encoding);
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_value_is_padded_base64url_and_round_trips() {
        // Every padding case: none, `=` and `==`.
        for len in [0usize, 1, 2, 3, 4, 5, 66, 67, 68, 18_137] {
            let token: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            let value = authorization_value(&token);
            let encoded = &value[PREFIX.len()..value.len() - SUFFIX.len()];
            assert_eq!(encoded.len() % 4, 0, "padded to a multiple of four: {len}");
            assert_eq!(token_bytes(&value), Ok(token), "round trip: {len}");
        }
    }

    /// A fixed vector, computed independently: Python's
    /// `base64.urlsafe_b64encode(bytes([0xfb, 0xff, 0xbf, 0xe5, 0xad]))`.
    #[test]
    fn a_known_token_has_its_rfc_spelling() {
        let token = [0xfb, 0xff, 0xbf, 0xe5, 0xad];
        assert_eq!(
            authorization_value(&token),
            "PrivateToken token=\"-_-_5a0=\""
        );
        assert_eq!(
            token_bytes("PrivateToken token=\"-_-_5a0=\""),
            Ok(token.to_vec())
        );
    }

    #[test]
    fn any_other_spelling_is_refused() {
        use TokenAuthorizationError::{Encoding, Scheme};
        let refused = [
            // Padding missing, short, or extra.
            ("PrivateToken token=\"-_-_5a0\"", Encoding),
            ("PrivateToken token=\"-_-_5a0==\"", Encoding),
            ("PrivateToken token=\"AA=\"", Encoding),
            ("PrivateToken token=\"AAAA=\"", Encoding),
            // Non-zero pad bits: `5a1=` decodes to the same bytes as `5a0=`.
            ("PrivateToken token=\"-_-_5a1=\"", Encoding),
            // The standard alphabet, and whitespace.
            ("PrivateToken token=\"+/+/5a0=\"", Encoding),
            ("PrivateToken token=\"-_-_ 5a0=\"", Encoding),
            ("PrivateToken token=\"-_-_5a0=\n\"", Encoding),
            // The scheme and parameter, spelled any other way.
            ("PrivateToken token=-_-_5a0=", Scheme),
            ("privatetoken token=\"-_-_5a0=\"", Scheme),
            ("PrivateToken  token=\"-_-_5a0=\"", Scheme),
            ("PrivateToken tokens=\"-_-_5a0=\"", Scheme),
            ("PrivateToken token=\"-_-_5a0=\" ", Scheme),
            ("Bearer -_-_5a0=", Scheme),
        ];
        for (value, why) in refused {
            assert_eq!(token_bytes(value), Err(why), "{value:?}");
        }
    }
}
