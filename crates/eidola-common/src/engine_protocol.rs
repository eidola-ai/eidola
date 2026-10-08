//! The wire contract between Eidola's clients, its gateway, and its inference
//! nodes (`eidola-server-engine`) that every side must spell identically.
//!
//! Two things live here:
//!
//! - **The prefix-cache key** a client may send as the request body's
//!   `cache_key` member: [`CACHE_KEY_BYTES`] bytes, encoded as unpadded
//!   base64url ([`CACHE_KEY_TEXT_LEN`] characters). [`is_cache_key_text`] is
//!   the shape rule: the gateway refuses anything else before a request is
//!   paid for, and the node's decoder accepts exactly what it accepts.
//! - **The weights header** ([`WEIGHTS_HEADER`]) the gateway sends on every
//!   chat request to a node, carrying the lowercase-hex weights hash it
//!   expects that node to serve. The node refuses a request without it, or
//!   with a different hash, before reading the body.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
