//! The refund recovery key: a secret only the client holds, sent with each
//! spend so the gateway can seal the refund it stores for that spend.
//!
//! A spend's refund carries the amount returned, which is the hold minus the
//! actual charge: a function of the request's content lengths. The gateway
//! stores each refund so a client whose response was lost can fetch it again,
//! and its database is outside the enclave and assumed adversarial. So the
//! stored copy is sealed under a key that needs this client secret: the
//! gateway uses the presented value only in enclave memory, never stores or
//! logs it, and the recovery endpoint opens the stored copy only when the
//! client presents the same value again.
//!
//! The wire form is fixed so the client and the gateway cannot disagree about
//! it: [`REFUND_KEY_BYTES`] bytes as exactly [`REFUND_KEY_TEXT_LEN`] lowercase
//! hex digits in the [`REFUND_KEY_HEADER`] request header. How the client
//! chooses the bytes is the client's business; the gateway only needs them to
//! be the same at spend and at recovery.

/// The request header carrying the refund recovery key, on a spend
/// (`POST /v1/chat/completions`) and on recovery (`POST /v1/credentials/refund`).
/// Header names are case-insensitive; this is the lowercase spelling both
/// sides use.
pub const REFUND_KEY_HEADER: &str = "x-eidola-refund-key";

/// Length of a refund recovery key, in bytes.
pub const REFUND_KEY_BYTES: usize = 32;

/// Length of a refund recovery key's wire form: lowercase hex.
pub const REFUND_KEY_TEXT_LEN: usize = REFUND_KEY_BYTES * 2;

const HEX: &[u8; 16] = b"0123456789abcdef";

/// The wire form of `key`: [`REFUND_KEY_TEXT_LEN`] lowercase hex digits.
pub fn encode_refund_key(key: &[u8; REFUND_KEY_BYTES]) -> [u8; REFUND_KEY_TEXT_LEN] {
    let mut out = [0u8; REFUND_KEY_TEXT_LEN];
    for (i, byte) in key.iter().enumerate() {
        out[2 * i] = HEX[usize::from(byte >> 4)];
        out[2 * i + 1] = HEX[usize::from(byte & 0x0f)];
    }
    out
}

/// A refund recovery key from its wire form, or `None` for anything that is
/// not exactly [`REFUND_KEY_TEXT_LEN`] lowercase hex digits. There is one
/// spelling of each key, so a value that parses re-encodes to itself.
pub fn decode_refund_key(text: &[u8]) -> Option<[u8; REFUND_KEY_BYTES]> {
    fn digit(c: u8) -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            _ => None,
        }
    }
    if text.len() != REFUND_KEY_TEXT_LEN {
        return None;
    }
    let mut key = [0u8; REFUND_KEY_BYTES];
    let (pairs, _) = text.as_chunks::<2>();
    for (byte, [hi, lo]) in key.iter_mut().zip(pairs) {
        *byte = (digit(*hi)? << 4) | digit(*lo)?;
    }
    Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refund_key_round_trips_through_its_one_spelling() {
        let key: [u8; REFUND_KEY_BYTES] = std::array::from_fn(|i| (i as u8).wrapping_mul(37));
        let text = encode_refund_key(&key);
        assert!(
            text.iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
        );
        assert_eq!(decode_refund_key(&text), Some(key));
    }

    #[test]
    fn anything_but_exactly_the_lowercase_hex_form_is_refused() {
        let text = encode_refund_key(&[0xab; REFUND_KEY_BYTES]);
        assert_eq!(decode_refund_key(&text[..REFUND_KEY_TEXT_LEN - 1]), None);
        let mut long = text.to_vec();
        long.push(b'0');
        assert_eq!(decode_refund_key(&long), None);
        let upper = text.to_ascii_uppercase();
        assert_eq!(decode_refund_key(&upper), None);
        let mut bad = text;
        bad[7] = b'g';
        assert_eq!(decode_refund_key(&bad), None);
        assert_eq!(decode_refund_key(b""), None);
    }
}
