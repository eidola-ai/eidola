//! The cipher idle prefix-cache KV is held under: ChaCha20 (RFC 8439: the 20-round block
//! function, a 32-bit block counter and a 96-bit nonce), as a bare stream cipher.
//!
//! This module is the normative definition every executor reproduces byte for byte: the
//! mock and the CPU reference executor call it, and the CUDA kernel (`kv_cipher.cu` in
//! `eidola-engine-kernels`) is a transliteration checked against it.
//!
//! # Why ChaCha20
//!
//! It is additions, rotations and XORs on 32-bit words: no tables and no secret-dependent
//! memory access or branch, so it is constant-time on a GPU as on a CPU. AES without AES
//! instructions (which GPUs do not have) is a table cipher whose lookups leak through the
//! cache and shared-memory banks.
//!
//! # Why no authentication tag
//!
//! Integrity is out of scope. The threat is a memory *reader* between requests: what it
//! must not learn is the KV of entries nobody is using. An attacker who can also *write*
//! device memory can already rewrite whatever a request in flight computes, plaintext and
//! all, so a tag over idle blocks would close nothing. A wrong key cannot reach a decrypt
//! either: the hash chain that finds an entry and the key that decrypts it come from the
//! same client secret, so a lookup that matches is a lookup under the right key.
//!
//! # A block's image and keystream
//!
//! An executor stores a KV block as one or more contiguous ranges (every layer's K and V,
//! boundary taps, and for the CPU reference its row tags), in an order it documents. Each
//! range is read as little-endian 32-bit words; range `r` is XORed with the keystream from
//! block counter `c_r` on, where `c_0 = 0` and `c_{r+1} = c_r + ceil(words_r / 16)`: every
//! range starts at a fresh 64-byte keystream block, so no keystream word is used twice
//! within a block. [`KvKeystream::range`] advances the counter that way.
//!
//! # Nonce: one per sealed block
//!
//! The nonce is derived from the block's salted block hash and its group
//! ([`crate::hash::BlockHash::kv_nonce`]), not drawn at random. A cached block is sealed:
//! by the executor's determinism contract its contents are a function of the token prefix
//! the hash names, so every time it is encrypted (after each idle period, or recomputed
//! after an eviction) the same nonce meets the same plaintext and yields the same
//! ciphertext. A nonce therefore never covers two different plaintexts under one key, and
//! the host stores nothing per encrypted block. Distinct blocks of one lineage (one key)
//! differ in hash or group; a collision of two of their 96-bit nonces needs about 2^48
//! blocks in one lineage. Different lineages have independent keys.

use zeroize::Zeroize;

use crate::secret::KvKey;

/// "expand 32-byte k", RFC 8439 §2.3.
const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// Words in one ChaCha20 block.
pub const BLOCK_WORDS: usize = 16;

/// The nonce a block is encrypted under: three little-endian words of RFC 8439's 96-bit
/// nonce. A content-free linkage tag like the block hash it is derived from, so `Debug` is
/// redacted; executors read it with [`KvNonce::words`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct KvNonce([u32; 3]);

impl std::fmt::Debug for KvNonce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KvNonce(<redacted>)")
    }
}

impl KvNonce {
    /// A nonce from its three words (tests; production nonces come from block hashes).
    pub fn from_words(words: [u32; 3]) -> Self {
        Self(words)
    }

    /// A nonce from RFC 8439's 12 bytes.
    pub fn from_bytes(bytes: [u8; 12]) -> Self {
        let w = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
        Self([w(0), w(1), w(2)])
    }

    /// The three nonce words, for an executor's device copy.
    pub fn words(&self) -> [u32; 3] {
        self.0
    }
}

#[inline(always)]
fn quarter(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

/// The RFC 8439 block function: the 16 keystream words of block `counter`. Written into
/// `out`, so the caller owns (and scrubs) every copy of the keystream.
pub fn chacha20_block(key: &[u32; 8], counter: u32, nonce: &[u32; 3], out: &mut [u32; 16]) {
    let mut input = [0u32; 16];
    input[..4].copy_from_slice(&SIGMA);
    input[4..12].copy_from_slice(key);
    input[12] = counter;
    input[13..].copy_from_slice(nonce);
    *out = input;
    for _ in 0..10 {
        quarter(out, 0, 4, 8, 12);
        quarter(out, 1, 5, 9, 13);
        quarter(out, 2, 6, 10, 14);
        quarter(out, 3, 7, 11, 15);
        quarter(out, 0, 5, 10, 15);
        quarter(out, 1, 6, 11, 12);
        quarter(out, 2, 7, 8, 13);
        quarter(out, 3, 4, 9, 14);
    }
    for (o, i) in out.iter_mut().zip(&input) {
        *o = o.wrapping_add(*i);
    }
    input.zeroize();
}

/// The keystream of one block's image (see the module docs): ranges in order, each from a
/// fresh keystream block. Holds the key's words, and zeroes them and its buffered keystream
/// when dropped.
pub struct KvKeystream {
    key: [u32; 8],
    nonce: [u32; 3],
    counter: u32,
    block: [u32; 16],
}

impl Drop for KvKeystream {
    fn drop(&mut self) {
        self.key.zeroize();
        self.block.zeroize();
    }
}

impl std::fmt::Debug for KvKeystream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KvKeystream(<redacted>)")
    }
}

impl KvKeystream {
    /// The keystream of a block image under `key` and `nonce`, from counter 0.
    pub fn new(key: &KvKey, nonce: &KvNonce) -> Self {
        Self::with_counter(key, nonce, 0)
    }

    /// The keystream from block `counter` on (RFC 8439's test vectors start at 1).
    pub fn with_counter(key: &KvKey, nonce: &KvNonce, counter: u32) -> Self {
        let mut k = [0u32; 8];
        key.words_into(&mut k);
        Self {
            key: k,
            nonce: nonce.0,
            counter,
            block: [0; 16],
        }
    }

    /// The block counter the next range starts at.
    pub fn counter(&self) -> u32 {
        self.counter
    }

    /// The next range of `words` words: calls `f(i, k)` for each `i` in `0..words` with
    /// keystream word `k` (the caller XORs it into word `i`), then moves the counter past
    /// the blocks the range used. Panics if the image outgrows the 32-bit block counter
    /// (256 GiB).
    pub fn range(&mut self, words: usize, mut f: impl FnMut(usize, u32)) {
        let blocks = u32::try_from(words.div_ceil(BLOCK_WORDS))
            .ok()
            .and_then(|b| self.counter.checked_add(b))
            .expect("a KV block image fits the 32-bit ChaCha20 counter");
        for (n, start) in (0..words).step_by(BLOCK_WORDS).enumerate() {
            chacha20_block(
                &self.key,
                self.counter + n as u32,
                &self.nonce,
                &mut self.block,
            );
            for (j, &k) in self.block[..(words - start).min(BLOCK_WORDS)]
                .iter()
                .enumerate()
            {
                f(start + j, k);
            }
        }
        self.counter = blocks;
    }

    /// XORs the next range, given as little-endian words.
    pub fn apply_words(&mut self, data: &mut [u32]) {
        let n = data.len();
        self.range(n, |i, k| data[i] ^= k);
    }

    /// XORs the next range, given as bytes (a whole number of words).
    pub fn apply_bytes(&mut self, data: &mut [u8]) {
        assert!(
            data.len().is_multiple_of(4),
            "a range is whole 32-bit words"
        );
        let n = data.len() / 4;
        self.range(n, |i, k| {
            for (b, kb) in data[4 * i..4 * i + 4].iter_mut().zip(k.to_le_bytes()) {
                *b ^= kb;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        let s: String = s.split_whitespace().collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn key_0_to_31() -> KvKey {
        KvKey::from_bytes(std::array::from_fn(|i| i as u8))
    }

    /// RFC 8439 §2.3.2: the block function's test vector.
    #[test]
    fn rfc8439_block_function() {
        let nonce = KvNonce::from_bytes(hex("000000090000004a00000000").try_into().unwrap());
        let mut ks = KvKeystream::with_counter(&key_0_to_31(), &nonce, 1);
        let mut out = vec![0u8; 64];
        ks.apply_bytes(&mut out);
        let want = hex(
            "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e
             d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e",
        );
        assert_eq!(out, want);
        assert_eq!(ks.counter(), 2);
    }

    /// RFC 8439 §2.4.2: encrypting the "sunscreen" plaintext from counter 1.
    #[test]
    fn rfc8439_encryption() {
        let nonce = KvNonce::from_bytes(hex("000000000000004a00000000").try_into().unwrap());
        let plain = b"Ladies and Gentlemen of the class of '99: If I could offer you only one \
                      tip for the future, sunscreen would be it.";
        // 114 bytes: pad to whole words (the trailing pad bytes are not compared).
        let mut buf = plain.to_vec();
        buf.resize(116, 0);
        KvKeystream::with_counter(&key_0_to_31(), &nonce, 1).apply_bytes(&mut buf);
        let want = hex(
            "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b
             f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8
             07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736
             5af90bbf74a35be6b40b8eedf2785e42874d",
        );
        assert_eq!(&buf[..114], &want[..]);
    }

    /// RFC 8439 Appendix A.1, test vectors 1 and 2 (all-zero key and nonce).
    #[test]
    fn rfc8439_appendix_a1() {
        let key = KvKey::from_bytes([0; 32]);
        let nonce = KvNonce::from_words([0; 3]);
        let mut out = vec![0u8; 128];
        KvKeystream::new(&key, &nonce).apply_bytes(&mut out);
        let want = hex(
            "76b8e0ada0f13d90405d6ae55386bd28bdd219b8a08ded1aa836efcc8b770dc7
             da41597c5157488d7724e03fb8d84a376a43b8f41518a11cc387b669b2ee6586
             9f07e7be5551387a98ba977c732d080dcb0f29a048e3656912c6533e32ee7aed
             29b721769ce64e43d57133b074d839d531ed1f28510afb45ace10a1f4b794d6f",
        );
        assert_eq!(out, want);
    }

    /// Against RustCrypto's `chacha20`, on random keys, nonces, counters and lengths:
    /// every range starts at its own block, at the counter the previous one left.
    #[test]
    fn matches_rustcrypto_chacha20() {
        use chacha20::cipher::{KeyIvInit, StreamCipher, StreamCipherSeek};
        let mut state = 0x0123_4567_89ab_cdefu64;
        let mut next = || {
            state = crate::sampling::mix64(state.wrapping_add(0x9e37_79b9_7f4a_7c15));
            state
        };
        for _ in 0..200 {
            let key_bytes: [u8; 32] = std::array::from_fn(|_| next() as u8);
            let nonce_bytes: [u8; 12] = std::array::from_fn(|_| next() as u8);
            let counter = (next() % 1000) as u32;
            let (key, nonce) = (
                KvKey::from_bytes(key_bytes),
                KvNonce::from_bytes(nonce_bytes),
            );
            let mut ours = KvKeystream::with_counter(&key, &nonce, counter);
            for _ in 0..3 {
                let words = (next() % 70) as usize;
                let data: Vec<u8> = (0..4 * words).map(|_| next() as u8).collect();
                let mut a = data.clone();
                let at = ours.counter();
                ours.apply_bytes(&mut a);
                let mut b = data.clone();
                let mut theirs = chacha20::ChaCha20::new(&key_bytes.into(), &nonce_bytes.into());
                theirs.seek(u64::from(at) * 64);
                theirs.apply_keystream(&mut b);
                assert_eq!(a, b);
                assert_eq!(ours.counter(), at + words.div_ceil(16) as u32);
            }
        }
    }

    #[test]
    fn applying_twice_restores_and_debug_is_redacted() {
        let key = key_0_to_31();
        let nonce = KvNonce::from_words([1, 2, 3]);
        let orig: Vec<u32> = (0..37).collect();
        let mut data = orig.clone();
        KvKeystream::new(&key, &nonce).apply_words(&mut data);
        assert_ne!(data, orig);
        KvKeystream::new(&key, &nonce).apply_words(&mut data);
        assert_eq!(data, orig);
        assert_eq!(format!("{nonce:?}"), "KvNonce(<redacted>)");
        assert_eq!(
            format!("{:?}", KvKeystream::new(&key, &nonce)),
            "KvKeystream(<redacted>)"
        );
    }
}
