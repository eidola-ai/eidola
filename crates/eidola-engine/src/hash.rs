//! The prefix-cache block-hash chain.
//!
//! ```text
//! hash(0) = SHA-256("eidola/kv-block/v1" ‖ 0x00 ‖ salt[32]   ‖ n:u32le ‖ tokens:u32le*)
//! hash(i) = SHA-256("eidola/kv-block/v1" ‖ 0x01 ‖ hash(i-1)  ‖ n:u32le ‖ tokens:u32le*)
//! ```
//!
//! The salt enters block 0 only; every later block chains its parent, so every block of a
//! sequence is salted. Only full blocks are hashed.

use sha2::{Digest, Sha256};

use crate::secret::EngineSalt;

const DOMAIN: &[u8] = b"eidola/kv-block/v1";

/// A block hash. It is an opaque but stable linkage tag for a (salt, prefix) pair: two
/// equal hashes mean "same salt and same tokens up to here", and nothing about the tokens
/// can be recovered from it. `Debug` is redacted, it has no `Display`, serialization or
/// byte accessor, and nothing outside this crate sees it.
///
/// It is not a secret and is not scrubbed. Hashes are `Copy` values held in growable
/// collections (the prefix-cache map's keys, each sequence's chain), and a rehash or
/// reallocation leaves earlier copies in freed memory that no type can reach, so zeroing
/// "every copy" could not be promised honestly. What bounds its exposure instead: it never
/// leaves the process, the process memory lives inside the encrypted confidential VM, and
/// it links only to a salt that is itself per-boot (or per-request for private requests).
/// The retention promises (TTL, zero on free) are about KV contents, not these tags.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash([u8; 32]);

impl std::fmt::Debug for BlockHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BlockHash(<redacted>)")
    }
}

impl BlockHash {
    /// Hash of block 0 of a sequence under `salt`.
    pub fn root(salt: &EngineSalt, tokens: &[u32]) -> Self {
        Self::digest(0, salt.expose(), tokens)
    }

    /// Hash of the block following `parent`.
    pub fn child(parent: &BlockHash, tokens: &[u32]) -> Self {
        Self::digest(1, &parent.0, tokens)
    }

    fn digest(tag: u8, prev: &[u8; 32], tokens: &[u32]) -> Self {
        let mut h = Sha256::new();
        h.update(DOMAIN);
        h.update([tag]);
        h.update(prev);
        h.update((tokens.len() as u32).to_le_bytes());
        for t in tokens {
            h.update(t.to_le_bytes());
        }
        let mut out = [0u8; 32];
        out.copy_from_slice(&h.finalize());
        BlockHash(out)
    }
}

/// Hashes of every full block of `tokens`, appended to `out` starting after the
/// `out.len()` blocks already present (which must be the chain for the same salt/prefix).
pub fn extend_chain(
    salt: &EngineSalt,
    tokens: &[u32],
    block_size: usize,
    out: &mut Vec<BlockHash>,
) {
    let full = tokens.len() / block_size;
    for i in out.len()..full {
        let blk = &tokens[i * block_size..(i + 1) * block_size];
        let h = match i {
            0 => BlockHash::root(salt, blk),
            _ => BlockHash::child(&out[i - 1], blk),
        };
        out.push(h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn salt_reaches_every_block() {
        let a = EngineSalt::from_bytes([1; 32]);
        let b = EngineSalt::from_bytes([2; 32]);
        let toks: Vec<u32> = (0..64).collect();
        let (mut ca, mut cb) = (Vec::new(), Vec::new());
        extend_chain(&a, &toks, 16, &mut ca);
        extend_chain(&b, &toks, 16, &mut cb);
        assert_eq!(ca.len(), 4);
        for (x, y) in ca.iter().zip(&cb) {
            assert_ne!(x, y);
        }
    }

    #[test]
    fn chain_is_incremental_and_prefix_stable() {
        let s = EngineSalt::from_bytes([5; 32]);
        let toks: Vec<u32> = (0..50).collect();
        let mut whole = Vec::new();
        extend_chain(&s, &toks, 16, &mut whole);
        let mut inc = Vec::new();
        extend_chain(&s, &toks[..20], 16, &mut inc);
        extend_chain(&s, &toks, 16, &mut inc);
        assert_eq!(whole, inc);
        let mut other = Vec::new();
        let mut toks2 = toks.clone();
        toks2[40] = 999;
        extend_chain(&s, &toks2, 16, &mut other);
        assert_eq!(whole[..2], other[..2]);
        assert_ne!(whole[2], other[2]);
    }

    #[test]
    fn debug_is_redacted() {
        let h = BlockHash::root(&EngineSalt::from_bytes([0; 32]), &[1, 2]);
        assert_eq!(format!("{h:?}"), "BlockHash(<redacted>)");
    }
}
