//! RFC 6962 Merkle inclusion-proof verification (sha256).
//!
//! Used by both [`super::ci_sigstore::rekor`] (CI sigstore bundle path,
//! `hashedrekord` entries) and [`super::human_attestation`] (engineer
//! SSH `rekord` path with `signature.format=ssh`)
//! to verify that a leaf hash is in a tree of declared `tree_size`
//! whose root is `proof_root_hash`.
//!
//! The `chain_inner` / `chain_border_right` / `decomp_inclusion_proof`
//! decomposition is adapted from sigstore-rs (Apache 2.0), originally
//! from the transparency-dev Merkle reference implementation. This
//! trimmed copy supports inclusion proofs only; consistency proofs are
//! not needed by either consumer.

use sha2::{Digest, Sha256};

use crate::error::AppError;

/// RFC 6962 leaf hash: `sha256(0x00 || leaf)`.
pub fn hash_leaf(leaf: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([0x00]);
    h.update(leaf);
    h.finalize().into()
}

/// RFC 6962 internal node hash: `sha256(0x01 || left || right)`.
pub fn hash_children(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update([0x01]);
    h.update(left);
    h.update(right);
    h.finalize().into()
}

/// Verify that `leaf_hash` is at position `index` in a tree of
/// `tree_size` leaves with root `proof_root_hash`, witnessed by the
/// sibling hashes in `proof_hashes` (ordered leaf-most up).
pub fn verify_inclusion_proof(
    index: u64,
    leaf_hash: &[u8; 32],
    tree_size: u64,
    proof_hashes: &[[u8; 32]],
    proof_root_hash: &[u8; 32],
) -> Result<(), AppError> {
    let computed = compute_root_from_proof(index, leaf_hash, tree_size, proof_hashes)?;
    if computed != *proof_root_hash {
        return Err(AppError::Update {
            message: format!(
                "Merkle inclusion proof's recomputed root `{}` ≠ declared rootHash `{}`",
                hex_encode(&computed),
                hex_encode(proof_root_hash)
            ),
        });
    }
    Ok(())
}

fn compute_root_from_proof(
    index: u64,
    leaf_hash: &[u8; 32],
    tree_size: u64,
    proof_hashes: &[[u8; 32]],
) -> Result<[u8; 32], AppError> {
    if index >= tree_size {
        return Err(AppError::Update {
            message: format!("Merkle inclusion proof: leaf index {index} >= tree size {tree_size}"),
        });
    }
    let (inner, border) = decomp_inclusion_proof(index, tree_size);
    let expected_len = inner + border;
    if proof_hashes.len() as u64 != expected_len {
        return Err(AppError::Update {
            message: format!(
                "Merkle inclusion proof has {} hashes, expected {}",
                proof_hashes.len(),
                expected_len
            ),
        });
    }
    let after_inner = chain_inner(*leaf_hash, &proof_hashes[..inner as usize], index);
    Ok(chain_border_right(
        after_inner,
        &proof_hashes[inner as usize..],
    ))
}

fn chain_inner(mut seed: [u8; 32], proof_hashes: &[[u8; 32]], index: u64) -> [u8; 32] {
    for (i, h) in proof_hashes.iter().enumerate() {
        seed = if ((index >> i) & 1) == 0 {
            hash_children(&seed, h)
        } else {
            hash_children(h, &seed)
        };
    }
    seed
}

fn chain_border_right(mut seed: [u8; 32], proof_hashes: &[[u8; 32]]) -> [u8; 32] {
    for h in proof_hashes {
        seed = hash_children(h, &seed);
    }
    seed
}

fn decomp_inclusion_proof(index: u64, tree_size: u64) -> (u64, u64) {
    let inner = inner_proof_size(index, tree_size);
    let border = (index >> inner).count_ones() as u64;
    (inner, border)
}

fn inner_proof_size(index: u64, tree_size: u64) -> u64 {
    u64::BITS as u64 - ((index ^ (tree_size - 1)).leading_zeros() as u64)
}

fn hex_encode(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(b.len() * 2);
    for byte in b {
        write!(out, "{byte:02x}").unwrap();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc6962_leaf_hash_empty() {
        let got = hash_leaf(b"");
        let expected = {
            let mut h = Sha256::new();
            h.update([0x00]);
            let d: [u8; 32] = h.finalize().into();
            d
        };
        assert_eq!(got, expected);
    }

    #[test]
    fn inclusion_proof_single_leaf_tree() {
        let leaf = hash_leaf(b"hello");
        verify_inclusion_proof(0, &leaf, 1, &[], &leaf).unwrap();
    }

    #[test]
    fn inclusion_proof_two_leaf_tree() {
        let l0 = hash_leaf(b"a");
        let l1 = hash_leaf(b"b");
        let root = hash_children(&l0, &l1);
        verify_inclusion_proof(0, &l0, 2, &[l1], &root).unwrap();
        verify_inclusion_proof(1, &l1, 2, &[l0], &root).unwrap();
    }

    #[test]
    fn inclusion_proof_rejects_wrong_proof_length() {
        let l0 = hash_leaf(b"a");
        let err = verify_inclusion_proof(0, &l0, 2, &[], &l0).unwrap_err();
        assert!(format!("{err}").contains("expected"));
    }

    #[test]
    fn inclusion_proof_rejects_index_out_of_range() {
        let l0 = hash_leaf(b"a");
        let err = verify_inclusion_proof(5, &l0, 2, &[], &l0).unwrap_err();
        assert!(format!("{err}").contains("leaf index"));
    }

    #[test]
    fn inclusion_proof_rejects_wrong_root() {
        let l0 = hash_leaf(b"a");
        let l1 = hash_leaf(b"b");
        let wrong_root = [0u8; 32];
        let err = verify_inclusion_proof(0, &l0, 2, &[l1], &wrong_root).unwrap_err();
        assert!(format!("{err}").contains("recomputed root"));
    }
}
