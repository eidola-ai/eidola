//! How a client names a model the Eidola gateway serves: the one grammar the
//! gateway resolves and the client writes, so the two cannot disagree.
//!
//! A model's identity is its **weights hash** (`docs/model-identity.md`).
//! A reference is one of:
//!
//! - `tag`, which follows the tag: whatever weights it names in the gateway's
//!   build;
//! - `tag@sha256:<64 lowercase hex>`, a pin: as in a container image
//!   reference, **the digest is authoritative** and the tag only the client's
//!   label for it;
//! - `sha256:<64 lowercase hex>`, the weights with no label.
//!
//! A tag matches `[A-Za-z0-9._-]+` and is at most [`MAX_TAG_LEN`] bytes, so
//! it never holds the separators `@` and `:`, and what a response echoes stays
//! bounded.

/// The longest tag accepted, in a catalog row and in a reference alike.
pub const MAX_TAG_LEN: usize = 64;

/// The prefix of a digest in a reference.
pub const DIGEST_PREFIX: &str = "sha256:";

/// Whether `tag` is a tag: `[A-Za-z0-9._-]+`, at most [`MAX_TAG_LEN`] bytes.
pub fn is_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.len() <= MAX_TAG_LEN
        && tag
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Whether `hex` is a full SHA-256 digest in lowercase hex.
pub fn is_digest_hex(hex: &str) -> bool {
    crate::weights_manifest::is_lower_hex_sha256(hex)
}

/// A reference, parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelReference<'a> {
    /// `tag`: whatever the tag resolves to.
    Follow { tag: &'a str },
    /// `tag@sha256:<hex>`: the weights `sha256`, labelled `tag`.
    Pin { tag: &'a str, sha256: &'a str },
    /// `sha256:<hex>`: the weights, unlabelled.
    Hash { sha256: &'a str },
}

/// A string that is not a reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Malformed;

impl<'a> ModelReference<'a> {
    /// Parse a reference. Only full digests are accepted.
    pub fn parse(text: &'a str) -> Result<Self, Malformed> {
        let digest = |d: &'a str| -> Result<&'a str, Malformed> {
            d.strip_prefix(DIGEST_PREFIX)
                .filter(|hex| is_digest_hex(hex))
                .ok_or(Malformed)
        };
        if text.starts_with(DIGEST_PREFIX) {
            return Ok(ModelReference::Hash {
                sha256: digest(text)?,
            });
        }
        let (tag, pinned) = match text.split_once('@') {
            Some((tag, rest)) => (tag, Some(digest(rest)?)),
            None => (text, None),
        };
        if !is_tag(tag) {
            return Err(Malformed);
        }
        Ok(match pinned {
            Some(sha256) => ModelReference::Pin { tag, sha256 },
            None => ModelReference::Follow { tag },
        })
    }

    /// The weights hash the reference names by digest, if it carries one.
    pub fn digest(&self) -> Option<&'a str> {
        match self {
            ModelReference::Follow { .. } => None,
            ModelReference::Pin { sha256, .. } | ModelReference::Hash { sha256 } => Some(sha256),
        }
    }

    /// The tag the reference carries, if any.
    pub fn tag(&self) -> Option<&'a str> {
        match self {
            ModelReference::Follow { tag } | ModelReference::Pin { tag, .. } => Some(tag),
            ModelReference::Hash { .. } => None,
        }
    }
}

/// The pinned spelling of weights `sha256`, labelled `tag` when there is
/// one: `tag@sha256:<hex>`, or `sha256:<hex>`.
pub fn pinned(tag: Option<&str>, sha256: &str) -> String {
    match tag {
        Some(tag) => format!("{tag}@{DIGEST_PREFIX}{sha256}"),
        None => format!("{DIGEST_PREFIX}{sha256}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_is_a_tag_a_pin_or_a_hash() {
        let hash = "a".repeat(64);
        assert_eq!(
            ModelReference::parse("mimo-v2.6-flash"),
            Ok(ModelReference::Follow {
                tag: "mimo-v2.6-flash"
            })
        );
        let pin_text = format!("mimo-v2.6-flash@sha256:{hash}");
        let pin = ModelReference::parse(&pin_text).unwrap();
        assert_eq!(
            pin,
            ModelReference::Pin {
                tag: "mimo-v2.6-flash",
                sha256: &hash
            }
        );
        assert_eq!(pin.digest(), Some(hash.as_str()));
        assert_eq!(
            pinned(pin.tag(), &hash),
            format!("mimo-v2.6-flash@sha256:{hash}")
        );
        let bare_text = format!("sha256:{hash}");
        let bare = ModelReference::parse(&bare_text).unwrap();
        assert_eq!(bare, ModelReference::Hash { sha256: &hash });
        assert_eq!(pinned(bare.tag(), &hash), format!("sha256:{hash}"));
        for bad in [
            String::new(),
            "tag@".into(),
            format!("@sha256:{hash}"),
            format!("tag@sha256:{}", &hash[..63]),
            format!("tag@sha256:{}", hash.to_uppercase()),
            format!("tag@sha512:{hash}"),
            format!("tag@{hash}"),
            format!("sha256:{}", &hash[..12]),
            format!("tag@sha256:{hash}@eidola"),
            format!("{}@sha256:{hash}", "x".repeat(MAX_TAG_LEN + 1)),
            "has space".into(),
            "a:b".into(),
        ] {
            assert_eq!(ModelReference::parse(&bad), Err(Malformed), "{bad:?}");
        }
        assert!(is_tag(&"x".repeat(MAX_TAG_LEN)));
    }
}
