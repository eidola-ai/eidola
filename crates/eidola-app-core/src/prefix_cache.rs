//! The prefix-cache key a turn sends to an Eidola-hosted engine.
//!
//! An Eidola-hosted engine reuses the computation behind a prompt prefix only
//! between requests that carry the same `cache_key`
//! ([`eidola_common::engine_protocol`]). The key therefore does two things at
//! once: it is what lets a conversation's next turn skip re-reading its
//! history, and it is an identifier that links every request carrying it. The
//! rules here keep the second no larger than the first needs:
//!
//! - **One key per lineage** — one participant answering in one space
//!   ([`crate::db::claim_prefix_cache_key`]). Never one key across spaces or
//!   across participants.
//! - **Sent only where the catalog says the engine caches**
//!   ([`PromptCachePolicy::from_catalog`]); every other model gets no key and
//!   a request body byte-for-byte the one it had before keys existed.
//! - **Rotated before the cache it unlocks is gone** ([`decide`]): idle past
//!   the engine's idle TTL, older than its maximum age, or a clock that
//!   disagrees with itself. Rotation mints a fresh random key, so a lineage's
//!   requests are linkable only within one key's window.
//! - **Never carried across a boundary**: a key is bound to the model and to
//!   the [`trust_domain`] — the eidola endpoint and trust bundle — it was minted
//!   under, and a change of either mints a new one.
//! - **Forgotten when it can no longer be sent**
//!   ([`crate::db::forget_unsendable_cache_keys`]), overwritten before it goes.
//!
//! The key is secret on this side as on the server's: [`CacheKey`] prints
//! redacted and scrubs its text on drop, and the bytes are scrubbed wherever
//! this crate copies them. Out of reach: turso's own buffers for the stored
//! row, and the serialized request body inside the HTTP stack.

use rand_core::{OsRng, RngCore};
use zeroize::{Zeroize, Zeroizing};

use eidola_common::engine_protocol::{CACHE_KEY_BYTES, CACHE_KEY_TEXT_LEN, encode_cache_key};

/// What a model's catalog row says about its engine's prefix cache, in the
/// units the rotation rule compares: milliseconds of this client's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PromptCachePolicy {
    /// A key idle this long since its last request is replaced.
    pub idle_ttl_ms: i64,
    /// A key this old is replaced, however busy.
    pub max_age_ms: i64,
}

impl PromptCachePolicy {
    /// Read the catalog's `capabilities.prompt_cache` leaf
    /// (`{supported, idle_ttl_secs, max_age_secs}`).
    ///
    /// **Anything short of a complete, positive declaration is no policy, and
    /// no policy means no key.** `supported` must be literally `true`, and both
    /// bounds present, integral and non-zero: a key whose lifetime this client
    /// cannot bound would be an identifier with no expiry, which is the one
    /// thing the rotation rule exists to prevent. Parsed from a raw value so a
    /// leaf of an unexpected shape degrades to "no key" rather than failing the
    /// catalog read every turn depends on.
    pub(crate) fn from_catalog(leaf: Option<&serde_json::Value>) -> Option<Self> {
        let leaf = leaf?;
        if leaf.get("supported")?.as_bool()? {
            let secs = |name: &str| {
                leaf.get(name)?
                    .as_u64()
                    .filter(|s| *s > 0)
                    .and_then(|s| i64::try_from(s).ok())
                    .map(|s| s.saturating_mul(1000))
            };
            Some(Self {
                idle_ttl_ms: secs("idle_ttl_secs")?,
                max_age_ms: secs("max_age_secs")?,
            })
        } else {
            None
        }
    }
}

/// The digest naming the trust domain a key is minted under: the eidola
/// backend's resolved base URL and its whole trust bundle (accepted
/// measurements, hardware root and intermediate CA overrides).
///
/// A key is a linking identifier, so it must never be seen by two servers
/// that answer to different trust: pointing the client at a dev stack and back
/// (`just client-local` / `client-reset`), or at any other endpoint, would
/// otherwise carry one secret into both and link the lineage across them.
/// Domain-separated and length-prefixed, so no two bundles share a digest by
/// concatenation.
pub(crate) fn trust_domain(
    base_url: &str,
    measurements_json: &str,
    hardware_root_ca: Option<&str>,
    hardware_intermediate_ca: Option<&str>,
) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"eidola/prefix-cache/trust-domain/v1");
    for part in [
        Some(base_url),
        Some(measurements_json),
        hardware_root_ca,
        hardware_intermediate_ca,
    ] {
        match part {
            Some(s) => {
                h.update([1u8]);
                h.update((s.len() as u64).to_be_bytes());
                h.update(s.as_bytes());
            }
            None => h.update([0u8]),
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// What a turn's requests may be keyed under: the model's retention and the
/// trust domain this turn talks to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheScope {
    pub policy: PromptCachePolicy,
    pub trust_domain: String,
}

/// A lineage's stored key, minus the key: what [`decide`] judges it by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StoredKeyAge {
    /// The model the key was minted for.
    pub model: String,
    /// The [`trust_domain`] the key was minted under.
    pub trust_domain: String,
    /// When the key was minted, by this client's clock.
    pub created_at: i64,
    /// The attempt time of the latest request known to have left with the key
    /// (its birth, until one has), by this client's clock.
    pub last_used_at: i64,
}

/// Whether a lineage keeps its key for the next request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Claim {
    /// The stored key is still inside every bound: send it again.
    Reuse,
    /// Mint a fresh key and replace the stored one.
    Mint,
}

/// The rotation rule. Every doubt resolves to [`Claim::Mint`]: a fresh key
/// costs one cache miss, a stale one links requests past the window the
/// catalog promised.
///
/// - No stored key, or one minted for another model (a participant whose
///   model changed is answered by another engine, under another retention) or
///   under another trust domain (a key never reaches two).
/// - **The clock disagrees with itself**: `now` earlier than either stored
///   time, or a last use before the birth. The client cannot tell a clock set
///   back from a clock that was ahead, and either way it cannot measure the
///   key's age or idleness, so it does not try.
/// - Idle for `idle_ttl_ms` or more since its last request.
/// - `max_age_ms` or more since it was minted.
///
/// `now` is the instant the request's body is first asked for, after its
/// connection and attestation (`keyed_body` in `lib.rs`), and both stored times
/// are such instants: no later than any engine could see the request, so both
/// measures run at least as long as the engine's own and the key is retired no
/// later than the cache could still answer for it; and no earlier than it could
/// be observed, so idleness is measured from the last request anyone could have
/// seen and never renewed by one whose connection never opened.
pub(crate) fn decide(
    stored: Option<&StoredKeyAge>,
    model: &str,
    trust_domain: &str,
    policy: &PromptCachePolicy,
    now: i64,
) -> Claim {
    let Some(stored) = stored else {
        return Claim::Mint;
    };
    if stored.model != model
        || stored.trust_domain != trust_domain
        || now < stored.created_at
        || now < stored.last_used_at
        || stored.last_used_at < stored.created_at
    {
        return Claim::Mint;
    }
    if now - stored.last_used_at >= policy.idle_ttl_ms
        || now - stored.created_at >= policy.max_age_ms
    {
        return Claim::Mint;
    }
    Claim::Reuse
}

/// 32 fresh bytes from the operating system's CSPRNG, or `None` when it
/// cannot supply them — in which case the turn sends no key at all rather
/// than a weaker one.
pub(crate) fn mint() -> Option<Zeroizing<[u8; CACHE_KEY_BYTES]>> {
    let mut bytes = Zeroizing::new([0u8; CACHE_KEY_BYTES]);
    OsRng.try_fill_bytes(&mut bytes[..]).ok()?;
    Some(bytes)
}

/// A prefix-cache key in its wire spelling, held for one request.
///
/// Prints redacted; its text is scrubbed on drop. Built only from key bytes,
/// through the shared encoder, so it is always a key the gateway accepts.
pub(crate) struct CacheKey {
    text: Zeroizing<[u8; CACHE_KEY_TEXT_LEN]>,
}

impl CacheKey {
    pub(crate) fn from_bytes(bytes: &[u8; CACHE_KEY_BYTES]) -> Self {
        Self {
            text: Zeroizing::new(encode_cache_key(bytes)),
        }
    }

    /// The key as the request body's `cache_key` carries it.
    pub(crate) fn as_str(&self) -> &str {
        std::str::from_utf8(&self.text[..]).expect("the encoder writes ASCII")
    }
}

impl std::fmt::Debug for CacheKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CacheKey(<redacted>)")
    }
}

/// Scrub a request body's `cache_key` in place: the copy of the key the body
/// holds once [`eidola_common::chat_completion_request_body`] has written it.
pub(crate) fn scrub_body_key(body: &mut serde_json::Value) {
    if let Some(serde_json::Value::String(text)) = body.get_mut("cache_key") {
        text.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY: PromptCachePolicy = PromptCachePolicy {
        idle_ttl_ms: 900_000,
        max_age_ms: 7_200_000,
    };

    fn stored(model: &str, created_at: i64, last_used_at: i64) -> StoredKeyAge {
        StoredKeyAge {
            model: model.to_string(),
            trust_domain: "d".to_string(),
            created_at,
            last_used_at,
        }
    }

    #[test]
    fn only_a_complete_positive_declaration_is_a_policy() {
        let read = |v: serde_json::Value| PromptCachePolicy::from_catalog(Some(&v));
        assert_eq!(
            read(
                serde_json::json!({"supported": true, "idle_ttl_secs": 900, "max_age_secs": 7200})
            ),
            Some(POLICY)
        );
        assert_eq!(PromptCachePolicy::from_catalog(None), None);
        for leaf in [
            serde_json::json!({"supported": false}),
            serde_json::json!({"supported": false, "idle_ttl_secs": 900, "max_age_secs": 7200}),
            serde_json::json!({"idle_ttl_secs": 900, "max_age_secs": 7200}),
            serde_json::json!({"supported": "true", "idle_ttl_secs": 900, "max_age_secs": 7200}),
            serde_json::json!({"supported": true}),
            serde_json::json!({"supported": true, "idle_ttl_secs": 900}),
            serde_json::json!({"supported": true, "max_age_secs": 7200}),
            serde_json::json!({"supported": true, "idle_ttl_secs": 0, "max_age_secs": 7200}),
            serde_json::json!({"supported": true, "idle_ttl_secs": 900, "max_age_secs": 0}),
            serde_json::json!({"supported": true, "idle_ttl_secs": -1, "max_age_secs": 7200}),
            serde_json::json!({"supported": true, "idle_ttl_secs": 1.5, "max_age_secs": 7200}),
            serde_json::json!({"supported": true, "idle_ttl_secs": "900", "max_age_secs": 7200}),
            serde_json::json!(true),
            serde_json::json!(null),
        ] {
            assert_eq!(read(leaf.clone()), None, "{leaf}");
        }
    }

    #[test]
    fn an_enormous_bound_saturates_rather_than_wrapping() {
        let leaf = serde_json::json!({
            "supported": true, "idle_ttl_secs": u64::MAX, "max_age_secs": i64::MAX as u64
        });
        // u64::MAX does not fit i64: no policy. i64::MAX seconds saturates.
        assert_eq!(PromptCachePolicy::from_catalog(Some(&leaf)), None);
        let leaf = serde_json::json!({
            "supported": true, "idle_ttl_secs": i64::MAX as u64, "max_age_secs": i64::MAX as u64
        });
        let policy = PromptCachePolicy::from_catalog(Some(&leaf)).unwrap();
        assert_eq!(policy.idle_ttl_ms, i64::MAX);
    }

    #[test]
    fn a_key_inside_every_bound_is_reused() {
        let s = stored("m", 1_000, 2_000);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, 2_000), Claim::Reuse);
        assert_eq!(
            decide(Some(&s), "m", "d", &POLICY, 2_000 + POLICY.idle_ttl_ms - 1),
            Claim::Reuse
        );
    }

    #[test]
    fn no_key_or_another_models_key_mints() {
        assert_eq!(decide(None, "m", "d", &POLICY, 5_000), Claim::Mint);
        let s = stored("other", 1_000, 2_000);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, 2_000), Claim::Mint);
    }

    #[test]
    fn another_trust_domains_key_mints() {
        let s = stored("m", 1_000, 2_000);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, 2_000), Claim::Reuse);
        assert_eq!(decide(Some(&s), "m", "e", &POLICY, 2_000), Claim::Mint);
    }

    #[test]
    fn a_trust_domain_names_the_whole_bundle() {
        let base = trust_domain("https://a", "[]", None, None);
        assert_eq!(base, trust_domain("https://a", "[]", None, None));
        for other in [
            trust_domain("https://b", "[]", None, None),
            trust_domain("https://a", "[1]", None, None),
            trust_domain("https://a", "[]", Some(""), None),
            trust_domain("https://a", "[]", None, Some("ask")),
            trust_domain("https://a", "[]", Some("ask"), None),
        ] {
            assert_ne!(base, other);
        }
        // Length-prefixed: moving a boundary changes the digest.
        assert_ne!(
            trust_domain("https://ab", "c", None, None),
            trust_domain("https://a", "bc", None, None)
        );
    }

    #[test]
    fn idleness_reaching_the_ttl_mints() {
        let s = stored("m", 1_000, 2_000);
        assert_eq!(
            decide(Some(&s), "m", "d", &POLICY, 2_000 + POLICY.idle_ttl_ms),
            Claim::Mint
        );
    }

    #[test]
    fn age_reaching_the_maximum_mints_however_busy() {
        // Used a moment ago, but born max_age ago.
        let now = 10_000_000;
        let s = stored("m", now - POLICY.max_age_ms, now - 1);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, now), Claim::Mint);
        let s = stored("m", now - POLICY.max_age_ms + 1, now - 1);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, now), Claim::Reuse);
    }

    #[test]
    fn a_clock_that_disagrees_with_itself_mints() {
        let s = stored("m", 1_000, 2_000);
        // Set back past the last use, and past the birth.
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, 1_999), Claim::Mint);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, 999), Claim::Mint);
        // A stored row whose last use precedes its birth.
        let s = stored("m", 2_000, 1_000);
        assert_eq!(decide(Some(&s), "m", "d", &POLICY, 2_500), Claim::Mint);
    }

    #[test]
    fn minted_keys_are_fresh_and_spell_canonically() {
        let a = mint().expect("the OS supplies randomness");
        let b = mint().expect("the OS supplies randomness");
        assert_ne!(&a[..], &b[..]);
        let key = CacheKey::from_bytes(&a);
        assert!(eidola_common::engine_protocol::is_cache_key_text(
            key.as_str()
        ));
    }

    #[test]
    fn a_key_never_prints() {
        let key = CacheKey::from_bytes(&[7; CACHE_KEY_BYTES]);
        let printed = format!("{key:?}");
        assert_eq!(printed, "CacheKey(<redacted>)");
        assert!(!printed.contains(key.as_str()));
    }

    #[test]
    fn a_bodys_key_is_scrubbed_in_place() {
        let key = CacheKey::from_bytes(&[7; CACHE_KEY_BYTES]);
        let mut body = eidola_common::chat_completion_request_body(
            "m",
            &[],
            16,
            &[],
            false,
            false,
            Some(key.as_str()),
        );
        scrub_body_key(&mut body);
        assert!(!body.to_string().contains(key.as_str()));
    }
}
