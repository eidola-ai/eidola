//! Prefix-cache secrets: the client's cache key, the per-boot key, and the engine salt.
//!
//! All three are content-class values. They never appear in `Debug` output (every type here
//! prints a fixed redaction marker), are never compared in variable time by this crate, are
//! zeroed on drop, and have no `Display`, serialization, or byte accessor outside this crate.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use zeroize::Zeroize;

/// Domain-separation label for deriving an engine salt from a client cache key.
pub const SALT_DERIVATION_LABEL: &[u8] = b"eidola/kv/v1";

/// Length in bytes of every secret in this module.
pub const SECRET_LEN: usize = 32;

macro_rules! redacted_secret {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name([u8; SECRET_LEN]);

        impl Drop for $name {
            fn drop(&mut self) {
                self.0.zeroize();
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(concat!(stringify!($name), "(<redacted>)"))
            }
        }

        impl $name {
            pub(crate) fn expose(&self) -> &[u8; SECRET_LEN] {
                &self.0
            }
        }
    };
}

redacted_secret!(
    /// The raw per-conversation cache key a client sends (256 random bits).
    ///
    /// The engine never uses it directly: [`SaltDeriver::derive`] turns it into an
    /// [`EngineSalt`] bound to this boot.
    CacheKey
);

redacted_secret!(
    /// The salt mixed into block 0 of every prefix-cache hash chain.
    ///
    /// Two sequences can share cached KV only if their salts are equal and their token
    /// prefixes are equal up to the shared block. There is no unsalted namespace.
    EngineSalt
);

redacted_secret!(
    /// A random key generated once per process; salts derived under it mean nothing after
    /// a restart, exactly as the cache itself does not survive one.
    BootKey
);

impl CacheKey {
    /// Wraps a client-supplied key.
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self(bytes)
    }
}

impl EngineSalt {
    /// A fresh random salt. Used for requests that carry no cache key: such a request can
    /// only ever hit blocks it computed itself (after a preemption), never anyone else's.
    pub fn fresh() -> Self {
        Self(random_bytes())
    }

    /// Constructs a salt from raw bytes. Test and integration use only; production salts
    /// come from [`SaltDeriver`] or [`EngineSalt::fresh`].
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self(bytes)
    }

    /// Constant-time equality.
    pub fn ct_eq(&self, other: &Self) -> bool {
        let mut acc = 0u8;
        for (a, b) in self.0.iter().zip(other.0.iter()) {
            acc |= a ^ b;
        }
        acc == 0
    }
}

impl BootKey {
    /// A fresh random per-boot key.
    pub fn generate() -> Self {
        Self(random_bytes())
    }

    /// Constructs a boot key from raw bytes (tests only need determinism).
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self(bytes)
    }
}

/// Derives engine salts from client cache keys under one per-boot key.
///
/// `engine_salt = HMAC-SHA256(boot_key, "eidola/kv/v1" ‖ client_key)`.
#[derive(Debug)]
pub struct SaltDeriver {
    boot_key: BootKey,
}

impl SaltDeriver {
    /// A deriver with a freshly generated per-boot key.
    pub fn new() -> Self {
        Self::with_boot_key(BootKey::generate())
    }

    /// A deriver with an explicit boot key.
    pub fn with_boot_key(boot_key: BootKey) -> Self {
        Self { boot_key }
    }

    /// Derives the engine salt for `key`.
    pub fn derive(&self, key: &CacheKey) -> EngineSalt {
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(self.boot_key.expose())
            .expect("HMAC accepts any key length");
        mac.update(SALT_DERIVATION_LABEL);
        mac.update(key.expose());
        let out = mac.finalize().into_bytes();
        let mut bytes = [0u8; SECRET_LEN];
        bytes.copy_from_slice(&out);
        EngineSalt(bytes)
    }

    /// Resolves the salt for a request: derived when a key is present, fresh otherwise.
    pub fn salt_for(&self, key: Option<&CacheKey>) -> EngineSalt {
        match key {
            Some(k) => self.derive(k),
            None => EngineSalt::fresh(),
        }
    }
}

impl Default for SaltDeriver {
    fn default() -> Self {
        Self::new()
    }
}

/// 32 bytes from the operating system's CSPRNG. Failure to obtain randomness is fatal: a
/// predictable salt or boot key would silently weaken isolation.
pub(crate) fn random_bytes() -> [u8; SECRET_LEN] {
    let mut b = [0u8; SECRET_LEN];
    getrandom::fill(&mut b).expect("operating-system randomness unavailable");
    b
}

/// A random `u64` from the operating system's CSPRNG.
pub(crate) fn random_u64() -> u64 {
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).expect("operating-system randomness unavailable");
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_is_redacted() {
        let k = CacheKey::from_bytes([0xAB; 32]);
        let s = EngineSalt::from_bytes([0xCD; 32]);
        let b = BootKey::from_bytes([0xEF; 32]);
        let d = SaltDeriver::with_boot_key(BootKey::from_bytes([0x11; 32]));
        for text in [
            format!("{k:?}"),
            format!("{s:?}"),
            format!("{b:?}"),
            format!("{d:?}"),
        ] {
            assert!(text.contains("<redacted>"), "{text}");
            for needle in [
                "ab", "AB", "171", "cd", "CD", "205", "ef", "EF", "239", "17,",
            ] {
                assert!(!text.contains(needle), "{text} leaks {needle}");
            }
        }
    }

    #[test]
    fn derivation_is_keyed_and_deterministic() {
        let d1 = SaltDeriver::with_boot_key(BootKey::from_bytes([1; 32]));
        let d2 = SaltDeriver::with_boot_key(BootKey::from_bytes([2; 32]));
        let k = CacheKey::from_bytes([7; 32]);
        let k2 = CacheKey::from_bytes([8; 32]);
        assert!(d1.derive(&k).ct_eq(&d1.derive(&k)));
        assert!(!d1.derive(&k).ct_eq(&d2.derive(&k)));
        assert!(!d1.derive(&k).ct_eq(&d1.derive(&k2)));
    }

    #[test]
    fn derivation_matches_hmac_definition() {
        let d = SaltDeriver::with_boot_key(BootKey::from_bytes([3; 32]));
        let k = CacheKey::from_bytes([9; 32]);
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(&[3; 32]).unwrap();
        let mut msg = SALT_DERIVATION_LABEL.to_vec();
        msg.extend_from_slice(&[9; 32]);
        mac.update(&msg);
        let expected = mac.finalize().into_bytes();
        assert_eq!(d.derive(&k).expose().as_slice(), expected.as_slice());
    }

    #[test]
    fn missing_key_gets_a_fresh_salt_every_time() {
        let d = SaltDeriver::new();
        assert!(!d.salt_for(None).ct_eq(&d.salt_for(None)));
    }
}
