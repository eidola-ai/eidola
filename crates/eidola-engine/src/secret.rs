//! Prefix-cache secrets: the client's cache key, the per-boot key, and the two values
//! derived from them for one lineage (one client key): the engine salt that names its
//! cache entries and the key its idle KV is encrypted under.
//!
//! All of them are content-class values. They never appear in `Debug` output (every type here
//! prints a fixed redaction marker), are never compared in variable time by this crate, and
//! have no `Display`, serialization, or byte accessor outside this crate.
//!
//! **Zeroed on drop, with no stale copies.** Each value keeps its bytes in one fixed heap
//! allocation that is overwritten with zeros when the value is dropped. Moving the value
//! (into a request, through a `HashMap` rehash or a `Vec` reallocation) moves only the
//! pointer, so the collections that hold salts never leave copies of the bytes behind. A
//! clone is a separate allocation, zeroed on its own drop. Constructors write the bytes
//! straight into that allocation; `from_bytes` copies its argument, whose own copy is the
//! caller's to scrub, and `CacheKey::from_buffer` scrubs the caller's buffer itself. The HMAC's internal state during [`SaltDeriver::derive`] is transient
//! stack memory and is not scrubbed by this crate.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;
use zeroize::Zeroize;

/// Domain-separation label for deriving an engine salt from a client cache key.
pub const SALT_DERIVATION_LABEL: &[u8] = b"eidola/kv/v1";

/// Domain-separation label for deriving the idle-KV encryption key from a client cache key.
pub const KV_KEY_DERIVATION_LABEL: &[u8] = b"eidola/kv-enc/v1";

/// Length in bytes of every secret in this module.
pub const SECRET_LEN: usize = 32;

macro_rules! redacted_secret {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name(Box<[u8; SECRET_LEN]>);

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

            /// Copies `bytes` into a fresh allocation.
            fn copy_from(bytes: &[u8; SECRET_LEN]) -> Self {
                let mut v = Self(Box::new([0u8; SECRET_LEN]));
                v.0.copy_from_slice(bytes);
                v
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
    /// The ChaCha20 key a lineage's idle prefix-cache KV is encrypted under
    /// ([`crate::kv_cipher`]).
    ///
    /// Derived from the client's key (with the boot key), never from anything that
    /// outlives a request (the salt, block hashes, the cache map), so it exists only while
    /// a request of the lineage does: it travels with the request, and with the step that
    /// runs that request's encryptions and decryptions, and is zeroed when the last of
    /// them is dropped. Between requests an idle entry is ciphertext that nothing in the
    /// engine can decrypt.
    KvKey
);

redacted_secret!(
    /// A random key generated once per process; salts derived under it mean nothing after
    /// a restart, exactly as the cache itself does not survive one.
    BootKey
);

impl CacheKey {
    /// Wraps a client-supplied key.
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self::copy_from(&bytes)
    }

    /// Wraps a client-supplied key held in the caller's buffer, and zeroes that buffer.
    /// Unlike [`CacheKey::from_bytes`], no by-value copy of the key is left behind: the
    /// bytes are copied straight from `bytes` into the zeroizing allocation, and `bytes`
    /// is scrubbed before this returns.
    pub fn from_buffer(bytes: &mut [u8; SECRET_LEN]) -> Self {
        let key = Self::copy_from(bytes);
        bytes.zeroize();
        key
    }
}

impl EngineSalt {
    /// A fresh random salt. Used for requests that carry no cache key: such a request can
    /// only ever hit blocks it computed itself (after a preemption), never anyone else's.
    pub fn fresh() -> Self {
        Self(random_secret())
    }

    /// Constructs a salt from raw bytes. Test and integration use only; production salts
    /// come from [`SaltDeriver`] or [`EngineSalt::fresh`].
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self::copy_from(&bytes)
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

impl KvKey {
    /// A fresh random key, for a request without a client key (its entries are its own).
    pub fn fresh() -> Self {
        Self(random_secret())
    }

    /// Constructs a key from raw bytes. Test and cross-check use only; production keys
    /// come from [`SaltDeriver::lineage`] or [`Lineage::fresh`].
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self::copy_from(&bytes)
    }

    /// Writes the key's eight little-endian words into `out`. The one way the key's bytes
    /// leave this crate: an executor staging it for the device scrubs `out` (and every copy
    /// it makes) once the step's encryptions have been queued.
    pub fn words_into(&self, out: &mut [u32; 8]) {
        for (w, b) in out.iter_mut().zip(self.expose().as_chunks::<4>().0) {
            *w = u32::from_le_bytes(*b);
        }
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

/// Constant time, so a [`crate::executor::StepInput`] compares without a timing channel on
/// the keys it carries.
impl PartialEq for KvKey {
    fn eq(&self, other: &Self) -> bool {
        self.ct_eq(other)
    }
}

/// What a request carries for the prefix cache: the salt that names its entries and the
/// key their idle KV is encrypted under, derived together from one client key (or drawn
/// together at random for a request without one). There is no other constructor, so the
/// entries a salt finds are always the ones its key decrypts.
#[derive(Clone, Debug)]
pub struct Lineage {
    salt: EngineSalt,
    kv_key: KvKey,
}

impl Lineage {
    /// A fresh random salt and key: a request without a client key can only ever hit
    /// blocks it computed itself (after a preemption), never anyone else's.
    pub fn fresh() -> Self {
        Self {
            salt: EngineSalt::fresh(),
            kv_key: KvKey::fresh(),
        }
    }

    /// The salt.
    pub fn salt(&self) -> &EngineSalt {
        &self.salt
    }

    /// The idle-KV key.
    pub fn kv_key(&self) -> &KvKey {
        &self.kv_key
    }
}

impl BootKey {
    /// A fresh random per-boot key.
    pub fn generate() -> Self {
        Self(random_secret())
    }

    /// Constructs a boot key from raw bytes (tests only need determinism).
    pub fn from_bytes(bytes: [u8; SECRET_LEN]) -> Self {
        Self::copy_from(&bytes)
    }
}

/// Derives engine salts and idle-KV keys from client cache keys under one per-boot key.
///
/// ```text
/// engine_salt = HMAC-SHA256(boot_key, "eidola/kv/v1"     ‖ client_key)
/// kv_key      = HMAC-SHA256(boot_key, "eidola/kv-enc/v1" ‖ client_key)
/// ```
///
/// Neither output yields the other or the client key, so a salt (or a block hash chained
/// from it) left in memory after a request says nothing about the key.
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
        let mut salt = EngineSalt(Box::new([0u8; SECRET_LEN]));
        self.hmac(SALT_DERIVATION_LABEL, key, &mut salt.0);
        salt
    }

    /// Derives the idle-KV key for `key`.
    pub fn derive_kv_key(&self, key: &CacheKey) -> KvKey {
        let mut kv_key = KvKey(Box::new([0u8; SECRET_LEN]));
        self.hmac(KV_KEY_DERIVATION_LABEL, key, &mut kv_key.0);
        kv_key
    }

    /// The lineage of `key`: its salt and its idle-KV key.
    pub fn lineage(&self, key: &CacheKey) -> Lineage {
        Lineage {
            salt: self.derive(key),
            kv_key: self.derive_kv_key(key),
        }
    }

    /// `HMAC-SHA256(boot_key, label ‖ key)`, written straight into `out`.
    fn hmac(&self, label: &[u8], key: &CacheKey, out: &mut [u8; SECRET_LEN]) {
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(self.boot_key.expose())
            .expect("HMAC accepts any key length");
        mac.update(label);
        mac.update(key.expose());
        let mut tag = mac.finalize().into_bytes();
        out.copy_from_slice(&tag);
        tag.as_mut_slice().zeroize();
    }

    /// Resolves the salt for a request: derived when a key is present, fresh otherwise.
    pub fn salt_for(&self, key: Option<&CacheKey>) -> EngineSalt {
        match key {
            Some(k) => self.derive(k),
            None => EngineSalt::fresh(),
        }
    }

    /// Resolves the lineage for a request: derived when a key is present, fresh otherwise.
    pub fn lineage_for(&self, key: Option<&CacheKey>) -> Lineage {
        match key {
            Some(k) => self.lineage(k),
            None => Lineage::fresh(),
        }
    }
}

impl Default for SaltDeriver {
    fn default() -> Self {
        Self::new()
    }
}

/// A fresh allocation filled from the operating system's CSPRNG in place.
fn random_secret() -> Box<[u8; SECRET_LEN]> {
    let mut b = Box::new([0u8; SECRET_LEN]);
    fill_random(&mut b);
    b
}

/// Fills `out` from the operating system's CSPRNG. Failure to obtain randomness is fatal: a
/// predictable salt or boot key would silently weaken isolation.
fn fill_random(out: &mut [u8; SECRET_LEN]) {
    getrandom::fill(out).expect("operating-system randomness unavailable");
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
        let e = KvKey::from_bytes([0xAB; 32]);
        let l = d.lineage(&k);
        for text in [
            format!("{k:?}"),
            format!("{s:?}"),
            format!("{b:?}"),
            format!("{d:?}"),
            format!("{e:?}"),
            format!("{l:?}"),
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
    fn kv_key_derivation_matches_hmac_definition_and_is_separated() {
        let d = SaltDeriver::with_boot_key(BootKey::from_bytes([3; 32]));
        let k = CacheKey::from_bytes([9; 32]);
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(&[3; 32]).unwrap();
        let mut msg = KV_KEY_DERIVATION_LABEL.to_vec();
        msg.extend_from_slice(&[9; 32]);
        mac.update(&msg);
        let expected = mac.finalize().into_bytes();
        let kv = d.derive_kv_key(&k);
        assert_eq!(kv.expose().as_slice(), expected.as_slice());
        // Domain separation: the key is not the salt, and a lineage holds both.
        let l = d.lineage(&k);
        assert_ne!(kv.expose(), d.derive(&k).expose());
        assert!(l.kv_key().ct_eq(&kv) && l.salt().ct_eq(&d.derive(&k)));
        let mut words = [0u32; 8];
        kv.words_into(&mut words);
        assert_eq!(words[0].to_le_bytes(), expected[..4]);
        assert_eq!(words[7].to_le_bytes(), expected[28..]);
        // Keyed by the boot key and the client key.
        let d2 = SaltDeriver::with_boot_key(BootKey::from_bytes([4; 32]));
        assert!(!d2.derive_kv_key(&k).ct_eq(&kv));
        assert!(!d.derive_kv_key(&CacheKey::from_bytes([8; 32])).ct_eq(&kv));
        assert!(!Lineage::fresh().kv_key().ct_eq(Lineage::fresh().kv_key()));
    }

    /// Moving a secret, including through the rehashes of a growing `HashMap`, never
    /// moves its bytes: there is one allocation to zero and no stale copy left behind.
    #[test]
    fn moves_never_copy_the_secret_bytes() {
        let salt = EngineSalt::from_bytes([0x5A; 32]);
        let addr = salt.expose().as_ptr();
        let mut map = std::collections::HashMap::new();
        map.insert(0u32, salt);
        for i in 1..10_000u32 {
            map.insert(i, EngineSalt::fresh());
        }
        let moved = map.remove(&0).unwrap();
        assert_eq!(moved.expose().as_ptr(), addr);
        assert_eq!(moved.expose(), &[0x5A; 32]);
        let key = CacheKey::from_bytes([1; 32]);
        let key_addr = key.expose().as_ptr();
        let mut grown = Vec::with_capacity(1);
        grown.push(key);
        for i in 0..=255u8 {
            grown.push(CacheKey::from_bytes([i; 32]));
        }
        assert_eq!(grown[0].expose().as_ptr(), key_addr);
    }

    #[test]
    fn from_buffer_scrubs_the_callers_copy() {
        let mut buf = [0x42u8; 32];
        let key = CacheKey::from_buffer(&mut buf);
        assert_eq!(buf, [0u8; 32]);
        assert_eq!(key.expose(), &[0x42u8; 32]);
    }

    #[test]
    fn missing_key_gets_a_fresh_salt_every_time() {
        let d = SaltDeriver::new();
        assert!(!d.salt_for(None).ct_eq(&d.salt_for(None)));
    }
}
