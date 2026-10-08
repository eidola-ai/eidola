//! Gateway authentication.
//!
//! The gateway authenticates with a bearer token. The token is a secret delivered to the
//! node as `GATEWAY_TOKEN`; its Argon2id hash is the measured half (`GATEWAY_TOKEN_HASH`),
//! exactly as the server binds its injected secrets. Boot verifies the token against the
//! hash once and refuses to start on a mismatch, a malformed hash, or either value
//! missing. Per request, the presented bearer is compared with the verified token in
//! constant time (over SHA-256 digests, so the comparison does not depend on where the
//! two differ or on the presented length); running Argon2 per request would hand any
//! caller a cheap way to burn the node's CPU.
//!
//! This check is about money, not privacy: it keeps anyone but the gateway (which bills)
//! from spending the node's GPUs. No privacy property depends on who can call the node.

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

/// The verified gateway token, held only as its SHA-256 digest.
pub struct GatewayToken {
    digest: [u8; 32],
}

impl std::fmt::Debug for GatewayToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GatewayToken(<redacted>)")
    }
}

impl Drop for GatewayToken {
    fn drop(&mut self) {
        self.digest.zeroize();
    }
}

impl GatewayToken {
    /// Verifies `token` against the measured Argon2id `hash`. The error names the
    /// variables, never a value.
    pub fn verify(mut token: String, hash: &str) -> Result<Self, String> {
        use argon2::PasswordVerifier;
        let result = (|| {
            let parsed = argon2::PasswordHash::new(hash).map_err(|_| {
                format!(
                    "{}: not a valid Argon2 hash string",
                    crate::config::env::GATEWAY_TOKEN_HASH
                )
            })?;
            if parsed.algorithm.as_str() != "argon2id" {
                return Err(format!(
                    "{}: must be an Argon2id hash",
                    crate::config::env::GATEWAY_TOKEN_HASH
                ));
            }
            argon2::Argon2::default()
                .verify_password(token.as_bytes(), &parsed)
                .map_err(|_| {
                    format!(
                        "{} does not match the measured hash in {}",
                        crate::config::env::GATEWAY_TOKEN,
                        crate::config::env::GATEWAY_TOKEN_HASH
                    )
                })?;
            Ok(GatewayToken {
                digest: Sha256::digest(token.as_bytes()).into(),
            })
        })();
        token.zeroize();
        result
    }

    /// Whether an `Authorization` header value is `Bearer <this token>`.
    pub fn accepts(&self, authorization: Option<&[u8]>) -> bool {
        let Some(value) = authorization else {
            return false;
        };
        let Some(presented) = value.strip_prefix(b"Bearer ") else {
            return false;
        };
        let digest: [u8; 32] = Sha256::digest(presented).into();
        bool::from(digest.ct_eq(&self.digest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argon2::PasswordHasher;

    fn hash(secret: &str) -> String {
        argon2::Argon2::default()
            .hash_password(secret.as_bytes())
            .unwrap()
            .to_string()
    }

    #[test]
    fn verifies_and_accepts_only_the_token() {
        let t = GatewayToken::verify("s3cret".into(), &hash("s3cret")).unwrap();
        assert!(t.accepts(Some(b"Bearer s3cret")));
        assert!(!t.accepts(Some(b"Bearer s3cret ")));
        assert!(!t.accepts(Some(b"bearer s3cret")));
        assert!(!t.accepts(Some(b"s3cret")));
        assert!(!t.accepts(None));
        assert_eq!(format!("{t:?}"), "GatewayToken(<redacted>)");
    }

    #[test]
    fn refuses_mismatch_and_malformed_hashes_without_echoing() {
        let e = GatewayToken::verify("s3cret".into(), &hash("other")).unwrap_err();
        assert!(!e.contains("s3cret") && !e.contains("other"), "{e}");
        assert!(GatewayToken::verify("s3cret".into(), "not-a-hash").is_err());
    }
}
