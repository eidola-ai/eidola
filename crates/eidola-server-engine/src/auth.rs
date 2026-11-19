//! Gateway authentication.
//!
//! The gateway authenticates with a bearer token. The token is a secret delivered to the
//! node as `GATEWAY_TOKEN` (under the environment's prefix), and nothing else about it is
//! configured: boot refuses to start when it is missing or not of the token shape
//! (`eidola_common::engine_deployment::is_gateway_token`), and the node then accepts the
//! bearer that equals it. Per request, the presented bearer is compared with the token in
//! constant time, over SHA-256 digests, so the comparison does not depend on where the two
//! differ or on the presented length.
//!
//! This check is about money, not privacy: it keeps anyone but the gateway (which bills)
//! from spending the node's GPUs. No privacy property depends on who can call the node, so
//! the token is not bound to the node's measurement: a measured binding would hold the
//! operator, who holds the token, to nothing a client can use, and would make rotating it
//! a new engine identity.

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

/// The gateway token, held only as its SHA-256 digest.
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
    /// Holds `token`, read from the variable `name`, if it is of the token shape the
    /// gateway holds its token to. The error names the variable, never the value.
    pub fn new(mut token: String, name: &str) -> Result<Self, String> {
        let result = if eidola_common::engine_deployment::is_gateway_token(&token) {
            Ok(GatewayToken {
                digest: Sha256::digest(token.as_bytes()).into(),
            })
        } else {
            Err(format!(
                "{name} must be {} to {} visible ASCII characters",
                eidola_common::engine_deployment::GATEWAY_TOKEN_MIN_LEN,
                eidola_common::engine_deployment::GATEWAY_TOKEN_MAX_LEN
            ))
        };
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

    #[test]
    fn accepts_only_the_token() {
        let t = GatewayToken::new("s3cret-gateway-token".into(), "GATEWAY_TOKEN").unwrap();
        assert!(t.accepts(Some(b"Bearer s3cret-gateway-token")));
        assert!(!t.accepts(Some(b"Bearer s3cret-gateway-token ")));
        assert!(!t.accepts(Some(b"Bearer other-gateway-token")));
        assert!(!t.accepts(Some(b"bearer s3cret-gateway-token")));
        assert!(!t.accepts(Some(b"s3cret-gateway-token")));
        assert!(!t.accepts(None));
        assert_eq!(format!("{t:?}"), "GatewayToken(<redacted>)");
    }

    #[test]
    fn refuses_a_malformed_token_without_echoing() {
        for bad in ["short", "has a space-in-it", "non-ascii-tøken-xx"] {
            let e = GatewayToken::new(bad.into(), "ENGINE_GATEWAY_TOKEN").unwrap_err();
            assert!(e.starts_with("ENGINE_GATEWAY_TOKEN must be"), "{e}");
            assert!(!e.contains(bad), "{e}");
        }
    }
}
