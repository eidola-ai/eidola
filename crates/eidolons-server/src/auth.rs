//! Authorization middleware for the Eidolons server.
//!
//! Supports Privacy Pass (RFC 9577) token validation.
//! Designed as a trait so implementations can be swapped at startup.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use http::HeaderValue;
use privacypass::auth::authorize::parse_authorization_header;
use privacypass::public_tokens::server::{OriginKeyStore, OriginServer};
use privacypass::public_tokens::{PublicKey, PublicToken};
use privacypass::{NonceStore, TruncatedTokenKeyId};
use serde::Serialize;
use tokio::sync::RwLock;
use utoipa::ToSchema;

use crate::error::ServerError;

/// Result of successful token validation.
#[derive(Debug, Clone)]
pub struct AuthContext {
    /// The authorization method that was used.
    pub method: AuthMethod,
}

/// Supported authorization methods.
#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum AuthMethod {
    /// Privacy Pass token (RFC 9577).
    PrivacyPass,
    /// No authentication (development only).
    None,
}

impl AuthMethod {
    /// Whether requests using this method are linkable across requests.
    pub fn linkable(&self) -> bool {
        match self {
            AuthMethod::PrivacyPass => false,
            AuthMethod::None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// TokenValidator trait
// ---------------------------------------------------------------------------

/// Trait for validating authorization tokens.
pub trait TokenValidator: Send + Sync {
    /// Validate an Authorization header value.
    fn validate(
        &self,
        authorization_header: &HeaderValue,
    ) -> impl std::future::Future<Output = Result<AuthContext, ServerError>> + Send;
}

// ---------------------------------------------------------------------------
// Enum dispatch (avoids dyn + async_trait on the hot path)
// ---------------------------------------------------------------------------

/// Runtime-selected token validator.
pub enum AnyValidator {
    PrivacyPass(PrivacyPassValidator),
    Noop(NoopValidator),
}

impl AnyValidator {
    pub async fn validate(
        &self,
        authorization_header: &HeaderValue,
    ) -> Result<AuthContext, ServerError> {
        match self {
            AnyValidator::PrivacyPass(v) => v.validate(authorization_header).await,
            AnyValidator::Noop(v) => v.validate(authorization_header).await,
        }
    }
}

// ---------------------------------------------------------------------------
// Privacy Pass validator (publicly verifiable tokens, RFC 9578 type 2)
// ---------------------------------------------------------------------------

/// In-memory nonce store for preventing double-spending of Privacy Pass tokens.
pub struct MemoryNonceStore {
    /// Nonces that have been seen. In production this would be backed by a
    /// persistent store, but for a single-server deployment in-memory is fine.
    nonces: RwLock<HashMap<[u8; 32], NonceState>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum NonceState {
    Reserved,
    Committed,
}

impl Default for MemoryNonceStore {
    fn default() -> Self {
        Self {
            nonces: RwLock::new(HashMap::new()),
        }
    }
}

impl MemoryNonceStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl NonceStore for MemoryNonceStore {
    async fn reserve(&self, nonce: &[u8; 32]) -> bool {
        let mut store = self.nonces.write().await;
        if store.contains_key(nonce) {
            false
        } else {
            store.insert(*nonce, NonceState::Reserved);
            true
        }
    }

    async fn commit(&self, nonce: &[u8; 32]) {
        let mut store = self.nonces.write().await;
        store.insert(*nonce, NonceState::Committed);
    }

    async fn release(&self, nonce: &[u8; 32]) {
        let mut store = self.nonces.write().await;
        if store.get(nonce) == Some(&NonceState::Reserved) {
            store.remove(nonce);
        }
    }
}

/// In-memory public key store for verifying Privacy Pass tokens.
pub struct MemoryOriginKeyStore {
    keys: RwLock<HashMap<TruncatedTokenKeyId, Vec<PublicKey>>>,
}

impl Default for MemoryOriginKeyStore {
    fn default() -> Self {
        Self {
            keys: RwLock::new(HashMap::new()),
        }
    }
}

impl MemoryOriginKeyStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a public key (DER-encoded SPKI) for token verification.
    pub async fn add_public_key(&self, truncated_id: TruncatedTokenKeyId, key: PublicKey) {
        let mut store = self.keys.write().await;
        store.entry(truncated_id).or_default().push(key);
    }
}

#[async_trait]
impl OriginKeyStore for MemoryOriginKeyStore {
    async fn insert(&self, truncated_token_key_id: TruncatedTokenKeyId, key: PublicKey) {
        let mut store = self.keys.write().await;
        store.entry(truncated_token_key_id).or_default().push(key);
    }

    async fn get(&self, truncated_token_key_id: &TruncatedTokenKeyId) -> Vec<PublicKey> {
        let store = self.keys.read().await;
        store
            .get(truncated_token_key_id)
            .cloned()
            .unwrap_or_default()
    }

    async fn remove(&self, truncated_token_key_id: &TruncatedTokenKeyId) -> bool {
        let mut store = self.keys.write().await;
        store.remove(truncated_token_key_id).is_some()
    }
}

/// Publicly verifiable Privacy Pass token validator.
///
/// Uses RSA Blind Signatures (RFC 9578, token type 2) with the
/// `privacypass` crate's `OriginServer` for token redemption.
pub struct PrivacyPassValidator {
    origin_server: OriginServer,
    key_store: Arc<MemoryOriginKeyStore>,
    nonce_store: Arc<MemoryNonceStore>,
}

impl PrivacyPassValidator {
    /// Create a new Privacy Pass validator.
    ///
    /// The `key_store` should be pre-populated with the issuer's public key(s).
    pub fn new(key_store: Arc<MemoryOriginKeyStore>, nonce_store: Arc<MemoryNonceStore>) -> Self {
        Self {
            origin_server: OriginServer::new(),
            key_store,
            nonce_store,
        }
    }
}

impl TokenValidator for PrivacyPassValidator {
    async fn validate(
        &self,
        authorization_header: &HeaderValue,
    ) -> Result<AuthContext, ServerError> {
        // Parse the PrivateToken from the Authorization header
        let token: PublicToken = parse_authorization_header(authorization_header).map_err(|e| {
            ServerError::Unauthorized {
                message: format!("invalid PrivateToken: {}", e),
            }
        })?;

        // Redeem the token (verifies signature + checks for double-spending)
        self.origin_server
            .redeem_token(self.key_store.as_ref(), self.nonce_store.as_ref(), token)
            .await
            .map_err(|e| ServerError::Unauthorized {
                message: format!("token redemption failed: {}", e),
            })?;

        Ok(AuthContext {
            method: AuthMethod::PrivacyPass,
        })
    }
}

// ---------------------------------------------------------------------------
// No-op validator (development)
// ---------------------------------------------------------------------------

/// No-op validator that accepts all requests.
pub struct NoopValidator;

impl TokenValidator for NoopValidator {
    async fn validate(
        &self,
        _authorization_header: &HeaderValue,
    ) -> Result<AuthContext, ServerError> {
        Ok(AuthContext {
            method: AuthMethod::None,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: extract and validate from a hyper request
// ---------------------------------------------------------------------------

/// Extract the Authorization header and validate it.
///
/// For `NoopValidator`, missing headers are accepted.
/// For other validators, a missing header returns 401.
pub async fn authenticate(
    req: &hyper::Request<hyper::body::Incoming>,
    validator: &AnyValidator,
) -> Result<AuthContext, ServerError> {
    match req.headers().get(http::header::AUTHORIZATION) {
        Some(header) => validator.validate(header).await,
        None => {
            // NoopValidator accepts missing headers
            if matches!(validator, AnyValidator::Noop(_)) {
                Ok(AuthContext {
                    method: AuthMethod::None,
                })
            } else {
                Err(ServerError::Unauthorized {
                    message: "missing Authorization header".to_string(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_noop_validator() {
        let validator = NoopValidator;
        let header = HeaderValue::from_static("anything");
        let result = validator.validate(&header).await;
        assert!(result.is_ok());
        assert!(matches!(result.unwrap().method, AuthMethod::None));
    }

    #[test]
    fn test_auth_method_linkable() {
        assert!(!AuthMethod::PrivacyPass.linkable());
        assert!(!AuthMethod::None.linkable());
    }
}
