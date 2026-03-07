//! Authorization middleware for the Eidolons server.
//!
//! Supports the PrivateToken authentication scheme with ACT (Anonymous Credit
//! Token) spend proofs, following draft-schlesinger-privacypass-act-01.

use anonymous_credit_tokens::SpendProof;
use axum::extract::FromRequestParts;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Serialize;
use utoipa::ToSchema;

use crate::AppState;
use crate::credentials::ACT_TOKEN_TYPE;
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
    /// Anonymous credential for privacy-preserving usage.
    AnonymousCredential,
    /// No authentication (development only).
    None,
}

impl AuthMethod {
    /// Whether requests using this method are linkable across requests.
    pub fn linkable(&self) -> bool {
        match self {
            AuthMethod::AnonymousCredential => false,
            AuthMethod::None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// ACT spend data extracted from the Authorization header
// ---------------------------------------------------------------------------

/// Parsed ACT Token from an `Authorization: PrivateToken token="..."` header.
///
/// Token structure (draft-schlesinger-privacypass-act-01, Section 9.1):
/// ```text
/// struct {
///     uint16_t token_type = 0xE5AD;
///     uint8_t challenge_digest[32];
///     uint8_t issuer_key_id[32];   // SHA-256(pkI_serialized)
///     uint8_t encoded_spend_proof[...];
/// } Token;
/// ```
pub struct ActSpend {
    /// SHA-256(TokenChallenge) — binds the token to the challenge context.
    pub challenge_digest: [u8; 32],
    /// SHA-256(pkI_serialized) — identifies which issuer key signed the credential.
    pub issuer_key_hash: [u8; 32],
    /// The zero-knowledge spend proof.
    pub spend_proof: SpendProof<128>,
}

// ---------------------------------------------------------------------------
// TokenAuth extractor (for chat completions / metered endpoints)
// ---------------------------------------------------------------------------

/// Axum extractor that parses an ACT Token from the Authorization header.
///
/// Expected format: `Authorization: PrivateToken token="<base64url(Token)>"`
///
/// Per RFC 9577 Section 2.2.2 and draft-schlesinger-privacypass-act-01 Section 9.1.
pub struct TokenAuth(pub ActSpend);

impl FromRequestParts<AppState> for TokenAuth {
    type Rejection = ServerError;

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        let header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| ServerError::Unauthorized {
                message: "missing Authorization header".to_string(),
            })?;

        // Parse: PrivateToken token="<base64url>"
        let payload = header
            .strip_prefix("PrivateToken token=\"")
            .and_then(|s| s.strip_suffix('"'))
            .ok_or_else(|| ServerError::Unauthorized {
                message: "expected PrivateToken authorization scheme".to_string(),
            })?;

        let token_bytes = URL_SAFE_NO_PAD
            .decode(payload)
            .map_err(|_| ServerError::Unauthorized {
                message: "invalid base64url in PrivateToken".to_string(),
            })?;

        // Minimum size: 2 (token_type) + 32 (challenge_digest) + 32 (issuer_key_id) + 1 (spend proof)
        if token_bytes.len() < 67 {
            return Err(ServerError::Unauthorized {
                message: "token too short".to_string(),
            });
        }

        // Parse token_type (2 bytes, big-endian)
        let token_type = u16::from_be_bytes([token_bytes[0], token_bytes[1]]);
        if token_type != ACT_TOKEN_TYPE {
            return Err(ServerError::Unauthorized {
                message: format!(
                    "unsupported token type: 0x{:04X}, expected 0x{:04X}",
                    token_type, ACT_TOKEN_TYPE
                ),
            });
        }

        // Parse challenge_digest (32 bytes)
        let mut challenge_digest = [0u8; 32];
        challenge_digest.copy_from_slice(&token_bytes[2..34]);

        // Parse issuer_key_id (32 bytes)
        let mut issuer_key_hash = [0u8; 32];
        issuer_key_hash.copy_from_slice(&token_bytes[34..66]);

        // Parse spend proof (remaining bytes are CBOR)
        let spend_proof =
            SpendProof::<128>::from_cbor(&token_bytes[66..]).map_err(|_| {
                ServerError::Unauthorized {
                    message: "invalid CBOR spend proof in token".to_string(),
                }
            })?;

        Ok(TokenAuth(ActSpend {
            challenge_digest,
            issuer_key_hash,
            spend_proof,
        }))
    }
}

// ---------------------------------------------------------------------------
// BasicAuth extractor (for account endpoints)
// ---------------------------------------------------------------------------

/// Axum extractor that validates HTTP Basic auth against the account table.
///
/// The username is the account UUID, and the password is the credential secret.
pub struct BasicAuth(pub uuid::Uuid);

impl FromRequestParts<AppState> for BasicAuth {
    type Rejection = ServerError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        use argon2::Argon2;
        use argon2::password_hash::{PasswordHash, PasswordVerifier};

        let header = parts
            .headers
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| ServerError::Unauthorized {
                message: "missing authorization header".to_string(),
            })?;

        let encoded = header
            .strip_prefix("Basic ")
            .ok_or_else(|| ServerError::Unauthorized {
                message: "expected Basic auth".to_string(),
            })?;

        let decoded = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| ServerError::Unauthorized {
                message: "invalid base64 in authorization header".to_string(),
            })?;

        let decoded_str = String::from_utf8(decoded).map_err(|_| ServerError::Unauthorized {
            message: "invalid utf-8 in authorization header".to_string(),
        })?;

        let (account_id_str, secret) =
            decoded_str
                .split_once(':')
                .ok_or_else(|| ServerError::Unauthorized {
                    message: "invalid Basic auth format".to_string(),
                })?;

        let account_id =
            uuid::Uuid::parse_str(account_id_str).map_err(|_| ServerError::Unauthorized {
                message: "invalid account_id".to_string(),
            })?;

        let account = crate::db::get_account_by_id(&state.db_pool, account_id)
            .await
            .map_err(|e| match e {
                ServerError::NotFound { .. } => ServerError::Unauthorized {
                    message: "invalid credentials".to_string(),
                },
                other => other,
            })?;

        let parsed_hash = PasswordHash::new(&account.secret_hash)
            .map_err(|_| ServerError::Internal("corrupt credential hash".to_string()))?;

        Argon2::default()
            .verify_password(secret.as_bytes(), &parsed_hash)
            .map_err(|_| ServerError::Unauthorized {
                message: "invalid credentials".to_string(),
            })?;

        Ok(BasicAuth(account_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_method_linkable() {
        assert!(!AuthMethod::AnonymousCredential.linkable());
        assert!(!AuthMethod::None.linkable());
    }
}
