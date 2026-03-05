//! Authorization middleware for the Eidolons server.
//!
//! Supports anonymous credentials for privacy-preserving rate-limited
//! authorization.

use axum::extract::FromRequestParts;
use axum::http::HeaderValue;
use axum::http::header::AUTHORIZATION;
use axum::http::request::Parts;
use serde::Serialize;
use utoipa::ToSchema;

use base64::Engine;

use crate::AppState;
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
// Enum dispatch (avoids dyn on the hot path)
// ---------------------------------------------------------------------------

/// Runtime-selected token validator.
pub enum AnyValidator {
    Noop(NoopValidator),
}

impl AnyValidator {
    pub async fn validate(
        &self,
        authorization_header: &HeaderValue,
    ) -> Result<AuthContext, ServerError> {
        match self {
            AnyValidator::Noop(v) => v.validate(authorization_header).await,
        }
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
// TokenAuth extractor (for chat completions / metered endpoints)
// ---------------------------------------------------------------------------

/// Axum extractor that validates token-based auth (ACT / noop).
pub struct TokenAuth(pub AuthContext);

impl FromRequestParts<AppState> for TokenAuth {
    type Rejection = ServerError;

    async fn from_request_parts(
        parts: &mut Parts,
        state: &AppState,
    ) -> Result<Self, Self::Rejection> {
        match parts.headers.get(AUTHORIZATION) {
            Some(header) => {
                let ctx = state.validator.validate(header).await?;
                Ok(TokenAuth(ctx))
            }
            None => {
                if matches!(state.validator, AnyValidator::Noop(_)) {
                    Ok(TokenAuth(AuthContext {
                        method: AuthMethod::None,
                    }))
                } else {
                    Err(ServerError::Unauthorized {
                        message: "missing Authorization header".to_string(),
                    })
                }
            }
        }
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
        assert!(!AuthMethod::AnonymousCredential.linkable());
        assert!(!AuthMethod::None.linkable());
    }
}
