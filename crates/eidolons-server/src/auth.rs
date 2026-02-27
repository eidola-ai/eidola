//! Authorization middleware for the Eidolons server.
//!
//! Supports Anonymous Credit Tokens (Privacy Pass ACT) for privacy-preserving
//! rate-limited authorization.

use http::HeaderValue;
use serde::Serialize;
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
    /// Anonymous Credit Token (Privacy Pass ACT).
    AnonymousCreditToken,
    /// No authentication (development only).
    None,
}

impl AuthMethod {
    /// Whether requests using this method are linkable across requests.
    pub fn linkable(&self) -> bool {
        match self {
            AuthMethod::AnonymousCreditToken => false,
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
        assert!(!AuthMethod::AnonymousCreditToken.linkable());
        assert!(!AuthMethod::None.linkable());
    }
}
