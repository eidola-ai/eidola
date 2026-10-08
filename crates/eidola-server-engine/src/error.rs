//! Request errors.
//!
//! An [`ApiError`] has two renderings. The response body (`to_body`) goes only to the
//! caller, over its own connection, and may describe what was wrong with the request.
//! `Display` is what reaches the logs, and it is the variant's fixed category alone: no
//! message here may interpolate anything derived from a request (a prompt, a token count,
//! a cache key, a serde message quoting a value). The same rule as the server's
//! `ServerError`, made structural: the only variants that carry text carry it for the
//! response, and `Display` never prints it.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Why a request was refused or failed.
#[derive(Debug)]
pub enum ApiError {
    /// No or wrong gateway token (401).
    Unauthorized,
    /// No `X-Eidola-Weights-Sha256` header (428).
    WeightsHashRequired,
    /// The header names other weights than this node serves (412). Refused before the
    /// body is read.
    WeightsHashMismatch,
    /// A model other than the one this node serves (404).
    ModelNotFound,
    /// The body is malformed or outside the accepted subset (400).
    InvalidRequest(String),
    /// The prompt does not fit the model length or the node's KV memory (400).
    ContextLengthExceeded(String),
    /// The body exceeds the size limit (413).
    PayloadTooLarge,
    /// Every admission slot is taken (503); retry elsewhere or later.
    Overloaded,
    /// The engine has stopped (503).
    Unavailable,
    /// A failure on this side (500). The text is authored here and content-free.
    Internal(&'static str),
}

impl ApiError {
    pub fn invalid(message: impl Into<String>) -> Self {
        ApiError::InvalidRequest(message.into())
    }

    /// A strict-deserialization failure. serde's message quotes offending values, which
    /// is fine for the caller (it sent them) and is why `Display` never prints it.
    pub fn from_serde(e: serde_json::Error) -> Self {
        ApiError::InvalidRequest(format!("invalid request body: {e}"))
    }

    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::WeightsHashRequired => StatusCode::PRECONDITION_REQUIRED,
            ApiError::WeightsHashMismatch => StatusCode::PRECONDITION_FAILED,
            ApiError::ModelNotFound => StatusCode::NOT_FOUND,
            ApiError::InvalidRequest(_) | ApiError::ContextLengthExceeded(_) => {
                StatusCode::BAD_REQUEST
            }
            ApiError::PayloadTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            ApiError::Overloaded | ApiError::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// The OpenAI `error.type`, spelled by `eidola_common::engine_protocol::error_type`.
    pub fn error_type(&self) -> &'static str {
        use eidola_common::engine_protocol::error_type as t;
        match self {
            ApiError::Unauthorized => t::AUTHENTICATION_ERROR,
            ApiError::WeightsHashRequired => t::WEIGHTS_HASH_REQUIRED,
            ApiError::WeightsHashMismatch => t::WEIGHTS_HASH_MISMATCH,
            ApiError::ModelNotFound => t::MODEL_NOT_FOUND,
            ApiError::InvalidRequest(_) => t::INVALID_REQUEST,
            ApiError::ContextLengthExceeded(_) => t::CONTEXT_LENGTH_EXCEEDED,
            ApiError::PayloadTooLarge => t::REQUEST_TOO_LARGE,
            ApiError::Overloaded => t::OVERLOADED,
            ApiError::Unavailable => t::ENGINE_UNAVAILABLE,
            ApiError::Internal(_) => t::INTERNAL_ERROR,
        }
    }

    fn message(&self) -> String {
        match self {
            ApiError::Unauthorized => "missing or invalid gateway token".into(),
            ApiError::WeightsHashRequired => {
                "X-Eidola-Weights-Sha256 is required on every chat request".into()
            }
            ApiError::WeightsHashMismatch => {
                "X-Eidola-Weights-Sha256 does not match the weights this node serves".into()
            }
            ApiError::ModelNotFound => "this node does not serve that model".into(),
            ApiError::InvalidRequest(m) | ApiError::ContextLengthExceeded(m) => m.clone(),
            ApiError::PayloadTooLarge => "request body too large".into(),
            ApiError::Overloaded => "the node is at capacity; retry later".into(),
            ApiError::Unavailable => "the engine is not running".into(),
            ApiError::Internal(m) => (*m).into(),
        }
    }

    /// The response body: `{"error": {"message", "type", "code"}}`.
    pub fn to_body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message(),
                "type": self.error_type(),
                "code": serde_json::Value::Null,
            }
        })
    }
}

/// The log-safe rendering: the category only.
impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Internal(m) => write!(f, "internal_error: {m}"),
            other => f.write_str(other.error_type()),
        }
    }
}

impl std::error::Error for ApiError {}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        if status.is_server_error() {
            tracing::warn!(status = status.as_u16(), "request failed: {self}");
        } else {
            tracing::debug!(status = status.as_u16(), "request refused: {self}");
        }
        (status, axum::Json(self.to_body())).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gateway sends a refused request to another node only when
    /// `refused_before_admission` says so, so exactly the refusals raised
    /// before anything is admitted may say so: the token and weights checks
    /// (before the body is read), the model check (before admission), and
    /// `Overloaded` (a read or admission slot not taken). `Unavailable` can
    /// come after the engine ran the request, and must not.
    #[test]
    fn only_pre_admission_refusals_say_so() {
        use eidola_common::engine_protocol::{PreAdmission, refused_before_admission};
        let cases = [
            (ApiError::Unauthorized, Some(PreAdmission::Misconfigured)),
            (
                ApiError::WeightsHashRequired,
                Some(PreAdmission::Misconfigured),
            ),
            (
                ApiError::WeightsHashMismatch,
                Some(PreAdmission::Misconfigured),
            ),
            (ApiError::ModelNotFound, Some(PreAdmission::Misconfigured)),
            (ApiError::Overloaded, Some(PreAdmission::Overloaded)),
            (ApiError::Unavailable, None),
            (ApiError::invalid("x"), None),
            (ApiError::ContextLengthExceeded("x".into()), None),
            (ApiError::PayloadTooLarge, None),
            (ApiError::Internal("x"), None),
        ];
        for (e, expected) in cases {
            assert_eq!(
                refused_before_admission(e.status().as_u16(), e.error_type()),
                expected,
                "{e}"
            );
        }
    }

    #[test]
    fn display_never_carries_the_response_message() {
        let e =
            ApiError::from_serde(serde_json::from_str::<u32>("\"my secret prompt\"").unwrap_err());
        assert_eq!(e.to_string(), "invalid_request_error");
        assert!(e.to_body().to_string().contains("my secret prompt"));
        let e = ApiError::ContextLengthExceeded("prompt is 9001 tokens".into());
        assert_eq!(e.to_string(), "context_length_exceeded");
    }
}
