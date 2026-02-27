//! Unified error handling for the Eidolons server.

use hyper::StatusCode;

use crate::types::ErrorResponse;

/// Errors that can occur during request processing.
#[derive(Debug)]
pub enum ServerError {
    /// Client sent a bad request (400).
    BadRequest { message: String },

    /// Authentication failed (401).
    Unauthorized { message: String },

    /// Upstream backend returned an error.
    Backend {
        status: u16,
        error_type: String,
        message: String,
    },

    /// Network error communicating with upstream.
    Network(String),

    /// Failed to parse upstream response.
    Parse(String),

    /// Internal server error (500).
    Internal(String),
}

impl ServerError {
    /// Map this error to an HTTP status code.
    pub fn status_code(&self) -> StatusCode {
        match self {
            ServerError::BadRequest { .. } => StatusCode::BAD_REQUEST,
            ServerError::Unauthorized { .. } => StatusCode::UNAUTHORIZED,
            ServerError::Backend { status, .. } => {
                StatusCode::from_u16(*status).unwrap_or(StatusCode::BAD_GATEWAY)
            }
            ServerError::Network(_) | ServerError::Parse(_) => StatusCode::BAD_GATEWAY,
            ServerError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// Convert to an OpenAI-format error response for the wire.
    pub fn to_error_response(&self) -> ErrorResponse {
        match self {
            ServerError::BadRequest { message } => {
                ErrorResponse::new(message, "invalid_request_error")
            }
            ServerError::Unauthorized { message } => {
                ErrorResponse::new(message, "authentication_error")
            }
            ServerError::Backend {
                error_type,
                message,
                ..
            } => ErrorResponse::new(message, error_type),
            ServerError::Network(msg) => ErrorResponse::new(msg, "upstream_error"),
            ServerError::Parse(msg) => ErrorResponse::new(msg, "upstream_error"),
            ServerError::Internal(msg) => ErrorResponse::new(msg, "internal_error"),
        }
    }
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::BadRequest { message } => write!(f, "bad request: {}", message),
            ServerError::Unauthorized { message } => write!(f, "unauthorized: {}", message),
            ServerError::Backend {
                status,
                error_type,
                message,
            } => write!(f, "backend error ({}): {}: {}", status, error_type, message),
            ServerError::Network(msg) => write!(f, "network error: {}", msg),
            ServerError::Parse(msg) => write!(f, "parse error: {}", msg),
            ServerError::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for ServerError {}
