//! Unified error handling for the Eidola server.

use axum::http::StatusCode;
use tracing::{error, warn};

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

    /// Resource not found (404).
    NotFound { message: String },

    /// Insufficient credit balance (402).
    PaymentRequired { message: String, available: i64 },

    /// Conflict with existing resource (409).
    Conflict { message: String },

    /// The account has not accepted the currently required terms of
    /// service / privacy policy versions (428). Clients fetch the required
    /// documents from `GET /v1/terms` and record acceptance via
    /// `POST /v1/account/terms`.
    TermsAcceptanceRequired { message: String },

    /// Service unavailable (503).
    ServiceUnavailable(String),

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
            ServerError::PaymentRequired { .. } => StatusCode::PAYMENT_REQUIRED,
            ServerError::NotFound { .. } => StatusCode::NOT_FOUND,
            ServerError::Conflict { .. } => StatusCode::CONFLICT,
            ServerError::TermsAcceptanceRequired { .. } => StatusCode::PRECONDITION_REQUIRED,
            ServerError::Network(_) | ServerError::Parse(_) => StatusCode::BAD_GATEWAY,
            ServerError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
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
            ServerError::PaymentRequired { message, .. } => {
                ErrorResponse::new(message, "insufficient_balance")
            }
            ServerError::Backend {
                error_type,
                message,
                ..
            } => ErrorResponse::new(message, error_type),
            ServerError::NotFound { message } => ErrorResponse::new(message, "not_found"),
            ServerError::Conflict { message } => ErrorResponse::new(message, "conflict"),
            ServerError::TermsAcceptanceRequired { message } => {
                ErrorResponse::new(message, "terms_acceptance_required")
            }
            ServerError::Network(msg) => ErrorResponse::new(msg, "upstream_error"),
            ServerError::Parse(msg) => ErrorResponse::new(msg, "upstream_error"),
            ServerError::ServiceUnavailable(msg) => ErrorResponse::new(msg, "service_unavailable"),
            ServerError::Internal(msg) => ErrorResponse::new(msg, "internal_error"),
        }
    }
}

/// The **log-safe** rendering of an error.
///
/// A `ServerError` has two renderings and they are deliberately not the same.
/// [`to_error_response`](ServerError::to_error_response) is the full detail,
/// and it goes only to the client that made the request — its own data, over
/// its own attested connection. `Display` is the rendering that reaches
/// stdout and the OTLP exporter, so it omits every field whose value is
/// derived from request content or authored by the upstream:
///
/// - `PaymentRequired` carries the worst-case charge, which is a
///   deterministic function of the prompt's chargeable bytes. Logging it
///   would publish a prompt-length estimate at request granularity, which
///   `docs/privacy-guarantees.md` §3.2 rules out.
/// - `Backend` carries the upstream's own error string. That string is
///   outside our control and can quote token counts or fragments of the
///   request, so only the status and a fixed-list resolution of the
///   upstream's error-type classifier are logged — the classifier is
///   itself upstream-authored free text, so anything outside
///   [`KNOWN_UPSTREAM_ERROR_TYPES`] collapses to `other`.
///
/// Every other variant's message is authored by this crate from constants (or
/// from an error that describes a *shape* failure, not a value), so it is
/// logged in full. Keep it that way: a message that interpolates anything
/// request-derived belongs in the redacted set above, not here.
impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::BadRequest { message } => write!(f, "bad request: {}", message),
            ServerError::Unauthorized { message } => write!(f, "unauthorized: {}", message),
            ServerError::Backend {
                status, error_type, ..
            } => write!(
                f,
                "backend error ({}): {}",
                status,
                upstream_error_type_label(error_type)
            ),
            ServerError::PaymentRequired { .. } => write!(f, "payment required"),
            ServerError::NotFound { message } => write!(f, "not found: {}", message),
            ServerError::Conflict { message } => write!(f, "conflict: {}", message),
            ServerError::TermsAcceptanceRequired { message } => {
                write!(f, "terms acceptance required: {}", message)
            }
            ServerError::Network(msg) => write!(f, "network error: {}", msg),
            ServerError::Parse(msg) => write!(f, "parse error: {}", msg),
            ServerError::ServiceUnavailable(msg) => write!(f, "service unavailable: {}", msg),
            ServerError::Internal(msg) => write!(f, "internal error: {}", msg),
        }
    }
}

impl std::error::Error for ServerError {}

/// Upstream error-type classifiers the log path recognizes. `error_type`
/// arrives as upstream-authored free text, so the `Display` rendering
/// resolves it against this list and collapses anything unrecognized to
/// `other` — the client response keeps the verbatim value. `unknown` is
/// this crate's own bucket for upstream error bodies that didn't parse.
const KNOWN_UPSTREAM_ERROR_TYPES: &[&str] = &[
    "authentication_error",
    "insufficient_quota",
    "invalid_request_error",
    "not_found_error",
    "permission_error",
    "rate_limit_error",
    "server_error",
    "unknown",
];

fn upstream_error_type_label(error_type: &str) -> &'static str {
    KNOWN_UPSTREAM_ERROR_TYPES
        .iter()
        .copied()
        .find(|known| *known == error_type)
        .unwrap_or("other")
}

/// Log-safe summary of a `serde_json` error.
///
/// serde's own messages echo offending values verbatim — `invalid type:
/// string "<the value>", expected u32`, ``unknown field `<the name>` `` —
/// so nothing a serde error `Display`s may reach the log path. Only the
/// error category and position survive.
pub(crate) fn parse_error_summary(e: &serde_json::Error) -> String {
    format!(
        "{:?} error at line {} column {}",
        e.classify(),
        e.line(),
        e.column()
    )
}

impl axum::response::IntoResponse for ServerError {
    fn into_response(self) -> axum::response::Response {
        let status = self.status_code();
        if status.is_server_error() {
            error!(status = status.as_u16(), "{self}");
        } else if status.is_client_error() {
            warn!(status = status.as_u16(), "{self}");
        }
        let body = self.to_error_response();
        (status, axum::Json(body)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The charge operands are a function of the prompt's chargeable bytes.
    /// They belong to the client and must not reach the logs.
    #[test]
    fn payment_required_display_omits_charge_operands() {
        let err = ServerError::PaymentRequired {
            message: "insufficient charge: 120 credits provided, 4096 required (worst case)"
                .to_string(),
            available: 120,
        };
        let logged = err.to_string();
        assert_eq!(logged, "payment required");
        assert!(
            !logged.contains("4096"),
            "worst-case charge leaked: {logged}"
        );
        assert!(!logged.contains("120"), "presented charge leaked: {logged}");

        // The client still gets the full detail on its own connection.
        assert!(err.to_error_response().error.message.contains("4096"));
    }

    /// Upstream error strings are outside our control and can quote token
    /// counts or request fragments. Only status + classifier are logged.
    #[test]
    fn backend_display_omits_upstream_message() {
        let err = ServerError::Backend {
            status: 400,
            error_type: "invalid_request_error".to_string(),
            message: "This model's maximum context length is 8192 tokens, however you \
                      requested 9001 tokens (8501 in the messages)"
                .to_string(),
        };
        let logged = err.to_string();
        assert_eq!(logged, "backend error (400): invalid_request_error");
        assert!(
            !logged.contains("8501"),
            "prompt token count leaked: {logged}"
        );
        assert!(
            !logged.contains("context length"),
            "upstream body leaked: {logged}"
        );

        assert!(err.to_error_response().error.message.contains("8501"));
    }

    /// `error_type` is upstream-authored free text; an unrecognized value
    /// must collapse to the shared bucket instead of reaching the logs.
    #[test]
    fn backend_display_buckets_unrecognized_error_type() {
        let err = ServerError::Backend {
            status: 400,
            error_type: "you requested 9001 tokens".to_string(),
            message: "irrelevant".to_string(),
        };
        assert_eq!(err.to_string(), "backend error (400): other");

        // The client still receives the verbatim classifier.
        assert_eq!(
            err.to_error_response().error.error_type,
            "you requested 9001 tokens"
        );
    }

    /// serde error messages quote offending values; the log-safe summary
    /// must carry only the category and position.
    #[test]
    fn parse_error_summary_omits_offending_values() {
        let err = serde_json::from_str::<u32>("\"sk-secret-value\"").unwrap_err();
        let summary = parse_error_summary(&err);
        assert!(
            !summary.contains("secret"),
            "offending value leaked: {summary}"
        );
        assert_eq!(summary, "Data error at line 1 column 17");
    }

    /// Server-authored messages stay in the logs — redaction is targeted, not
    /// blanket, so operators keep the diagnostics that carry no request data.
    #[test]
    fn server_authored_messages_are_still_logged() {
        let err = ServerError::Conflict {
            message: "credential already spent (duplicate nullifier)".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "conflict: credential already spent (duplicate nullifier)"
        );
    }
}
