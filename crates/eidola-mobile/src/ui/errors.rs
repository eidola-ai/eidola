//! Typed errors become words here, at render: state holds the `AppError`, and
//! this chooses what the reader is told in their locale (the desktop app's
//! `SpaceView::error_copy` rule, with its translations). An error this layer
//! has no words of its own for shows the error's own text inside a localized
//! sentence — app-core ships no user-facing strings, so that text is English.

use eidola_app_core::error::{AppError, IssuerKeyRefusal};
use gpui::{App, SharedString};

use super::i18n::msg;

/// What the conversation's recovery notice says about a failed request.
pub fn error_copy(e: &AppError, cx: &App) -> SharedString {
    match e {
        AppError::NoAccount => msg::conversation_error_no_account(cx),
        AppError::InsufficientBalance { .. } => msg::conversation_error_insufficient_balance(cx),
        // Not a balance problem: adding credit would not lift it.
        AppError::HoldExceedsDenomination { .. } => {
            msg::conversation_error_hold_exceeds_denomination(cx)
        }
        AppError::SpaceArchived { .. } => msg::conversation_error_archived(cx),
        AppError::UncachedRetryUnfunded => msg::conversation_error_uncached_retry_unfunded(cx),
        AppError::NotJoined { .. } => msg::conversation_error_not_joined(cx),
        // Names the model, and says nothing answered in its place.
        AppError::ModelUnavailable { model } => {
            msg::conversation_error_model_unavailable(cx, model.clone())
        }
        AppError::IssuerKeyRefused { refusal } => match refusal {
            IssuerKeyRefusal::NotOnCalendar => msg::conversation_error_issuer_key_off_calendar(cx),
            IssuerKeyRefusal::SecondKeyForEpoch { .. } => {
                msg::conversation_error_issuer_key_second_for_epoch(cx)
            }
            IssuerKeyRefusal::OneKeyForSeveralEpochs { .. } => {
                msg::conversation_error_issuer_key_reused(cx)
            }
        },
        other => msg::conversation_error_other(cx, other.to_string()),
    }
}

/// Whether a Retry could ever succeed: an archived conversation stays
/// archived, so offering one would only meet the same refusal.
pub fn is_retryable(e: &AppError) -> bool {
    !matches!(e, AppError::SpaceArchived { .. })
}
