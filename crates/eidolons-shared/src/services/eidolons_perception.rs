//! Perception service stub.
//!
//! The real implementation (on-device inference via eidolons-perception) is
//! temporarily shelved. This module keeps the UniFFI surface intact so the
//! macOS shell compiles, but every method returns a placeholder response.

/// Role of a chat message sender (UniFFI-compatible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum ServiceRole {
    /// Message from the user
    User,
    /// Message from the AI assistant
    Assistant,
}

/// A chat message for the perception service (UniFFI-compatible).
#[derive(Debug, Clone, uniffi::Record)]
pub struct ServiceChatMessage {
    /// The role of the message sender
    pub role: ServiceRole,
    /// The message content
    pub content: String,
}

/// Callback interface for streaming text generation.
///
/// Swift/Kotlin shells implement this trait to receive streaming tokens
/// as they are generated.
#[uniffi::export(callback_interface)]
pub trait StreamingCallback: Send + Sync {
    /// Called when a new chunk of text is generated.
    fn on_chunk(&self, text: String);
    /// Called when generation is complete.
    fn on_complete(&self);
    /// Called when an error occurs during generation.
    fn on_error(&self, error: String);
}

/// Error type for perception service operations.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum PerceptionError {
    #[error("Model not initialized. Call initialize() first.")]
    NotInitialized,
    #[error("Model loading failed: {message}")]
    LoadFailed { message: String },
    #[error("Model already initialized")]
    AlreadyInitialized,
    #[error("Inference failed: {message}")]
    InferenceFailed { message: String },
}

/// Stub perception service.
///
/// Maintains the same UniFFI interface as the real implementation so the
/// macOS shell compiles unchanged. All inference methods return a placeholder
/// message indicating that on-device inference is not yet available.
#[derive(uniffi::Object)]
pub struct PerceptionService {
    initialized: std::sync::atomic::AtomicBool,
}

#[uniffi::export]
impl PerceptionService {
    /// Creates a new perception service stub.
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self {
            initialized: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Stub initialize — always succeeds immediately.
    pub async fn initialize(&self) -> Result<(), PerceptionError> {
        self.initialized
            .store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Returns whether the service has been initialized.
    pub async fn is_ready(&self) -> bool {
        self.initialized
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Returns a placeholder response.
    pub async fn chat(
        &self,
        _messages: Vec<ServiceChatMessage>,
    ) -> Result<String, PerceptionError> {
        Ok("On-device inference is not yet available.".to_string())
    }

    /// Sends a single placeholder chunk and completes.
    pub async fn chat_streaming(
        &self,
        _messages: Vec<ServiceChatMessage>,
        callback: Box<dyn StreamingCallback>,
    ) -> Result<(), PerceptionError> {
        callback.on_chunk("On-device inference is not yet available.".to_string());
        callback.on_complete();
        Ok(())
    }

    /// Returns stub model info.
    pub async fn model_info(&self) -> Result<String, PerceptionError> {
        Ok(r#"{"status": "stub", "inference": "not available"}"#.to_string())
    }
}

impl Default for PerceptionService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_lifecycle() {
        let service = PerceptionService::new();
        assert!(!service.is_ready().await);

        service.initialize().await.unwrap();
        assert!(service.is_ready().await);

        let messages = vec![ServiceChatMessage {
            role: ServiceRole::User,
            content: "test".to_string(),
        }];
        let result = service.chat(messages).await.unwrap();
        assert!(!result.is_empty());
    }
}
