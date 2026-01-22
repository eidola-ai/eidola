use std::sync::{Arc, LazyLock, RwLock};

/// Shared Tokio runtime for all async services (Perception, Memory, etc.)
///
/// This runtime is lazily initialized on first use and provides the async
/// execution context for operations like model downloading and inference.
///
/// The runtime is completely isolated from Crux's synchronous FFI calls.
pub(crate) static TOKIO_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime for async services")
});

/// Error type for perception service operations.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum PerceptionError {
    #[error("Model not initialized. Call initialize() first.")]
    NotInitialized,
    #[error("Model loading failed: {message}")]
    LoadFailed { message: String },
    #[error("Model already initialized")]
    AlreadyInitialized,
}

/// State of the perception service.
enum ServiceState {
    /// Service created but model not loaded.
    Uninitialized,
    /// Model is currently being loaded.
    Loading,
    /// Model loaded and ready for inference.
    Ready(eidolons_perception::TextGenerationModel),
}

/// Service for text generation using ML models.
///
/// This service wraps the perception crate's TextGenerationModel and exposes
/// it via UniFFI for use in Swift/Kotlin shells.
///
/// # Usage
///
/// ```swift
/// let service = PerceptionService()
/// try await service.initialize()
/// let response = try await service.chat(message: "Hello!")
/// ```
#[derive(uniffi::Object)]
pub struct PerceptionService {
    state: Arc<RwLock<ServiceState>>,
}

#[uniffi::export]
impl PerceptionService {
    /// Creates a new uninitialized perception service.
    ///
    /// This is a cheap operation that does not download any model weights.
    /// Call `initialize()` to download and load the model.
    #[uniffi::constructor]
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ServiceState::Uninitialized)),
        }
    }

    /// Initializes the service by downloading and loading the model.
    ///
    /// This is an async operation that may take time depending on network
    /// speed and whether the model is already cached.
    ///
    /// # Errors
    ///
    /// Returns `PerceptionError::AlreadyInitialized` if called twice.
    /// Returns `PerceptionError::LoadFailed` if model download or loading fails.
    pub async fn initialize(&self) -> Result<(), PerceptionError> {
        // Check current state and mark as loading (sync lock, no await needed)
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| PerceptionError::LoadFailed {
                    message: "Lock poisoned".to_string(),
                })?;
            match &*state {
                ServiceState::Ready(_) => return Err(PerceptionError::AlreadyInitialized),
                ServiceState::Loading => return Err(PerceptionError::AlreadyInitialized),
                ServiceState::Uninitialized => {
                    *state = ServiceState::Loading;
                }
            }
        }

        // Load the model on the Tokio runtime
        let model_result = TOKIO_RUNTIME
            .spawn(async { eidolons_perception::TextGenerationModel::load().await })
            .await
            .map_err(|e| PerceptionError::LoadFailed {
                message: format!("Task failed: {e}"),
            })?
            .map_err(|e| PerceptionError::LoadFailed {
                message: e.to_string(),
            })?;

        // Store the loaded model
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| PerceptionError::LoadFailed {
                    message: "Lock poisoned".to_string(),
                })?;
            *state = ServiceState::Ready(model_result);
        }

        Ok(())
    }

    /// Returns whether the model is initialized and ready for inference.
    pub async fn is_ready(&self) -> bool {
        let state = self.state.read().ok();
        state.is_some_and(|s| matches!(&*s, ServiceState::Ready(_)))
    }

    /// Generates a response for the given message.
    ///
    /// # Arguments
    ///
    /// * `message` - The input message/prompt
    ///
    /// # Returns
    ///
    /// The generated response string.
    ///
    /// # Errors
    ///
    /// Returns `PerceptionError::NotInitialized` if `initialize()` hasn't been called.
    pub async fn chat(&self, message: String) -> Result<String, PerceptionError> {
        let state = self
            .state
            .read()
            .map_err(|_| PerceptionError::NotInitialized)?;
        match &*state {
            ServiceState::Ready(model) => Ok(model.generate(&message)),
            ServiceState::Loading => Err(PerceptionError::NotInitialized),
            ServiceState::Uninitialized => Err(PerceptionError::NotInitialized),
        }
    }

    /// Returns model configuration information if initialized.
    ///
    /// Returns a JSON string with model details, or an error if not initialized.
    pub async fn model_info(&self) -> Result<String, PerceptionError> {
        let state = self
            .state
            .read()
            .map_err(|_| PerceptionError::NotInitialized)?;
        match &*state {
            ServiceState::Ready(model) => {
                let config = model.config();
                Ok(format!(
                    "{{\"architectures\": {:?}, \"hidden_size\": {}, \"num_layers\": {}, \"vocab_size\": {}}}",
                    config.architectures,
                    config.hidden_size,
                    config.num_hidden_layers,
                    config.vocab_size
                ))
            }
            _ => Err(PerceptionError::NotInitialized),
        }
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

        // Should not be ready initially
        assert!(!service.is_ready().await);

        // Chat should fail before initialization
        let result = service.chat("test".to_string()).await;
        assert!(matches!(result, Err(PerceptionError::NotInitialized)));
    }
}
