//! Perception service for text generation using ML models.
//!
//! Uses a dedicated inference thread to avoid `Send + Sync` requirements on GPU types.

use eidolons_perception::{ChatRole, FormatMessage, StreamChunk};
use std::sync::{Arc, LazyLock, RwLock, mpsc as std_mpsc};
use tokio::sync::{mpsc, oneshot};

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

/// Shared Tokio runtime for all async services (Perception, Memory, etc.)
///
/// This runtime is lazily initialized on first use and provides the async
/// execution context for operations like model downloading and inference.
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
    #[error("Inference failed: {message}")]
    InferenceFailed { message: String },
}

/// Commands sent to the inference thread.
enum InferenceCommand {
    /// Generate a response for the given conversation.
    Generate {
        messages: Vec<ServiceChatMessage>,
        response_tx: oneshot::Sender<Result<String, String>>,
    },
    /// Generate a response with streaming output.
    GenerateStreaming {
        messages: Vec<ServiceChatMessage>,
        callback: Box<dyn StreamingCallback>,
    },
    /// Get model info.
    ModelInfo {
        response_tx: oneshot::Sender<Result<String, String>>,
    },
}

/// Handle to communicate with the inference thread.
struct InferenceHandle {
    command_tx: mpsc::Sender<InferenceCommand>,
}

impl InferenceHandle {
    /// Spawns a new inference thread that loads the model and processes commands.
    ///
    /// The model is loaded and owned entirely by this thread, avoiding `Send + Sync`
    /// requirements on WGPU types.
    async fn spawn() -> Result<Self, PerceptionError> {
        let (command_tx, mut command_rx) = mpsc::channel::<InferenceCommand>(32);
        let (init_tx, init_rx) = oneshot::channel::<Result<(), String>>();

        // Spawn the inference thread
        std::thread::spawn(move || {
            // Configure environment for macOS app bundles.
            // Sandboxed apps have HOME pointing to the container, but we need
            // the real home for HuggingFace cache and WGPU shader cache.
            if let Some(real_home) = dirs::home_dir() {
                // Set HF_HOME to ensure HuggingFace cache uses real user cache
                let hf_cache = real_home.join(".cache").join("huggingface");
                // SAFETY: Setting HF_HOME at thread startup before any HF operations.
                // This thread is newly spawned and HF_HOME hasn't been read yet.
                unsafe { std::env::set_var("HF_HOME", &hf_cache) };

                // Change to real home directory for WGPU shader cache
                let _ = std::env::set_current_dir(&real_home);
            }

            // Create a thread-local Tokio runtime for async model loading
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create inference thread runtime");

            // Load the model (this is async due to HuggingFace downloads)
            // Wrap in catch_unwind since WGPU initialization may panic
            let model_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rt.block_on(async { eidolons_perception::TextGenerationModel::load().await })
            }));

            let model = match model_result {
                Ok(Ok(m)) => {
                    let _ = init_tx.send(Ok(()));
                    m
                }
                Ok(Err(e)) => {
                    let _ = init_tx.send(Err(e.to_string()));
                    return;
                }
                Err(panic) => {
                    let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "Unknown panic during model loading".to_string()
                    };
                    let _ = init_tx.send(Err(format!("Model loading panicked: {}", msg)));
                    return;
                }
            };

            // Process commands until shutdown
            rt.block_on(async {
                while let Some(cmd) = command_rx.recv().await {
                    match cmd {
                        InferenceCommand::Generate { messages, response_tx } => {
                            // Convert messages to the format expected by the model
                            let format_messages: Vec<FormatMessage<'_>> = messages
                                .iter()
                                .map(|m| FormatMessage {
                                    role: match m.role {
                                        ServiceRole::User => ChatRole::User,
                                        ServiceRole::Assistant => ChatRole::Assistant,
                                    },
                                    content: &m.content,
                                })
                                .collect();

                            // Catch panics during inference
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                model.generate_from_conversation(&format_messages)
                            }));
                            match result {
                                Ok(output) => {
                                    let _ = response_tx.send(Ok(output));
                                }
                                Err(e) => {
                                    let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                        s.to_string()
                                    } else if let Some(s) = e.downcast_ref::<String>() {
                                        s.clone()
                                    } else {
                                        "Unknown panic during inference".to_string()
                                    };
                                    let _ = response_tx.send(Err(format!("Inference panicked: {}", msg)));
                                }
                            }
                        }
                        InferenceCommand::GenerateStreaming { messages, callback } => {
                            // Convert messages to the format expected by the model
                            let format_messages: Vec<FormatMessage<'_>> = messages
                                .iter()
                                .map(|m| FormatMessage {
                                    role: match m.role {
                                        ServiceRole::User => ChatRole::User,
                                        ServiceRole::Assistant => ChatRole::Assistant,
                                    },
                                    content: &m.content,
                                })
                                .collect();

                            // Create channel for streaming chunks
                            let (chunk_tx, chunk_rx) = std_mpsc::channel::<StreamChunk>();

                            // Spawn thread to process chunks CONCURRENTLY with generation.
                            // This ensures callbacks fire as chunks are produced, not after
                            // generation completes.
                            let callback_handle = std::thread::spawn(move || {
                                for chunk in chunk_rx {
                                    match chunk {
                                        StreamChunk::Text(text) => {
                                            callback.on_chunk(text);
                                        }
                                        StreamChunk::Done => {
                                            callback.on_complete();
                                            break;
                                        }
                                        StreamChunk::Error(e) => {
                                            callback.on_error(e);
                                            break;
                                        }
                                    }
                                }
                            });

                            // Run generation (sends chunks while callback thread processes them)
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                model.generate_from_conversation_streaming(
                                    &format_messages,
                                    eidolons_perception::GenerationConfig::default(),
                                    chunk_tx,
                                );
                            }));

                            // Wait for callback thread to finish processing all chunks
                            let _ = callback_handle.join();

                            // Handle generation panic (callback thread may have already
                            // received an error via the channel if generation failed gracefully)
                            if let Err(e) = result {
                                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                    s.to_string()
                                } else if let Some(s) = e.downcast_ref::<String>() {
                                    s.clone()
                                } else {
                                    "Unknown panic during streaming inference".to_string()
                                };
                                // Note: callback thread may have already exited, but we log
                                // the panic for debugging purposes
                                eprintln!("Streaming inference panicked: {}", msg);
                            }
                        }
                        InferenceCommand::ModelInfo { response_tx } => {
                            let config = model.config();
                            let info = format!(
                                "{{\"architectures\": {:?}, \"hidden_size\": {}, \"num_layers\": {}, \"vocab_size\": {}, \"gpu_accelerated\": {}}}",
                                config.architectures,
                                config.hidden_size,
                                config.num_hidden_layers,
                                config.vocab_size,
                                model.is_gpu_accelerated()
                            );
                            let _ = response_tx.send(Ok(info));
                        }
                    }
                }
            });
        });

        // Wait for initialization to complete
        init_rx
            .await
            .map_err(|_| PerceptionError::LoadFailed {
                message: "Inference thread died during initialization".to_string(),
            })?
            .map_err(|e| PerceptionError::LoadFailed { message: e })?;

        Ok(Self { command_tx })
    }
}

/// State of the perception service.
enum ServiceState {
    /// Service created but model not loaded.
    Uninitialized,
    /// Model is currently being loaded.
    Loading,
    /// Model loaded and ready for inference.
    Ready(InferenceHandle),
}

/// Service for text generation using ML models.
///
/// This service wraps the perception crate's TextGenerationModel and exposes
/// it via UniFFI for use in Swift/Kotlin shells.
///
/// The model runs on a dedicated inference thread, allowing GPU-accelerated
/// inference without requiring WGPU types to be `Send + Sync`.
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
    /// This spawns a dedicated inference thread that owns the model,
    /// enabling GPU acceleration without thread-safety constraints.
    ///
    /// # Errors
    ///
    /// Returns `PerceptionError::AlreadyInitialized` if called twice.
    /// Returns `PerceptionError::LoadFailed` if model download or loading fails.
    pub async fn initialize(&self) -> Result<(), PerceptionError> {
        // Check current state and mark as loading
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

        // Spawn inference thread and wait for model to load
        let handle = TOKIO_RUNTIME
            .spawn(InferenceHandle::spawn())
            .await
            .map_err(|e| PerceptionError::LoadFailed {
                message: format!("Task failed: {e}"),
            })??;

        // Store the handle
        {
            let mut state = self
                .state
                .write()
                .map_err(|_| PerceptionError::LoadFailed {
                    message: "Lock poisoned".to_string(),
                })?;
            *state = ServiceState::Ready(handle);
        }

        Ok(())
    }

    /// Returns whether the model is initialized and ready for inference.
    pub async fn is_ready(&self) -> bool {
        let state = self.state.read().ok();
        state.is_some_and(|s| matches!(&*s, ServiceState::Ready(_)))
    }

    /// Generates a response for the given conversation history.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a list of messages
    ///
    /// # Returns
    ///
    /// The generated response string.
    ///
    /// # Errors
    ///
    /// Returns `PerceptionError::NotInitialized` if `initialize()` hasn't been called.
    pub async fn chat(&self, messages: Vec<ServiceChatMessage>) -> Result<String, PerceptionError> {
        let handle = {
            let state = self
                .state
                .read()
                .map_err(|_| PerceptionError::NotInitialized)?;
            match &*state {
                ServiceState::Ready(handle) => handle.command_tx.clone(),
                _ => return Err(PerceptionError::NotInitialized),
            }
        };

        // Send command through the cloned sender
        let (response_tx, response_rx) = oneshot::channel();
        handle
            .send(InferenceCommand::Generate {
                messages,
                response_tx,
            })
            .await
            .map_err(|_| PerceptionError::InferenceFailed {
                message: "Inference thread not responding".to_string(),
            })?;

        response_rx
            .await
            .map_err(|_| PerceptionError::InferenceFailed {
                message: "Inference thread died".to_string(),
            })?
            .map_err(|e| PerceptionError::InferenceFailed { message: e })
    }

    /// Generates a response for the given conversation with streaming output.
    ///
    /// Tokens are streamed through the callback as they are generated,
    /// enabling real-time display in the UI.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a list of messages
    /// * `callback` - Callback to receive streaming chunks
    ///
    /// # Errors
    ///
    /// Returns `PerceptionError::NotInitialized` if `initialize()` hasn't been called.
    pub async fn chat_streaming(
        &self,
        messages: Vec<ServiceChatMessage>,
        callback: Box<dyn StreamingCallback>,
    ) -> Result<(), PerceptionError> {
        let handle = {
            let state = self
                .state
                .read()
                .map_err(|_| PerceptionError::NotInitialized)?;
            match &*state {
                ServiceState::Ready(handle) => handle.command_tx.clone(),
                _ => return Err(PerceptionError::NotInitialized),
            }
        };

        // Send streaming command
        handle
            .send(InferenceCommand::GenerateStreaming { messages, callback })
            .await
            .map_err(|_| PerceptionError::InferenceFailed {
                message: "Inference thread not responding".to_string(),
            })?;

        Ok(())
    }

    /// Returns model configuration information if initialized.
    ///
    /// Returns a JSON string with model details, or an error if not initialized.
    pub async fn model_info(&self) -> Result<String, PerceptionError> {
        let handle = {
            let state = self
                .state
                .read()
                .map_err(|_| PerceptionError::NotInitialized)?;
            match &*state {
                ServiceState::Ready(handle) => handle.command_tx.clone(),
                _ => return Err(PerceptionError::NotInitialized),
            }
        };

        let (response_tx, response_rx) = oneshot::channel();
        handle
            .send(InferenceCommand::ModelInfo { response_tx })
            .await
            .map_err(|_| PerceptionError::InferenceFailed {
                message: "Inference thread not responding".to_string(),
            })?;

        response_rx
            .await
            .map_err(|_| PerceptionError::InferenceFailed {
                message: "Inference thread died".to_string(),
            })?
            .map_err(|e| PerceptionError::InferenceFailed { message: e })
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
        let messages = vec![ServiceChatMessage {
            role: ServiceRole::User,
            content: "test".to_string(),
        }];
        let result = service.chat(messages).await;
        assert!(matches!(result, Err(PerceptionError::NotInitialized)));
    }

    /// Full integration test that initializes the model and runs inference.
    /// This test requires network access and a GPU.
    /// Run with: cargo test --release -p eidolons-shared test_full_inference -- --ignored --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_full_inference() {
        let service = PerceptionService::new();

        // Initialize the model
        println!("Initializing model...");
        service.initialize().await.expect("Model should initialize");

        // Should be ready now
        assert!(
            service.is_ready().await,
            "Service should be ready after init"
        );

        // Get model info
        println!("Getting model info...");
        if let Ok(info) = service.model_info().await {
            println!("Model info: {}", info);
        }

        // Run inference with single message
        println!("Running inference...");
        let messages = vec![ServiceChatMessage {
            role: ServiceRole::User,
            content: "Hello!".to_string(),
        }];
        let response = service.chat(messages).await.expect("Chat should succeed");
        println!("Chat response: {}", response);

        // Run multi-turn inference
        println!("Running multi-turn inference...");
        let messages = vec![
            ServiceChatMessage {
                role: ServiceRole::User,
                content: "What is 2 + 2?".to_string(),
            },
            ServiceChatMessage {
                role: ServiceRole::Assistant,
                content: "2 + 2 equals 4.".to_string(),
            },
            ServiceChatMessage {
                role: ServiceRole::User,
                content: "And what is 3 + 3?".to_string(),
            },
        ];
        let response = service
            .chat(messages)
            .await
            .expect("Multi-turn chat should succeed");
        println!("Multi-turn response: {}", response);
    }
}
