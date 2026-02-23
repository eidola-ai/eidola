//! Model manager for text generation models.
//!
//! Handles downloading, loading, and running inference with Llama and Qwen3 models.

use crate::NdArrayBackend;
#[cfg(feature = "gpu")]
use crate::WgpuBackend;
use crate::generation::{GenerationConfig, StreamToken, generate, generate_streaming};
use crate::llama::{Llama, LlamaConfig};
use crate::qwen3::{GenerationEvent, GenerationParams, QuantizationMode, Qwen3Model, Sampler};
use crate::tokenizer::{Qwen3Tokenizer, TinyLlamaTokenizer, load_qwen3_tokenizer, load_tokenizer};
use crate::weights::load_llama_from_safetensors;
use anyhow::{Context, Result};
use burn_ndarray::NdArrayDevice;
#[cfg(feature = "gpu")]
use burn_wgpu::WgpuDevice;
use hf_hub::api::tokio::Api;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Once;
use std::sync::mpsc;

/// Ensures the TLS crypto provider is installed exactly once.
static CRYPTO_PROVIDER_INIT: Once = Once::new();

/// Installs the pure-Rust crypto provider for TLS connections.
/// This must be called before any TLS operations (e.g., HTTPS requests).
fn ensure_crypto_provider() {
    CRYPTO_PROVIDER_INIT.call_once(|| {
        rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
            .expect("failed to install rustls crypto provider");
    });
}

/// Configuration loaded from a model's config.json file.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Model architecture type (e.g., "LlamaForCausalLM").
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Hidden size dimension.
    #[serde(default)]
    pub hidden_size: usize,
    /// Number of attention heads.
    #[serde(default)]
    pub num_attention_heads: usize,
    /// Number of hidden layers.
    #[serde(default)]
    pub num_hidden_layers: usize,
    /// Vocabulary size.
    #[serde(default)]
    pub vocab_size: usize,
    /// Additional fields we don't explicitly handle.
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Backend selection for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InferenceBackend {
    /// Apple Silicon native acceleration via MLX backend.
    /// Only available with the `mlx` feature.
    #[cfg(feature = "mlx")]
    Mlx,
    /// GPU-accelerated inference (Metal on macOS, Vulkan elsewhere).
    /// Only available with the `gpu` feature.
    #[cfg(feature = "gpu")]
    Wgpu,
    /// CPU inference using ndarray.
    #[default]
    NdArray,
}

/// Supported model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    /// Llama-family models (TinyLlama, Llama 2, etc.)
    Llama,
    /// Qwen3 models with QK-Norm
    Qwen3,
}

/// Chunk emitted during streaming text generation.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A chunk of decoded text.
    Text(String),
    /// Generation is complete.
    Done,
    /// An error occurred during generation.
    Error(String),
}

/// Internal model representation that can use either backend and architecture.
///
/// Qwen3 variants use `RefCell` because qwen3-burn's `generate` methods take
/// `&mut self` for KV cache mutation. Since the model runs on a single inference
/// thread (with a Mutex at the service layer), `RefCell` is safe here.
enum LoadedModel {
    #[cfg(feature = "gpu")]
    LlamaWgpu {
        model: Box<Llama<WgpuBackend>>,
        device: WgpuDevice,
    },
    LlamaNdArray {
        model: Box<Llama<NdArrayBackend>>,
        device: NdArrayDevice,
    },
    #[cfg(feature = "gpu")]
    Qwen3Wgpu {
        model: Box<RefCell<Qwen3Model<WgpuBackend>>>,
    },
    Qwen3NdArray {
        model: Box<RefCell<Qwen3Model<NdArrayBackend>>>,
    },
}

/// Tokenizer that can handle multiple model types.
enum LoadedTokenizer {
    TinyLlama(Box<TinyLlamaTokenizer>),
    Qwen3(Box<Qwen3Tokenizer>),
}

/// Represents a loaded text generation model.
///
/// Supports both GPU (WGPU) and CPU (NdArray) backends for inference.
/// Supports Llama and Qwen3 architectures.
/// Thread-safety for FFI is handled by wrapping in a Mutex at the service layer.
pub struct TextGenerationModel {
    /// The model configuration from HuggingFace.
    config: ModelConfig,
    /// The detected architecture.
    architecture: ModelArchitecture,
    /// The Llama configuration (if Llama architecture).
    llama_config: Option<LlamaConfig>,
    /// Path to the cached model weights.
    weights_path: PathBuf,
    /// The tokenizer for encoding/decoding text.
    tokenizer: LoadedTokenizer,
    /// The loaded model (either WGPU or NdArray backend, Llama or Qwen3).
    model: LoadedModel,
}

// Manual Debug impl since LoadedModel contains non-Debug types
impl std::fmt::Debug for TextGenerationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend_name = match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::LlamaWgpu { .. } | LoadedModel::Qwen3Wgpu { .. } => "Wgpu",
            LoadedModel::LlamaNdArray { .. } | LoadedModel::Qwen3NdArray { .. } => "NdArray",
        };
        f.debug_struct("TextGenerationModel")
            .field("config", &self.config)
            .field("architecture", &self.architecture)
            .field("weights_path", &self.weights_path)
            .field("backend", &backend_name)
            .finish()
    }
}

impl TextGenerationModel {
    /// The default model repository to use.
    /// Using Qwen3-0.6B as a reasonable default - small enough for CPU inference
    /// while still having the QK-Norm architecture.
    pub const DEFAULT_REPO: &'static str = "Qwen/Qwen3-0.6B";

    /// Loads a text generation model from HuggingFace Hub.
    ///
    /// With the `gpu` feature enabled, attempts GPU acceleration (WGPU) first,
    /// falling back to CPU (NdArray) if unavailable. Without `gpu`, uses CPU only.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Network request fails
    /// - Model files cannot be downloaded
    /// - Configuration cannot be parsed
    pub async fn load() -> Result<Self> {
        Self::load_from_repo(Self::DEFAULT_REPO).await
    }

    /// Loads a text generation model from a specific HuggingFace repository.
    ///
    /// With `gpu` feature: tries GPU first, falls back to CPU.
    /// Without `gpu` feature: uses CPU only.
    pub async fn load_from_repo(repo_id: &str) -> Result<Self> {
        #[cfg(feature = "mlx")]
        {
            match Self::load_from_repo_with_backend(repo_id, InferenceBackend::Mlx).await {
                Ok(model) => return Ok(model),
                Err(e) => {
                    eprintln!("MLX backend failed: {e}. Falling back.");
                }
            }
        }
        #[cfg(feature = "gpu")]
        {
            match Self::load_from_repo_with_backend(repo_id, InferenceBackend::Wgpu).await {
                Ok(model) => return Ok(model),
                Err(e) => {
                    eprintln!("WGPU backend failed: {e}. Falling back to CPU.");
                }
            }
        }
        Self::load_from_repo_with_backend(repo_id, InferenceBackend::NdArray).await
    }

    /// Loads a text generation model with a specific backend.
    pub async fn load_with_backend(backend: InferenceBackend) -> Result<Self> {
        Self::load_from_repo_with_backend(Self::DEFAULT_REPO, backend).await
    }

    /// Loads a text generation model from a specific repository with a specific backend.
    pub async fn load_from_repo_with_backend(
        repo_id: &str,
        backend: InferenceBackend,
    ) -> Result<Self> {
        // Ensure TLS crypto provider is installed before making HTTPS requests
        ensure_crypto_provider();

        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.model(repo_id.to_string());

        // Download config.json
        let config_path = repo
            .get("config.json")
            .await
            .context("Failed to download config.json")?;

        // Parse the configuration
        let config_content = tokio::fs::read_to_string(&config_path)
            .await
            .context("Failed to read config.json")?;
        let config: ModelConfig =
            serde_json::from_str(&config_content).context("Failed to parse config.json")?;

        // Detect architecture from config
        let architecture = Self::detect_architecture(&config)?;

        // Load model based on architecture
        match architecture {
            ModelArchitecture::Llama => {
                // Llama uses our custom weight loading pipeline
                let weights_path = Self::download_weights(&repo).await?;
                let weights_data = Self::load_weights_data(&weights_path).await?;
                Self::load_llama_model(
                    config,
                    config_content,
                    weights_data,
                    weights_path,
                    &repo,
                    backend,
                )
                .await
            }
            ModelArchitecture::Qwen3 => {
                // Qwen3 uses qwen3-burn's from_pretrained which loads weights itself
                Self::load_qwen3_model(config, &config_path, &repo, backend).await
            }
        }
    }

    /// Detects the model architecture from the config.
    fn detect_architecture(config: &ModelConfig) -> Result<ModelArchitecture> {
        if let Some(arch) = config.architectures.first() {
            match arch.as_str() {
                "LlamaForCausalLM" => Ok(ModelArchitecture::Llama),
                // Only Qwen3 is supported (has QK-Norm). Qwen2 is not supported.
                "Qwen3ForCausalLM" => Ok(ModelArchitecture::Qwen3),
                other => anyhow::bail!(
                    "Unsupported model architecture: {}. Supported: LlamaForCausalLM, Qwen3ForCausalLM",
                    other
                ),
            }
        } else {
            // Default to Llama if no architecture specified
            Ok(ModelArchitecture::Llama)
        }
    }

    /// Loads a Llama model.
    async fn load_llama_model(
        config: ModelConfig,
        config_content: String,
        weights_data: Vec<u8>,
        weights_path: PathBuf,
        repo: &hf_hub::api::tokio::ApiRepo,
        backend: InferenceBackend,
    ) -> Result<Self> {
        // Parse LlamaConfig for model construction
        let llama_config: LlamaConfig =
            serde_json::from_str(&config_content).context("Failed to parse Llama config")?;

        // Download tokenizer
        let tokenizer = load_tokenizer(repo)
            .await
            .context("Failed to load tokenizer")?;

        // Initialize the model with loaded weights
        let model = match backend {
            #[cfg(feature = "mlx")]
            InferenceBackend::Mlx => {
                anyhow::bail!("MLX backend is not yet supported for Llama models");
            }
            #[cfg(feature = "gpu")]
            InferenceBackend::Wgpu => {
                let device = try_init_wgpu()?;
                let model = load_llama_from_safetensors::<WgpuBackend>(
                    &weights_data,
                    &llama_config,
                    &device,
                )
                .context("Failed to load model weights for WGPU backend")?;
                LoadedModel::LlamaWgpu {
                    model: Box::new(model),
                    device,
                }
            }
            InferenceBackend::NdArray => {
                let device = NdArrayDevice::Cpu;
                let model = load_llama_from_safetensors::<NdArrayBackend>(
                    &weights_data,
                    &llama_config,
                    &device,
                )
                .context("Failed to load model weights for NdArray backend")?;
                LoadedModel::LlamaNdArray {
                    model: Box::new(model),
                    device,
                }
            }
        };

        Ok(Self {
            config,
            architecture: ModelArchitecture::Llama,
            llama_config: Some(llama_config),
            weights_path,
            tokenizer: LoadedTokenizer::TinyLlama(Box::new(tokenizer)),
            model,
        })
    }

    /// Loads a Qwen3 model using qwen3-burn's `from_pretrained`.
    ///
    /// Downloads all required files (weights, tokenizer) via hf-hub first,
    /// then passes the HF cache snapshot directory to from_pretrained.
    async fn load_qwen3_model(
        config: ModelConfig,
        config_path: &std::path::Path,
        repo: &hf_hub::api::tokio::ApiRepo,
        backend: InferenceBackend,
    ) -> Result<Self> {
        // Ensure weight files are downloaded so from_pretrained can find them
        let weights_path = Self::download_weights(repo).await?;

        // Ensure tokenizer is downloaded
        repo.get("tokenizer.json")
            .await
            .context("Failed to download tokenizer.json")?;

        // The HF cache snapshot directory contains all downloaded files
        let model_dir = config_path
            .parent()
            .context("Failed to determine model directory from config path")?;

        // Load our tokenizer wrapper (for chat formatting + qwen3-burn tokenizer access)
        let tokenizer = load_qwen3_tokenizer(repo)
            .await
            .context("Failed to load tokenizer")?;

        // Load model using qwen3-burn's from_pretrained
        let model = match backend {
            #[cfg(feature = "mlx")]
            InferenceBackend::Mlx => {
                anyhow::bail!(
                    "MLX backend for Qwen3 requires the `mlx` feature and burn-mlx crate"
                );
            }
            #[cfg(feature = "gpu")]
            InferenceBackend::Wgpu => {
                let device = try_init_wgpu()?;
                let qwen3 = Qwen3Model::<WgpuBackend>::from_pretrained(
                    model_dir,
                    4096,
                    QuantizationMode::None,
                    &device,
                )
                .map_err(|e| anyhow::anyhow!("Failed to load Qwen3 model (WGPU): {}", e))?;
                LoadedModel::Qwen3Wgpu {
                    model: Box::new(RefCell::new(qwen3)),
                }
            }
            InferenceBackend::NdArray => {
                let device = NdArrayDevice::Cpu;
                let qwen3 = Qwen3Model::<NdArrayBackend>::from_pretrained(
                    model_dir,
                    4096,
                    QuantizationMode::None,
                    &device,
                )
                .map_err(|e| anyhow::anyhow!("Failed to load Qwen3 model (NdArray): {}", e))?;
                LoadedModel::Qwen3NdArray {
                    model: Box::new(RefCell::new(qwen3)),
                }
            }
        };

        Ok(Self {
            config,
            architecture: ModelArchitecture::Qwen3,
            llama_config: None,
            weights_path,
            tokenizer: LoadedTokenizer::Qwen3(Box::new(tokenizer)),
            model,
        })
    }

    /// Attempts to download model weights, trying safetensors files first.
    ///
    /// Prioritizes `.safetensors` format as it's the only format currently supported
    /// for weight loading. Falls back to checking for other formats to provide
    /// helpful error messages.
    async fn download_weights(repo: &hf_hub::api::tokio::ApiRepo) -> Result<PathBuf> {
        // Try safetensors first (preferred and only supported format)
        if let Ok(path) = repo.get("model.safetensors").await {
            return Ok(path);
        }

        // Check for sharded safetensors (will produce helpful error later)
        if let Ok(path) = repo.get("model.safetensors.index.json").await {
            return Ok(path);
        }

        // Try other formats (will produce helpful error messages)
        let fallback_files = ["pytorch_model.bin", "model.bin"];
        for filename in fallback_files {
            if let Ok(path) = repo.get(filename).await {
                return Ok(path);
            }
        }

        // Check for sharded PyTorch weights
        if let Ok(path) = repo.get("pytorch_model.bin.index.json").await {
            return Ok(path);
        }

        anyhow::bail!(
            "Could not find model weights. Expected 'model.safetensors'. \
             Make sure the model repository contains safetensors weights."
        )
    }

    /// Loads weight data from the weights file.
    ///
    /// Currently supports:
    /// - Single `.safetensors` files
    ///
    /// Returns an error for:
    /// - Sharded weights (index.json files)
    /// - PyTorch `.bin` files (would need conversion)
    async fn load_weights_data(weights_path: &PathBuf) -> Result<Vec<u8>> {
        let path_str = weights_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid weights path"))?;

        // Check if this is a safetensors file
        if path_str.ends_with(".safetensors") {
            return tokio::fs::read(weights_path)
                .await
                .with_context(|| format!("Failed to read weights file: {:?}", weights_path));
        }

        // Check for sharded weights (not yet supported)
        if path_str.ends_with(".index.json") {
            anyhow::bail!(
                "Sharded model weights are not yet supported. \
                 Found index file: {:?}. \
                 Consider using a model with single-file weights.",
                weights_path
            );
        }

        // Check for PyTorch bin files (not supported)
        if path_str.ends_with(".bin") {
            anyhow::bail!(
                "PyTorch .bin weight files are not supported. \
                 Please use a model with .safetensors weights. \
                 Found: {:?}",
                weights_path
            );
        }

        anyhow::bail!(
            "Unsupported weights file format: {:?}. \
             Only .safetensors files are supported.",
            weights_path
        )
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the detected model architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        self.architecture
    }

    /// Returns the Llama-specific configuration (if Llama architecture).
    pub fn llama_config(&self) -> Option<&LlamaConfig> {
        self.llama_config.as_ref()
    }

    /// Returns the path to the cached model weights.
    pub fn weights_path(&self) -> &PathBuf {
        &self.weights_path
    }

    /// Returns which backend is being used.
    pub fn backend(&self) -> InferenceBackend {
        match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::LlamaWgpu { .. } => InferenceBackend::Wgpu,
            LoadedModel::LlamaNdArray { .. } => InferenceBackend::NdArray,
            #[cfg(feature = "gpu")]
            LoadedModel::Qwen3Wgpu { .. } => InferenceBackend::Wgpu,
            LoadedModel::Qwen3NdArray { .. } => InferenceBackend::NdArray,
        }
    }

    /// Returns whether the model is using GPU acceleration.
    pub fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "gpu")]
        match &self.model {
            LoadedModel::LlamaWgpu { .. } | LoadedModel::Qwen3Wgpu { .. } => return true,
            _ => {}
        }
        false
    }

    /// Generates text based on the given prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input text prompt
    ///
    /// # Returns
    ///
    /// Generated text response.
    pub fn generate(&self, prompt: &str) -> String {
        self.generate_with_config(prompt, GenerationConfig::default())
    }

    /// Generates text from a multi-turn conversation.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of messages
    ///
    /// # Returns
    ///
    /// Generated text response that continues the conversation.
    pub fn generate_from_conversation(
        &self,
        messages: &[crate::tokenizer::FormatMessage<'_>],
    ) -> String {
        self.generate_from_conversation_with_config(messages, GenerationConfig::default())
    }

    /// Generates text from a multi-turn conversation with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of messages
    /// * `gen_config` - Generation configuration (temperature, top-p, etc.)
    ///
    /// # Returns
    ///
    /// Generated text response.
    pub fn generate_from_conversation_with_config(
        &self,
        messages: &[crate::tokenizer::FormatMessage<'_>],
        mut gen_config: GenerationConfig,
    ) -> String {
        // Qwen3: use qwen3-burn's generate directly (handles tokenization internally)
        if self.architecture == ModelArchitecture::Qwen3 {
            return self.generate_qwen3_text(
                &Qwen3Tokenizer::format_multi_turn_prompt(messages),
                &gen_config,
            );
        }

        // Llama path: use our CausalLM-based generation pipeline
        self.apply_model_eos_tokens(&mut gen_config);

        let input_ids = match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => {
                let formatted = TinyLlamaTokenizer::format_multi_turn_prompt(messages);
                match tok.encode(&formatted, true) {
                    Ok(ids) => ids,
                    Err(e) => {
                        eprintln!("Tokenization error: {}", e);
                        return format!("[Tokenization error: {}]", e);
                    }
                }
            }
            LoadedTokenizer::Qwen3(_) => unreachable!(),
        };

        let output_ids = match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::LlamaWgpu { model, device } => {
                generate(&**model, input_ids, &gen_config, device)
            }
            LoadedModel::LlamaNdArray { model, device } => {
                generate(&**model, input_ids, &gen_config, device)
            }
            _ => unreachable!("Expected Llama model variant"),
        };

        self.decode_llama_output(&output_ids, gen_config.max_new_tokens)
    }

    /// Generates text from a conversation with streaming output.
    ///
    /// Each chunk of generated text is sent through the provided channel
    /// as it's produced, enabling real-time streaming to the UI.
    ///
    /// # Arguments
    ///
    /// * `messages` - The conversation history as a slice of messages
    /// * `gen_config` - Generation configuration (temperature, top-p, etc.)
    /// * `chunk_tx` - Channel sender for streaming text chunks
    pub fn generate_from_conversation_streaming(
        &self,
        messages: &[crate::tokenizer::FormatMessage<'_>],
        mut gen_config: GenerationConfig,
        chunk_tx: mpsc::Sender<StreamChunk>,
    ) {
        // Qwen3: use qwen3-burn's streaming generate
        if self.architecture == ModelArchitecture::Qwen3 {
            self.generate_qwen3_streaming(
                &Qwen3Tokenizer::format_multi_turn_prompt(messages),
                &gen_config,
                chunk_tx,
            );
            return;
        }

        // Llama path: use our CausalLM-based streaming pipeline
        self.apply_model_eos_tokens(&mut gen_config);

        let input_ids = match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => {
                let formatted = TinyLlamaTokenizer::format_multi_turn_prompt(messages);
                match tok.encode(&formatted, true) {
                    Ok(ids) => ids,
                    Err(e) => {
                        eprintln!("Tokenization error: {}", e);
                        let _ =
                            chunk_tx.send(StreamChunk::Error(format!("Tokenization error: {}", e)));
                        return;
                    }
                }
            }
            LoadedTokenizer::Qwen3(_) => unreachable!(),
        };

        // Create channel for token streaming
        let (token_tx, token_rx) = mpsc::channel::<StreamToken>();

        // Use scoped threads so we can borrow the tokenizer
        std::thread::scope(|scope| {
            let tokenizer_ref = &self.tokenizer;
            let chunk_tx_ref = &chunk_tx;

            let receiver_handle = scope.spawn(move || {
                let mut generated_tokens = Vec::new();
                let mut last_decoded_len = 0;

                for stream_token in token_rx {
                    match stream_token {
                        StreamToken::Token(token_id) => {
                            generated_tokens.push(token_id);

                            let decode_result = match tokenizer_ref {
                                LoadedTokenizer::TinyLlama(tok) => {
                                    tok.decode(&generated_tokens, true)
                                }
                                LoadedTokenizer::Qwen3(_) => unreachable!(),
                            };

                            match decode_result {
                                Ok(text) => {
                                    let text = text.trim_start();
                                    if text.len() > last_decoded_len {
                                        let new_text = &text[last_decoded_len..];
                                        if !new_text.is_empty() {
                                            let _ = chunk_tx_ref
                                                .send(StreamChunk::Text(new_text.to_string()));
                                        }
                                        last_decoded_len = text.len();
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Incremental decode error: {}", e);
                                }
                            }
                        }
                        StreamToken::Done => {
                            let _ = chunk_tx_ref.send(StreamChunk::Done);
                            break;
                        }
                        StreamToken::Error(e) => {
                            let _ = chunk_tx_ref.send(StreamChunk::Error(e));
                            break;
                        }
                    }
                }
            });

            // Run generation - only Llama variants reach here
            match &self.model {
                #[cfg(feature = "gpu")]
                LoadedModel::LlamaWgpu { model, device } => {
                    generate_streaming(&**model, input_ids, &gen_config, device, token_tx);
                }
                LoadedModel::LlamaNdArray { model, device } => {
                    generate_streaming(&**model, input_ids, &gen_config, device, token_tx);
                }
                _ => unreachable!("Expected Llama model variant"),
            }

            let _ = receiver_handle.join();
        });
    }

    /// Generates text with custom configuration.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input text prompt
    /// * `config` - Generation configuration (temperature, top-p, etc.)
    ///
    /// # Returns
    ///
    /// Generated text response.
    pub fn generate_with_config(&self, prompt: &str, gen_config: GenerationConfig) -> String {
        // Qwen3: use qwen3-burn's generate directly
        if self.architecture == ModelArchitecture::Qwen3 {
            return self
                .generate_qwen3_text(&Qwen3Tokenizer::format_chat_prompt(prompt), &gen_config);
        }

        // Llama path: use our CausalLM-based generation pipeline
        let mut gen_config = gen_config;
        self.apply_model_eos_tokens(&mut gen_config);

        let input_ids = match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => {
                let formatted = TinyLlamaTokenizer::format_chat_prompt(prompt);
                match tok.encode(&formatted, true) {
                    Ok(ids) => ids,
                    Err(e) => {
                        eprintln!("Tokenization error: {}", e);
                        return format!("[Tokenization error: {}]", e);
                    }
                }
            }
            LoadedTokenizer::Qwen3(_) => unreachable!(),
        };

        let output_ids = match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::LlamaWgpu { model, device } => {
                generate(&**model, input_ids, &gen_config, device)
            }
            LoadedModel::LlamaNdArray { model, device } => {
                generate(&**model, input_ids, &gen_config, device)
            }
            _ => unreachable!("Expected Llama model variant"),
        };

        self.decode_llama_output(&output_ids, gen_config.max_new_tokens)
    }

    /// Returns a reference to the TinyLlama tokenizer (if using Llama architecture).
    /// For architecture-agnostic access, use the generate methods instead.
    pub fn tokenizer(&self) -> Option<&TinyLlamaTokenizer> {
        match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => Some(tok),
            LoadedTokenizer::Qwen3(_) => None,
        }
    }

    /// Returns a reference to the Qwen3 tokenizer (if using Qwen3 architecture).
    pub fn qwen3_tokenizer(&self) -> Option<&Qwen3Tokenizer> {
        match &self.tokenizer {
            LoadedTokenizer::TinyLlama(_) => None,
            LoadedTokenizer::Qwen3(tok) => Some(tok),
        }
    }

    /// Returns the EOS token ID for this model.
    pub fn eos_token_id(&self) -> u32 {
        match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => tok.eos_token_id(),
            LoadedTokenizer::Qwen3(tok) => tok.eos_token_id(),
        }
    }

    /// Returns all EOS token IDs for this model.
    pub fn all_eos_token_ids(&self) -> Vec<u32> {
        match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => vec![tok.eos_token_id()],
            LoadedTokenizer::Qwen3(tok) => tok.all_eos_token_ids(),
        }
    }

    /// Returns a GenerationConfig with the correct EOS tokens for this model.
    pub fn default_generation_config(&self) -> GenerationConfig {
        let all_eos = self.all_eos_token_ids();
        let (primary, additional) = if all_eos.is_empty() {
            (2, Vec::new()) // fallback
        } else {
            (all_eos[0], all_eos[1..].to_vec())
        };
        GenerationConfig {
            eos_token_id: primary,
            additional_eos_ids: additional,
            ..GenerationConfig::default()
        }
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    /// Applies this model's EOS tokens to a GenerationConfig if defaults are in use.
    fn apply_model_eos_tokens(&self, config: &mut GenerationConfig) {
        if config.eos_token_id == 2 && config.additional_eos_ids.is_empty() {
            let all_eos = self.all_eos_token_ids();
            if !all_eos.is_empty() {
                config.eos_token_id = all_eos[0];
                config.additional_eos_ids = all_eos[1..].to_vec();
            }
        }
    }

    /// Decodes Llama output tokens, skipping the input portion.
    fn decode_llama_output(&self, output_ids: &[u32], max_new_tokens: usize) -> String {
        let new_tokens = &output_ids[output_ids.len().saturating_sub(max_new_tokens)..];
        let decoded = match &self.tokenizer {
            LoadedTokenizer::TinyLlama(tok) => tok.decode(new_tokens, true),
            LoadedTokenizer::Qwen3(_) => unreachable!(),
        };
        match decoded {
            Ok(text) => text.trim().to_string(),
            Err(e) => {
                eprintln!("Decoding error: {}", e);
                format!("[Decoding error: {}]", e)
            }
        }
    }

    /// Generates text using qwen3-burn's non-streaming generate method.
    fn generate_qwen3_text(&self, prompt: &str, config: &GenerationConfig) -> String {
        let tokenizer = match &self.tokenizer {
            LoadedTokenizer::Qwen3(tok) => tok,
            _ => unreachable!(),
        };
        let hf_tok = tokenizer.inner();
        let mut sampler = Self::make_sampler(config);

        macro_rules! run_generate {
            ($model_cell:expr) => {{
                $model_cell.borrow_mut().generate(
                    hf_tok,
                    prompt,
                    config.max_new_tokens,
                    config.temperature as f64,
                    &mut sampler,
                )
            }};
        }

        let result = match &self.model {
            LoadedModel::Qwen3NdArray { model } => run_generate!(model),
            #[cfg(feature = "gpu")]
            LoadedModel::Qwen3Wgpu { model } => run_generate!(model),
            _ => unreachable!("Expected Qwen3 model variant"),
        };

        match result {
            Ok(output) => output.text,
            Err(e) => {
                eprintln!("Qwen3 generation error: {}", e);
                format!("[Generation error: {}]", e)
            }
        }
    }

    /// Generates text using qwen3-burn's streaming generate method.
    ///
    /// Maps qwen3-burn's callback-based `GenerationEvent` to `StreamChunk`
    /// messages sent through the provided channel. Tracks accumulated text
    /// to compute deltas for each token event.
    fn generate_qwen3_streaming(
        &self,
        prompt: &str,
        config: &GenerationConfig,
        chunk_tx: mpsc::Sender<StreamChunk>,
    ) {
        let tokenizer = match &self.tokenizer {
            LoadedTokenizer::Qwen3(tok) => tok,
            _ => unreachable!(),
        };
        let hf_tok = tokenizer.inner();
        let mut sampler = Self::make_sampler(config);

        macro_rules! run_streaming {
            ($model_cell:expr) => {{
                let mut model = $model_cell.borrow_mut();
                let mut last_text_len = 0usize;
                let tx = &chunk_tx;
                model.generate_streaming(
                    hf_tok,
                    GenerationParams {
                        prompt,
                        max_new_tokens: config.max_new_tokens,
                        temperature: config.temperature as f64,
                        sampler: &mut sampler,
                        prefill_chunk_size: None,
                    },
                    |event: GenerationEvent| match event {
                        GenerationEvent::Token { text, .. } => {
                            if text.len() > last_text_len {
                                let delta = text[last_text_len..].to_string();
                                last_text_len = text.len();
                                if !delta.is_empty() && tx.send(StreamChunk::Text(delta)).is_err() {
                                    return ControlFlow::Break(());
                                }
                            }
                            ControlFlow::Continue(())
                        }
                        GenerationEvent::Done { .. } => {
                            let _ = tx.send(StreamChunk::Done);
                            ControlFlow::Break(())
                        }
                        GenerationEvent::PrefillProgress { .. } => ControlFlow::Continue(()),
                    },
                )
            }};
        }

        let result = match &self.model {
            LoadedModel::Qwen3NdArray { model } => run_streaming!(model),
            #[cfg(feature = "gpu")]
            LoadedModel::Qwen3Wgpu { model } => run_streaming!(model),
            _ => unreachable!("Expected Qwen3 model variant"),
        };

        if let Err(e) = result {
            let _ = chunk_tx.send(StreamChunk::Error(format!("Qwen3 streaming error: {}", e)));
        }
    }

    /// Creates a Sampler from GenerationConfig settings.
    fn make_sampler(config: &GenerationConfig) -> Sampler {
        if config.temperature == 0.0 {
            Sampler::Argmax
        } else {
            Sampler::new_top_p(config.top_p as f64, rand::random::<u64>())
        }
    }
}

/// Attempts to initialize the WGPU backend.
///
/// WGPU may panic if no GPU adapter is available (e.g., in CI/sandbox environments).
/// We catch the panic and convert it to an error for graceful fallback.
#[cfg(feature = "gpu")]
fn try_init_wgpu() -> Result<WgpuDevice> {
    use std::panic;

    let result = panic::catch_unwind(|| {
        let device = WgpuDevice::default();

        // Verify we can create a simple tensor on the device
        use burn::tensor::Tensor;
        let _tensor: Tensor<WgpuBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);

        device
    });

    result.map_err(|e| {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic during WGPU initialization".to_string()
        };
        anyhow::anyhow!("WGPU initialization failed: {}", msg)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test requires network access and system configuration.
    /// Run with: cargo test --ignored test_model_load
    #[tokio::test]
    #[ignore]
    async fn test_model_load() {
        let load_result = TextGenerationModel::load().await;

        match load_result {
            Ok(model) => {
                let config = model.config();
                println!("Loaded model config:");
                println!("  Architectures: {:?}", config.architectures);
                println!("  Hidden size: {}", config.hidden_size);
                println!("  Num layers: {}", config.num_hidden_layers);
                println!("  Vocab size: {}", config.vocab_size);
                println!("  Backend: {:?}", model.backend());
                println!("  GPU accelerated: {}", model.is_gpu_accelerated());

                let weights_path = model.weights_path();
                println!("  Weights path: {:?}", weights_path);
                assert!(weights_path.exists(), "Weights file should exist");

                let response = model.generate("Hello!");
                println!("  Generate output: {}", response);
            }
            Err(e) => {
                panic!("Model loading failed: {e}");
            }
        }
    }

    /// Test that model config deserialization works correctly.
    #[test]
    fn test_model_config_deserialization() {
        let config_json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 2048,
            "num_attention_heads": 32,
            "num_hidden_layers": 22,
            "vocab_size": 32000,
            "extra_field": "ignored"
        }"#;

        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        assert_eq!(config.architectures, vec!["LlamaForCausalLM"]);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.vocab_size, 32000);
        assert!(config.extra.contains_key("extra_field"));
    }

    /// Test Llama config deserialization.
    #[test]
    fn test_llama_config_deserialization() {
        let config_json = r#"{
            "vocab_size": 32000,
            "hidden_size": 2048,
            "intermediate_size": 5632,
            "num_hidden_layers": 22,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 2048,
            "rope_theta": 10000.0,
            "bos_token_id": 1,
            "eos_token_id": 2
        }"#;

        let config: LlamaConfig = serde_json::from_str(config_json).unwrap();

        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.intermediate_size, 5632);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_kv_heads(), 4);
        assert_eq!(config.head_dim(), 64);
    }

    /// Test that model generates coherent output (not gibberish).
    /// This test requires network access to download the model.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_model_output_coherence --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_output_coherence() {
        println!("Loading model...");
        let model = TextGenerationModel::load()
            .await
            .expect("Failed to load model");

        println!("Model loaded. Backend: {:?}", model.backend());

        // Test with a simple prompt
        let prompt = "What is 2 + 2?";
        println!("\nPrompt: {}", prompt);

        let response = model.generate(prompt);
        println!("Response: {}", response);

        // Basic coherence checks
        assert!(!response.is_empty(), "Response should not be empty");

        assert!(
            !response.contains("[Tokenization error"),
            "Should not have tokenization errors"
        );

        assert!(
            !response.contains("[Decoding error"),
            "Should not have decoding errors"
        );

        // Check that response contains actual words (not just random characters)
        // A coherent response should have spaces and common letters
        let has_spaces = response.contains(' ');
        let has_common_letters = response.chars().any(|c| "aeiouAEIOU".contains(c));

        assert!(
            has_spaces || response.len() < 10,
            "Response should contain spaces (unless very short): '{}'",
            response
        );

        assert!(
            has_common_letters,
            "Response should contain common vowels (not gibberish): '{}'",
            response
        );

        // Check that response doesn't have too many special/control characters
        let special_char_ratio = response
            .chars()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace() && !".,!?'-:;\"()".contains(*c))
            .count() as f32
            / response.len().max(1) as f32;

        assert!(
            special_char_ratio < 0.3,
            "Response has too many special characters ({:.0}%): '{}'",
            special_char_ratio * 100.0,
            response
        );

        // Check for repeated character patterns (sign of broken generation)
        let has_long_repetition = response
            .as_bytes()
            .windows(10)
            .any(|window| window.iter().all(|&b| b == window[0]));

        assert!(
            !has_long_repetition,
            "Response has repeated character patterns (broken generation): '{}'",
            response
        );

        // Check for repeated word patterns (another sign of broken generation)
        let words: Vec<&str> = response.split_whitespace().collect();
        if words.len() >= 4 {
            // Count word frequency
            let mut word_counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for word in &words {
                *word_counts.entry(*word).or_insert(0) += 1;
            }

            // Check if any short word appears too frequently (> 20% of words)
            let max_repetition = word_counts
                .iter()
                .filter(|(w, _)| w.len() <= 6) // Only check short words
                .map(|(_, count)| *count)
                .max()
                .unwrap_or(0);

            let repetition_ratio = max_repetition as f32 / words.len() as f32;
            assert!(
                repetition_ratio < 0.2,
                "Response has too many repeated words ({:.0}% repetition): '{}'",
                repetition_ratio * 100.0,
                response
            );
        }

        // For a math question, check if response contains any numbers or math-related words
        let math_related = response.contains('4')
            || response.to_lowercase().contains("four")
            || response.to_lowercase().contains("equal")
            || response.to_lowercase().contains("answer")
            || response.to_lowercase().contains("two");

        // NOTE: This assertion is expected to fail until the model inference is fixed.
        // The model currently produces gibberish output despite correct weight loading.
        // Further investigation needed into attention/RoPE/MLP computation.
        assert!(
            math_related,
            "Response to math question should contain numbers or math words.\n\
             Got: '{}'\n\n\
             This indicates the model inference is not working correctly.\n\
             Weight loading and architecture have been verified - the issue is likely in:\n\
             - Attention computation\n\
             - RoPE (Rotary Position Embedding) implementation\n\
             - Layer normalization\n\
             - Or numerical precision issues",
            response
        );

        println!("\n✓ Response appears coherent!");
    }

    /// Test generation with different prompts to verify consistency.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_multiple_prompts --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_multiple_prompts() {
        println!("Loading model...");
        let model = TextGenerationModel::load()
            .await
            .expect("Failed to load model");

        let prompts = [
            "Hello, how are you?",
            "Write a haiku about coding.",
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
        ];

        for prompt in prompts {
            println!("\n--- Prompt: {} ---", prompt);
            let response = model.generate(prompt);
            println!("Response: {}", response);

            // Basic sanity check
            assert!(
                !response.is_empty() && !response.contains("[error"),
                "Generation failed for prompt: {}",
                prompt
            );
        }
    }

    /// Test that single-turn and multi-turn produce same output for single message.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_single_vs_multi_turn --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_single_vs_multi_turn() {
        use crate::tokenizer::{ChatRole, FormatMessage};

        println!("Loading model...");
        let model = TextGenerationModel::load()
            .await
            .expect("Failed to load model");

        let prompt = "Hello!";
        println!("\nTesting prompt: {}", prompt);

        // Test single-turn
        println!("\n=== Single-turn (original generate) ===");
        let single_response = model.generate(prompt);
        println!("Response: {}", single_response);

        // Test multi-turn with single message
        println!("\n=== Multi-turn (generate_from_conversation) ===");
        let messages = vec![FormatMessage {
            role: ChatRole::User,
            content: prompt,
        }];
        let multi_response = model.generate_from_conversation(&messages);
        println!("Response: {}", multi_response);

        // The responses should be similar (not necessarily identical due to sampling)
        println!("\n=== Comparison ===");
        println!("Single-turn length: {}", single_response.len());
        println!("Multi-turn length: {}", multi_response.len());
    }

    /// Diagnostic test to examine model logits and greedy decoding.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_model_logits_diagnostic --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_model_logits_diagnostic() {
        use crate::generation::generate_greedy;

        println!("Loading model...");
        let model = TextGenerationModel::load()
            .await
            .expect("Failed to load model");

        // Simple prompt
        let prompt = "Hello";
        println!("\nPrompt: {}", prompt);

        // Tokenize
        let formatted = TinyLlamaTokenizer::format_chat_prompt(prompt);
        println!("Formatted prompt: {}", formatted);

        let tokenizer = model.tokenizer().expect("Expected TinyLlama tokenizer");
        let input_ids = tokenizer.encode(&formatted, true).unwrap();
        println!("Input token IDs: {:?}", input_ids);
        println!(
            "Input tokens: {:?}",
            input_ids
                .iter()
                .map(|&id| tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| format!("<{}>", id)))
                .collect::<Vec<_>>()
        );

        // Get device based on backend
        let device = match model.backend() {
            #[cfg(feature = "mlx")]
            InferenceBackend::Mlx => {
                println!("\nUsing MLX backend - skipping detailed logits analysis");
                return;
            }
            #[cfg(feature = "gpu")]
            InferenceBackend::Wgpu => {
                // For WGPU, we need to use greedy generation which handles device internally
                println!("\nUsing WGPU backend - running greedy generation...");
                let gen_config = GenerationConfig {
                    max_new_tokens: 10,
                    temperature: 1.0,
                    top_p: 1.0,
                    ..GenerationConfig::default()
                };
                let response = model.generate_with_config(prompt, gen_config);
                println!("Greedy-ish response (10 tokens): {}", response);
                return;
            }
            InferenceBackend::NdArray => burn_ndarray::NdArrayDevice::Cpu,
        };

        // Run greedy generation to eliminate sampling randomness
        println!("\nRunning greedy decoding (no sampling randomness)...");
        match &model.model {
            LoadedModel::LlamaNdArray { model: llama, .. } => {
                let output_ids = generate_greedy(&**llama, input_ids.clone(), 20, 2, &device);
                let new_tokens: Vec<u32> = output_ids[input_ids.len()..].to_vec();

                println!("Generated token IDs: {:?}", new_tokens);
                println!("Generated tokens:");
                for (i, &id) in new_tokens.iter().enumerate() {
                    let token_str = tokenizer
                        .decode(&[id], false)
                        .unwrap_or_else(|_| format!("<{}>", id));
                    println!("  {}: {} -> '{}'", i, id, token_str);
                }

                let full_response = tokenizer.decode(&new_tokens, true).unwrap();
                println!("\nFull greedy response: {}", full_response);
            }
            _ => {
                println!("Skipping detailed analysis for non-Llama-NdArray backend");
            }
        }
    }

    /// Test that tokenizer roundtrips correctly.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_tokenizer_roundtrip --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_tokenizer_roundtrip() {
        println!("Loading model (for tokenizer)...");
        let model = TextGenerationModel::load()
            .await
            .expect("Failed to load model");

        let test_strings = [
            "Hello, world!",
            "The answer is 42.",
            "What is 2 + 2?",
            "I am a helpful assistant.",
        ];

        let tokenizer = model.tokenizer().expect("Expected TinyLlama tokenizer");
        for test_str in test_strings {
            let tokens = tokenizer.encode(test_str, false).unwrap();
            let decoded = tokenizer.decode(&tokens, false).unwrap();

            println!("Original:  '{}'", test_str);
            println!("Tokens:    {:?}", tokens);
            println!("Decoded:   '{}'", decoded);
            println!();

            // Check roundtrip (may not be exact due to tokenization)
            assert!(
                decoded.contains(&test_str[..test_str.len().min(10)]),
                "Tokenizer roundtrip failed for: {}",
                test_str
            );
        }
    }

    /// Diagnostic test to verify weight loading from safetensors.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_weight_loading_diagnostic --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_weight_loading_diagnostic() {
        use crate::weights::LlamaWeightLoader;

        println!("Setting up HuggingFace API...");

        // Ensure crypto provider is installed
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();
        });

        let api = hf_hub::api::tokio::Api::new().unwrap();
        let repo = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

        println!("Downloading model.safetensors...");
        let weights_path = repo.get("model.safetensors").await.unwrap();
        println!("Weights path: {:?}", weights_path);

        println!("Loading weights file...");
        let weights_data = tokio::fs::read(&weights_path).await.unwrap();
        println!("Weights file size: {} MB", weights_data.len() / 1024 / 1024);

        let loader = LlamaWeightLoader::new(&weights_data).unwrap();

        println!("\nTensor names in safetensors:");
        let names = loader.tensor_names();
        for name in names.iter().take(20) {
            println!("  {}", name);
        }
        println!("  ... ({} total tensors)", names.len());

        // Load a few weights and check their statistics
        use crate::NdArrayBackend;
        use burn_ndarray::NdArrayDevice;

        let device = NdArrayDevice::Cpu;

        println!("\n--- Checking embedding weights ---");
        let embed_weight = loader.load_embed_tokens::<NdArrayBackend>(&device).unwrap();
        let embed_shape = embed_weight.dims();
        let embed_data: Vec<f32> = embed_weight.val().to_data().to_vec().unwrap();

        println!("Embedding shape: {:?}", embed_shape);
        println!("Expected: [32000, 2048]");

        let embed_mean: f32 = embed_data.iter().sum::<f32>() / embed_data.len() as f32;
        let embed_std: f32 = (embed_data
            .iter()
            .map(|x| (x - embed_mean).powi(2))
            .sum::<f32>()
            / embed_data.len() as f32)
            .sqrt();
        let embed_min = embed_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let embed_max = embed_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!(
            "Embedding stats: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
            embed_mean, embed_std, embed_min, embed_max
        );

        // Check for NaN/Inf
        let nan_count = embed_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = embed_data.iter().filter(|x| x.is_infinite()).count();
        println!("NaN count: {}, Inf count: {}", nan_count, inf_count);

        println!("\n--- Checking layer 0 attention weights ---");
        let attn_weights = loader
            .load_attention_weights::<NdArrayBackend>(0, &device)
            .unwrap();

        let q_shape = attn_weights.q_proj.dims();
        let q_data: Vec<f32> = attn_weights.q_proj.val().to_data().to_vec().unwrap();
        println!(
            "Q projection shape: {:?} (expected [2048, 2048] after transpose)",
            q_shape
        );
        let q_mean: f32 = q_data.iter().sum::<f32>() / q_data.len() as f32;
        let q_std: f32 =
            (q_data.iter().map(|x| (x - q_mean).powi(2)).sum::<f32>() / q_data.len() as f32).sqrt();
        println!("Q stats: mean={:.6}, std={:.6}", q_mean, q_std);

        let k_shape = attn_weights.k_proj.dims();
        println!(
            "K projection shape: {:?} (expected [2048, 256] after transpose)",
            k_shape
        );

        let v_shape = attn_weights.v_proj.dims();
        println!(
            "V projection shape: {:?} (expected [2048, 256] after transpose)",
            v_shape
        );

        let o_shape = attn_weights.o_proj.dims();
        println!(
            "O projection shape: {:?} (expected [2048, 2048] after transpose)",
            o_shape
        );

        println!("\n--- Checking LM head weights ---");
        let lm_head = loader.load_lm_head::<NdArrayBackend>(&device).unwrap();
        let lm_shape = lm_head.dims();
        println!(
            "LM head shape: {:?} (expected [2048, 32000] after transpose)",
            lm_shape
        );

        // Verify shapes are correct
        assert_eq!(embed_shape, [32000, 2048], "Embedding shape mismatch");
        assert_eq!(q_shape, [2048, 2048], "Q projection shape mismatch");
        assert_eq!(k_shape, [2048, 256], "K projection shape mismatch");
        assert_eq!(v_shape, [2048, 256], "V projection shape mismatch");
        assert_eq!(o_shape, [2048, 2048], "O projection shape mismatch");
        assert_eq!(lm_shape, [2048, 32000], "LM head shape mismatch");

        println!("\n✓ All weight shapes are correct!");
    }

    /// Diagnostic test to examine model forward pass with NdArray backend.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_forward_pass_diagnostic --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_forward_pass_diagnostic() {
        use crate::NdArrayBackend;
        use crate::weights::load_llama_from_safetensors;
        use burn::tensor::TensorData;
        use burn_ndarray::NdArrayDevice;

        println!("Setting up...");

        // Ensure crypto provider is installed
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();
        });

        let api = hf_hub::api::tokio::Api::new().unwrap();
        let repo = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

        // Download config and weights
        let config_path = repo.get("config.json").await.unwrap();
        let config_content = tokio::fs::read_to_string(&config_path).await.unwrap();
        let llama_config: LlamaConfig = serde_json::from_str(&config_content).unwrap();

        println!("Model config:");
        println!("  vocab_size: {}", llama_config.vocab_size);
        println!("  hidden_size: {}", llama_config.hidden_size);
        println!("  num_layers: {}", llama_config.num_hidden_layers);
        println!("  num_heads: {}", llama_config.num_attention_heads);
        println!("  num_kv_heads: {}", llama_config.num_kv_heads());

        let weights_path = repo.get("model.safetensors").await.unwrap();
        let weights_data = tokio::fs::read(&weights_path).await.unwrap();

        println!("\nLoading model with NdArray backend...");
        let device = NdArrayDevice::Cpu;
        let model =
            load_llama_from_safetensors::<NdArrayBackend>(&weights_data, &llama_config, &device)
                .unwrap();

        // Create a simple input: just the BOS token
        let input_ids: Vec<i32> = vec![1]; // BOS token
        let input_tensor: burn::tensor::Tensor<NdArrayBackend, 2, burn::tensor::Int> =
            burn::tensor::Tensor::from_data(TensorData::new(input_ids.clone(), [1, 1]), &device);

        println!("\nInput: BOS token (id=1)");
        println!("Input tensor shape: {:?}", input_tensor.dims());

        // Forward pass
        println!("\nRunning forward pass...");
        let logits = model.forward(input_tensor, 0);

        let logits_shape = logits.dims();
        println!("Output logits shape: {:?}", logits_shape);

        // Get logits as Vec
        let logits_flat = logits.reshape([logits_shape[2]]);
        let logits_data: Vec<f32> = logits_flat.to_data().to_vec().unwrap();

        // Statistics
        let mean: f32 = logits_data.iter().sum::<f32>() / logits_data.len() as f32;
        let std: f32 = (logits_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / logits_data.len() as f32)
            .sqrt();
        let min = logits_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = logits_data
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        println!("\nLogits statistics:");
        println!("  mean: {:.4}", mean);
        println!("  std:  {:.4}", std);
        println!("  min:  {:.4}", min);
        println!("  max:  {:.4}", max);

        // Check for NaN/Inf
        let nan_count = logits_data.iter().filter(|x| x.is_nan()).count();
        let inf_count = logits_data.iter().filter(|x| x.is_infinite()).count();
        println!("  NaN count: {}", nan_count);
        println!("  Inf count: {}", inf_count);

        // Get top 10 tokens by logit value
        let mut indexed: Vec<(usize, f32)> = logits_data.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("\nTop 10 predicted tokens:");
        // Load tokenizer for decoding
        let tokenizer = crate::tokenizer::load_tokenizer(&repo).await.unwrap();
        for (idx, logit) in indexed.iter().take(10) {
            let token_str = tokenizer
                .decode(&[*idx as u32], false)
                .unwrap_or_else(|_| format!("<{}>", idx));
            println!(
                "  {}: id={}, logit={:.4}, token='{}'",
                indexed.iter().position(|(i, _)| i == idx).unwrap(),
                idx,
                logit,
                token_str
            );
        }

        // Softmax to get probabilities for top tokens
        let softmax_logits = burn::tensor::activation::softmax(
            burn::tensor::Tensor::<NdArrayBackend, 1>::from_data(
                TensorData::new(logits_data.clone(), [logits_data.len()]),
                &device,
            ),
            0,
        );
        let probs: Vec<f32> = softmax_logits.to_data().to_vec().unwrap();

        println!("\nTop 10 token probabilities:");
        for (idx, _logit) in indexed.iter().take(10) {
            let prob = probs[*idx];
            let token_str = tokenizer
                .decode(&[*idx as u32], false)
                .unwrap_or_else(|_| format!("<{}>", idx));
            println!("  id={}: prob={:.4}, token='{}'", idx, prob, token_str);
        }

        // Basic sanity check
        assert!(nan_count == 0, "Logits contain NaN values");
        assert!(inf_count == 0, "Logits contain Inf values");
        assert!(
            std > 0.1,
            "Logits have very low variance (std={:.4}), model may not be working",
            std
        );
    }

    /// Diagnostic test to verify embedding produces distinct vectors.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_embedding_distinctness --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_embedding_distinctness() {
        use crate::NdArrayBackend;
        use crate::llama::embedding::Embedding;
        use crate::weights::LlamaWeightLoader;
        use burn::tensor::TensorData;
        use burn_ndarray::NdArrayDevice;

        println!("Setting up...");

        // Ensure crypto provider
        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();
        });

        let api = hf_hub::api::tokio::Api::new().unwrap();
        let repo = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

        let weights_path = repo.get("model.safetensors").await.unwrap();
        let weights_data = tokio::fs::read(&weights_path).await.unwrap();

        let loader = LlamaWeightLoader::new(&weights_data).unwrap();
        let device = NdArrayDevice::Cpu;

        // Load embedding
        let embed_weight = loader.load_embed_tokens::<NdArrayBackend>(&device).unwrap();
        let embed = Embedding::from_weights(embed_weight);

        // Test different tokens
        let test_tokens = [1u32, 2, 100, 1000, 10000]; // BOS, EOS, and some random tokens

        println!("\nEmbedding vectors for different tokens:");

        let mut embeddings = Vec::new();

        for &token_id in &test_tokens {
            let input_ids: burn::tensor::Tensor<NdArrayBackend, 2, burn::tensor::Int> =
                burn::tensor::Tensor::from_data(
                    TensorData::new(vec![token_id as i32], [1, 1]),
                    &device,
                );

            let embed_output = embed.forward(input_ids);
            let embed_data: Vec<f32> = embed_output.reshape([2048]).to_data().to_vec().unwrap();

            let mean: f32 = embed_data.iter().sum::<f32>() / embed_data.len() as f32;
            let std: f32 = (embed_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / embed_data.len() as f32)
                .sqrt();
            let first_5: Vec<f32> = embed_data.iter().take(5).cloned().collect();

            println!(
                "Token {}: mean={:.6}, std={:.6}, first_5={:?}",
                token_id, mean, std, first_5
            );

            embeddings.push((token_id, embed_data));
        }

        // Check that embeddings are distinct
        println!("\nPairwise cosine similarities:");
        for i in 0..embeddings.len() {
            for j in (i + 1)..embeddings.len() {
                let (id_i, ref vec_i) = embeddings[i];
                let (id_j, ref vec_j) = embeddings[j];

                let dot: f32 = vec_i.iter().zip(vec_j.iter()).map(|(a, b)| a * b).sum();
                let norm_i: f32 = vec_i.iter().map(|x| x * x).sum::<f32>().sqrt();
                let norm_j: f32 = vec_j.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cosine = dot / (norm_i * norm_j);

                println!("  cos({}, {}) = {:.4}", id_i, id_j, cosine);

                // Embeddings for different tokens should not be identical
                assert!(
                    cosine < 0.99,
                    "Embeddings for tokens {} and {} are too similar (cosine={})",
                    id_i,
                    id_j,
                    cosine
                );
            }
        }

        println!("\n✓ Embeddings are distinct for different tokens");
    }

    /// Diagnostic test to check if model produces varied logits for a sequence.
    /// Run with: cargo test -p eidolons-perception --release -- --ignored test_sequence_logits --nocapture
    #[tokio::test]
    #[ignore]
    async fn test_sequence_logits() {
        use crate::NdArrayBackend;
        use crate::weights::load_llama_from_safetensors;
        use burn::tensor::TensorData;
        use burn_ndarray::NdArrayDevice;

        println!("Setting up...");

        use std::sync::Once;
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider()).ok();
        });

        let api = hf_hub::api::tokio::Api::new().unwrap();
        let repo = api.model("TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string());

        let config_path = repo.get("config.json").await.unwrap();
        let config_content = tokio::fs::read_to_string(&config_path).await.unwrap();
        let llama_config: LlamaConfig = serde_json::from_str(&config_content).unwrap();

        let weights_path = repo.get("model.safetensors").await.unwrap();
        let weights_data = tokio::fs::read(&weights_path).await.unwrap();

        let device = NdArrayDevice::Cpu;
        let model =
            load_llama_from_safetensors::<NdArrayBackend>(&weights_data, &llama_config, &device)
                .unwrap();

        // Test with a short sequence: "Hello" -> [1, 15043] (BOS + Hello)
        let input_ids = vec![1i32, 15043]; // BOS + "Hello" token
        let input_tensor: burn::tensor::Tensor<NdArrayBackend, 2, burn::tensor::Int> =
            burn::tensor::Tensor::from_data(TensorData::new(input_ids.clone(), [1, 2]), &device);

        println!("Input: BOS + 'Hello' (ids: {:?})", input_ids);

        let logits = model.forward(input_tensor, 0);
        println!("Logits shape: {:?}", logits.dims());

        // Check logits for each position
        for pos in 0..2 {
            let pos_logits = logits.clone().slice([0..1, pos..pos + 1, 0..32000]);
            let pos_logits_flat = pos_logits.reshape([32000]);
            let logits_data: Vec<f32> = pos_logits_flat.to_data().to_vec().unwrap();

            let mean: f32 = logits_data.iter().sum::<f32>() / logits_data.len() as f32;
            let std: f32 = (logits_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                / logits_data.len() as f32)
                .sqrt();

            // Get top 5 tokens
            let mut indexed: Vec<(usize, f32)> = logits_data.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            println!("\nPosition {}: mean={:.4}, std={:.4}", pos, mean, std);
            println!("Top 5 tokens:");

            let tokenizer = crate::tokenizer::load_tokenizer(&repo).await.unwrap();
            for (idx, logit) in indexed.iter().take(5) {
                let token_str = tokenizer
                    .decode(&[*idx as u32], false)
                    .unwrap_or_else(|_| "?".into());
                println!("  id={}: logit={:.4}, token='{}'", idx, logit, token_str);
            }
        }

        // The logits at position 0 and position 1 should be different
        let logits0 = logits
            .clone()
            .slice([0..1, 0..1, 0..32000])
            .reshape([32000]);
        let logits1 = logits.slice([0..1, 1..2, 0..32000]).reshape([32000]);

        let logits0_data: Vec<f32> = logits0.to_data().to_vec().unwrap();
        let logits1_data: Vec<f32> = logits1.to_data().to_vec().unwrap();

        // Compute cosine similarity
        let dot: f32 = logits0_data
            .iter()
            .zip(logits1_data.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm0: f32 = logits0_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1: f32 = logits1_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (norm0 * norm1);

        println!(
            "\nCosine similarity between pos 0 and pos 1 logits: {:.4}",
            cosine
        );

        assert!(
            cosine < 0.99,
            "Logits at different positions are too similar (cosine={})",
            cosine
        );
    }
}
