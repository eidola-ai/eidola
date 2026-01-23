//! Model manager for text generation models.
//!
//! Handles downloading, loading, and running inference with Llama models.

use crate::NdArrayBackend;
#[cfg(feature = "gpu")]
use crate::WgpuBackend;
use crate::generation::{GenerationConfig, generate};
use crate::llama::{Llama, LlamaConfig};
use crate::tokenizer::{TinyLlamaTokenizer, load_tokenizer};
use anyhow::{Context, Result};
use burn_ndarray::NdArrayDevice;
#[cfg(feature = "gpu")]
use burn_wgpu::WgpuDevice;
use hf_hub::api::tokio::Api;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Once;

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
    /// GPU-accelerated inference (Metal on macOS, Vulkan elsewhere).
    /// Only available with the `gpu` feature.
    #[cfg(feature = "gpu")]
    Wgpu,
    /// CPU inference using ndarray.
    #[default]
    NdArray,
}

/// Internal model representation that can use either backend.
enum LoadedModel {
    #[cfg(feature = "gpu")]
    Wgpu {
        model: Llama<WgpuBackend>,
        device: WgpuDevice,
    },
    NdArray {
        model: Llama<NdArrayBackend>,
        device: NdArrayDevice,
    },
}

/// Represents a loaded text generation model.
///
/// Supports both GPU (WGPU) and CPU (NdArray) backends for inference.
/// Thread-safety for FFI is handled by wrapping in a Mutex at the service layer.
pub struct TextGenerationModel {
    /// The model configuration from HuggingFace.
    config: ModelConfig,
    /// The Llama configuration parsed for model construction.
    llama_config: LlamaConfig,
    /// Path to the cached model weights.
    weights_path: PathBuf,
    /// The tokenizer for encoding/decoding text.
    tokenizer: TinyLlamaTokenizer,
    /// The loaded model (either WGPU or NdArray backend).
    model: LoadedModel,
}

// Manual Debug impl since LoadedModel contains non-Debug types
impl std::fmt::Debug for TextGenerationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextGenerationModel")
            .field("config", &self.config)
            .field("weights_path", &self.weights_path)
            .field(
                "backend",
                &match &self.model {
                    #[cfg(feature = "gpu")]
                    LoadedModel::Wgpu { .. } => "Wgpu",
                    LoadedModel::NdArray { .. } => "NdArray",
                },
            )
            .finish()
    }
}

impl TextGenerationModel {
    /// The default model repository to use.
    pub const DEFAULT_REPO: &'static str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

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
        #[cfg(feature = "gpu")]
        {
            // Try WGPU first, fall back to NdArray
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

        // Parse LlamaConfig for model construction
        let llama_config: LlamaConfig =
            serde_json::from_str(&config_content).context("Failed to parse Llama config")?;

        // Download tokenizer
        let tokenizer = load_tokenizer(&repo)
            .await
            .context("Failed to load tokenizer")?;

        // Download model weights
        let weights_path = Self::download_weights(&repo).await?;

        // Initialize the model with the selected backend
        let model = match backend {
            #[cfg(feature = "gpu")]
            InferenceBackend::Wgpu => {
                let device = try_init_wgpu()?;
                let model = Llama::<WgpuBackend>::new(llama_config.clone(), &device);
                // TODO: Load weights from safetensors
                LoadedModel::Wgpu { model, device }
            }
            InferenceBackend::NdArray => {
                let device = NdArrayDevice::Cpu;
                let model = Llama::<NdArrayBackend>::new(llama_config.clone(), &device);
                // TODO: Load weights from safetensors
                LoadedModel::NdArray { model, device }
            }
        };

        Ok(Self {
            config,
            llama_config,
            weights_path,
            tokenizer,
            model,
        })
    }

    /// Attempts to download model weights, trying common file names.
    async fn download_weights(repo: &hf_hub::api::tokio::ApiRepo) -> Result<PathBuf> {
        // Try different common weight file names
        let weight_files = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
            "model.mpk",
        ];

        for filename in weight_files {
            match repo.get(filename).await {
                Ok(path) => return Ok(path),
                Err(_) => continue,
            }
        }

        // If no single file found, check for sharded weights
        if let Ok(path) = repo.get("model.safetensors.index.json").await {
            return Ok(path);
        }

        if let Ok(path) = repo.get("pytorch_model.bin.index.json").await {
            return Ok(path);
        }

        anyhow::bail!("Could not find model weights. Tried: {:?}", weight_files)
    }

    /// Returns the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the Llama-specific configuration.
    pub fn llama_config(&self) -> &LlamaConfig {
        &self.llama_config
    }

    /// Returns the path to the cached model weights.
    pub fn weights_path(&self) -> &PathBuf {
        &self.weights_path
    }

    /// Returns which backend is being used.
    pub fn backend(&self) -> InferenceBackend {
        match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::Wgpu { .. } => InferenceBackend::Wgpu,
            LoadedModel::NdArray { .. } => InferenceBackend::NdArray,
        }
    }

    /// Returns whether the model is using GPU acceleration.
    pub fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "gpu")]
        if matches!(self.model, LoadedModel::Wgpu { .. }) {
            return true;
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
        // Format prompt for TinyLlama chat
        let formatted_prompt = TinyLlamaTokenizer::format_chat_prompt(prompt);

        // Encode prompt
        let input_ids = match self.tokenizer.encode(&formatted_prompt, true) {
            Ok(ids) => ids,
            Err(e) => {
                eprintln!("Tokenization error: {}", e);
                return format!("[Tokenization error: {}]", e);
            }
        };

        // Run generation based on backend
        let output_ids = match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::Wgpu { model, device } => generate(model, input_ids, &gen_config, device),
            LoadedModel::NdArray { model, device } => {
                generate(model, input_ids, &gen_config, device)
            }
        };

        // Decode output (skip the input tokens)
        let new_tokens = &output_ids[output_ids.len().saturating_sub(gen_config.max_new_tokens)..];

        match self.tokenizer.decode(new_tokens, true) {
            Ok(text) => text.trim().to_string(),
            Err(e) => {
                eprintln!("Decoding error: {}", e);
                format!("[Decoding error: {}]", e)
            }
        }
    }

    /// Returns a reference to the tokenizer.
    pub fn tokenizer(&self) -> &TinyLlamaTokenizer {
        &self.tokenizer
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
}
