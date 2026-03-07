//! Model manager for text generation models.
//!
//! Handles downloading, loading, and running inference with Qwen3 models.

use crate::NdArrayBackend;
#[cfg(feature = "gpu")]
use crate::WgpuBackend;
use crate::generation::GenerationConfig;
use crate::qwen3::{GenerationEvent, GenerationParams, QuantizationMode, Qwen3Model, Sampler};
use crate::tokenizer::{Qwen3Tokenizer, load_qwen3_tokenizer};
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
    /// Model architecture type (e.g., "Qwen3ForCausalLM").
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

/// Internal model representation that can use either backend.
///
/// Uses `RefCell` because qwen3-burn's `generate` methods take
/// `&mut self` for KV cache mutation. Since the model runs on a single inference
/// thread (with a Mutex at the service layer), `RefCell` is safe here.
enum LoadedModel {
    #[cfg(feature = "gpu")]
    Qwen3Wgpu {
        model: Box<RefCell<Qwen3Model<WgpuBackend>>>,
    },
    Qwen3NdArray {
        model: Box<RefCell<Qwen3Model<NdArrayBackend>>>,
    },
}

/// Represents a loaded text generation model.
///
/// Supports both GPU (WGPU) and CPU (NdArray) backends for inference.
/// Thread-safety for FFI is handled by wrapping in a Mutex at the service layer.
pub struct TextGenerationModel {
    /// The model configuration from HuggingFace.
    config: ModelConfig,
    /// Path to the cached model weights.
    weights_path: PathBuf,
    /// The tokenizer for encoding/decoding text.
    tokenizer: Qwen3Tokenizer,
    /// The loaded model (either WGPU or NdArray backend).
    model: LoadedModel,
}

// Manual Debug impl since LoadedModel contains non-Debug types
impl std::fmt::Debug for TextGenerationModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let backend_name = match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::Qwen3Wgpu { .. } => "Wgpu",
            LoadedModel::Qwen3NdArray { .. } => "NdArray",
        };
        f.debug_struct("TextGenerationModel")
            .field("config", &self.config)
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

        // Verify architecture
        Self::verify_architecture(&config)?;

        // Ensure weight files are downloaded so from_pretrained can find them
        let weights_path = Self::download_weights(&repo).await?;

        // Ensure tokenizer is downloaded
        repo.get("tokenizer.json")
            .await
            .context("Failed to download tokenizer.json")?;

        // The HF cache snapshot directory contains all downloaded files
        let model_dir = config_path
            .parent()
            .context("Failed to determine model directory from config path")?;

        // Load our tokenizer wrapper (for chat formatting + qwen3-burn tokenizer access)
        let tokenizer = load_qwen3_tokenizer(&repo)
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
            weights_path,
            tokenizer,
            model,
        })
    }

    /// Verifies the model architecture is supported (Qwen3 only).
    fn verify_architecture(config: &ModelConfig) -> Result<ModelArchitecture> {
        if let Some(arch) = config.architectures.first() {
            match arch.as_str() {
                "Qwen3ForCausalLM" => Ok(ModelArchitecture::Qwen3),
                other => anyhow::bail!(
                    "Unsupported model architecture: {}. Only Qwen3ForCausalLM is supported.",
                    other
                ),
            }
        } else {
            anyhow::bail!(
                "No architecture specified in config.json. Only Qwen3ForCausalLM is supported."
            )
        }
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

    /// Returns the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Returns the detected model architecture.
    pub fn architecture(&self) -> ModelArchitecture {
        ModelArchitecture::Qwen3
    }

    /// Returns the path to the cached model weights.
    pub fn weights_path(&self) -> &PathBuf {
        &self.weights_path
    }

    /// Returns which backend is being used.
    pub fn backend(&self) -> InferenceBackend {
        match &self.model {
            #[cfg(feature = "gpu")]
            LoadedModel::Qwen3Wgpu { .. } => InferenceBackend::Wgpu,
            LoadedModel::Qwen3NdArray { .. } => InferenceBackend::NdArray,
        }
    }

    /// Returns whether the model is using GPU acceleration.
    pub fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "gpu")]
        if matches!(&self.model, LoadedModel::Qwen3Wgpu { .. }) {
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
        gen_config: GenerationConfig,
    ) -> String {
        self.generate_qwen3_text(
            &Qwen3Tokenizer::format_multi_turn_prompt(messages),
            &gen_config,
        )
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
        gen_config: GenerationConfig,
        chunk_tx: mpsc::Sender<StreamChunk>,
    ) {
        self.generate_qwen3_streaming(
            &Qwen3Tokenizer::format_multi_turn_prompt(messages),
            &gen_config,
            chunk_tx,
        );
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
        self.generate_qwen3_text(&Qwen3Tokenizer::format_chat_prompt(prompt), &gen_config)
    }

    /// Returns a reference to the Qwen3 tokenizer.
    pub fn qwen3_tokenizer(&self) -> &Qwen3Tokenizer {
        &self.tokenizer
    }

    /// Returns the EOS token ID for this model.
    pub fn eos_token_id(&self) -> u32 {
        self.tokenizer.eos_token_id()
    }

    /// Returns all EOS token IDs for this model.
    pub fn all_eos_token_ids(&self) -> Vec<u32> {
        self.tokenizer.all_eos_token_ids()
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

    /// Generates text using qwen3-burn's non-streaming generate method.
    fn generate_qwen3_text(&self, prompt: &str, config: &GenerationConfig) -> String {
        let hf_tok = self.tokenizer.inner();
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
        let hf_tok = self.tokenizer.inner();
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
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "vocab_size": 151936,
            "extra_field": "ignored"
        }"#;

        let config: ModelConfig = serde_json::from_str(config_json).unwrap();

        assert_eq!(config.architectures, vec!["Qwen3ForCausalLM"]);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.vocab_size, 151936);
        assert!(config.extra.contains_key("extra_field"));
    }

    /// Test architecture detection.
    #[test]
    fn test_architecture_detection() {
        // Qwen3 should be accepted
        let config = ModelConfig {
            architectures: vec!["Qwen3ForCausalLM".to_string()],
            hidden_size: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            vocab_size: 0,
            extra: HashMap::new(),
        };
        assert!(TextGenerationModel::verify_architecture(&config).is_ok());

        // Unknown architecture should be rejected
        let config = ModelConfig {
            architectures: vec!["UnknownModel".to_string()],
            hidden_size: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            vocab_size: 0,
            extra: HashMap::new(),
        };
        assert!(TextGenerationModel::verify_architecture(&config).is_err());

        // No architecture should be rejected
        let config = ModelConfig {
            architectures: vec![],
            hidden_size: 0,
            num_attention_heads: 0,
            num_hidden_layers: 0,
            vocab_size: 0,
            extra: HashMap::new(),
        };
        assert!(TextGenerationModel::verify_architecture(&config).is_err());
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
            !response.contains("[Generation error"),
            "Should not have generation errors"
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

        // For a math question, check if response contains any numbers or math-related words
        let math_related = response.contains('4')
            || response.to_lowercase().contains("four")
            || response.to_lowercase().contains("equal")
            || response.to_lowercase().contains("answer")
            || response.to_lowercase().contains("two");

        assert!(
            math_related,
            "Response to math question should contain numbers or math words.\nGot: '{}'",
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
}
