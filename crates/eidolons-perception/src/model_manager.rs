use anyhow::{Context, Result};
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

/// Represents a loaded text generation model.
///
/// Currently a placeholder that demonstrates model weight downloading
/// and configuration loading. Real inference will be implemented in Phase 2.
#[derive(Debug)]
pub struct TextGenerationModel {
    /// The model configuration.
    config: ModelConfig,
    /// Path to the cached model weights.
    weights_path: PathBuf,
    /// Whether the model is ready for inference.
    initialized: bool,
}

impl TextGenerationModel {
    /// The default model repository to use.
    pub const DEFAULT_REPO: &'static str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

    /// Loads a text generation model from HuggingFace Hub.
    ///
    /// Downloads the model configuration and weights to a local cache,
    /// then initializes the model for inference.
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
    /// # Arguments
    ///
    /// * `repo_id` - The HuggingFace repository ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    pub async fn load_from_repo(repo_id: &str) -> Result<Self> {
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

        // Try to download model weights (try common file names)
        // Models can use different naming conventions
        let weights_path = Self::download_weights(&repo).await?;

        Ok(Self {
            config,
            weights_path,
            initialized: true,
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
        // For now, just try to get the index file to verify weights exist
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

    /// Returns the path to the cached model weights.
    pub fn weights_path(&self) -> &PathBuf {
        &self.weights_path
    }

    /// Returns whether the model is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Generates text based on the given prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The input text prompt
    ///
    /// # Returns
    ///
    /// Generated text response. Currently returns a mocked response;
    /// real inference will be implemented in Phase 2.
    pub fn generate(&self, prompt: &str) -> String {
        // Phase 1: Return mocked response
        // Phase 2: Implement actual inference using Burn
        format!("ECHO [Model Loaded]: {}", prompt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This test requires network access and system configuration.
    /// It is ignored by default because it:
    /// - Requires network access to download from HuggingFace
    /// - May panic in sandboxed environments (CI, Nix builds) where
    ///   macOS system-configuration APIs are unavailable
    ///
    /// Run with: cargo test --ignored test_model_load
    #[tokio::test]
    #[ignore]
    async fn test_model_load() {
        let load_result = TextGenerationModel::load().await;

        match load_result {
            Ok(model) => {
                // Verify the model is initialized
                assert!(model.is_initialized());

                // Verify config was loaded
                let config = model.config();
                println!("Loaded model config:");
                println!("  Architectures: {:?}", config.architectures);
                println!("  Hidden size: {}", config.hidden_size);
                println!("  Num layers: {}", config.num_hidden_layers);
                println!("  Vocab size: {}", config.vocab_size);

                // Verify weights path exists
                let weights_path = model.weights_path();
                println!("  Weights path: {:?}", weights_path);
                assert!(weights_path.exists(), "Weights file should exist");

                // Test generate (mocked)
                let response = model.generate("Hello, world!");
                assert!(response.contains("ECHO [Model Loaded]"));
                assert!(response.contains("Hello, world!"));
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
}
