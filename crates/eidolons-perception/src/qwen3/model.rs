//! Main Qwen3 model implementation.

use super::attention::{Qwen3Attention, create_causal_mask};
use super::config::Qwen3Config;
use crate::llama::embedding::Embedding;
use crate::llama::mlp::LlamaMlp;
use crate::llama::norm::RmsNorm;
use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig, LinearRecord};
use burn::prelude::*;

/// Creates a Linear layer from pre-loaded weights (no bias).
///
/// Expects weights in Burn format [in_features, out_features].
fn linear_from_weight<B: Backend>(weight: Param<Tensor<B, 2>>) -> Linear<B> {
    let [in_features, out_features] = weight.dims();
    let device = weight.device();

    let linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(&device);

    let record = LinearRecord { weight, bias: None };
    linear.load_record(record)
}

/// A single transformer layer in Qwen3.
#[derive(Module, Debug)]
pub struct Qwen3Layer<B: Backend> {
    /// Self-attention layer with QK-Norm
    self_attn: Qwen3Attention<B>,
    /// Feed-forward network (SwiGLU, same as Llama)
    mlp: LlamaMlp<B>,
    /// Pre-attention normalization
    input_layernorm: RmsNorm<B>,
    /// Pre-FFN normalization
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3Layer<B> {
    /// Creates a new transformer layer with random initialization.
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            self_attn: Qwen3Attention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads(),
                config.head_dim(),
                config.rms_norm_eps,
                config.rope_theta,
                device,
            ),
            mlp: LlamaMlp::new(config.hidden_size, config.intermediate_size, device),
            input_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
            post_attention_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
        }
    }

    /// Creates a transformer layer from pre-loaded weights.
    pub fn from_weights(
        self_attn: Qwen3Attention<B>,
        mlp: LlamaMlp<B>,
        input_layernorm: RmsNorm<B>,
        post_attention_layernorm: RmsNorm<B>,
    ) -> Self {
        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    /// Forward pass through the transformer layer.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        start_pos: usize,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        // Self-attention with residual
        let h = self.input_layernorm.forward(x.clone());
        let h = self.self_attn.forward(h, start_pos, mask);
        let x = x + h;

        // MLP with residual
        let h = self.post_attention_layernorm.forward(x.clone());
        let h = self.mlp.forward(h);
        x + h
    }
}

/// Main Qwen3 model.
///
/// Key differences from Llama:
/// - QK-Norm in attention (RMSNorm applied to Q and K before RoPE)
/// - Untied embeddings (separate embed_tokens and lm_head)
/// - Higher RoPE theta (1,000,000 vs 10,000)
/// - Larger vocabulary (151,936 tokens)
#[derive(Module, Debug)]
pub struct Qwen3<B: Backend> {
    /// Token embeddings
    embed_tokens: Embedding<B>,
    /// Transformer layers
    layers: Vec<Qwen3Layer<B>>,
    /// Final normalization
    norm: RmsNorm<B>,
    /// LM head (output projection to vocabulary)
    /// Note: Always separate in Qwen3 (not tied to embeddings)
    lm_head: burn::nn::Linear<B>,
    /// Vocabulary size (stored for reference)
    #[module(skip)]
    vocab_size: usize,
}

impl<B: Backend> Qwen3<B> {
    /// Creates a new Qwen3 model with random weights.
    pub fn new(config: Qwen3Config, device: &B::Device) -> Self {
        let embed_tokens: Embedding<B> =
            Embedding::new(config.vocab_size, config.hidden_size, device);

        let layers: Vec<Qwen3Layer<B>> = (0..config.num_hidden_layers)
            .map(|_| Qwen3Layer::new(&config, device))
            .collect();

        let norm: RmsNorm<B> = RmsNorm::new(config.hidden_size, config.rms_norm_eps, device);

        let lm_head = burn::nn::LinearConfig::new(config.hidden_size, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size: config.vocab_size,
        }
    }

    /// Creates a Qwen3 model from pre-loaded weights.
    ///
    /// # Arguments
    ///
    /// * `embed_tokens` - Token embedding layer
    /// * `layers` - Transformer layers
    /// * `norm` - Final layer normalization
    /// * `lm_head_weight` - LM head weights (always separate in Qwen3)
    /// * `vocab_size` - Vocabulary size
    pub fn from_weights(
        embed_tokens: Embedding<B>,
        layers: Vec<Qwen3Layer<B>>,
        norm: RmsNorm<B>,
        lm_head_weight: Param<Tensor<B, 2>>,
        vocab_size: usize,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            norm,
            lm_head: linear_from_weight(lm_head_weight),
            vocab_size,
        }
    }

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Token IDs of shape [batch, seq_len]
    /// * `start_pos` - Starting position for rotary encoding
    ///
    /// # Returns
    ///
    /// Logits tensor of shape [batch, seq_len, vocab_size]
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        let [_batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Get embeddings
        let mut x = self.embed_tokens.forward(input_ids);

        // Create causal mask for full sequence
        let mask = if seq_len > 1 {
            Some(create_causal_mask::<B>(seq_len, &device))
        } else {
            None
        };

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(x, start_pos, mask.clone());
        }

        // Final norm and LM head
        let x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_qwen3_forward_shape() {
        let device = NdArrayDevice::Cpu;

        // Use small config for testing
        let test_config = Qwen3Config {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 2048,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
        };

        let model = Qwen3::<TestBackend>::new(test_config, &device);

        let input: Tensor<TestBackend, 2, Int> = Tensor::from_ints([[1, 2, 3, 4, 5]], &device);
        let output = model.forward(input, 0);

        assert_eq!(output.dims(), [1, 5, 100]);
    }

    #[test]
    fn test_qwen3_layer_residual() {
        let device = NdArrayDevice::Cpu;

        let config = Qwen3Config {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            head_dim: Some(16),
            rms_norm_eps: 1e-6,
            max_position_embeddings: 2048,
            rope_theta: 1_000_000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_ids: vec![2],
        };

        let layer = Qwen3Layer::<TestBackend>::new(&config, &device);

        let input: Tensor<TestBackend, 3> = Tensor::ones([1, 5, 64], &device);
        let output = layer.forward(input.clone(), 0, None);

        // Output should have same shape
        assert_eq!(output.dims(), input.dims());

        // Output should be finite
        let output_data: Vec<f32> = output.reshape([320]).to_data().to_vec().unwrap();
        assert!(output_data.iter().all(|&v| v.is_finite()));
    }
}
