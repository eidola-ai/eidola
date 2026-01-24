//! Text generation with autoregressive sampling.
//!
//! Provides utilities for generating text from causal language models using
//! temperature-scaled sampling and nucleus (top-p) filtering.

use crate::llama::Llama;
use crate::qwen3::Qwen3;
use burn::prelude::*;
use burn::tensor::TensorData;
use rand::Rng;
use rand::distributions::Distribution;
use std::sync::mpsc;

/// Trait for causal language models that can be used for text generation.
///
/// Both Llama and Qwen3 implement this trait, allowing the generation functions
/// to be generic over different model architectures.
pub trait CausalLM<B: Backend> {
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
    fn forward(&self, input_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3>;

    /// Returns the vocabulary size.
    fn vocab_size(&self) -> usize;
}

impl<B: Backend> CausalLM<B> for Llama<B> {
    fn forward(&self, input_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        Llama::forward(self, input_ids, start_pos)
    }

    fn vocab_size(&self) -> usize {
        Llama::vocab_size(self)
    }
}

impl<B: Backend> CausalLM<B> for Qwen3<B> {
    fn forward(&self, input_ids: Tensor<B, 2, Int>, start_pos: usize) -> Tensor<B, 3> {
        Qwen3::forward(self, input_ids, start_pos)
    }

    fn vocab_size(&self) -> usize {
        Qwen3::vocab_size(self)
    }
}

/// Configuration for text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (higher = more random).
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f32,
    /// Token ID that signals end of generation (primary).
    pub eos_token_id: u32,
    /// Additional EOS token IDs that should also stop generation.
    /// Useful for models like Qwen3 that have multiple stop tokens.
    pub additional_eos_ids: Vec<u32>,
    /// Whether to include the EOS token in output.
    pub include_eos: bool,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 1024,
            temperature: 0.7,
            top_p: 0.9,
            eos_token_id: 2,
            additional_eos_ids: Vec::new(),
            include_eos: false,
        }
    }
}

impl GenerationConfig {
    /// Returns true if the given token is an EOS token.
    pub fn is_eos_token(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id || self.additional_eos_ids.contains(&token_id)
    }
}

/// Token emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamToken {
    /// A newly generated token.
    Token(u32),
    /// Generation is complete.
    Done,
    /// An error occurred during generation.
    Error(String),
}

/// Generates tokens autoregressively from a causal language model.
///
/// # Arguments
///
/// * `model` - The model to generate from (Llama or Qwen3)
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `device` - Device to run inference on
///
/// # Returns
///
/// Vector of generated token IDs (including the input).
pub fn generate<B: Backend, M: CausalLM<B>>(
    model: &M,
    input_ids: Vec<u32>,
    config: &GenerationConfig,
    device: &B::Device,
) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let mut tokens = input_ids.clone();
    let mut generated_count = 0;

    while generated_count < config.max_new_tokens {
        // Create input tensor from current tokens
        let token_data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let data = TensorData::new(token_data.clone(), [1, tokens.len()]);
        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(data, device);

        // Forward pass through model
        let logits = model.forward(input_tensor, 0);

        // Get logits for the last token only
        let [_batch, seq_len, vocab_size] = logits.dims();
        let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
        let last_logits = last_logits.reshape([vocab_size]);

        // Sample next token
        let next_token = sample_token(&last_logits, config, &mut rng);

        // Check for EOS (including additional EOS tokens)
        if config.is_eos_token(next_token) {
            if config.include_eos {
                tokens.push(next_token);
            }
            break;
        }

        tokens.push(next_token);
        generated_count += 1;
    }

    tokens
}

/// Generates tokens autoregressively, sending each token through a channel.
///
/// This is the streaming variant of `generate()`. Each generated token is sent
/// through the provided channel immediately after generation, enabling real-time
/// streaming of generated text.
///
/// # Arguments
///
/// * `model` - The model to generate from (Llama or Qwen3)
/// * `input_ids` - Initial token IDs (prompt)
/// * `config` - Generation configuration
/// * `device` - Device to run inference on
/// * `token_tx` - Channel sender for streaming tokens
///
/// # Returns
///
/// Vector of generated token IDs (including the input).
pub fn generate_streaming<B: Backend, M: CausalLM<B>>(
    model: &M,
    input_ids: Vec<u32>,
    config: &GenerationConfig,
    device: &B::Device,
    token_tx: mpsc::Sender<StreamToken>,
) -> Vec<u32> {
    let mut rng = rand::thread_rng();
    let mut tokens = input_ids.clone();
    let mut generated_count = 0;

    while generated_count < config.max_new_tokens {
        // Create input tensor from current tokens
        let token_data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let data = TensorData::new(token_data.clone(), [1, tokens.len()]);
        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(data, device);

        // Forward pass through model
        let logits = model.forward(input_tensor, 0);

        // Get logits for the last token only
        let [_batch, seq_len, vocab_size] = logits.dims();
        let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
        let last_logits = last_logits.reshape([vocab_size]);

        // Sample next token
        let next_token = sample_token(&last_logits, config, &mut rng);

        // Check for EOS (including additional EOS tokens)
        if config.is_eos_token(next_token) {
            if config.include_eos {
                tokens.push(next_token);
                // Send EOS token if included
                let _ = token_tx.send(StreamToken::Token(next_token));
            }
            // Signal completion
            let _ = token_tx.send(StreamToken::Done);
            break;
        }

        tokens.push(next_token);
        generated_count += 1;

        // Send the newly generated token
        if token_tx.send(StreamToken::Token(next_token)).is_err() {
            // Receiver dropped, stop generation
            break;
        }
    }

    // If we reached max tokens without EOS, still signal completion
    if generated_count >= config.max_new_tokens {
        let _ = token_tx.send(StreamToken::Done);
    }

    tokens
}

/// Samples a token from logits using temperature and top-p sampling.
fn sample_token<B: Backend, R: Rng>(
    logits: &Tensor<B, 1>,
    config: &GenerationConfig,
    rng: &mut R,
) -> u32 {
    // Apply temperature
    let scaled_logits = if config.temperature != 1.0 {
        logits.clone() / config.temperature
    } else {
        logits.clone()
    };

    // Convert to probabilities
    let probs = burn::tensor::activation::softmax(scaled_logits, 0);

    // Get probabilities as Vec<f32>
    let probs_data: Vec<f32> = probs.to_data().to_vec().expect("Failed to convert probs");

    // Apply top-p (nucleus) sampling
    let filtered_probs = apply_top_p(&probs_data, config.top_p);

    // Sample from the distribution
    sample_from_probs(&filtered_probs, rng)
}

/// Applies top-p (nucleus) filtering to probabilities.
fn apply_top_p(probs: &[f32], p: f32) -> Vec<f32> {
    if p >= 1.0 {
        return probs.to_vec();
    }

    // Create (index, prob) pairs and sort by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cumulative sum threshold
    let mut cumsum = 0.0;
    let mut cutoff_idx = indexed.len();

    for (i, (_, prob)) in indexed.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out tokens below threshold
    let mut filtered = vec![0.0; probs.len()];
    for (idx, prob) in indexed.into_iter().take(cutoff_idx) {
        filtered[idx] = prob;
    }

    // Renormalize
    let sum: f32 = filtered.iter().sum();
    if sum > 0.0 {
        for p in &mut filtered {
            *p /= sum;
        }
    }

    filtered
}

/// Samples an index from a probability distribution.
fn sample_from_probs<R: Rng>(probs: &[f32], rng: &mut R) -> u32 {
    let dist = WeightedIndex::new(probs);

    match dist {
        Ok(d) => d.sample(rng) as u32,
        Err(_) => {
            // Fallback: return argmax if distribution is invalid
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        }
    }
}

/// A weighted index distribution for sampling.
struct WeightedIndex {
    cumulative: Vec<f32>,
}

impl WeightedIndex {
    fn new(weights: &[f32]) -> Result<Self, &'static str> {
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return Err("Sum of weights must be positive");
        }

        let mut cumulative = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in weights {
            acc += w / sum;
            cumulative.push(acc);
        }

        Ok(Self { cumulative })
    }
}

impl Distribution<usize> for WeightedIndex {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let r: f32 = rng.r#gen();
        self.cumulative
            .iter()
            .position(|&c| r < c)
            .unwrap_or(self.cumulative.len() - 1)
    }
}

/// Greedy decoding - always picks the most likely token.
pub fn generate_greedy<B: Backend, M: CausalLM<B>>(
    model: &M,
    input_ids: Vec<u32>,
    max_new_tokens: usize,
    eos_token_id: u32,
    device: &B::Device,
) -> Vec<u32> {
    let mut tokens = input_ids;
    let mut generated_count = 0;

    while generated_count < max_new_tokens {
        let token_data: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let data = TensorData::new(token_data.clone(), [1, tokens.len()]);
        let input_tensor: Tensor<B, 2, Int> = Tensor::from_data(data, device);

        let logits = model.forward(input_tensor, 0);

        let [_batch, seq_len, vocab_size] = logits.dims();
        let last_logits = logits.slice([0..1, (seq_len - 1)..seq_len, 0..vocab_size]);
        let last_logits = last_logits.reshape([vocab_size]);

        // Argmax
        let next_token_idx = last_logits.argmax(0);
        let next_token_data: Vec<i32> = next_token_idx
            .to_data()
            .to_vec()
            .expect("Failed to get argmax");
        let next_token = next_token_data[0] as u32;

        if next_token == eos_token_id {
            break;
        }

        tokens.push(next_token);
        generated_count += 1;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_top_p() {
        let probs = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        let filtered = apply_top_p(&probs, 0.9);

        // Should keep tokens with cumulative prob < 0.9
        assert!(filtered[0] > 0.0);
        assert!(filtered[1] > 0.0);
        assert!(filtered[2] > 0.0);

        // Sum should be ~1.0 after renormalization
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weighted_index() {
        let weights = vec![0.7, 0.2, 0.1];
        let dist = WeightedIndex::new(&weights).unwrap();

        // Sample many times and check distribution is roughly correct
        let mut rng = rand::thread_rng();
        let mut counts = [0; 3];
        let n = 10000;

        for _ in 0..n {
            let idx = dist.sample(&mut rng);
            counts[idx] += 1;
        }

        // First should be sampled most often (~70%)
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }
}
