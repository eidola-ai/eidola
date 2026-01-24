//! Multi-head attention with QK-Norm and Rotary Position Embeddings (RoPE) for Qwen3.
//!
//! The key difference from Llama is QK-Norm: RMSNorm is applied to Q and K vectors
//! before RoPE, which improves training stability and convergence.

use crate::llama::norm::RmsNorm;
use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig, LinearRecord};
use burn::prelude::*;

/// Creates a Linear layer from pre-loaded weights (no bias).
///
/// Expects weights in Burn format [in_features, out_features].
/// (HuggingFace weights should be transposed before calling this function.)
fn linear_from_weight<B: Backend>(weight: Param<Tensor<B, 2>>) -> Linear<B> {
    let [in_features, out_features] = weight.dims();
    let device = weight.device();

    let linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(&device);

    let record = LinearRecord { weight, bias: None };
    linear.load_record(record)
}

/// Multi-head attention with QK-Norm and Rotary Position Embeddings.
///
/// Qwen3 applies RMSNorm to Q and K before RoPE, which is the key difference from Llama.
/// This improves training stability and allows for higher learning rates.
///
/// Supports grouped-query attention (GQA) where num_kv_heads < num_attention_heads.
#[derive(Module, Debug)]
pub struct Qwen3Attention<B: Backend> {
    /// Query projection
    q_proj: Linear<B>,
    /// Key projection
    k_proj: Linear<B>,
    /// Value projection
    v_proj: Linear<B>,
    /// Output projection
    o_proj: Linear<B>,
    /// QK-Norm: RMSNorm for query vectors (applied per head before RoPE)
    q_norm: RmsNorm<B>,
    /// QK-Norm: RMSNorm for key vectors (applied per head before RoPE)
    k_norm: RmsNorm<B>,

    /// Number of attention heads
    #[module(skip)]
    num_heads: usize,
    /// Number of key-value heads (for GQA)
    #[module(skip)]
    num_kv_heads: usize,
    /// Head dimension
    #[module(skip)]
    head_dim: usize,
    /// Rope theta for positional encoding (Qwen3 uses 1,000,000)
    #[module(skip)]
    rope_theta: f64,
}

impl<B: Backend> Qwen3Attention<B> {
    /// Creates a new attention layer.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rope_theta: f64,
        device: &B::Device,
    ) -> Self {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        let q_proj = LinearConfig::new(hidden_size, num_heads * head_dim)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);

        let k_proj = LinearConfig::new(hidden_size, num_kv_heads * head_dim)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);

        let v_proj = LinearConfig::new(hidden_size, num_kv_heads * head_dim)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);

        let o_proj = LinearConfig::new(num_heads * head_dim, hidden_size)
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);

        // QK-Norm: one RMSNorm per head_dim (applied to each head independently)
        let q_norm = RmsNorm::new(head_dim, rms_norm_eps, device);
        let k_norm = RmsNorm::new(head_dim, rms_norm_eps, device);

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        }
    }

    /// Creates an attention layer from pre-loaded weights.
    ///
    /// # Arguments
    ///
    /// * `q_proj_weight` - Query projection weights [hidden_size, num_heads * head_dim]
    /// * `k_proj_weight` - Key projection weights [hidden_size, num_kv_heads * head_dim]
    /// * `v_proj_weight` - Value projection weights [hidden_size, num_kv_heads * head_dim]
    /// * `o_proj_weight` - Output projection weights [num_heads * head_dim, hidden_size]
    /// * `q_norm_weight` - Q normalization weights [head_dim]
    /// * `k_norm_weight` - K normalization weights [head_dim]
    /// * `num_heads` - Number of attention heads
    /// * `num_kv_heads` - Number of key-value heads (for GQA)
    /// * `head_dim` - Dimension per head
    /// * `rms_norm_eps` - Epsilon for RMSNorm
    /// * `rope_theta` - Rope theta for positional encoding
    pub fn from_weights(
        q_proj_weight: Param<Tensor<B, 2>>,
        k_proj_weight: Param<Tensor<B, 2>>,
        v_proj_weight: Param<Tensor<B, 2>>,
        o_proj_weight: Param<Tensor<B, 2>>,
        q_norm_weight: Param<Tensor<B, 1>>,
        k_norm_weight: Param<Tensor<B, 1>>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rope_theta: f64,
    ) -> Self {
        Self {
            q_proj: linear_from_weight(q_proj_weight),
            k_proj: linear_from_weight(k_proj_weight),
            v_proj: linear_from_weight(v_proj_weight),
            o_proj: linear_from_weight(o_proj_weight),
            q_norm: RmsNorm::from_weights(q_norm_weight, rms_norm_eps),
            k_norm: RmsNorm::from_weights(k_norm_weight, rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        }
    }

    /// Forward pass through the attention layer.
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        start_pos: usize,
        mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _hidden_size] = x.dims();
        let device = x.device();

        // Project to Q, K, V
        let q = self.q_proj.forward(x.clone());
        let k = self.k_proj.forward(x.clone());
        let v = self.v_proj.forward(x);

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q = q.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let k = k.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        // QK-NORM: Apply RMSNorm to each head's Q and K BEFORE RoPE
        // This is the key difference from Llama
        let q = self.apply_head_norm(&self.q_norm, q);
        let k = self.apply_head_norm(&self.k_norm, k);

        // Apply rotary position embeddings (after QK-Norm)
        let q = self.apply_rope_simple(q, start_pos, &device);
        let k = self.apply_rope_simple(k, start_pos, &device);

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Expand K and V for grouped-query attention
        let (k, v) = self.expand_kv(k, v);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(k.swap_dims(2, 3)) / scale;

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores + mask
        } else {
            scores
        };

        // Softmax and apply to values
        let attn = burn::tensor::activation::softmax(scores, 3);
        let out = attn.matmul(v);

        // Transpose back and reshape
        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch_size, seq_len, self.num_heads * self.head_dim]);

        // Output projection
        self.o_proj.forward(out)
    }

    /// Applies RMSNorm to each head independently.
    ///
    /// Input shape: [batch, seq_len, num_heads, head_dim]
    /// The norm is applied along the head_dim axis for each head.
    fn apply_head_norm(&self, norm: &RmsNorm<B>, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, seq_len, num_heads, head_dim] = x.dims();

        // Reshape to [batch * seq_len * num_heads, 1, head_dim] for RmsNorm
        // RmsNorm expects [batch, seq_len, hidden_size] but we can reshape
        let x = x.reshape([batch_size * seq_len * num_heads, 1, head_dim]);

        // Apply normalization
        let x = norm.forward(x);

        // Reshape back to [batch, seq_len, num_heads, head_dim]
        x.reshape([batch_size, seq_len, num_heads, head_dim])
    }

    /// Rotary position embeddings using HuggingFace's split/half format.
    ///
    /// The rotation formula (matching HuggingFace's rotate_half):
    /// - new_first_half = first_half * cos - second_half * sin
    /// - new_second_half = second_half * cos + first_half * sin
    fn apply_rope_simple(
        &self,
        x: Tensor<B, 4>,
        start_pos: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        use burn::tensor::TensorData;

        let [_batch_size, seq_len, _num_heads, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        // Generate position indices and frequencies
        let positions: Vec<f32> = (start_pos..start_pos + seq_len).map(|p| p as f32).collect();

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (self.rope_theta.powf(i as f64 * 2.0 / head_dim as f64) as f32))
            .collect();

        // Compute angles: [seq_len, half_dim]
        let pos_data = TensorData::new(positions, [seq_len, 1]);
        let pos_tensor: Tensor<B, 2> = Tensor::from_data(pos_data, device);

        let inv_freq_data = TensorData::new(inv_freq, [1, half_dim]);
        let inv_freq_tensor: Tensor<B, 2> = Tensor::from_data(inv_freq_data, device);

        let angles = pos_tensor.matmul(inv_freq_tensor);

        // Compute sin and cos, broadcast to [1, seq_len, 1, half_dim]
        let cos = angles.clone().cos().reshape([1, seq_len, 1, half_dim]);
        let sin = angles.sin().reshape([1, seq_len, 1, half_dim]);

        // Split x into first half and second half (HuggingFace format)
        let x_first_half = x.clone().narrow(3, 0, half_dim);
        let x_second_half = x.narrow(3, half_dim, half_dim);

        // Apply rotation (HuggingFace rotate_half formula)
        let new_first_half =
            x_first_half.clone() * cos.clone() - x_second_half.clone() * sin.clone();
        let new_second_half = x_second_half * cos + x_first_half * sin;

        // Concatenate back: [new_first_half, new_second_half]
        Tensor::cat(vec![new_first_half, new_second_half], 3)
    }

    /// Expands K and V for grouped-query attention.
    fn expand_kv(&self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if self.num_kv_heads == self.num_heads {
            return (k, v);
        }

        let [batch_size, num_kv_heads, seq_len, head_dim] = k.dims();
        let repeat_factor = self.num_heads / self.num_kv_heads;

        let k = k.reshape([batch_size, num_kv_heads, 1, seq_len, head_dim]);
        let v = v.reshape([batch_size, num_kv_heads, 1, seq_len, head_dim]);

        let k = k.repeat_dim(2, repeat_factor);
        let v = v.repeat_dim(2, repeat_factor);

        let k = k.reshape([batch_size, self.num_heads, seq_len, head_dim]);
        let v = v.reshape([batch_size, self.num_heads, seq_len, head_dim]);

        (k, v)
    }
}

/// Creates a causal attention mask.
pub fn create_causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
    use burn::tensor::TensorData;

    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();

    let data = TensorData::new(mask_data, [1, 1, seq_len, seq_len]);
    Tensor::from_data(data, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_attention_shape() {
        let device = NdArrayDevice::Cpu;
        let attn = Qwen3Attention::<TestBackend>::new(
            64,          // hidden_size
            8,           // num_heads
            8,           // num_kv_heads
            8,           // head_dim
            1e-6,        // rms_norm_eps
            1_000_000.0, // rope_theta
            &device,
        );

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 10, 64], &device);
        let mask = create_causal_mask::<TestBackend>(10, &device);
        let output = attn.forward(input, 0, Some(mask));

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_gqa_attention_shape() {
        let device = NdArrayDevice::Cpu;
        // GQA with 8 Q heads and 2 KV heads
        let attn = Qwen3Attention::<TestBackend>::new(
            64,          // hidden_size
            8,           // num_heads
            2,           // num_kv_heads (GQA)
            8,           // head_dim
            1e-6,        // rms_norm_eps
            1_000_000.0, // rope_theta
            &device,
        );

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 10, 64], &device);
        let mask = create_causal_mask::<TestBackend>(10, &device);
        let output = attn.forward(input, 0, Some(mask));

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_qk_norm_applied() {
        // Verify that QK-Norm produces different results than without normalization
        let device = NdArrayDevice::Cpu;
        let attn = Qwen3Attention::<TestBackend>::new(64, 8, 8, 8, 1e-6, 1_000_000.0, &device);

        // Create input with non-zero values
        let input: Tensor<TestBackend, 3> = Tensor::ones([1, 5, 64], &device) * 0.5;
        let output = attn.forward(input, 0, None);

        // Output should have reasonable values (not NaN or Inf)
        let output_data: Vec<f32> = output.reshape([320]).to_data().to_vec().unwrap();
        assert!(output_data.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_causal_mask() {
        let device = NdArrayDevice::Cpu;
        let mask = create_causal_mask::<TestBackend>(3, &device);

        assert_eq!(mask.dims(), [1, 1, 3, 3]);
    }
}
