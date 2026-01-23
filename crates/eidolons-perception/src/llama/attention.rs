//! Multi-head attention with Rotary Position Embeddings (RoPE) for Llama.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;

/// Multi-head attention with Rotary Position Embeddings.
///
/// Supports grouped-query attention (GQA) where num_kv_heads < num_attention_heads.
#[derive(Module, Debug)]
pub struct LlamaAttention<B: Backend> {
    /// Query projection
    q_proj: Linear<B>,
    /// Key projection
    k_proj: Linear<B>,
    /// Value projection
    v_proj: Linear<B>,
    /// Output projection
    o_proj: Linear<B>,

    /// Number of attention heads
    #[module(skip)]
    num_heads: usize,
    /// Number of key-value heads (for GQA)
    #[module(skip)]
    num_kv_heads: usize,
    /// Head dimension
    #[module(skip)]
    head_dim: usize,
    /// Rope theta for positional encoding
    #[module(skip)]
    rope_theta: f64,
}

impl<B: Backend> LlamaAttention<B> {
    /// Creates a new attention layer.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        rope_theta: f64,
        device: &B::Device,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
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

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
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

        // Apply rotary position embeddings (simplified version)
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

    /// Simplified rotary position embeddings.
    /// Uses a simpler implementation that avoids complex tensor slicing.
    fn apply_rope_simple(
        &self,
        x: Tensor<B, 4>,
        start_pos: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        use burn::tensor::TensorData;

        let [batch_size, seq_len, num_heads, head_dim] = x.dims();
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

        // Split x into even and odd parts using narrow
        // x shape: [batch, seq_len, num_heads, head_dim]
        // We need to interleave the rotation, but for simplicity let's use a
        // reshape-based approach that splits the last dim in half

        // Reshape to [batch, seq_len, num_heads, half_dim, 2]
        let x_paired = x.reshape([batch_size, seq_len, num_heads, half_dim, 2]);

        // Extract even (index 0) and odd (index 1) elements using narrow
        let x_even = x_paired
            .clone()
            .narrow(4, 0, 1)
            .reshape([batch_size, seq_len, num_heads, half_dim]);
        let x_odd = x_paired
            .narrow(4, 1, 1)
            .reshape([batch_size, seq_len, num_heads, half_dim]);

        // Apply rotation: [cos, -sin; sin, cos] to [even; odd]
        let x_even_rot = x_even.clone() * cos.clone() - x_odd.clone() * sin.clone();
        let x_odd_rot = x_even * sin + x_odd * cos;

        // Interleave back: stack along last dim then reshape
        let x_even_rot = x_even_rot.reshape([batch_size, seq_len, num_heads, half_dim, 1]);
        let x_odd_rot = x_odd_rot.reshape([batch_size, seq_len, num_heads, half_dim, 1]);
        let x_rot = Tensor::cat(vec![x_even_rot, x_odd_rot], 4);

        x_rot.reshape([batch_size, seq_len, num_heads, head_dim])
    }

    /// Expands K and V for grouped-query attention.
    fn expand_kv(&self, k: Tensor<B, 4>, v: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 4>) {
        if self.num_kv_heads == self.num_heads {
            return (k, v);
        }

        let [batch_size, _num_kv_heads, seq_len, head_dim] = k.dims();
        let repeat_factor = self.num_heads / self.num_kv_heads;

        // Repeat K and V along the head dimension
        let k = k.repeat_dim(1, repeat_factor);
        let v = v.repeat_dim(1, repeat_factor);

        // Reshape to correct dimensions
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
        let attn = LlamaAttention::<TestBackend>::new(64, 8, 8, 10000.0, &device);

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 10, 64], &device);
        let mask = create_causal_mask::<TestBackend>(10, &device);
        let output = attn.forward(input, 0, Some(mask));

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_gqa_attention_shape() {
        let device = NdArrayDevice::Cpu;
        let attn = LlamaAttention::<TestBackend>::new(64, 8, 2, 10000.0, &device);

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 10, 64], &device);
        let mask = create_causal_mask::<TestBackend>(10, &device);
        let output = attn.forward(input, 0, Some(mask));

        assert_eq!(output.dims(), [2, 10, 64]);
    }

    #[test]
    fn test_causal_mask() {
        let device = NdArrayDevice::Cpu;
        let mask = create_causal_mask::<TestBackend>(3, &device);

        assert_eq!(mask.dims(), [1, 1, 3, 3]);
    }
}
