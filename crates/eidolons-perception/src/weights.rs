//! Weight loading utilities for Llama models from safetensors files.

use anyhow::{Context, Result};
use burn::module::Param;
use burn::prelude::*;
use burn::tensor::TensorData;
use half::f16;
use safetensors::SafeTensors;

/// Loads a tensor from safetensors by name and converts to the appropriate type.
pub fn load_tensor<B: Backend, const D: usize>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, D>> {
    let tensor_view = tensors
        .tensor(name)
        .with_context(|| format!("Failed to load tensor '{}'", name))?;

    let shape: Vec<usize> = tensor_view.shape().to_vec();
    let dtype = tensor_view.dtype();

    // Convert to f32 data
    let float_data: Vec<f32> = match dtype {
        safetensors::Dtype::F32 => {
            let data = tensor_view.data();
            // Safety: data is known to be f32 aligned from safetensors
            let float_slice: &[f32] = bytemuck::cast_slice(data);
            float_slice.to_vec()
        }
        safetensors::Dtype::F16 => {
            let data = tensor_view.data();
            // Convert from f16 bytes to f32
            let f16_slice: &[u16] = bytemuck::cast_slice(data);
            f16_slice.iter().map(|&x| f16::from_bits(x).to_f32()).collect()
        }
        safetensors::Dtype::BF16 => {
            let data = tensor_view.data();
            // Convert from bf16 bytes to f32
            let bf16_slice: &[u16] = bytemuck::cast_slice(data);
            bf16_slice
                .iter()
                .map(|&x| half::bf16::from_bits(x).to_f32())
                .collect()
        }
        other => anyhow::bail!("Unsupported tensor dtype: {:?}", other),
    };

    // Create tensor data with the correct shape
    let shape_arr: [usize; D] = shape
        .try_into()
        .map_err(|_| anyhow::anyhow!("Shape dimension mismatch for tensor '{}': expected {} dims", name, D))?;

    let tensor_data = TensorData::new(float_data, shape_arr);
    Ok(Tensor::from_data(tensor_data, device))
}

/// Loads a 1D tensor as a Param.
pub fn load_param_1d<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Param<Tensor<B, 1>>> {
    let tensor = load_tensor::<B, 1>(tensors, name, device)?;
    Ok(Param::from_tensor(tensor))
}

/// Loads a 2D tensor as a Param.
pub fn load_param_2d<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Param<Tensor<B, 2>>> {
    let tensor = load_tensor::<B, 2>(tensors, name, device)?;
    Ok(Param::from_tensor(tensor))
}

/// Loads a 2D tensor as a Param, transposing from HuggingFace format to Burn format.
///
/// HuggingFace/PyTorch stores linear weights as [out_features, in_features].
/// Burn stores linear weights as [in_features, out_features].
/// This function loads and transposes the weights accordingly.
pub fn load_linear_weight<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Param<Tensor<B, 2>>> {
    let tensor = load_tensor::<B, 2>(tensors, name, device)?;
    // Transpose from [out, in] to [in, out]
    let transposed = tensor.transpose();
    Ok(Param::from_tensor(transposed))
}

/// Weight loader for Llama models.
pub struct LlamaWeightLoader<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> LlamaWeightLoader<'a> {
    /// Creates a new weight loader from safetensors file data.
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes of the safetensors file (must outlive the loader)
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(data)
            .context("Failed to deserialize safetensors file")?;
        Ok(Self { tensors })
    }

    /// Lists all tensor names in the file.
    pub fn tensor_names(&self) -> Vec<String> {
        self.tensors.names().into_iter().cloned().collect()
    }

    /// Loads the token embedding weights.
    pub fn load_embed_tokens<B: Backend>(&self, device: &B::Device) -> Result<Param<Tensor<B, 2>>> {
        load_param_2d(&self.tensors, "model.embed_tokens.weight", device)
    }

    /// Loads the final layer norm weights.
    pub fn load_final_norm<B: Backend>(&self, device: &B::Device) -> Result<Param<Tensor<B, 1>>> {
        load_param_1d(&self.tensors, "model.norm.weight", device)
    }

    /// Loads the LM head weights.
    /// Note: Some models tie embed_tokens and lm_head weights.
    pub fn load_lm_head<B: Backend>(&self, device: &B::Device) -> Result<Param<Tensor<B, 2>>> {
        // Try lm_head.weight first, fall back to embed_tokens if tied
        match load_linear_weight(&self.tensors, "lm_head.weight", device) {
            Ok(weights) => Ok(weights),
            Err(_) => {
                // Weights might be tied - use embed_tokens (also needs transpose for lm_head use)
                load_linear_weight(&self.tensors, "model.embed_tokens.weight", device)
            }
        }
    }

    /// Loads attention weights for a specific layer.
    /// Linear weights are transposed from HuggingFace format [out, in] to Burn format [in, out].
    pub fn load_attention_weights<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<AttentionWeights<B>> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        Ok(AttentionWeights {
            q_proj: load_linear_weight(&self.tensors, &format!("{}.q_proj.weight", prefix), device)?,
            k_proj: load_linear_weight(&self.tensors, &format!("{}.k_proj.weight", prefix), device)?,
            v_proj: load_linear_weight(&self.tensors, &format!("{}.v_proj.weight", prefix), device)?,
            o_proj: load_linear_weight(&self.tensors, &format!("{}.o_proj.weight", prefix), device)?,
        })
    }

    /// Loads MLP weights for a specific layer.
    /// Linear weights are transposed from HuggingFace format [out, in] to Burn format [in, out].
    pub fn load_mlp_weights<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<MlpWeights<B>> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        Ok(MlpWeights {
            gate_proj: load_linear_weight(&self.tensors, &format!("{}.gate_proj.weight", prefix), device)?,
            up_proj: load_linear_weight(&self.tensors, &format!("{}.up_proj.weight", prefix), device)?,
            down_proj: load_linear_weight(&self.tensors, &format!("{}.down_proj.weight", prefix), device)?,
        })
    }

    /// Loads layer norm weights for a specific layer.
    pub fn load_layer_norms<B: Backend>(
        &self,
        layer_idx: usize,
        device: &B::Device,
    ) -> Result<LayerNormWeights<B>> {
        let prefix = format!("model.layers.{}", layer_idx);

        Ok(LayerNormWeights {
            input_layernorm: load_param_1d(
                &self.tensors,
                &format!("{}.input_layernorm.weight", prefix),
                device,
            )?,
            post_attention_layernorm: load_param_1d(
                &self.tensors,
                &format!("{}.post_attention_layernorm.weight", prefix),
                device,
            )?,
        })
    }
}

/// Attention layer weights.
pub struct AttentionWeights<B: Backend> {
    pub q_proj: Param<Tensor<B, 2>>,
    pub k_proj: Param<Tensor<B, 2>>,
    pub v_proj: Param<Tensor<B, 2>>,
    pub o_proj: Param<Tensor<B, 2>>,
}

/// MLP layer weights.
pub struct MlpWeights<B: Backend> {
    pub gate_proj: Param<Tensor<B, 2>>,
    pub up_proj: Param<Tensor<B, 2>>,
    pub down_proj: Param<Tensor<B, 2>>,
}

/// Layer normalization weights.
pub struct LayerNormWeights<B: Backend> {
    pub input_layernorm: Param<Tensor<B, 1>>,
    pub post_attention_layernorm: Param<Tensor<B, 1>>,
}

use crate::llama::{
    Llama, LlamaConfig,
    attention::LlamaAttention,
    embedding::Embedding,
    mlp::LlamaMlp,
    model::LlamaLayer,
    norm::RmsNorm,
};

/// Loads a complete Llama model from safetensors data.
///
/// # Arguments
///
/// * `data` - Raw bytes of the safetensors file
/// * `config` - Llama model configuration
/// * `device` - Device to load tensors on
///
/// # Returns
///
/// A Llama model with weights loaded from the safetensors file.
pub fn load_llama_from_safetensors<B: Backend>(
    data: &[u8],
    config: &LlamaConfig,
    device: &B::Device,
) -> Result<Llama<B>> {
    let loader = LlamaWeightLoader::new(data)?;

    // Load embeddings
    let embed_tokens_weight = loader.load_embed_tokens(device)?;
    let embed_tokens = Embedding::from_weights(embed_tokens_weight);

    // Load transformer layers
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for layer_idx in 0..config.num_hidden_layers {
        let layer = load_llama_layer(&loader, layer_idx, config, device)?;
        layers.push(layer);
    }

    // Load final norm
    let norm_weight = loader.load_final_norm(device)?;
    let norm = RmsNorm::from_weights(norm_weight, config.rms_norm_eps);

    // Load LM head
    let lm_head_weight = loader.load_lm_head(device)?;

    Ok(Llama::from_weights(
        embed_tokens,
        layers,
        norm,
        lm_head_weight,
        config.vocab_size,
    ))
}

/// Loads a single transformer layer from the weight loader.
fn load_llama_layer<B: Backend>(
    loader: &LlamaWeightLoader<'_>,
    layer_idx: usize,
    config: &LlamaConfig,
    device: &B::Device,
) -> Result<LlamaLayer<B>> {
    // Load attention weights
    let attn_weights = loader.load_attention_weights(layer_idx, device)?;
    let self_attn = LlamaAttention::from_weights(
        attn_weights.q_proj,
        attn_weights.k_proj,
        attn_weights.v_proj,
        attn_weights.o_proj,
        config.num_attention_heads,
        config.num_kv_heads(),
        config.rope_theta,
    );

    // Load MLP weights
    let mlp_weights = loader.load_mlp_weights(layer_idx, device)?;
    let mlp = LlamaMlp::from_weights(
        mlp_weights.gate_proj,
        mlp_weights.up_proj,
        mlp_weights.down_proj,
    );

    // Load layer norms
    let norm_weights = loader.load_layer_norms(layer_idx, device)?;
    let input_layernorm = RmsNorm::from_weights(norm_weights.input_layernorm, config.rms_norm_eps);
    let post_attention_layernorm = RmsNorm::from_weights(
        norm_weights.post_attention_layernorm,
        config.rms_norm_eps,
    );

    Ok(LlamaLayer::from_weights(
        self_attn,
        mlp,
        input_layernorm,
        post_attention_layernorm,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::{Linear, LinearConfig};
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_shape_conversion() {
        let shape: Vec<usize> = vec![32000, 2048];
        let result: Result<[usize; 2], _> = shape.try_into();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), [32000, 2048]);
    }

    #[test]
    fn test_burn_linear_weight_shape() {
        // Test what shape Burn's Linear layer expects for its weights
        let device = NdArrayDevice::Cpu;

        // Create a linear layer: 2048 input features -> 256 output features
        let linear: Linear<TestBackend> = LinearConfig::new(2048, 256)
            .with_bias(false)
            .init(&device);

        // Check the shape of the weight
        let weight_shape = linear.weight.dims();
        println!("Burn Linear weight shape (in=2048, out=256): {:?}", weight_shape);

        // Burn stores weights as [in_features, out_features]
        // This is OPPOSITE to PyTorch which stores [out_features, in_features]
        assert_eq!(weight_shape, [2048, 256], "Burn stores weights as [in, out]");
    }

    #[test]
    fn test_linear_forward_with_manual_weights() {
        // Test that a Linear layer with manually set weights produces correct output shapes
        use burn::nn::LinearRecord;
        use burn::tensor::TensorData;

        let device = NdArrayDevice::Cpu;

        let in_features = 64;
        let out_features = 32;

        // Create weights in Burn format: [in_features, out_features]
        let weight_data: Vec<f32> = vec![0.1; in_features * out_features];
        let weight_tensor: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(weight_data, [in_features, out_features]),
            &device,
        );

        // Create a Linear layer with matching dimensions
        let linear: Linear<TestBackend> = LinearConfig::new(in_features, out_features)
            .with_bias(false)
            .init(&device);

        // Verify native Linear weight shape
        println!("Native Linear weight shape: {:?}", linear.weight.dims());
        assert_eq!(linear.weight.dims(), [in_features, out_features]);

        // Load our weights into the linear layer
        let record = LinearRecord {
            weight: Param::from_tensor(weight_tensor),
            bias: None,
        };
        let linear = linear.load_record(record);

        // Test forward pass
        let input: Tensor<TestBackend, 2> = Tensor::ones([4, in_features], &device);
        let output = linear.forward(input);

        println!("Input shape: {:?}, Output shape: {:?}", [4, in_features], output.dims());
        assert_eq!(output.dims(), [4, out_features]);
    }

    #[test]
    fn test_attention_weight_shapes() {
        // Test that attention layer works with correct weight shapes
        let device = NdArrayDevice::Cpu;

        // TinyLlama dimensions
        let hidden_size = 2048;
        let num_heads = 32;
        let num_kv_heads = 4;
        let head_dim = hidden_size / num_heads; // 64

        // Burn format: [in_features, out_features]
        // Q projection: hidden_size -> num_heads * head_dim
        let q_out = num_heads * head_dim; // 2048
        let q_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([hidden_size, q_out], &device);

        // K projection: hidden_size -> num_kv_heads * head_dim
        let k_out = num_kv_heads * head_dim; // 256
        let k_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([hidden_size, k_out], &device);

        // V projection: same as K
        let v_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([hidden_size, k_out], &device);

        // O projection: num_heads * head_dim -> hidden_size
        let o_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([q_out, hidden_size], &device);

        println!("Q weight shape (Burn format [in, out]): {:?}", q_weight.dims());
        println!("K weight shape (Burn format [in, out]): {:?}", k_weight.dims());
        println!("V weight shape (Burn format [in, out]): {:?}", v_weight.dims());
        println!("O weight shape (Burn format [in, out]): {:?}", o_weight.dims());

        // Create attention layer from weights
        let attn = LlamaAttention::from_weights(
            Param::from_tensor(q_weight),
            Param::from_tensor(k_weight),
            Param::from_tensor(v_weight),
            Param::from_tensor(o_weight),
            num_heads,
            num_kv_heads,
            10000.0,
        );

        // Test forward pass with small input
        let batch_size = 1;
        let seq_len = 10;
        let input: Tensor<TestBackend, 3> = Tensor::zeros([batch_size, seq_len, hidden_size], &device);
        let mask = crate::llama::attention::create_causal_mask::<TestBackend>(seq_len, &device);

        let output = attn.forward(input, 0, Some(mask));
        assert_eq!(output.dims(), [batch_size, seq_len, hidden_size]);
    }

    #[test]
    fn test_mlp_weight_shapes() {
        let device = NdArrayDevice::Cpu;

        // TinyLlama dimensions
        let hidden_size = 2048;
        let intermediate_size = 5632;

        // Burn format: [in_features, out_features]
        // Gate/Up: hidden_size -> intermediate_size
        let gate_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([hidden_size, intermediate_size], &device);
        let up_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([hidden_size, intermediate_size], &device);

        // Down: intermediate_size -> hidden_size
        let down_weight: Tensor<TestBackend, 2> =
            Tensor::zeros([intermediate_size, hidden_size], &device);

        let mlp = LlamaMlp::from_weights(
            Param::from_tensor(gate_weight),
            Param::from_tensor(up_weight),
            Param::from_tensor(down_weight),
        );

        // Test forward
        let input: Tensor<TestBackend, 3> = Tensor::zeros([1, 10, hidden_size], &device);
        let output = mlp.forward(input);
        assert_eq!(output.dims(), [1, 10, hidden_size]);
    }

    #[test]
    fn test_huggingface_weight_transpose() {
        // Test that transposing HuggingFace weights to Burn format works correctly
        use burn::tensor::TensorData;

        let device = NdArrayDevice::Cpu;

        let in_features = 4;
        let out_features = 2;

        // Simulate HuggingFace weight format: [out_features, in_features]
        // Values are arranged so we can verify transpose is correct
        // Row 0: [1, 2, 3, 4]
        // Row 1: [5, 6, 7, 8]
        let hf_weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let hf_weight: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(hf_weight_data.clone(), [out_features, in_features]),
            &device,
        );

        println!("HuggingFace weight shape: {:?}", hf_weight.dims());
        assert_eq!(hf_weight.dims(), [out_features, in_features]);

        // Transpose to Burn format: [in_features, out_features]
        let burn_weight = hf_weight.transpose();
        println!("Burn weight shape after transpose: {:?}", burn_weight.dims());
        assert_eq!(burn_weight.dims(), [in_features, out_features]);

        // Verify values are correctly transposed
        // Should now be:
        // Row 0: [1, 5]
        // Row 1: [2, 6]
        // Row 2: [3, 7]
        // Row 3: [4, 8]
        let burn_data = burn_weight.to_data();
        let values: Vec<f32> = burn_data.to_vec().unwrap();
        assert_eq!(values, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn test_full_model_forward_with_small_config() {
        // Test a complete model forward pass with small dimensions
        let device = NdArrayDevice::Cpu;

        // Small test config
        let config = LlamaConfig {
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            rms_norm_eps: 1e-5,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            bos_token_id: 1,
            eos_token_id: 2,
        };

        // Create model with random weights
        let model = Llama::<TestBackend>::new(config.clone(), &device);

        // Test forward pass
        use burn::prelude::Int;
        let input_ids: Tensor<TestBackend, 2, Int> =
            Tensor::from_ints([[1, 2, 3, 4, 5]], &device);
        let output = model.forward(input_ids, 0);

        println!("Model output shape: {:?}", output.dims());
        assert_eq!(output.dims(), [1, 5, config.vocab_size]);
    }

    #[test]
    fn test_embedding_to_lm_head_direct() {
        // Test embedding -> norm -> lm_head directly (bypassing transformer layers)
        // This verifies the basic input/output pipeline works
        use burn::nn::{Linear, LinearConfig};

        let device = NdArrayDevice::Cpu;

        let vocab_size = 100;
        let hidden_size = 64;

        // Create embedding
        let embed = Embedding::<TestBackend>::new(vocab_size, hidden_size, &device);

        // Create norm
        let norm = RmsNorm::<TestBackend>::new(hidden_size, 1e-5, &device);

        // Create lm_head
        let lm_head: Linear<TestBackend> = LinearConfig::new(hidden_size, vocab_size)
            .with_bias(false)
            .init(&device);

        // Forward pass
        use burn::prelude::Int;
        let input_ids: Tensor<TestBackend, 2, Int> = Tensor::from_ints([[1]], &device);

        let x = embed.forward(input_ids);
        println!("After embedding: shape={:?}", x.dims());

        let x_data: Vec<f32> = x.clone().reshape([hidden_size]).to_data().to_vec().unwrap();
        let mean = x_data.iter().sum::<f32>() / x_data.len() as f32;
        let std = (x_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x_data.len() as f32).sqrt();
        println!("Embedding stats: mean={:.4}, std={:.4}", mean, std);

        let x = norm.forward(x);
        println!("After norm: shape={:?}", x.dims());

        let logits = lm_head.forward(x);
        let logits_shape = logits.dims();
        println!("After lm_head: shape={:?}", logits_shape);

        // Get top prediction
        let logits_flat = logits.reshape([vocab_size]);
        let logits_data: Vec<f32> = logits_flat.to_data().to_vec().unwrap();

        let mean = logits_data.iter().sum::<f32>() / logits_data.len() as f32;
        let std = (logits_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / logits_data.len() as f32).sqrt();
        println!("Logits stats: mean={:.4}, std={:.4}", mean, std);

        // This should work without errors
        assert_eq!(logits_shape, [1, 1, vocab_size]);
    }

    #[test]
    fn test_loaded_embedding_values() {
        // Test that loaded embeddings have expected properties
        use burn::tensor::TensorData;

        let device = NdArrayDevice::Cpu;

        let vocab_size = 100;
        let hidden_size = 64;

        // Create embedding with known values
        // In Burn, embedding weight is [vocab_size, hidden_size]
        let weight_data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| {
                let row = i / hidden_size;
                let col = i % hidden_size;
                // Make each row have a distinct pattern: row index normalized
                (row as f32) / (vocab_size as f32) + (col as f32) / (hidden_size as f32) * 0.1
            })
            .collect();

        let weight_tensor: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(weight_data.clone(), [vocab_size, hidden_size]),
            &device,
        );

        let embed = Embedding::from_weights(Param::from_tensor(weight_tensor));

        // Lookup token 5
        use burn::prelude::Int;
        let input_ids: Tensor<TestBackend, 2, Int> = Tensor::from_ints([[5]], &device);
        let output = embed.forward(input_ids);

        let output_data: Vec<f32> = output.reshape([hidden_size]).to_data().to_vec().unwrap();

        // Verify output matches row 5 of the weight matrix
        let expected_row_start = 5 * hidden_size;
        for col in 0..hidden_size {
            let expected = weight_data[expected_row_start + col];
            let actual = output_data[col];
            assert!(
                (expected - actual).abs() < 1e-5,
                "Embedding lookup mismatch at col {}: expected {}, got {}",
                col, expected, actual
            );
        }

        println!("✓ Embedding lookup produces correct values");
    }

    /// Test a single transformer layer to verify it produces reasonable outputs.
    #[test]
    fn test_single_layer_forward() {
        let device = NdArrayDevice::Cpu;

        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let intermediate_size = 128;
        let head_dim = hidden_size / num_heads;

        // Create attention weights in Burn format [in, out]
        let q_weight: Tensor<TestBackend, 2> = Tensor::ones([hidden_size, hidden_size], &device) * 0.01;
        let k_weight: Tensor<TestBackend, 2> = Tensor::ones([hidden_size, num_kv_heads * head_dim], &device) * 0.01;
        let v_weight: Tensor<TestBackend, 2> = Tensor::ones([hidden_size, num_kv_heads * head_dim], &device) * 0.01;
        let o_weight: Tensor<TestBackend, 2> = Tensor::ones([hidden_size, hidden_size], &device) * 0.01;

        // Create MLP weights
        let gate_weight: Tensor<TestBackend, 2> = Tensor::ones([hidden_size, intermediate_size], &device) * 0.01;
        let up_weight: Tensor<TestBackend, 2> = Tensor::ones([hidden_size, intermediate_size], &device) * 0.01;
        let down_weight: Tensor<TestBackend, 2> = Tensor::ones([intermediate_size, hidden_size], &device) * 0.01;

        // Create norm weights (all ones)
        let input_norm_weight: Tensor<TestBackend, 1> = Tensor::ones([hidden_size], &device);
        let post_norm_weight: Tensor<TestBackend, 1> = Tensor::ones([hidden_size], &device);

        // Create attention
        let attn = LlamaAttention::from_weights(
            Param::from_tensor(q_weight),
            Param::from_tensor(k_weight),
            Param::from_tensor(v_weight),
            Param::from_tensor(o_weight),
            num_heads,
            num_kv_heads,
            10000.0,
        );

        // Create MLP
        let mlp = LlamaMlp::from_weights(
            Param::from_tensor(gate_weight),
            Param::from_tensor(up_weight),
            Param::from_tensor(down_weight),
        );

        // Create norms
        let input_layernorm = RmsNorm::from_weights(Param::from_tensor(input_norm_weight), 1e-5);
        let post_attention_layernorm = RmsNorm::from_weights(Param::from_tensor(post_norm_weight), 1e-5);

        // Create layer
        let layer = LlamaLayer::from_weights(attn, mlp, input_layernorm, post_attention_layernorm);

        // Create input
        let batch_size = 1;
        let seq_len = 3;
        let input: Tensor<TestBackend, 3> = Tensor::ones([batch_size, seq_len, hidden_size], &device) * 0.1;

        println!("Input: shape={:?}", input.dims());
        let input_data: Vec<f32> = input.clone().reshape([batch_size * seq_len * hidden_size]).to_data().to_vec().unwrap();
        let mean = input_data.iter().sum::<f32>() / input_data.len() as f32;
        println!("Input mean: {:.4}", mean);

        // Create mask
        let mask = crate::llama::attention::create_causal_mask::<TestBackend>(seq_len, &device);

        // Forward
        let output = layer.forward(input, 0, Some(mask));

        println!("Output: shape={:?}", output.dims());
        let output_data: Vec<f32> = output.reshape([batch_size * seq_len * hidden_size]).to_data().to_vec().unwrap();
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let std = (output_data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / output_data.len() as f32).sqrt();
        let has_nan = output_data.iter().any(|v| v.is_nan());
        let has_inf = output_data.iter().any(|v| v.is_infinite());

        println!("Output stats: mean={:.4}, std={:.4}, has_nan={}, has_inf={}", mean, std, has_nan, has_inf);

        assert!(!has_nan, "Output contains NaN");
        assert!(!has_inf, "Output contains Inf");
        assert!(std > 0.0, "Output has zero variance");
    }
}
