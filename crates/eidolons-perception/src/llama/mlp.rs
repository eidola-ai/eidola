//! Feed-forward network (MLP) with SwiGLU activation for Llama.

use burn::module::{Module, Param};
use burn::nn::{Initializer, Linear, LinearConfig, LinearRecord};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

/// Creates a Linear layer from pre-loaded weights (no bias).
///
/// Expects weights in Burn format [in_features, out_features].
fn linear_from_weight<B: Backend>(weight: Param<Tensor<B, 2>>) -> Linear<B> {
    // Burn stores weights as [in_features, out_features]
    let [in_features, out_features] = weight.dims();
    let device = weight.device();

    let linear = LinearConfig::new(in_features, out_features)
        .with_bias(false)
        .init(&device);

    let record = LinearRecord {
        weight,
        bias: None,
    };
    linear.load_record(record)
}

/// Feed-forward network with SwiGLU activation.
///
/// Llama uses the SwiGLU variant:
/// FFN(x) = (Swish(x W_gate) * (x W_up)) W_down
#[derive(Module, Debug)]
pub struct LlamaMlp<B: Backend> {
    /// Gate projection: hidden_size -> intermediate_size
    gate_proj: Linear<B>,
    /// Up projection: hidden_size -> intermediate_size
    up_proj: Linear<B>,
    /// Down projection: intermediate_size -> hidden_size
    down_proj: Linear<B>,
}

impl<B: Backend> LlamaMlp<B> {
    /// Creates a new MLP layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Input/output dimension
    /// * `intermediate_size` - Hidden dimension of FFN
    /// * `device` - Device to create parameters on
    pub fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        let initializer = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        };

        let gate_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);

        let up_proj = LinearConfig::new(hidden_size, intermediate_size)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);

        let down_proj = LinearConfig::new(intermediate_size, hidden_size)
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);

        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Creates an MLP layer from pre-loaded weights.
    ///
    /// # Arguments
    ///
    /// * `gate_proj_weight` - Gate projection weights [hidden_size, intermediate_size]
    /// * `up_proj_weight` - Up projection weights [hidden_size, intermediate_size]
    /// * `down_proj_weight` - Down projection weights [intermediate_size, hidden_size]
    pub fn from_weights(
        gate_proj_weight: Param<Tensor<B, 2>>,
        up_proj_weight: Param<Tensor<B, 2>>,
        down_proj_weight: Param<Tensor<B, 2>>,
    ) -> Self {
        Self {
            gate_proj: linear_from_weight(gate_proj_weight),
            up_proj: linear_from_weight(up_proj_weight),
            down_proj: linear_from_weight(down_proj_weight),
        }
    }

    /// Forward pass through the MLP.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Output tensor of the same shape.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // SwiGLU: (swish(gate(x)) * up(x)) -> down
        let gate = self.gate_proj.forward(x.clone());
        let up = self.up_proj.forward(x);

        // Swish activation: x * sigmoid(x)
        let gate_swish = gate.clone() * sigmoid(gate);

        // Element-wise multiply and project down
        self.down_proj.forward(gate_swish * up)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mlp_shape() {
        let device = NdArrayDevice::Cpu;
        let mlp = LlamaMlp::<TestBackend>::new(64, 128, &device);

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 10, 64], &device);
        let output = mlp.forward(input);

        assert_eq!(output.dims(), [2, 10, 64]);
    }
}
