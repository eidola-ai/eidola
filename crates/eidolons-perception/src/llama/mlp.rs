//! Feed-forward network (MLP) with SwiGLU activation for Llama.

use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::sigmoid;

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
