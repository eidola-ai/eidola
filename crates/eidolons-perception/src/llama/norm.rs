//! RMS Normalization layer.

use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::prelude::*;

/// RMS Normalization layer.
///
/// Unlike LayerNorm, RMSNorm only normalizes by the root mean square,
/// without centering (subtracting mean).
#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    /// Scale parameter (gamma).
    weight: Param<Tensor<B, 1>>,
    /// Epsilon for numerical stability.
    #[module(skip)]
    eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    /// Creates a new RMSNorm layer.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - The dimension of the input tensor
    /// * `eps` - Small value for numerical stability
    /// * `device` - The device to create the parameters on
    pub fn new(hidden_size: usize, eps: f64, device: &B::Device) -> Self {
        // Initialize weight to ones
        let weight: Param<Tensor<B, 1>> = Initializer::Ones.init([hidden_size], device);

        Self { weight, eps }
    }

    /// Forward pass of RMS normalization.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    ///
    /// Normalized tensor of the same shape.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Compute RMS: sqrt(mean(x^2) + eps)
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let rms = (variance + self.eps).sqrt();

        // Normalize and scale
        let x_norm = x / rms;

        // Apply learned scale (weight)
        // weight shape: [hidden_size] -> need to broadcast
        x_norm * self.weight.val().unsqueeze::<3>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_rms_norm_shape() {
        let device = NdArrayDevice::Cpu;
        let norm = RmsNorm::<TestBackend>::new(64, 1e-5, &device);

        let input: Tensor<TestBackend, 3> = Tensor::zeros([2, 10, 64], &device);
        let output = norm.forward(input);

        assert_eq!(output.dims(), [2, 10, 64]);
    }
}
