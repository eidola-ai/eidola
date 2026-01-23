//! Token embedding layer for Llama.

use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::prelude::*;

/// Token embedding layer.
///
/// Maps token IDs to dense vectors.
#[derive(Module, Debug)]
pub struct Embedding<B: Backend> {
    /// Embedding weight matrix [vocab_size, hidden_size].
    weight: Param<Tensor<B, 2>>,
}

impl<B: Backend> Embedding<B> {
    /// Creates a new embedding layer.
    ///
    /// # Arguments
    ///
    /// * `vocab_size` - Number of tokens in vocabulary
    /// * `hidden_size` - Dimension of embedding vectors
    /// * `device` - Device to create parameters on
    pub fn new(vocab_size: usize, hidden_size: usize, device: &B::Device) -> Self {
        // Initialize with normal distribution
        let weight: Param<Tensor<B, 2>> = Initializer::Normal {
            mean: 0.0,
            std: 0.02,
        }
        .init([vocab_size, hidden_size], device);

        Self { weight }
    }

    /// Forward pass - look up embeddings for token IDs.
    ///
    /// # Arguments
    ///
    /// * `ids` - Token IDs tensor of shape [batch, seq_len]
    ///
    /// # Returns
    ///
    /// Embedding tensor of shape [batch, seq_len, hidden_size]
    pub fn forward(&self, ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = ids.dims();

        // Flatten to 1D for gathering
        let flat_ids = ids.reshape([batch_size * seq_len]);

        // Select rows from embedding matrix
        let embeddings = self.weight.val().select(0, flat_ids);

        // Reshape back to [batch, seq_len, hidden_size]
        let hidden_size = self.weight.dims()[1];
        embeddings.reshape([batch_size, seq_len, hidden_size])
    }

    /// Returns the embedding weight matrix.
    pub fn weight(&self) -> &Param<Tensor<B, 2>> {
        &self.weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::{NdArray, NdArrayDevice};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_embedding_shape() {
        let device = NdArrayDevice::Cpu;
        let embedding = Embedding::<TestBackend>::new(100, 64, &device);

        let ids: Tensor<TestBackend, 2, Int> = Tensor::from_ints([[1, 2, 3], [4, 5, 6]], &device);
        let output = embedding.forward(ids);

        assert_eq!(output.dims(), [2, 3, 64]);
    }
}
