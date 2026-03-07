use anyhow::{Context, Result};
use burn::backend::Autodiff;
use burn_ndarray::NdArray;
#[cfg(feature = "gpu")]
use burn_wgpu::{Wgpu, WgpuDevice};

pub mod generation;
pub mod model_manager;
pub mod qwen3;
pub mod tokenizer;

pub use generation::GenerationConfig;
pub use model_manager::{
    InferenceBackend, ModelArchitecture, ModelConfig, StreamChunk, TextGenerationModel,
};
pub use qwen3::{QuantizationMode, Qwen3Config, Qwen3Model};
pub use tokenizer::{ChatRole, FormatMessage, Qwen3Tokenizer};

/// Backend type aliases for convenience.
#[cfg(feature = "gpu")]
pub type WgpuBackend = Wgpu<f32, i32, u8>;
pub type NdArrayBackend = NdArray<f32>;
#[cfg(feature = "gpu")]
pub type AutodiffWgpuBackend = Autodiff<WgpuBackend>;
pub type AutodiffNdArrayBackend = Autodiff<NdArrayBackend>;

/// Represents the initialized compute backend.
#[derive(Debug, Clone)]
pub enum Backend {
    /// GPU-accelerated backend (Metal on macOS, Vulkan on Linux/Windows).
    /// Only available with the `gpu` feature.
    #[cfg(feature = "gpu")]
    Wgpu,
    /// CPU fallback using ndarray.
    NdArray,
}

/// Manages backend initialization and selection for ML inference.
#[derive(Debug)]
pub struct BackendManager {
    backend: Backend,
}

impl BackendManager {
    /// Returns which backend was initialized.
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Check if using GPU acceleration.
    pub fn is_gpu_accelerated(&self) -> bool {
        #[cfg(feature = "gpu")]
        if matches!(self.backend, Backend::Wgpu) {
            return true;
        }
        false
    }
}

/// Attempts to initialize the best available compute backend.
///
/// With `gpu` feature: tries WGPU (GPU via Metal/Vulkan) first, then falls back to NdArray (CPU).
/// Without `gpu` feature: uses NdArray (CPU) only.
pub fn init_backend() -> Result<BackendManager> {
    #[cfg(feature = "gpu")]
    {
        // Try WGPU backend first (provides Metal on macOS, Vulkan elsewhere)
        match try_init_wgpu() {
            Ok(()) => {
                return Ok(BackendManager {
                    backend: Backend::Wgpu,
                });
            }
            Err(e) => {
                eprintln!("WGPU backend unavailable: {e}. Falling back to CPU.");
            }
        }
    }

    // Fall back to NdArray (CPU)
    try_init_ndarray().context("Failed to initialize any backend")?;

    Ok(BackendManager {
        backend: Backend::NdArray,
    })
}

/// Attempts to initialize the WGPU backend.
#[cfg(feature = "gpu")]
fn try_init_wgpu() -> Result<()> {
    use std::panic;

    // WGPU may panic if no GPU adapter is available (e.g., in CI/sandbox environments).
    // We catch the panic and convert it to an error for graceful fallback.
    let result = panic::catch_unwind(|| {
        // Attempt to get default device - this will fail if no GPU is available
        let device = WgpuDevice::default();

        // Verify we can create a simple tensor on the device
        use burn::tensor::Tensor;
        let _tensor: Tensor<WgpuBackend, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);
    });

    result.map_err(|e| {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic during WGPU initialization".to_string()
        };
        anyhow::anyhow!("WGPU initialization panicked: {}", msg)
    })
}

/// Attempts to initialize the NdArray (CPU) backend.
fn try_init_ndarray() -> Result<()> {
    use burn::tensor::Tensor;
    use burn_ndarray::NdArrayDevice;

    let device = NdArrayDevice::Cpu;

    // Verify we can create a simple tensor
    let _tensor: Tensor<NdArray<f32>, 1> = Tensor::from_floats([1.0, 2.0, 3.0], &device);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_init() {
        let manager = init_backend().expect("Backend initialization should not crash");

        // Should have initialized some backend
        match manager.backend() {
            #[cfg(feature = "gpu")]
            Backend::Wgpu => {
                println!("Initialized with WGPU (GPU) backend");
                assert!(manager.is_gpu_accelerated());
            }
            Backend::NdArray => {
                println!("Initialized with NdArray (CPU) backend");
                assert!(!manager.is_gpu_accelerated());
            }
        }
    }
}
