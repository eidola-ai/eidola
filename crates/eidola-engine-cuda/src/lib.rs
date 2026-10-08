//! The CUDA executor for the MiMo-V2.6 engine.
//!
//! Host code is Rust over the CUDA driver API (`cudarc`, loaded at run time);
//! there is no CUDA runtime library and no runtime compilation. Every kernel
//! comes from the ahead-of-time, hash-manifested build in
//! `eidola-engine-kernels` and is loaded only after its bytes match the
//! manifest ([`module`]).

pub mod attention;
pub mod bf16;
pub mod device;
pub mod engine_ops;
pub mod executor;
pub mod gemm;
pub mod kv;
pub mod launch;
pub mod model;
pub mod module;
pub mod moe_gemm;
pub mod ops;
pub mod sampler;
pub mod support;
pub mod tma;
pub mod weights;

use std::fmt;

pub use device::{DeviceInfo, Gpu, ImageArch};
pub use executor::{CudaExecutor, CudaExecutorConfig};
pub use module::{ImageSource, Kernel, KernelDir, KernelModule};
pub use support::{Unsupported, check_supported};

/// A CUDA executor failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaError {
    /// A driver call, a launch precondition or a malformed input failed.
    Failed(String),
    /// The model is outside the one configuration the kernels are built for.
    Unsupported(Unsupported),
}

impl CudaError {
    pub fn new(what: impl Into<String>) -> CudaError {
        CudaError::Failed(what.into())
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::Failed(what) => write!(f, "cuda: {what}"),
            CudaError::Unsupported(u) => write!(f, "cuda: unsupported model: {u}"),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> CudaError {
        CudaError::Failed(format!("{e:?}"))
    }
}

impl From<Unsupported> for CudaError {
    fn from(u: Unsupported) -> CudaError {
        CudaError::Unsupported(u)
    }
}

pub type Result<T> = std::result::Result<T, CudaError>;
