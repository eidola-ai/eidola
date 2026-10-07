//! The CUDA executor for the MiMo-V2.6 engine.
//!
//! Host code is Rust over the CUDA driver API (`cudarc`, loaded at run time);
//! there is no CUDA runtime library and no runtime compilation. Every kernel
//! comes from the ahead-of-time, hash-manifested build in
//! `eidola-engine-kernels` and is loaded only after its bytes match the
//! manifest ([`module`]).

pub mod bf16;
pub mod device;
pub mod module;
pub mod ops;

use std::fmt;

pub use device::{DeviceInfo, Gpu, ImageArch};
pub use module::{ImageSource, Kernel, KernelModule};

/// A CUDA executor failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaError(pub String);

impl CudaError {
    pub fn new(what: impl Into<String>) -> CudaError {
        CudaError(what.into())
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cuda: {}", self.0)
    }
}

impl std::error::Error for CudaError {}

impl From<cudarc::driver::DriverError> for CudaError {
    fn from(e: cudarc::driver::DriverError) -> CudaError {
        CudaError(format!("{e:?}"))
    }
}

pub type Result<T> = std::result::Result<T, CudaError>;
