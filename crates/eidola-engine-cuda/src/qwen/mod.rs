//! Qwen3.8-27B (`eidola-engine-cuda/AGENTS.md` → Qwen3.8): the kernels' parameter
//! structs ([`abi`]), the kernel families and their grids ([`kernels`]), the devices the
//! executor runs on ([`device`], [`host`]), its weights ([`weights`]) and the executor
//! ([`executor`]).

pub mod abi;
pub mod attention;
pub mod cuda;
pub mod device;
pub mod executor;
pub mod host;
pub mod kernels;
pub mod plan;
pub mod weights;

pub use device::{QwenDevice, QwenProfile};
pub use executor::{
    CudaGraphs, DecodePath, DecodeStats, QwenCudaExecutor, QwenCudaExecutorConfig, StateStrategy,
};
