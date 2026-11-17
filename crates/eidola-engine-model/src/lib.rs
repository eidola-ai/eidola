//! Model definitions for the engine: configuration, weight loading, and the
//! dense reference forward pass, one module per architecture.
//!
//! - [`mimo`] is MiMo-V2.6; [`deepseek`] is DeepSeek-V4.1; [`qwen`] is
//!   Qwen3.8. None knows the others exist.
//!
//! What they share stays at the crate root:
//!
//! - [`safetensors`] memory-maps checkpoint shards with lazy per-tensor access
//!   and optional sha256 verification.
//! - [`numeric`] decodes the scalar storage formats exactly.
//! - [`tensor`] is the row-major matrix and the matrix kernels.
//! - [`compare`] holds the metrics golden checks report.
//!
//! See `AGENTS.md` in this crate for what is verified against what.

pub mod compare;
pub mod deepseek;
mod error;
pub mod mimo;
pub mod numeric;
pub mod qwen;
pub mod safetensors;
pub mod tensor;

pub use error::{Error, Result};
pub use safetensors::WeightSet;
pub use tensor::Matrix;
