//! Qwen3 model via the qwen3-burn crate.
//!
//! Re-exports from `qwen3_burn` for Qwen3 architecture support including
//! KV cache, chunked prefill, GGUF support, and quantization.

pub use qwen3_burn::model::{
    GenerationEvent, GenerationOutput, GenerationParams, QuantizationMode, Qwen3 as Qwen3Model,
    Qwen3Config, StopReason,
};
pub use qwen3_burn::sampling::Sampler;
