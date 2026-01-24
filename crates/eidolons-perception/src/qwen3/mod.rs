//! Qwen3 model implementation using burn framework.
//!
//! This module provides a Qwen3 architecture with QK-Norm (RMSNorm applied to Q and K
//! before RoPE). The key differences from Llama are:
//!
//! - **QK-Norm**: RMSNorm is applied to Q and K vectors before RoPE, improving training stability
//! - **Untied embeddings**: Separate `embed_tokens` and `lm_head` weights (not shared)
//! - **Higher RoPE theta**: 1,000,000 for extended context (vs Llama's 10,000)
//! - **Larger vocabulary**: 151,936 tokens
//!
//! The MLP (SwiGLU) and RMSNorm layers are reused from the Llama implementation since
//! they are identical.

pub mod attention;
pub mod config;
pub mod model;

pub use config::Qwen3Config;
pub use model::Qwen3;
