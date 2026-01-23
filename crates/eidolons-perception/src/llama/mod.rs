//! Llama model implementation using burn framework.
//!
//! This module provides a minimal Llama architecture compatible with TinyLlama weights.

pub mod attention;
pub mod config;
pub mod embedding;
pub mod mlp;
pub mod model;
pub mod norm;

pub use config::LlamaConfig;
pub use model::Llama;
