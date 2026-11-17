//! CPU reference executor: runs the serving core's steps with the model crate's f32
//! reference numerics, over paged KV, with drafting and on-"device" sampling, one module
//! per architecture.
//!
//! * [`mimo`] — [`mimo::CpuExecutor`], the executor for MiMo-V2.6 over
//!   [`eidola_engine_model::mimo::ReferenceModel`], with MTP drafting, and its dense
//!   oracle.
//! * [`deepseek`] — [`deepseek::DsCpuExecutor`], the executor for DeepSeek-V4.1 over
//!   [`eidola_engine_model::deepseek::DsModel`]: KV paged in the model's packed formats,
//!   the model split across two logical devices, and its own dense oracle.
//!
//! * [`qwen`] — [`qwen::QwenCpuExecutor`], the executor for Qwen3.8 over
//!   [`eidola_engine_model::qwen::QwenModel`]: full-attention KV paged per group beside a
//!   recurrent state kept as buffers at block boundaries and a per-slot tail of inputs,
//!   with MTP drafting, and its own dense oracle.
//!
//! Shared: `page`, the page-owned memory every pool is built from, and `pool`, the
//! per-layer K|V block pool MiMo-V2.6 and Qwen3.8 page their KV in.
//!
//! Doctrine lives in this crate's `AGENTS.md`.

pub mod deepseek;
pub mod mimo;
mod page;
mod pool;
pub mod qwen;
