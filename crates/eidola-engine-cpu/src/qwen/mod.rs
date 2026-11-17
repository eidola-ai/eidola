//! [`QwenCpuExecutor`]: the serving core's executor for Qwen3.8 over the model crate's
//! reference numerics: the attention layers' KV and the drafter's paged per group, the
//! DeltaNet layers' recurrent state kept as buffers at block boundaries and a per-slot
//! tail of inputs, and MTP drafting. Doctrine: this crate's `AGENTS.md`.
//!
//! * [`executor`] — [`QwenCpuExecutor`], the [`eidola_engine::executor::Executor`] over
//!   [`eidola_engine_model::qwen::QwenModel`].
//! * [`oracle`] — the dense oracle it is diffed against: the reference forward over a
//!   whole sequence, the drafter's rows, and the draft chain.

pub mod executor;
pub mod oracle;
mod state;

pub use executor::{QwenCpuExecutor, QwenCpuExecutorConfig, QwenRowRecord, StoredState};
pub use oracle::{QwenDenseOracle, QwenDenseRun, qwen_check_generation};
