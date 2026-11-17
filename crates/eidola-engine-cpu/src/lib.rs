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
//! Shared by both: `page`, the page-owned memory every KV pool is built from, and `pool`,
//! the per-layer K|V block pool MiMo-V2.6 pages its KV in.
//!
//! Doctrine lives in this crate's `AGENTS.md`.

pub mod deepseek;
pub mod mimo;
mod page;
mod pool;
