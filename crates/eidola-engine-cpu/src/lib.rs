//! CPU reference executor: runs the serving core's steps with the model crate's f32
//! reference numerics, over paged KV, with MTP drafting and on-"device" sampling.
//!
//! * [`executor`] — [`CpuExecutor`], the [`eidola_engine::executor::Executor`] for
//!   MiMo-V2.6 over [`eidola_engine_model::ReferenceModel`].
//! * [`oracle`] — the dense oracle it is diffed against: the reference forward over a
//!   whole sequence plus the MTP draft chain in the executor's row layout.
//! * [`deepseek`] — [`deepseek::DsCpuExecutor`], the executor for DeepSeek-V4.1 over
//!   [`eidola_engine_model::deepseek::DsModel`]: KV paged in the model's packed formats,
//!   the model split across two logical devices, and its own dense oracle.
//!
//! Doctrine lives in this crate's `AGENTS.md`.

pub mod deepseek;
pub mod executor;
pub mod oracle;
mod pool;

pub use executor::{CpuExecutor, CpuExecutorConfig, MtpHidden, RowRecord};
pub use oracle::{DenseOracle, DenseRun};
