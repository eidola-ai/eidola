//! [`CpuExecutor`]: the serving core's executor for MiMo-V2.6 over the model crate's
//! reference numerics, with f32 KV paged per group and MTP drafting. Doctrine: this
//! crate's `AGENTS.md`.
//!
//! * [`executor`] — [`CpuExecutor`], the [`eidola_engine::executor::Executor`] over
//!   [`eidola_engine_model::mimo::ReferenceModel`].
//! * [`oracle`] — the dense oracle it is diffed against: the reference forward over a
//!   whole sequence plus the MTP draft chain in the executor's row layout.

pub mod executor;
pub mod oracle;

pub use executor::{CpuExecutor, CpuExecutorConfig, MtpHidden, RowRecord};
pub use oracle::{DenseOracle, DenseRun};
