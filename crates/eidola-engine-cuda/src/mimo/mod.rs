//! The MiMo-V2.6 executor's modules.
//!
//! [`executor`] is the seam ([`CudaExecutor`]), over the target forward in [`model`]
//! with the checkpoint laid out by [`weights`], KV memory in [`kv`], drafting and
//! acceptance in [`draft`] and decode-graph replay in [`graph`]. [`attention`]
//! (FlashInfer FA2 with sinks), [`gemm`] (CUTLASS), [`moe_gemm`] (DeepGEMM, with the
//! [`tma`] descriptors both GEMM paths embed) and [`engine_ops`] (the executor's own
//! kernels) are the launch wrappers. [`support`] is the one model configuration the
//! kernels are built for.

pub mod attention;
pub mod draft;
pub mod engine_ops;
pub mod executor;
pub mod gemm;
pub mod graph;
pub mod kv;
pub mod model;
pub mod moe_gemm;
pub mod support;
pub mod tma;
pub mod weights;

pub use draft::MtpHidden;
pub use executor::{CudaExecutor, CudaExecutorConfig, DecodeStats, DraftRecord, KvBlocks};
pub use graph::{CudaGraphs, DecodePath};
pub use support::{check_device, check_supported};
