//! The DeepSeek-V4.1 executor's modules, beside the MiMo-V2.6 ones.
//!
//! [`abi`] is the kernels' parameter structs; [`ops`], [`gemm_ref`], [`head`],
//! [`attention`] and [`index`] are the launch wrappers of the kernel images, and
//! [`kernels`] the launches the executor makes through them. [`executor`] is the seam,
//! over two [`half`]s whose forward is [`model`], with [`plan`] ordering a step's rows,
//! [`kv`] and [`weights`] laying out each half's memory, and [`boundary`] what crosses
//! between them. [`host`] runs the same executor on host memory with the reference
//! numerics, and [`dry`] with no memory at all: the accounting and the launch checks
//! of a configuration at any scale. `staged` (dev-only, behind the
//! `dev-staged-experts` feature) runs the whole model on one GPU with the routed experts
//! cached from host memory.

pub mod abi;
pub mod attention;
pub mod boundary;
pub mod dry;
pub mod executor;
pub mod gemm_ref;
pub mod half;
pub mod head;
pub mod host;
pub mod index;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod ops;
pub mod plan;
#[cfg(feature = "dev-staged-experts")]
pub mod staged;
pub mod support;
pub mod weights;

pub use executor::{DsCudaExecutor, DsCudaExecutorConfig, StepStats};
