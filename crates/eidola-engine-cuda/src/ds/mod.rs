//! The DeepSeek-V4.1 executor's modules, beside the MiMo-V2.6 ones.
//!
//! [`abi`] is the kernels' parameter structs and [`kernels`] the launches made with
//! them; [`executor`] is the seam, over two [`half`]s whose forward is [`model`], with
//! [`plan`] ordering a step's rows, [`kv`] and [`weights`] laying out each half's
//! memory, and [`boundary`] what crosses between them. [`host`] runs the same executor
//! on host memory with the reference numerics.

pub mod abi;
pub mod boundary;
pub mod executor;
pub mod half;
pub mod host;
pub mod kernels;
pub mod kv;
pub mod model;
pub mod plan;
pub mod support;
pub mod weights;

pub use executor::{DsCudaExecutor, DsCudaExecutorConfig, StepStats};
