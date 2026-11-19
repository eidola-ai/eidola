//! Qwen3.8-27B (`eidola-engine-cuda/AGENTS.md` → Qwen3.8): the kernels' parameter
//! structs ([`abi`]), the kernel families and their grids ([`kernels`]), the devices the
//! executor runs on ([`device`], [`host`]), its weights ([`weights`]) and the executor
//! ([`executor`]).

pub mod abi;
pub mod attention;
pub mod cuda;
pub mod device;
pub mod executor;
pub mod host;
pub mod kernels;
pub mod plan;
pub mod sampler;
pub mod weights;

pub use device::{QwenDevice, QwenProfile};

/// The prefill chunk Qwen3.8 is served at (the scheduler's `max_prefill_chunk`, the
/// node's `EIDOLA_ENGINE_MAX_PREFILL_CHUNK`): 256 positions. The DeltaNet's token
/// recurrence is serial within a run, one position after another, so a step's prefill
/// cost follows its longest run; a smaller chunk spreads a long prompt over more steps
/// while several sequences prefill side by side, which keeps the recurrence's longest
/// run short and the decode rows' steps short with it. A row's bits do not depend on the
/// chunking (asserted: `tests/qwen_fast.rs`, the invariance scenarios), so the chunk is a
/// choice of speed alone. On a B200 it met every serving point of the yardstick workloads
/// where 2,048 left the drafted ones at concurrency 32 and 64 below their mark, and 512 sat
/// between the two.
pub const SERVING_PREFILL_CHUNK: u32 = 256;
pub use executor::{
    CudaGraphs, DecodePath, DecodeStats, QwenCudaExecutor, QwenCudaExecutorConfig, StateStrategy,
};
