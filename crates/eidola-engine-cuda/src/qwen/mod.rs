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

use eidola_engine_model::qwen::{QwenConfig, WeightFormat};

/// The prefill chunk Qwen3.8 is served at (the scheduler's `max_prefill_chunk`, the
/// node's `MAX_PREFILL_CHUNK`): 256 positions. The DeltaNet's token
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

/// Refuses a checkpoint this executor has no kernels for: one whose projections are not
/// FP8 block-scaled (Qwen3.5's BF16), or whose head is tied to the embedding (Qwen3.5-2B
/// and -0.8B). The CPU reference executor serves those. Needs no device, so a node calls it
/// before opening one; [`QwenCudaExecutor::new`] calls it first too.
pub fn check_supported(config: &QwenConfig) -> crate::Result<()> {
    if config.weights != WeightFormat::Fp8Block128 {
        return Err(crate::CudaError::new(
            "the Qwen CUDA executor runs FP8 block-scaled projections; this checkpoint stores \
             BF16 ones, which the cpu executor serves",
        ));
    }
    if config.tie_word_embeddings {
        return Err(crate::CudaError::new(
            "the Qwen CUDA executor runs a head of its own; this checkpoint ties its head to \
             the embedding, which the cpu executor serves",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(fixture: &str) -> QwenConfig {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../eidola-engine-model/tests/fixtures")
            .join(fixture)
            .join("config.json");
        QwenConfig::from_json_bytes(&std::fs::read(path).unwrap(), "config.json").unwrap()
    }

    /// Qwen3.8's FP8 storage is run; Qwen3.5's BF16 storage is refused with or without a
    /// tied head, and a tied head is refused on its own.
    #[test]
    fn only_fp8_checkpoints_with_their_own_head_are_supported() {
        assert!(check_supported(&config("qwen_tiny")).is_ok());
        for fixture in ["qwen_tiny_bf16", "qwen_tiny_bf16_tied"] {
            let e = check_supported(&config(fixture)).unwrap_err().to_string();
            assert!(e.contains("BF16"), "{fixture}: {e}");
        }
        let mut tied = config("qwen_tiny");
        tied.tie_word_embeddings = true;
        let e = check_supported(&tied).unwrap_err().to_string();
        assert!(e.contains("ties its head"), "{e}");
    }
}
