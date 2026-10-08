//! `eidola-server-engine`: the process on each GPU inference node. It serves exactly one
//! model through Eidola's own engine crates, in process, behind a narrow HTTP API that
//! only Eidola's gateway calls.
//!
//! * [`config`] — the node's configuration, from the environment; nothing defaulted.
//! * [`auth`] — the gateway token (measured Argon2id hash, constant-time per request).
//! * [`model`] — the weights hash, checked before loading, and the loaded model.
//! * [`api`] — the strict request subset; [`error`] — refusals, with log-safe `Display`.
//! * [`pipeline`] — template, tokens, sampling and cache scope in; detokenizer, stop
//!   sequences, reasoning and tool calls out.
//! * [`worker`] — the engine's step loop on its own thread, and the bridge to it.
//! * [`http`] — the routes.
//!
//! Doctrine lives in this crate's `AGENTS.md`.

pub mod api;
pub mod auth;
pub mod config;
pub mod error;
pub mod http;
pub mod model;
pub mod pipeline;
pub mod worker;

use std::sync::Arc;
use std::time::Duration;

use eidola_engine::engine::SchedulerConfig;
use eidola_engine::kv::CachePolicy;
use eidola_engine::secret::SaltDeriver;
use eidola_engine::spec::Bucket;
use eidola_engine_cpu::{CpuExecutor, CpuExecutorConfig, MtpHidden};
use tokio::sync::oneshot;

use crate::config::{Config, ExecutorKind};
use crate::http::AppState;
use crate::model::LoadedModel;
use crate::worker::{Admission, EngineHandle};

/// How often an idle engine sweeps the prefix cache for expired entries.
pub const SWEEP_INTERVAL: Duration = Duration::from_secs(1);

/// Why the node did not start. Content-free: the messages name variables, files and
/// limits only.
#[derive(Debug)]
pub struct BootError(pub String);

impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BootError {}

/// A started node: its router, its engine, and a signal that fires if the engine stops.
pub struct Node {
    pub router: axum::Router,
    pub engine: EngineHandle,
    pub weights_hash: String,
    /// Resolves if the engine thread stops (an executor failure).
    pub engine_stopped: oneshot::Receiver<()>,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("weights_hash", &self.weights_hash)
            .finish_non_exhaustive()
    }
}

/// Verifies and loads the configured weights, then starts the node.
pub fn boot(config: Config) -> Result<Node, BootError> {
    let model = LoadedModel::load(&config.weights_dir, &config.expected_weights_sha256)
        .map_err(|e| BootError(e.to_string()))?;
    start(config, Arc::new(model))
}

/// Starts the node over an already verified model. Refuses unless the model's verified
/// weights hash is the one `config` expects.
pub fn start(config: Config, model: Arc<LoadedModel>) -> Result<Node, BootError> {
    if model.weights_hash() != config.expected_weights_sha256 {
        return Err(BootError(
            "the loaded model's weights hash is not the configured one".into(),
        ));
    }
    let sizing = config.sizing;
    let weights = &model.model().weights;
    let max_positions = weights.config.max_position_embeddings as u32;
    if sizing.max_model_len > max_positions {
        return Err(BootError(format!(
            "{} ({}) exceeds the model's max_position_embeddings ({max_positions})",
            config::env::MAX_MODEL_LEN,
            sizing.max_model_len
        )));
    }
    let mtp_layers = weights.mtp.len() as u32;
    if sizing.draft_tokens > mtp_layers {
        return Err(BootError(format!(
            "{} ({}) exceeds the model's {mtp_layers} MTP layers",
            config::env::DRAFT_TOKENS,
            sizing.draft_tokens
        )));
    }
    let sampleable_vocab_size = model.tokenizer().vocab_size() as u32;
    let scheduler = SchedulerConfig {
        max_batched_tokens: sizing.max_batched_tokens,
        max_seqs: sizing.max_seqs,
        max_prefill_chunk: sizing.max_prefill_chunk,
        eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
        speculative: sizing.draft_tokens > 0,
        cache: CachePolicy {
            enabled: config.cache.enabled,
            idle_ttl_ms: config.cache.idle_ttl_secs * 1000,
            max_age_ms: config.cache.max_age_secs * 1000,
        },
        sweep_interval_ms: SWEEP_INTERVAL.as_millis() as u64,
    };

    let (stopped_tx, stopped_rx) = oneshot::channel();
    let engine = match config.executor {
        ExecutorKind::Cpu => {
            // One captured shape: the CPU executor runs any batch within it unpadded.
            let exec_config = CpuExecutorConfig {
                block_size: sizing.kv_block_size,
                num_blocks: sizing.kv_blocks,
                num_state_slots: sizing.max_seqs,
                max_model_len: sizing.max_model_len,
                buckets: vec![Bucket {
                    max_seqs: sizing.max_seqs,
                    max_tokens: sizing.max_batched_tokens,
                }],
                mtp_depths: (0..sizing.draft_tokens as usize).collect(),
                mtp_hidden: MtpHidden::Normed,
                sampleable_vocab_size,
                pad_batches: false,
                record: false,
            };
            let reference = model.model().clone();
            worker::spawn(
                move || Ok(CpuExecutor::new(reference, exec_config)),
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            )
        }
        #[cfg(feature = "cuda")]
        ExecutorKind::Cuda => {
            drop(stopped_tx);
            Err("the cuda executor is not implemented in this build".to_string())
        }
    }
    .map_err(|e| BootError(format!("the engine refused its configuration: {e}")))?;

    let weights_hash = model.weights_hash().to_string();
    let state = Arc::new(AppState {
        model_id: config.model_id,
        executor: config.executor.as_str(),
        max_model_len: sizing.max_model_len,
        model,
        token: config.gateway_token,
        salts: SaltDeriver::new(),
        engine: engine.clone(),
        admission: Admission::new(sizing.max_requests),
    });
    Ok(Node {
        router: http::router(state),
        engine,
        weights_hash,
        engine_stopped: stopped_rx,
    })
}
