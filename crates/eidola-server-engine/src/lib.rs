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
    /// The admission bound (its count is content-free).
    pub admission: Arc<Admission>,
    pub weights_hash: String,
    /// Resolves when the engine thread stops, however it stops.
    pub engine_stopped: oneshot::Receiver<()>,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("weights_hash", &self.weights_hash)
            .finish_non_exhaustive()
    }
}

/// How long the runtime may take to wind down remaining tasks (open connections, a
/// preparation still running on the blocking pool) once serving has ended.
pub const TEARDOWN_LIMIT: Duration = Duration::from_secs(2);

/// Serves `router` on `listener` until `shutdown` resolves (`Ok`, after draining open
/// requests gracefully) or the engine thread stops for any reason (`Err`, at once):
/// `engine_stopped` resolving, by its signal or by its sender going away, is fatal, and
/// nothing waits for open connections, however stalled. Those connection tasks are left
/// to the runtime's bounded teardown ([`run`]).
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: axum::Router,
    engine_stopped: oneshot::Receiver<()>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), BootError> {
    let server = axum::serve(listener, router).with_graceful_shutdown(shutdown);
    tokio::select! {
        served = std::future::IntoFuture::into_future(server) => {
            served.map_err(|e| BootError(format!("serving failed: {}", e.kind())))
        }
        _ = engine_stopped => {
            tracing::error!("the engine thread stopped");
            Err(BootError("the engine stopped".into()))
        }
    }
}

/// Runs the node's HTTP side to completion on its own runtime: binds `bind`, reports the
/// bound address to `on_listening`, serves (see [`serve`]), and then tears the runtime
/// down within [`TEARDOWN_LIMIT`], so a fatal engine stop ends the process promptly even
/// with requests stalled mid-upload. `main` maps `Err` to a non-zero exit.
pub fn run(
    bind: std::net::SocketAddr,
    router: axum::Router,
    engine_stopped: oneshot::Receiver<()>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    on_listening: impl FnOnce(std::net::SocketAddr),
) -> Result<(), BootError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| BootError(format!("cannot start the runtime: {}", e.kind())))?;
    let result = runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(bind)
            .await
            .map_err(|e| BootError(format!("cannot bind: {}", e.kind())))?;
        on_listening(
            listener
                .local_addr()
                .map_err(|e| BootError(format!("cannot bind: {}", e.kind())))?,
        );
        serve(listener, router, engine_stopped, shutdown).await
    });
    runtime.shutdown_timeout(TEARDOWN_LIMIT);
    result
}

/// Resolves on SIGINT or SIGTERM: the graceful shutdown.
pub async fn os_shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("install the SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = term.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
    tracing::info!("shutting down");
}

/// Verifies and loads the configured weights, then starts the node.
pub fn boot(config: Config) -> Result<Node, BootError> {
    let model = LoadedModel::load(
        &config.weights_dir,
        &config.expected_weights_sha256,
        config.weights_storage,
    )
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
    if model.storage() != config.weights_storage {
        return Err(BootError(
            "the loaded model's weights storage is not the configured one".into(),
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
    let admission = Admission::new(sizing.max_requests);
    let state = Arc::new(AppState {
        model_id: config.model_id,
        executor: config.executor.as_str(),
        max_model_len: sizing.max_model_len,
        model,
        token: config.gateway_token,
        salts: SaltDeriver::new(),
        engine: engine.clone(),
        admission: admission.clone(),
    });
    Ok(Node {
        router: http::router(state),
        engine,
        admission,
        weights_hash,
        engine_stopped: stopped_rx,
    })
}
