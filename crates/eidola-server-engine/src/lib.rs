//! `eidola-server-engine`: the process on each GPU inference node. It serves exactly one
//! model through Eidola's own engine crates, in process, behind a narrow HTTP API that
//! only Eidola's gateway calls.
//!
//! * [`config`] — the node's configuration, from the environment; nothing defaulted.
//! * `cuda` (the `cuda` feature) — the CUDA executor's refusals, KV sizing and device.
//! * [`auth`] — the gateway token (measured Argon2id hash, constant-time per request).
//! * [`model`] — the weights hash, checked before loading, and the loaded model.
//! * [`storage`] — the `verified-readonly` storage check (mount and superblock read-only).
//! * [`swap`] — the boot refusal of a guest with active swap.
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
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod error;
pub mod gpu_cc;
pub mod http;
pub mod model;
pub mod pipeline;
pub mod storage;
pub mod swap;
pub mod telemetry;
pub mod worker;

use std::sync::Arc;
use std::time::Duration;

use eidola_engine::engine::SchedulerConfig;
use eidola_engine::kv::CachePolicy;
use eidola_engine::secret::SaltDeriver;
use eidola_engine::spec::Bucket;
use eidola_engine_cpu::deepseek::{DsCpuExecutor, DsCpuExecutorConfig};
use eidola_engine_cpu::mimo::{CpuExecutor, CpuExecutorConfig, MtpHidden};
use eidola_engine_cpu::qwen::{QwenCpuExecutor, QwenCpuExecutorConfig};
use tokio::sync::oneshot;

use crate::config::{Config, ExecutorConfig};
use crate::http::AppState;
use crate::model::{LoadedModel, Served};
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
    /// What the node's telemetry instruments read ([`telemetry::Source::register`]).
    pub telemetry: telemetry::Source,
}

impl std::fmt::Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Node")
            .field("weights_hash", &self.weights_hash)
            .finish_non_exhaustive()
    }
}

/// How long a graceful shutdown waits for open requests to finish on their own before
/// the rest are cancelled. With [`TEARDOWN_LIMIT`] and [`ENGINE_SETTLE_LIMIT`] it bounds
/// the time from the signal to the exit, so a long generation or a stalled reader cannot
/// hold the process until it is killed with KV still unzeroed.
pub const DRAIN_LIMIT: Duration = Duration::from_secs(5);

/// How long the runtime may take to wind down remaining tasks (open connections, a
/// preparation still running on the blocking pool) once serving has ended.
pub const TEARDOWN_LIMIT: Duration = Duration::from_secs(2);

/// How long a graceful shutdown waits, after the runtime is down, for the engine thread
/// to release what the cancelled requests held (their KV zeroed or left encrypted, their
/// keys dropped) and stop.
pub const ENGINE_SETTLE_LIMIT: Duration = Duration::from_secs(2);

/// Serves `router` on `listener` until `shutdown` resolves (`Ok`, after open requests
/// have drained or [`DRAIN_LIMIT`] has passed, whichever is first) or the engine thread
/// stops for any reason (`Err`, at once): `engine_stopped` resolving, by its signal or by
/// its sender going away, is fatal, and nothing waits for open connections, however
/// stalled. Connection tasks still open either way are left to the runtime's bounded
/// teardown ([`run`]), which drops them: each request's guard then cancels it in the
/// engine.
pub async fn serve(
    listener: tokio::net::TcpListener,
    router: axum::Router,
    mut engine_stopped: oneshot::Receiver<()>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), BootError> {
    serve_until(listener, router, &mut engine_stopped, shutdown).await
}

/// [`serve`], leaving `engine_stopped` with the caller when serving ends without it.
async fn serve_until(
    listener: tokio::net::TcpListener,
    router: axum::Router,
    engine_stopped: &mut oneshot::Receiver<()>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), BootError> {
    let (draining_tx, draining_rx) = oneshot::channel::<()>();
    let server = axum::serve(listener, router).with_graceful_shutdown(async move {
        shutdown.await;
        let _ = draining_tx.send(());
    });
    let drain_limit = async move {
        match draining_rx.await {
            Ok(()) => tokio::time::sleep(DRAIN_LIMIT).await,
            // The server ended without a shutdown; its own arm reports that.
            Err(_) => std::future::pending().await,
        }
    };
    tokio::select! {
        served = std::future::IntoFuture::into_future(server) => {
            served.map_err(|e| BootError(format!("serving failed: {}", e.kind())))
        }
        _ = drain_limit => {
            tracing::info!("open requests outlasted the drain limit; cancelling them");
            Ok(())
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
///
/// After a graceful shutdown the teardown drops every request still open, whose guards
/// cancel them in the engine, and with the router gone the engine thread stops once it
/// has run the maintenance they left. This waits for that, up to
/// [`ENGINE_SETTLE_LIMIT`], so the process does not exit under the engine while it zeroes
/// freed KV. The caller must hold no [`EngineHandle`] of its own by then: the engine
/// thread runs for as long as one exists.
pub fn run(
    bind: std::net::SocketAddr,
    router: axum::Router,
    mut engine_stopped: oneshot::Receiver<()>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    on_listening: impl FnOnce(std::net::SocketAddr),
) -> Result<(), BootError> {
    // An explicit worker count, so the runtime never reads `TOKIO_WORKER_THREADS`
    // (or any other ambient variable): one worker per vCPU the measured VM shape
    // gives, at least one.
    let workers = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .map_err(|e| BootError(format!("cannot start the runtime: {}", e.kind())))?;
    let result = runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind(bind)
            .await
            .map_err(|e| BootError(format!("cannot bind: {}", e.kind())))?;
        on_listening(
            listener
                .local_addr()
                .map_err(|e| BootError(format!("cannot bind: {}", e.kind())))?,
        );
        serve_until(listener, router, &mut engine_stopped, shutdown).await
    });
    runtime.shutdown_timeout(TEARDOWN_LIMIT);
    if result.is_ok() {
        let settle = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .map_err(|e| BootError(format!("cannot start the runtime: {}", e.kind())))?;
        let settled = settle
            .block_on(async { tokio::time::timeout(ENGINE_SETTLE_LIMIT, engine_stopped).await });
        if settled.is_err() {
            tracing::warn!("the engine did not stop within its limit");
        }
    }
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

/// Verifies and loads the configured weights, then starts the node. In order: no active
/// swap ([`swap::check_guest`]), the weights storage, the weights hash, the model's files
/// (and, for the CPU executor, the reference model), then the executor ([`start`]); a
/// refusal at any step ends the boot with nothing served.
pub fn boot(config: Config) -> Result<Node, BootError> {
    boot_with_swap_list(config, std::path::Path::new(swap::PROC_SWAPS))
}

/// [`boot`] with the swap check reading the list at `swaps` instead of
/// [`swap::PROC_SWAPS`]: for tests whose machine has swap (CI runners do) and that check
/// what boot does after it. The node itself only ever calls [`boot`].
pub fn boot_with_swap_list(config: Config, swaps: &std::path::Path) -> Result<Node, BootError> {
    swap::check_guest_at(config.weights_storage, config.kv_host_bytes, swaps)?;
    let model = LoadedModel::load(
        &config.weights_dir,
        &config.expected_weights_sha256,
        config.weights_storage,
        config.executor.kind(),
    )
    .map_err(|e| BootError(e.to_string()))?;
    start(config, Arc::new(model))
}

/// The CPU executor's host-tier block count from `kv_host_bytes`: as many blocks of every
/// group together as the budget holds (one count for every group, as its pools have; the
/// null block among them). 0 is none; a budget under two blocks is refused.
fn cpu_host_blocks(
    model: &Arc<eidola_engine_model::mimo::ReferenceModel>,
    cfg: &CpuExecutorConfig,
    kv_host_bytes: u64,
) -> Result<u32, BootError> {
    if kv_host_bytes == 0 {
        return Ok(0);
    }
    let per_block = CpuExecutor::kv_block_bytes(model, cfg).max(1);
    let blocks = (kv_host_bytes / per_block).min(i32::MAX as u64);
    if blocks < 2 {
        return Err(BootError(format!(
            "{} ({kv_host_bytes}) holds fewer than two host blocks of {per_block} bytes",
            config::env::KV_HOST_BYTES
        )));
    }
    Ok(blocks as u32)
}

/// Refuses unless `KV_DEVICE_BYTES` names one budget for each device the
/// served architecture runs on (`eidola_common::architecture::Architecture::devices`):
/// the node opens exactly the devices its configuration names, device `d` on GPU `d`.
pub fn check_device_budgets(
    architecture: eidola_common::architecture::Architecture,
    budgets: usize,
) -> Result<(), BootError> {
    let devices = architecture.devices().len();
    if budgets == devices {
        Ok(())
    } else {
        Err(BootError(format!(
            "{} names {budgets} device budgets; {architecture} runs on {devices}",
            config::env::KV_DEVICE_BYTES
        )))
    }
}

/// The engine core's scheduler configuration for `sizing` and `cache` over `model`: the
/// node's, whichever executor runs it.
pub fn scheduler_config(
    sizing: &config::Sizing,
    cache: &config::CacheConfig,
    model: &LoadedModel,
) -> SchedulerConfig {
    SchedulerConfig {
        max_batched_tokens: sizing.max_batched_tokens,
        max_seqs: sizing.max_seqs,
        max_prefill_chunk: sizing.max_prefill_chunk,
        eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
        speculative: sizing.draft_tokens > 0,
        cache: CachePolicy {
            enabled: cache.enabled,
            idle_ttl_ms: cache.idle_ttl_secs * 1000,
            max_age_ms: cache.max_age_secs * 1000,
        },
        sweep_interval_ms: SWEEP_INTERVAL.as_millis() as u64,
    }
}

/// The DeepSeek-V4.1 CPU executor's configuration: one captured shape, as for MiMo; every
/// group `kv_blocks` blocks. Refuses what the executor would panic on: a block size the
/// pooling ratios do not divide, or more drafts than the drafter's slots (or any, without
/// a drafter). A host tier is not offered for this executor.
pub fn deepseek_cpu_config(
    ds: &eidola_engine_model::deepseek::DsConfig,
    sizing: &config::Sizing,
    kv_blocks: u32,
    kv_host_bytes: u64,
    sampleable_vocab_size: u32,
) -> Result<DsCpuExecutorConfig, BootError> {
    if let Some(ratio) = ds
        .layers
        .iter()
        .map(|l| l.compress_ratio as u32)
        .find(|&r| r > 1 && !sizing.kv_block_size.is_multiple_of(r))
    {
        return Err(BootError(format!(
            "{} ({}) must be a multiple of the model's pooling ratio {ratio}",
            config::env::KV_BLOCK_SIZE,
            sizing.kv_block_size
        )));
    }
    let slots = ds.drafter.as_ref().map_or(0, |d| d.slots as u32);
    if sizing.draft_tokens > slots {
        return Err(BootError(format!(
            "{} ({}) exceeds the model's {slots} draft slots",
            config::env::DRAFT_TOKENS,
            sizing.draft_tokens
        )));
    }
    if kv_host_bytes > 0 {
        return Err(BootError(format!(
            "{} must be 0: the DeepSeek-V4.1 cpu executor takes no host tier",
            config::env::KV_HOST_BYTES
        )));
    }
    Ok(DsCpuExecutorConfig {
        block_size: sizing.kv_block_size,
        num_blocks: kv_blocks,
        window_blocks: kv_blocks,
        host_blocks: 0,
        num_state_slots: sizing.max_seqs,
        max_model_len: sizing.max_model_len,
        buckets: vec![Bucket {
            max_seqs: sizing.max_seqs,
            max_tokens: sizing.max_batched_tokens,
        }],
        draft_width: sizing.draft_tokens,
        sampleable_vocab_size,
        record: false,
    })
}

/// The Qwen3.8 CPU executor's configuration: one captured shape, as for the others; both
/// KV groups `kv_blocks` blocks; a state pool of
/// `eidola_common::engine_deployment::recurrent_state_buffers` buffers (the CUDA
/// executor's rule: per seat its base, a fresh buffer and, with the prefix cache, its
/// retention points). Refuses what the executor would panic on: drafting without the
/// model's MTP layer. A host tier is not offered for this executor.
pub fn qwen_cpu_config(
    qwen: &eidola_engine_model::qwen::QwenConfig,
    sizing: &config::Sizing,
    cache: &config::CacheConfig,
    kv_blocks: u32,
    kv_host_bytes: u64,
    sampleable_vocab_size: u32,
) -> Result<QwenCpuExecutorConfig, BootError> {
    if sizing.draft_tokens > 0 && qwen.mtp_layers == 0 {
        return Err(BootError(format!(
            "{} ({}) needs the model's MTP layer, and it has none",
            config::env::DRAFT_TOKENS,
            sizing.draft_tokens
        )));
    }
    if kv_host_bytes > 0 {
        return Err(BootError(format!(
            "{} must be 0: the Qwen3.8 cpu executor takes no host tier",
            config::env::KV_HOST_BYTES
        )));
    }
    let state_buffers = eidola_common::engine_deployment::recurrent_state_buffers(sizing, cache);
    let state_buffers = u32::try_from(state_buffers).map_err(|_| {
        BootError(format!(
            "{} overflows the state pool",
            config::env::MAX_SEQS
        ))
    })?;
    Ok(QwenCpuExecutorConfig {
        block_size: sizing.kv_block_size,
        num_blocks: kv_blocks,
        host_blocks: 0,
        state_buffers,
        host_state_buffers: 0,
        num_state_slots: sizing.max_seqs,
        max_model_len: sizing.max_model_len,
        buckets: vec![Bucket {
            max_seqs: sizing.max_seqs,
            max_tokens: sizing.max_batched_tokens,
        }],
        draft_tokens: sizing.draft_tokens,
        sampleable_vocab_size,
        record: false,
    })
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
    let max_positions = model.max_positions();
    if sizing.max_model_len > max_positions {
        return Err(BootError(format!(
            "{} ({}) exceeds the model's max_position_embeddings ({max_positions})",
            config::env::MAX_MODEL_LEN,
            sizing.max_model_len
        )));
    }
    let sampleable_vocab_size = model.sampleable_vocab_size();
    let scheduler = scheduler_config(&sizing, &config.cache, &model);

    let (stopped_tx, stopped_rx) = oneshot::channel();
    let (engine, device) = match (&config.executor, model.served()) {
        (
            ExecutorConfig::Cpu { kv_blocks },
            Served::DeepSeek {
                config: ds,
                model: reference,
                ..
            },
        ) => {
            let reference = reference.clone().ok_or_else(|| {
                BootError("the loaded model has no reference weights for the cpu executor".into())
            })?;
            let exec_config = deepseek_cpu_config(
                ds,
                &sizing,
                *kv_blocks,
                config.kv_host_bytes,
                sampleable_vocab_size,
            )?;
            let engine = worker::spawn(
                move || {
                    // The executor panics on a shape it does not run; the checks above
                    // refuse every configuration the node can be given that would.
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        DsCpuExecutor::new(reference, exec_config)
                    }))
                    .map_err(|_| "the DeepSeek-V4.1 executor refused this model".to_string())
                },
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            );
            (engine, None)
        }
        // DeepSeek-V4.1 on two GPUs, device `d` on GPU `d`. Never a fallback: a refusal
        // here ends the boot.
        #[cfg(feature = "cuda")]
        (
            ExecutorConfig::Cuda {
                kernels_dir,
                kv_device_bytes,
                further_kv_device_bytes,
                graphs,
                gpu_cc,
            },
            Served::DeepSeek { .. },
        ) => {
            check_device_budgets(model.architecture(), 1 + further_kv_device_bytes.len())?;
            let budgets = [*kv_device_bytes, further_kv_device_bytes[0]];
            let prepared = cuda::prepare_deepseek(
                &model,
                &sizing,
                &config.cache,
                kernels_dir,
                budgets,
                config.kv_host_bytes,
                *graphs,
                *gpu_cc,
            )?;
            let report = prepared.report();
            let c = prepared.config();
            tracing::info!(
                devices = ?c.devices,
                bank_kv_blocks = c.num_blocks,
                window_kv_blocks = c.window_blocks,
                host_kv_blocks = c.host_blocks,
                draft_tokens = c.draft_width,
                "loading the model onto the devices"
            );
            let engine = worker::spawn(
                move || prepared.build(),
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            );
            (engine, Some(report))
        }
        (
            ExecutorConfig::Cpu { kv_blocks },
            Served::Qwen {
                model: reference, ..
            },
        ) => {
            let exec_config = qwen_cpu_config(
                reference.config(),
                &sizing,
                &config.cache,
                *kv_blocks,
                config.kv_host_bytes,
                sampleable_vocab_size,
            )?;
            let reference = reference.clone();
            let engine = worker::spawn(
                move || {
                    // The executor panics on a configuration it does not run; the checks
                    // above refuse every one the node can be given that would.
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        QwenCpuExecutor::new(reference, exec_config)
                    }))
                    .map_err(|_| "the Qwen3.8 executor refused this model".to_string())
                },
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            );
            (engine, None)
        }
        // Qwen3.8 on one GPU. Never a fallback: a refusal here ends the boot.
        #[cfg(feature = "cuda")]
        (
            ExecutorConfig::Cuda {
                kernels_dir,
                kv_device_bytes,
                further_kv_device_bytes,
                graphs,
                gpu_cc,
            },
            Served::Qwen { .. },
        ) => {
            check_device_budgets(model.architecture(), 1 + further_kv_device_bytes.len())?;
            let prepared = cuda::prepare_qwen(
                &model,
                &sizing,
                &config.cache,
                kernels_dir,
                *kv_device_bytes,
                config.kv_host_bytes,
                *graphs,
                *gpu_cc,
            )?;
            let report = prepared.report();
            let c = prepared.config();
            tracing::info!(
                device = %report.name,
                compute_capability = %report.compute_capability,
                kv_blocks = c.num_blocks,
                state_buffers = c.state_buffers,
                host_kv_blocks = c.host_blocks,
                host_state_buffers = c.host_state_buffers,
                draft_tokens = c.draft_tokens,
                profile = prepared.profile().name(),
                "loading the model onto the device"
            );
            let engine = worker::spawn(
                move || prepared.build(),
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            );
            (engine, Some(report))
        }
        (ExecutorConfig::Cpu { kv_blocks }, Served::Mimo { .. }) => {
            let reference = model.reference().cloned().ok_or_else(|| {
                BootError("the loaded model has no reference weights for the cpu executor".into())
            })?;
            let mtp_layers = reference.weights.mtp.len() as u32;
            if sizing.draft_tokens > mtp_layers {
                return Err(BootError(format!(
                    "{} ({}) exceeds the model's {mtp_layers} MTP layers",
                    config::env::DRAFT_TOKENS,
                    sizing.draft_tokens
                )));
            }
            // One captured shape: the CPU executor runs any batch within it unpadded.
            let mut exec_config = CpuExecutorConfig {
                block_size: sizing.kv_block_size,
                num_blocks: *kv_blocks,
                host_blocks: 0,
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
            exec_config.host_blocks =
                cpu_host_blocks(&reference, &exec_config, config.kv_host_bytes)?;
            let engine = worker::spawn(
                move || Ok(CpuExecutor::new(reference, exec_config)),
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            );
            (engine, None)
        }
        // Never a fallback: a refusal here ends the boot.
        #[cfg(feature = "cuda")]
        (
            ExecutorConfig::Cuda {
                kernels_dir,
                kv_device_bytes,
                further_kv_device_bytes,
                graphs,
                gpu_cc,
            },
            Served::Mimo { .. },
        ) => {
            check_device_budgets(model.architecture(), 1 + further_kv_device_bytes.len())?;
            let prepared = cuda::prepare(
                &model,
                &sizing,
                &config.cache,
                kernels_dir,
                *kv_device_bytes,
                config.kv_host_bytes,
                *graphs,
                *gpu_cc,
            )?;
            let report = prepared.report();
            let blocks = prepared.config().num_blocks;
            tracing::info!(
                device = %report.name,
                compute_capability = %report.compute_capability,
                image = report.image.unwrap_or("none"),
                global_kv_blocks = blocks.global,
                sliding_kv_blocks = blocks.sliding,
                drafter_kv_blocks = blocks.drafter,
                global_host_kv_blocks = prepared.config().host_blocks.global,
                sliding_host_kv_blocks = prepared.config().host_blocks.sliding,
                drafter_host_kv_blocks = prepared.config().host_blocks.drafter,
                draft_tokens = prepared.config().draft_tokens,
                cuda_graphs = prepared.config().graphs.as_str(),
                "loading the model onto the device"
            );
            let engine = worker::spawn(
                move || prepared.build(),
                scheduler,
                SWEEP_INTERVAL,
                stopped_tx,
            );
            (engine, Some(report))
        }
    };
    let engine =
        engine.map_err(|e| BootError(format!("the engine refused its configuration: {e}")))?;

    let weights_hash = model.weights_hash().to_string();
    let admission = Admission::new(sizing.max_requests);
    // The read bound has as many slots as admission (`http` module docs).
    let reading = Admission::new(sizing.max_requests);
    let state = Arc::new(AppState {
        model_id: config.model_id,
        executor: config.executor.kind().as_str(),
        device,
        max_model_len: sizing.max_model_len,
        model,
        token: config.gateway_token,
        salts: SaltDeriver::new(),
        engine: engine.clone(),
        admission: admission.clone(),
        reading,
        overloaded: Default::default(),
        failed: Default::default(),
        streams_failed: Default::default(),
    });
    let telemetry = telemetry::Source::new(&state);
    Ok(Node {
        telemetry,
        router: http::router(state),
        engine,
        admission,
        weights_hash,
        engine_stopped: stopped_rx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_common::architecture::Architecture;

    #[test]
    fn a_node_names_one_budget_per_device() {
        assert!(check_device_budgets(Architecture::MimoV26Flash, 1).is_ok());
        assert!(check_device_budgets(Architecture::DeepSeekV41Flash, 2).is_ok());
        assert!(check_device_budgets(Architecture::Qwen38Dense27B, 1).is_ok());
        for (arch, n) in [
            (Architecture::MimoV26Flash, 2),
            (Architecture::Qwen38Dense27B, 2),
            (Architecture::DeepSeekV41Flash, 1),
            (Architecture::DeepSeekV41Flash, 3),
        ] {
            let e = check_device_budgets(arch, n).unwrap_err();
            assert!(e.0.contains("device budgets"), "{e}");
        }
    }
}
