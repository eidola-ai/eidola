//! The inference node's process. Configuration is environment-only (`config.rs`); see
//! this crate's `AGENTS.md` for the contract.
//!
//! Logs go to stdout through `tracing`, and over OTLP with the node's metrics when the
//! deployment's telemetry secrets name an endpoint (`telemetry.rs`). The filter is part
//! of the measured image rather than read from the environment: everything that changes
//! what the node logs is measured, and a level (or a per-crate directive) delivered any
//! other way could turn on a dependency's debug logging, which this crate's content-free
//! discipline does not cover. Colour follows whether stdout is a terminal, so the logger
//! reads no variable either (its default would read a bare `NO_COLOR`). Nothing
//! request-derived is ever logged: no prompt, output, cache key, salt or per-request size.
//!
//! Exit status: 0 after a graceful shutdown; 1 when boot refuses or the engine thread
//! stops; `worker::STALL_EXIT_STATUS` when the step watchdog finds the engine wedged.

use std::process::ExitCode;

use eidola_server_engine::config::Config;
use eidola_server_engine::telemetry;
use eidola_server_engine::zeroing::ZeroOnFree;

/// Every heap block is zeroed before it is freed, so no request-derived data the node or
/// its executor computed on the host outlives the memory that held it
/// (`eidola_server_engine::zeroing`).
#[global_allocator]
static ALLOCATOR: ZeroOnFree<std::alloc::System> = ZeroOnFree(std::alloc::System);

fn main() -> ExitCode {
    telemetry::install_crypto_provider();
    let otel = telemetry::init(|name| std::env::var(name).ok());
    let status = match run(otel.as_ref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e}; refusing to serve");
            ExitCode::FAILURE
        }
    };
    if let Some(otel) = otel {
        otel.shutdown();
    }
    status
}

fn run(otel: Option<&telemetry::OtelGuard>) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    tracing::info!("environment variable prefix: {:?}", config.env.prefix());
    let bind = config.bind_addr;
    tracing::info!(
        model = %config.model_id,
        executor = config.executor.kind().as_str(),
        "verifying and loading weights"
    );
    let node = eidola_server_engine::boot(config)?;
    tracing::info!(
        weights_sha256 = %node.weights_hash,
        instance_id = telemetry::instance_id(),
        "weights verified; engine started"
    );
    if let Some(otel) = otel {
        node.telemetry.register(&otel.meter());
    }

    // Only the router keeps the engine running from here on, so a shutdown that drops it
    // lets the engine thread finish its work and stop (`run`). A `..` pattern would leave
    // the handle alive in `node` until this function returns.
    let eidola_server_engine::Node {
        router,
        engine,
        engine_stopped,
        admission: _,
        weights_hash: _,
        telemetry: _,
    } = node;
    drop(engine);
    eidola_server_engine::run(
        bind,
        router,
        engine_stopped,
        eidola_server_engine::os_shutdown_signal(),
        |addr| tracing::info!(%addr, "listening"),
    )?;
    Ok(())
}
