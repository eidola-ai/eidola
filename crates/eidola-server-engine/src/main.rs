//! The inference node's process. Configuration is environment-only (`config.rs`); see
//! this crate's `AGENTS.md` for the contract.
//!
//! Logs go to stdout through `tracing` (`RUST_LOG` filters, default `info`). Nothing
//! request-derived is ever logged: no prompt, output, cache key, salt or per-request size.

use std::process::ExitCode;

use eidola_server_engine::config::Config;
use tracing_subscriber::EnvFilter;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("{e}; refusing to serve");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env()?;
    let bind = config.bind_addr;
    tracing::info!(
        model = %config.model_id,
        executor = config.executor.kind().as_str(),
        "verifying and loading weights"
    );
    let node = eidola_server_engine::boot(config)?;
    tracing::info!(weights_sha256 = %node.weights_hash, "weights verified; engine started");

    eidola_server_engine::run(
        bind,
        node.router,
        node.engine_stopped,
        eidola_server_engine::os_shutdown_signal(),
        |addr| tracing::info!(%addr, "listening"),
    )?;
    Ok(())
}
