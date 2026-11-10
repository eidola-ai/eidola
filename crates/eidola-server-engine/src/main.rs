//! The inference node's process. Configuration is environment-only (`config.rs`); see
//! this crate's `AGENTS.md` for the contract.
//!
//! Logs go to stdout through `tracing` at a fixed `info` level. The filter is part of the
//! measured image rather than read from the environment: everything that changes the
//! node's behaviour is measured, and a level (or a per-crate directive) delivered any other
//! way could turn on a dependency's debug logging, which this crate's content-free
//! discipline does not cover. Nothing request-derived is ever logged: no prompt, output,
//! cache key, salt or per-request size.

use std::process::ExitCode;

use eidola_server_engine::config::Config;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
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

    // Only the router keeps the engine running from here on, so a shutdown that drops it
    // lets the engine thread finish its work and stop (`run`). A `..` pattern would leave
    // the handle alive in `node` until this function returns.
    let eidola_server_engine::Node {
        router,
        engine,
        engine_stopped,
        admission: _,
        weights_hash: _,
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
