//! Greedy generation from a DeepSeek-V4.1 checkpoint on the CPU reference executor,
//! through the node's own loading, prompt encoding and output parsing, with drafting on
//! and its acceptance measured per draft slot. Development only: it is not part of the
//! node and serves nothing. `session.sh` beside it runs it on a machine holding the
//! checkpoint; doctrine in this crate's `AGENTS.md` → Generating from the released
//! checkpoint on the CPU.
//!
//! ```sh
//! EIDOLA_DSGEN_WEIGHTS_DIR=<checkpoint> EIDOLA_DSGEN_WEIGHTS_SHA256=<hash> \
//! EIDOLA_DSGEN_OUT=<dir> cargo run --release -p eidola-server-engine --example ds_generate
//! ```
//!
//! Environment (the first three required):
//!
//! | Variable | Meaning | Default |
//! |---|---|---|
//! | `EIDOLA_DSGEN_WEIGHTS_DIR` | the checkpoint directory | |
//! | `EIDOLA_DSGEN_WEIGHTS_SHA256` | its weights hash, checked by the node's loader | |
//! | `EIDOLA_DSGEN_OUT` | results, checkpoints and transcripts | |
//! | `EIDOLA_DSGEN_PROMPTS` | a prompts file | `prompts.json` beside this file |
//! | `EIDOLA_DSGEN_MAX_NEW_TOKENS` | new tokens per request, at most | 96 |
//! | `EIDOLA_DSGEN_DRAFT_WIDTH` | the main pass's draft width | the drafter's slots |
//! | `EIDOLA_DSGEN_TIME_LIMIT_HOURS` | refuse a run estimated longer | 12 |
//! | `EIDOLA_DSGEN_PREFILL_CHUNK` | positions per prefill step | 512 |
//! | `EIDOLA_DSGEN_MIN_AVAILABLE_GB` | refuse to load with less available memory (Linux) | 64 |
//!
//! Exit status: 0 when every request has a result and every width-0 check matched; 2
//! when a check differed; 3 when the estimate exceeded the limit; 1 on any failure.

mod runner;

use std::path::PathBuf;

fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

fn required(name: &str) -> Result<String, String> {
    var(name).ok_or_else(|| format!("{name} is not set"))
}

fn number<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    match var(name) {
        None => Ok(default),
        Some(v) => v
            .parse()
            .map_err(|_| format!("{name} ({v}) is not a number")),
    }
}

fn config() -> Result<runner::RunConfig, String> {
    let prompts_path = var("EIDOLA_DSGEN_PROMPTS").map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/examples/ds_generate/prompts.json"
            ))
        },
        PathBuf::from,
    );
    let prompts = std::fs::read(&prompts_path)
        .map_err(|e| format!("read {}: {e}", prompts_path.display()))?;
    let hash = required("EIDOLA_DSGEN_WEIGHTS_SHA256")?.to_ascii_lowercase();
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("EIDOLA_DSGEN_WEIGHTS_SHA256 is not 64 hex digits".into());
    }
    Ok(runner::RunConfig {
        weights_dir: PathBuf::from(required("EIDOLA_DSGEN_WEIGHTS_DIR")?),
        weights_sha256: hash,
        out_dir: PathBuf::from(required("EIDOLA_DSGEN_OUT")?),
        prompts: runner::PromptSpec::parse_file(&prompts)?,
        max_new_tokens: number("EIDOLA_DSGEN_MAX_NEW_TOKENS", 96)?,
        draft_width: var("EIDOLA_DSGEN_DRAFT_WIDTH")
            .map(|v| {
                v.parse()
                    .map_err(|_| format!("EIDOLA_DSGEN_DRAFT_WIDTH ({v}) is not a number"))
            })
            .transpose()?,
        time_limit_secs: number("EIDOLA_DSGEN_TIME_LIMIT_HOURS", 12.0f64)? * 3600.0,
        prefill_chunk: number("EIDOLA_DSGEN_PREFILL_CHUNK", 512)?,
        block_size: 16,
        min_available_bytes: number("EIDOLA_DSGEN_MIN_AVAILABLE_GB", 64u64)? * 1_000_000_000,
    })
}

fn main() {
    let outcome = config().and_then(|cfg| {
        runner::log(format_args!(
            "ds_generate: {} prompts, up to {} new tokens each, output in {}",
            cfg.prompts.len(),
            cfg.max_new_tokens,
            cfg.out_dir.display()
        ));
        runner::run(&cfg)
    });
    match outcome {
        Ok(runner::Outcome::Complete { mismatches: 0 }) => runner::log("ds_generate: complete"),
        Ok(runner::Outcome::Complete { mismatches }) => {
            runner::log(format_args!(
                "FAILED: {mismatches} prompts' greedy tokens differ between draft widths"
            ));
            std::process::exit(2);
        }
        Ok(runner::Outcome::Refused(why)) => {
            runner::log(format_args!("REFUSED: {why}"));
            std::process::exit(3);
        }
        Err(e) => {
            runner::log(format_args!("FAILED: {e}"));
            std::process::exit(1);
        }
    }
}
