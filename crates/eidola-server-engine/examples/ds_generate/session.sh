#!/usr/bin/env bash
# One CPU session generating from a DeepSeek-V4.1 checkpoint with the node's own path
# (`ds_generate`, beside this script; doctrine in crates/eidola-server-engine/AGENTS.md).
#
#   CHECKPOINT=<dir> session.sh [step ...]      steps: setup hash build generate summary
#                                               (none given: all of them, in that order)
#
# Run from an unpacked tree of this repository on a Linux machine holding the
# checkpoint, with Rust installed (the toolchain comes from rust-toolchain.toml).
# Every step is idempotent and resumable: a finished step leaves a marker under
# $OUT/done/ and is skipped on a rerun (delete the marker to repeat it). Everything a
# step prints is also in $OUT/logs/<step>.log. `generate` checkpoints every request
# it finishes, so a rerun after an interruption runs only the rest. Copy $OUT back.
#
# Environment:
#   CHECKPOINT   the checkpoint directory as published (required)
#   OUT          results                                     (default ~/eidola-dsgen)
#   EIDOLA_DSGEN_WEIGHTS_SHA256  the checkpoint's weights hash; when unset, `hash`
#                computes it once with the offline procedure and caches it in
#                $OUT/weights.sha256
#   EIDOLA_DSGEN_*  the runner's other settings (examples/ds_generate/main.rs):
#                MAX_NEW_TOKENS (96), DRAFT_WIDTH (the drafter's 5), TIME_LIMIT_HOURS
#                (12), PREFILL_CHUNK (512), MIN_AVAILABLE_GB (64), PROMPTS
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
OUT="${OUT:-$HOME/eidola-dsgen}"
[ -n "${CHECKPOINT:-}" ] || { echo "CHECKPOINT is not set" >&2; exit 1; }
CHECKPOINT="$(cd "$CHECKPOINT" && pwd)"
export PATH="$HOME/.cargo/bin:$PATH"
# Absolute, since steps change directory before writing under it.
mkdir -p "$OUT/done" "$OUT/logs"
OUT="$(cd "$OUT" && pwd)"
export HERE ROOT OUT CHECKPOINT

say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { say "FAILED: $*" >&2; exit 1; }

# run_step <name> <function>: skip when done, log, mark, fail loudly.
run_step() {
    local name="$1" fn="$2"
    if [ -e "$OUT/done/$name" ]; then say "$name: already done"; return 0; fi
    say "$name: start"
    local began=$SECONDS
    set +e
    bash -euo pipefail -c "$(declare -f); $fn" 2>&1 | tee "$OUT/logs/$name.log"
    local status=${PIPESTATUS[0]}
    set -e
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    touch "$OUT/done/$name"
    say "$name: done in $((SECONDS - began)) s"
}

setup() {
    command -v cargo >/dev/null || die "cargo not found"
    command -v cc >/dev/null || die "no C compiler: apt-get install -y build-essential"
    say "machine: $(nproc) CPUs"
    free -g || true
    swapon --show || true
    df -h "$CHECKPOINT" "$OUT"
    [ -f "$CHECKPOINT/config.json" ] || die "$CHECKPOINT has no config.json"
    [ -f "$CHECKPOINT/tokenizer.json" ] || die "$CHECKPOINT has no tokenizer.json"
    local shards; shards=$(find "$CHECKPOINT" -maxdepth 1 -name '*.safetensors' | wc -l)
    say "$shards shards, $(du -sh "$CHECKPOINT" | cut -f1)"
    # Shard 1 holds the vision tower, which the text model never reads. A checkpoint
    # fetched without it loads and runs the same, but its weights hash is not the
    # published model's (the hash covers every shard present).
    [ -f "$CHECKPOINT/model-00001-of-00048.safetensors" ] \
        || say "note: model-00001-of-00048.safetensors (vision) is absent; the weights hash covers the shards present"
}

# The weights hash by the offline procedure (crates/eidola-server-engine/AGENTS.md →
# The weights hash), files hashed in parallel; the manifest is kept beside it. The
# runner checks it again through the node's loader on every run.
hash_weights() {
    if [ -n "${EIDOLA_DSGEN_WEIGHTS_SHA256:-}" ]; then
        say "using EIDOLA_DSGEN_WEIGHTS_SHA256 from the environment"
        printf '%s\n' "$EIDOLA_DSGEN_WEIGHTS_SHA256" > "$OUT/weights.sha256"
        return 0
    fi
    cd "$CHECKPOINT"
    for f in *.safetensors config.json model.safetensors.index.json \
        tokenizer.json chat_template.jinja generation_config.json; do
        if [ -f "$f" ]; then printf '%s\n' "$f"; fi
    done | xargs -P "$(nproc)" -n 1 sha256sum | LC_ALL=C sort -k 2 > "$OUT/weights.manifest"
    sha256sum < "$OUT/weights.manifest" | cut -d' ' -f1 > "$OUT/weights.sha256"
    say "$(wc -l < "$OUT/weights.manifest") files; weights hash $(cat "$OUT/weights.sha256")"
}

build() {
    cd "$ROOT"
    cargo build --release --locked -p eidola-server-engine --example ds_generate
}

generate() {
    [ -s "$OUT/weights.sha256" ] || die "no weights hash: run the hash step"
    local status=0
    EIDOLA_DSGEN_WEIGHTS_DIR="$CHECKPOINT" \
        EIDOLA_DSGEN_WEIGHTS_SHA256="$(cat "$OUT/weights.sha256")" \
        EIDOLA_DSGEN_OUT="$OUT/run" \
        "$ROOT/target/release/examples/ds_generate" || status=$?
    case "$status" in
        0) ;;
        2) die "greedy tokens differ between draft widths (results in $OUT/run/summary.json)" ;;
        3) die "refused: the estimate exceeds EIDOLA_DSGEN_TIME_LIMIT_HOURS (see the log; finished requests are kept)" ;;
        *) die "ds_generate exited with $status" ;;
    esac
}

summary() {
    # The runner writes the transcripts of exactly the requests summary.json lists;
    # transcripts/ may also hold files an earlier run of this directory left.
    [ -s "$OUT/run/transcripts.txt" ] || die "no transcripts: run the generate step"
    [ -s "$OUT/run/summary.json" ] || die "no summary: run the generate step"
    {
        echo "DeepSeek-V4.1 CPU generation, $(date -u +%Y-%m-%dT%H:%MZ), $(nproc) CPUs"
        echo "weights hash $(cat "$OUT/weights.sha256") (checked by the node's loader)"
        echo
        cat "$OUT/run/transcripts.txt"
        echo "== summary.json"
        cat "$OUT/run/summary.json"
    } > "$OUT/summary.txt"
    say "wrote $OUT/summary.txt"
}

step() {
    case "$1" in
        setup)    run_step setup setup ;;
        hash)     run_step hash hash_weights ;;
        build)    run_step build build ;;
        generate) run_step generate generate ;;
        summary)  summary ;;
        *)        die "unknown step $1 (setup hash build generate summary)" ;;
    esac
}

if [ $# -eq 0 ]; then set -- setup hash build generate summary; fi
for s in "$@"; do step "$s"; done
