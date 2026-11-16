#!/usr/bin/env bash
# One GPU session for the DeepSeek-V4.1 executor with its routed experts staged
# (`ds::staged`, dev-only, the `dev-staged-experts` feature): the whole released
# model on one GPU.
#
#   ds_staged_session.sh [step ...]     (no argument: every step, in order)
#
#   steps: setup build staged_host staged_tiny staged_tiny_drafting
#          whole_compare whole_generate summary
#
# `staged_host` is the staged executor against the CPU reference executor on the
# host (the tiny fixture, caches smaller than its experts; no GPU). `staged_tiny`
# and `staged_tiny_drafting` are the same on the device, both halves on device 0:
# the plain scenarios, then every draft width and the memory scenarios. Each is
# held to the CPU reference executor step for step and needs only the device and
# the kernel directory.
#
# `whole_compare` and `whole_generate` need the released checkpoint
# (`examples/ds_whole.rs`). Both halves load on the one device, both Engram tables
# included, with the routed experts read once into host memory (about 289 GB; the
# step checks the memory available first) and a cache of about 3,200 experts in
# what the device has left. `whole_compare` runs the text the one-half-at-a-time
# measurement recorded and compares every layer's streams, the boundary, the bank
# and the logits with its recordings bit for bit: RECORDED names that output
# (`ds_gpu_session.sh`'s `$OUT/ds_halves`, which it uses by default when this
# session shares its OUT; its `halves_encoder` and `halves_decoder` steps make it).
# `whole_generate` generates greedily from the `ds_generate` example's chat prompts
# through the engine at draft widths 0 and 5 and requires the same tokens at
# both; it records each step's time and each draft slot's acceptance.
#
# Run from a checkout (or an unpacked `git archive`) of this repository on a Linux
# x86_64 machine with one NVIDIA Blackwell GPU and its driver (CUDA 13 driver API),
# with the kernel images of the reproducible build (KERNELS, as for
# `ds_gpu_session.sh`). Steps are idempotent and resumable (markers under
# $OUT/done/), logged to $OUT/logs/<step>.log; results are JSON under $OUT and
# $OUT/staged_session.json lists the steps. A device test that skipped fails its
# step.
#
# Environment (all optional):
#   WORK, OUT, KERNELS   as `ds_gpu_session.sh`
#   CHECKPOINT           the released checkpoint directory (the `whole_*` steps)
#   RECORDED             the ds_halves output to compare with (default $OUT/ds_halves)
#   HOST_MODE            `read` (experts read once into host memory) or `mapped`
#                        (read from the mapped checkpoint on every miss)  (default read)
#   MAX_NEW, WIDTHS      new tokens per prompt and draft widths of `whole_generate`
#                                                      (default 64 and 0,5)
#
# Expected time on one B300, after `setup` and `build`: staged_host 3 to 5
# minutes; staged_tiny and staged_tiny_drafting a few minutes each. A `whole_*`
# step reads the experts (289 GB, logged with its rate; a few minutes at 1 to
# 2 GB/s) and loads the rest (222 GB, about 3 minutes), then: whole_compare runs
# the recorded text (555 tokens in steps of 96 on the measurement's settings: a
# step about 4 seconds of compute plus the first copies of each layer's
# experts); whole_generate decodes 4 prompts twice, at roughly a second or less a
# step (unmeasured: the copies of missed experts dominate).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${WORK:-$HOME/eidola-gpu}"
OUT="${OUT:-$WORK/out}"
export PATH="$HOME/.cargo/bin:$PATH"
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$REPO/rust-toolchain.toml")"
export RUSTUP_TOOLCHAIN
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/target}"

if [ -z "${KERNELS:-}" ]; then
    for dir in "$REPO/target/engine-kernels/amd64" "$REPO/target/engine-kernels/arm64" "$WORK/kernels"; do
        if [ -d "$dir/cubin" ]; then KERNELS="$dir"; break; fi
    done
fi
export CHECKPOINT="${CHECKPOINT:-}"
export RECORDED="${RECORDED:-$OUT/ds_halves}"
export HOST_MODE="${HOST_MODE:-read}" MAX_NEW="${MAX_NEW:-64}" WIDTHS="${WIDTHS:-0,5}"
export HERE REPO WORK OUT KERNELS="${KERNELS:-}"
export EIDOLA_ENGINE_KERNELS_DIR="$KERNELS"
export EIDOLA_DS_SUMMARY_DIR="$OUT"
export FEATURE=dev-staged-experts

mkdir -p "$WORK" "$OUT/done" "$OUT/logs"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "FAILED: $*" >&2; exit 1; }

run_step() {
    local name="$1" fn="$2"
    if [ -e "$OUT/done/$name" ]; then say "$name: already done"; return 0; fi
    say "$name: start"
    local began=$SECONDS
    set +e
    bash -euo pipefail -c "$(declare -f); $fn" 2>&1 | tee "$OUT/logs/$name.log"
    local status=${PIPESTATUS[0]}
    set -e
    local seconds=$((SECONDS - began))
    printf '%s %s %s\n' "$name" "$status" "$seconds" >> "$OUT/staged_steps.txt"
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    if grep -q 'skipping:' "$OUT/logs/$name.log"; then
        die "$name skipped a device test (see $OUT/logs/$name.log)"
    fi
    touch "$OUT/done/$name"
    say "$name: done in $seconds s"
}

setup() {
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    if ! command -v cc >/dev/null || ! command -v c++ >/dev/null || ! command -v curl >/dev/null; then
        say "installing a C toolchain and curl"
        sudo apt-get update -y
        sudo apt-get install -y build-essential curl ca-certificates
    fi
    if ! command -v rustup >/dev/null; then
        say "installing rustup"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
            sh -s -- -y --profile minimal --default-toolchain none
    fi
    rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal
    cargo --version
    free -g
}

build() {
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda --features "$FEATURE" \
        --test ds_staged --test ds_staged_release --example ds_whole --no-run &&
        cargo build --release --locked -p eidola-engine-cuda --features "$FEATURE" --example ds_whole)
}

# staged_test <filter>: tests of ds_staged on this machine.
staged_test() {
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda --features "$FEATURE" \
        --test ds_staged -- --nocapture --test-threads 1 "$@")
}

# The host scenarios (and the release exclusion), then the device ones.
staged_host() {
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda --test ds_staged_release)
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda --features "$FEATURE" \
        --test ds_staged -- --test-threads 4 --skip on_a_device)
}
staged_tiny() {
    [ -n "$KERNELS" ] && [ -d "$KERNELS/cubin" ] || die "no kernel build output (set KERNELS)"
    staged_test --exact the_tiny_model_staged_on_a_device_equals_the_cpu_reference_executor
}
staged_tiny_drafting() {
    [ -n "$KERNELS" ] && [ -d "$KERNELS/cubin" ] || die "no kernel build output (set KERNELS)"
    staged_test --exact the_tiny_model_staged_on_a_device_drafts_and_keeps_its_memory
}

# whole <phase>: the released model whole, its experts staged.
whole() {
    [ -n "${CHECKPOINT:-}" ] && [ -d "$CHECKPOINT" ] || die "CHECKPOINT names no directory"
    [ -n "$KERNELS" ] && [ -d "$KERNELS/cubin" ] || die "no kernel build output (set KERNELS)"
    if [ "$HOST_MODE" = read ]; then
        local avail
        avail=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
        say "host memory available: $avail GiB"
        [ "$avail" -ge 280 ] ||
            die "the experts need about 270 GiB of host memory; $avail GiB available (HOST_MODE=mapped reads them from the checkpoint instead)"
    fi
    (cd "$REPO" && EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_WHOLE_OUT="$OUT/ds_whole" \
        EIDOLA_DS_WHOLE_PHASES="$1" EIDOLA_DS_WHOLE_RECORDED="$RECORDED" \
        EIDOLA_DS_WHOLE_HOST="$HOST_MODE" EIDOLA_DS_WHOLE_MAX_NEW="$MAX_NEW" \
        EIDOLA_DS_WHOLE_WIDTHS="$WIDTHS" \
        cargo run --release --locked -p eidola-engine-cuda --features "$FEATURE" --example ds_whole)
    cp "$OUT/ds_whole/ds_whole.json" "$OUT/ds_whole_$1.json"
}
whole_compare() {
    [ -f "$RECORDED/ds_halves.json" ] && [ -f "$RECORDED/logits.bin" ] ||
        die "RECORDED ($RECORDED) holds no ds_halves recording: run ds_gpu_session.sh halves_encoder halves_decoder"
    whole compare
}
whole_generate() { whole generate; }

summary() {
    {
        printf '{\n  "gpu": "%s",\n  "kernels": "%s",\n  "steps": [\n' \
            "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)" "$KERNELS"
        awk '{ printf "%s    {\"step\": \"%s\", \"status\": %s, \"seconds\": %s}", sep, $1, $2, $3; sep = ",\n" }
             END { print "" }' "$OUT/staged_steps.txt"
        printf '  ]\n}\n'
    } > "$OUT/staged_session.json"
    cat "$OUT/staged_session.json"
}

ALL="setup build staged_host staged_tiny staged_tiny_drafting whole_compare whole_generate summary"
for step in ${*:-$ALL}; do
    case " $ALL " in
        *" $step "*) ;;
        *) die "unknown step $step (steps: $ALL)" ;;
    esac
    if [ "$step" = summary ]; then
        summary
    else
        run_step "$step" "$step"
    fi
done
say "results in $OUT"
