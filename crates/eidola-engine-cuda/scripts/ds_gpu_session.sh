#!/usr/bin/env bash
# One GPU session for the DeepSeek-V4.1 kernels and executor: every device
# test of this crate that needs no checkpoint, in order, unattended.
#
#   ds_gpu_session.sh [step ...]        (no argument: every step, in order)
#
#   steps: setup build kernels ds_ops ds_gemm_ref ds_attention ds_index ds_tiny mimo summary
#
# Run from a checkout (or an unpacked `git archive`) of this repository on a
# Linux x86_64 machine with one NVIDIA Blackwell GPU and its driver (CUDA 13
# driver API). Nothing is compiled for the GPU here: the kernel images are the
# reproducible build's output, copied from the machine that built them
# (`eidola-engine-kernels`' `scripts/build-in-docker.sh`; its
# `target/engine-kernels/<builder arch>` directory, which is the same bytes
# from either builder architecture). Every image is checked against the
# manifest compiled into the tests before it is loaded.
#
# Every step is idempotent and resumable: a finished step leaves a marker
# under $OUT/done/ and is skipped on a rerun (delete the marker to repeat it).
# Everything a step prints is also in $OUT/logs/<step>.log; every result is
# JSON under $OUT, and $OUT/session.json lists the steps. A device step that
# skipped (no device, no kernel directory) fails. Copy $OUT back.
#
# Environment (all optional):
#   WORK     scratch                                   (default ~/eidola-gpu)
#   OUT      results                                   (default $WORK/out)
#   KERNELS  the kernel build output directory         (default: target/engine-kernels/amd64,
#            else .../arm64, under the checkout, else $WORK/kernels)
#
# Expected time on one B300, after `setup` and `build` (about 10 minutes
# together on a fresh machine): kernels under 1 minute; ds_ops 5 to 10
# (the casts over every f32 through the probe, the released shapes);
# ds_gemm_ref 3 to 8 (the whole vocabulary through the head);
# ds_attention and ds_index 1 to 3 each; ds_tiny 10 to 15 (the CPU reference
# executor runs beside the device); mimo 3 to 6. None of these was measured.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${WORK:-$HOME/eidola-gpu}"
OUT="${OUT:-$WORK/out}"
export PATH="$HOME/.cargo/bin:$PATH"
# The toolchain the repository pins, without its cross-compilation targets.
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$REPO/rust-toolchain.toml")"
export RUSTUP_TOOLCHAIN
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/target}"

if [ -z "${KERNELS:-}" ]; then
    for dir in "$REPO/target/engine-kernels/amd64" "$REPO/target/engine-kernels/arm64" "$WORK/kernels"; do
        if [ -d "$dir/cubin" ]; then KERNELS="$dir"; break; fi
    done
fi
export HERE REPO WORK OUT KERNELS="${KERNELS:-}"
export EIDOLA_ENGINE_KERNELS_DIR="$KERNELS"
export EIDOLA_DS_SUMMARY_DIR="$OUT" EIDOLA_DS_BENCH_DIR="$OUT" EIDOLA_DS_TINY_SUMMARY="$OUT/ds_tiny.json"

mkdir -p "$WORK" "$OUT/done" "$OUT/logs"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "FAILED: $*" >&2; exit 1; }

# run_step <name> <function>: skip when done, log, mark, fail loudly.
run_step() {
    local name="$1" fn="$2"
    if [ -e "$OUT/done/$name" ]; then say "$name: already done"; return 0; fi
    say "$name: start"
    local began=$SECONDS
    # A fresh shell, so that errexit is live in the step whatever context
    # this function was called from.
    set +e
    bash -euo pipefail -c "$(declare -f); $fn" 2>&1 | tee "$OUT/logs/$name.log"
    local status=${PIPESTATUS[0]}
    set -e
    local seconds=$((SECONDS - began))
    printf '%s %s %s\n' "$name" "$status" "$seconds" >> "$OUT/steps.txt"
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    # A device test without a device passes after saying it skipped.
    if grep -q '^skipping:' "$OUT/logs/$name.log"; then
        die "$name skipped a device test (see $OUT/logs/$name.log)"
    fi
    touch "$OUT/done/$name"
    say "$name: done in $seconds s"
}

# device_tests <test binary>...: this crate's tests, one at a time on the device.
device_tests() {
    local args=()
    for t in "$@"; do args+=(--test "$t"); done
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda "${args[@]}" -- \
        --nocapture --test-threads 1)
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
}

build() {
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda -p eidola-engine-kernels --no-run)
}

# The kernel directory holds the images the tests' manifest names, byte for
# byte, and the device loads every one (the launch contracts of MiMo's and the
# probe's among them).
kernels() {
    [ -n "$KERNELS" ] && [ -d "$KERNELS/cubin" ] ||
        die "no kernel build output: copy target/engine-kernels/<arch> here and set KERNELS"
    say "kernels: $KERNELS"
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-kernels --test manifest)
    device_tests smoke
}

# The scalar kernels; its suite replays the numerics probe (the casts over
# every f32) before the kernels' cases.
ds_ops() { device_tests ds_ops; }
ds_gemm_ref() { device_tests ds_gemm_ref; }
ds_attention() { device_tests ds_attention; }
ds_index() { device_tests ds_index; }
# The executor: the tiny fixture and its wide-window configuration through the
# engine, equal to the CPU reference executor step for step.
ds_tiny() { device_tests ds_tiny; }

# MiMo's device suites that need no checkpoint: nothing of the first model's
# kernels or wrappers moved.
mimo() {
    device_tests gemm moe_gemm attention sampler kv kv_cipher copy_rows qkv_rope_kv \
        quant_ops moe_ops preconditions
}

summary() {
    {
        printf '{\n  "gpu": "%s",\n  "kernels": "%s",\n  "steps": [\n' \
            "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)" "$KERNELS"
        awk '{ printf "%s    {\"step\": \"%s\", \"status\": %s, \"seconds\": %s}", sep, $1, $2, $3; sep = ",\n" }
             END { print "" }' "$OUT/steps.txt"
        printf '  ],\n  "results": ['
        local sep=""
        for f in "$OUT"/*.json; do
            [ "$(basename "$f")" = session.json ] && continue
            printf '%s"%s"' "$sep" "$(basename "$f")"
            sep=", "
        done
        printf ']\n}\n'
    } > "$OUT/session.json"
    cat "$OUT/session.json"
}

ALL="setup build kernels ds_ops ds_gemm_ref ds_attention ds_index ds_tiny mimo summary"
for step in ${*:-$ALL}; do
    case " $ALL " in
        *" $step "*) ;;
        *) die "unknown step $step (steps: $ALL)" ;;
    esac
    if [ "$step" = summary ]; then
        # Always rewritten: it reports whatever ran.
        summary
    else
        run_step "$step" "$step"
    fi
done
say "results in $OUT"
