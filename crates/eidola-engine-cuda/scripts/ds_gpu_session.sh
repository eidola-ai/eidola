#!/usr/bin/env bash
# One GPU session for the DeepSeek-V4.1 kernels and executor: every device
# test of this crate that needs no checkpoint, in order, unattended.
#
#   ds_gpu_session.sh [step ...]        (no argument: every step, in order)
#
#   steps: setup build kernels ds_ops ds_gemm_ref ds_gemm ds_gemm_bench
#          ds_attention ds_index ds_tiny ds_tiny_banded ds_tiny_drafting
#          ds_tiny_memory mimo summary
#   named only:  ds_gemm_profile
#   with the released checkpoint (never run by default; name them):
#          halves_build halves_encoder halves_decoder halves_reference
#          halves_invariance halves_banded
#
# `ds_gemm` is the quantised GEMMs on tensor cores: the probe of their block
# product, both forms against the scalar reference in every block shape, and
# their batch invariance (`ds_gemm.json`); `ds_gemm_bench` times every form
# and block shape per row count (`ds_gemm_bench.txt`, and its JSON lines in
# `ds_gemm_bench.jsonl`). The GEMMs alone, as one command:
#
#   ds_gpu_session.sh setup build kernels ds_gemm ds_gemm_bench summary
#
# `ds_gemm_profile` (never run by default; name it) profiles the tensor-core
# GEMMs with NVIDIA Nsight Compute, installing it from NVIDIA's apt repository
# when no `ncu` is found: both forms of `wo_b` (5120 × 8192) and `wq_b`
# (32768 × 1280) at 1 and 8 rows in the split block and the 8-row tile, and
# the routed experts' gate and up at 1 token and at 4,096 tokens (about 64
# rows an expert) in every block shape, each launched once
# (`ds_gemm_bench --once`). `ncu` runs under sudo (profiling counters are
# for administrators unless the driver allows them to all users), with the
# speed-of-light, memory, launch, occupancy, warp-state, scheduler,
# instruction and source-counter sections. Under $OUT/ncu: per case the
# report (`.ncu-rep`), its details page as text and as CSV, a per-kernel
# summary, and the bench's `launch` lines, which name each profiled launch in
# order. `DRY_RUN=1` prints every command of the step instead of running it.
#
#   ds_gpu_session.sh setup build kernels ds_gemm_profile
#
# `ds_tiny_banded` is `ds_tiny` with the executor's quantised GEMMs in the
# banded form (`EIDOLA_DS_GEMM_FORM=banded`), held to the same CPU reference
# executor bit for bit (`ds_tiny_banded.json`). `halves_banded` runs both
# device phases of the released model again in the banded form, under
# $OUT/ds_halves_banded, and requires every recording (boundary, both halves'
# layers, logits) to be byte for byte the exact run's: run it after
# `halves_encoder` and `halves_decoder`.
#
# `ds_tiny` is the executor without drafts; `ds_tiny_drafting` is every draft
# width and acceptance pattern, seeded sampling and the scenarios again with
# every draft slot drafted; `ds_tiny_memory` is idle encryption, resealing and
# the host tier (page-locked host memory) with drafts on. Each is the tiny
# fixture through the engine, held to the CPU reference executor step for
# step, and needs only the device and the kernel directory. `ds_tiny_memory`
# also locks a few megabytes of host memory for the reference executor's own
# host tier: the memory-lock hard limit (`ulimit -H -l`) must allow that.
#
# The `halves_*` steps are the released model on the exact executor, one half
# at a time (`examples/ds_halves.rs`): the encoder half's boundary and the
# decoder half's logits over real text, then the model crate's forward on the
# CPU over the same tokens, compared bit for bit. Each device phase ends with
# its invariance pass (the first positions again in other slots, blocks, rows
# and steps, equal bit for bit to the first run) and fails if it differs.
# `halves_reference` needs no GPU. Their recordings and `ds_halves.json` are
# under $OUT/ds_halves. `halves_invariance` is both device phases over the
# text's first 192 tokens only, for the invariance pass alone, under
# $OUT/ds_halves_invariance: it loads both halves again and leaves the
# recordings of the other steps as they are.
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
#   CHECKPOINT  the released checkpoint directory, for the `halves_*` steps; its
#            golden/ holds a reference-*.safetensors with the tokens and the Engram map
#   EIDOLA_DS_HALVES_TOKENS, EIDOLA_DS_HALVES_CHUNK   tokens of the text and positions
#            per step                                  (default 256 and 96)
#   EIDOLA_DS_HALVES_INVARIANCE   positions the invariance pass repeats; 0 for none
#                                                      (default twice the chunk)
#
# Expected time on one B300, after `setup` and `build` (about 10 minutes
# together on a fresh machine): kernels under 1 minute; ds_ops 5 to 10
# (the casts over every f32 through the probe, the released shapes);
# ds_gemm_ref 3 to 8 (the whole vocabulary through the head); ds_gemm about
# 12 (the probe is 8.6 billion dot products a case, 48 cases an image) and
# ds_gemm_bench about 3;
# ds_attention and ds_index 1 to 3 each; ds_tiny about 1 (the CPU reference
# executor runs beside the device and is most of it); ds_tiny_drafting 3 to 6
# and ds_tiny_memory 1 to 3 (not measured); mimo 3 to 6. With the checkpoint:
# a half loads in 2 to 3 minutes, a step of 96 positions computes in about 2
# seconds a half, the invariance pass adds about 15 seconds a half, and the
# CPU reference takes about 11 minutes; halves_invariance is two loads and
# under a minute of steps.
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
export CHECKPOINT="${CHECKPOINT:-}"
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
    if grep -q 'skipping:' "$OUT/logs/$name.log"; then
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
ds_gemm() { device_tests ds_gemm; }
ds_gemm_bench() {
    (cd "$REPO" && cargo run --release --locked -p eidola-engine-cuda --example ds_gemm_bench -- \
        "$KERNELS" --all-tiles) | tee "$OUT/ds_gemm_bench.txt"
    grep '^{' "$OUT/ds_gemm_bench.txt" > "$OUT/ds_gemm_bench.jsonl"
}
ds_attention() { device_tests ds_attention; }
ds_index() { device_tests ds_index; }
# A command, or with DRY_RUN set the command printed.
x() {
    if [ -n "${DRY_RUN:-}" ]; then
        printf 'would run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

find_ncu() {
    local c
    for c in "$(command -v ncu 2>/dev/null || true)" /usr/local/cuda/bin/ncu \
        /opt/nvidia/nsight-compute/*/ncu /usr/local/cuda-*/bin/ncu; do
        if [ -n "$c" ] && [ -x "$c" ]; then printf '%s\n' "$c"; return 0; fi
    done
    return 1
}

# Nsight Compute from NVIDIA's CUDA apt repository (its keyring first, when
# the repository is not configured): the newest `nsight-compute-<version>`.
install_ncu() {
    . /etc/os-release 2>/dev/null || true
    local release="ubuntu${VERSION_ID:-24.04}"
    release="${release//./}"
    if ! grep -rqs 'developer.download.nvidia.com/compute/cuda/repos' /etc/apt/; then
        x curl -fsSL -o "$WORK/cuda-keyring.deb" \
            "https://developer.download.nvidia.com/compute/cuda/repos/$release/x86_64/cuda-keyring_1.1-1_all.deb"
        x sudo dpkg -i "$WORK/cuda-keyring.deb"
    fi
    x sudo apt-get update -y
    local pkg
    pkg="$( (apt-cache search --names-only '^nsight-compute-[0-9]' 2>/dev/null || true) |
        cut -d' ' -f1 | sort -V | tail -1)"
    [ -n "$pkg" ] || pkg=nsight-compute
    x sudo apt-get install -y "$pkg"
}

# profile_case <name> <bench arguments>...: one bench run under ncu, and its
# report as text.
profile_case() {
    local name="$1"
    shift
    local ncu="$NCU" bin="$CARGO_TARGET_DIR/release/examples/ds_gemm_bench" dir="$OUT/ncu"
    local sudo=(sudo -E)
    [ "$(id -u)" -eq 0 ] && sudo=()
    x ${sudo[@]+"${sudo[@]}"} "$ncu" --target-processes all --kernel-name regex:'^eidola_ds_lin[48]_' \
        --section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
        --section LaunchStats --section Occupancy --section WarpStateStats --section SchedulerStats \
        --section ComputeWorkloadAnalysis --section InstructionStats --section SourceCounters \
        --export "$dir/$name" --force-overwrite \
        "$bin" "$KERNELS" --once --forms banded,tolerance "$@" > "$dir/$name.run.txt"
    grep '^launch ' "$dir/$name.run.txt" > "$dir/$name.launches.txt" || [ -n "${DRY_RUN:-}" ]
    x ${sudo[@]+"${sudo[@]}"} chown "$(id -u):$(id -g)" "$dir/$name.ncu-rep"
    x "$ncu" --import "$dir/$name.ncu-rep" --page details > "$dir/$name.details.txt"
    x "$ncu" --import "$dir/$name.ncu-rep" --page details --csv > "$dir/$name.details.csv"
    x "$ncu" --import "$dir/$name.ncu-rep" --print-summary per-kernel > "$dir/$name.summary.txt"
}

# The tensor-core GEMMs under Nsight Compute (the header says what).
ds_gemm_profile() {
    [ -n "${DRY_RUN:-}" ] || { [ -n "$KERNELS" ] && [ -d "$KERNELS/cubin" ]; } ||
        die "no kernel build output: set KERNELS"
    NCU="$(find_ncu || true)"
    if [ -z "$NCU" ]; then
        say "installing Nsight Compute"
        install_ncu
        NCU="$(find_ncu || echo /usr/local/cuda/bin/ncu)"
    fi
    export NCU
    x "$NCU" --version
    (cd "$REPO" && x cargo build --release --locked -p eidola-engine-cuda --example ds_gemm_bench)
    mkdir -p "$OUT/ncu"
    profile_case wo_b --only dense --dense 5120,8192 --rows 1,8 --tiles s8,t8
    profile_case wq_b --only dense --dense 32768,1280 --rows 1,8 --tiles s8,t8
    profile_case experts --only experts --rows 1,4096 --tiles s8,t8,t32
    ls -l "$OUT/ncu"
}

# device_test_named <test binary> <test name>: one test of a binary.
device_test_named() {
    (cd "$REPO" && cargo test --release --locked -p eidola-engine-cuda --test "$1" -- \
        --nocapture --test-threads 1 --exact "$2")
}

# The executor: the tiny fixture and its wide-window configuration through the
# engine, equal to the CPU reference executor step for step; then with drafts,
# then its memory (idle encryption, resealing, the host tier).
ds_tiny() { device_test_named ds_tiny the_tiny_model_equals_the_cpu_reference_executor; }
ds_tiny_drafting() {
    device_test_named ds_tiny the_tiny_model_drafts_as_the_cpu_reference_executor
}
ds_tiny_banded() {
    EIDOLA_DS_GEMM_FORM=banded EIDOLA_DS_TINY_SUMMARY="$OUT/ds_tiny_banded.json" \
        device_test_named ds_tiny the_tiny_model_equals_the_cpu_reference_executor
}
ds_tiny_memory() {
    device_test_named ds_tiny the_tiny_model_keeps_its_memory_as_the_cpu_reference_executor
}

# MiMo's device suites that need no checkpoint: nothing of the first model's
# kernels or wrappers moved.
mimo() {
    device_tests gemm moe_gemm attention sampler kv kv_cipher copy_rows qkv_rope_kv \
        quant_ops moe_ops preconditions
}

# The released model, one half at a time. Each device step loads its half
# (about 251 GB and 149 GB of weights from the checkpoint's files, the long
# part), runs the tokens in chunks and leaves its recording; the reference is
# the CPU's.
halves() {
    [ -n "${CHECKPOINT:-}" ] && [ -d "$CHECKPOINT" ] || die "CHECKPOINT names no directory"
    (cd "$REPO" && EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_HALVES_OUT="$OUT/ds_halves" \
        EIDOLA_DS_HALVES_PHASES="$1" \
        cargo run --release --locked -p eidola-engine-cuda --example ds_halves)
}
halves_build() {
    (cd "$REPO" && cargo build --release --locked -p eidola-engine-cuda --example ds_halves)
}
halves_encoder() { halves encoder; }
halves_decoder() { halves decoder; }
halves_reference() { halves reference; }
# Both device phases over the first 192 tokens, for their invariance passes.
halves_invariance() {
    [ -n "${CHECKPOINT:-}" ] && [ -d "$CHECKPOINT" ] || die "CHECKPOINT names no directory"
    (cd "$REPO" && EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" \
        EIDOLA_DS_HALVES_OUT="$OUT/ds_halves_invariance" \
        EIDOLA_DS_HALVES_PHASES=encoder,decoder EIDOLA_DS_HALVES_TOKENS=192 \
        EIDOLA_DS_HALVES_CHUNK=96 EIDOLA_DS_HALVES_INVARIANCE=192 \
        cargo run --release --locked -p eidola-engine-cuda --example ds_halves)
}

# The released model's device phases with the banded GEMMs, and every
# recording against the exact run's.
halves_banded() {
    [ -n "${CHECKPOINT:-}" ] && [ -d "$CHECKPOINT" ] || die "CHECKPOINT names no directory"
    [ -f "$OUT/ds_halves/logits.bin" ] || die "run halves_encoder and halves_decoder first"
    (cd "$REPO" && EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_HALVES_OUT="$OUT/ds_halves_banded" \
        EIDOLA_DS_HALVES_PHASES=encoder,decoder EIDOLA_DS_GEMM_FORM=banded \
        cargo run --release --locked -p eidola-engine-cuda --example ds_halves)
    local f
    for f in boundary_entries.bin boundary_streams.bin boundary_pre.bin encoder_layers.bin \
        decoder_layers.bin logits.bin; do
        cmp "$OUT/ds_halves/$f" "$OUT/ds_halves_banded/$f" || die "banded $f differs from the exact run's"
        say "banded $f: equal"
    done
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

ALL="setup build kernels ds_ops ds_gemm_ref ds_gemm ds_gemm_bench ds_attention ds_index ds_tiny ds_tiny_banded ds_tiny_drafting ds_tiny_memory mimo summary"
NAMED="ds_gemm_profile halves_build halves_encoder halves_decoder halves_reference halves_invariance halves_banded"
for step in ${*:-$ALL}; do
    case " $ALL $NAMED " in
        *" $step "*) ;;
        *) die "unknown step $step (steps: $ALL $NAMED)" ;;
    esac
    if [ "$step" = summary ]; then
        # Always rewritten: it reports whatever ran.
        summary
    else
        run_step "$step" "$step"
    fi
done
say "results in $OUT"
