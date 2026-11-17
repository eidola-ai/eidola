#!/usr/bin/env bash
# What the Qwen3.8 GPU sessions share (sourced by qwen_probe_session.sh,
# qwen_gpu_session.sh and qwen_real_session.sh; not run on its own).
#
# A session is a list of steps. A finished step leaves a marker under
# $OUT/done/ and is skipped on a rerun, so a session resumes, on the same
# machine or on a fresh one with $OUT copied back (a preempted instance).
# `setup`, `build` and `kernels_build` are never marked: they rebuild what a
# fresh machine lacks and cost little where it is already there. A step that
# fails exits non-zero, leaves no marker and stops the session; a device test
# that printed `SKIPPED` fails its step. Everything a step prints is in
# $OUT/logs/<step>.log, and `summary` writes $OUT/session.json.
#
# DRY_RUN=1 prints every command a step would run, runs none and marks
# nothing.
#
# Environment (all optional):
#   WORK       scratch: the Rust target directory, the kernel build, the venvs
#              (default ~/eidola-qwen)
#   OUT        results (default $WORK/out)
#   KERNELS    a kernel build output directory; default: built on this
#              machine by `kernels_build` into $WORK/kernels
#   DRY_RUN    1: print, run nothing
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${WORK:-$HOME/eidola-qwen}"
OUT="${OUT:-$WORK/out}"
DRY_RUN="${DRY_RUN:-}"
KERNELS="${KERNELS:-$WORK/kernels}"
export PATH="$HOME/.cargo/bin:$PATH"
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$REPO/rust-toolchain.toml")"
export RUSTUP_TOOLCHAIN
# One target directory for the session: every test and example is built once
# (`build`) and reused by the steps after it.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/target}"
export EIDOLA_ENGINE_KERNELS_DIR="$KERNELS"
export EIDOLA_QWEN_SUMMARY_DIR="$OUT"
export HERE REPO WORK OUT KERNELS DRY_RUN

mkdir -p "$WORK" "$OUT/done" "$OUT/logs"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "FAILED: $*" >&2; exit 1; }

# x <command...>: run it, or under DRY_RUN print it.
x() {
    if [ -n "$DRY_RUN" ]; then
        printf 'would run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# Steps a rerun repeats (cheap where their output is already there).
UNMARKED=" setup build kernels_build summary "

# run_step <name>: skip when done, log, mark, fail loudly.
run_step() {
    local name="$1"
    if [ -e "$OUT/done/$name" ]; then say "$name: already done"; return 0; fi
    say "$name: start"
    local began=$SECONDS
    # A fresh shell, so that errexit is live in the step whatever called it.
    set +e
    bash -euo pipefail -c "$(declare -f); $name" 2>&1 | tee "$OUT/logs/$name.log"
    local status=${PIPESTATUS[0]}
    set -e
    local seconds=$((SECONDS - began))
    [ -n "$DRY_RUN" ] && { say "$name: dry run ($status)"; [ "$status" -eq 0 ] || die "$name: dry run exited with $status"; return 0; }
    printf '%s %s %s\n' "$name" "$status" "$seconds" >> "$OUT/steps.txt"
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    if grep -q 'SKIPPED' "$OUT/logs/$name.log"; then
        die "$name skipped a device test (see $OUT/logs/$name.log)"
    fi
    case "$UNMARKED" in *" $name "*) ;; *) touch "$OUT/done/$name" ;; esac
    say "$name: done in $seconds s"
}

# device_tests <test binary>...: this crate's tests, one at a time on the device.
device_tests() {
    local args=()
    for t in "$@"; do args+=(--test "$t"); done
    (cd "$REPO" && x cargo test --release --locked -p eidola-engine-cuda "${args[@]}" -- \
        --nocapture --test-threads 1)
}

facts() {
    x nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
    x nvidia-smi -q -d COMPUTE
    say "memory-lock limit: $(ulimit -l) (hard $(ulimit -H -l))"
    if [ -z "$DRY_RUN" ]; then
        say "cpus: $(nproc), memory: $(awk '/MemTotal/ {print $2 " kB"}' /proc/meminfo)"
    fi
    x git -C "$REPO" rev-parse HEAD
}

setup() {
    command -v nvidia-smi >/dev/null || [ -n "$DRY_RUN" ] || die "nvidia-smi not found: no NVIDIA driver"
    if ! command -v cc >/dev/null || ! command -v c++ >/dev/null || ! command -v curl >/dev/null; then
        x sudo apt-get update -y
        x sudo apt-get install -y build-essential curl ca-certificates
    fi
    if ! command -v rustup >/dev/null; then
        if [ -n "$DRY_RUN" ]; then
            say "would install rustup"
        else
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
                sh -s -- -y --profile minimal --default-toolchain none
        fi
    fi
    x rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal
}

# Every test and example the steps run, built once, release.
build() {
    (cd "$REPO" && x cargo test --release --locked -p eidola-engine-cuda -p eidola-engine-kernels --no-run)
    (cd "$REPO" && x cargo build --release --locked -p eidola-engine-cuda --examples)
}

# The kernel set built on this machine (x86_64, Docker, pinned Nix) and held to
# the committed manifest byte for byte: the second builder architecture's
# reproducibility check, and the images every device step loads. Skipped when
# $KERNELS already holds a build that verifies.
kernels_build() {
    if [ -f "$KERNELS/manifest.json" ] && cmp -s "$KERNELS/manifest.json" \
        "$REPO/crates/eidola-engine-kernels/kernels.manifest.json"; then
        say "kernels_build: $KERNELS already matches the committed manifest"
        return 0
    fi
    command -v docker >/dev/null || [ -n "$DRY_RUN" ] || die "docker not found"
    x "$REPO/crates/eidola-engine-kernels/scripts/build-in-docker.sh" --platform linux/amd64 \
        --out "$KERNELS" --verify
}

# The images load and every entry resolves; the manifest holds.
kernels() {
    [ -n "$DRY_RUN" ] || [ -d "$KERNELS/cubin" ] || die "no kernel build output in $KERNELS"
    (cd "$REPO" && x cargo test --release --locked -p eidola-engine-kernels --test manifest)
    device_tests smoke
}

summary() {
    [ -n "$DRY_RUN" ] && { say "would write $OUT/session.json"; return 0; }
    {
        printf '{\n  "steps": [\n'
        local sep=""
        if [ -f "$OUT/steps.txt" ]; then
            while read -r name status seconds; do
                printf '%s    {"step": "%s", "status": %s, "seconds": %s}' "$sep" "$name" "$status" "$seconds"
                sep=$',\n'
            done < "$OUT/steps.txt"
        fi
        printf '\n  ],\n  "results": ['
        sep=""
        for f in "$OUT"/*.json "$OUT"/*.txt; do
            [ -e "$f" ] || continue
            case "$(basename "$f")" in session.json|steps.txt) continue ;; esac
            printf '%s"%s"' "$sep" "$(basename "$f")"
            sep=", "
        done
        printf ']\n}\n'
    } > "$OUT/session.json"
    cat "$OUT/session.json"
}

# main <all steps> <named-only steps> [steps...]: run the steps given, or all.
main() {
    local all="$1" named="$2"
    shift 2
    for step in ${*:-$all}; do
        case " $all $named " in
            *" $step "*) ;;
            *) die "unknown step $step (steps: $all; named only: $named)" ;;
        esac
        if [ "$step" = summary ]; then summary; else run_step "$step"; fi
    done
    say "results in $OUT"
}
