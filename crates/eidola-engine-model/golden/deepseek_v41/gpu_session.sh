#!/usr/bin/env bash
# One GPU session with the reference's real tilelang kernels.
#
#   gpu_session.sh <step> [...]
#
#   steps over the two truncations:  setup probes kernels fetch real rust summary   (all = these but rust)
#   steps over the whole checkpoint: full-fetch full-real full-pinned full-rust     (full = these, then summary)
#
# Run from a checkout of this repository on a Linux machine with one NVIDIA
# GPU and CUDA. Every step is idempotent and resumable: a finished step leaves
# a marker under $OUT/done/ and is skipped on a rerun (delete the marker to
# repeat it). Everything a step prints is also in $OUT/logs/<step>.log, and
# every result is JSON or a golden file under $OUT. Copy $OUT back.
#
# Environment (all optional):
#   WORK         scratch for the venv, the reference code and weights   (default ~/eidola-gpu)
#   OUT          results                                                (default $WORK/out)
#   SPREAD_DIR   the 12-layer truncation, pre-fetched or to fetch into  (default $WORK/spread)
#   PREFIX_DIR   the whole-layers-0-3 truncation, likewise              (default $WORK/prefix)
#   TORCH_INDEX  the PyTorch wheel index for this machine's CUDA        (default .../whl/cu130)
#   TORCH_VERSION                                                        (default 2.10.0)
#   SPREAD_TOKENS  tokens of the 12-layer run                            (default 256, as measured locally)
#   FULL_DIR     the whole checkpoint as published (about 510 GB)       (default $WORK/full)
#   FULL_TOKENS  tokens of the whole-model runs, 0 for the whole text   (default 0)
#   PY           the Python to run with                                 (default the venv `setup` makes)
#
# The Engram table rows a run needs cost two small range reads each, and the
# weight host rate-limits them (about 25 minutes per 12,000 requests). Upload
# <dir>/golden/engram-rows.safetensors from a machine that already fetched
# them and the run fetches none.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${WORK:-$HOME/eidola-gpu}"
OUT="${OUT:-$WORK/out}"
SPREAD_DIR="${SPREAD_DIR:-$WORK/spread}"
PREFIX_DIR="${PREFIX_DIR:-$WORK/prefix}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
SPREAD_TOKENS="${SPREAD_TOKENS:-256}"
FULL_DIR="${FULL_DIR:-$WORK/full}"
FULL_TOKENS="${FULL_TOKENS:-0}"
REPO="deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION="08c8657f02291e11b6f6b63e0e16014f8e748518"
REFERENCE="$WORK/reference"
PY="${PY:-$WORK/venv/bin/python}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HERE WORK OUT SPREAD_DIR PREFIX_DIR FULL_DIR FULL_TOKENS TORCH_INDEX TORCH_VERSION SPREAD_TOKENS REPO REVISION REFERENCE PY

mkdir -p "$WORK" "$OUT/done" "$OUT/logs" "$OUT/goldens"

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
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    touch "$OUT/done/$name"
    say "$name: done in $((SECONDS - began)) s"
}

setup() {
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    if ! command -v uv >/dev/null; then
        say "installing uv"
        curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    [ -x "$PY" ] || uv venv "$WORK/venv" --python 3.12
    uv pip install --python "$PY" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
    uv pip install --python "$PY" tilelang==0.1.12 numpy==2.3.4 sympy==1.14.0 tokenizers==0.22.1
    mkdir -p "$REFERENCE"
    for file in model.py engram.py kernel.py; do
        [ -s "$REFERENCE/$file" ] || curl -fsSL -A python-urllib -o "$REFERENCE/$file" \
            "https://huggingface.co/$REPO/resolve/$REVISION/inference/$file"
    done
    "$PY" - <<'PY'
import torch, tilelang
assert torch.cuda.is_available(), "PyTorch does not see a CUDA device (wrong TORCH_INDEX for this driver?)"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "tilelang", tilelang.__version__)
for name in ("float8_e4m3fn", "float8_e8m0fnu", "float4_e2m1fn_x2"):
    torch.empty(2, 2, dtype=getattr(torch, name), device="cuda")
PY
}

probes() {
    "$PY" -I "$HERE/gpu_probes.py" --reference-dir "$REFERENCE" --json "$OUT/probes.json"
}

# Every kernel once, real against emulated, on generated inputs at the
# released dimensions. (The committed fixture cannot be used: tilelang refuses
# to compile sparse_attn at its head dimension.)
kernels() {
    "$PY" -I "$HERE/compare_real_kernels.py" --reference-dir "$REFERENCE" --inputs generated --json "$OUT/kernels-generated.json"
}

fetch_spread() {
    "$PY" -I "$HERE/fetch_truncated.py" --dir "$SPREAD_DIR" --truncation spread
}

fetch_prefix() {
    "$PY" -I "$HERE/fetch_truncated.py" --dir "$PREFIX_DIR" --truncation prefix --reuse "$SPREAD_DIR"
}

# real <truncation> <dir> <mode> [extra args]: the reference with its real
# kernels; the golden is copied to $OUT/goldens/<truncation>/. The unpinned
# run also gives every kernel call to the emulation (real activations).
real() {
    local truncation="$1" dir="$2" mode="$3"; shift 3
    local flag=(); [ "$mode" = native ] && flag=(--native-arithmetic --kernel-stats "$OUT/kernels-$truncation.json")
    "$PY" -I "$HERE/real_golden.py" --dir "$dir" --reference-dir "$REFERENCE" --truncation "$truncation" \
        --kernels real --rows-from "$SPREAD_DIR" --rows-from "$PREFIX_DIR" "${flag[@]}" "$@"
    mkdir -p "$OUT/goldens/$truncation"
    cp "$dir/golden/reference-gpu-$mode.safetensors" "$dir/golden/engram-rows.safetensors" "$OUT/goldens/$truncation/"
}
real_spread_native() { real spread "$SPREAD_DIR" native --tokens "$SPREAD_TOKENS"; }
real_spread_pinned() { real spread "$SPREAD_DIR" pinned --tokens "$SPREAD_TOKENS"; }
real_prefix_native() { real prefix "$PREFIX_DIR" native; }
real_prefix_pinned() { real prefix "$PREFIX_DIR" pinned; }

# The reference on the CPU with emulated kernels and pinned arithmetic: the
# golden our forward must equal exactly, which the tests assert when present.
pinned_cpu() {
    local truncation="$1" dir="$2"; shift 2
    if [ -e "$dir/golden/reference-pinned.safetensors" ]; then say "$truncation: reference-pinned present"; return 0; fi
    "$PY" -I "$HERE/real_golden.py" --dir "$dir" --reference-dir "$REFERENCE" --truncation "$truncation" \
        --rows-from "$SPREAD_DIR" --rows-from "$PREFIX_DIR" "$@"
    mkdir -p "$OUT/goldens/$truncation"
    cp "$dir/golden/reference-pinned.safetensors" "$OUT/goldens/$truncation/"
}
pinned_cpu_spread() { pinned_cpu spread "$SPREAD_DIR" --tokens "$SPREAD_TOKENS"; }
pinned_cpu_prefix() { pinned_cpu prefix "$PREFIX_DIR"; }

rust_toolchain() {
    if ! command -v cargo >/dev/null; then
        say "installing rustup (the toolchain comes from rust-toolchain.toml)"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    fi
    command -v cc >/dev/null || die "no C compiler: apt-get install -y build-essential"
}

# Our forward against every golden in the two truncation directories. With
# reference-pinned present (pinned_cpu makes it) the tests assert exact
# equality with it; the GPU goldens are reported beside it.
rust_tests() {
    rust_toolchain
    local root; root="$(cd "$HERE/../../../.." && pwd)"
    ( cd "$root" && EIDOLA_DEEPSEEK_TRUNCATED_DIR="$SPREAD_DIR" cargo test --release -p eidola-engine-model \
        --test deepseek_real -- --ignored --nocapture )
    ( cd "$root" && EIDOLA_DEEPSEEK_PREFIX_DIR="$PREFIX_DIR" cargo test --release -p eidola-engine-model \
        --test deepseek_real_prefix -- --ignored --nocapture )
}

# --- The whole checkpoint, one half of the model at a time -----------------

full_fetch() {
    mkdir -p "$FULL_DIR"
    local base="https://huggingface.co/$REPO/resolve/$REVISION"
    for file in config.json tokenizer.json model.safetensors.index.json; do
        [ -s "$FULL_DIR/$file" ] || curl -fsSL -A python-urllib -o "$FULL_DIR/$file" "$base/$file"
    done
    # Shard 00001 is the vision tower, which nothing here reads.
    for i in $(seq 2 48); do
        local shard; shard="$(printf 'model-%05d-of-00048.safetensors' "$i")"
        [ -e "$FULL_DIR/$shard" ] && continue
        say "fetching $shard"
        curl -fL -C - --retry 8 --retry-delay 10 -A python-urllib -o "$FULL_DIR/$shard.partial" "$base/$shard"
        mv "$FULL_DIR/$shard.partial" "$FULL_DIR/$shard"
    done
}

full_halves() {
    local tag="$1"; shift
    "$PY" -I "$HERE/full_halves.py" --dir "$FULL_DIR" --reference-dir "$REFERENCE" --tokens "$FULL_TOKENS" "$@"
    mkdir -p "$OUT/goldens/full"
    cp "$FULL_DIR/golden/reference-$tag.safetensors" "$OUT/goldens/full/"
}
full_real_native() { full_halves gpu-native --kernels real --native-arithmetic; }
# On the CPU with emulated kernels and pinned arithmetic, one prefill chunk
# only (one token per call decodes every routed expert again at each step).
full_pinned() { full_halves pinned --patterns prefill; }

full_rust() {
    rust_toolchain
    local root; root="$(cd "$HERE/../../../.." && pwd)"
    ( cd "$root" && EIDOLA_DEEPSEEK_FULL_DIR="$FULL_DIR" cargo test --release -p eidola-engine-model \
        --test deepseek_real_full -- --ignored --nocapture )
}

summary() {
    {
        echo "GPU session summary, $(date -u +%Y-%m-%dT%H:%MZ)"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
        for log in probes kernels real-spread-native real-spread-pinned real-prefix-native real-prefix-pinned rust full-real-native full-pinned full-rust; do
            [ -e "$OUT/logs/$log.log" ] || { echo; echo "== $log: not run"; continue; }
            echo; echo "== $log"
            grep -vE '^[0-9:]{8} (Engram layer|running)' "$OUT/logs/$log.log" | tail -n 45
        done
    } > "$OUT/summary.txt"
    say "wrote $OUT/summary.txt"
}

step() {
    case "$1" in
        setup)   run_step setup setup ;;
        probes)  run_step probes probes ;;
        kernels) run_step kernels kernels ;;
        fetch)   run_step fetch-spread fetch_spread; run_step fetch-prefix fetch_prefix ;;
        real)    run_step real-spread-native real_spread_native; run_step real-prefix-native real_prefix_native
                 run_step real-spread-pinned real_spread_pinned; run_step real-prefix-pinned real_prefix_pinned ;;
        rust)    run_step pinned-cpu-prefix pinned_cpu_prefix; run_step pinned-cpu-spread pinned_cpu_spread
                 run_step rust rust_tests ;;
        summary) summary ;;
        all)     for s in setup probes kernels fetch real summary; do step "$s"; done ;;
        full-fetch)  run_step full-fetch full_fetch ;;
        full-real)   run_step full-real-native full_real_native ;;
        full-pinned) run_step full-pinned full_pinned ;;
        full-rust)   run_step full-rust full_rust ;;
        full)        for s in setup full-fetch full-real full-pinned full-rust summary; do step "$s"; done ;;
        *)       die "unknown step $1 (setup probes kernels fetch real rust summary all; full-fetch full-real full-pinned full-rust full)" ;;
    esac
}

[ $# -gt 0 ] || die "usage: gpu_session.sh <step> [...]   steps: setup probes kernels fetch real rust summary all; full-fetch full-real full-pinned full-rust full"
for s in "$@"; do step "$s"; done
