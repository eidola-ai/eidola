#!/usr/bin/env bash
# One GPU session with the reference's real tilelang kernels.
#
#   gpu_session.sh <step> [...]      steps: setup probes fixture fetch real rust summary all
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
REPO="deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION="08c8657f02291e11b6f6b63e0e16014f8e748518"
REFERENCE="$WORK/reference"
PY="$WORK/venv/bin/python"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HERE WORK OUT SPREAD_DIR PREFIX_DIR TORCH_INDEX TORCH_VERSION SPREAD_TOKENS REPO REVISION REFERENCE PY

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

fixture_native() {
    "$PY" -I "$HERE/compare_real_kernels.py" --reference-dir "$REFERENCE" --json "$OUT/fixture-native.json"
}

fixture_pinned() {
    "$PY" -I "$HERE/compare_real_kernels.py" --reference-dir "$REFERENCE" --pinned --json "$OUT/fixture-pinned.json"
}

fetch_spread() {
    "$PY" -I "$HERE/fetch_truncated.py" --dir "$SPREAD_DIR" --truncation spread
}

fetch_prefix() {
    "$PY" -I "$HERE/fetch_truncated.py" --dir "$PREFIX_DIR" --truncation prefix --reuse "$SPREAD_DIR"
}

# real <truncation> <dir> <mode> [extra args]: the reference with its real
# kernels; the golden is copied to $OUT/goldens/<truncation>/.
real() {
    local truncation="$1" dir="$2" mode="$3"; shift 3
    local flag=(); [ "$mode" = native ] && flag=(--native-arithmetic)
    "$PY" -I "$HERE/real_golden.py" --dir "$dir" --reference-dir "$REFERENCE" --truncation "$truncation" \
        --kernels real --rows-from "$SPREAD_DIR" --rows-from "$PREFIX_DIR" "${flag[@]}" "$@"
    mkdir -p "$OUT/goldens/$truncation"
    cp "$dir/golden/reference-gpu-$mode.safetensors" "$dir/golden/engram-rows.safetensors" "$OUT/goldens/$truncation/"
}
real_spread_native() { real spread "$SPREAD_DIR" native --tokens "$SPREAD_TOKENS"; }
real_spread_pinned() { real spread "$SPREAD_DIR" pinned --tokens "$SPREAD_TOKENS"; }
real_prefix_native() { real prefix "$PREFIX_DIR" native; }
real_prefix_pinned() { real prefix "$PREFIX_DIR" pinned; }

# Optional: our forward against the goldens made here, on this machine.
rust_tests() {
    if ! command -v cargo >/dev/null; then
        say "installing rustup (the toolchain comes from rust-toolchain.toml)"
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    fi
    command -v cc >/dev/null || die "no C compiler: apt-get install -y build-essential"
    local root; root="$(cd "$HERE/../../../.." && pwd)"
    ( cd "$root" && EIDOLA_DEEPSEEK_TRUNCATED_DIR="$SPREAD_DIR" cargo test --release -p eidola-engine-model \
        --test deepseek_real -- --ignored --nocapture )
    ( cd "$root" && EIDOLA_DEEPSEEK_PREFIX_DIR="$PREFIX_DIR" cargo test --release -p eidola-engine-model \
        --test deepseek_real_prefix -- --ignored --nocapture )
}

summary() {
    {
        echo "GPU session summary, $(date -u +%Y-%m-%dT%H:%MZ)"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
        for log in probes fixture-native fixture-pinned real-spread-native real-spread-pinned real-prefix-native real-prefix-pinned rust; do
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
        fixture) run_step fixture-native fixture_native; run_step fixture-pinned fixture_pinned ;;
        fetch)   run_step fetch-spread fetch_spread; run_step fetch-prefix fetch_prefix ;;
        real)    run_step real-spread-native real_spread_native; run_step real-prefix-native real_prefix_native
                 run_step real-spread-pinned real_spread_pinned; run_step real-prefix-pinned real_prefix_pinned ;;
        rust)    run_step rust rust_tests ;;
        summary) summary ;;
        all)     for s in setup probes fixture fetch real summary; do step "$s"; done ;;
        *)       die "unknown step $1 (setup probes fixture fetch real rust summary all)" ;;
    esac
}

[ $# -gt 0 ] || die "usage: gpu_session.sh <step> [...]   steps: setup probes fixture fetch real rust summary all"
for s in "$@"; do step "$s"; done
