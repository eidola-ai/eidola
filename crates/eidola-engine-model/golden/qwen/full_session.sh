#!/usr/bin/env bash
# The whole released model: our forward, the pinned reference, the reference
# as released on a GPU, and the health check, with comparable outputs.
#
#   full_session.sh <step> [...]     steps: setup fetch tokens pinned native released released-torch health rust summary
#   full_session.sh all              every step this machine can run, in that order
#   DRY_RUN=1 full_session.sh all    print what each step would run, run nothing
#
# Run from a checkout of this repository. On a Mac or any CPU machine
# (DEVICE=cpu, the default) `released`, `released-torch` and the GPU health
# run are skipped; on a Linux machine with one NVIDIA GPU (DEVICE=cuda) every
# step runs. Every step is idempotent and resumable: a finished step leaves a
# marker under $OUT/done/ and is skipped on a rerun (delete the marker to repeat
# it). A step that fails exits non-zero, leaves no marker and stops the
# session; everything it printed is in $OUT/logs/<step>.log.
#
# Every checkpoint file is checked against the recorded upstream SHA-256
# (upstream-sha256.json) before it gets the name anything loads, and again by
# every step that reads it: a download that resumes or runs in parts can leave
# the right size and the wrong bytes.
#
# Environment (all optional):
#   WORK          scratch: venvs, results                           (default ~/eidola-qwen)
#   CKPT          the whole checkpoint, pre-fetched or to fetch into (default $WORK/checkpoint)
#   OUT           results                                           (default $WORK/out)
#   DEVICE        cpu or cuda                                       (default cpu)
#   TORCH_INDEX   PyTorch wheel index for this machine's CUDA        (default .../whl/cu130)
#   TORCH_VERSION                                                    (default 2.13.0, the CPU lock's)
#   DRY_RUN       1 prints the commands and runs none
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
WORK="${WORK:-$HOME/eidola-qwen}"
CKPT="${CKPT:-$WORK/checkpoint}"
OUT="${OUT:-$WORK/out}"
DEVICE="${DEVICE:-cpu}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
DRY_RUN="${DRY_RUN:-0}"
GPU_PY="$WORK/venv-gpu/bin/python"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export HERE ROOT WORK CKPT OUT DEVICE TORCH_INDEX TORCH_VERSION DRY_RUN GPU_PY

case "$DEVICE" in cpu|cuda) ;; *) echo "DEVICE is cpu or cuda, not $DEVICE" >&2; exit 2 ;; esac
mkdir -p "$WORK" "$OUT/done" "$OUT/logs"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "FAILED: $*" >&2; exit 1; }

# run <command...>: run it, or under DRY_RUN print it.
run() {
    if [ "$DRY_RUN" = 1 ]; then printf 'would run:'; printf ' %q' "$@"; printf '\n'; return 0; fi
    "$@"
}

# The CPU tooling: this directory's locked project.
cpu_py() { (cd "$HERE" && run uv run --frozen python -I "$@"); }

# run_step <name> <function>: skip when done, log, mark, fail loudly.
run_step() {
    local name="$1" fn="$2"
    if [ -e "$OUT/done/$name" ]; then say "$name: already done"; return 0; fi
    say "$name: start"
    local began=$SECONDS status
    set +e
    bash -euo pipefail -c "$(declare -f); $fn" 2>&1 | tee "$OUT/logs/$name.log"
    status=${PIPESTATUS[0]}
    set -e
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    if [ "$DRY_RUN" = 1 ]; then say "$name: dry run, no marker"; return 0; fi
    touch "$OUT/done/$name"
    say "$name: done in $((SECONDS - began)) s"
}

setup() {
    command -v uv >/dev/null || run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    (cd "$HERE" && run uv sync --frozen)
    if [ "$DEVICE" = cuda ]; then
        command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver"
        run nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        [ -x "$GPU_PY" ] || run uv venv "$WORK/venv-gpu" --python 3.12
        run uv pip install --python "$GPU_PY" "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
        run uv pip install --python "$GPU_PY" transformers==5.23.0 numpy==2.3.4 kernels==0.18.3 \
            flash-linear-attention==0.6.0
        run "$GPU_PY" -c 'import torch; assert torch.cuda.is_available(), "PyTorch sees no CUDA device"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
    fi
    command -v cargo >/dev/null || run sh -c 'curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none'
}

fetch() { cpu_py fetch_checkpoint.py --dir "$CKPT" --truncation full; }

# The health check's token sequences, from the checkpoint's own tokenizer and
# chat template.
tokens() { cpu_py health_check.py --dir "$CKPT" --tokens-only; }

# The reference on the CPU with its unspecified operations pinned: what our
# forward must equal bit for bit at full depth.
pinned() { cpu_py real_golden.py --dir "$CKPT" --layers 64 --mode pinned; }

# The reference on the CPU with PyTorch's own arithmetic and its chunked
# DeltaNet prefill. PyTorch's BF16 depthwise convolution is slow on a CPU, so
# it runs eight one-token calls, not 32.
native() { cpu_py real_golden.py --dir "$CKPT" --layers 64 --mode native --step-tokens 72; }

# The reference as released, on the GPU: Transformers' own FP8 dispatch and
# FLA's DeltaNet kernels.
released() {
    (cd "$HERE" && run "$GPU_PY" -I real_golden.py --dir "$CKPT" --layers 64 --mode released --device cuda)
}

# The same with the model code's own PyTorch DeltaNet (FLA hidden).
released_torch() {
    (cd "$HERE" && run "$GPU_PY" -I real_golden.py --dir "$CKPT" --layers 64 --mode released --device cuda \
        --torch-deltanet --name gpu-torch)
}

# The reference's health: pinned on the CPU, or as released on the GPU.
health() {
    if [ "$DEVICE" = cuda ]; then
        (cd "$HERE" && run "$GPU_PY" -I health_check.py --dir "$CKPT" --mode released --device cuda)
    else
        cpu_py health_check.py --dir "$CKPT" --mode pinned
    fi
}

# Our forward at full depth against every golden, and its own health.
rust() {
    cd "$ROOT"
    run env EIDOLA_QWEN_DIR="$CKPT" EIDOLA_QWEN_LAYERS=64 cargo test --release -p eidola-engine-model \
        --test qwen_real -- --ignored --nocapture
    run env EIDOLA_QWEN_DIR="$CKPT" cargo test --release -p eidola-engine-model \
        --test qwen_health -- --ignored --nocapture
}

summary() {
    [ "$DRY_RUN" = 1 ] && { say "summary: dry run"; return 0; }
    {
        for f in "$OUT"/logs/*.log; do
            echo "== $(basename "$f" .log)"
            grep -E 'accuracy|capital|against|differ|top-1|FAILED|healthy|one call|RSS|wrote' "$f" || true
        done
    } > "$OUT/summary.txt"
    cat "$OUT/summary.txt"
    # Every step that ran must have finished.
    for f in "$OUT"/logs/*.log; do
        local step; step="$(basename "$f" .log)"
        [ "$step" = summary ] || [ -e "$OUT/done/$step" ] || die "$step has a log and no done marker"
    done
}

STEPS=(setup fetch tokens pinned native released released-torch health rust summary)
gpu_only() { [ "$1" = released ] || [ "$1" = released-torch ]; }
[ $# -gt 0 ] || { sed -n '2,30p' "$0"; exit 2; }
for arg in "$@"; do
    if [ "$arg" = all ]; then
        for s in "${STEPS[@]}"; do
            if gpu_only "$s" && [ "$DEVICE" != cuda ]; then say "$s: needs DEVICE=cuda; not run"; continue; fi
            run_step "$s" "${s//-/_}"
        done
        continue
    fi
    printf '%s\n' "${STEPS[@]}" | grep -qx -- "$arg" || die "unknown step $arg"
    if gpu_only "$arg" && [ "$DEVICE" != cuda ]; then die "$arg needs DEVICE=cuda"; fi
    run_step "$arg" "${arg//-/_}"
done
