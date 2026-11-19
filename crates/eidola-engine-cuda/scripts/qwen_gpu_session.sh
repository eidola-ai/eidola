#!/usr/bin/env bash
# The Qwen3.8 device session on one B200: every kernel and the tiny scenarios on the
# device, no checkpoint.
#
#   qwen_gpu_session.sh [step ...]      (no argument: every step, in order)
#   DRY_RUN=1 qwen_gpu_session.sh       print every command, run none
#
#   steps: facts setup build kernels_build kernels qwen_kernels qwen_host
#          qwen_tiny qwen_fast regressions qwen_bench summary
#   named only: qwen_profile
#
# - `qwen_kernels`: every kernel against the model crate bit for bit; the fast
#   linears' deviation recorded and their row invariance asserted;
# - `qwen_host`: the executor on the kernels' bodies compiled for this host,
#   against the CPU reference executor (as on a developer machine);
# - `qwen_tiny`: the executor's exact profile on the device against the CPU
#   reference executor step for step (tokens, logits bits, state images, the
#   memory scan): chunkings, both state strategies, drafting, hits, preemption,
#   the host tier;
# - `qwen_fast`: the fast profile's reproducibility between calls, asserted
#   (alone and beside traffic, chunked, decoded and prefilled, a cache hit and
#   fresh, every draft width), and its deviation from the exact profile,
#   recorded in $OUT/qwen_fast_deviation.txt;
# - `regressions`: MiMo's and DeepSeek's device suites that need no
#   checkpoint, unchanged;
# - `qwen_bench`: as in qwen_probe_session.sh;
# - `qwen_profile` (named only): Nsight Compute over one bench run, reports
#   under $OUT/ncu.
#
# Expected: build 25 to 35 minutes on 20 vCPUs; the device steps 1 to 15
# minutes each (estimates). See qwen_session_lib.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen_session_lib.sh"

qwen_kernels() { device_tests qwen_ops; }
qwen_host() { device_tests qwen_kernels_host qwen_host; }
qwen_tiny() { device_tests qwen_tiny; }
qwen_fast() { device_tests qwen_fast; }
regressions() {
    device_tests smoke sampler kv_cipher mimo_gemm mimo_moe_gemm mimo_attention mimo_kv \
        mimo_copy_rows mimo_qkv_rope_kv mimo_quant_ops mimo_moe_ops mimo_preconditions \
        deepseek_ops deepseek_gemm_ref deepseek_attention deepseek_index deepseek_tiny
}

qwen_bench() {
    if [ -n "$DRY_RUN" ]; then
        (cd "$REPO" && x cargo run --release --locked -p eidola-engine-cuda --example qwen_kernel_bench -- "$KERNELS")
        return 0
    fi
    (cd "$REPO" && cargo run --release --locked -p eidola-engine-cuda --example qwen_kernel_bench -- \
        "$KERNELS") | tee "$OUT/qwen_bench.txt"
    grep '^{' "$OUT/qwen_bench.txt" > "$OUT/qwen_bench.jsonl" || die "the bench printed no measurement"
}

qwen_profile() {
    local ncu
    ncu="$(command -v ncu || ls /usr/local/cuda*/bin/ncu 2>/dev/null | head -1 || true)"
    if [ -z "$ncu" ]; then
        x sudo apt-get install -y nsight-compute || die "no ncu, and nsight-compute did not install"
        ncu="$(command -v ncu || ls /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)"
    fi
    mkdir -p "$OUT/ncu"
    local bin
    bin="$CARGO_TARGET_DIR/release/examples/qwen_kernel_bench"
    x sudo "$ncu" --set full --launch-count 40 -o "$OUT/ncu/qwen_kernels" -f "$bin" "$KERNELS" --quick
}

main "facts setup build kernels_build kernels qwen_kernels qwen_host qwen_tiny qwen_fast regressions qwen_bench summary" \
    "qwen_profile" "$@"
