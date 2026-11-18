#!/usr/bin/env bash
# The Qwen3.8 performance session on one B200: the step program (decode graphs
# per rung, FA2 at head dimension 256, sampling and chain acceptance on the
# device) held to the CPU reference executor and to reproducibility between
# calls, then the released checkpoint served at the reference stack's
# workloads.
#
#   qwen_perf_session.sh [step ...]     (no argument: every step, in order)
#   DRY_RUN=1 qwen_perf_session.sh      print every command, run none
#
#   steps: facts setup build kernels_build kernels perf_host perf_tiny
#          perf_fast perf_kernels serve_b serve_b_mtp serve_c serve_c_mtp
#          summary
#   named only: serve_b_eager serve_profile
#
# - `perf_host`: every kernel body compiled for this host against the model
#   crate (the quantiser's warp form against its reference form too), the
#   blocked DeltaNet's host form against its definition and reproducible, and
#   the executor on those bodies against the CPU reference executor, eagerly and
#   on the padded rung path; the blocked DeltaNet through the scheduler,
#   reproducible; the fast profile's plans packed on every path;
# - `perf_tiny`: every kernel on the device against the model crate, the
#   quantiser's kernel against its reference form, the fast linears and BF16
#   GEMM against the exact ones (a row's bits alone and in batches, asserted;
#   deviation in $OUT/qwen_ops_fast_*.txt), every FP8 GEMM the fast profile
#   routes rows to (the swapped kernel's three tiles and the unswapped one,
#   and the split instances at the split shapes) giving each row the same
#   bits, the split's ranges and sum in their order, a stack's parts against
#   each part alone, the small-N BF16 GEMM against its order and
#   the f32 gates against the exact ones (asserted within two BF16 units;
#   histogram in $OUT/qwen_ops_fast_gates.txt), the blocked DeltaNet (its intra
#   kernel bit for bit its host body, the whole form within a bound of its
#   definition, its distance from the token recurrence in
#   $OUT/qwen_ops_blocked_deltanet_*.txt, and reproducible bit for bit: a
#   block's rows with 1 to 16 of its positions present (the earlier ones from
#   the tail cache), chunks of 1 to 65, drafts accepted and rejected,
#   the fused form bit for bit the unfused one's at 1, 8 and 64 sequences,
#   beside other runs); then the exact profile on the
#   device against the CPU reference executor, eagerly and replaying graphs
#   (graphs keep it exact);
# - `perf_fast`: both profiles' reproducibility between calls, asserted, at KV
#   blocks of 4 and of 16 (a 70-token prompt in chunks of 1 to 65 across the
#   blocked DeltaNet's blocks), eagerly and replaying graphs (a 1,800-position
#   prompt among them, past several FA2 chunks and prefilled in two attention
#   passes), and the same workloads on every path (eager,
#   a rung run directly, its graph replayed) equal in tokens and logits bits;
#   the fast profile's deviation recorded in $OUT/qwen_fast_deviation.txt;
# - `perf_kernels`: the kernel bench (`examples/qwen_kernel_bench.rs`) as JSON
#   lines in $OUT/qwen_kernel_bench.jsonl: every FP8 GEMM form at 1 to 2,048
#   rows at the stacked shapes (a split shape's split and unsplit forms side
#   by side), a target layer's FA2 decode launch at 1 to 64 rows and 1,024 to
#   12,288 keys split at 256, 512 and 1,024, the small-N and CUTLASS `a`/`b`
#   projection,
#   the norm, quantiser, gates, DeltaNet preparation and gated norm at 1, 8, 64
#   and 2,048 rows, the recurrence, the blocked DeltaNet (prefills of 2,048
#   positions in one to four runs, decode rows), the cipher and the host copies;
# - `serve_b`, `serve_b_mtp`: workload B (1,024 in, 256 out, greedy) at
#   concurrency 1, 8, 32, 64 and 128, without drafts and with three, graphs
#   on: $OUT/serve_b*.jsonl;
# - `serve_c`, `serve_c_mtp`: workload C (7-turn conversations, each turn 1,760
#   tokens past the last, 256 out, temperature 1, top-p 0.95, top-k 20) at
#   concurrency 1, 8, 32 and 64: $OUT/serve_c*.jsonl;
# - `serve_b_eager` (named only): workload B with three drafts and graphs off,
#   the graphs' gain;
# - `serve_profile` (named only): Nsight Systems over workload B at
#   concurrency 1 and 64 with three drafts: $OUT/nsys.
#
# The serving steps need the verified checkpoint in $CHECKPOINT (default
# $WORK/checkpoint: qwen_real_session.sh's `fetch`). PROMPTS, when set, names
# the reference runs' tokenised prompts (`{"b": [...], "conversations":
# [...]}`); without it the prompts are seeded random ids, on which drafts are
# rarely accepted.
#
# Expected (estimates): build 25 to 35 minutes; perf_tiny 5 to 10, perf_fast
# 10 to 20; each serving step a checkpoint load (2 to 4 minutes, graphs
# captured after it) and 5 to 20 minutes of traffic. See qwen_session_lib.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen_session_lib.sh"

CHECKPOINT="${CHECKPOINT:-$WORK/checkpoint}"
PROMPTS="${PROMPTS:-}"
export CHECKPOINT PROMPTS

perf_host() { device_tests qwen_kernels_host qwen_host; }
perf_tiny() { device_tests qwen_ops qwen_tiny; }
perf_fast() { device_tests qwen_fast; }

perf_kernels() {
    if [ -n "$DRY_RUN" ]; then
        (cd "$REPO" && x cargo run --release --locked -p eidola-engine-cuda --example qwen_kernel_bench -- "$KERNELS")
        return 0
    fi
    (cd "$REPO" && cargo run --release --locked -p eidola-engine-cuda --example qwen_kernel_bench -- \
        "$KERNELS") | tee "$OUT/qwen_kernel_bench.txt"
    grep '^{' "$OUT/qwen_kernel_bench.txt" > "$OUT/qwen_kernel_bench.jsonl" ||
        die "the bench printed no measurement"
}

# serve <name> <workload> <draft> <graphs> [concurrency]
serve() {
    [ -n "$DRY_RUN" ] || [ -f "$CHECKPOINT/config.json" ] ||
        die "no checkpoint in $CHECKPOINT (qwen_real_session.sh fetch)"
    local args=(--draft "$3" --graphs "$4" --profile fast)
    [ -z "${5:-}" ] || args+=(--concurrency "$5")
    [ -z "$PROMPTS" ] || args+=(--prompts "$PROMPTS")
    (cd "$REPO" && x cargo run --release --locked -p eidola-engine-cuda --example qwen_serve_bench -- \
        "$KERNELS" "$CHECKPOINT" "$OUT/$1.jsonl" "$2" "${args[@]}")
    [ -n "$DRY_RUN" ] || [ -s "$OUT/$1.jsonl" ] || die "$1 wrote no measurement"
}

serve_b() { serve serve_b B 0 on; }
serve_b_mtp() { serve serve_b_mtp B 3 on; }
serve_c() { serve serve_c C 0 on; }
serve_c_mtp() { serve serve_c_mtp C 3 on; }
serve_b_eager() { serve serve_b_eager B 3 off; }

serve_profile() {
    local nsys
    nsys="$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1 || true)"
    if [ -z "$nsys" ]; then
        x sudo apt-get install -y nsight-systems || die "no nsys, and nsight-systems did not install"
        nsys="$(command -v nsys || ls /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1 || true)"
        [ -n "$nsys" ] || [ -n "$DRY_RUN" ] || die "nsight-systems installed no nsys"
        nsys="${nsys:-nsys}"
    fi
    mkdir -p "$OUT/nsys"
    local bin="$CARGO_TARGET_DIR/release/examples/qwen_serve_bench"
    for c in 1 64; do
        x "$nsys" profile --trace cuda --cuda-graph-trace node -o "$OUT/nsys/serve_b_c$c" -f true \
            "$bin" "$KERNELS" "$CHECKPOINT" "$OUT/nsys/serve_b_c$c.jsonl" B \
            --draft 3 --graphs on --concurrency "$c"
    done
}

main "facts setup build kernels_build kernels perf_host perf_tiny perf_fast perf_kernels serve_b serve_b_mtp serve_c serve_c_mtp summary" \
    "serve_b_eager serve_profile" "$@"
