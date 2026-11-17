#!/usr/bin/env bash
# The Qwen3.8 real-weights session on one B200: the released checkpoint.
#
#   qwen_real_session.sh [step ...]     (no argument: every step, in order)
#   DRY_RUN=1 qwen_real_session.sh      print every command, run none
#
#   steps: facts setup build kernels_build kernels fetch golden_tokens
#          whole_compare whole_health whole_generate whole_bench
#          golden_gpu vllm_generate vllm_bench vllm_mtp_bench summary
#   named only: golden_rust whole_bench_lagging
#
# The checkpoint comes from `eidola-engine-model`'s golden/qwen/full_session.sh
# (`fetch`: every file hashed against its recorded upstream SHA-256 as it
# arrives, never resumed), into $WORK/checkpoint; that session's own markers and
# logs stay under $WORK/golden-out, on this machine, and its logs are copied
# into $OUT/golden after each step. `fetch` and `golden_tokens` are never
# marked, since what they write lives under $WORK: on a fresh machine (a
# preempted instance, $OUT copied back) they fetch, verify and write it again,
# and on this one the model crate's session skips them by its own markers.
#
# - `golden_tokens`: the health check's token sequences
#   (`$WORK/checkpoint/golden/health.safetensors`);
# - `whole_compare`: the passage through the model crate's forward on this CPU,
#   the executor's exact profile equal to it bit for bit (required), its fast
#   profile's deviation (recorded);
# - `whole_health`: licence accuracy at least 0.95 and "Paris", both profiles;
# - `whole_generate`: the fast profile's greedy tokens equal at draft widths 0
#   to 3;
# - `whole_bench`: the fast profile through the engine at widths 0 and 3: the
#   workloads of `scripts/qwen_workloads.json` at concurrency 1 to 128, and
#   prefill alone at 1k and 8k;
# - `golden_gpu`: the model crate's GPU reference steps (`released`,
#   `released-torch`, `health` with DEVICE=cuda): the reference as released on
#   this GPU, against which ours is compared;
# - `vllm_generate`, `vllm_bench`, `vllm_mtp_bench`: vLLM 0.33.0 (pinned
#   image, `scripts/qwen_vllm.py`) on the same files and GPU: its greedy
#   replies to the same prompts (beside ours), and the same workloads without
#   and with its MTP drafter (acceptance from its metrics).
#
# Expected (estimates): fetch 15 to 30 minutes, whole_compare 10 to 20 (the
# CPU forward and the exact profile's scalar GEMMs), whole_health 10 to 20,
# whole_generate 5 to 10, whole_bench 20 to 30, golden_gpu 30 to 60, vLLM 40 to
# 60 with its start-ups. See qwen_session_lib.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen_session_lib.sh"

CKPT="${CKPT:-$WORK/checkpoint}"
GOLDEN="$REPO/crates/eidola-engine-model/golden/qwen/full_session.sh"
export CKPT GOLDEN

# golden <steps...>: the model crate's session on this machine, its logs kept.
golden() {
    x env WORK="$WORK/golden-work" CKPT="$CKPT" OUT="$WORK/golden-out" DEVICE=cuda DRY_RUN="$DRY_RUN" \
        bash "$GOLDEN" "$@"
    mkdir -p "$OUT/golden"
    [ -n "$DRY_RUN" ] || cp -r "$WORK/golden-out/logs" "$OUT/golden/" 2>/dev/null || true
}

fetch() { golden setup fetch; }
golden_tokens() {
    golden tokens
    [ -n "$DRY_RUN" ] || [ -f "$CKPT/golden/health.safetensors" ] || die "no health.safetensors after tokens"
}
golden_gpu() { golden released released-torch health; }
golden_rust() { golden rust; }

whole() {
    (cd "$REPO" && x cargo run --release --locked -p eidola-engine-cuda --example qwen_whole -- \
        "$1" "$KERNELS" "$CKPT" "$OUT/whole")
}
whole_compare() { whole compare; }
whole_health() { whole health; }
whole_generate() { whole generate; }
whole_bench() { whole bench; }
whole_bench_lagging() { EIDOLA_QWEN_STRATEGY=lagging whole bench; }

vllm() { x python3 -I "$HERE/qwen_vllm.py" "$@"; }
vllm_generate() {
    vllm serve "$CKPT" "$OUT/vllm_base.log"
    vllm generate "$CKPT/golden/health.safetensors" "$OUT/vllm_generate.json"
    vllm stop
}
vllm_bench() {
    vllm serve "$CKPT" "$OUT/vllm_base.log"
    vllm bench "$OUT/vllm_bench.json"
    vllm stop
}
vllm_mtp_bench() {
    vllm serve "$CKPT" "$OUT/vllm_mtp.log" --mtp
    vllm bench "$OUT/vllm_mtp_bench.json" --mtp
    vllm stop
}

UNMARKED=" setup build kernels_build fetch golden_tokens summary "
main "facts setup build kernels_build kernels fetch golden_tokens whole_compare whole_health whole_generate whole_bench golden_gpu vllm_generate vllm_bench vllm_mtp_bench summary" \
    "golden_rust whole_bench_lagging" "$@"
