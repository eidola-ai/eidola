#!/usr/bin/env bash
# The first Qwen3.8 session on one B200: the cheapest evidence, no checkpoint.
#
#   qwen_probe_session.sh [step ...]     (no argument: every step, in order)
#   DRY_RUN=1 qwen_probe_session.sh      print every command, run none
#
#   steps: facts setup build kernels_build kernels qwen_kernels qwen_bench summary
#
# - `kernels_build` builds the kernel set on this x86_64 machine (Docker, the
#   pinned Nix) and requires the committed manifest byte for byte;
# - `qwen_kernels`: every Qwen kernel against the model crate bit for bit on the
#   device, and the fast profile's linears (deviation recorded, row invariance
#   asserted) (`tests/qwen_ops.rs`);
# - `qwen_bench`: copy bandwidth, the FP8 linears at the released shapes (fast
#   and exact), the head, the DeltaNet recurrence, the cipher on a 157 MB image
#   and the host copies (`examples/qwen_kernel_bench.rs`), as JSON lines in
#   $OUT/qwen_bench.jsonl. This decides split-K and checks the bandwidth and
#   GEMM-rate assumptions.
#
# Expected on a fresh 20-vCPU box: setup and build 25 to 35 minutes,
# kernels_build 10 to 20 (first toolkit download), kernels and qwen_kernels a
# few minutes, qwen_bench 5 to 10. See qwen_session_lib.sh for markers,
# resumption, DRY_RUN and the environment.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen_session_lib.sh"

qwen_kernels() { device_tests qwen_ops; }

qwen_bench() {
    if [ -n "$DRY_RUN" ]; then
        (cd "$REPO" && x cargo run --release --locked -p eidola-engine-cuda --example qwen_kernel_bench -- "$KERNELS")
        return 0
    fi
    (cd "$REPO" && cargo run --release --locked -p eidola-engine-cuda --example qwen_kernel_bench -- \
        "$KERNELS") | tee "$OUT/qwen_bench.txt"
    grep '^{' "$OUT/qwen_bench.txt" > "$OUT/qwen_bench.jsonl" || die "the bench printed no measurement"
}

main "facts setup build kernels_build kernels qwen_kernels qwen_bench summary" "" "$@"
