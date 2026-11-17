#!/usr/bin/env bash
# The Qwen3.8 session scripts' failure paths, without a GPU: every way a step can
# fail must fail the session and leave no marker, and a dry run must run and mark
# nothing. Run anywhere: `bash crates/eidola-engine-cuda/scripts/qwen_session_test.sh`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
fails=0
check() { # check <description> <condition...>
    local what="$1"; shift
    if "$@"; then echo "ok      $what"; else echo "FAILED  $what"; fails=$((fails + 1)); fi
}

# A session of stub steps over the real library.
cat > "$T/session.sh" <<STUB
source "$HERE/qwen_session_lib.sh"
good() { echo fine; }
bad() { echo about to fail; false; }
late() { echo late; }
skipper() { echo "SKIPPED some_test: no CUDA device"; }
pipefail() { false | cat; }
printer() { x touch "\$WORK/printed-effect"; }
build() { echo building; }
main "good bad late skipper pipefail printer build summary" "" "\$@"
STUB

run() { WORK="$T/work-$1" bash "$T/session.sh" "${@:2}" > "$T/log-$1" 2>&1; }

! run a good bad late
rc_bad=$?
check "a failing step fails the session" [ "$rc_bad" -eq 0 ]
check "a failing step leaves no marker" [ ! -e "$T/work-a/out/done/bad" ]
check "the step before it is marked" [ -e "$T/work-a/out/done/good" ]
check "the steps after it do not run" [ ! -e "$T/work-a/out/done/late" ]
check "the failure is named" grep -q "FAILED: bad exited" "$T/log-a"

run a good late
check "a rerun skips a finished step" grep -q "good: already done" "$T/log-a"

! run b skipper
check "a skipped device test fails its step" grep -q "skipped a device test" "$T/log-b"
check "and leaves no marker" [ ! -e "$T/work-b/out/done/skipper" ]

! run c pipefail
check "a failure inside a pipe fails the step" [ ! -e "$T/work-c/out/done/pipefail" ]

DRY_RUN=1 run d good printer build
check "a dry run prints commands" grep -q "would run: touch" "$T/log-d"
check "a dry run runs nothing" [ ! -e "$T/work-d/printed-effect" ]
check "a dry run marks nothing" [ -z "$(ls "$T/work-d/out/done")" ]

run e build
check "an unmarked step (build) is never marked" [ ! -e "$T/work-e/out/done/build" ]

! run f nonesuch
check "an unknown step fails" grep -q "unknown step nonesuch" "$T/log-f"

# The vLLM helper fails loudly with nothing serving and on a missing file.
! python3 -I "$HERE/qwen_vllm.py" bench "$T/x.json" > "$T/log-g" 2>&1
check "vLLM bench with no server fails" [ ! -e "$T/x.json" ]
! python3 -I "$HERE/qwen_vllm.py" generate "$T/none.safetensors" "$T/y.json" > "$T/log-h" 2>&1
check "vLLM generate without its token file fails" [ ! -e "$T/y.json" ]
! python3 -I "$HERE/qwen_vllm.py" > "$T/log-i" 2>&1
check "vLLM helper without a command fails" grep -q "FAILED" "$T/log-i"

# Every session script dry-runs to its end, marking nothing.
for s in qwen_probe_session.sh qwen_gpu_session.sh qwen_real_session.sh; do
    WORK="$T/dry-$s" DRY_RUN=1 bash "$HERE/$s" > "$T/log-$s" 2>&1
    check "$s dry-runs" grep -q "results in" "$T/log-$s"
    check "$s dry run marks nothing" [ -z "$(ls "$T/dry-$s/out/done")" ]
done

[ "$fails" -eq 0 ] || { echo "$fails check(s) failed"; exit 1; }
echo "all checks passed"
