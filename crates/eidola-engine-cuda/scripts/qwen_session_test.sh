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

# The vLLM helper reads token tensors as the health check writes them (I32), and I64.
python3 -I - "$HERE/qwen_vllm.py" "$T" > "$T/log-ids" 2>&1 <<'PY' || true
import importlib.util, json, struct, sys
spec = importlib.util.spec_from_file_location("qwen_vllm", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
ids = [3, 70000, 248044]
for dtype, fmt in (("I32", "i"), ("I64", "q")):
    data = struct.pack(f"<{len(ids)}{fmt}", *ids)
    meta = {"capital.prompt": {"dtype": dtype, "shape": [len(ids)], "data_offsets": [0, len(data)]}}
    header = json.dumps(meta).encode()
    path = f"{sys.argv[2]}/ids-{dtype}.safetensors"
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)) + header + data)
    try:
        got = m.safetensors_ids(path, "capital.prompt")
    except SystemExit:
        continue
    print("read" if got == ids else "misread", dtype)
PY
check "vLLM helper reads I32 token ids" grep -q "read I32" "$T/log-ids"
check "vLLM helper reads I64 token ids" grep -q "read I64" "$T/log-ids"

# The throughput benches run one workload definition: qwen_whole's reads the shared
# file, which holds the yardsticks, and the vLLM bench runs what the file says
# (against a stand-in server, on a small copy of it): its request counts, prompt
# lengths, conversation turns and sampling.
check "qwen_whole's bench reads the shared workloads" \
    grep -q 'include_str!("../scripts/qwen_workloads.json")' "$HERE/../examples/qwen_whole.rs"
python3 -I - "$HERE/qwen_workloads.json" > "$T/log-yardsticks" 2>&1 <<'PY' || true
import json, sys
w = json.load(open(sys.argv[1]))
lanes = [1, 8, 32, 64, 128]
b = {"prompt_tokens": 1024, "output_tokens": 256, "temperature": 0.0, "concurrency": lanes,
     "requests_per_concurrent": 2, "requests_at_least": 4}
c = {"turns": 7, "turn_tokens": 1760, "output_tokens": 256, "temperature": 1.0, "top_p": 0.95,
     "top_k": 20, "concurrency": lanes, "conversations_per_concurrent": 1, "conversations_at_least": 2}
print("yardsticks" if w == {"B": b, "C": c} else f"other workloads: {w}")
PY
check "the shared workloads are the yardsticks B and C" grep -qx "yardsticks" "$T/log-yardsticks"
python3 -I - "$HERE/qwen_vllm.py" "$T" > "$T/log-bench" 2>&1 <<'PY' || true
import importlib.util, json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
spec = importlib.util.spec_from_file_location("qwen_vllm", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
seen, lock = [], threading.Lock()

class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b"vllm:prefix_cache_queries_total 0\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with lock:
            seen.append(req)
        self.send_response(200)
        self.end_headers()
        for i in range(req["max_tokens"]):
            chunk = {"choices": [{"text": "x", "token_ids": [9000 + i]}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=server.serve_forever, daemon=True).start()
m.BASE = f"http://127.0.0.1:{server.server_port}"
m.WORKLOADS = {
    "B": {"prompt_tokens": 5, "output_tokens": 3, "temperature": 0.0, "concurrency": [1, 3],
          "requests_per_concurrent": 2, "requests_at_least": 4},
    "C": {"turns": 3, "turn_tokens": 6, "output_tokens": 2, "temperature": 1.0, "top_p": 0.95,
          "top_k": 20, "concurrency": [1, 2], "conversations_per_concurrent": 1,
          "conversations_at_least": 2},
}
m.cmd_bench(f"{sys.argv[2]}/bench.json", False)
rows = json.load(open(f"{sys.argv[2]}/bench.json"))
b = [r for r in seen if r["temperature"] == 0.0]
c = [r for r in seen if r["temperature"] == 1.0]
prompts = [r["prompt"] for r in c]
checks = {
    "rows": [(r["workload"], r["concurrency"], r["requests"]) for r in rows]
    == [("B", 1, 4), ("B", 3, 6), ("C", 1, 6), ("C", 2, 6)],
    "B requests": len(b) == 10 and all(len(r["prompt"]) == 5 and r["max_tokens"] == 3 for r in b),
    "C requests": len(c) == 12 and all(r["max_tokens"] == 2 and r["top_p"] == 0.95
                                       and r["top_k"] == 20 and "seed" in r for r in c),
    "C turns": sorted(len(p) for p in prompts) == [6] * 4 + [12] * 4 + [18] * 4,
    "C extends": all(p[-6:-4] == [9000, 9001] and p[:-6] in prompts for p in prompts if len(p) > 6),
    "ignore eos": all(r["ignore_eos"] for r in seen),
}
for name, ok in checks.items():
    print("ok" if ok else "FAILED", name)
PY
check "the vLLM bench runs the shared workloads" \
    [ "$(grep -c '^ok' "$T/log-bench")" -eq 6 ]

# The real-weights session resumed on a fresh machine ($OUT copied back, $WORK
# empty) writes the token sequences again, since they live under $WORK. The
# model crate's session is a stub keeping its own markers, as the real one does.
R="$T/repo"
mkdir -p "$R/crates/eidola-engine-cuda/scripts" "$R/crates/eidola-engine-model/golden/qwen"
cp "$HERE/qwen_session_lib.sh" "$HERE/qwen_real_session.sh" "$R/crates/eidola-engine-cuda/scripts/"
cp "$HERE/../../../rust-toolchain.toml" "$R/"
cat > "$R/crates/eidola-engine-model/golden/qwen/full_session.sh" <<'STUB'
mkdir -p "$OUT/done"
for step in "$@"; do
    [ -e "$OUT/done/$step" ] && continue
    echo "golden $step" >> "$GOLDEN_CALLS"
    if [ "$step" = tokens ]; then
        mkdir -p "$CKPT/golden"
        echo ids > "$CKPT/golden/health.safetensors"
    fi
    touch "$OUT/done/$step"
done
STUB
resume() {
    WORK="$T/$1" OUT="$T/resume-out" GOLDEN_CALLS="$T/calls-$1" \
        bash "$R/crates/eidola-engine-cuda/scripts/qwen_real_session.sh" golden_tokens \
        > "$T/log-resume-$1" 2>&1 || true
}
resume first
resume first
check "golden_tokens is never marked" [ ! -e "$T/resume-out/done/golden_tokens" ]
check "a rerun on its machine leaves it to the model crate's markers" \
    [ "$(grep -c "golden tokens" "$T/calls-first")" -eq 1 ]
touch "$T/resume-out/done/golden_tokens" # as an earlier session may have left
resume fresh
check "a fresh machine writes the token sequences again" \
    [ -f "$T/fresh/checkpoint/golden/health.safetensors" ]

# Every session script dry-runs to its end, marking nothing.
for s in qwen_probe_session.sh qwen_gpu_session.sh qwen_real_session.sh; do
    WORK="$T/dry-$s" DRY_RUN=1 bash "$HERE/$s" > "$T/log-$s" 2>&1
    check "$s dry-runs" grep -q "results in" "$T/log-$s"
    check "$s dry run marks nothing" [ -z "$(ls "$T/dry-$s/out/done")" ]
done

[ "$fails" -eq 0 ] || { echo "$fails check(s) failed"; exit 1; }
echo "all checks passed"
