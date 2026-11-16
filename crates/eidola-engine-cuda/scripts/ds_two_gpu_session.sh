#!/usr/bin/env bash
# One session on a machine with two B300s for DeepSeek-V4.1 on two GPUs: our executor
# with each half on its own GPU, the node serving the released checkpoint over its
# HTTP API, and vLLM on the same checkpoint and prompts as a yardstick.
#
#   ds_two_gpu_session.sh [step ...]        (no argument: every default step, in order)
#
# Steps in their default order, the cheapest evidence first (minutes on two B300s,
# after the first two; estimates unless marked measured):
#
#   facts         the GPUs, driver, topology, CPUs, memory, disk, memory-lock limit,
#                 image and vLLM version (facts.json)                               1
#   fetch_start   starts the checkpoint's download in the background (it runs while
#                 the steps up to fetch_wait do): every file the golden tooling
#                 records, from the pinned revision, 8 files at a time, each a fresh
#                 download hashed as it arrives and named only when it is the
#                 recorded bytes (ds_two_gpu.py fetch; fetch.json logs the rate)      0
#   setup         a C toolchain and rustup with the repository's toolchain            4
#   build         every binary and test the steps run, release                     10-15
#   kernels       the kernel manifest test; every image loads (tests/smoke.rs)        1
#   dev_two_gpu   eidola-server-engine's ignored test: the node on the DeepSeek dev
#                 model with each half on its own GPU answers as the CPU executor,
#                 draft widths 0 and 2                                              2-3
#   tiny_two_gpu  the executor's tiny scenarios (plain, then every draft width) with
#                 the encoder on GPU 0 and the decoder on GPU 1, banded GEMMs, held
#                 to the CPU reference executor step for step (tiny_two_gpu/)       4-8
#   tiny_two_gpu_memory   the memory scenarios the same way (idle encryption, the
#                 page-locked host tier); not run, and says why, when the memory-lock
#                 hard limit is under 64 MiB                                        1-3
#   fetch_wait    waits for the download (about 510 GB: 10-20 at 1 GB/s)        0-15
#   verify        every file read back and hashed against its record, in parallel;
#                 the node's weights hash from the same digests (verify.json,
#                 weights.sha256)                                                   3-5
#   sequences     the health texts from the golden tooling, then the sequences every
#                 later step runs, tokenized by our tokenizer (ds_two/sequences.json)
#                 and the passage golden ds_halves reads                              1
#   halves        the one-half-at-a-time measurement on GPU 0, exact GEMMs, over the
#                 passage (555 tokens, steps of 96), with its invariance pass: the
#                 recordings the two-GPU run is held to (ds_halves/). Skipped when
#                 RECORDED names an earlier recording                       6 (measured
#                                                         loads 148 s and 106 s)
#   two_resident  both halves resident, encoder on GPU 0 and decoder (with its Engram
#                 table) on GPU 1, banded GEMMs: the recorded text compared with the
#                 halves' recordings bit for bit (44 arrays), then the health sequences
#                 (hello, story, passage) scored (ds_two/compare.json, score.json)    5-7
#   two_serving   the executor itself on both GPUs under the engine, banded GEMMs:
#                 ds_generate's prompts at draft widths 0 and 5, the same tokens at
#                 both and, with CPU_RECORDED, the CPU reference executor's; then the
#                 throughput prompts (1,024 in, 256 out) at widths 0 and 5 and
#                 concurrencies 1, 8 and 32 (ds_two/generate.json, bench.json)     10-15
#   node_w0, node_w3  the node binary on the checkpoint, EIDOLA_ENGINE_EXECUTOR=cuda,
#                 a 131,072-position context, the measured GEMM form (exact), draft
#                 width 0 then 3 (the most the node's grammar takes with cuda): the
#                 health requests and the generation prompts over its HTTP API, each
#                 against the CPU reference executor's answer (node_w*.json)    8-10 each
#   vllm_setup    vLLM present (installed with pip when not), its version; a model
#                 directory of links to the verified files and tokenizer_config.json   1-5
#   vllm_base     vLLM, tensor parallel 2, Engram tables in pinned host memory,
#                 bounded replay off, prefix caching off, no drafting: the health
#                 sequences scored from prompt log-probabilities over our token ids,
#                 the hello reply and the generation prompts greedily, and the same
#                 throughput runs (vllm_base/)                                     20-35
#   vllm_spec     the same with DSpark drafting 5 tokens (greedy drafts, standard
#                 acceptance, no adaptive verification) (vllm_spec/)               15-25
#   summary       summary.json and summary.txt from everything above (always rewritten)
#
#   named only:   halves_reference   the model crate's forward on the CPU over the
#                 halves' recordings (ds_halves' reference phase; needs no GPU)     11
#
# Whole session: about 2 to 3 hours, most of it the download and vLLM's start-ups. A
# failure in a step of ours stops the session (its log says why); vLLM's steps record
# what failed (server.json: each launch tried, the command, the end of its log) and
# go on, so a stack that cannot run the model on two B300s still leaves its reason.
#
# Run from a checkout (or an unpacked `git archive`) of this repository on Linux
# x86_64 with two Blackwell GPUs and their driver, as root or with sudo (apt). The
# kernel images are the reproducible build's output, copied here (KERNELS, as for
# ds_gpu_session.sh). Nothing needs a secret: the checkpoint is public and the node's
# gateway token is made here and thrown away.
#
# Every step is idempotent and resumable: a finished step leaves a marker under
# $OUT/done/ and is skipped on a rerun (delete the marker to repeat it). Everything a
# step prints is also in $OUT/logs/<step>.log. DRY_RUN=1 prints every command a step
# would run instead of running it, and marks nothing. Copy $OUT back.
#
# Environment (all optional):
#   WORK          scratch                                      (default ~/eidola-gpu)
#   OUT           results                                      (default $WORK/out)
#   CHECKPOINT    where the checkpoint is fetched to           (default $WORK/checkpoint)
#   KERNELS       the kernel build output                      (as ds_gpu_session.sh)
#   RECORDED      an earlier ds_halves recording of the passage, instead of `halves`
#   CPU_RECORDED  a ds_generate output on the CPU (its run/ directory, with requests/)
#   FETCH_WORKERS files fetched at a time                      (default 8)
#   NODE_WIDTHS   the node's draft widths                      (default "0 3")
#   VLLM_PY       the Python vLLM runs in                      (default python3)
#   VLLM_VERSION  installed with pip when vLLM is missing      (default 0.33.0)
#   VLLM_READY_MINUTES  how long a vLLM start-up may take      (default 60)
#   IMAGE         the container image, recorded in facts.json
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
WORK="${WORK:-$HOME/eidola-gpu}"
OUT="${OUT:-$WORK/out}"
CHECKPOINT="${CHECKPOINT:-$WORK/checkpoint}"
export PATH="$HOME/.cargo/bin:$PATH"
RUSTUP_TOOLCHAIN="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$REPO/rust-toolchain.toml")"
export RUSTUP_TOOLCHAIN
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$WORK/target}"

if [ -z "${KERNELS:-}" ]; then
    for dir in "$REPO/target/engine-kernels/amd64" "$REPO/target/engine-kernels/arm64" "$WORK/kernels"; do
        if [ -d "$dir/cubin" ]; then KERNELS="$dir"; break; fi
    done
fi
export HERE REPO WORK OUT CHECKPOINT KERNELS="${KERNELS:-}"
export EIDOLA_ENGINE_KERNELS_DIR="$KERNELS"
export RECORDED="${RECORDED:-}" CPU_RECORDED="${CPU_RECORDED:-}"
export FETCH_WORKERS="${FETCH_WORKERS:-8}" NODE_WIDTHS="${NODE_WIDTHS:-0 3}"
export PY="${PY:-python3}" VLLM_PY="${VLLM_PY:-python3}" VLLM_VERSION="${VLLM_VERSION:-0.33.0}"
export VLLM_READY_MINUTES="${VLLM_READY_MINUTES:-60}" IMAGE="${IMAGE:-}"
export TOOL="$HERE/ds_two_gpu.py"
export VLLM_DIR="$WORK/vllm-model" EXTRAS="$WORK/extras"
export DRY_RUN="${DRY_RUN:-}"

mkdir -p "$WORK" "$OUT/done" "$OUT/logs"

say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { say "FAILED: $*" >&2; exit 1; }

# A command, or with DRY_RUN set the command printed.
x() {
    if [ -n "$DRY_RUN" ]; then
        printf 'would run:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# run_step <name>: skip when done, log, mark, fail loudly.
run_step() {
    local name="$1"
    if [ -e "$OUT/done/$name" ]; then say "$name: already done"; return 0; fi
    say "$name: start"
    local began=$SECONDS
    set +e
    bash -euo pipefail -c "$(declare -f); $name" 2>&1 | tee "$OUT/logs/$name.log"
    local status=${PIPESTATUS[0]}
    set -e
    local seconds=$((SECONDS - began))
    [ -n "$DRY_RUN" ] || printf '%s %s %s\n' "$name" "$status" "$seconds" >> "$OUT/steps.txt"
    [ "$status" -eq 0 ] || die "$name exited with $status (see $OUT/logs/$name.log)"
    if grep -q 'skipping:' "$OUT/logs/$name.log"; then
        die "$name skipped a device test (see $OUT/logs/$name.log)"
    fi
    [ -n "$DRY_RUN" ] || touch "$OUT/done/$name"
    say "$name: done in $seconds s"
}

tool() { x "$PY" -I "$TOOL" "$@"; }
cargo_r() { (cd "$REPO" && x cargo "$@"); }
need_kernels() {
    [ -n "$DRY_RUN" ] || { [ -n "$KERNELS" ] && [ -d "$KERNELS/cubin" ]; } ||
        die "no kernel build output: copy target/engine-kernels/<arch> here and set KERNELS"
}
need_checkpoint() {
    [ -n "$DRY_RUN" ] || [ -e "$OUT/done/verify" ] || die "the checkpoint is not verified: run fetch_wait and verify"
}
weights() { if [ -n "$DRY_RUN" ]; then echo "<weights hash>"; else cat "$OUT/weights.sha256"; fi; }

# Waits until no process holds either GPU (a server just stopped frees them late).
gpus_free() {
    [ -n "$DRY_RUN" ] && return 0
    local i
    for i in $(seq 1 60); do
        [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0
        sleep 2
    done
    nvidia-smi
    die "the GPUs are still in use"
}

# stop_group <pid file>: the process group it names, TERM then KILL.
stop_group() {
    [ -f "$1" ] || return 0
    local pid; pid="$(cat "$1")"
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        local i
        for i in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$pid" 2>/dev/null || true
    fi
    rm -f "$1"
}

facts() {
    if [ -n "$DRY_RUN" ]; then say "would record nvidia-smi, the topology, CPUs, memory, disk and versions"; return 0; fi
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found: no NVIDIA driver"
    nvidia-smi
    nvidia-smi topo -m || true
    local gpus driver vllm_version
    gpus="$(nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader | tr '\n' ';')"
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    vllm_version="$("$VLLM_PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo none)"
    [ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -ge 2 ] || die "two GPUs are needed"
    nproc; free -g; df -h "$WORK"; ulimit -H -l; cat /etc/os-release | head -3
    "$PY" -I -c 'import json, sys; json.dump(dict(zip(["gpus", "driver", "image", "vllm_version", "cpus", "memlock_hard", "kernel", "python"], sys.argv[1:])), open(sys.argv[-1], "w"), indent=1)' \
        "$gpus" "$driver" "$IMAGE" "$vllm_version" "$(nproc)" "$(ulimit -H -l)" "$(uname -r)" "$("$PY" --version 2>&1)" \
        "$OUT/facts.json"
    cat "$OUT/facts.json"
}

fetch_start() {
    if [ -e "$OUT/done/fetch" ]; then say "fetched already"; return 0; fi
    if [ -f "$OUT/fetch.pid" ] && kill -0 "$(cat "$OUT/fetch.pid")" 2>/dev/null; then
        say "the fetch is running (pid $(cat "$OUT/fetch.pid"))"; return 0
    fi
    df -h "$(dirname "$CHECKPOINT")"
    if [ -n "$DRY_RUN" ]; then
        tool fetch --dir "$CHECKPOINT" --workers "$FETCH_WORKERS" --extras-dir "$EXTRAS" --out "$OUT/fetch.json"
        return 0
    fi
    # Its own session and process group, writing its marker when it succeeds.
    setsid bash -c '"$0" -I "$1" fetch --dir "$2" --workers "$3" --extras-dir "$4" --out "$5/fetch.json" && touch "$5/done/fetch"' \
        "$PY" "$TOOL" "$CHECKPOINT" "$FETCH_WORKERS" "$EXTRAS" "$OUT" > "$OUT/logs/fetch.log" 2>&1 < /dev/null &
    echo $! > "$OUT/fetch.pid"
    say "fetching in the background (pid $!, log $OUT/logs/fetch.log)"
}

setup() {
    if ! command -v cc >/dev/null || ! command -v c++ >/dev/null || ! command -v curl >/dev/null; then
        say "installing a C toolchain and curl"
        local sudo=(); [ "$(id -u)" -eq 0 ] || sudo=(sudo)
        x ${sudo[@]+"${sudo[@]}"} apt-get update -y
        x ${sudo[@]+"${sudo[@]}"} env DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential curl ca-certificates
    fi
    if ! command -v rustup >/dev/null; then
        say "installing rustup"
        if [ -n "$DRY_RUN" ]; then
            echo "would run: curl https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none"
        else
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
        fi
    fi
    x rustup toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal
    x cargo --version
}

build() {
    cargo_r build --release --locked -p eidola-engine-cuda --example ds_two --example ds_halves
    cargo_r test --release --locked -p eidola-engine-cuda -p eidola-engine-kernels \
        --test ds_tiny --test smoke --test manifest --no-run
    cargo_r build --release --locked -p eidola-server-engine --features cuda --bin eidola-server-engine
    cargo_r test --release --locked -p eidola-server-engine --features cuda --test cuda --no-run
    cargo_r build --release --locked -p hash-secret
}

kernels() {
    need_kernels
    say "kernels: $KERNELS"
    cargo_r test --release --locked -p eidola-engine-kernels --test manifest
    cargo_r test --release --locked -p eidola-engine-cuda --test smoke -- --nocapture --test-threads 1
}

dev_two_gpu() {
    need_kernels
    cargo_r test --release --locked -p eidola-server-engine --features cuda --test cuda \
        deepseek_dev_model_on_two_gpus -- --ignored --nocapture --test-threads 1
}

# tiny_test <test name>: one of ds_tiny's tests with the halves on GPUs 0 and 1.
tiny_test() {
    need_kernels
    mkdir -p "$OUT/tiny_two_gpu"
    (cd "$REPO" && x env EIDOLA_DS_TINY_DEVICES=0,1 EIDOLA_DS_GEMM_FORM=banded \
        EIDOLA_DS_SUMMARY_DIR="$OUT/tiny_two_gpu" EIDOLA_DS_TINY_SUMMARY="$OUT/tiny_two_gpu/ds_tiny.json" \
        cargo test --release --locked -p eidola-engine-cuda --test ds_tiny -- \
        --nocapture --test-threads 1 --exact "$1")
}
tiny_two_gpu() {
    tiny_test the_tiny_model_equals_the_cpu_reference_executor
    tiny_test the_tiny_model_drafts_as_the_cpu_reference_executor
}
tiny_two_gpu_memory() {
    local limit; limit="$(ulimit -H -l)"
    if [ "$limit" != unlimited ] && [ "$limit" -lt 65536 ]; then
        say "not run: the memory-lock hard limit is $limit KiB (the reference executor's host tier locks a few MiB; start the container with --ulimit memlock=-1)"
        echo "not run: memlock $limit KiB" > "$OUT/tiny_two_gpu/memory.txt"
        return 0
    fi
    tiny_test the_tiny_model_keeps_its_memory_as_the_cpu_reference_executor
}

fetch_wait() {
    [ -n "$DRY_RUN" ] && { say "would wait for the fetch"; return 0; }
    while [ ! -e "$OUT/done/fetch" ]; do
        if ! { [ -f "$OUT/fetch.pid" ] && kill -0 "$(cat "$OUT/fetch.pid")" 2>/dev/null; }; then
            tail -20 "$OUT/logs/fetch.log" || true
            die "the fetch is not running and did not finish: rerun fetch_start"
        fi
        tail -1 "$OUT/logs/fetch.log" || true
        sleep 30
    done
    say "fetched: $(head -c 400 "$OUT/fetch.json")"
}

verify() {
    tool verify --dir "$CHECKPOINT" --workers 32 --out "$OUT/verify.json"
}

sequences() {
    need_checkpoint
    tool texts --out "$OUT/texts.json"
    (cd "$REPO" && x env EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_TWO_OUT="$OUT/ds_two" \
        EIDOLA_DS_TWO_TEXTS="$OUT/texts.json" EIDOLA_DS_TWO_PHASES=tokens \
        cargo run --release --locked -p eidola-engine-cuda --example ds_two)
}

# The recording the two-GPU run is held to: RECORDED, else this session's.
recorded() { if [ -n "$RECORDED" ]; then echo "$RECORDED"; else echo "$OUT/ds_halves"; fi; }

halves() {
    need_kernels; need_checkpoint
    if [ -n "$RECORDED" ]; then say "using the recording in $RECORDED"; return 0; fi
    gpus_free
    (cd "$REPO" && x env EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_HALVES_OUT="$OUT/ds_halves" \
        EIDOLA_DS_HALVES_GOLDEN="$OUT/ds_two/passage.safetensors" EIDOLA_DS_HALVES_PHASES=encoder,decoder \
        EIDOLA_DS_HALVES_TOKENS=555 EIDOLA_DS_HALVES_CHUNK=96 EIDOLA_DS_GEMM_FORM=exact \
        cargo run --release --locked -p eidola-engine-cuda --example ds_halves)
}
halves_reference() {
    need_checkpoint
    (cd "$REPO" && x env EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_HALVES_OUT="$(recorded)" \
        EIDOLA_DS_HALVES_GOLDEN="$OUT/ds_two/passage.safetensors" EIDOLA_DS_HALVES_PHASES=reference \
        EIDOLA_DS_HALVES_TOKENS=555 EIDOLA_DS_HALVES_CHUNK=96 \
        cargo run --release --locked -p eidola-engine-cuda --example ds_halves)
}

# ds_two <phases> [VAR=value ...]: the two-GPU example.
ds_two() {
    local phases="$1"; shift
    need_kernels; need_checkpoint; gpus_free
    (cd "$REPO" && x env EIDOLA_DEEPSEEK_FULL_DIR="$CHECKPOINT" EIDOLA_DS_TWO_OUT="$OUT/ds_two" \
        EIDOLA_DS_TWO_PHASES="$phases" EIDOLA_DS_GEMM_FORM=banded EIDOLA_DS_TWO_DEVICES=0,1 \
        EIDOLA_DS_HALVES_GOLDEN="$OUT/ds_two/passage.safetensors" "$@" \
        cargo run --release --locked -p eidola-engine-cuda --example ds_two)
}
two_resident() { ds_two compare,score EIDOLA_DS_TWO_RECORDED="$(recorded)"; }
two_serving() {
    local cpu=()
    [ -n "$CPU_RECORDED" ] && cpu=(EIDOLA_DS_TWO_CPU="$CPU_RECORDED")
    ds_two generate,bench ${cpu[@]+"${cpu[@]}"}
}

# node <draft width>: the node on the checkpoint, then the client.
node() {
    local width="$1" name="node_w$1"
    need_kernels; need_checkpoint; gpus_free
    local bin="$CARGO_TARGET_DIR/release/eidola-server-engine" token hash
    token="$("$PY" -I -c 'import secrets; print(secrets.token_urlsafe(32))')"
    if [ -n "$DRY_RUN" ]; then hash="<argon2id hash>"; else
        hash="$("$CARGO_TARGET_DIR/release/hash-secret" "$token")"
    fi
    local env=(
        EIDOLA_ENGINE_MODEL_ID=ds EIDOLA_ENGINE_WEIGHTS_DIR="$CHECKPOINT"
        EIDOLA_ENGINE_WEIGHTS_SHA256="$(weights)" EIDOLA_ENGINE_WEIGHTS_STORAGE=dev-writable
        GATEWAY_TOKEN="$token" GATEWAY_TOKEN_HASH="$hash"
        EIDOLA_ENGINE_EXECUTOR=cuda EIDOLA_ENGINE_BIND_ADDR=127.0.0.1:8090
        EIDOLA_ENGINE_KV_BLOCK_SIZE=256 EIDOLA_ENGINE_KV_DEVICE_BYTES="$((8 << 30)),$((8 << 30))"
        EIDOLA_ENGINE_KERNELS_DIR="$KERNELS" EIDOLA_ENGINE_CUDA_GRAPHS=off EIDOLA_ENGINE_GPU_CC=not-required
        EIDOLA_ENGINE_KV_HOST_BYTES=0 EIDOLA_ENGINE_MAX_MODEL_LEN=131072
        EIDOLA_ENGINE_MAX_SEQS=4 EIDOLA_ENGINE_MAX_BATCHED_TOKENS=2048 EIDOLA_ENGINE_MAX_PREFILL_CHUNK=2048
        EIDOLA_ENGINE_DRAFT_TOKENS="$width" EIDOLA_ENGINE_MAX_REQUESTS=8 EIDOLA_ENGINE_PREFIX_CACHE=true
        EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS=600 EIDOLA_ENGINE_CACHE_MAX_AGE_SECS=3600
    )
    # The driver's libraries, where an image keeps them off the loader's default path.
    [ -n "${LD_LIBRARY_PATH:-}" ] && env+=(LD_LIBRARY_PATH="$LD_LIBRARY_PATH")
    local cpu=()
    [ -n "$CPU_RECORDED" ] && cpu=(--cpu "$CPU_RECORDED")
    if [ -n "$DRY_RUN" ]; then
        x env -i PATH="$PATH" "${env[@]}" "$bin"
    else
        # Exactly the measured environment and the token, in a process group of its own.
        setsid env -i PATH="$PATH" "${env[@]}" "$bin" > "$OUT/logs/$name.server.log" 2>&1 < /dev/null &
        echo $! > "$OUT/$name.pid"
        trap 'stop_group "$OUT/'"$name"'.pid"' EXIT
        say "node at width $width: pid $!, log $OUT/logs/$name.server.log"
    fi
    local pid=0
    [ -n "$DRY_RUN" ] || pid="$(cat "$OUT/$name.pid")"
    tool node --base http://127.0.0.1:8090 --token "$token" --weights "$(weights)" --model ds --pid "$pid" \
        --texts "$OUT/texts.json" ${cpu[@]+"${cpu[@]}"} --out "$OUT/$name.json"
    [ -n "$DRY_RUN" ] || tail -5 "$OUT/logs/$name.server.log"
}
node_w0() { node 0; }
node_w3() { node 3; }

vllm_setup() {
    if ! "$VLLM_PY" -c 'import vllm' 2>/dev/null; then
        say "installing vllm==$VLLM_VERSION"
        x "$VLLM_PY" -m pip install "vllm==$VLLM_VERSION"
    fi
    [ -n "$DRY_RUN" ] || "$VLLM_PY" -c 'import vllm; print("vLLM", vllm.__version__)' | tee "$OUT/vllm_version.txt"
    need_checkpoint
    # The verified files as links, and the tokenizer configuration vLLM's tokenizer reads
    # (fetched beside the checkpoint so the weights hash never covers it).
    x mkdir -p "$VLLM_DIR"
    local f
    for f in "$CHECKPOINT"/*; do x ln -sfn "$f" "$VLLM_DIR/$(basename "$f")"; done
    x ln -sfn "$EXTRAS/tokenizer_config.json" "$VLLM_DIR/tokenizer_config.json"
    [ -n "$DRY_RUN" ] || ls -l "$VLLM_DIR" | head
}

# vllm_serve <dir> <attempt> <extra args...>: one start-up, waited for; 0 when serving.
vllm_serve() {
    local dir="$1" attempt="$2"; shift 2
    local args=(
        serve "$VLLM_DIR" --served-model-name ds --host 127.0.0.1 --port 8000
        --tensor-parallel-size 2 --tokenizer-mode deepseek_v41
        --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'
        --kv-cache-dtype fp8 --engram-config '{"cpu_offload":true}'
        --no-swa-bounded-replay --no-enable-prefix-caching
        --max-model-len 8192 --max-num-seqs 64 --max-num-batched-tokens 8192
        "$@"
    )
    # vLLM's CLI from the interpreter it was installed and checked in.
    local vllm=("$VLLM_PY" -m vllm.entrypoints.cli.main)
    printf '%q ' "${vllm[@]}" "${args[@]}" > "$dir/command.$attempt.txt"
    if [ -n "$DRY_RUN" ]; then x env VLLM_ENGINE_READY_TIMEOUT_S=3600 "${vllm[@]}" "${args[@]}"; return 0; fi
    setsid env VLLM_ENGINE_READY_TIMEOUT_S=3600 "${vllm[@]}" "${args[@]}" > "$dir/server.$attempt.log" 2>&1 < /dev/null &
    echo $! > "$dir/server.pid"
    local began=$SECONDS
    while [ $((SECONDS - began)) -lt $((VLLM_READY_MINUTES * 60)) ]; do
        if ! kill -0 "$(cat "$dir/server.pid")" 2>/dev/null; then
            say "vLLM exited during start-up ($attempt)"; tail -40 "$dir/server.$attempt.log"; return 1
        fi
        if curl -fs http://127.0.0.1:8000/health >/dev/null 2>&1; then
            echo $((SECONDS - began)) > "$dir/ready_seconds.$attempt"
            say "vLLM serving after $((SECONDS - began)) s ($attempt)"; return 0
        fi
        sleep 10
    done
    say "vLLM did not start within $VLLM_READY_MINUTES minutes ($attempt)"
    stop_group "$dir/server.pid"
    return 1
}

# vllm_run <name> <capture sizes> [speculative config]: start-up (our capture sizes,
# then the recipe's), quality, throughput, stop; what failed is recorded, not fatal.
vllm_run() {
    local name="$1" sizes="$2" spec="${3:-}" dir="$OUT/$1"
    need_checkpoint; gpus_free
    mkdir -p "$dir"
    local extra=()
    [ -n "$spec" ] && extra=(--speculative-config "$spec")
    local tuned="{\"mode\":\"VLLM_COMPILE\",\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[$sizes]}"
    local status=failed attempt=""
    trap 'stop_group "'"$dir"'/server.pid"' EXIT
    # Our capture sizes; the recipe's compilation; and without the text-only switch.
    if vllm_serve "$dir" tuned --language-model-only --compilation-config "$tuned" \
        --max-cudagraph-capture-size "${sizes##*,}" ${extra[@]+"${extra[@]}"}; then
        status=served attempt=tuned
    elif vllm_serve "$dir" recipe --language-model-only ${extra[@]+"${extra[@]}"}; then
        status=served attempt=recipe
    elif vllm_serve "$dir" bare ${extra[@]+"${extra[@]}"}; then
        status=served attempt=bare
    fi
    local label="{\"stack\":\"vllm\",\"name\":\"$name\"}"
    if [ "$status" = served ]; then
        tool quality --base http://127.0.0.1:8000 --sequences "$OUT/ds_two/sequences.json" \
            --texts "$OUT/texts.json" --out "$dir/quality.json" || status=quality-failed
        tool bench --base http://127.0.0.1:8000 --sequences "$OUT/ds_two/sequences.json" \
            --concurrency 1,8,32 --out-len 256 --label "$label" --out "$dir/bench.json" || status=bench-failed
    fi
    stop_group "$dir/server.pid"
    [ -n "$DRY_RUN" ] && return 0
    "$PY" -I -c 'import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
tries = []
for c in sorted(d.glob("command.*.txt")):
    a = c.name.split(".")[1]
    log = d / f"server.{a}.log"
    ready = d / f"ready_seconds.{a}"
    tries.append({"attempt": a, "command": c.read_text(), "ready_seconds": int(ready.read_text()) if ready.exists() else None,
                  "log_tail": log.read_text(errors="replace").splitlines()[-30:] if log.exists() else []})
json.dump({"status": sys.argv[2], "served_by": sys.argv[3] or None, "attempts": tries,
           "error": None if sys.argv[2] == "served" else (tries[-1]["log_tail"][-5:] if tries else "no start-up")},
          open(d / "server.json", "w"), indent=1)' "$dir" "$status" "$attempt"
    say "$name: $status"
}
vllm_base() { vllm_run vllm_base "1,2,4,8,16,24,32,48,64"; }
vllm_spec() {
    vllm_run vllm_spec "6,12,24,48,96,192,288,384" \
        '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy","rejection_sample_method":"standard","enable_adaptive_verification":false}'
}

summary() {
    tool summarize --out "$OUT"
}

ALL="facts fetch_start setup build kernels dev_two_gpu tiny_two_gpu tiny_two_gpu_memory fetch_wait verify sequences halves two_resident two_serving node_w0 node_w3 vllm_setup vllm_base vllm_spec summary"
NAMED="halves_reference"
for step in ${*:-$ALL}; do
    case " $ALL $NAMED " in
        *" $step "*) ;;
        *) die "unknown step $step (steps: $ALL $NAMED)" ;;
    esac
    case "$step" in
        summary) summary ;;
        node_w*)
            # NODE_WIDTHS chooses which of the node's widths run.
            case " $NODE_WIDTHS " in *" ${step#node_w} "*) run_step "$step" ;; *) say "$step: not in NODE_WIDTHS" ;; esac ;;
        *) run_step "$step" ;;
    esac
done
say "results in $OUT"
