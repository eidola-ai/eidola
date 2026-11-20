#!/usr/bin/env python3
"""vLLM on the same B200 and checkpoint as the Qwen3.8 session (standard library only).

    qwen_vllm.py serve <checkpoint> <log> [--mtp]     start the pinned image, wait until it serves
    qwen_vllm.py stop
    qwen_vllm.py bench <out.json> [--mtp]             the yardstick workloads of
                                                      qwen_workloads.json, as `qwen_whole bench`
                                                      runs them on our engine (B: random-id
                                                      prompts, greedy; C: conversations, each
                                                      turn extending the last, sampled), at each
                                                      concurrency; output tokens/s, time to first
                                                      token, per-stream decode, prefix-cache hits
                                                      and (with --mtp) the drafts' acceptance
    qwen_vllm.py generate <health.safetensors> <out.json>
                                                      greedy replies to the capital prompt and to the
                                                      passage's first 200 tokens (token ids in, ids out)

Every failure exits non-zero with its reason; a server that does not come up prints its
log's tail.
"""

import json
import os
import struct
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# vLLM 0.33.0, the version measured on a B200 before (multi-arch index digest).
IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
NAME = "eidola-qwen-vllm"
BASE = "http://127.0.0.1:8000"
MTP = '{"method":"mtp","num_speculative_tokens":3}'
# The workloads, shared with `examples/qwen_whole.rs`.
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_workloads.json")) as workloads:
    WORKLOADS = json.load(workloads)


def requests_at(c):
    b = WORKLOADS["B"]
    return max(b["requests_at_least"], b["requests_per_concurrent"] * c)


def conversations_at(c):
    w = WORKLOADS["C"]
    return max(w["conversations_at_least"], w["conversations_per_concurrent"] * c)


def die(msg):
    print(f"FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metrics():
    """The speculative-decoding and prefix-cache counters, summed over labels (a counter's
    `_total` suffix dropped, its `_created` sample skipped)."""
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#") or not line.startswith(("vllm:spec_decode_num_", "vllm:prefix_cache_")):
                continue
            name, value = line.rsplit(" ", 1)
            key = name.split("{")[0]
            if "per_pos" in key or key.endswith("_created"):
                continue
            key = key.removesuffix("_total")
            out[key] = out.get(key, 0.0) + float(value)
    return out


def cmd_serve(ckpt, log, mtp):
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    args = ["docker", "run", "-d", "--name", NAME, "--gpus", "all", "--ipc", "host", "-p", "8000:8000",
            "-v", f"{os.path.abspath(ckpt)}:/model:ro", IMAGE, "/model", "--served-model-name", "qwen",
            "--enable-prefix-caching", "--max-model-len", "16384", "--max-num-seqs", "256",
            "--language-model-only"]
    if mtp:
        args += ["--speculative-config", MTP]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        die(f"docker run: {r.stderr.strip()}")
    for _ in range(180):
        time.sleep(10)
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", NAME], capture_output=True, text=True)
        if state.stdout.strip() != "true":
            break
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=5) as h:
                if h.status == 200:
                    subprocess.run(f"docker logs {NAME} > {log} 2>&1", shell=True)
                    print("serving")
                    return
        except OSError:
            pass  # not answering yet; try again
    subprocess.run(f"docker logs {NAME} > {log} 2>&1", shell=True)
    tail = subprocess.run(["tail", "-40", log], capture_output=True, text=True).stdout
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    die(f"vLLM did not serve; log tail:\n{tail}")


def stream_one(ids, out_len, sampling):
    """One request, streamed: its time to first token, its total, and the ids it produced
    (exactly `out_len`, or the run fails)."""
    body = {"model": "qwen", "prompt": ids, "max_tokens": out_len, "ignore_eos": True, "stream": True,
            "return_token_ids": True, **sampling}
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, last, out = None, None, []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            for choice in chunk.get("choices", []):
                if choice.get("token_ids"):
                    now = time.time()
                    first = first or now
                    last = now
                    out += choice["token_ids"]
    total = time.time() - t0
    if len(out) != out_len:
        die(f"a request produced {len(out)} of {out_len} token ids")
    decode = (out_len - 1) / (last - first) if last > first else None
    return {"ttft": first - t0, "total": total, "decode": decode, "ids": out, "input_len": len(ids)}


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def row(workload, c, res, wall, before, mtp):
    tokens = sum(len(r["ids"]) for r in res)
    inputs = sum(r["input_len"] for r in res)
    out = {"workload": workload, "concurrency": c, "requests": len(res), "wall_s": wall,
           "output_tokens": tokens, "output_tok_per_s": tokens / wall, "input_tok_per_s": inputs / wall,
           "mean_input_len": inputs / len(res), "ttft_median_s": median(r["ttft"] for r in res),
           "decode_median": median(r["decode"] for r in res)}
    after = metrics()
    d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    queries = d.get("vllm:prefix_cache_queries", 0)
    if queries > 0:
        out["prefix_hit_fraction"] = d.get("vllm:prefix_cache_hits", 0) / queries
    drafts = d.get("vllm:spec_decode_num_drafts", 0)
    if mtp and drafts > 0:
        out["mean_acceptance_length"] = 1 + d.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts
    return out


def cmd_bench(out, mtp):
    import random
    rng = random.Random(1)
    ids = lambda n: [rng.randrange(240000) for _ in range(n)]
    rows = []
    b, cw = WORKLOADS["B"], WORKLOADS["C"]
    greedy = {"temperature": b["temperature"]}
    for c in b["concurrency"]:
        prompts = [ids(b["prompt_tokens"]) for _ in range(requests_at(c))]
        before = metrics()
        t0 = time.time()
        with ThreadPoolExecutor(c) as pool:
            res = list(pool.map(lambda p: stream_one(p, b["output_tokens"], greedy), prompts))
        rows.append(row("B", c, res, time.time() - t0, before, mtp))
        print(json.dumps(rows[-1]), flush=True)
    sampled = {k: cw[k] for k in ("temperature", "top_p", "top_k")}
    new = cw["turn_tokens"] - cw["output_tokens"]
    for c in cw["concurrency"]:
        convs = [(ids(cw["turn_tokens"]), [ids(new) for _ in range(cw["turns"] - 1)])
                 for _ in range(conversations_at(c))]

        def conversation(i):
            prompt, later = convs[i]
            res = []
            for t in range(cw["turns"]):
                r = stream_one(prompt, cw["output_tokens"], {**sampled, "seed": 7919 * c + 31 * i + t})
                res.append(r)
                if t + 1 < cw["turns"]:
                    prompt = prompt + r["ids"] + later[t]
            return res

        before = metrics()
        t0 = time.time()
        with ThreadPoolExecutor(c) as pool:
            res = [r for conv in pool.map(conversation, range(len(convs))) for r in conv]
        rows.append(row("C", c, res, time.time() - t0, before, mtp))
        rows[-1]["conversations"] = len(convs)
        print(json.dumps(rows[-1]), flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


# Token tensors' element formats: the health check writes I32.
ID_FORMATS = {"I32": "i", "I64": "q", "U32": "I"}


def safetensors_ids(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        if name not in header:
            die(f"{path} holds no {name}")
        meta = header[name]
        fmt = ID_FORMATS.get(meta["dtype"])
        if fmt is None:
            die(f"{name} is {meta['dtype']}")
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        data = f.read(b - a)
    size = struct.calcsize(fmt)
    if len(data) % size:
        die(f"{name}: {len(data)} bytes of {meta['dtype']}")
    return list(struct.unpack(f"<{len(data) // size}{fmt}", data))


def cmd_generate(health, out):
    capital = safetensors_ids(health, "capital.prompt")
    passage = safetensors_ids(health, "passage.tokens")[:200]
    res = {}
    for name, ids, n in (("capital", capital, 48), ("passage", passage, 96)):
        r = post("/v1/completions", {"model": "qwen", "prompt": ids, "max_tokens": n, "temperature": 0.0,
                                     "ignore_eos": True, "return_token_ids": True})
        choice = r["choices"][0]
        res[name] = {"text": choice.get("text"), "token_ids": choice.get("token_ids")}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: (v["token_ids"] or [])[:16] for k, v in res.items()}))


def main():
    a = sys.argv[1:]
    if not a:
        die(__doc__)
    if a[0] == "serve" and len(a) >= 3:
        cmd_serve(a[1], a[2], "--mtp" in a)
    elif a[0] == "stop":
        subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    elif a[0] == "bench" and len(a) >= 2:
        cmd_bench(a[1], "--mtp" in a)
    elif a[0] == "generate" and len(a) == 3:
        cmd_generate(a[1], a[2])
    else:
        die(__doc__)


if __name__ == "__main__":
    main()
