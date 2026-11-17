#!/usr/bin/env python3
"""vLLM on the same B200 and checkpoint as the Qwen3.8 session (standard library only).

    qwen_vllm.py serve <checkpoint> <log> [--mtp]     start the pinned image, wait until it serves
    qwen_vllm.py stop
    qwen_vllm.py bench <out.json> [--mtp]             1,024 random ids in, 256 out, greedy, at
                                                      concurrency 1, 32 and 64; output tokens/s,
                                                      time to first token, per-stream decode, and
                                                      (with --mtp) the drafts' acceptance
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


def die(msg):
    print(f"FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metrics():
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("vllm:spec_decode_num_") and not line.startswith("#"):
                name, value = line.rsplit(" ", 1)
                key = name.split("{")[0]
                if "per_pos" in key:
                    continue
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
            pass
    subprocess.run(f"docker logs {NAME} > {log} 2>&1", shell=True)
    tail = subprocess.run(["tail", "-40", log], capture_output=True, text=True).stdout
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    die(f"vLLM did not serve; log tail:\n{tail}")


def stream_one(ids, out_len):
    body = {"model": "qwen", "prompt": ids, "max_tokens": out_len, "temperature": 0.0,
            "ignore_eos": True, "stream": True}
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    first, n = None, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            text = chunk["choices"][0].get("text", "")
            if text and first is None:
                first = time.time() - t0
            n += 1
    total = time.time() - t0
    return {"ttft": first or total, "total": total, "chunks": n}


def cmd_bench(out, mtp):
    import random
    rng = random.Random(1)
    rows = []
    for c in (1, 32, 64):
        prompts = [[rng.randrange(240000) for _ in range(1024)] for _ in range(2 * c)]
        before = metrics() if mtp else {}
        t0 = time.time()
        with ThreadPoolExecutor(c) as pool:
            res = list(pool.map(lambda p: stream_one(p, 256), prompts))
        wall = time.time() - t0
        row = {"concurrency": c, "requests": len(prompts), "wall_s": wall,
               "output_tok_per_s": 256 * len(prompts) / wall,
               "ttft_median_s": sorted(r["ttft"] for r in res)[len(res) // 2],
               "decode_per_stream": sorted(256 / max(r["total"] - r["ttft"], 1e-9) for r in res)[len(res) // 2]}
        if mtp:
            after = metrics()
            d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
            drafts = d.get("vllm:spec_decode_num_drafts", 0)
            if drafts > 0:
                row["mean_acceptance_length"] = 1 + d.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(rows, open(out, "w"), indent=1)


def safetensors_ids(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        meta = header[name]
        if meta["dtype"] != "I64":
            die(f"{name} is {meta['dtype']}")
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        data = f.read(b - a)
    return list(struct.unpack(f"<{len(data) // 8}q", data))


def cmd_generate(health, out):
    capital = safetensors_ids(health, "capital.prompt")
    passage = safetensors_ids(health, "passage.tokens")[:200]
    res = {}
    for name, ids, n in (("capital", capital, 48), ("passage", passage, 96)):
        r = post("/v1/completions", {"model": "qwen", "prompt": ids, "max_tokens": n, "temperature": 0.0,
                                     "ignore_eos": True, "return_token_ids": True})
        choice = r["choices"][0]
        res[name] = {"text": choice.get("text"), "token_ids": choice.get("token_ids")}
    json.dump(res, open(out, "w"), indent=1)
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
