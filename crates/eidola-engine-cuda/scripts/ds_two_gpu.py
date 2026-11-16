"""The two-GPU session's host side: the checkpoint's fetch and verification, the
health texts, the clients of the node and of the yardstick, and the summary.

    python3 -I ds_two_gpu.py <command> [options]      (standard library only)

Commands:

- `texts --out F`: the health check's texts (the system prompt, `hello` with its
  publicly reported reply and near tie, the story request and story, the passage),
  read from the golden tooling's sources without importing them.
- `fetch --dir D`: every file `upstream-sha256.json` records, fetched in parallel from
  the pinned revision. Each download starts from nothing into a file of its own (never
  resumed), is hashed as it arrives, and gets its name only when its size and SHA-256
  are the recorded ones; a file already under its name is left to `verify`.
  `--extras-dir E` also fetches the small files a yardstick needs that no hash is
  recorded for (`tokenizer_config.json`), recording the SHA-256 they had.
- `verify --dir D`: every recorded file present and read back from disk, hashed in
  parallel, against the recorded hashes; and the node's weights hash (the manifest of
  the covered files, `crates/eidola-server-engine/AGENTS.md` -> The weights hash) from
  the same digests.
- `node --base URL ...`: the node's chat API, one request at a time: each request's
  reasoning, content, tool calls, finish reason, token counts and time to the first
  delta, and the generation prompts against the CPU reference executor's results.
- `quality --base URL ...`: an OpenAI-compatible completions server given our token
  ids: next-token accuracy and the most likely token at every scored position of the
  health sequences (`prompt_logprobs`), the `hello` reply generated greedily, and the
  generation prompts' greedy tokens.
- `bench --base URL ...`: the throughput prompts at each concurrency, streamed, a fixed
  output length: time to the first token, decode rate, output tokens per second.
- `summarize --out OUT`: `summary.json` and `summary.txt` from everything under OUT.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
GOLDEN = REPO / "crates/eidola-engine-model/golden/deepseek_v41"
PROMPTS = REPO / "crates/eidola-server-engine/examples/ds_generate/prompts.json"
AGENT = "python-urllib"
# Files the node's weights hash covers when present (eidola_common::weights_manifest).
COVERED = ("config.json", "model.safetensors.index.json", "tokenizer.json", "chat_template.jinja", "generation_config.json")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def upstream_list() -> dict:
    return json.loads((GOLDEN / "upstream-sha256.json").read_text())


# ---------------------------------------------------------------------------------------
# texts


def _constants(path: Path) -> dict:
    """Module-level assignments of `path` whose values are literals, or tuples and lists
    of literals and names already collected."""
    out: dict = {}
    for node in ast.parse(path.read_text()).body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue

        def value(n):
            if isinstance(n, ast.Name) and n.id in out:
                return out[n.id]
            if isinstance(n, (ast.Tuple, ast.List)):
                return [value(e) for e in n.elts]
            return ast.literal_eval(n)

        try:
            out[node.targets[0].id] = value(node.value)
        except (ValueError, TypeError, SyntaxError):
            pass
    return out


def texts(golden: Path = GOLDEN) -> dict:
    h = _constants(golden / "ds_golden/health.py")
    r = _constants(golden / "real_golden.py")
    hello = dict(h["HELLO_MESSAGES"])
    story = dict(h["STORY_MESSAGES"])
    assert hello["system"] == story["system"] == h["SYSTEM"]
    return {
        "system": h["SYSTEM"],
        "hello_user": hello["user"],
        "hello_reply": h["HELLO_REPLY"],
        "hello_fork": h["HELLO_FORK"],
        "hello_fork_top3": list(h["HELLO_FORK_TOP3"]),
        "story_user": story["user"],
        "story": h["STORY"],
        "passage": r["TEXT"],
        "story_floor": h["STORY_FLOOR"],
        "passage_floor": h["PASSAGE_FLOOR"],
    }


# ---------------------------------------------------------------------------------------
# fetch, verify


class Rate:
    """Bytes received across threads, logged every `every` seconds."""

    def __init__(self, every: float = 30.0):
        self.lock = threading.Lock()
        self.bytes = 0
        self.began = time.time()
        self.last = self.began
        self.every = every

    def add(self, n: int) -> None:
        with self.lock:
            self.bytes += n
            now = time.time()
            if now - self.last >= self.every:
                self.last = now
                log(f"fetched {self.bytes / 1e9:.1f} GB at {self.bytes / 1e9 / (now - self.began):.2f} GB/s")


def fetch_one(base: str, name: str, size: int | None, sha256: str | None, dest: Path, rate: Rate, attempts: int = 5) -> dict:
    """`name` into `dest/name`: a fresh download into a file of its own, hashed as it
    arrives, renamed only when it is the recorded bytes (when a hash is recorded)."""
    final = dest / name
    for attempt in range(1, attempts + 1):
        partial = dest / f"{name}.partial-{os.getpid()}-{threading.get_ident()}"
        partial.unlink(missing_ok=True)
        h = hashlib.sha256()
        got = 0
        began = time.time()
        try:
            req = urllib.request.Request(f"{base}/{name}", headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(partial, "wb") as f:
                while chunk := resp.read(1 << 22):
                    f.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
                    rate.add(len(chunk))
            digest = h.hexdigest()
            if size is not None and got != size:
                raise ValueError(f"{got} bytes, recorded {size}")
            if sha256 is not None and digest != sha256:
                raise ValueError(f"sha256 {digest}, recorded {sha256}")
            partial.replace(final)
            seconds = time.time() - began
            log(f"{name}: {got / 1e9:.2f} GB in {seconds:.0f} s, matches")
            return {"name": name, "bytes": got, "seconds": seconds, "sha256": digest, "recorded": sha256 is not None}
        except (OSError, ValueError, urllib.error.URLError) as e:
            partial.unlink(missing_ok=True)
            log(f"{name}: attempt {attempt} failed: {e}")
            if attempt == attempts:
                raise
            time.sleep(min(60, 5 * attempt))
    raise AssertionError("unreachable")


def fetch(dest: Path, files: dict, base: str, workers: int, extras: list[str], extras_dir: Path | None, every: float = 30.0) -> dict:
    """Every file of `files` (name -> {size, sha256}) not yet under its name in `dest`,
    then `extras` (no recorded hash) into `extras_dir`."""
    dest.mkdir(parents=True, exist_ok=True)
    for stale in dest.glob("*.partial-*"):
        stale.unlink()
    todo = [n for n in sorted(files, key=lambda n: -files[n]["size"]) if not (dest / n).exists()]
    log(f"{len(todo)} of {len(files)} files to fetch, {sum(files[n]['size'] for n in todo) / 1e9:.1f} GB, {workers} at a time")
    rate = Rate(every)
    began = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(fetch_one, base, n, files[n]["size"], files[n]["sha256"], dest, rate) for n in todo]
        for j in concurrent.futures.as_completed(jobs):
            results.append(j.result())
    if extras_dir is not None:
        extras_dir.mkdir(parents=True, exist_ok=True)
        for n in extras:
            if not (extras_dir / n).exists():
                results.append(fetch_one(base, n, None, None, extras_dir, rate))
    seconds = time.time() - began
    total = sum(r["bytes"] for r in results)
    return {"files": results, "bytes": total, "seconds": seconds, "gb_per_second": total / 1e9 / seconds if seconds else None, "workers": workers}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def weights_hash(digests: dict) -> tuple[str, str]:
    """The node's weights hash over `digests` (file name -> SHA-256) of the covered
    files: every `*.safetensors` and the named small files present. Returns the hash
    and the manifest."""
    names = sorted(n for n in digests if n.endswith(".safetensors") or n in COVERED)
    manifest = "".join(f"{digests[n]}  {n}\n" for n in sorted(names, key=lambda n: n.encode()))
    return hashlib.sha256(manifest.encode()).hexdigest(), manifest


def verify(dest: Path, files: dict, workers: int) -> dict:
    """Every recorded file read back from `dest` and hashed against its record; the
    weights hash from the same digests. Raises on a missing or different file."""
    missing = [n for n in files if not (dest / n).is_file()]
    if missing:
        raise SystemExit(f"REFUSED: {len(missing)} recorded files are missing: {missing[:5]}")
    present = sorted(p.name for p in dest.iterdir() if p.is_file() and (p.name.endswith(".safetensors") or p.name in COVERED))
    unknown = [n for n in present if n not in files]
    if unknown:
        raise SystemExit(f"REFUSED: files no hash is recorded for would enter the weights hash: {unknown}")
    began = time.time()
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        digests = dict(zip(present, pool.map(lambda n: sha256_file(dest / n), present)))
    seconds = time.time() - began
    bad = [n for n in present if digests[n] != files[n]["sha256"] or (dest / n).stat().st_size != files[n]["size"]]
    if bad:
        raise SystemExit(f"REFUSED: {bad} are not the recorded bytes")
    total = sum((dest / n).stat().st_size for n in present)
    whash, manifest = weights_hash(digests)
    log(f"{len(present)} files ({total / 1e9:.1f} GB) match their recorded hashes in {seconds:.0f} s; weights hash {whash}")
    return {"files": len(present), "bytes": total, "seconds": seconds, "gb_per_second": total / 1e9 / seconds if seconds else None, "weights_sha256": whash, "manifest": manifest}


# ---------------------------------------------------------------------------------------
# HTTP


def post(url: str, body: dict, headers: dict | None = None, timeout: float = 3600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", "User-Agent": AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def get(url: str, headers: dict | None = None, timeout: float = 30):
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def sse_events(resp):
    """The data payloads of a server-sent event stream, comments (padding) skipped, in
    arrival order, each with the time it was read."""
    data: list[str] = []
    event = None
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line == "":
            if data:
                yield event, "\n".join(data), time.time()
            data, event = [], None
        elif line.startswith(":"):
            continue
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif line.startswith("event:"):
            event = line[6:].strip()
    if data:
        yield event, "\n".join(data), time.time()


def wait_healthy(url: str, seconds: float, alive=None) -> float:
    began = time.time()
    while time.time() - began < seconds:
        if alive is not None and not alive():
            raise SystemExit(f"the server exited before {url} answered")
        try:
            status, _ = get(url, timeout=5)
            if status == 200:
                return time.time() - began
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise SystemExit(f"{url} did not answer within {seconds:.0f} s")


# ---------------------------------------------------------------------------------------
# node


def node_chat(base: str, token: str, weights: str, model: str, body: dict) -> dict:
    """One streamed chat request to the node: its parsed answer and timings."""
    body = {**body, "model": model, "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Authorization": f"Bearer {token}", "X-Eidola-Weights-Sha256": weights}
    began = time.time()
    out = {"reasoning_content": "", "content": "", "tool_calls": [], "finish_reason": None, "usage": None, "continuable": None}
    first = None
    with post(f"{base}/v1/chat/completions", body, headers) as resp:
        for event, data, at in sse_events(resp):
            if event == "error":
                raise SystemExit(f"the node answered with an error: {data}")
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                for key in ("reasoning_content", "content"):
                    if delta.get(key):
                        out[key] += delta[key]
                        first = first or at
                for call in delta.get("tool_calls") or []:
                    out["tool_calls"].append({"name": call["function"]["name"], "arguments": call["function"]["arguments"]})
                    first = first or at
                if choice.get("finish_reason"):
                    out["finish_reason"] = choice["finish_reason"]
                    out["continuable"] = choice.get("continuable")
            if chunk.get("usage") and not chunk.get("choices"):
                out["usage"] = chunk["usage"]
    if out["finish_reason"] is None or out["usage"] is None:
        raise SystemExit("the node's stream ended without a finish reason and usage")
    end = time.time()
    out["seconds"] = end - began
    out["first_delta_seconds"] = None if first is None else first - began
    return out


def cpu_result(cpu: Path | None, name: str) -> dict | None:
    if cpu is None:
        return None
    files = sorted((cpu / "requests").glob(f"{name}.w*.json"))
    return read_json(files[0]) if files else None


def node_session(base: str, token: str, weights: str, model: str, prompts: list, texts_: dict, cpu: Path | None, max_new: int, health_new: int) -> dict:
    """The health requests and the generation prompts through the node, greedily; the
    generation prompts against the CPU reference executor's results."""
    status, info = get(f"{base}/v1/engine/info", {"Authorization": f"Bearer {token}"})
    info = json.loads(info.decode().rstrip())
    results, checks = [], []
    health = {
        "hello": [{"role": "system", "content": texts_["system"]}, {"role": "user", "content": texts_["hello_user"]}],
        "story": [{"role": "system", "content": texts_["system"]}, {"role": "user", "content": texts_["story_user"]}],
    }
    for name, messages in health.items():
        r = node_chat(base, token, weights, model, {"messages": messages, "max_completion_tokens": health_new, "temperature": 0.0})
        r["prompt"] = name
        log(f"node {name}: {r['usage']} first delta {r['first_delta_seconds']}")
        results.append(r)
    for p in prompts:
        body = {"messages": p["messages"], "max_completion_tokens": max_new, "temperature": 0.0}
        if "tools" in p:
            body["tools"] = p["tools"]
        r = node_chat(base, token, weights, model, body)
        r["prompt"] = p["name"]
        log(f"node {p['name']}: {r['usage']} first delta {r['first_delta_seconds']}")
        results.append(r)
        theirs = cpu_result(cpu, p["name"])
        if theirs is not None:
            usage = r["usage"] or {}
            same = {
                "reasoning_content": r["reasoning_content"] == theirs["reasoning_content"],
                "content": r["content"] == theirs["content"],
                "tool_calls": [(c["name"], c["arguments"]) for c in r["tool_calls"]] == [(c["name"], c["arguments"]) for c in theirs["tool_calls"]],
                "completion_tokens": usage.get("completion_tokens") == theirs["generated_tokens"] if theirs["max_tokens"] == max_new else None,
                "prompt_tokens": usage.get("prompt_tokens") == theirs["prompt_tokens"],
            }
            # A run to a different limit is compared on the shorter one's text.
            if theirs["max_tokens"] != max_new:
                shorter = min((r, theirs), key=lambda x: len(x["reasoning_content"]) + len(x["content"]))
                longer = theirs if shorter is r else r
                same["reasoning_content"] = longer["reasoning_content"].startswith(shorter["reasoning_content"])
                same["content"] = shorter["content"] == "" or longer["content"].startswith(shorter["content"])
                same["tool_calls"] = None
            checks.append({"prompt": p["name"], "cpu_max_tokens": theirs["max_tokens"], "same": same, "equal": all(v is not False for v in same.values())})
    return {"info": info, "results": results, "cpu_checks": checks, "equal_to_cpu": all(c["equal"] for c in checks) if checks else None}


def compare_widths(runs: dict) -> dict:
    """Whether the node answered every prompt alike at every draft width."""
    names = None
    per = {}
    for width, run in runs.items():
        answers = {r["prompt"]: (r["reasoning_content"], r["content"], [(c["name"], c["arguments"]) for c in r["tool_calls"]], (r["usage"] or {}).get("completion_tokens")) for r in run["results"]}
        names = names or list(answers)
        per[width] = answers
    widths = list(per)
    same = {n: all(per[w][n] == per[widths[0]][n] for w in widths) for n in names or []}
    return {"widths": widths, "same_at_every_width": same, "equal": all(same.values())}


# ---------------------------------------------------------------------------------------
# quality (a completions server given our token ids)


def completion(base: str, model: str, body: dict) -> dict:
    with post(f"{base}/v1/completions", {"model": model, **body}) as r:
        return json.loads(r.read())


def score_prompt_logprobs(tokens: list, start: int, entries: list) -> dict:
    """Next-token accuracy from `start` and the most likely token at each scored
    position, from a completions server's `prompt_logprobs` for `tokens` (entry `i` is
    token `i`'s, holding the most likely tokens at its position with their ranks)."""
    argmax, hits = [], 0
    for i in range(max(start, 1), len(tokens)):
        entry = {int(k): v for k, v in (entries[i] or {}).items()}
        best = next((t for t, v in entry.items() if v.get("rank") == 1), None)
        argmax.append(best)
        hits += best == tokens[i]
    return {"tokens": len(tokens), "start": start, "scored": len(argmax), "accuracy": hits / max(len(argmax), 1), "argmax": argmax}


def quality(base: str, model: str, seqs: dict, texts_: dict, max_new: int) -> dict:
    out = {"health": {}, "prompts": []}
    for name in ("hello", "story", "passage"):
        h = seqs["health"][name]
        k = 3 if name == "hello" else 1
        r = completion(base, model, {"prompt": h["tokens"], "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": k})
        entries = r["choices"][0]["prompt_logprobs"]
        s = score_prompt_logprobs(h["tokens"], h["start"], entries)
        if name == "hello":
            fork = h["fork"]
            entry = {int(t): v for t, v in (entries[fork + 1] or {}).items()}
            top = sorted(((v["rank"], t, v["logprob"]) for t, v in entry.items() if v.get("rank") is not None))[:3]
            s["fork_top3"] = [[t, lp] for _, t, lp in top]
            s["fork_ok"] = bool(top) and top[0][1] in h["fork_tokens"]
            first = s["start"] if s["start"] >= 1 else 1
            wrong = [first - 1 + j for j, (a, t) in enumerate(zip(s["argmax"], h["tokens"][first:])) if a != t and first - 1 + j != fork]
            s["wrong_positions"] = wrong
            s["healthy"] = not wrong and s["fork_ok"]
            head = h["tokens"][: h["start"]]
            g = completion(base, model, {"prompt": head, "max_tokens": len(h["tokens"]) - h["start"] + 4, "temperature": 0.0, "return_token_ids": True})
            s["greedy_text"] = g["choices"][0]["text"]
            s["greedy_is_the_reported_reply"] = s["greedy_text"].startswith(texts_["hello_reply"])
        else:
            s["floor"] = texts_[f"{name}_floor"]
            s["healthy"] = s["accuracy"] >= s["floor"]
        log(f"quality {name}: accuracy {s['accuracy']:.4f} over {s['scored']} positions")
        out["health"][name] = s
    for p in seqs["prompts"]:
        g = completion(base, model, {"prompt": p["tokens"], "max_tokens": max_new, "temperature": 0.0, "return_token_ids": True})
        c = g["choices"][0]
        out["prompts"].append({"prompt": p["name"], "token_ids": c.get("token_ids"), "text": c["text"], "finish_reason": c.get("finish_reason")})
    return out


# ---------------------------------------------------------------------------------------
# bench


def stream_completion(base: str, model: str, prompt: list, out_len: int) -> dict:
    began = time.time()
    first = None
    tokens = None
    body = {"model": model, "prompt": prompt, "max_tokens": out_len, "temperature": 0.0, "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    with post(f"{base}/v1/completions", body) as resp:
        for event, data, at in sse_events(resp):
            if event == "error":
                raise SystemExit(f"the yardstick answered with an error: {data}")
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise SystemExit(f"the yardstick answered with an error: {data}")
            if any(c.get("text") for c in chunk.get("choices", [])) and first is None:
                first = at
            if chunk.get("usage"):
                tokens = chunk["usage"].get("completion_tokens")
    # A fixed-length benchmark counts only requests that produced their whole output.
    if tokens != out_len:
        raise SystemExit(f"a yardstick request produced {tokens} of {out_len} tokens")
    end = time.time()
    decode = (tokens - 1) / (end - first) if tokens and tokens > 1 and first and end > first else None
    return {"tokens": tokens, "submitted": began, "ttft_seconds": None if first is None else first - began, "seconds": end - began, "decode_tokens_per_second": decode}


def bench(base: str, model: str, prompts: list, concurrency: list, out_len: int, label: dict) -> dict:
    runs = []
    for c in concurrency:
        n = 3 if c == 1 else c
        began = time.time()
        with concurrent.futures.ThreadPoolExecutor(c) as pool:
            results = list(pool.map(lambda p: stream_completion(base, model, p, out_len), prompts[:n]))
        wall = time.time() - began
        ttft = [r["ttft_seconds"] for r in results if r["ttft_seconds"] is not None]
        decode = [r["decode_tokens_per_second"] for r in results if r["decode_tokens_per_second"] is not None]
        tokens = sum(r["tokens"] or 0 for r in results)
        run = {
            **label, "concurrency": c, "requests": n, "input_len": len(prompts[0]), "output_len": out_len,
            "wall_seconds": wall, "output_tokens": tokens, "output_tokens_per_second": tokens / wall,
            "mean_ttft_seconds": statistics.mean(ttft) if ttft else None, "median_ttft_seconds": statistics.median(ttft) if ttft else None,
            "mean_decode_tokens_per_second": statistics.mean(decode) if decode else None,
            "median_decode_tokens_per_second": statistics.median(decode) if decode else None,
            "requests_detail": results,
        }
        log(f"bench {label} concurrency {c}: {run['output_tokens_per_second']:.1f} output tok/s, ttft {run['median_ttft_seconds']}")
        runs.append(run)
    return {"runs": runs}


# ---------------------------------------------------------------------------------------
# summarize


def _maybe(path: Path):
    try:
        return read_json(path)
    except (OSError, ValueError):
        return None


def agreement(a: list, b: list) -> dict:
    n = min(len(a), len(b))
    same = sum(1 for x, y in zip(a, b) if x == y)
    prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), n)
    return {"compared": n, "same": same, "share": same / n if n else None, "common_prefix": prefix, "lengths": [len(a), len(b)]}


def summarize(out: Path) -> dict:
    s: dict = {"facts": _maybe(out / "facts.json"), "steps": []}
    steps = out / "steps.txt"
    if steps.exists():
        for line in steps.read_text().splitlines():
            name, status, seconds = line.split()[:3]
            s["steps"].append({"step": name, "status": int(status), "seconds": int(seconds)})
    s["fetch"] = {k: v for k, v in (_maybe(out / "fetch.json") or {}).items() if k != "files"} or None
    v = _maybe(out / "verify.json")
    s["verify"] = {k: v[k] for k in ("files", "bytes", "seconds", "gb_per_second", "weights_sha256")} if v else None
    two = out / "ds_two"
    s["two_gpu"] = {
        "compare": (_maybe(two / "compare.json") or {}).get("equal"),
        "compare_checks": len((_maybe(two / "compare.json") or {}).get("checks", [])),
        "score": {k: {f: x.get(f) for f in ("accuracy", "scored", "healthy", "fork_top3", "equals_the_comparisons_logits")} for k, x in ((_maybe(two / "score.json") or {}).get("sequences") or {}).items()},
        "generate": (_maybe(two / "generate.json") or {}).get("checks"),
        "generate_equal": (_maybe(two / "generate.json") or {}).get("equal"),
        "bench": [{k: v for k, v in r.items() if k != "requests_detail"} for r in (_maybe(two / "bench.json") or {}).get("runs", [])],
    }
    s["dev_two_gpu"] = (out / "done/dev_two_gpu").exists()
    s["tiny_two_gpu"] = (out / "done/tiny_two_gpu").exists()
    node = {}
    for f in sorted(out.glob("node_w*.json")):
        node[f.stem] = {k: x for k, x in read_json(f).items() if k != "results"}
        node[f.stem]["answers"] = [{k: r.get(k) for k in ("prompt", "usage", "finish_reason", "first_delta_seconds", "seconds")} for r in read_json(f)["results"]]
    if len(node) > 1:
        node["across_widths"] = compare_widths({k: read_json(out / f"{k}.json") for k in node})
    s["node"] = node or None
    yard = {}
    ours = (_maybe(two / "score.json") or {}).get("sequences") or {}
    ours_gen = {r["prompt"]: r["token_ids"] for r in (_maybe(two / "generate.json") or {}).get("results", []) if r.get("width") == 0}
    for d in sorted(out.glob("vllm_*")):
        if not d.is_dir():
            continue
        y = {"server": _maybe(d / "server.json")}
        q = _maybe(d / "quality.json")
        if q:
            y["health"] = {k: {f: x.get(f) for f in ("accuracy", "scored", "healthy", "fork_top3", "greedy_text", "greedy_is_the_reported_reply")} for k, x in q["health"].items()}
            y["argmax_agreement_with_ours"] = {k: agreement(x["argmax"], ours[k]["argmax"]) for k, x in q["health"].items() if k in ours}
            y["greedy_agreement_with_ours"] = {p["prompt"]: agreement(p["token_ids"] or [], ours_gen[p["prompt"]]) for p in q["prompts"] if p["prompt"] in ours_gen}
        b = _maybe(d / "bench.json")
        if b:
            y["bench"] = [{k: v for k, v in r.items() if k != "requests_detail"} for r in b["runs"]]
        yard[d.name] = y
    s["yardstick"] = yard or None
    write_json(out / "summary.json", s)
    (out / "summary.txt").write_text(render(s))
    return s


def render(s: dict) -> str:
    """The short human summary."""
    L = []
    f = s.get("facts") or {}
    L.append(f"GPUs: {f.get('gpus')}; driver {f.get('driver')}; image {f.get('image')}; vLLM {f.get('vllm_version')}")
    for st in s["steps"]:
        L.append(f"  step {st['step']:<22} status {st['status']}  {st['seconds'] / 60:6.1f} min")
    if s.get("fetch"):
        L.append(f"fetch: {s['fetch'].get('bytes', 0) / 1e9:.1f} GB at {s['fetch'].get('gb_per_second') or 0:.2f} GB/s with {s['fetch'].get('workers')} streams")
    if s.get("verify"):
        L.append(f"verify: {s['verify']['files']} files match their recorded hashes; weights hash {s['verify']['weights_sha256']}")
    L.append(f"dev model on two GPUs (node test): {'passed' if s['dev_two_gpu'] else 'not passed'}; tiny executor scenarios on two GPUs: {'passed' if s['tiny_two_gpu'] else 'not passed'}")
    t = s["two_gpu"]
    L.append(f"two GPUs, resident, against the one-half-at-a-time recordings: equal={t['compare']} over {t['compare_checks']} arrays")
    for k, v in (t["score"] or {}).items():
        L.append(f"  health {k}: accuracy {v.get('accuracy')} over {v.get('scored')} positions; healthy={v.get('healthy')}")
    L.append(f"two GPUs, executor, greedy tokens across widths and against the CPU: equal={t['generate_equal']}")
    for r in t["bench"]:
        L.append(f"  ours w{r['width']} c{r['concurrency']}: {r['output_tokens_per_second']:.1f} out tok/s; ttft {r['median_ttft_seconds']}; decode {r['median_decode_tokens_per_second']} tok/s")
    for k, v in (s.get("node") or {}).items():
        if k == "across_widths":
            L.append(f"node: same answers at widths {v['widths']}: {v['equal']}")
        else:
            L.append(f"node {k}: equal to the CPU: {v.get('equal_to_cpu')}")
    for k, y in (s.get("yardstick") or {}).items():
        srv = y.get("server") or {}
        L.append(f"{k}: {srv.get('status')} {srv.get('error') or ''}")
        for n, h in (y.get("health") or {}).items():
            L.append(f"  health {n}: accuracy {h.get('accuracy')} healthy={h.get('healthy')}")
        for n, a in (y.get("argmax_agreement_with_ours") or {}).items():
            L.append(f"  most likely token agrees with ours on {n}: {a['share']}")
        for n, a in (y.get("greedy_agreement_with_ours") or {}).items():
            L.append(f"  greedy {n}: common prefix with ours {a['common_prefix']} of {a['compared']}")
        for r in y.get("bench") or []:
            L.append(f"  c{r['concurrency']}: {r['output_tokens_per_second']:.1f} out tok/s; ttft {r['median_ttft_seconds']}; decode {r['median_decode_tokens_per_second']} tok/s")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("texts")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("fetch")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--extras-dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("verify")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("node")
    for a in ("--base", "--token", "--weights", "--model"):
        p.add_argument(a, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=PROMPTS)
    p.add_argument("--cpu", type=Path)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--health-new", type=int, default=64)
    p.add_argument("--wait", type=float, default=3600)
    p.add_argument("--pid", type=int, default=0, help="the node's process: waiting ends if it exits")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("quality")
    p.add_argument("--base", required=True)
    p.add_argument("--model", default="ds")
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--texts", type=Path, required=True)
    p.add_argument("--max-new", type=int, default=96)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("bench")
    p.add_argument("--base", required=True)
    p.add_argument("--model", default="ds")
    p.add_argument("--sequences", type=Path, required=True)
    p.add_argument("--concurrency", default="1,8,32")
    p.add_argument("--out-len", type=int, default=256)
    p.add_argument("--label", default="{}")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("summarize")
    p.add_argument("--out", type=Path, required=True)
    o = ap.parse_args()

    if o.cmd == "texts":
        write_json(o.out, texts())
    elif o.cmd == "fetch":
        up = upstream_list()
        base = f"https://huggingface.co/{up['repository']}/resolve/{up['revision']}"
        write_json(o.out, fetch(o.dir, up["files"], base, o.workers, ["tokenizer_config.json"], o.extras_dir))
    elif o.cmd == "verify":
        v = verify(o.dir, upstream_list()["files"], o.workers)
        write_json(o.out, v)
        (o.out.parent / "weights.sha256").write_text(v["weights_sha256"] + "\n")
    elif o.cmd == "node":
        def alive() -> bool:
            try:
                os.kill(o.pid, 0)
                return True
            except OSError:
                return False

        wait_healthy(f"{o.base}/healthz", o.wait, alive if o.pid else None)
        r = node_session(o.base, o.token, o.weights, o.model, read_json(o.prompts), read_json(o.texts), o.cpu, o.max_new, o.health_new)
        write_json(o.out, r)
        if r["equal_to_cpu"] is False:
            sys.exit("the node's answers differ from the CPU reference executor's")
    elif o.cmd == "quality":
        write_json(o.out, quality(o.base, o.model, read_json(o.sequences), read_json(o.texts), o.max_new))
    elif o.cmd == "bench":
        seqs = read_json(o.sequences)
        write_json(o.out, bench(o.base, o.model, seqs["bench"]["prompts"], [int(c) for c in o.concurrency.split(",")], o.out_len, json.loads(o.label)))
    elif o.cmd == "summarize":
        s = summarize(o.out)
        print(render(s), end="")
        # The report is written either way; a greedy disagreement across draft widths fails it.
        if (s.get("node") or {}).get("across_widths", {}).get("equal") is False:
            sys.exit("the node's answers differ across draft widths")


if __name__ == "__main__":
    main()
