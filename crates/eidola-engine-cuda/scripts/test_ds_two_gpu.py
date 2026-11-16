"""Tests of the two-GPU session's host side on synthetic inputs, with local servers
standing in for the checkpoint's host, the node and the yardstick.

    python3 -I crates/eidola-engine-cuda/scripts/test_ds_two_gpu.py
"""

from __future__ import annotations

import hashlib
import http.server
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds_two_gpu as t  # noqa: E402

t.log = lambda msg: None


class Server:
    """A local HTTP server answering with `handler(method, path, body) -> (status,
    content type, bytes or a list of byte chunks)`."""

    def __init__(self, handler):
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def answer(self, method):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else None
                outer.requests.append((method, self.path, body, dict(self.headers)))
                status, ctype, data = handler(method, self.path, body)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                if isinstance(data, list):
                    self.end_headers()
                    for chunk in data:
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            def do_GET(self):
                self.answer("GET")

            def do_POST(self):
                self.answer("POST")

        self.requests = []
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()


def sse(events, pad=True):
    """Server-sent events, each followed by padding a framed stream carries."""
    out = []
    for e in events:
        data = e if isinstance(e, str) else json.dumps(e)
        out.append(f"data: {data}  \n\n".encode())
        if pad:
            out.append(b":" + b" " * 37 + b"\n\n")
    return out


class Texts(unittest.TestCase):
    def test_the_golden_sources_give_the_health_texts(self):
        x = t.texts()
        self.assertEqual(x["hello_reply"], "Hello! How can\n can I help you today?")
        self.assertEqual(x["hello_fork_top3"], ["\n", "#", "<"])
        self.assertEqual(x["hello_user"], "hello")
        self.assertTrue(x["story_user"].startswith("Write a short story"))
        self.assertTrue(x["story"].startswith("The ferry to the island"))
        self.assertTrue(x["passage"].startswith("The keeper of the northern light"))
        self.assertGreater(len(x["passage"]), 2000)
        self.assertEqual(x["story_floor"], 0.55)


class FetchVerify(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.files = {"a.safetensors": b"A" * 5000, "config.json": b"{}", "tokenizer.json": b"tok", "tokenizer_config.json": b"tc"}
        self.server = Server(lambda m, path, b: (200, "application/octet-stream", self.files[path.lstrip("/")]) if path.lstrip("/") in self.files else (404, "text/plain", b""))
        self.recorded = {n: {"size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for n, d in self.files.items() if n != "tokenizer_config.json"}

    def tearDown(self):
        self.server.close()

    def test_a_fetch_names_only_the_recorded_bytes_and_verify_hashes_them(self):
        dest, extras = self.tmp / "ck", self.tmp / "extras"
        r = t.fetch(dest, self.recorded, self.server.base, 2, ["tokenizer_config.json"], extras)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["a.safetensors", "config.json", "tokenizer.json"])
        self.assertEqual((extras / "tokenizer_config.json").read_bytes(), b"tc")
        self.assertEqual(r["bytes"], 5000 + 2 + 3 + 2)
        v = t.verify(dest, self.recorded, 4)
        manifest = "".join(f"{hashlib.sha256(self.files[n]).hexdigest()}  {n}\n" for n in sorted(["a.safetensors", "config.json", "tokenizer.json"]))
        self.assertEqual(v["manifest"], manifest)
        self.assertEqual(v["weights_sha256"], hashlib.sha256(manifest.encode()).hexdigest())
        # A second fetch fetches nothing.
        self.assertEqual(t.fetch(dest, self.recorded, self.server.base, 2, [], None)["bytes"], 0)
        # A changed byte on disk is refused.
        (dest / "a.safetensors").write_bytes(b"A" * 4999 + b"B")
        with self.assertRaises(SystemExit):
            t.verify(dest, self.recorded, 4)

    def test_a_download_that_is_not_the_recorded_bytes_never_gets_its_name(self):
        wrong = dict(self.recorded)
        wrong["config.json"] = {"size": 2, "sha256": "0" * 64}
        dest = self.tmp / "ck2"
        orig = t.time.sleep
        t.time.sleep = lambda s: None
        try:
            with self.assertRaises(ValueError):
                t.fetch(dest, {"config.json": wrong["config.json"]}, self.server.base, 1, [], None)
        finally:
            t.time.sleep = orig
        self.assertEqual(list(dest.iterdir()), [])

    def test_a_missing_or_unrecorded_file_is_refused(self):
        dest = self.tmp / "ck3"
        t.fetch(dest, self.recorded, self.server.base, 2, [], None)
        (dest / "config.json").unlink()
        with self.assertRaises(SystemExit):
            t.verify(dest, self.recorded, 2)
        t.fetch(dest, self.recorded, self.server.base, 2, [], None)
        (dest / "b.safetensors").write_bytes(b"x")
        with self.assertRaises(SystemExit):
            t.verify(dest, self.recorded, 2)


class Node(unittest.TestCase):
    def test_answers_are_read_through_the_padding_and_held_to_the_cpu(self):
        answers = {"plain": ("think", "Canberra.", []), "tool": ("", "", [("get_weather", '{"city":"Paris"}')])}

        def handler(method, path, body):
            if path == "/healthz":
                return 200, "text/plain", b"ok" + b" " * 100
            if path == "/v1/engine/info":
                return 200, "application/json", json.dumps({"model": "m", "device": {"name": "B300"}}).encode() + b" " * 50
            user = body["messages"][-1]["content"]
            name = "tool" if "weather" in user else "plain"
            r, c, calls = answers[name]
            events = [{"choices": [{"delta": {"role": "assistant"}}], "usage": {"completion_tokens": 0}}]
            if r:
                events.append({"choices": [{"delta": {"reasoning_content": r}}]})
            if c:
                events.append({"choices": [{"delta": {"content": c}}]})
            for i, (n, a) in enumerate(calls):
                events.append({"choices": [{"delta": {"tool_calls": [{"index": i, "id": "call_0", "type": "function", "function": {"name": n, "arguments": a}}]}}]})
            events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            events.append({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 7}})
            events.append("[DONE]")
            return 200, "text/event-stream", sse(events)

        s = Server(handler)
        tmp = Path(tempfile.mkdtemp())
        (tmp / "requests").mkdir()
        cpu = {"plain": ("think", "Canberra.", []), "tool": ("", "", [("get_weather", '{"city":"Lyon"}')])}
        for n, (r, c, calls) in cpu.items():
            t.write_json(tmp / f"requests/{n}.w5.json", {"reasoning_content": r, "content": c, "tool_calls": [{"name": a, "arguments": b} for a, b in calls], "generated_tokens": 7, "prompt_tokens": 10, "max_tokens": 96})
        prompts = [{"name": "plain", "messages": [{"role": "user", "content": "capital?"}]}, {"name": "tool", "messages": [{"role": "user", "content": "weather?"}], "tools": [{"type": "function"}]}]
        texts = {"system": "s", "hello_user": "hello", "story_user": "story"}
        try:
            t.wait_healthy(f"{s.base}/healthz", 10)
            r = t.node_session(s.base, "tok", "ab" * 32, "m", prompts, texts, tmp, 96, 64)
        finally:
            s.close()
        self.assertEqual([x["prompt"] for x in r["results"]], ["hello", "story", "plain", "tool"])
        self.assertEqual(r["results"][2]["content"], "Canberra.")
        self.assertEqual(r["results"][2]["usage"]["completion_tokens"], 7)
        self.assertIsNotNone(r["results"][2]["first_delta_seconds"])
        self.assertTrue(r["cpu_checks"][0]["equal"])
        self.assertFalse(r["cpu_checks"][1]["equal"])
        self.assertFalse(r["cpu_checks"][1]["same"]["tool_calls"])
        self.assertFalse(r["equal_to_cpu"])
        method, path, body, headers = s.requests[-1]
        self.assertEqual(headers["Authorization"], "Bearer tok")
        self.assertEqual(headers["X-Eidola-Weights-Sha256"], "ab" * 32)
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["tools"], [{"type": "function"}])
        w = t.compare_widths({"w0": r, "w3": r})
        self.assertTrue(w["equal"])
        other = json.loads(json.dumps(r))
        other["results"][0]["content"] = "x"
        self.assertFalse(t.compare_widths({"w0": r, "w3": other})["equal"])


def model(prefix):
    """A deterministic stand-in model: the most likely next token of a prefix."""
    return (sum(prefix) * 7 + len(prefix)) % 50


class Yardstick(unittest.TestCase):
    def setUp(self):
        def handler(method, path, body):
            prompt = body["prompt"]
            if body.get("stream"):
                n = body["max_tokens"]
                events = [{"choices": [{"text": "x"}]} for _ in range(n)]
                events.append({"choices": [], "usage": {"completion_tokens": n}})
                events.append("[DONE]")
                return 200, "text/event-stream", sse(events, pad=False)
            if body.get("prompt_logprobs") is not None:
                k = body["prompt_logprobs"]
                entries = [None]
                for i in range(1, len(prompt)):
                    best = model(prompt[:i])
                    e = {str((best + j) % 50): {"logprob": -0.1 * (j + 1), "rank": j + 1} for j in range(k)}
                    if str(prompt[i]) not in e:
                        e[str(prompt[i])] = {"logprob": -9.0, "rank": 17}
                    entries.append(e)
                return 200, "application/json", json.dumps({"choices": [{"text": "", "prompt_logprobs": entries}]}).encode()
            ids, p = [], list(prompt)
            for _ in range(body["max_tokens"]):
                ids.append(model(p))
                p.append(ids[-1])
            return 200, "application/json", json.dumps({"choices": [{"text": "Hello! How can\n can I help you today?!", "token_ids": ids, "finish_reason": "length"}]}).encode()

        self.s = Server(handler)

    def tearDown(self):
        self.s.close()

    def test_scores_and_greedy_tokens_from_prompt_logprobs(self):
        # A sequence the stand-in model predicts from position 3 on, but for one token.
        tokens = [1, 2, 3]
        for _ in range(9):
            tokens.append(model(tokens))
        tokens[8] = (tokens[8] + 1) % 50
        fork = 6
        seqs = {
            "health": {
                "hello": {"tokens": tokens, "start": 3, "fork": fork, "fork_tokens": [model(tokens[: fork + 1]), 99, 98]},
                "story": {"tokens": tokens, "start": 3},
                "passage": {"tokens": tokens, "start": 1},
            },
            "prompts": [{"name": "plain", "tokens": [4, 5, 6]}],
        }
        texts = {"hello_reply": "Hello! How can\n can I help you today?", "story_floor": 0.55, "passage_floor": 0.99}
        q = t.quality(self.s.base, "ds", seqs, texts, 5)
        hello = q["health"]["hello"]
        self.assertEqual(hello["scored"], 9)
        # Token 8 is not the most likely one, and every prediction after it reads it.
        self.assertEqual(hello["wrong_positions"], [7, 8, 9, 10])
        self.assertFalse(hello["healthy"])
        self.assertTrue(hello["fork_ok"])
        self.assertEqual(len(hello["fork_top3"]), 3)
        self.assertTrue(hello["greedy_is_the_reported_reply"])
        self.assertAlmostEqual(q["health"]["story"]["accuracy"], 5 / 9)
        self.assertTrue(q["health"]["story"]["healthy"])  # 0.556 against 0.55
        self.assertFalse(q["health"]["passage"]["healthy"])
        self.assertEqual(len(q["prompts"][0]["token_ids"]), 5)
        self.assertEqual(q["health"]["passage"]["argmax"][0], model(tokens[:1]))

    def test_the_bench_streams_fixed_lengths_at_each_concurrency(self):
        prompts = [[1, 2, 3, i] for i in range(8)]
        b = t.bench(self.s.base, "ds", prompts, [1, 4], 6, {"stack": "fake"})
        self.assertEqual([r["requests"] for r in b["runs"]], [3, 4])
        for r in b["runs"]:
            self.assertEqual(r["output_tokens"], 6 * r["requests"])
            self.assertEqual(r["stack"], "fake")
            self.assertIsNotNone(r["median_ttft_seconds"])
        bodies = [b for m, p, b, h in self.s.requests]
        self.assertTrue(all(b["ignore_eos"] and b["temperature"] == 0.0 for b in bodies))


class Summary(unittest.TestCase):
    def test_the_summary_reads_every_result(self):
        out = Path(tempfile.mkdtemp())
        (out / "done").mkdir()
        (out / "done/dev_two_gpu").touch()
        (out / "steps.txt").write_text("setup 0 120\nbuild 0 600\n")
        t.write_json(out / "facts.json", {"gpus": "2x B300", "driver": "580", "image": "img", "vllm_version": "0.33.0"})
        t.write_json(out / "verify.json", {"files": 51, "bytes": 1, "seconds": 2, "gb_per_second": 3, "weights_sha256": "ab"})
        two = out / "ds_two"
        t.write_json(two / "compare.json", {"equal": True, "checks": [{}] * 44})
        t.write_json(two / "score.json", {"sequences": {"story": {"accuracy": 0.66, "scored": 800, "healthy": True, "argmax": [1, 2, 3]}}})
        t.write_json(two / "generate.json", {"equal": True, "checks": [], "results": [{"prompt": "plain", "width": 0, "token_ids": [1, 2, 3, 4]}]})
        t.write_json(two / "bench.json", {"runs": [{"width": 0, "concurrency": 1, "output_tokens_per_second": 10.0, "median_ttft_seconds": 0.5, "median_decode_tokens_per_second": 11.0, "requests_detail": []}]})
        r = {"results": [{"prompt": "plain", "reasoning_content": "a", "content": "b", "tool_calls": [], "usage": {"completion_tokens": 3}}], "equal_to_cpu": True}
        t.write_json(out / "node_w0.json", r)
        t.write_json(out / "node_w3.json", r)
        v = out / "vllm_base"
        t.write_json(v / "server.json", {"status": "served"})
        t.write_json(v / "quality.json", {"health": {"story": {"accuracy": 0.6, "scored": 800, "healthy": True, "argmax": [1, 2, 4]}}, "prompts": [{"prompt": "plain", "token_ids": [1, 2, 9]}]})
        t.write_json(v / "bench.json", {"runs": [{"concurrency": 1, "output_tokens_per_second": 90.0, "median_ttft_seconds": 0.1, "median_decode_tokens_per_second": 95.0, "requests_detail": []}]})
        s = t.summarize(out)
        self.assertTrue(s["two_gpu"]["compare"])
        self.assertEqual(s["two_gpu"]["compare_checks"], 44)
        self.assertTrue(s["node"]["across_widths"]["equal"])
        self.assertAlmostEqual(s["yardstick"]["vllm_base"]["argmax_agreement_with_ours"]["story"]["share"], 2 / 3)
        self.assertEqual(s["yardstick"]["vllm_base"]["greedy_agreement_with_ours"]["plain"]["common_prefix"], 2)
        self.assertNotIn("requests_detail", s["yardstick"]["vllm_base"]["bench"][0])
        text = (out / "summary.txt").read_text()
        self.assertIn("weights hash ab", text)
        self.assertIn("90.0 out tok/s", text)


class NodeFailures(unittest.TestCase):
    """A request the node fails is a failed step, never a recorded answer."""

    def run_stream(self, chunks):
        s = Server(lambda m, path, b: (200, "text/event-stream", chunks))
        try:
            with self.assertRaises(SystemExit):
                t.node_chat(s.base, "tok", "ab" * 32, "m", {"messages": [{"role": "user", "content": "q"}]})
        finally:
            s.close()

    def test_an_error_event_fails_the_request(self):
        start = sse([{"choices": [{"delta": {"role": "assistant"}}]}])
        self.run_stream(start + [b'event: error\ndata: {"error": {"message": "engine stopped"}}\n\n'])

    def test_a_stream_without_a_finish_reason_and_usage_fails_the_request(self):
        self.run_stream(sse([{"choices": [{"delta": {"content": "partial"}}]}, "[DONE]"]))


class YardstickFailures(unittest.TestCase):
    """A failed or short yardstick request fails the benchmark, never counts as tokens."""

    def run_stream(self, chunks):
        s = Server(lambda m, path, b: (200, "text/event-stream", chunks))
        try:
            with self.assertRaises(SystemExit):
                t.stream_completion(s.base, "m", [1, 2, 3], 4)
        finally:
            s.close()

    def test_an_error_payload_fails_the_request(self):
        self.run_stream(sse([{"choices": [{"text": "a"}]}, {"error": {"message": "out of memory"}}]))

    def test_a_stream_without_usage_fails_the_request(self):
        self.run_stream(sse([{"choices": [{"text": "a"}]}, "[DONE]"]))

    def test_a_short_output_fails_the_request(self):
        self.run_stream(sse([{"choices": [{"text": "a"}]}, {"choices": [], "usage": {"completion_tokens": 2}}, "[DONE]"]))

    def test_a_whole_output_is_measured(self):
        s = Server(lambda m, path, b: (200, "text/event-stream", sse([{"choices": [{"text": "a"}]}, {"choices": [], "usage": {"completion_tokens": 4}}, "[DONE]"])))
        try:
            self.assertEqual(t.stream_completion(s.base, "m", [1, 2, 3], 4)["tokens"], 4)
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
