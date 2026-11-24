#!/usr/bin/env python3
"""A development stand-in for a model: an OpenAI-compatible server that streams a
fixed answer, so the app's conversation flow runs end to end in the Simulator and
the emulator — through the shell's pipe, app-core's turn and its stream — with no
account and no engine. Never part of a release.

  openai_stub.py [--port 8790] [--delay 0.03]

GET /v1/models lists one model, `stub`. POST /v1/chat/completions answers the last
user message: streamed (server-sent events), its reasoning first
(`delta.reasoning_content`) and then a markdown answer with a heading, a list,
code and a table; or, unstreamed, a short line (what a title request gets).
Standard library only.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "stub"


def last_user_text(body):
    """The last thing a person wrote (not a directive app-core adds, such as
    "Respond to #post.")."""
    for message in reversed(body.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        # app-core opens a post with a header naming its author and time; the words are the
        # last paragraph.
        paragraphs = [p.strip() for p in (content or "").split("\n\n") if p.strip()]
        if paragraphs and not paragraphs[-1].startswith("Respond to #"):
            return paragraphs[-1]

    return ""


def reasoning_for(prompt):
    topic = prompt.splitlines()[0][:60] if prompt else "this"
    return (
        f"The reader wrote “{topic}”. A useful answer shows the whole range of what "
        "the page can set: a heading, a short list, a code block and a small table, "
        "then a closing line. Keep it brief and calm."
    )


def answer_for(prompt):
    topic = prompt.splitlines()[0][:48] if prompt else "your message"
    return (
        f"## On “{topic}”\n\n"
        "Here is a short answer, set in the same face as the post you wrote.\n\n"
        "- **First**, a point with *emphasis*.\n"
        "- Second, with `inline code` in it.\n"
        "- Third, and last.\n\n"
        "```rust\n"
        "fn main() {\n"
        "    println!(\"hello from the development server\");\n"
        "}\n"
        "```\n\n"
        "| Part | What it shows |\n"
        "|---|---|\n"
        "| Heading | the prose ramp |\n"
        "| Table | columns on a phone |\n\n"
        "> That is the whole of it — reply to continue."
    )


def chunks(text):
    """Split into small word-ish pieces, keeping whitespace and newlines."""
    piece = ""
    for ch in text:
        piece += ch
        if ch in " \n" and len(piece) >= 3:
            yield piece
            piece = ""
    if piece:
        yield piece


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    delay = 0.03

    def log_message(self, fmt, *args):
        print(f"stub: {self.command} {self.path} {fmt % args}", flush=True)

    def send_json(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self.send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "stub"}]})
        else:
            self.send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_json(400, {"error": {"message": "bad json"}})
            return
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_json(404, {"error": {"message": "not found"}})
            return
        prompt = last_user_text(body)
        created = int(time.time())
        if not body.get("stream"):
            words = prompt.split()[:5] or ["Conversation"]
            self.send_json(200, {
                "id": "stub-1", "object": "chat.completion", "created": created, "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": " ".join(words)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def event(delta, finish=None):
            payload = {
                "id": "stub-1", "object": "chat.completion.chunk", "created": created, "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            data = f"data: {json.dumps(payload)}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        try:
            event({"role": "assistant"})
            for piece in chunks(reasoning_for(prompt)):
                event({"reasoning_content": piece})
                time.sleep(self.delay)
            for piece in chunks(answer_for(prompt)):
                event({"content": piece})
                time.sleep(self.delay)
            event({}, "stop")
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):x}\r\n".encode() + done + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--delay", type=float, default=0.03)
    args = parser.parse_args()
    Handler.delay = args.delay
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"stub: listening on 127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
