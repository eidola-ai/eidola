#!/usr/bin/env python3
"""A development stand-in for the payment processor, at the gateway's edge: it
sits between the attesting shim and a locally run gateway that has no Stripe
key, forwards every request to the gateway unchanged, and answers the three
requests that need Stripe itself:

- GET /v1/prices: three plans (two one-time top-ups and a monthly subscription);
- POST /v1/account/checkout: a checkout link, https://example.com/?eidola-dev-checkout=<price>,
  which the app hands to the system browser; and it remembers the purchase;
- GET /v1/account/subscription: no subscription ("none").

With `--refuse-terms N` it also refuses the first N acceptance submissions
(POST /v1/account/terms) with 503, as a gateway that failed mid-creation
would, and then refuses a checkout for an account whose acceptance it refused
with 428 (the terms gate) until a later submission for it goes through: how
the UI tests reach an account created with its agreement unrecorded.

GET /_stub/secret-sha256 answers the SHA-256 of the account secret the last
authenticated request carried: what a UI test compares the clipboard with
after copying that secret while it is masked on screen.

A balance read for an account that "bought" here (by its Basic credentials)
then carries the plan's credits as a pool of their own, so the app's re-read
on coming back from the browser shows the purchase. Development and the UI
tests only; it is never in front of a real gateway.

    billing_edge_stub.py --port 18081 --upstream http://127.0.0.1:18080

Standard library only.
"""

import argparse
import base64
import hashlib
import http.client
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PLANS = [
    {"id": "price_dev_topup_5", "product_name": "Top-up", "unit_amount": 500, "currency": "usd",
     "type": "one_time", "credits": 5_000_000},
    {"id": "price_dev_topup_20", "product_name": "Large top-up", "unit_amount": 2000, "currency": "usd",
     "type": "one_time", "credits": 20_000_000},
    {"id": "price_dev_monthly", "product_name": "Monthly", "unit_amount": 1000, "currency": "usd",
     "type": "recurring", "recurring": {"interval": "month", "interval_count": 1}, "credits": 10_000_000},
]
CREDITS = {p["id"]: p["credits"] for p in PLANS}
BOUGHT = {}
NEEDS_TERMS = set()
REFUSE = {"terms": 0}
LAST_SECRET = {"sha256": ""}
LOCK = threading.Lock()


class Edge(BaseHTTPRequestHandler):
    upstream = None

    def log_message(self, fmt, *args):
        print("edge:", fmt % args, flush=True)

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def forward(self, body):
        u = urllib.parse.urlsplit(self.upstream)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        conn.request(self.command, self.path, body=body or None, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        return resp, data

    def handle_any(self):
        body = self.body()
        path = urllib.parse.urlsplit(self.path).path
        who = self.headers.get("Authorization")
        if who and who.startswith("Basic "):
            _, _, secret = base64.b64decode(who[6:]).decode().partition(":")
            with LOCK:
                LAST_SECRET["sha256"] = hashlib.sha256(secret.encode()).hexdigest()
        if self.command == "GET" and path == "/_stub/secret-sha256":
            with LOCK:
                return self.reply(200, {"sha256": LAST_SECRET["sha256"]})
        if self.command == "GET" and path == "/v1/prices":
            return self.reply(200, {"data": PLANS})
        if self.command == "POST" and path == "/v1/account/terms":
            with LOCK:
                refuse = REFUSE["terms"] > 0
                if refuse:
                    REFUSE["terms"] -= 1
                    NEEDS_TERMS.add(who)
            if refuse:
                return self.reply(503, {"error": {"message": "the stand-in refused this submission"}})
        if self.command == "POST" and path == "/v1/account/checkout":
            with LOCK:
                gated = who in NEEDS_TERMS
            if gated:
                return self.reply(428, {"error": {"message": "terms acceptance required"}})
            price = json.loads(body or b"{}").get("price_id", "")
            if price not in CREDITS:
                return self.reply(400, {"error": {"message": "unknown price"}})
            with LOCK:
                BOUGHT.setdefault(who, []).append(price)
            return self.reply(200, {"checkout_url": f"https://example.com/?eidola-dev-checkout={price}"})
        if self.command == "GET" and path == "/v1/account/subscription":
            return self.reply(200, {"state": "none"})
        resp, data = self.forward(body)
        if self.command == "POST" and path == "/v1/account/terms" and resp.status < 300:
            with LOCK:
                NEEDS_TERMS.discard(who)
        if self.command == "GET" and path == "/v1/account/balances" and resp.status == 200:
            with LOCK:
                bought = list(BOUGHT.get(who, []))
            if bought:
                balances = json.loads(data)
                for price in bought:
                    balances["available"] += CREDITS[price]
                    balances["pools"].append({"amount": CREDITS[price], "source": "purchase",
                                              "expires_at": "2027-11-23T00:00:00Z"})
                data = json.dumps(balances).encode()
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() not in ("content-length", "transfer-encoding", "connection"):
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = handle_any


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18081)
    ap.add_argument("--upstream", default="http://127.0.0.1:18080")
    ap.add_argument("--refuse-terms", type=int, default=0)
    args = ap.parse_args()
    Edge.upstream = args.upstream
    REFUSE["terms"] = args.refuse_terms
    ThreadingHTTPServer(("127.0.0.1", args.port), Edge).serve_forever()


if __name__ == "__main__":
    main()
