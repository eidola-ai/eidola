#!/usr/bin/env python3
"""Serve the update a development bundle's update check finds (its `update_fixtures`
URL, `build-bundle.sh --dev-update`), for the phone interface's UI tests.

    update_fixtures.py BUNDLE_DIR WORK_DIR --port 8791 --base http://HOST:8791 [--bind 127.0.0.1]
        [--broken-version 0.0.10]

(`--bind 0.0.0.0` for a physical device, which reaches it over the LAN at HOST. The shell
remembers the last version that failed its trial and the app refuses to install it again, so a
device that keeps its install from one run to the next needs a `--broken-version` of its own each
run.)

Two releases, built here into WORK_DIR and served from it over plain HTTP:

- v0.0.9, BUNDLE_DIR plus a marker file (so a digest of its own): an update that
  boots and confirms itself;
- v0.0.10, a page with no app, which never confirms: the shell falls back.

`GET /update.json` answers v0.0.9 until its container has been downloaded (the
approval of it), then v0.0.10 until that one's has, then v0.0.9 again. Each
answer is the app's `StagedRelease`: a release index and its manifest (schema
6, recording the bundle), which a development bundle takes as verified, and
from which the real bundle reader, download, byte checks and staging run.
Every request is logged to stdout, so a run shows when a container was fetched
(and that none was before an approval). Standard library only.
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def container(src, out):
    row = subprocess.run([sys.executable, os.path.join(HERE, "bundle_container.py"), src, out],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(row)


def staged(base, version, row):
    tag = f"v{version}"
    manifest = json.dumps({"schema_version": 6, "artifacts": {"eidola-mobile-bundle": row}}).encode()
    return {"index": {
        "schema_version": 2, "version": version,
        "git_commit": "9c3a" + "0" * 34 + version.replace(".", "")[-2:],
        "git_tag": tag, "released_at": "2026-11-23T00:00:00Z",
        "artifact_manifest": {"url": f"{base}/download/{tag}/artifact-manifest.json",
                              "sigstore_bundle_url": f"{base}/download/{tag}/artifact-manifest.json.sigstore"},
        "human_attestations": [],
        "artifacts": {"eidola-mobile-bundle": {"url": f"{base}/download/{tag}/eidola-mobile-bundle.tar"}},
    }, "manifest": base64.b64encode(manifest).decode()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("work")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--base", required=True)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--broken-version", default="0.0.10")
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(args.bundle, f"{work}/update")
    with open(f"{work}/update/update-marker.txt", "w") as f:
        f.write("the update a UI test approves\n")
    os.makedirs(f"{work}/broken")
    with open(f"{work}/broken/index.html", "w") as f:
        f.write("<!doctype html><title>a bundle that never confirms</title>\n")
    releases = []
    for version, src in [("0.0.9", "update"), (args.broken_version, "broken")]:
        os.makedirs(f"{work}/www/download/v{version}", exist_ok=True)
        row = container(f"{work}/{src}", f"{work}/www/download/v{version}/eidola-mobile-bundle.tar")
        releases.append((f"/download/v{version}/eidola-mobile-bundle.tar", staged(args.base, version, row), row))
        print(f"v{version}: bundle digest {row['digest']}", flush=True)
    state = {"next": 0}
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=f"{work}/www", **k)

        def log_message(self, fmt, *a):
            print("fixtures:", fmt % a, flush=True)

        def do_GET(self):
            if self.path.split("?")[0] == "/update.json":
                with lock:
                    body = json.dumps(releases[state["next"]][1]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with lock:
                for i, (path, _, _) in enumerate(releases):
                    if self.path.split("?")[0] == path:
                        state["next"] = (i + 1) % len(releases)
                        print(f"fixtures: container of release {i} fetched; next offer {state['next']}", flush=True)
            super().do_GET()

    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
