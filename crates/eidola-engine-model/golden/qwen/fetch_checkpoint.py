"""Fetch the checkpoint's files at the pinned revision, each checked against
`upstream-sha256.json` before it gets the name anything loads.

    uv run python fetch_checkpoint.py --dir <dir> [--truncation prefix|full]

`prefix` is checkpoint layers 0-3, the drafter and the file holding the
embedding and head (about 8 GB); `full` is every shard (about 31 GB). Each file
is downloaded whole, from its first byte, under a name nothing loads, hashed
while it streams, and renamed only when its size and SHA-256 are the recorded
ones. A partial download is never resumed: a failed attempt is deleted and the
next one starts again from the first byte. A file already present under its
final name is checked again, never trusted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_golden import upstream  # noqa: E402

AGENT = "eidola-golden-fetch"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_one(name: str, dest: Path, attempts: int) -> int:
    rec = upstream.record(name)
    url = f"https://huggingface.co/{upstream.REPO}/resolve/{upstream.REVISION}/{name}"
    final = dest / name
    for attempt in range(1, attempts + 1):
        partial = dest / f".{name}.partial-{os.getpid()}-{threading.get_ident()}"
        partial.unlink(missing_ok=True)
        h, got, began = hashlib.sha256(), 0, time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(partial, "wb") as f:
                while chunk := resp.read(1 << 22):
                    f.write(chunk)
                    h.update(chunk)
                    got += len(chunk)
                f.flush()
                os.fsync(f.fileno())
            if got != rec["size"]:
                raise upstream.Mismatch(f"{name}: {got} bytes, recorded {rec['size']}")
            if h.hexdigest() != rec["sha256"]:
                raise upstream.Mismatch(f"{name}: sha256 {h.hexdigest()}, recorded {rec['sha256']}")
            partial.replace(final)
            log(f"{name}: {got / 1e9:.2f} GB in {time.time() - began:.0f} s, matches")
            return got
        except Exception as e:  # noqa: BLE001 - every failure retries from byte 0; the last one raises
            partial.unlink(missing_ok=True)
            log(f"{name}: attempt {attempt} failed: {e}")
            if attempt == attempts:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--truncation", choices=sorted(upstream.TRUNCATIONS), default="prefix")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=4)
    a = ap.parse_args()
    a.dir.mkdir(parents=True, exist_ok=True)
    for stale in a.dir.glob(".*.partial-*"):
        stale.unlink()
    names = upstream.files_of(a.truncation)
    todo = []
    for name in names:
        path = a.dir / name
        if path.exists():
            try:
                upstream.check_file(path)
                continue
            except upstream.Mismatch as e:
                log(f"{e}; setting it aside as {name}.refused and fetching it again")
                path.replace(a.dir / f"{name}.refused")
        todo.append(name)
    todo.sort(key=lambda n: -upstream.record(n)["size"])
    total = sum(upstream.record(n)["size"] for n in todo)
    log(f"{len(todo)} of {len(names)} files to fetch, {total / 1e9:.2f} GB")
    began = time.time()
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        jobs = {pool.submit(fetch_one, n, a.dir, a.attempts): n for n in todo}
        failed = []
        for job in concurrent.futures.as_completed(jobs):
            if job.exception() is not None:
                failed.append(jobs[job])
    if failed:
        log(f"FAILED: {len(failed)} files did not fetch: {', '.join(sorted(failed))}")
        return 1
    log(f"fetched {total / 1e9:.2f} GB in {time.time() - began:.0f} s; checking every file again")
    upstream.check_dir(a.dir, a.truncation)
    log(f"all {len(names)} files match the recorded hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
