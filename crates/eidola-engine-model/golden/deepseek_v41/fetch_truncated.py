"""Fetch a truncation of the released checkpoint by HTTP range reads.

    uv run python fetch_truncated.py --dir <checkpoint dir>

About 14 GB: `config.json`, `tokenizer.json`, and the tensors
`ds_golden/truncated.py` keeps, re-packed byte for byte into one local file
per upstream shard. Engram tables are not fetched here; `real_golden.py`
fetches only the rows its tokens touch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ds_golden import truncated  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    opts = parser.parse_args()
    opts.dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json"):
        (opts.dir / name).write_bytes(truncated.get(name))
    index = json.loads(truncated.get("model.safetensors.index.json"))["weight_map"]
    shards = sorted({shard for name, shard in index.items() if truncated.wanted(name)})
    total = 0
    for shard in shards:
        out = opts.dir / shard
        if out.exists():
            print(f"{shard}: present")
            continue
        partial = out.with_suffix(".partial")
        size = truncated.fetch_shard(shard, partial)
        partial.rename(out)
        total += size
        print(f"{shard}: {size / 1e9:.2f} GB", flush=True)
    digests = {}
    for path in sorted(opts.dir.glob("*.safetensors")) + [opts.dir / "config.json"]:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        digests[path.name] = h.hexdigest()
    (opts.dir / "local-sha256.json").write_text(json.dumps(digests, indent=2) + "\n")
    print(f"fetched {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
