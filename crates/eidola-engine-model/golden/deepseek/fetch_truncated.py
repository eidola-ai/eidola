"""Fetch a truncation of the released checkpoint by HTTP range reads.

    uv run python fetch_truncated.py --dir <checkpoint dir> [--truncation prefix] [--reuse <other dir>]

`config.json`, `tokenizer.json`, and the tensors the chosen truncation of
`deepseek_golden/truncated.py` keeps (`spread`: about 14 GB; `prefix`: about 32 GB),
re-packed byte for byte into one local file per upstream shard. Every file is
checked against the recorded hashes of `upstream-sha256.json` and refused on a
mismatch, a shard already present included, so an interrupted fetch resumes
shard by shard. Engram tables are not fetched here; `real_golden.py` fetches
only the rows its tokens touch.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek_golden import truncated, upstream  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--truncation", choices=sorted(truncated.TRUNCATIONS), default="spread")
    parser.add_argument("--reuse", type=Path, help="another fetched truncation whose tensors are copied, not fetched again")
    opts = parser.parse_args()
    t = truncated.TRUNCATIONS[opts.truncation]
    opts.dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "tokenizer.json"):
        (opts.dir / name).write_bytes(upstream.check_bytes(name, truncated.get(name)))
    index_name = "model.safetensors.index.json"
    index = json.loads(upstream.check_bytes(index_name, truncated.get(index_name)))["weight_map"]
    shards = sorted({shard for name, shard in index.items() if truncated.wanted(name, t)})
    total = 0
    for shard in shards:
        out = opts.dir / shard
        if out.exists():
            upstream.check_file(out, opts.truncation)
            print(f"{shard}: present, matches its recorded hash")
            continue
        partial = out.with_suffix(".partial")
        size = truncated.fetch_shard(shard, partial, t, opts.reuse)
        # The partial file stays behind, under a name nothing loads, when refused.
        upstream.check_file(partial, opts.truncation, name=shard)
        partial.rename(out)
        total += size
        print(f"{time.strftime('%H:%M:%S')} {shard}: fetched {size / 1e9:.2f} GB", flush=True)
    digests = {name: upstream.expected(name, opts.truncation) for name in sorted(p.name for p in opts.dir.glob("*.safetensors")) + ["config.json"]}
    (opts.dir / "local-sha256.json").write_text(json.dumps(digests, indent=2) + "\n")
    print(f"fetched {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
