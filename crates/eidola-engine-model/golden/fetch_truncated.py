"""Fetch a truncated real checkpoint into the engine cache.

    uv run python fetch_truncated.py [--layers 0,1,2,5] [--no-mtp] [--out DIR]
"""

import argparse
from pathlib import Path

from mimo_golden import truncated

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(map(str, truncated.DEFAULT_LAYERS)))
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    mtp = not a.no_mtp
    out = a.out or truncated.default_dir(layers, mtp)
    print(truncated.fetch(out, layers, mtp))
