"""Check fetched checkpoint files against the recorded upstream hashes.

    uv run python verify_checkpoint.py --dir <checkpoint dir> [--truncation spread|prefix]
    uv run python verify_checkpoint.py --file <path> [--name <upstream file name>]

Exits non-zero, naming the file, on the first one whose bytes are not the
recorded ones. Standard library only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepseek_golden import upstream  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path)
    parser.add_argument("--file", type=Path)
    parser.add_argument("--name", help="the upstream name of --file, when the path's differs")
    parser.add_argument("--truncation", choices=sorted(upstream.LIST["truncations"]))
    opts = parser.parse_args()
    assert (opts.dir is None) != (opts.file is None), "give --dir or --file"
    try:
        if opts.file is not None:
            upstream.check_file(opts.file, opts.truncation, opts.name)
            print(f"{opts.name or opts.file.name}: matches the recorded hash")
        else:
            names = upstream.check_dir(opts.dir, opts.truncation)
            print(f"{len(names)} files match their recorded hashes")
    except upstream.Mismatch as error:
        sys.exit(f"REFUSED {error}")


if __name__ == "__main__":
    main()
