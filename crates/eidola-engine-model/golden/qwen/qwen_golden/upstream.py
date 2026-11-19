"""The released checkpoint's recorded bytes: every file is checked against
`upstream-sha256.json` before anything reads it.

The list holds, for the pinned revision of the repository, the size and
SHA-256 of every published file used here. A weight shard's SHA-256 (and
`tokenizer.json`'s) is its Git LFS object id from the repository's tree
listing; a small file's was computed from bytes whose Git blob id (also
recorded) equals the listing's.

A truncation here is a set of whole upstream files (each shard holds whole
layers), so it has no re-packed bytes of its own: every file of every
truncation is checked against the same upstream hash.

A size that matches says nothing about the bytes: a resumed or multi-part
download can leave a file of the right length and the wrong contents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LIST = json.loads((Path(__file__).resolve().parent.parent / "upstream-sha256.json").read_text())
REPO = LIST["repository"]
REVISION = LIST["revision"]
assert (REPO, REVISION) == ("Qwen/Qwen3.8-27B-FP8", "c3a462cc3d38cf351f3b26cbd763469eeb74ef81")

# The small files every run reads beside the weights.
SMALL = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"]
SHARDS = [f"layers-{i}.safetensors" for i in range(64)] + ["mtp.safetensors", "outside.safetensors"]
# Checkpoint layers 0-3 (three DeltaNet layers and one attention layer), the
# drafter, and the file holding the embedding, the head and the final norm.
PREFIX = [f"layers-{i}.safetensors" for i in range(4)] + ["mtp.safetensors", "outside.safetensors"]
TRUNCATIONS = {"prefix": PREFIX, "full": SHARDS + ["model.safetensors.index.json"]}


class Mismatch(Exception):
    """A file's bytes are not the recorded ones."""


def record(name: str) -> dict:
    if name not in LIST["files"]:
        raise Mismatch(f"{name}: not a recorded upstream file")
    return LIST["files"][name]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def check_file(path: Path, name: str | None = None) -> None:
    """Refuse `path` unless its bytes are the recorded ones for `name`
    (default: its own file name)."""
    rec = record(name or path.name)
    if path.stat().st_size != rec["size"]:
        raise Mismatch(f"{path.name}: {path.stat().st_size} bytes, recorded {rec['size']}")
    got = sha256_of(path)
    if got != rec["sha256"]:
        raise Mismatch(f"{path.name}: sha256 {got}, recorded {rec['sha256']}")


def files_of(truncation: str) -> list[str]:
    return SMALL + TRUNCATIONS[truncation]


def check_dir(directory: Path, truncation: str) -> list[str]:
    """Check every file `truncation` needs, and every safetensors file present,
    and return their names. A missing file, or a weight file that is not
    recorded, is refused like a wrong one."""
    names = sorted(set(files_of(truncation)) | {p.name for p in directory.glob("*.safetensors")})
    for name in names:
        path = directory / name
        if not path.exists():
            raise Mismatch(f"{name}: missing from {directory}")
        check_file(path)
    return names
