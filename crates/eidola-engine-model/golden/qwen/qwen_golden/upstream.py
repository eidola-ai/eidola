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

HERE = Path(__file__).resolve().parent.parent

# The small files every run reads beside the weights.
SMALL = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE"]

_QWEN38_SHARDS = [f"layers-{i}.safetensors" for i in range(64)] + ["mtp.safetensors", "outside.safetensors"]
_QWEN35_SHARD = "model.safetensors-00001-of-00001.safetensors"

# Per checkpoint: its recorded list, the repository and revision the list must
# name, and its truncations (sets of whole upstream files). Qwen3.8-27B-FP8's
# `prefix` is checkpoint layers 0-3 (three DeltaNet layers and one attention
# layer), the drafter, and the file holding the embedding, the head and the
# final norm. Qwen3.5's checkpoints are one shard each, so only whole.
MODELS = {
    "qwen3.8-27b": (
        "upstream-sha256.json",
        ("Qwen/Qwen3.8-27B-FP8", "c3a462cc3d38cf351f3b26cbd763469eeb74ef81"),
        {
            "prefix": [f"layers-{i}.safetensors" for i in range(4)] + ["mtp.safetensors", "outside.safetensors"],
            "full": _QWEN38_SHARDS + ["model.safetensors.index.json"],
        },
    ),
    "qwen3.5-2b": (
        "upstream-qwen3.5-2b-sha256.json",
        ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc"),
        {"full": [_QWEN35_SHARD, "model.safetensors.index.json"]},
    ),
    "qwen3.5-0.8b": (
        "upstream-qwen3.5-0.8b-sha256.json",
        ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
        {"full": [_QWEN35_SHARD, "model.safetensors.index.json"]},
    ),
}
DEFAULT_MODEL = "qwen3.8-27b"

LIST: dict = {}
REPO = REVISION = ""
TRUNCATIONS: dict[str, list[str]] = {}


def use(model: str) -> None:
    """Make `model`'s recorded list the one every check below reads."""
    global LIST, REPO, REVISION, TRUNCATIONS
    path, pinned, truncations = MODELS[model]
    LIST = json.loads((HERE / path).read_text())
    REPO, REVISION = LIST["repository"], LIST["revision"]
    assert (REPO, REVISION) == pinned, (path, REPO, REVISION)
    TRUNCATIONS = truncations


def truncation_for(layers: int | None) -> str:
    """The truncation a run of the first `layers` layers (None: all) reads."""
    if "prefix" in TRUNCATIONS and layers is not None and layers <= 4:
        return "prefix"
    return "full"


use(DEFAULT_MODEL)


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
