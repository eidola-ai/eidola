"""The released checkpoint's recorded bytes: every fetched file is checked
against `upstream-sha256.json` before anything reads it.

The list holds, for the pinned revision of the repository:

- `files`: size and SHA-256 of every published file used here. A weight
  shard's is its Git LFS object id from the repository's tree listing; a
  small file's was computed from bytes whose Git blob id equals the listing's.
- `truncations`: the SHA-256 of every re-packed file of each truncation in
  `truncated.py`. A re-pack is a function of the upstream shard and the
  truncation alone, and these were recorded from re-packs whose every tensor
  equals the same tensor of shards matching `files`.

A size that matches says nothing about the bytes: a resumed or multi-part
download can leave a file of the right length and the wrong contents.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import truncated

LIST = json.loads((Path(__file__).resolve().parent.parent / "upstream-sha256.json").read_text())
assert (LIST["repository"], LIST["revision"]) == (truncated.REPO, truncated.REVISION), "the hash list is for another revision"


class Mismatch(Exception):
    """A file's bytes are not the recorded ones."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def expected(name: str, truncation: str | None = None) -> str:
    """The recorded SHA-256 of a file: the upstream file, or with `truncation`
    that truncation's re-pack of the shard (small files are never re-packed)."""
    if truncation is not None and name.endswith(".safetensors"):
        table = LIST["truncations"][truncation]
        if name not in table:
            raise Mismatch(f"{name}: the {truncation} truncation has no such file")
        return table[name]
    if name not in LIST["files"]:
        raise Mismatch(f"{name}: not a recorded upstream file")
    return LIST["files"][name]["sha256"]


def check_bytes(name: str, data: bytes) -> bytes:
    want = expected(name)
    got = hashlib.sha256(data).hexdigest()
    if got != want:
        raise Mismatch(f"{name}: sha256 {got}, recorded {want}")
    return data


def check_file(path: Path, truncation: str | None = None, name: str | None = None) -> None:
    """Refuse `path` unless its bytes are the recorded ones for `name`
    (default: its own file name)."""
    name = name or path.name
    want = expected(name, truncation)
    if truncation is None and path.stat().st_size != LIST["files"][name]["size"]:
        raise Mismatch(f"{name}: {path.stat().st_size} bytes, recorded {LIST['files'][name]['size']}")
    got = sha256_of(path)
    if got != want:
        raise Mismatch(f"{name}: sha256 {got}, recorded {want}")


def check_dir(directory: Path, truncation: str | None = None) -> list[str]:
    """Check every checkpoint file in `directory` (weight shards, `config.json`,
    `tokenizer.json`, the index) and return their names. A weight file that is
    not recorded is refused like a wrong one."""
    names = sorted(p.name for p in directory.glob("*.safetensors"))
    names += [n for n in ("config.json", "tokenizer.json", "model.safetensors.index.json") if (directory / n).exists()]
    for name in names:
        check_file(directory / name, truncation)
    return names
