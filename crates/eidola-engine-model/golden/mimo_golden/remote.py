"""Fetch and pin the Hugging Face artifacts the goldens depend on.

Nothing here is vendored: the MiMo remote code and checkpoint files are
downloaded at a pinned revision and checked against recorded sha256 digests.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
REVISION = "2479e2d0029eca9a34cc7e7f55a121925f81908e"

REMOTE_CODE_SHA256 = {
    "modeling_mimo_v2.py": "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad",
    "configuration_mimo_v2.py": "773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e",
}


def cache_root() -> Path:
    base = os.environ.get("EIDOLA_ENGINE_CACHE")
    if base:
        return Path(base)
    return Path.home() / ".cache" / "eidola" / "engine"


def url(path: str, repo: str = REPO, revision: str = REVISION) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def _open(req: urllib.request.Request):
    for attempt in range(6):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except Exception as e:  # noqa: BLE001 - retried, then re-raised
            if attempt == 5:
                raise
            print(f"retrying {req.full_url}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_bytes(path: str, start: int | None = None, end: int | None = None) -> bytes:
    """GET a repo file, or the inclusive byte range [start, end] of it."""
    req = urllib.request.Request(url(path))
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    for attempt in range(6):
        try:
            with _open(req) as r:
                data = r.read()
            if start is not None and len(data) != end - start + 1:
                raise IOError(f"short range read {len(data)} of {end - start + 1}")
            return data
        except Exception as e:  # noqa: BLE001
            if attempt == 5:
                raise
            print(f"retrying range {path} {start}-{end}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def fetch_file(path: str, dest: Path, sha256: str | None = None) -> Path:
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_bytes(path)
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise RuntimeError(f"{path}: sha256 {digest} != pinned {sha256}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remote_code_package() -> Path:
    """Directory to put on sys.path so `import mimo_remote.modeling_mimo_v2` works."""
    root = cache_root() / "hf-remote-code" / REVISION
    pkg = root / "mimo_remote"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").touch()
    for name, digest in REMOTE_CODE_SHA256.items():
        fetch_file(name, pkg / name, digest)
    return root


def import_remote_code():
    root = str(remote_code_package())
    if root not in sys.path:
        sys.path.insert(0, root)
    from mimo_remote import configuration_mimo_v2, modeling_mimo_v2  # type: ignore

    return configuration_mimo_v2, modeling_mimo_v2


def safetensors_header(path: str) -> tuple[int, dict]:
    """(data start offset, header) of a remote safetensors file via range reads."""
    n = struct.unpack("<Q", fetch_bytes(path, 0, 7))[0]
    header = json.loads(fetch_bytes(path, 8, 8 + n - 1))
    return 8 + n, header
