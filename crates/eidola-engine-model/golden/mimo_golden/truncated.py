"""Fetch a truncated MiMo-V2.6-Flash-MOPD checkpoint by HTTP range reads.

Only the tensors of the chosen layers (plus embeddings, final norm, head, and
optionally the MTP layers) are downloaded. They are re-packed, byte for byte,
into one local safetensors file per upstream shard, next to the upstream
`config.json` and an index whose metadata (including `tp_size`) is copied
from upstream. `local-sha256.json` records the digests of the re-packed files
so later runs can check the cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import remote

# Layer 0: global, dense FFN. Layers 1 and 2: sliding-window MoE. Layer 5:
# global MoE, whose fused-QKV chunks (3392 rows) are padded in the scale grid.
DEFAULT_LAYERS = [0, 1, 2, 5]
TOP_LEVEL = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]


def default_dir(layers: list[int], mtp: bool) -> Path:
    tag = "-".join(str(i) for i in layers) + ("-mtp" if mtp else "")
    return remote.cache_root() / "mimo-v2.6-flash-mopd" / remote.REVISION[:12] / f"layers-{tag}"


def wanted(name: str, layers: list[int], mtp: bool) -> bool:
    if name in TOP_LEVEL:
        return True
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1)) in layers
    return mtp and name.startswith("model.mtp.layers.")


def fetch(out: Path, layers: list[int], mtp: bool, workers: int = 16) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    done = out / "local-sha256.json"
    if done.exists():
        print(f"already fetched: {out}")
        return out

    remote.fetch_file("config.json", out / "config.json")
    index = json.loads(remote.fetch_bytes("model.safetensors.index.json"))
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, f in index["weight_map"].items():
        if wanted(name, layers, mtp):
            by_file[f].append(name)

    def one_shard(fname: str) -> tuple[str, str]:
        dest = out / fname
        if dest.exists():
            return fname, _sha256(dest)
        data_start, header = remote.safetensors_header(fname)
        names = sorted(by_file[fname], key=lambda n: header[n]["data_offsets"][0])
        new_header = {}
        blobs = []
        offset = 0
        for n in names:
            a, b = header[n]["data_offsets"]
            blobs.append(remote.fetch_bytes(fname, data_start + a, data_start + b - 1))
            new_header[n] = {
                "dtype": header[n]["dtype"],
                "shape": header[n]["shape"],
                "data_offsets": [offset, offset + (b - a)],
            }
            offset += b - a
        if "__metadata__" in header:
            new_header["__metadata__"] = header["__metadata__"]
        hjson = json.dumps(new_header, separators=(",", ":")).encode()
        hjson += b" " * ((8 - len(hjson) % 8) % 8)
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            for blob in blobs:
                f.write(blob)
        tmp.replace(dest)
        print(f"{fname}: {len(names)} tensors, {offset / 1e6:.1f} MB", flush=True)
        return fname, _sha256(dest)

    with ThreadPoolExecutor(workers) as pool:
        digests = dict(pool.map(one_shard, sorted(by_file)))

    new_map = {n: f for f, ns in by_file.items() for n in ns}
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": index.get("metadata", {}), "weight_map": new_map}, indent=1)
    )
    (out / "truncation.json").write_text(
        json.dumps(
            {"repo": remote.REPO, "revision": remote.REVISION, "layers": layers, "mtp": mtp},
            indent=1,
        )
    )
    done.write_text(json.dumps(dict(sorted(digests.items())), indent=1))
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
