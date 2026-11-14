"""A truncation of the released checkpoint: a subset of layers and the first
few routed experts of each, fetched by HTTP range reads.

The kept layers still form a model: each layer with a compress ratio has its
KV source and an index source among them, and the drafter's target layers are
kept. Every tensor is the checkpoint's own bytes.
"""

from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import dataclass
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
REVISION = "08c8657f02291e11b6f6b63e0e16014f8e748518"
BASE = f"https://huggingface.co/{REPO}/resolve/{REVISION}"



@dataclass(frozen=True)
class Truncation:
    """Which checkpoint layers are kept, and how many routed experts of each
    backbone layer and of each draft block (0 draft experts: no drafter)."""

    layers: tuple[int, ...]
    backbone_experts: int
    drafter_experts: int

    @property
    def drafter(self) -> bool:
        return self.drafter_experts > 0


TRUNCATIONS = {
    # Every layer kind and the drafter, with few experts. 0: window only.
    # 1: window only + Engram. 2 / 3: ratio-2 KV source and a consumer.
    # 14 / 15: the same with Engram. 20: ratio-1 KV, index and candidate
    # source. 21: consumer. 24: its own indexer inside 20's candidates.
    # 37-39: the drafter's target layers.
    "spread": Truncation((0, 1, 2, 3, 14, 15, 20, 21, 24, 37, 38, 39), 32, 16),
    # The model's first four layers whole, every routed expert: the residual
    # streams after each are the full model's, so their sensitivity is too.
    "prefix": Truncation((0, 1, 2, 3), 384, 0),
}


def get(path: str, start: int | None = None, end: int | None = None) -> bytes:
    """Bytes `[start, end)` of a repository file (the whole file without a range)."""
    request = urllib.request.Request(f"{BASE}/{path}", headers={"User-Agent": "python-urllib"})
    if start is not None:
        request.add_header("Range", f"bytes={start}-{end - 1}")
    for attempt in range(10):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if start is not None and len(data) != end - start:
                raise OSError(f"{path}: got {len(data)} bytes for a range of {end - start}")
            return data
        except OSError as error:
            if attempt == 9:
                raise
            # Back off when the host asks for fewer requests.
            limited = isinstance(error, urllib.error.HTTPError) and error.code == 429
            time.sleep(15 * (attempt + 1) if limited else 1)
    raise AssertionError


def remote_header(shard: str) -> tuple[dict, int]:
    n = int.from_bytes(get(shard, 0, 8), "little")
    return json.loads(get(shard, 8, 8 + n)), 8 + n


def wanted(name: str, t: Truncation) -> bool:
    """Whether the truncation keeps a checkpoint tensor."""
    if name in ("embed.weight", "head.weight", "norm.weight"):
        return True
    m = re.match(r"(layers|mtp)\.(\d+)\.(.*)", name)
    if not m:
        return False
    kind, index, rest = m.group(1), int(m.group(2)), m.group(3)
    if (kind == "layers" and index not in t.layers) or (kind == "mtp" and not t.drafter):
        return False
    if rest.endswith("bias_vl") or rest.startswith("engram.embed."):
        return False
    expert = re.match(r"ffn\.experts\.(\d+)\.", rest)
    if expert:
        return int(expert.group(1)) < (t.backbone_experts if kind == "layers" else t.drafter_experts)
    return True


def _local_tensors(directory: Path | None, shard: str) -> dict[str, tuple[int, int]]:
    """Absolute byte ranges of the tensors a local re-pack of `shard` holds."""
    path = directory / shard if directory else None
    if path is None or not path.exists():
        return {}
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return {name: (8 + n + m["data_offsets"][0], 8 + n + m["data_offsets"][1]) for name, m in header.items() if name != "__metadata__"}


def fetch_shard(shard: str, out: Path, t: Truncation, reuse: Path | None = None, workers: int = 16) -> int:
    """Re-pack the wanted tensors of one upstream shard into `out`, byte for
    byte. Tensors an earlier re-pack of the same shard under `reuse` already
    holds are copied from it (their bytes are the checkpoint's); the rest are
    range reads. Returns the bytes fetched."""
    header, base = remote_header(shard)
    names = sorted(n for n in header if n != "__metadata__" and wanted(n, t))
    if not names:
        return 0
    local, offset = {}, 0
    for name in names:
        lo, hi = header[name]["data_offsets"]
        local[name] = {"dtype": header[name]["dtype"], "shape": header[name]["shape"], "data_offsets": [offset, offset + hi - lo]}
        offset += hi - lo
    blob = json.dumps(local, sort_keys=True, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    have = _local_tensors(reuse, shard)
    # Pieces of at most 32 MB, so a retry repeats little.
    pieces, copies = [], []
    for name in names:
        lo, hi = header[name]["data_offsets"]
        at = 8 + len(blob) + local[name]["data_offsets"][0]
        if name in have and have[name][1] - have[name][0] == hi - lo:
            copies.append((have[name], at))
            continue
        for start in range(lo, hi, 32 << 20):
            end = min(start + (32 << 20), hi)
            pieces.append((base + start, base + end, at + start - lo))
    with open(out, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        f.truncate(8 + len(blob) + offset)
    with open(out, "r+b") as f:
        if copies:
            with open(reuse / shard, "rb") as src:
                for (lo, hi), at in copies:
                    src.seek(lo)
                    f.seek(at)
                    f.write(src.read(hi - lo))
        with concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for at, data in pool.map(lambda p: (p[2], get(shard, p[0], p[1])), pieces):
                f.seek(at)
                f.write(data)
    return sum(end - start for start, end, _ in pieces)


def truncate_args(args: dict, t: Truncation) -> dict:
    """The reference's arguments for the kept layers as a model of their own.

    Layer indices are renumbered. The Engram hash depends on the original
    layer ids, so the caller supplies the table rows itself (`real_golden.py`)
    and only the count of Engram layers is kept here. Without the drafter's
    target layers (or its experts) the drafter is dropped."""
    a = copy.deepcopy(args)
    n, renumber = args["n_layers"], {src: i for i, src in enumerate(t.layers)}
    a["n_layers"] = len(t.layers)
    for key in ("kv_source_layers", "index_source_layers", "engram_layer_ids", "dspark_target_layer_ids"):
        a[key] = [renumber[s] for s in args[key] if s in renumber]
    drafter = t.drafter and len(a["dspark_target_layer_ids"]) == len(args["dspark_target_layer_ids"])
    assert drafter == t.drafter, "a drafter target layer was dropped"
    if not drafter:
        a.update(n_mtp_layers=0, dspark_block_size=0, dspark_target_layer_ids=[])
    a["compress_ratios"] = [args["compress_ratios"][s] for s in t.layers] + list(args["compress_ratios"][n:])[: a["n_mtp_layers"]]
    a["candidate_source_layer"] = renumber.get(args["candidate_source_layer"], -1)
    a["n_routed_experts"] = t.backbone_experts
    if drafter:
        a["dspark_n_routed_experts"] = t.drafter_experts
    return a
