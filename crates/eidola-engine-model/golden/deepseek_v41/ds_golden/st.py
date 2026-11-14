"""A minimal safetensors reader and deterministic writer.

Tensors are raw little-endian bytes with a dtype string, so the checkpoint's
`F8_E4M3`, `F8_E8M0` and packed-FP4 `I8` tensors pass through untouched.
"""

from __future__ import annotations

import json
import mmap
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ELEMENT_SIZE = {"BF16": 2, "F32": 4, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "U8": 1, "I32": 4, "I64": 8}
TORCH_DTYPE = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "I64": torch.int64,
}
DTYPE_NAME = {v: k for k, v in TORCH_DTYPE.items()}


@dataclass
class Entry:
    dtype: str
    shape: tuple[int, ...]
    data: memoryview

    def tensor(self) -> torch.Tensor:
        """A read-only tensor over the stored bytes, without copying them."""
        if 0 in self.shape:
            return torch.empty(self.shape, dtype=TORCH_DTYPE[self.dtype])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the buffer is not writable
            raw = torch.frombuffer(self.data, dtype=torch.uint8)
        return raw.view(TORCH_DTYPE[self.dtype]).view(self.shape)


def read_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


def read_file(path: Path) -> dict[str, Entry]:
    """Every tensor of one file, backed by a read-only memory map."""
    header, base = read_header(path)
    f = open(path, "rb")
    view = memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        lo, hi = meta["data_offsets"]
        out[name] = Entry(meta["dtype"], tuple(meta["shape"]), view[base + lo : base + hi])
    return out


def read_dir(path: Path) -> dict[str, Entry]:
    out: dict[str, Entry] = {}
    for file in sorted(Path(path).glob("*.safetensors")):
        out.update(read_file(file))
    return out


def entry(t: torch.Tensor | np.ndarray, dtype: str | None = None) -> tuple[str, tuple[int, ...], bytes]:
    """(dtype, shape, bytes) for a tensor; `dtype` overrides the header string
    for byte-typed storage (`F8_E4M3`, `F8_E8M0`, `I8`)."""
    if isinstance(t, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(t))
    t = t.contiguous()
    name = dtype or DTYPE_NAME[t.dtype]
    size = ELEMENT_SIZE[name]
    raw = t.view(torch.uint8) if t.dtype.itemsize == size and t.numel() else t
    data = raw.numpy().tobytes() if t.numel() else b""
    assert len(data) == t.numel() * size, (name, t.dtype, t.shape)
    return name, tuple(t.shape), data


def write_file(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Sorted names, sorted header keys, header padded to 8 bytes: the file's
    bytes are a function of the tensors alone."""
    header, offset = {}, 0
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for name in sorted(tensors):
            f.write(tensors[name][2])
