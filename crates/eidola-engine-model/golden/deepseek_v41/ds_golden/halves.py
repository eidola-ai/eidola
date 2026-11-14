"""The reference model run one half at a time.

The released checkpoint does not fit one GPU, and the reference's `model.py`
builds every layer at once. A half is the unmodified `model.py` built for a
contiguous range of layers as a model of its own (`truncated.truncate_args`
renumbers them), driven from outside:

- the **encoder half** (layers `0..split`, with their Engram modules) runs the
  tokens and its last layer's output is recorded: the residual copies and the
  collapse weights the next sub-layer applies. That pair is the boundary;
- the **decoder half** (layers `split..`, the final norm and the head) is
  teacher-forced from it: a forward pre-hook on its first layer replaces the
  streams and collapse weights the model computed from its (ignored) token
  embedding with the boundary's rows for the same positions.

Each half is loaded, run for every call pattern, and freed before the other
is built. A decoder layer may only read global KV built in the decoder half,
which holds when the split is at a KV source (20 in the released model); the
causal structure makes the two halves, run for the same call pattern, the
same function as the whole model.

A call pattern is a list of `(start, end)` position ranges, each one forward
call: `[(0, n)]` is one prefill chunk, `[(0, 1), (1, 2), ...]` one token per
call.
"""

from __future__ import annotations

import gc

import torch
import torch.nn.functional as F

from . import kernel_shim, truncated
from .reference import Reference


def prefill(n: int) -> list[tuple[int, int]]:
    return [(0, n)]


def steps(n: int, first: int = 1) -> list[tuple[int, int]]:
    """A prefill chunk of `first` tokens, then one token per call."""
    return [(0, first)] + [(p, p + 1) for p in range(first, n)]


class Half:
    """Layers `layers` (a contiguous range of the checkpoint's) as a model."""

    def __init__(self, module, full_args: dict, ckpt, layers: range, tokenizer, max_seq_len: int, device: str, engram_tables=None, engram_rows=None):
        t = truncated.Truncation(tuple(layers), full_args["n_routed_experts"], 0)
        args = truncated.truncate_args(full_args, t)
        kept = [l for l in full_args["engram_layer_ids"] if l in layers]
        # Engram hashing depends on the checkpoint's layer ids, which a half
        # starting at layer 0 keeps. Table rows are the caller's to supply
        # (`engram_tables` compact tables with `engram_rows` mapping positions
        # to their rows), or the checkpoint's whole tables.
        assert not kept or layers.start == 0, "Engram layers are only supported in a half that starts at layer 0"
        args["engram_num_embeddings"] = [
            (engram_tables[l][0].size(0) if engram_tables else full_args["engram_num_embeddings"][full_args["engram_layer_ids"].index(l)]) for l in kept
        ]
        self.first = layers.start
        self.ref = Reference(module, args, tokenizer if kept else None, max_seq_len=max_seq_len, device=device)
        self.ref.place(
            ckpt,
            layer_map=tuple(layers),
            routed_experts=(t.backbone_experts, 0),
            engram_tables={l - layers.start: engram_tables[l] for l in kept} if engram_tables else None,
        )
        model = self.ref.model
        if engram_rows is not None and kept:
            rows = engram_rows.to(device)
            model.engram_hash.register_forward_hook(lambda _m, inputs, _out: rows[inputs[1] : inputs[1] + inputs[0].size(1)].unsqueeze(0))
        self.boundary_out: list[tuple[torch.Tensor, torch.Tensor]] = []
        model.layers[-1].register_forward_hook(lambda _m, _i, out: self.boundary_out.append((out[0][0].cpu().clone(), out[1][0].cpu().clone())))
        self.inject = None  # (streams, pre_mix) rows for the current call

        def teacher_force(_module, args):
            if self.inject is None:
                return None
            streams, pre_mix = self.inject
            return (streams.to(args[0].dtype).to(args[0].device).unsqueeze(0), args[1], pre_mix.to(args[2].dtype).to(args[2].device).unsqueeze(0), *args[3:])

        model.layers[0].register_forward_pre_hook(teacher_force)

    def run(self, n: int, calls, tokens=None, boundary=None, logits: bool = False) -> dict[str, torch.Tensor]:
        """Run the calls. The encoder half takes `tokens`; the decoder half
        takes the encoder's `boundary` (`streams`, `pre_mix`) and ignores its
        own embedding. Returns the recorded tensors by name, layers numbered
        as in the checkpoint, plus `boundary.streams` / `boundary.pre_mix`."""
        ref = self.ref
        ref.records.clear()
        self.boundary_out.clear()
        ids = torch.tensor([tokens if tokens is not None else [0] * n], dtype=torch.long, device=ref.device)
        for start, end in calls:
            if boundary is not None:
                self.inject = (boundary[0][start:end], boundary[1][start:end])
            ref.model.forward(ids[:, start:end], start)
        self.inject = None
        out = {}
        for key, parts in ref.records.items():
            kind, _, index = key.partition(".")
            name = f"{kind}.{int(index) + self.first}" if index else kind
            out[name] = torch.cat(parts, dim=0)
        out["boundary.streams"] = torch.cat([b[0] for b in self.boundary_out], dim=0)
        out["boundary.pre_mix"] = torch.cat([b[1] for b in self.boundary_out], dim=0)
        if logits:
            head = ref.model.head.weight
            out["logits"] = torch.stack([F.linear(row.to(head.device).float(), head).cpu() for row in out["normed"]])
        return out

    def free(self) -> None:
        """Drop the model and everything cached for it."""
        self.ref.model = None
        self.ref = None
        kernel_shim.clear_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_halves(module, full_args: dict, ckpt, tokens: list[int], split: int, patterns: dict, tokenizer, device: str, engram_tables=None, engram_rows=None, log=print) -> dict[str, dict[str, torch.Tensor]]:
    """Every pattern through the encoder half, then, with the encoder freed,
    through the decoder half teacher-forced from that pattern's boundary.
    Returns `{pattern: tensors}` with the logits of every position."""
    n = len(tokens)
    layers = full_args["n_layers"]
    encoder = Half(module, full_args, ckpt, range(0, split), tokenizer, n + 8, device, engram_tables, engram_rows)
    log(f"encoder half (layers 0-{split - 1}) loaded")
    encoded = {}
    for name, calls in patterns.items():
        encoded[name] = encoder.run(n, calls, tokens=tokens)
        log(f"encoder half: {name} done")
    encoder.free()
    decoder = Half(module, full_args, ckpt, range(split, layers), tokenizer, n + 8, device)
    log(f"decoder half (layers {split}-{layers - 1}) loaded")
    out = {}
    for name, calls in patterns.items():
        e = encoded[name]
        d = decoder.run(n, calls, boundary=(e["boundary.streams"], e["boundary.pre_mix"]), logits=True)
        d["boundary.streams"], d["boundary.pre_mix"] = e["boundary.streams"], e["boundary.pre_mix"]
        for key, value in e.items():
            if not key.startswith(("boundary.", "normed")):
                d[key] = value
        out[name] = d
        log(f"decoder half: {name} done")
    decoder.free()
    return out
