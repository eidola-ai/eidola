"""Rewrite the committed synthetic fixture and its goldens.

    uv run python make_synthetic.py --reference-dir <dir holding model.py and engram.py>
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from ds_golden import pinned_ops, st, synthetic  # noqa: E402
from ds_golden.reference import Reference, args_from_hf_config, load_reference_modules, place_weights  # noqa: E402

SHORT_PREFILL = 40
FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "deepseek_tiny"


def record(prefix: str, out: dict[str, torch.Tensor], n_layers: int) -> dict[str, tuple]:
    tensors = {}
    for i in range(n_layers):
        tensors[f"{prefix}.layer.{i}"] = st.entry(out[f"layer.{i}"])
    tensors[f"{prefix}.normed"] = st.entry(out["normed"])
    tensors[f"{prefix}.logits"] = st.entry(out["logits"])
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=FIXTURE)
    parser.add_argument("--native-topk", action="store_true", help="measure only: leave torch.topk's tie order as it is")
    parser.add_argument("--native-arithmetic", action="store_true", help="measure only: leave PyTorch's own arithmetic as it is")
    opts = parser.parse_args()

    pinned_ops.install(arithmetic=not opts.native_arithmetic, topk=not opts.native_topk)
    module = load_reference_modules(opts.reference_dir)

    config = copy.deepcopy(synthetic.CONFIG)
    text = config["text_config"]
    tokenizer = synthetic.SyntheticTokenizer(text["vocab_size"])
    # The compressed vocabulary's size is an output of the reference's own map.
    text["engram_compressed_vocab_size"] = sys.modules["engram"].build_compressed_token_map(tokenizer)[1]

    weights = synthetic.build_weights(config)
    opts.out.mkdir(parents=True, exist_ok=True)
    (opts.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    st.write_file(opts.out / "model.safetensors", weights)

    tokens = synthetic.tokens(config)
    n_layers = text["num_hidden_layers"]
    reference = Reference(module, args_from_hf_config(config), tokenizer, max_seq_len=128)
    place_weights(reference.model, st.read_file(opts.out / "model.safetensors"))

    golden = {
        "tokens": st.entry(torch.tensor(tokens, dtype=torch.int32)),
        "engram.token_map": st.entry(reference.token_map()),
    }
    # The reference's decode layout: one token per call.
    step = reference.run(tokens, prefill=1, drafter=True)
    golden.update(record("step", step, n_layers))
    for i in range(n_layers):
        golden[f"step.attn.{i}"] = st.entry(step[f"attn.{i}"])
        golden[f"step.ffn.{i}"] = st.entry(step[f"ffn.{i}"])
    golden["step.main_hidden"] = st.entry(step["main_hidden"])
    golden["step.engram_rows"] = st.entry(step["engram_rows"])
    for key in ("draft_tokens", "draft_logits", "draft_confidence"):
        golden[f"step.{key}"] = st.entry(step[key])
    # The reference's prefill layout: the whole sequence as one chunk.
    golden.update(record("prefill", reference.run(tokens, prefill=len(tokens)), n_layers))
    # A prefill chunk shorter than the window, whose layout differs from ours.
    short = reference.run(tokens[:SHORT_PREFILL], prefill=SHORT_PREFILL)
    golden["short.logits"] = st.entry(short["logits"])
    st.write_file(opts.out / "golden.safetensors", golden)
    for name in ("config.json", "model.safetensors", "golden.safetensors"):
        print(f"{name}: {(opts.out / name).stat().st_size} bytes")


if __name__ == "__main__":
    main()
