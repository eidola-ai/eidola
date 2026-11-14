"""HF remote-code goldens for a truncated real checkpoint (local check, not CI).

    uv run python fetch_truncated.py
    uv run python real_golden.py [--dir DIR] [--layers 0,1,2,5]

Writes `golden/hf-golden.safetensors` under the checkpoint directory: the token ids
of the text below, logits at every position, the residual stream after every
kept layer, and MTP layer 0's outputs over the shifted sequence. The Rust side
is the ignored test `tests/real_flash.rs`.
"""

import argparse
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer

from mimo_golden import hf_run, remote, truncated

# Public-domain prose and a little code (363 tokens).
TEXT = """When in the Course of human events, it becomes necessary for one people to \
dissolve the political bands which have connected them with another, and to assume \
among the powers of the earth, the separate and equal station to which the Laws of \
Nature and of Nature's God entitle them, a decent respect to the opinions of mankind \
requires that they should declare the causes which impel them to the separation.
We hold these truths to be self-evident, that all men are created equal, that they \
are endowed by their Creator with certain unalienable Rights, that among these are \
Life, Liberty and the pursuit of Happiness. That to secure these rights, Governments \
are instituted among Men, deriving their just powers from the consent of the governed. \
That whenever any Form of Government becomes destructive of these ends, it is the Right \
of the People to alter or to abolish it, and to institute new Government, laying its \
foundation on such principles and organizing its powers in such form, as to them shall \
seem most likely to effect their Safety and Happiness. Prudence, indeed, will dictate \
that Governments long established should not be changed for light and transient causes.

def fibonacci(n: int) -> list[int]:
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]

print(fibonacci(12))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=",".join(map(str, truncated.DEFAULT_LAYERS)))
    ap.add_argument("--dir", type=Path)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    d = a.dir or truncated.default_dir(layers, True)

    tok_path = remote.fetch_file("tokenizer.json", d / "tokenizer.json")
    tokens = Tokenizer.from_file(str(tok_path)).encode(TEXT, add_special_tokens=False).ids
    print(f"{len(tokens)} tokens")

    torch.set_num_threads(max(1, torch.get_num_threads()))
    ckpt = hf_run.Checkpoint(d)
    t0 = time.time()
    main_out = hf_run.run_main(ckpt, tokens, keep=layers)
    print(f"main forward {time.time() - t0:.1f}s")
    golden = {
        "tokens": torch.tensor(tokens, dtype=torch.int64),
        "logits": main_out["logits"],
        "hidden_normed": main_out["hidden_normed"],
    }
    for i in range(len(layers)):
        golden[f"hidden.{i}"] = main_out[f"hidden.{i}"]

    if ckpt.has("model.mtp.layers.0.eh_proj.weight"):
        n = len(tokens) - 1
        prev = main_out["hidden"][:n]
        t0 = time.time()
        mtp = hf_run.run_mtp(ckpt, 0, tokens[1:], prev)
        print(f"mtp forward {time.time() - t0:.1f}s")
        golden["mtp.0.tokens"] = torch.tensor(tokens[1:], dtype=torch.int64)
        golden["mtp.0.prev_hidden"] = prev.clone()
        golden["mtp.0.hidden"] = mtp["hidden"]
        golden["mtp.0.logits"] = mtp["logits"]

    out = d / "golden" / "hf-golden.safetensors"
    out.parent.mkdir(exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in golden.items()},
        str(out),
        metadata={"layers": ",".join(map(str, layers))},
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
