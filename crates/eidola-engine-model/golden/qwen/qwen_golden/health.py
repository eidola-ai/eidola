"""The health check's inputs and measures (see `health_check.py`)."""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from . import st
from .passage import TEXT

LICENSE_TOKENS = 1024
LICENSE_FLOOR = 0.95
QUESTION = "What is the capital of France? Answer with one word."
ANSWER = "Paris"
REPLY_TOKENS = 16


def _tokenizer(directory: Path) -> Tokenizer:
    return Tokenizer.from_file(str(directory / "tokenizer.json"))


def sequences(directory: Path) -> dict[str, list[int]]:
    """Token sequences of the checks, from the checkpoint's own tokenizer and
    chat template."""
    from transformers import AutoTokenizer

    tok = _tokenizer(directory)
    license_ids = tok.encode((directory / "LICENSE").read_text(), add_special_tokens=False).ids[:LICENSE_TOKENS]
    assert len(license_ids) == LICENSE_TOKENS
    chat = AutoTokenizer.from_pretrained(str(directory))
    prompt = chat.apply_chat_template(
        [{"role": "user", "content": QUESTION}], add_generation_prompt=True, enable_thinking=False, tokenize=False
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), prompt
    prompt_ids = tok.encode(prompt, add_special_tokens=False).ids
    stops = [tok.token_to_id("<|im_end|>"), tok.token_to_id("<|endoftext|>")]
    assert None not in stops
    return {
        "license.tokens": license_ids,
        "passage.tokens": tok.encode(TEXT, add_special_tokens=False).ids,
        "capital.prompt": prompt_ids,
        "capital.stop": stops,
        "capital.answer": tok.encode(ANSWER, add_special_tokens=False).ids,
    }


def accuracy(top1: list[int], tokens: list[int]) -> float:
    """The share of positions whose most likely next token is the one that follows."""
    hits = sum(top1[i] == tokens[i + 1] for i in range(len(tokens) - 1))
    return hits / (len(tokens) - 1)


@torch.no_grad()
def greedy(model, prompt: list[int], stops: list[int], device: str = "cpu") -> list[int]:
    out = model.model(input_ids=torch.tensor([prompt], device=device), use_cache=True)
    cache, reply = out.past_key_values, []
    for _ in range(REPLY_TOKENS):
        t = int(model.lm_head(out.last_hidden_state[:, -1:]).float().argmax(-1))
        reply.append(t)
        if t in stops:
            break
        out = model.model(input_ids=torch.tensor([[t]], device=device), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
    return reply


def decode(directory: Path, ids: list[int]) -> str:
    return _tokenizer(directory).decode(ids, skip_special_tokens=False)


def write(directory: Path, tensors: dict) -> None:
    out = directory / "golden" / "health.safetensors"
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_name(out.name + ".partial")
    st.write_file(tmp, tensors)
    tmp.replace(out)
