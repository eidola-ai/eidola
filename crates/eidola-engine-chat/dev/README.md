# Fixture generators

Dev-only Python that produces the committed fixtures under `tests/fixtures/`. Nothing here is a build input and `cargo test` never runs it. The MiMo generators and their pinned versions (`requirements.txt`) are described in the crate's `AGENTS.md`.

## DeepSeek-V4.1

`deepseek_gen_fixtures.py` runs the reference prompt encoder and output parser that the model repository ships (`encoding/encoding.py` in [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT) and the reference `tokenizers` build, and writes `tests/fixtures/deepseek/`. The reference is read from a checkout you supply; it is not vendored.

1. Fetch the model repository's non-weight files at the revision to pin: `encoding/`, `inference/engram.py`, `tokenizer.json` and `config.json`.
2. From the crate directory, run the generator with [`uv`](https://docs.astral.sh/uv/), which installs the Python and packages pinned in the script's header and in `deepseek_gen_fixtures.py.lock`:

   ```sh
   uv run --script dev/deepseek_gen_fixtures.py \
       --model-dir <checkout> --revision <commit sha of the checkout>
   ```

3. Review the fixture diff like code, and run `cargo test -p eidola-engine-chat`.

It also runs `build_compressed_token_map` from `inference/engram.py` (that function alone, compiled out of the file, since the rest of it needs torch) for the Engram map: `token_map.json` (class count and metadata, with `engram.py`'s SHA-256) and `token_map.bin.gz` (one little-endian u32 per id: 0 opens the next class, n > 0 is class n − 1).

The revision is recorded in each fixture file's `meta`, with the SHA-256 of `encoding.py` and `tokenizer.json`. Regenerating at the recorded revision reproduces the committed files byte for byte. A new revision whose `tokenizer.json` differs also needs its hash pinned in `src/tokenizer.rs`.

After changing the script's dependencies, refresh the lock with `uv lock --script dev/deepseek_gen_fixtures.py`.

The generator executes the reference code it is pointed at: run it only on a checkout you trust.

## Qwen3.8

Two generators, run with `uv` like the DeepSeek one, each with its dependencies pinned in its header and its lock file. Both read a directory holding the non-weight files of [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (Apache-2.0) at the revision to pin: `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and `config.json`.

- `qwen_gen_template_fixtures.py` renders every case with `transformers.apply_chat_template` and encodes with the reference `tokenizers`, and writes `tests/fixtures/qwen/template_cases.json`, `template_token_ids.json` and copies of the template, generation config and tokenizer (gzip with a zero timestamp). The revision and the template's and tokenizer's SHA-256 are recorded in `meta`; a new revision whose files differ also needs their hashes pinned (`src/qwen/template.rs`, `src/tokenizer.rs`).

  ```sh
  uv run --script dev/qwen_gen_template_fixtures.py \
      --model-dir <dir> --revision <commit sha>
  ```

- `qwen_gen_vllm_fixtures.py` records what vLLM's Qwen3 output parser (`vllm/parser/qwen3.py` on its parser engine, which serves both `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`) reads from outputs the template renders and from truncated and malformed ones, and writes `tests/fixtures/qwen/vllm_parser_cases.json`. It runs those files from a vLLM checkout you supply, with the few serving-layer modules they import replaced by stand-ins, and records the commit and each file's SHA-256.

  ```sh
  uv run --script dev/qwen_gen_vllm_fixtures.py \
      --vllm-src <vllm checkout> --vllm-commit <commit sha> --model-dir <dir>
  ```

Both execute code they are pointed at (the template, vLLM's parser): run them only on files you trust. Refresh a lock with `uv lock --script <script>` after changing its dependencies.

## Qwen3.5

`qwen_gen_template_fixtures.py --format qwen3.5` generates the Qwen3.5 corpus, `tests/fixtures/qwen35/`, from a directory holding the non-weight files of [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) (Apache-2.0) at the revision to pin: `chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json` and `config.json`. Qwen3.5-0.8B ships the same template and tokenizer. The case families are Qwen3.8's, a case setting `preserve_thinking` or `reasoning_effort` recorded as refused, plus the template's reading of reasoning out of content (`inline_reasoning`); it copies `config.json` (the end-of-sequence id's file) where Qwen3.8's copies `generation_config.json`. Regenerating Qwen3.8's corpus with the default `--format qwen3.8` reproduces its committed files byte for byte.

```sh
uv run --script dev/qwen_gen_template_fixtures.py --format qwen3.5 \
    --model-dir <dir> --revision <commit sha>
```
