# Fixture generators

Dev-only Python that produces the committed fixtures under `tests/fixtures/`. Nothing here is a build input and `cargo test` never runs it. The MiMo generators and their pinned versions (`requirements.txt`) are described in the crate's `AGENTS.md`.

## DeepSeek-V4.1

`gen_deepseek_fixtures.py` runs the reference prompt encoder and output parser that the model repository ships (`encoding/encoding.py` in [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT) and the reference `tokenizers` build, and writes `tests/fixtures/deepseek_v4_1/`. The reference is read from a checkout you supply; it is not vendored.

1. Fetch the model repository's non-weight files at the revision to pin: `encoding/`, `tokenizer.json` and `config.json`.
2. From the crate directory, run the generator with [`uv`](https://docs.astral.sh/uv/), which installs the Python and packages pinned in the script's header and in `gen_deepseek_fixtures.py.lock`:

   ```sh
   uv run --script dev/gen_deepseek_fixtures.py \
       --model-dir <checkout> --revision <commit sha of the checkout>
   ```

3. Review the fixture diff like code, and run `cargo test -p eidola-engine-chat`.

The revision is recorded in each fixture file's `meta`, with the SHA-256 of `encoding.py` and `tokenizer.json`. Regenerating at the recorded revision reproduces the committed files byte for byte. A new revision whose `tokenizer.json` differs also needs its hash pinned in `src/tokenizer.rs`.

After changing the script's dependencies, refresh the lock with `uv lock --script dev/gen_deepseek_fixtures.py`.

The generator executes the reference code it is pointed at: run it only on a checkout you trust.
