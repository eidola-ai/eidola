# Model catalog

One hand-written file per model the gateway sells: `models/<family>/<label>.toml` (each component `[a-z0-9][a-z0-9._-]*`; the names are for people). A model is identified by its **weights hash**, which the file holds as `sha256`, never by its file name or its tags. The gateway's build reads every file here, checks it, binds it to the engine pins in `releases/trust/engine-enclaves.json` and compiles the catalog in, so what `/v1/models` publishes is a function of this directory and the measured image. It holds one model, `mimo/mimo-v2.6-flash.toml`, pinned to the deployment under `deploy/engine/mimo-v2.6-flash/`.

The hash, its per-file manifest and how to reproduce both from the published files: [`docs/model-identity.md`](../docs/model-identity.md).

## A file

```toml
sha256 = "<the weights hash: 64 lowercase hex digits>"
architecture = "MimoV26Flash"          # an implemented eidola_common::architecture::Architecture
tags = ["mimo-v2.6-flash"]             # [A-Za-z0-9._-]+; each tag names exactly one hash
name = "MiMo-V2.6-Flash"
description = "…"

[source]
repo = "<owner>/<name>"                # on Hugging Face
revision = "<40-hex-digit commit>"

[[files]]                              # one per file the hash covers
path = "config.json"
size = 1234
sha256 = "<64 lowercase hex digits>"

[capabilities]
context_length = 1048576
tool_calling = true
reasoning = true
input_modalities = ["text"]
output_modalities = ["text"]
max_output_tokens = 32768              # below context_length; required for a token-priced row
output_budget_class = "reasoning"      # or "standard"

[price]                                # USD, as charged; `PRICING_OVERRIDES` can replace it at boot
input_per_m = 1.80
output_per_m = 5.75                    # or per_request = … alone
cached_input_per_m = 0.45              # optional: a prompt token declared cached; at most input_per_m

[lifecycle]
status = "current"                     # or "deprecated" with retire_after = "YYYY-MM-DD", or "retired"
```

The build refuses: a hash listed twice; a tag on two files; a tag outside `[A-Za-z0-9._-]+`; a `files` list that is not exactly the files the engine hashes (every `*.safetensors` shard, `config.json`, `tokenizer.json`, `chat_template.jinja`, `generation_config.json`, and `model.safetensors.index.json` when present) or whose aggregate is not `sha256`; an architecture that is not implemented; a token-priced row without `max_output_tokens`; a `cached_input_per_m` above `input_per_m` or beside `per_request`; a served (current or deprecated) row without an engine pin for its hash, or a pin without one; a retired row that is still pinned.

## Lifecycle

- **current**: served; what its tags are for.
- **deprecated**: still served, its retirement announced for `retire_after`. A release retires it; the date is a notice, not a timer.
- **retired**: not served and not pinned. The file stays so a request naming it is refused as retired rather than unknown; its tags may be moved to another file.

Moving a tag is editing two files in one change: removing it from the old row and adding it to the new one. Every quantization served is its own set of standard-format (safetensors) files with its own hash and its own file here, never a transform applied at load time.
