# Model identity

## Why model names aren't enough

If you have used hosted AI for any length of time, you have probably felt a quiet doubt: you pick a model by name, come to rely on how it writes or reasons, and one day it feels different. A provider can quietly swap in a smaller quantization, update a checkpoint, alter a prompt template, or route overflow traffic to another model — and a name alone gives you no way to tell.

Even with hardware-attested enclaves running open-source code, a model name still leaves a gap: *which weights is that attested engine actually loading?*

In a plain chat app with no external capabilities, that gap is a real but bounded risk; once a model can wield tools and act on your behalf, the stakes change completely. Even tiny changes to a model's weights can substantially alter its behavior — and a service free to swap in different weights than the ones you evaluated leaves you exposed to broad or even targeted manipulation.

Eidola closes that gap by identifying every model by its **weights hash** — a single cryptographic fingerprint (SHA-256) computed over the exact open-weight files the inference engine loads. Human-readable names are just convenience labels on top of that hash. Because Eidola sells private compute over open-weight models rather than proprietary AI, that fingerprint points directly to public files anyone can download, inspect, and evaluate on their own hardware or another service.

## Tags and pinned weights

Every model in Eidola's catalog is compiled into both the gateway (making the catalog part of its measured enclave identity) and your app. When you choose a hosted model, you can reference it in two ways:

- **Follow a tag** (for example, `qwen3.8-27b`). A tag points to one specific weights hash in your current Eidola release. Tags never move silently on the server behind your back — they change only with a new, attested release. When a tag you follow moves to newer weights, the app notifies you so you know what changed.
- **Pin exact weights** (`tag@sha256:<hash>`, or the bare `sha256:<hash>`). As with a container image digest, the hash is authoritative and the tag is just your label. A pinned conversation stays on those exact weights even if a future release moves the tag, while offering a one-click option to switch when newer weights are available.

Across the entire serving path, identity is enforced structurally:

- **One answer, one set of weights.** Even when you follow a tag, your app resolves it to its current weights hash before sending the turn — and sends that exact hash on every follow-up tool call or continuation — so a single answer never mixes two models. The app records which weights hash produced each answer.
- **Engines refuse mismatches.** Every inference node hashes its read-only weights directory on startup, and the gateway sends the requested weights hash on every call. If a node does not hold those exact weights, it refuses the request before doing any compute.
- **No silent substitutions.** Every model has an explicit lifecycle: **current**, **deprecated** (still served, with an announced retirement date shown in the app), or **retired**. Once weights are retired, requests naming them are refused rather than quietly routed to something else.

## What the hash does and doesn't cover

**What it pins:** the exact bytes of the model's weight shards, along with its configuration, tokenizer, and any chat template and default sampling settings. Every quantization Eidola serves is stored as standard-format `safetensors` files with its own weights hash and catalog entry, never as a hidden transform applied when weights are loaded. The files you can download and hash are the exact files the engine computes with.

**What it does not pin:** identical token-by-token output across different software or hardware. The weights hash pins *the model files*; the inference engine's code and compiled kernels are pinned separately by each [engine deployment's measurement](architecture.md#pinning-and-attestation). Loading the same verified weights in another engine will produce outputs that agree in substance, though differences in floating-point execution and sampling mean they won't match to the exact token.

## How the hash is computed

The weights hash is deliberately simple to compute without Eidola's code:

1. Collect the covered files at the top level of the model's weights directory: every `.safetensors` shard, plus `config.json`, `tokenizer.json`, and — when present — `chat_template.jinja`, `generation_config.json`, and `model.safetensors.index.json`. Subdirectories are not read by the engine and are excluded.
2. Compute the lowercase hex SHA-256 of each covered file and build a manifest with one line per file in standard `sha256sum` format: `<sha256>  <filename>\n` (two spaces between hash and name), sorted by filename in ASCII byte order. (Filenames are restricted to ASCII letters, digits, `.`, `_`, and `-`, never starting with `.` or `-`, so standard shell tools reproduce the manifest identically.)
3. Take the lowercase hex SHA-256 of that manifest's bytes. That digest is the **weights hash**.

The gateway publishes each served model's weights hash, upstream repository and revision, per-file hashes and sizes, lifecycle status, and pinned engine measurements on `/v1/models`. Inside Eidola, the model picker shows the hash and source repository directly — and on the command line, `eidola model show <tag>` prints the full manifest and any update or retirement notices.

## Checking weights yourself

To check a model independently, you don't need to trust Eidola's app or website — you can fetch the files directly from their upstream Hugging Face repository at the published revision and recompute the hash yourself.

The repository includes [`scripts/weights-hash.py`](../scripts/weights-hash.py), which uses only the Python 3.11+ standard library:

```sh
# Download every file in a catalog entry from its upstream repo and verify each hash
python3 scripts/weights-hash.py download models/qwen/qwen3.8-27b.toml ./weights

# Verify a weights directory you already have against a catalog file or /v1/models entry
python3 scripts/weights-hash.py verify models/qwen/qwen3.8-27b.toml ./weights

# Compute the weights hash of any local checkpoint directory
python3 scripts/weights-hash.py hash ./weights
```

You can also pass a single model's JSON object saved from `/v1/models` in place of the catalog file to verify what a live gateway advertises. The resulting directory is an ordinary Hugging Face checkpoint — you can load those exact same files in another engine on your own hardware or rented compute and compare for yourself.
