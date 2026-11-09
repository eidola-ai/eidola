# Model identity

A model the Eidola server sells is identified by its **weights hash**: a SHA-256 over every file the inference engine reads from its weights directory. The hash is the model's key everywhere: in the server's catalog, in the engine deployments it pins, in the engine's own refusal of a request naming other weights, and in what the server publishes. Names are a convenience on top of it. This page specifies the hash well enough to reimplement it without Eidola's code, says what it does and does not pin, and shows how to check it against files you download yourself.

## Naming a model

A request's `model` field takes one of three forms:

- **`tag`**, such as `mimo-v2.6-flash`: whatever weights the tag resolves to in this server build. A tag matches `[A-Za-z0-9._-]+`, is at most 64 characters, and names exactly one hash at a time. Tags move only with a server release, so following a tag means accepting the release's choice.
- **`tag@sha256:<64 lowercase hex digits>`**: a pin. As in a container image reference, **the digest is authoritative**: the server serves exactly those weights, and reads the tag only as the client's label for them, whether it names that hash now, named it once, or never did. A pin keeps its weights across a release that moves its tag, for as long as the server serves them.
- **`sha256:<64 lowercase hex digits>`**: those weights, with no label.

Only full digests are accepted. The server resolves the reference to a hash as the request arrives; everything after that point works with the hash alone. A response's `model` names what was resolved: the reference as the client sent it, normalised (`tag@sha256:<hex>` for a tag, with the hash it resolved to, or for a pin; `sha256:<hex>` for a hash named alone). A reference that is malformed, names no model the server serves, or names a retired model is refused, with a full refund of the request's charge. Telling a user that a tag they pinned now names other weights is the client's job, from the catalog compiled into it.

Every model's catalog entry is a file in the source tree, `models/<family>/<label>.toml`, compiled into the server, so the catalog a server publishes is part of what its enclave measurement covers. A model is **current** or **deprecated** (still served, with a retirement date announced) while it is listed, and **retired** when a release stops serving it; a retired model is refused by name, tag or hash.

## What `/v1/models` publishes

For each model the server sells, `/v1/models` publishes:

- `sha256`, the weights hash, and `tags`, every tag that resolves to it;
- `weights`: the source repository and revision (on Hugging Face) and the per-file manifest, each file's `path`, `size` and `sha256`;
- `architecture`, the model family the engine runs it as;
- `lifecycle`: `status` (`current` or `deprecated`) and, for a deprecated model, `retire_after`;
- `deployments`: for every engine deployment the server accepts for these weights, the SHA-256 of its committed configuration and the attestation measurement the server holds every connection to it to.

Its `id` is the model's first tag, or `sha256:<hex>` when it has none.

## The weights hash

The engine's weights directory is flat: file names only, no subdirectories. The hash **covers** these files:

- every file whose name ends in `.safetensors` (the shards);
- `config.json`, `tokenizer.json`, `chat_template.jinja` and `generation_config.json`, all required;
- `model.safetensors.index.json`, when present.

Every covered name must match `[A-Za-z0-9_][A-Za-z0-9._-]*`: ASCII letters, digits, `.`, `_` and `-`, not starting with `.` or `-`. The engine refuses a directory holding a covered file named otherwise. Files that are not covered are not read.

The **manifest** is one line per covered file, `<sha256>␠␠<name>\n`: the file's SHA-256 in lowercase hex, two spaces, its name, a newline. That is exactly the output of `sha256sum`. Lines are sorted by name in byte order, which for these names is ASCII order. The **weights hash** is the SHA-256 of the manifest's bytes, in lowercase hex.

In a shell, inside the directory:

```sh
LC_ALL=C ls *.safetensors config.json model.safetensors.index.json \
    tokenizer.json chat_template.jinja generation_config.json 2>/dev/null \
  | LC_ALL=C sort | xargs sha256sum | sha256sum      # macOS: shasum -a 256
```

The name rule is what makes this reproduce the manifest byte for byte: `sha256sum` escapes a name holding a backslash or newline, `xargs` splits on whitespace and quotes, a leading `-` reads as an option, and the shell's `*` skips a leading `.`.

The published per-file manifest lets you check one file at a time, and its sizes let you check a download before hashing it. The sizes are not part of the hash.

## What the hash pins, and what it does not

The hash pins **the bytes of the weights**, and with them the tokenizer, the chat template and the default sampling settings. A node refuses a request naming any other hash before the request is admitted, and the server names the hash on every request it sends.

It does **not** pin exact behaviour. The engine's code and version, its numerics (the kernels, the order floating-point work is done in, the batch a request shares) and the sampling seed are outside it. The engine build is pinned separately, by the deployment's launch identity ([Inference engine](engine.md#what-pins-the-engine)); two engines serving the same hash run the same model, not necessarily to the same bit.

**Every quantization Eidola serves is its own set of standard-format files**: safetensors shards with their own hash and their own catalog entry, never a transform applied when the weights are loaded. So the files you can download and hash are the files the engine computes with.

## Checking it yourself

[`scripts/weights-hash.py`](../scripts/weights-hash.py) does all of this with the Python standard library (3.11 or later):

```sh
# Download every file a catalog entry lists, from its repository at its
# revision, checking each file's size and SHA-256, then the aggregate.
python3 scripts/weights-hash.py download models/<family>/<label>.toml ./weights

# Check a directory you already have against an entry.
python3 scripts/weights-hash.py verify models/<family>/<label>.toml ./weights

# Print the weights hash of a directory, as the engine computes it.
python3 scripts/weights-hash.py hash ./weights
```

The entry can also be one model's object from `/v1/models`, saved as a `.json` file, so you can check what a running server publishes rather than what the source tree says.

The directory that verifies is an ordinary Hugging Face checkpoint. Load the same files in another engine to compare: `vllm serve ./weights`, or convert them with llama.cpp's `convert_hf_to_gguf.py ./weights` (the conversion is then your transform, outside the hash). Outputs will agree in substance, not to the token: the hash pins the weights, not the engine (see above).
