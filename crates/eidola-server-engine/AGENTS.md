# eidola-server-engine

The process on each inference node: it serves **exactly one model** through Eidola's own engine crates (`eidola-engine`, `-chat`, `-model`, `-cpu`), in process, behind a narrow HTTP API that only Eidola's gateway calls. Requests flow client → gateway → node; TLS and attestation are the enclave shim's job, so the node speaks plain HTTP behind it. AGPL-3.0-only. Workspace conventions live in the root `AGENTS.md`; the engine's own doctrine in `crates/eidola-engine/AGENTS.md` and its siblings.

| Module | Contents |
|---|---|
| `config.rs` | The environment, read once at boot. Every variable is required; nothing is defaulted. |
| `auth.rs` | The gateway token: Argon2id-verified at boot, constant-time per request. |
| `model.rs` | The weights hash (definition below), checked before any weight is loaded; `LoadedModel`. |
| `api.rs` | The strict request subset and its validation. |
| `error.rs` | `ApiError`: full detail to the caller, category-only `Display` for logs. |
| `pipeline.rs` | Template → tokens → engine request; tokens → detokenizer → stop sequences → reasoning / tool calls. |
| `worker.rs` | The engine's step loop on its own thread; the channel bridge, admission bound, cancellation. |
| `http.rs` | Routes and response shapes. |
| `lib.rs` | `boot` (verify, load, start) and `start` (over an already verified model). |

## What the node trusts, and what it refuses

The node trusts its measured configuration (the environment; see below) and the bytes of its weights directory **only after** they hash to the configured weights hash. It trusts nothing in a request beyond what the strict subset names.

**At boot** it refuses to start (non-zero exit, a content-free message naming the variable or file) on:

- any configuration variable missing, empty or malformed;
- a gateway token that does not verify against `GATEWAY_TOKEN_HASH`, or a hash that is not Argon2id;
- `EIDOLA_ENGINE_WEIGHTS_STORAGE=verified-readonly` with any weights file (the directory, shards, semantic and chat files) not on read-only storage (`storage.rs`; checked before any weights file is opened, mapped or parsed);
- a file in the weights directory whose name is not UTF-8 (the manifest is keyed by exact names; checked from the listing, before hashing);
- a weights directory whose weights hash differs from `EIDOLA_ENGINE_WEIGHTS_SHA256` (checked before dequantising anything);
- an unpinned tokenizer or chat template (the chat crate's pins), a tokenizer larger than the model's head, or an unusable `generation_config.json`;
- sizing the model or the engine core cannot honour: `MAX_MODEL_LEN` above `max_position_embeddings`, more draft tokens than MTP layers, or a scheduler configuration `Engine::new` refuses (for example a step budget below one speculative decode row);
- `cuda` as the executor (a build without the `cuda` feature cannot parse it; a build with it refuses it until the executor exists).

**Per chat request**, in this order, each before anything is admitted:

| Check | Status | `error.type` |
|---|---|---|
| `Authorization: Bearer <gateway token>` | 401 | `authentication_error` |
| `X-Eidola-Weights-Sha256` present | 428 | `weights_hash_required` |
| … and equal to this node's weights hash (hex, any case); **checked before the body is read** | 412 | `weights_hash_mismatch` |
| body ≤ 32 MiB | 413 | `request_too_large` |
| strict subset (`deny_unknown_fields` at every level, every content-part variant included), text-only content, `tool_choice` `auto`/`none`, ≤ 4 non-empty stops of ≤ 256 bytes each, `max_completion_tokens` ≥ 1, valid `cache_key` | 400 | `invalid_request_error` |
| `model` is the configured model id | 404 | `model_not_found` |
| an admission slot is free | 503 | `overloaded` |
| template accepts the messages; sampling parameters valid (`SamplingParams::new`) | 400 | `invalid_request_error` |
| prompt + 1 fits `MAX_MODEL_LEN` and the KV pools | 400 | `context_length_exceeded` |

The authentication check is about money, not privacy: it keeps anyone but the gateway (which bills) from spending the node's compute. No privacy property depends on who can call the node. Authentication precedes the weights check, so an unauthenticated caller learns nothing about the weights.

The weights-hash check is redundant with the node's measurement (which covers the weights) but early, cheap and independent: a gateway sends the hash from its own compiled-in catalog, never from placement data, so a misregistered node refuses traffic rather than serving the wrong weights. `GET /v1/engine/info` reports the verified hash so a misregistration is visible before any traffic.

## The weights hash

The node's weights hash identifies every file it reads from the weights directory:

- every `*.safetensors` file in the directory;
- `config.json`, and `model.safetensors.index.json` when present (the model crate's `SEMANTIC_FILES`);
- `tokenizer.json`, `chat_template.jinja` and `generation_config.json` (the chat artifacts). The tokenizer and template are also pinned by the chat crate; `generation_config.json` is not, and it decides the EOS ids and default sampling, so it must be covered here.

The **manifest** is one line per file, `<sha256, lowercase hex>␠␠<file name>\n` (exactly `sha256sum`'s output), sorted by file name in byte order. The weights hash is the SHA-256 of the manifest's bytes, lowercase hex. A file name containing a newline, or any name in the directory that is not UTF-8, is refused (the model crate's `WeightSet` also refuses a non-UTF-8 shard name rather than converting it lossily, so distinct names can never share a manifest entry). Offline, inside the directory:

```sh
LC_ALL=C ls *.safetensors config.json model.safetensors.index.json \
    tokenizer.json chat_template.jinja generation_config.json 2>/dev/null \
  | LC_ALL=C sort | xargs sha256sum | sha256sum      # macOS: shasum -a 256
```

Every byte used is a byte hashed: shards are memory-mapped once and both hashed and loaded from that mapping; the other files are read once and parsed from the hashed bytes. `LoadedModel` can only be built by `LoadedModel::load`, which checks the hash first, and `start` re-checks a loaded model's hash and storage mode against the configuration.

**What binds the bytes after boot is the storage, not this process.** The hash is checked once; the shards stay memory-mapped (shared, file-backed), and routed experts are dequantised from them on later forwards, so a file modified in place after the check would change the model while `/v1/engine/info` still reports the verified hash. Copying hundreds of gigabytes into private memory is not an option, so the binding is the deployment's: in production the weights directory is a dm-verity volume, mounted read-only, whose every read the kernel checks against the root hash in the measured configuration, so no read can return bytes other than those the hash covered. The process does not assume that: `EIDOLA_ENGINE_WEIGHTS_STORAGE` is part of the measured configuration, and `verified-readonly` refuses deployments that clearly are not immutable, **before any weights file is opened, mapped or parsed** (the directory, then each file named in its listing; only metadata is read):

- every platform: each path's mount is read-only (`statvfs` `ST_RDONLY`). That flag is per mount: a read-only bind mount can sit over a filesystem still writable through another mount of the same superblock, so on its own it is dev-grade, and macOS (development only) stops here;
- Linux: each path also resolves through `/proc/self/mountinfo` (its device id, and the longest mount point containing its canonical path, the last such mount winning) to a mount whose mount options **and superblock options** both contain `ro`. A filesystem on a dm-verity device always has a read-only superblock, so production passes; a read-only bind over a writable filesystem is refused.

Verified in containers: a superblock-read-only tmpfs is accepted, while a read-only bind mount of a writable filesystem and Docker's `--tmpfs …:ro` (a read-only mount over a writable tmpfs superblock) are refused; on macOS a read-only disk image is accepted and an ordinary directory refused. What this does not establish: that the bytes are the right ones (the weights hash does that, once), or that a privileged process in the VM will not remount the superblock read-write later; dm-verity is what binds every read. In turn `dev-writable` boots anywhere but says so in `/v1/engine/info` (`weights_storage`) and in `/healthz`'s body. A gateway pins the production configuration, so it never routes to a `dev-writable` node.

## Wire contract

`POST /v1/chat/completions` accepts the server's strict request subset (`eidola-server/src/types.rs`: `model`, `messages`, `max_completion_tokens`, `temperature`, `top_p`, `stream`, `stream_options.include_usage`, `stop`, `tools`, `tool_choice`, with the same nullability and nested strictness) plus `cache_key`. Keep the two in step: a field the gateway starts forwarding must be added here, or every such request is refused.

- **Content parts** are stricter than the server's: every variant denies unknown fields. The template renders parts from the raw body and treats an `image_url`, `image`, `audio` or `video` key on any part as multimodal content, so a key the strict type does not name must never reach it. `image_url` parts are refused outright (text-only model).
- **`tool_choice`**: absent or `auto` offers the tools and parses calls; `none` renders the prompt **without** the tool definitions (the model is never offered a tool it may not call) and parses nothing, so it changes the prompt, and with it the reusable cache prefix, relative to the same request under `auto`; `required` and named functions are refused (no constrained decoding). App-core's chat path never sets `tool_choice`, but its local inference proxy relays whatever a local caller sends, so `none` is reachable and is supported exactly rather than refused.
- **`cache_key`**: 32 bytes, base64url without padding. Decoded at parse time, the text scrubbed, and turned into the engine salt `HMAC-SHA256(boot_key, "eidola/kv/v1" ‖ key)` under the core's per-boot key (`SaltDeriver`). No key ⇒ `CacheScope::Private`: a fresh salt, nothing reusable by any other request.
- **Messages and tools render from the body as sent.** The strict serde types only validate; the prompt is rendered from the same bytes parsed by the chat crate's order-preserving JSON parser, because `serde_json::Value` would sort keys and round big integers that the template prints.
- **Sampling**: `temperature` and `top_p` from the request, else `generation_config.json`'s (as vLLM applies a model's generation config), else 1.0; `top_k` from the generation config; a fresh random seed per request.
- **`max_completion_tokens`**: clamped to the room left in `MAX_MODEL_LEN`; absent means that room.
- **Stop sequences** are matched here on the decoded text (special tokens skipped), before reasoning/tool parsing, by one KMP automaton per sequence (at most 4 sequences of at most 256 bytes, `api::MAX_STOP_BYTES`: stops are delimiters, Eidola's chat path sends none and its local proxy relays only a local caller's, and the cap bounds the matcher's tables and held-back text to a few kilobytes per request, which an uncapped 32 MiB body would amplify to hundreds of MiB): only the longest suffix that is a prefix of some sequence is held back, so text that cannot begin one is released at once, and the work is linear in the text however long a sequence is. The first occurrence to complete wins; on a match the text before it is released, the sequence and everything after it dropped, `finish_reason` is `stop` (or `tool_calls` if calls were emitted), and the engine request is cancelled. `completion_tokens` counts through the token that completed the match.
- **Responses** are OpenAI shapes: `message.reasoning_content` / `content` (null when empty) / `tool_calls`; streaming sends a role chunk at once, one chunk per non-empty delta (`reasoning_content`, `content`, `tool_calls` with `index`), a finish chunk with an empty delta, a usage chunk with `choices: []` when `stream_options.include_usage` (the gateway always sets it), then `[DONE]`. A mid-stream failure ends the stream with an `event: error` frame.
- **Usage** includes `prompt_tokens_details.cached_tokens`, the prompt positions served from the prefix cache. It is for the client's information only: nothing here prices, and the gateway's charge never depends on it.

`GET /v1/engine/info` (gateway token, no weights header) returns `model`, `weights_sha256`, `weights_storage`, `executor`, and `build` (`crate`, `version`, and `git_sha`: the `EIDOLA_GIT_SHA` value the build was given at compile time, `null` otherwise; nothing is read from the build environment implicitly). `GET /healthz` is unauthenticated and content-free: `200 ok` while the engine thread runs (`ok; weights-storage=dev-writable` on a development node), `503` after it stopped.

## Concurrency

The executor seam is synchronous, so one dedicated thread owns the `Engine` and runs its step loop. HTTP tasks reach it over channels: commands (`Submit`, `Cancel`) in, one bounded event channel per request out (`worker::EVENT_BUFFER` = 256 events, each one step's few tokens; the engine thread never blocks on a reader). A reader that has gone away, or has stopped reading long enough to fill its buffer, is cut off at the next send: the request is cancelled in the engine (no more compute) and its channel closed, so the reader gets an error after draining. An SSE response drains as fast as its socket accepts data, and kernel socket buffers absorb far more than 256 events, so only a reader that has truly stopped is cut off; a non-streaming response drains in process. Between steps the thread drains every pending command; when idle it sleeps until a command arrives or the one-second sweep is due, and sweeps the prefix cache so expired KV is zeroed without traffic.

- **Admission** is bounded by `EIDOLA_ENGINE_MAX_REQUESTS` (rendering, tokenizing, running or queued). A permit is taken before rendering or tokenizing and is **owned by the work it bounds** for that work's whole life: it moves into the blocking preparation task (which runs to completion even when a disconnect drops the handler), comes back with the prepared request, travels with the submission, and is then shared between the engine's record of the request and the response's guard, which lives exactly as long as the request's receiver: the slot is released only when the request has left the engine **and** its buffered output has been drained or dropped. Buffered output is therefore bounded by `EIDOLA_ENGINE_MAX_REQUESTS × EVENT_BUFFER` events. With none free the request is refused at once (`overloaded`); nothing queues unboundedly in front of the engine. Rendering and tokenization run on the blocking pool.
- **Cancellation**: each request holds a guard that sends `Cancel` when dropped, so a client disconnect (the SSE stream or the handler future dropped) releases the sequence before the engine's next step, including a request still waiting for a seat. Independently, the engine thread cancels a request whose reader has gone away the next time it has output for it.
- **Failure**: an `ExecutorError` is fatal (the host cannot know which writes landed). **Any** exit of the engine thread (that, a panic, every handle dropped) is fatal, structurally: the thread owns an `ExitSignal` whose drop (on return or unwind alike) turns health `503` and resolves `Node::engine_stopped`; `serve` treats that receiver resolving in any way, signal or dropped sender, as fatal and returns an error at once, without draining (graceful draining is only for SIGINT/SIGTERM); `run` then tears the runtime down within `TEARDOWN_LIMIT`, so open connections (even one stalled mid-upload) cannot keep the process alive, and `main` exits non-zero. In-flight requests see their channel close and get an error.

## Content-free by construction

Prompts, outputs, cache keys and salts are content; so is anything derived from one request (its token counts, its error details). None of it is logged:

- `ApiError`'s `Display` (the only rendering that reaches logs) is the error category alone; the detail goes only into the caller's response body (`error.rs`, tested).
- Configuration and boot errors name variables, files and limits, never values; the gateway token is held only as its SHA-256 and prints as a redaction marker.
- The engine core's secret types (`CacheKey`, `EngineSalt`) print redacted and are zeroed on drop. The cache key is decoded into a fixed stack buffer handed to `CacheKey::from_buffer`, which copies it into the zeroizing allocation and scrubs the buffer; the key's JSON text (the serde `String` and the order-preserving parse's copy) is scrubbed too. Not reachable from here: the body's bytes in the HTTP stack's shared read buffers, and serde_json's scratch buffer when the key's JSON string uses escapes. The gateway token is scrubbed after boot verification, but stays in the process environment (removing a variable is `unsafe` in this edition).
- `ValidRequest` prints a marker, not its fields.
- Per-request logging is limited to refusal categories at `debug` and failures at `warn`, with no sizes.

**Metrics**: `worker::Stats` keeps content-free counters (submitted, finished, cancelled, rejected, in-engine, engine steps, preemptions, drafts proposed and accepted). They are in-process only today (the tests read them); exporting them over OTLP like the server's telemetry is not wired yet. Aggregate token totals are deliberately not counted: at low traffic a short window's total is one request's size.

## The executor seam

`EIDOLA_ENGINE_EXECUTOR` selects the executor; `lib.rs::start` builds it on the engine thread and the rest of the node is executor-agnostic (`worker::spawn` is generic over `Executor`). `cpu` is `eidola-engine-cpu`'s f32 reference executor with one step shape (`MAX_SEQS` × `MAX_BATCHED_TOKENS`, unpadded) and MTP drafting over `DRAFT_TOKENS` depths. `cuda` is a typed placeholder behind the `cuda` feature: adding the CUDA executor means implementing its arm in `start` (its bucket ladder, graph capture, device memory) and nothing else here.

## Configuration

All required; an empty value counts as missing.

| Variable | Meaning |
|---|---|
| `EIDOLA_ENGINE_MODEL_ID` | The one model id served. |
| `EIDOLA_ENGINE_WEIGHTS_DIR` | Weights and chat artifacts. |
| `EIDOLA_ENGINE_WEIGHTS_SHA256` | The expected weights hash (64 hex digits). |
| `EIDOLA_ENGINE_WEIGHTS_STORAGE` | `verified-readonly` (production: read-only, kernel-verified mount; checked) or `dev-writable` (reported in info and health). |
| `GATEWAY_TOKEN` / `GATEWAY_TOKEN_HASH` | The gateway's bearer (secret) and its Argon2id hash (measured; make one with `cargo run -p hash-secret`). |
| `EIDOLA_ENGINE_EXECUTOR` | `cpu` (or `cuda` in a `cuda` build). |
| `EIDOLA_ENGINE_BIND_ADDR` | `host:port`. |
| `EIDOLA_ENGINE_KV_BLOCK_SIZE`, `EIDOLA_ENGINE_KV_BLOCKS` | KV geometry (blocks per group, including the null block). |
| `EIDOLA_ENGINE_MAX_MODEL_LEN` | Longest sequence; ≤ `max_position_embeddings`. |
| `EIDOLA_ENGINE_MAX_SEQS`, `EIDOLA_ENGINE_MAX_BATCHED_TOKENS`, `EIDOLA_ENGINE_MAX_PREFILL_CHUNK` | Scheduler step limits. |
| `EIDOLA_ENGINE_DRAFT_TOKENS` | Speculative width `k` (0 disables); ≤ the model's MTP layers. |
| `EIDOLA_ENGINE_MAX_REQUESTS` | Admission bound. |
| `EIDOLA_ENGINE_PREFIX_CACHE` | `true` / `false`. |
| `EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS`, `EIDOLA_ENGINE_CACHE_MAX_AGE_SECS` | Prefix-cache retention bounds (idle ≤ max age). |

`RUST_LOG` filters logs (default `info`).

## Running it locally

`just run engine-server` builds the synthetic dev model into `target/engine-dev-model` (`examples/dev_model.rs`), prints its weights hash, and serves it on `127.0.0.1:8090` with gateway token `dev-gateway-token`:

```sh
W=$(curl -s -H 'Authorization: Bearer dev-gateway-token' 127.0.0.1:8090/v1/engine/info | jq -r .weights_sha256)
curl -N 127.0.0.1:8090/v1/chat/completions -H 'Authorization: Bearer dev-gateway-token' \
  -H "X-Eidola-Weights-Sha256: $W" -H 'content-type: application/json' \
  -d '{"model":"mimo-dev","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":16,"stream":true}'
```

**The dev model** is the model crate's committed synthetic fixture with `embed_tokens` and `lm_head` widened from 256 rows to MiMo's padded 152,576 (the fixture's rows first, then deterministic random rows at the same scale), so the real pinned tokenizer and template can drive it; every other tensor is the fixture's. Its weights are random: the output is gibberish, and nothing about it is a quality signal. A built directory records a fingerprint of its inputs (`.fixture-fingerprint`: SHA-256 over a generator version, the generator's own source, and every input file); a directory whose fingerprint does not match is rebuilt beside it and swapped in by rename, so neither the tests nor `just run engine-server` can use a stale model.

## Tests

`cargo test -p eidola-server-engine` (about 5 s after the first build; no network, no GPU). `tests/server.rs` runs the real node in process on a loopback port, over the dev model (built into the target directory when its input fingerprint changes, loaded once per test binary) and the CPU executor:

- gateway token required (chat and info), checked before the weights hash; health unauthenticated;
- weights-hash mismatch and absence refused on the header alone (an unparseable body gives the same answer), with the engine's counters untouched;
- wrong model; the strict subset refusing unknown fields at every level, images, unsupported `tool_choice`, bad stops, sampling and cache keys, without reaching the engine, and accepting the body shape the gateway forwards today (explicit nulls included);
- non-streaming output equal, token for token, to the engine core driven directly over the same CPU executor (without speculation, while the node speculates with two MTP depths), and the chat layer's decoding of those tokens; usage arithmetic;
- streaming chunk shapes, the usage chunk with and without `include_usage`, and text equal to non-streaming;
- stop sequences (streaming and not) truncating at the first match and cancelling the engine request;
- client disconnect cancelling a running request and a request still queued for a seat (the latter only the guard can cancel);
- the admission bound refusing with `overloaded` and recovering, and a client disconnect during preparation keeping the slot until the abandoned preparation ends;
- `tool_choice: none` rendering the same prompt as no tools; a multimodal key (`image_url`, `image`, `audio`, `video`, …) on any content part refused;
- an injected executor panic (`tests/engine_exit.rs`, over the core's mock executor) turning health unhealthy, closing the in-flight request's channel, releasing its permit, and ending `serve` with an error;
- `cache_key`: the same key reports whole cached blocks with identical output; another key, and no key, report none;
- tool definitions and tool-call history with string arguments rendering, and non-object history arguments refused;
- stop sequences over the byte cap refused (alone or among four) and four at exactly the cap accepted; a long stop sequence not delaying the stream; the stop matcher against a naive reference on thousands of random stop sets and splits;
- `verified-readonly` refusing a writable directory before parsing an unparseable shard in it, and a model loaded under the other mode; the mountinfo parser and resolution on captured samples (a read-only bind over a writable superblock refused, read-only superblocks accepted, a read-write mount refused, shadowing, device and component-prefix matching, escapes); `a_read_only_mount_is_accepted` and `a_read_only_bind_of_a_writable_filesystem_is_refused` run where `EIDOLA_TEST_READ_ONLY_DIR` / `EIDOLA_TEST_READ_ONLY_BIND_DIR` name such mounts (a privileged container: `mount -t tmpfs -o ro none <dir>`, and a `:ro` bind);
- non-UTF-8 directory entries refused (the model crate's Linux-only test covers its own refusal);
- a reader that stops reading cut off after `EVENT_BUFFER` events, with no compute after it, and its slot held until its buffered output is dropped (`tests/backpressure.rs`, over the core's mock executor);
- an engine panic ending a real child process non-zero within the teardown bound while a request is stalled mid-upload (`tests/engine_exit.rs`);
- a stale dev model rebuilt, a current one reused;
- prompts beyond the model length; a full `boot` from the directory; boot refusing a wrong expected hash (before loading) and a loaded model's hash mismatching the configuration; every configuration variable's absence and malformations; sizing the model or the core cannot honour.

Each of these was checked to fail under a deliberate bug: no cancel on guard drop (the queued-disconnect test), no cancel on either path, a dropped first output token, a different prompt rendering, a salt not derived from the key, the weights check moved after body parsing, the permit kept by the handler instead of the preparation task, the exit signal skipped on panic, `serve` reacting only to a sent stop signal, tools rendered under `tool_choice: none`, content parts accepting unknown fields, the read-only check skipped, graceful draining on an engine stop, the stop matcher holding everything, a cached dev model reused whatever its fingerprint, `from_buffer` not scrubbing, the stop byte cap removed or off by one, the storage check moved after opening, the superblock check dropped, non-UTF-8 names converted lossily (in this crate, and in the model crate on Linux), an unbounded event buffer, the permit released when the engine finishes rather than with the response. Keep that true when changing the tests.
