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
| strict subset (`deny_unknown_fields`), text-only content, `tool_choice` `auto`/`none`, ≤ 4 non-empty stops, `max_completion_tokens` ≥ 1, valid `cache_key` | 400 | `invalid_request_error` |
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

The **manifest** is one line per file, `<sha256, lowercase hex>␠␠<file name>\n` (exactly `sha256sum`'s output), sorted by file name in byte order. The weights hash is the SHA-256 of the manifest's bytes, lowercase hex. A file name containing a newline is refused. Offline, inside the directory:

```sh
LC_ALL=C ls *.safetensors config.json model.safetensors.index.json \
    tokenizer.json chat_template.jinja generation_config.json 2>/dev/null \
  | LC_ALL=C sort | xargs sha256sum | sha256sum      # macOS: shasum -a 256
```

Every byte used is a byte hashed: shards are memory-mapped once and both hashed and loaded from that mapping; the other files are read once and parsed from the hashed bytes. `LoadedModel` can only be built by `LoadedModel::load`, which checks the hash first, and `start` re-checks a loaded model's hash against the configuration.

## Wire contract

`POST /v1/chat/completions` accepts the server's strict request subset (`eidola-server/src/types.rs`: `model`, `messages`, `max_completion_tokens`, `temperature`, `top_p`, `stream`, `stream_options.include_usage`, `stop`, `tools`, `tool_choice`, with the same nullability and nested strictness) plus `cache_key`. Keep the two in step: a field the gateway starts forwarding must be added here, or every such request is refused.

- **`cache_key`**: 32 bytes, base64url without padding. Decoded at parse time, the text scrubbed, and turned into the engine salt `HMAC-SHA256(boot_key, "eidola/kv/v1" ‖ key)` under the core's per-boot key (`SaltDeriver`). No key ⇒ `CacheScope::Private`: a fresh salt, nothing reusable by any other request.
- **Messages and tools render from the body as sent.** The strict serde types only validate; the prompt is rendered from the same bytes parsed by the chat crate's order-preserving JSON parser, because `serde_json::Value` would sort keys and round big integers that the template prints.
- **Sampling**: `temperature` and `top_p` from the request, else `generation_config.json`'s (as vLLM applies a model's generation config), else 1.0; `top_k` from the generation config; a fresh random seed per request.
- **`max_completion_tokens`**: clamped to the room left in `MAX_MODEL_LEN`; absent means that room.
- **Stop sequences** are matched here on the decoded text (special tokens skipped), before reasoning/tool parsing: text that could begin a sequence is held back; on a match the text before it is released, the sequence and everything after it dropped, `finish_reason` is `stop` (or `tool_calls` if calls were emitted), and the engine request is cancelled. `completion_tokens` counts through the token that completed the match.
- **Responses** are OpenAI shapes: `message.reasoning_content` / `content` (null when empty) / `tool_calls`; streaming sends a role chunk at once, one chunk per non-empty delta (`reasoning_content`, `content`, `tool_calls` with `index`), a finish chunk with an empty delta, a usage chunk with `choices: []` when `stream_options.include_usage` (the gateway always sets it), then `[DONE]`. A mid-stream failure ends the stream with an `event: error` frame.
- **Usage** includes `prompt_tokens_details.cached_tokens`, the prompt positions served from the prefix cache. It is for the client's information only: nothing here prices, and the gateway's charge never depends on it.

`GET /v1/engine/info` (gateway token, no weights header) returns `model`, `weights_sha256`, `executor`, and `build` (`crate`, `version`, and `git_sha`: the `EIDOLA_GIT_SHA` value the build was given at compile time, `null` otherwise; nothing is read from the build environment implicitly). `GET /healthz` is unauthenticated and content-free: `200 ok` while the engine thread runs, `503` after it stopped.

## Concurrency

The executor seam is synchronous, so one dedicated thread owns the `Engine` and runs its step loop. HTTP tasks reach it over channels: commands (`Submit`, `Cancel`) in, one unbounded event channel per request out (bounded in practice by `max_tokens`; the engine thread never blocks on a slow reader). Between steps the thread drains every pending command; when idle it sleeps until a command arrives or the one-second sweep is due, and sweeps the prefix cache so expired KV is zeroed without traffic.

- **Admission** is bounded by `EIDOLA_ENGINE_MAX_REQUESTS` (running plus queued in the engine). A permit is taken before rendering or tokenizing, travels with the submission, and is released when the request leaves the engine. With none free the request is refused at once (`overloaded`); nothing queues unboundedly in front of the engine. Rendering and tokenization run on the blocking pool.
- **Cancellation**: each request holds a guard that sends `Cancel` when dropped, so a client disconnect (the SSE stream or the handler future dropped) releases the sequence before the engine's next step, including a request still waiting for a seat. Independently, the engine thread cancels a request whose reader has gone away the next time it has output for it.
- **Failure**: an `ExecutorError` is fatal (the host cannot know which writes landed). In-flight requests get an error, health turns `503`, and the process exits non-zero.

## Content-free by construction

Prompts, outputs, cache keys and salts are content; so is anything derived from one request (its token counts, its error details). None of it is logged:

- `ApiError`'s `Display` (the only rendering that reaches logs) is the error category alone; the detail goes only into the caller's response body (`error.rs`, tested).
- Configuration and boot errors name variables, files and limits, never values; the gateway token is held only as its SHA-256 and prints as a redaction marker.
- The engine core's secret types (`CacheKey`, `EngineSalt`) print redacted and are zeroed on drop; the decoded key's buffers here are scrubbed (`zeroize`).
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

**The dev model** is the model crate's committed synthetic fixture with `embed_tokens` and `lm_head` widened from 256 rows to MiMo's padded 152,576 (the fixture's rows first, then deterministic random rows at the same scale), so the real pinned tokenizer and template can drive it; every other tensor is the fixture's. Its weights are random: the output is gibberish, and nothing about it is a quality signal.

## Tests

`cargo test -p eidola-server-engine` (about 5 s after the first build; no network, no GPU). `tests/server.rs` runs the real node in process on a loopback port, over the dev model (built once into the target directory, loaded once per test binary) and the CPU executor:

- gateway token required (chat and info), checked before the weights hash; health unauthenticated;
- weights-hash mismatch and absence refused on the header alone (an unparseable body gives the same answer), with the engine's counters untouched;
- wrong model; the strict subset refusing unknown fields at every level, images, unsupported `tool_choice`, bad stops, sampling and cache keys, without reaching the engine, and accepting the body shape the gateway forwards today (explicit nulls included);
- non-streaming output equal, token for token, to the engine core driven directly over the same CPU executor (without speculation, while the node speculates with two MTP depths), and the chat layer's decoding of those tokens; usage arithmetic;
- streaming chunk shapes, the usage chunk with and without `include_usage`, and text equal to non-streaming;
- stop sequences (streaming and not) truncating at the first match and cancelling the engine request;
- client disconnect cancelling a running request and a request still queued for a seat (the latter only the guard can cancel);
- the admission bound refusing with `overloaded` and recovering;
- `cache_key`: the same key reports whole cached blocks with identical output; another key, and no key, report none;
- tool definitions and tool-call history with string arguments rendering, and non-object history arguments refused;
- prompts beyond the model length; a full `boot` from the directory; boot refusing a wrong expected hash (before loading) and a loaded model's hash mismatching the configuration; every configuration variable's absence and malformations; sizing the model or the core cannot honour.

Each of these was checked to fail under a deliberate bug: no cancel on guard drop (the queued-disconnect test), no cancel on either path, a dropped first output token, a different prompt rendering, a salt not derived from the key, the weights check moved after body parsing. Keep that true when changing the tests.
