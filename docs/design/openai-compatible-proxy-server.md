# OpenAI-Compatible Proxy Server

## Context

The project needs a server component that provides AI capabilities to clients.
There are two fundamental approaches:

1. **Run inference locally** on the server (requires GPU hardware, large
   binaries, complex deployment).
2. **Proxy to upstream providers** and translate between API formats
   (lightweight, stateless, deployable anywhere).

Separately, the AI API ecosystem has fragmented across incompatible formats
(OpenAI, Anthropic, Google, Mistral, etc.), but the OpenAI Chat Completions
format has become the de facto standard. Most client libraries, tools, and
integrations speak OpenAI format first.

## Decision

### OpenAI API as the canonical interface

The server exposes the **OpenAI Chat Completions API** (`POST
/v1/chat/completions`) as its external interface. Internally, it translates
to whatever upstream provider is configured. Clients never need to know which
provider is actually serving the request.

This means any tool that speaks OpenAI format — client SDKs, CLI tools, IDE
plugins, other proxies — works without modification.

Model name translation is handled transparently: OpenAI model names
(e.g., `gpt-4o`) are mapped to provider equivalents, and provider-native
model names (e.g., `claude-sonnet-4-20250514`) pass through unchanged.

### Proxy-first, not inference-first

The server is a stateless proxy. It does not load models, manage GPU memory,
or run inference. It translates request formats, forwards to an upstream API,
and translates the response back.

The current upstream is Anthropic's Messages API. Adding new upstreams
(Google, Mistral, local inference) means implementing new translation modules,
not changing the server architecture.

Local on-device inference exists separately in the macOS app via
`eidolons-perception` (see [On-Device Inference with Burn](on-device-inference-with-burn.md)).
The server does not depend on any ML crate.

### Stateless design

Each request is independent. No conversation history, session state, token
counting, rate limiting, or response caching. The server is a pure function:
`request -> upstream call -> response`.

This maximizes horizontal scalability (any instance can handle any request)
and simplifies deployment (no shared state, no database, no session store).

### Minimal HTTP stack

The server is built directly on **hyper + tokio** without a web framework
(no Axum, Actix, Warp, or Rocket). HTTP/1.1 only. Routing is a pattern match
in a single `handle_request` function.

This minimizes binary size (~9MB distroless OCI image) and dependencies. The
server's scope is narrow enough that framework ergonomics (middleware,
extractors, error handling) don't justify the added dependency weight.

### Distroless OCI deployment

Server containers contain only the statically-linked musl binary. No shell,
package manager, libc, or any other runtime dependency. The container runs as
UID 65534 (nobody) with no capabilities.

See [Reproducible Builds](reproducible-builds.md) for the full build and
deployment pipeline.

### OpenAPI specification from code

The API is documented via `utoipa` annotations on Rust types. The OpenAPI spec
is generated as a build artifact, committed to the repository, and
CI-verified for freshness. This ensures documentation never drifts from
implementation.

## Consequences

**Benefits:**

- Any OpenAI-compatible client works out of the box.
- The server is tiny (~9MB), starts instantly, and scales horizontally.
- Switching upstream providers is a configuration change, not an architecture
  change.
- No GPU hardware required for the server. Cloud inference is available
  anywhere with network access.
- Statelessness eliminates an entire class of bugs (session corruption, cache
  invalidation, state synchronization).

**Trade-offs we accept:**

- Every request pays full upstream API latency. No caching means repeated
  identical prompts are re-processed.
- No conversation history means the client must send the full context with
  every request (standard for the OpenAI API pattern, but larger payloads).
- The minimal HTTP stack means no middleware. Cross-cutting concerns (logging,
  auth, rate limiting) must be added manually or handled by an external
  reverse proxy.
- HTTP/1.1 only. HTTP/2 would improve multiplexing for streaming responses
  but adds complexity.
- Only one upstream provider at a time. Multi-provider routing (e.g., route
  by model name) would require additional infrastructure.

## Future Considerations

- **Local inference as an upstream.** The server could gain an "upstream" that
  routes to `eidolons-perception` for on-device inference, making the proxy
  architecture work for both cloud and local models with the same client API.

- **Additional upstream providers.** Google (Gemini), Mistral, and others
  can be added as translation modules alongside the existing Anthropic module.

- **Streaming improvements.** The current SSE streaming implementation
  translates Anthropic's streaming format to OpenAI's. More complex streaming
  behaviors (tool use, multi-turn) may require buffering or state.

- **Rate limiting and auth.** If the server is exposed beyond localhost,
  it will need authentication and rate limiting. These could be added as
  middleware or delegated to a reverse proxy (nginx, Caddy, etc.).
