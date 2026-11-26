# Architecture

Eidola is built from a small set of components arranged around explicit trust boundaries. Everything that touches your prompts and answers runs either on your own device or inside hardware-attested confidential enclaves whose exact code is verified before a request is sent. Everything else — the database, billing processes, telemetry, and third-party services — is treated as untrusted and shown as little as possible.

This page walks through the system from end to end: your device, the gateway, Eidola's inference engine, and each connection between them.

## System map

```mermaid
flowchart TB
  accTitle: Eidola's components and the connections between them
  accDescr: The Eidola app on your device connects to the attested gateway for chat and for its account. The gateway connects to the attested inference engines, to Postgres, to Stripe and to Grafana. Your browser pays at Stripe, Stripe notifies a webhook receiver, and a billing worker applies payments from Stripe to Postgres. The gateway, the engines and both billing processes send telemetry to Grafana. Prompts and answers travel only between the app, the gateway and the engines, and nothing on those connections is tied to an account.

  subgraph device [Your device]
    client[Eidola app]
    browser[Browser]
  end

  subgraph attested [Eidola, attested]
    gateway[Gateway]
    engines@{ shape: processes, label: "Engines" }
  end

  subgraph unattested [Eidola, unattested]
    pg[(Postgres)]
    webhook[Webhook receiver]
    worker[Billing worker]
    grafana[Grafana]
  end

  subgraph external [Third parties]
    stripe[Stripe]
  end

  client -- account --> gateway
  client == chat ==> gateway
  gateway == inference ==> engines
  gateway -. spends .-> pg
  gateway -- account records --> pg
  gateway -- checkout --> stripe
  grafana <-. telemetry .- gateway
  browser -- payment --> stripe
  stripe -- events --> webhook
  webhook -- event queue --> pg
  worker -- billing work --> pg
  worker -- payment facts --> stripe
  grafana <-. telemetry .- webhook
  grafana <-. telemetry .- worker
  grafana <-. telemetry .- engines

  class client,gateway,engines content
  classDef content stroke-width:3px

  click client "#the-client"
  click gateway "#the-gateway"
  click engines "#the-inference-engine"
  click pg "anonymous-credits.md#what-the-database-may-hold"
  click webhook "#events"
  click worker "#billing-work"
  click stripe "vendors.md#processors--parties-handling-data-on-our-behalf"
  click grafana "#telemetry-and-logs"
```

### Reading the diagram

- **Boxes are trust boundaries.**
  - *Eidola, attested* components run inside confidential-compute enclaves. Whoever connects to them verifies their hardware measurement on every new connection before sending any data.
  - *Eidola, unattested* components are supporting services run by Eidola or on its behalf outside enclaves. The app cannot pin or verify them, so the system treats them as untrusted and never sends them prompt content or linkable spend records.
  - *Third parties* are external services Eidola relies on (currently Stripe for payment processing).
- **Dotted vs. solid lines.** A dotted line carries nothing that can be tied to an account or person; a solid line carries data tied to a conversation, an account, or a payment.
- **Thick lines and outlines** mark the only paths and components that ever see your prompts and answers in the clear: your device, the gateway enclave, and the inference engine enclave.
- **Arrows** point away from the side that opens the connection (data flows both ways on almost every link).

Click any box to jump to its description, or any labelled arrow to jump to its privacy breakdown in [How the pieces connect](#how-the-pieces-connect).

**Not shown:** connections outside Eidola's hosted compute path. Your device checks GitHub for releases and downloads updates there; Stripe redirects your browser back to this website after checkout, and the gateway reads the current legal terms from it; your device downloads on-device models from Hugging Face when you ask it to, and talks directly to any external model server you configure yourself. Beneath the attested enclaves, the hosting provider operates the physical machines and network and supplies deployment secrets ([Infrastructure & vendors](vendors.md#infrastructure-inside-the-trust-boundary)).

## The client

Eidola ships as a desktop app (macOS and Linux), a command-line tool (`eidola`), and a mobile app (iOS and Android). All of them run **one shared Rust core** — the same wallet, the same local database, the same attestation verifier, and the same fail-safe rules. On mobile, that core compiles to WebAssembly and runs inside a minimal native shell that passes only encrypted network bytes ([Apps & platform verification](apps.md#mobile-apps)).

### Fail-safe verification

Every trust decision happens locally on your device, against a **trust root** compiled directly into the app at build time. There is no runtime API and no server call that can alter what a running client trusts.

- **One client release pairs with one gateway build.** A client does not accept "any gateway at or above version X" — a floor like that would let an attacker downgrade you to an older, vulnerable server build. Instead, each client release pins one exact gateway launch identity, and that gateway pins the exact inference engines it will talk to. Updating the server side reaches users only through a new, attested client release.
- **Fresh attestation on every connection.** Every time the client opens a new TLS connection to the gateway, it generates a fresh random nonce and requests a hardware attestation report over that same connection before sending any application data. Offline, using hardware vendor roots compiled into the app, it verifies that a genuine confidential-compute processor signed the report, that the report binds both the fresh nonce and the TLS certificate's public key, that the enclave's debug mode is disabled, that its firmware meets minimum security floors, and that its boot image and configuration hash match the compiled-in pin. There is no "verified once" cache.
- **Refusal on any failure.** If attestation fails, if an update lacks the required human signatures, or if a release document uses a schema version the client doesn't understand, the client refuses to proceed. There is no fallback and no prompt to ignore a security warning.

### What stays on your device

Eidola keeps everything personal on your machine, never in a cloud profile:

- **Conversations and account state** live in a local database in your device's application storage. Your device is the only place your account identity and your conversations ever meet.
- **Your anonymous wallet** holds your credit tokens locally. The client generates each token's secrets, requests blind signatures from the gateway, and spends tokens on inference without revealing which account funded them ([Anonymous credits](anonymous-credits.md)).
- **Stream assembly.** The gateway serves every chat answer as a stream. Even when you ask for one complete answer (from the CLI or a non-streaming local tool), your client reads the stream and assembles it locally — so if a connection drops halfway through, you still keep the text you paid for and settle cleanly on what was generated.
- **Self-update verification.** Checking whether an update is legitimate runs entirely inside the client, checking signatures and transparency-log proofs locally against its compiled-in trust root ([Releases](releases.md)).
- **Prefix-cache keys.** Reusing work an inference engine has already done on the earlier turns of a conversation makes follow-up turns much faster and cheaper. To scope that reuse strictly to one conversation, the client generates a random cache key for each conversation lineage (one participant in one space) and only sends it for models that support prefix caching. Because a shared key links the turns that carry it, the client automatically rotates it whenever the conversation goes idle past the model's declared retention window, hits its maximum age, or switches models — and never sends cache keys for requests coming through the local inference proxy. Keys live only in your local database and are never logged or exported.

> [!NOTE]
> **Configuration overrides.** Developers and self-hosters can override where the client connects (the server URL, accepted enclave measurements, and hardware root CAs) through Settings or `eidola configure`. Overrides never disable attestation — they only change which enclave identity is expected — and release-verification rules cannot be overridden at all. Eidola's [privacy guarantees](privacy-guarantees.md) apply only to official releases running under the built-in trust root with no overrides set.

## The gateway

The gateway is the single front door to Eidola's hosted compute. It runs inside a confidential-compute VM whose entire boot image and deployment configuration are measured at launch and verified by every connecting client. TLS terminates **inside that enclave**: the private key is generated in enclave memory and has no export interface, so neither the hosting provider nor Eidola's operators can inspect traffic in transit. The gateway is purely request-based — it keeps no conversation history, no cross-request memory, and no operator interface for inspecting, flagging, or replaying traffic.

### Linked vs. unlinked surfaces

The gateway's core privacy boundary is a strict structural split across its HTTP endpoints:

| Surface | Authentication | Knows your account? | Sees prompts or answers? |
| --- | --- | --- | --- |
| **Public** | None | No | No |
| **Linked** (account & billing) | Account ID + secret (HTTP Basic header) | Yes | No |
| **Unlinked** (inference) | Anonymous credit tokens (request body) | **No** | Only in enclave memory while routing |

Which surface an endpoint belongs to is fixed in code, not configuration. Account endpoints accept only account credentials and never handle inference; inference endpoints accept only anonymous credit tokens.

With full access to the database, nobody at Eidola can connect an inference request to the account that paid for it, or link two inference requests to the same user. For the full story of how tokens are issued, rotated, buffered, spent, and recovered across crashes without leaking identity, see [Anonymous credits](anonymous-credits.md).

Account billing reconciliation — receiving Stripe webhooks, handling refunds, disputes, and account closures — runs in separate billing processes sharing Postgres. Those processes never touch the inference path or anonymous spend records, and cannot identify or recall tokens that have already been issued.

### Response shaping

Even over encrypted TLS, a streamed AI answer can leak information through packet sizes and timing: each token arrives as its own encrypted chunk, and published attacks can recover topics and phrases from token lengths and pacing alone.

To hide token-level shape on **both** network hops — between your device and the gateway, and between the gateway and the inference engine — every streamed chat answer is re-framed into fixed-size frames sent on a fixed clock tick (padding any spare space in each frame with whitespace ignored by standard stream readers). Every non-streaming response, including errors and refusals, is padded to a fixed power-of-two size bucket, and refusals never echo back what your request contained.

Response shaping hides individual token sizes and inter-token rhythm. It does **not** hide that a connection happened, when it started, how long the whole answer took, the size of your outgoing request (requests are not padded yet; see issue [#620](https://github.com/eidola-ai/eidola/issues/620)), or the engine's own hardware activity as observed by its host. See [Known gaps](gaps.md#network-and-metadata).

### Telemetry and logs

Operating a real service requires knowing whether engines are healthy and where latency is going, but does not require creating a per-request audit trail.

- **Aggregated in the enclave.** The gateway and engines record internal timing spans, convert them into aggregate counters and histograms *inside the enclave*, and discard the individual spans. Only coarse periodic aggregates leave for Grafana, labelled strictly from fixed lists (models are resolved to their weights hash on arrival; unrecognized inputs collapse to a shared bucket). Nothing per-request is exported by default.
- **No per-request logs.** Because every log line carries a timestamp, logging routine per-request events (like refusals) could let an observer correlate timing across surfaces. Routine conditions are counted in aggregate metrics instead. Logs are reserved for genuine incidents — misconfiguration, engine health changes, and rare server errors — with internal error messages redacted before leaving the enclave.
- **Opt-in tracing.** A request produces an exported trace only if the caller explicitly requests one by sending a sampled W3C `traceparent` header (used when we profile our own traffic).

## The inference engine

Models do not run inside the gateway. They run on **Eidola's own Rust inference engine**, deployed in separate confidential VMs where all participating hardware — the CPU and any attached GPUs — operates in confidential-compute mode and is transitively bound into the checked measurement. Each deployment serves exactly one open-weight model.

### Why we built our own engine

Most hosted AI providers wrap off-the-shelf serving stacks like vLLM. We chose to write a compact, purpose-built Rust engine instead, for four reasons that matter directly to users:

1. **Auditability and reproducibility.** General-purpose serving engines are massive, fast-moving codebases spanning Python, C++, and runtime-compiled CUDA, built to support hundreds of models across dozens of hardware setups. Auditing every path a prompt takes — or reproducing the exact binary from source — is practically impossible at that scale. Eidola serves a small, curated set of open-weight models on a tightly bounded set of confidential hardware. That narrow scope makes a small, self-contained Rust engine tractable: its binary is built reproducibly from source, and its GPU kernels are compiled ahead of time (with no runtime JIT compilation) and checked against a committed hash manifest.
2. **Batch isolation.** In conventional inference engines, floating-point execution order varies with batching and scheduling — meaning the exact output your prompt produces can depend subtly on what other users' prompts happen to be running in the same batch. That may not look like a blatant data leak, but it is a weak isolation boundary between users sharing a GPU, and weak boundaries get exploited. Eidola's engine enforces **batch invariance**: your logits and outputs depend strictly on your own tokens, never on who else is scheduled beside you, how requests are chunked, or which memory slots they occupy.
3. **Client-encrypted prefix caching.** Standard serving engines keep cached conversation state sitting in plaintext in GPU or host memory, shared across requests and discounted after the fact. Owning the engine lets Eidola isolate each conversation's cache under a key the client supplies and rotates, encrypt idle KV cache entries under a key derived from the client's (which the engine does not retain), and verify a client's [declared cached prefix](anonymous-credits.md#declared-prefix-caching) *before* doing any compute.
4. **Clean continuation.** To keep upfront anonymous-credit holds small without leaking wallet balances, the client requests bounded output lengths and automatically continues long answers across follow-up requests ([Output ceilings and automatic continuation](anonymous-credits.md#output-ceilings-and-automatic-continuation)). General-purpose engines rarely guarantee that an answer stopped at a token limit — especially mid-reasoning — can be resumed cleanly. Owning the engine lets Eidola explicitly signal when a length-limited output is cleanly continuable and resume it seamlessly from that boundary.

### Pinning and attestation

You don't take our word that a model runs in confidential compute — that property is enforced structurally by the same verification chain that pins the gateway:

- **The gateway pins its engines at build time.** Every accepted engine deployment is compiled into the gateway's own measured image: its confidential-VM boot image measurement, the SHA-256 of its committed deployment configuration, its host and hardware policy (including any attached GPUs), and the exact **weights hash** of the model it serves ([Model identity](model-identity.md)). Because the client pins the gateway's measurement, one client release transitively fixes the entire inference path — gateway, engine, hardware mode, and model weights.
- **Database placement matters for availability only.** Postgres tells the gateway *where* running engines can be reached, but never *which* ones to trust. On every new TLS connection to an engine, the gateway runs the same per-handshake hardware attestation verifier the client uses, and probes the engine's weights hash before sending any user request. An engine whose attestation or weights don't match the compiled-in pin is refused before a single prompt crosses the wire.
- **Updates roll in order.** Nothing can add an engine pin at runtime. Adding a model or changing an engine image requires a new gateway build and a new client release that pins it.

**What remains trusted at this layer** (as distinct from what is verified):

- **Hardware vendors.** AMD, Intel, and NVIDIA are trusted to manufacture genuine confidential-compute hardware and honest attestation keys ([Threat model](threat-model.md#hardware-vendor-forgery)).
- **GPU checks inside the measured guest.** Where a deployment uses GPUs, the measured boot image runs NVIDIA's attestation verifier on each GPU at startup and refuses to open them unless it passes, and our engine checks before using any GPU that confidential-compute mode is active and developer tools are off. On each request handshake, the gateway verifies the VM's hardware attestation and checks the nonce-labelled GPU evidence count, rather than re-appraising NVIDIA's certificate chain itself ([Known gaps](gaps.md#gpu-attestation-and-host-endorsements)).
- **Platform boot images and host endorsements.** Both the gateway and engines boot confidential-VM images published by our hosting platform (Tinfoil), pinned by hash and reproduced by us from their public build definition, though some upstream inputs (Ubuntu packages, Docker runtime, NVIDIA driver/CUDA toolkit) are binary inputs rather than source-bootstrapped. For endorsed-host engine deployments, the allowlist of machines and firmware floors comes from Tinfoil's signed platform endorsements committed into the gateway build ([Known gaps](gaps.md#non-source-bootstrapped-build-inputs)).

### Memory, caching, and shutdown

Everything an engine computes from your prompt stays strictly in ephemeral enclave memory — never logged and never written to disk.

- **Unkeyed requests leave nothing behind.** When a request without a prefix-cache key finishes, the engine zeroes and releases the memory it used. (Released memory in general host allocations is reclaimed on normal reuse; KV pools and host-tier buffers are explicitly zeroed on free.)
- **Encrypted idle prefix cache.** When a request carries a client prefix-cache key, the engine retains the computed prompt state only within the model's declared idle and maximum-age limits, and only a later request presenting the same key can reuse it. Between requests, idle cache entries are encrypted under a key derived from the client's own cache key — which the engine does not keep — so the engine itself cannot read an idle entry on its own. Under GPU memory pressure, encrypted entries can be demoted to page-locked enclave RAM and promoted back on a hit, still encrypted while idle. Lookup keys are one-way hashes that reveal neither your prompt nor your cache key.
- **Clean failure on hangs.** If an engine encounters an unrecoverable error — or if a GPU call hangs past a generous watchdog bound — the engine stops serving, exits for the platform to restart it, and the gateway settles every in-flight request fairly on what was already generated. (Note that a forced watchdog exit cannot run the normal zero-on-free pass before the process terminates; what was in flight stays inside enclave memory until the driver reclaims it.)

## How the pieces connect

Every labelled arrow in the [System map](#system-map) links directly to its entry below: who can observe the connection, what they can see, and what protects it.

### Inference and accounts

#### Chat

The app sends each inference request to the gateway, paid for with anonymous credit, and receives the streamed answer.

- **Who can see it:** Your network, internet transit providers, the hosting provider's network, and — inside its enclave — the gateway's measured code.
- **What they see:** On the wire, encrypted traffic: your IP address, when requests happen, request sizes (requests are not padded), and how long each response runs. Inside the gateway enclave: the prompt, the answer, and the anonymous spend paying for it (but never which account funded that spend).
- **What protects it:** Per-handshake attested TLS terminated inside the gateway enclave; anonymous credit tokens with no account credentials allowed; fixed-frame, fixed-tick response shaping hiding token sizes and rhythm.

#### Account

The app creates an account, records terms acceptance, converts purchased balance into anonymous credit tokens, and opens Stripe checkout or billing-portal sessions.

- **Who can see it:** The same network observers as Chat, and the gateway.
- **What they see:** On the wire, encrypted traffic metadata. Inside the gateway: a random account ID (no name or email), when credits are converted and in what denomination, terms acceptance, and billing requests. Never a prompt.
- **What protects it:** The same per-handshake attested TLS as Chat. Tokens are blind-signed so the gateway cannot recognize them when later spent, and the app converts credits ahead of need at random intervals in a single standard denomination so conversions don't line up with requests ([Anonymous credits](anonymous-credits.md)).

#### Inference

The gateway forwards each inference request to an attested engine serving the requested model weights and relays the answer back.

- **Who can see it:** The hosting provider (which operates the machines and the network between enclaves), Eidola as operator (network metadata only), and — inside its confidential VM — the engine's measured code.
- **What they see:** On the wire, encrypted traffic timing and request sizes. Because requests carrying the same prefix-cache key route to the same engine for cache locality, network observers between enclaves can correlate turns sharing a key by destination as well as timing. Inside the engine: the prompt, the answer, the requested weights hash, and the cache key — never a credit, an account, or a trace.
- **What protects it:** Per-handshake attested TLS terminated inside the engine's confidential VM against pins compiled into the gateway; weights-hash verification on every request; fixed-frame response shaping between enclaves; encrypted idle prefix caching.

### Database and telemetry

#### Spends

The gateway records each anonymous credit spend in Postgres so no token can be spent twice.

- **Who can see it:** Whoever can read the database — PlanetScale (our database host) and anyone with database credentials.
- **What they see:** One row per spend: the issuer key ID, a one-time serial number (nullifier) revealed only at spend time, and an encrypted refund blob of fixed length. No account ID, no cost, no model, and no timestamp column. Because each row is written when a request starts and updated when it finishes, a live observer of database writes (or replication logs) can observe request start times and durations ([Known gaps](gaps.md#database-spend-timing)).
- **What protects it:** Strict schema separation (nothing on the inference path names an account, amount, or content) and client-keyed refund encryption that neither the database nor Eidola can open ([Anonymous credits](anonymous-credits.md#what-the-database-may-hold)).

#### Account records

The gateway reads and writes account balances, credit conversions, terms acceptances, encrypted issuer keys, and engine availability in Postgres.

- **Who can see it:** The same parties as Spends.
- **What they see:** Random account IDs, hashed secrets, Stripe customer references, terms records, conversion ledger rows (amounts and times), queued repair requests, and encrypted issuer keys — plus the list of reachable engine addresses.
- **What protects it:** No personal contact information is collected, issuer private keys are encrypted with their validity windows authenticated inside the enclave, and account records share no join key with spend records.

#### Telemetry

The gateway, inference engines, webhook receiver, and billing worker export operational metrics and incident logs to Grafana Cloud.

- **Who can see it:** Grafana Cloud, and the networks in between.
- **What they see:** From the gateway and engines: periodic aggregate counters and histograms (latencies, token totals, stream outcomes, queue and memory levels) labelled only from fixed sets, plus redacted incident logs. (An interval containing only a single request necessarily reflects that one request.) From the billing processes: job counts, Stripe call counts, and aggregate purchase/refund/dispute amounts by currency, but never an account or Stripe identifier.
- **What protects it:** In-enclave aggregation before export, fixed label lists, coarsened engine cache counters on a fixed export timer, and measured gateway/engine code fixing exactly what can be sent ([Telemetry and logs](#telemetry-and-logs)).

### Payments and billing

#### Checkout

The gateway fetches current plan prices from Stripe, creates a Stripe customer for an account, opens checkout and billing-portal sessions, and checks subscription status.

- **Who can see it:** Stripe, and the network between the gateway and Stripe.
- **What they see:** The account's random ID, the plan chosen, and when. Nothing about inference.
- **What protects it:** Measured gateway code fixes what is sent, and no trace context propagates outbound. Stripe can link a payment to a person; Eidola ensures payment cannot be linked to usage.

#### Payment

Your browser opens Stripe's own hosted pages to pay for a subscription or credit top-up.

- **Who can see it:** Stripe, and your network.
- **What they see:** Your payment method, email, and transaction history, linked to your account's Stripe customer record.
- **What protects it:** Payment details never pass through Eidola's servers, and spending credits crosses the anonymous-credential boundary before reaching inference.

#### Events

Stripe sends thin webhook notifications to Eidola's webhook receiver when payment events occur.

- **Who can see it:** Stripe, the network, and infrastructure in front of the unattested webhook receiver.
- **What they see:** Only an event ID, event type, and the ID of the related Stripe object — no customer details.
- **What protects it:** The receiver verifies Stripe's webhook signature before accepting an event and holds no Stripe API key that could fetch customer details.

#### Event queue

The webhook receiver writes each verified notification ID into Postgres for the billing worker to pick up.

- **Who can see it:** Whoever can read the database.
- **What they see:** The event ID, type, Stripe object reference, and arrival time. The receiver touches no account rows and no spend records.
- **What protects it:** Minimal data stored — only the identifiers Stripe already sent.

#### Billing work

The billing worker processes queued events and applies purchases, refunds, and disputes to account balances in Postgres.

- **Who can see it:** Whoever can read the database.
- **What they see:** Account credit grants, prices paid, refunds, and dispute adjustments against the account ID.
- **What protects it:** Billing processes never read or write anonymous spend records, so nothing in billing can locate or recall anonymous tokens already issued ([Anonymous credits](anonymous-credits.md#what-the-database-may-hold)).

#### Payment facts

The billing worker queries Stripe directly for the authoritative state behind each queued notification (checkouts, invoices, charges, refunds, disputes, subscriptions) and executes refunds or cancellations.

- **Who can see it:** Stripe, and the network in between.
- **What they see:** Which Stripe objects the worker inspects or updates, and when.
- **What protects it:** The worker treats webhook notifications solely as a prompt to check Stripe directly, holds no webhook secret, and accepts no inbound connections.
