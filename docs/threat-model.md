# Threat model

This document names who Eidola defends against, who it does not, and what is left as residual trust.

For each adversary, the question is the same: *given the [privacy guarantees](privacy-guarantees.md), what can this adversary do, and what stops them?*

## Adversaries we defend against

### A1. A curious or malicious Eidola operator

**Capability:** Operates Eidola's infrastructure and can observe its logs, its database, and its network.

**What stops them:** The boundary between the *linked* account surface and the *unlinked* inference surface is structural in the gateway: the two take different kinds of credential and no endpoint accepts both, and the inference surface uses anonymous credentials (Privacy Pass ACT). The operator can see *that* a credential was redeemed, never *whose* it was. Inference content never reaches a log, and exists in cleartext only inside the gateway's and the engine's enclaves. Telemetry carries no per-request records: the operator sees counters and histograms aggregated across requests. The operator cannot single out a request for closer inspection either: a per-request trace is produced only when the requester asks for one by sending a `traceparent`.

**Residual exposure:** The database runs outside the enclave, so the operator sees when each spend is written and so roughly how long each paid request ran ([gaps.md](gaps.md#spend-timing-at-the-database)). The operator could also attempt to deploy a gateway or an engine that violates these properties. The client refuses to talk to any gateway whose attestation does not match the launch identity pinned in the client build, and the gateway refuses any engine that does not match the pins in its own, so deploying either requires the operator to also ship a violating *client release*. That path is blocked by the release-integrity invariants in [privacy-guarantees.md §6](privacy-guarantees.md#6-release-integrity) — in particular, the embedded trust root (§6.5), the human attestation requirement (§6.2), the attestant's no-known-backdoor claim (§6.3), and the no-compulsion claims (§6.4).

### A2. A passive network observer

**Capability:** Sees all packets between the user's device and Eidola's servers, including TLS metadata (SNI, IP, timing).

**What stops them:** TLS confidentiality and integrity. (The deeper attestation binding that protects against an *active* attacker with a valid WebPKI cert is described in [A3](#a3-an-active-network-attacker-mitm) — a passive observer has nothing to gain there, since they can't impersonate the gateway in the first place.)

**Residual exposure:** Network metadata remains visible: that a connection occurred, when, from where, the size of each request, and how long each response ran. The gateway pads its responses and writes a streamed answer in fixed-size frames at a fixed interval, so the size and timing of individual tokens are hidden ([privacy-guarantees.md §3.8](privacy-guarantees.md#3-content)); nothing else about the traffic is. Users who need more should layer Eidola behind Tor or a similar anonymity network. See [gaps.md](gaps.md#network--metadata).

### A3. An active network attacker (MITM)

**Capability:** Can intercept, modify, and re-route traffic between the user's device and Eidola.

**What stops them:** Binding between the TLS key and a *fresh* enclave attestation. The client issues an inline `GET /.well-known/tinfoil-attestation?nonce=<hex>` (a fresh random nonce per handshake) over the same TCP+TLS connection as the real request. It hashes the exact endorsed crypto-material and device-evidence section bytes and verifies `REPORT_DATA = sha256(report-data-v1 URI ‖ nonce ‖ crypto_material_hash ‖ device_evidence_hash)`, with the crypto section's `tls` item equal to `sha256(SPKI(peer_cert))`. An attacker presenting a valid certificate for the same hostname without the matching attestation fails the check; an attacker presenting a stale or replayed attestation document fails either the SPKI binding or nonce check.

**Residual exposure:** The per-handshake nonce proves *freshness*: the report was generated for this handshake by a genuine confidential-compute machine that holds the certificate's key, which defeats replay of a captured document. It does not defeat an attacker who has **exfiltrated** the enclave's TLS private key. The report binds the enclave's TLS *key* (the certificate's SPKI), not the TLS *session*, so such an attacker can terminate the client's connection with the stolen key and relay a fresh, nonce-bound report fetched from the enclave's public attestation endpoint, which serves any nonce. Every client check passes. The defense is that the key stays sealed inside the enclave. Removing the exposure would require channel binding: committing a TLS-session value (such as an RFC 5705 exporter) into `REPORT_DATA` ([gaps.md](gaps.md#tls-key-exfiltration-and-channel-binding)).

### A3.5. A rogue deployer

**Capability:** Anyone with deploy access to the legitimate Eidola infrastructure (the hosting platform's operators, an Eidola employee with deploy credentials, or an attacker who has compromised one) could attempt to alter what runs: substitute a binary, change injected secrets, point the gateway at a different inference engine, or roll back to an older, vulnerable build.

**What stops them:** What the gateway runs is determined by its **attested configuration** (`tinfoil-config.yml`), whose SHA-256 every attestation report carries as HOST_DATA. The enclave checks that hash against its configuration before it boots, and the client pins it beside the launch digest. The configuration covers the gateway image digest, the argument list, the environment, and the hashes of the injected secrets, which the gateway checks at startup. Any change produces a different launch identity, which the client refuses to connect to. The pairing of client and gateway (see [client.md](client.md#one-release-pairs-exactly-one-client-with-one-server)) also blocks rollback: each client release pins exactly one gateway launch identity, so a genuinely attested *older* gateway build does not match the pin in an updated client. The inference engines are fixed the same way, one level down: their deployments are pinned in the gateway's own build, the placement data the gateway reads from its database says only where they are, and an engine whose attestation matches no pin for the requested model is refused ([engine.md](engine.md#what-pins-the-engine)).

**Residual exposure:** Configuration that is not committed into the attested boot path is not bound into the measurement, so any new sensitive setting has to be added there. The boot images themselves are the hosting platform's: pinned by hash and measured, but not built by Eidola ([gaps.md](gaps.md#build-chain-opacity)).

### A3.6. A server impersonator

**Capability:** Operates a server pretending to be Eidola from outside the legitimate infrastructure: a DNS hijack, a BGP redirection, a server the attacker controls at the real hostname, or a stack not running confidential compute at all. The active attacker of [A3](#a3-an-active-network-attacker-mitm) is one route to this.

**What stops them:** The same controls as [A3.5](#a35-a-rogue-deployer). The client's pinned launch identity names exactly one gateway build; only an enclave running that build can produce a matching attestation, and only confidential-compute hardware with valid vendor keys can produce an attestation at all. WebPKI is an additional layer: an impersonator without a valid TLS certificate for the real hostname never reaches the attestation step, and one with a fraudulent certificate still fails the measurement check.

**Residual exposure:** The same as A3: an attacker who has exfiltrated the enclave's TLS private key can relay a fresh attestation.

### A4. An attacker who compromises the Eidola release pipeline

**Capability:** Has gained write access to the GitHub repo, the CI system, or the OIDC identity used to sign artifact manifests.

**What stops them:** The release pipeline's signature alone is not sufficient for the client to accept a release. The client *also* requires a [human attestation](releases.md) signed by a pinned attestant from hardware under their exclusive physical control, posted to a public transparency log (Rekor). An attacker who controls CI cannot mint that signature. An attacker who has additionally compromised an attestant's hardware-bound key has crossed a higher bar; the verifier enforces a minimum number of independent attestations, raising the cost as more attestants come online.

**Residual exposure:** A compromise of every pinned attestant simultaneously, sufficient to extract or coerce each hardware-bound signing key, defeats this. The mitigation is distribution of attestants across hardware tokens, jurisdictions, and physical custody, plus a public minimum threshold pinned in the *prior* client so the bar cannot be lowered by the compromise itself.

### A5. A compromised dependency (supply chain)

**Capability:** Has subverted a third-party crate, container base image, or build tool that Eidola depends on.

**What stops them:** The structural defenses, in order of how much of the attack surface they actually close:

- **Pinned dependency surface.** Every dependency has a fixed version and an expected hash. Updates are explicit commits.
- **Minimal runtime dependencies.** Distribution of compiled binaries limits exposure to outdated or vulnerable libraries on the end-user's machine.
- **Source-bootstrapped reproducible builds for the Linux OCI images** (StageX). Every binary in the toolchain, down to the compiler's own bootstrap, is built from source whose hash is committed and verified.
- **Hermetic builds for the desktop artifacts** (Nix: the macOS universal binaries and the Linux GUI). These are hermetic and reproducible but not source-bootstrapped in the StageX sense. On macOS, the Apple SDK and toolchain are opaque inputs, because cross-compiling to macOS from Linux is not viable today. The Linux GUI is Nix-built because a desktop GUI must be a glibc dynamic binary to load the host GPU stack, which StageX's musl toolchain cannot produce. See [gaps.md](gaps.md#build-chain-opacity).
- **A hermetic build for the inference engine image** (Nix). The engine must load NVIDIA's driver libraries at runtime, so it is a glibc dynamic binary and cannot come from StageX. Its GPU kernels are compiled with NVIDIA's closed CUDA toolkit; the output is reproducible and checked against committed hashes on every image build.
- **Hash-pinned platform boot images.** The images the gateway and the engines boot are the hosting platform's, pinned by hash and bound into the measurements, so they cannot change silently. They are not source-bootstrapped ([gaps.md](gaps.md#build-chain-opacity)).
- **A committed `artifact-manifest.json` as a merge invariant.** The pipeline that promotes the integration branch to `main` re-derives the manifest from source and refuses to advance if the result differs from the committed copy, so reproducibility is checked before every release rather than only asserted at it.
- **Personal reproduction in the release attestation.** Every release attestation includes a claim that the attestant has personally reproduced the manifest from source on hardware under their control.

**Residual exposure:** A dependency compromise that occurred, was incorporated, and was not caught by any of the above (PR review, attestant diff review, downstream public review) is the remaining gap. Operational practices supplement these (advisory monitoring, a preference for pure-Rust dependencies, a small direct dependency count), but they are ordinary good practice rather than structural defenses.

### A6. A legally-compelled Eidola engineer

**Capability:** A court order, technical capability notice, or other legal compulsion directs an engineer to weaken the privacy guarantees or introduce a backdoor.

**What stops them:** The release-attestation no-compulsion claims ([privacy-guarantees.md §6.4](privacy-guarantees.md#6-release-integrity)) require the attestant to sign, under their legal identity, that they are *not* under such compulsion. A coerced engineer who is also gagged must either (a) decline to sign, breaking the release and sending a public signal, or (b) sign falsely and incur the legal exposure they were trying to avoid. As attestant counts grow, the required-threshold check forces the adversary to compel multiple independent engineers, multiplying the legal and operational risk on the adversary's side.

**Residual exposure:** A jurisdiction able to compel every pinned attestant in coordinated secrecy defeats this silently. The mitigation is distributing attestants across jurisdictions ([gaps.md](gaps.md#multi-jurisdiction-attestant-distribution)). Today one attestant is required ([gaps.md](gaps.md#single-attestant-policy)).

## Adversaries we do not defend against

### N1. A compromised local environment

If the user's device is compromised — malicious app with sufficient privilege, OS-level surveillance, malware in the Eidola binary's host process — most user-facing privacy properties fall, regardless of what Eidola does. The client's fail-safe defaults and embedded trust root limit *what code runs*, but they cannot prevent an already-trusted process from observing the user. This is named explicitly in [privacy-guarantees.md §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise).

### N2. A hardware-manufacturer forgery

If AMD, Intel, or NVIDIA were to issue fraudulent attestation chains (for example, signing measurements for an enclave that does not in fact provide confidential compute), the corresponding layer of the verification chain falls. This is residual trust we currently accept; the long-term mitigation is open hardware roots like OpenTitan, tracked in [gaps.md](gaps.md).

### N3. A traffic-analysis adversary

An adversary observing network metadata (connection patterns, sizes, timing) can infer a great deal even when content is encrypted. Eidola hides the size and timing of an answer's individual tokens from an observer between the client and the gateway ([privacy-guarantees.md §3.8](privacy-guarantees.md#3-content)) and nothing more: connection patterns, network identity, request sizes and response durations remain visible. Users who need metadata privacy should layer Eidola behind Tor or a similar anonymity network.

### N4. A user who voluntarily leaks

A user who pastes their email into a prompt, copies inference output to a third-party service, or screenshots their chat history has exited Eidola's privacy boundary by their own action. Eidola does not redact, classify, or filter user input.

### N5. Microarchitectural side channels against confidential-compute hardware

Confidential-compute hardware proves which code is running in an enclave. It does not prove that the host platform is free of microarchitectural side channels (speculative-execution leaks, cache timing, power analysis, fault injection). Published attacks on SEV-SNP and TDX show this is an active research area. Mitigations live in firmware and microcode: every attestation is held to a TCB floor, on the gateway's SEV-SNP hosts and on the engines' TDX hosts alike. The engines' hosts may run with SMT enabled, which admits cross-thread channels ([gaps.md](gaps.md#tdx-acceptance)). Eidola does not promise that no further vulnerability will be found.

## Residual trust, named explicitly

Every system rests on something it cannot verify from inside itself. Ours is listed so a reader can decide whether it is acceptable for their threat model:

| Trust anchor | What we rely on it for | What lowers the exposure |
|---|---|---|
| **Confidential compute vendors** (AMD, Intel, NVIDIA) | The attestation chains prove real enclave execution | An inference request depends on all three: AMD for the gateway's enclave, Intel and NVIDIA for the engine's. Nothing lowers this today; open hardware roots would over time. See [N2](#n2-a-hardware-manufacturer-forgery) and [N5](#n5-microarchitectural-side-channels-against-confidential-compute-hardware) |
| **The hosting platform** (Tinfoil) | The boot images the enclaves launch, and its signed endorsements of which hosts an engine may run on and the firmware levels they must meet | The images are pinned by hash and measured, and the endorsements are verified and committed with each gateway build, so neither can change silently. See [gaps.md](gaps.md#build-chain-opacity) and [gaps.md](gaps.md#tdx-acceptance) |
| **WebPKI** (Let's Encrypt) | The TLS cert presented by the gateway is genuinely the one issued for the hostname | The cert is bound to a fresh enclave attestation by the client (the report's `REPORT_DATA` commits to `tls_key_fp == sha256(SPKI(peer_cert))` alongside the per-handshake nonce); a forged WebPKI cert alone is not enough |
| **Sigstore Rekor** (Linux Foundation) | The transparency log entry for a release attestation is genuine and never removed | Rekor public keys are pinned in the client via `sigstore-trusted-root.json`, so a WebPKI MITM against `rekor.sigstore.dev` still can't sign valid Rekor responses. If Rekor is unreachable, update verification fails closed. Checkpoint signature verification is a [gap](gaps.md#rekor-checkpoint-signature-verification) |
| **GitHub OIDC** (Fulcio identity binding) | The release-signing CI workflow ran under the identity it claims | Pinned identity pattern + tag, plus the human attestation requirement makes OIDC compromise alone insufficient. Manual escape via [rotation](../releases/README.md#rotating-the-ci-signing-workflow) |
| **The user's prior client binary** | Embedded trust root has not been silently subverted before install | Public release record + signed continuity check between releases |
| **The user's hardware and OS** | Process isolation, key storage, code execution integrity | Outside Eidola's scope; named in [N1](#n1-a-compromised-local-environment) |

Each of these is a place where a capable adversary could break the chain. Where mitigations are in progress, they are in [gaps.md](gaps.md).
