# Threat Model

This document names who Eidola defends against, who it does not, and what is left as residual trust. It is deliberately complete on the *scope* side — knowing what we **do not** protect against is part of knowing what protection means.

For each adversary, the question is the same: *given the [privacy guarantees](privacy-guarantees.md), what can this adversary do, and what stops them?*

## Adversaries we defend against

### A1. A curious or malicious Eidola operator

**Capability:** Operates Eidola's account-management infrastructure and could in principle observe its logs, databases, and network.

**What stops them:** The boundary between the *linked* account surface and the *unlinked* inference surface is enforced at the type level in the server, and the inference surface uses anonymous credentials (Privacy Pass ACT). The operator can see *that* a credential was redeemed, never *whose* credential it was. Inference content never reaches the operator's logging surface at all. Telemetry on the inference path is restricted to model name, token counts, status, and latency.

**Residual exposure:** The operator could attempt to deploy a server binary that violates these properties. The client refuses to talk to any server whose attestation does not match the measurement pinned in the client build, so deploying a violating server requires the operator to also ship a violating *client release*. That path is blocked by the release-integrity invariants in [privacy-guarantees.md §6](privacy-guarantees.md#6-release-integrity) — in particular, the embedded trust root (§6.5), the human attestation requirement (§6.2), the attestant's no-known-backdoor claim (§6.3), and the no-compulsion claims (§6.4).

### A2. A passive network observer

**Capability:** Sees all packets between the user's device and Eidola's servers, including TLS metadata (SNI, IP, timing).

**What stops them:** TLS confidentiality and integrity. (The deeper attestation binding that protects against an *active* attacker with a valid WebPKI cert is described in [A3](#a3-an-active-network-attacker-mitm) — a passive observer has nothing to gain there, since they can't impersonate the server in the first place.)

**Residual exposure:** Network metadata (the fact that a connection occurred, its size, its timing) remains visible. Eidola does not defend against traffic analysis. Users who need that property need to layer Eidola behind Tor or a similar anonymity network. See [gaps.md](gaps.md) for ongoing work.

### A3. An active network attacker (MITM)

**Capability:** Can intercept, modify, and re-route traffic between the user's device and Eidola.

**What stops them:** Mutual binding between the TLS handshake and a *fresh* enclave attestation. The client issues an inline `GET /.well-known/tinfoil-attestation?nonce=<hex>` (a fresh random nonce per handshake) over the same TCP+TLS connection as the real request, and verifies that the report's `REPORT_DATA` equals `sha256(tls_key_fp ‖ hpke_key ‖ nonce ‖ …)` with `tls_key_fp == sha256(SPKI(peer_cert))`. An attacker presenting a valid certificate for the same hostname without the matching attestation fails the check; an attacker presenting a stale or replayed attestation document fails either the SPKI binding or the nonce-freshness check.

**Residual exposure:** The per-handshake nonce guarantees *freshness* — the report was generated for this request by a live, genuine CC machine that currently holds the cert's key — which defeats replay of a stale or captured document. It does **not**, on its own, defeat an attacker who has **exfiltrated** the enclave's TLS private key. The report binds the enclave's long-term TLS *key* (the cert SPKI), not the live TLS *session*, so such an attacker can terminate the client's connection with the stolen key — reading the plaintext, since an active MITM who holds the signing key derives the session keys regardless of TLS 1.3 forward secrecy — while relaying a fresh, nonce-bound report fetched from the enclave's public attestation endpoint (which serves any nonce). Every client check still passes. The residual exposure is therefore the same as before the nonce existed: it rests on the TLS key staying sealed inside the enclave by the confidential-compute runtime. Fully removing it would require channel binding — committing a TLS-session value (e.g. an RFC 5705 exporter) into `report_data` rather than only the cert key.

### A3.5. A rogue deployer

**Capability:** Anyone with deploy access to the legitimate Eidola infrastructure — a confidential-compute platform operator (e.g. Tinfoil), an Eidola employee with deploy credentials, or an attacker who has compromised one — could attempt to alter what runs: substitute the binary, change injected secrets, point the server at a different upstream, or roll back to a known-vulnerable older Eidola server build.

**What stops them:** The server's behavior is fully determined by the **attested configuration** (`tinfoil-config.yml`), which is bound into the enclave measurement via the kernel command line. This covers the server image digest, the argument list and environment-variable schema, and hashes of any injected secrets the server enforces at boot. Any change produces a different measurement, which the client refuses to connect to. The client:server pairing (see [client.md](client.md#one-release-pairs-exactly-one-client-with-one-server)) additionally blocks rollback: each client release pins exactly one server measurement, so even a genuinely-attested *older* server build doesn't match the pin compiled into an updated client.

**Residual exposure:** Configuration that *isn't* committed into the attested boot path is, by definition, not bound into the measurement. The defense scales with the rigor of what's actually committed; the same discipline has to extend to any new sensitive configuration we add.

### A3.6. A server impersonator

**Capability:** Operates a server pretending to be Eidola from somewhere outside the legitimate infrastructure — DNS hijack, BGP redirection, a server the attacker controls at the real hostname, or a fake stack not running confidential compute at all. This generalizes the active-network MITM of [A3](#a3-an-active-network-attacker-mitm), which is one route by which an impersonator can redirect traffic to their fake endpoint.

**What stops them:** The same controls as [A3.5](#a35-a-rogue-deployer). The client's pinned measurement names exactly one valid server build; only an enclave actually running that build can produce a matching attestation, and only confidential-compute hardware with valid vendor signing keys can produce a real attestation at all. **WebPKI is an additional layer**: an impersonator without a valid TLS cert for the real Eidola hostname can't reach the attestation step at all, while an impersonator who *has* a fraudulent cert (or who tricks the user into a lookalike hostname) still fails the measurement check on a real or fake enclave.

**Residual exposure:** Same as A3: the per-handshake nonce proves freshness but binds the cert *key*, not the TLS *session*, so an attacker who has exfiltrated the TLS private key can still MITM — reading the session with the stolen key and relaying a fresh nonce-bound report from the enclave's public endpoint. Bounded by the hardware-sealed-key property; closing it fully would require channel binding.

### A4. An attacker who compromises the Eidola release pipeline

**Capability:** Has gained write access to the GitHub repo, the CI system, or the OIDC identity used to sign artifact manifests.

**What stops them:** The release pipeline's signature alone is not sufficient for the client to accept a release. The client *also* requires a [human attestation](releases.md) signed by a pinned attestant from hardware under their exclusive physical control, posted to a public transparency log (Rekor). An attacker who controls CI cannot mint that signature. An attacker who has additionally compromised an attestant's hardware-bound key has crossed a higher bar; the verifier enforces a minimum number of independent attestations, raising the cost as more attestants come online.

**Residual exposure:** A compromise of every pinned attestant simultaneously, sufficient to extract or coerce each hardware-bound signing key, defeats this. The mitigation is distribution of attestants across hardware tokens, jurisdictions, and physical custody, plus a public minimum threshold pinned in the *prior* client so the bar cannot be lowered by the compromise itself.

### A5. A compromised dependency (supply chain)

**Capability:** Has subverted a third-party crate, container base image, or build tool that Eidola depends on.

**What stops them:** The structural defenses, in order of how much of the attack surface they actually close:

- **Pinned dependency surface.** Every dependency has a fixed version and an expected hash. Updates are explicit commits.
- **Minimal runtime dependencies.** Distribution of compiled binaries limits exposure to outdated or vulnerable libraries on the end-user's machine.
- **Source-bootstrapped reproducible builds on Linux** (StageX). This eliminates an entire class of "what was actually in the toolchain" risks: every binary in the chain — down to the assembly that built the c compiler — is built from source whose hash is committed and verified.
- **Hermetic builds on macOS** (Nix). This is *hermetic*, not fully source-bootstrapped in the StageX sense, but provides similar properties for components of the build environment above the OS. The macOS operating system and parts of its toolchain are opaque inputs we must accept, because cross-compiling to macOS from Linux is not viable today. See [gaps.md](gaps.md#build-chain-opacity).
- **A committed `artifact-manifest.json` as a merge invariant.** CI re-derives the manifest from source on every PR and refuses to merge if the result differs from the committed copy. This makes reproducibility enforced in the PR flow itself, not just at release time.
- **Personal reproduction in the release attestation.** Every release attestation includes a claim that the attestant has personally reproduced the manifest from source on hardware under their control.

**Residual exposure:** A dependency compromise that occurred, was incorporated, and was not caught by any of the above (PR review, attestant diff review, downstream public review) is the remaining gap. We supplement with operational practices — Dependabot for CVE monitoring, preference for pure-Rust dependencies where it reduces audit surface, minimal direct dependency count — but these are normal-best-practice rather than structural defenses.

### A6. A legally-compelled Eidola engineer

**Capability:** A court order, technical capability notice, or other legal compulsion directs an engineer to weaken the privacy guarantees or introduce a backdoor.

**What stops them:** The release-attestation no-compulsion claims ([privacy-guarantees.md §6.4](privacy-guarantees.md#6-release-integrity)) require the attestant to sign, under their legal identity, that they are *not* under such compulsion. A coerced engineer who is also gagged must either (a) decline to sign, breaking the release and sending a public signal, or (b) sign falsely and incur the legal exposure they were trying to avoid. As attestant counts grow, the required-threshold check forces the adversary to compel multiple independent engineers, multiplying the legal and operational risk on the adversary's side.

**Residual exposure:** A jurisdiction able to compel every pinned attestant in coordinated secrecy defeats this silently. The mitigation is multi-jurisdiction attestant distribution; this is [ongoing work](gaps.md).

## Adversaries we do not defend against

### N1. A compromised local environment

If the user's device is compromised — malicious app with sufficient privilege, OS-level surveillance, malware in the Eidola binary's host process — most user-facing privacy properties fall, regardless of what Eidola does. The client's fail-safe defaults and embedded trust root limit *what code runs*, but they cannot prevent an already-trusted process from observing the user. This is named explicitly in [privacy-guarantees.md §8.1](privacy-guarantees.md#8-bounded-claims-what-this-document-does-not-promise).

### N2. A hardware-manufacturer forgery

If AMD, Intel, or NVIDIA were to issue fraudulent attestation chains (for example, signing measurements for an enclave that does not in fact provide confidential compute), the corresponding layer of the verification chain falls. This is residual trust we currently accept; the long-term mitigation is open hardware roots like OpenTitan, tracked in [gaps.md](gaps.md).

### N3. A traffic-analysis adversary

An adversary observing network metadata — connection patterns, packet sizes, timing — can infer a great deal even when content is encrypted. Eidola does not defend against this. Users who need metadata privacy should layer Eidola behind Tor or a similar anonymity network.

### N4. A user who voluntarily leaks

A user who pastes their email into a prompt, copies inference output to a third-party service, or screenshots their chat history has exited Eidola's privacy boundary by their own action. Eidola does not redact, classify, or filter user input.

## Residual trust, named explicitly

Every system has a foundation it cannot verify from inside itself. Ours is named so a reader can decide whether the foundation is acceptable for their threat model:

| Trust anchor | What we rely on it for | What lowers the exposure |
|---|---|---|
| **Confidential compute vendors** (AMD, Intel, NVIDIA) | The attestation chain proves real enclave execution | A specific deployment is attested under one vendor's root, not both; supporting multiple vendors broadens the *deployment* surface, not the per-connection trust surface. Future OpenTitan-style roots would reduce vendor count over time. See [N5](#n5-microarchitectural-side-channels-against-confidential-compute-hardware) for the hardware-vulnerability side |
| **WebPKI** (Let's Encrypt) | The TLS cert presented by the server is genuinely the one issued for the hostname | The cert is bound to a fresh enclave attestation by the client (the report's `REPORT_DATA` commits to `tls_key_fp == sha256(SPKI(peer_cert))` alongside the per-handshake nonce); a forged WebPKI cert alone is not enough |
| **Sigstore Rekor** (Linux Foundation) | The transparency log entry for a release attestation is genuine and never removed | Rekor public keys are pinned in the client via `sigstore-trusted-root.json`, so a WebPKI MITM against `rekor.sigstore.dev` still can't sign valid Rekor responses. If Rekor is unreachable, update verification fails closed. Checkpoint signature verification is a [gap](gaps.md#rekor-checkpoint-signature-verification) |
| **GitHub OIDC** (Fulcio identity binding) | The release-signing CI workflow ran under the identity it claims | Pinned identity pattern + tag, plus the human attestation requirement makes OIDC compromise alone insufficient. Manual escape via [rotation](../releases/README.md#rotating-the-ci-signing-workflow) |
| **The user's prior client binary** | Embedded trust root has not been silently subverted before install | Public release record + signed continuity check between releases |
| **The user's hardware and OS** | Process isolation, key storage, code execution integrity | Outside Eidola's scope; named in [N1](#n1-a-compromised-local-environment) |

Each of these is a place where a sufficiently motivated and capable adversary could break the chain. They are not weaknesses we are hiding; they are the cost of building software at all. Where we have in-progress mitigations, they are in [gaps.md](gaps.md).

### N5. Microarchitectural side channels against confidential-compute hardware

Confidential-compute hardware proves *which code is running* in an enclave; it does not prove that the host platform is free of microarchitectural side channels (Spectre-class branch prediction leaks, cache-timing attacks, power-analysis, fault-injection, etc.). Past CVEs against SEV-SNP and TDX have shown this is a live research area. Mitigations are at the firmware-and-microcode layer — the verifier enforces a TCB floor (bl ≥ 0x07, snp ≥ 0x0e, ucode ≥ 0x48) on every connection — but Eidola does not promise that no future vulnerability will be discovered. A reader for whom this is the dominant concern should weigh it against the remote-compute threat surface as a whole, not against Eidola specifically.
