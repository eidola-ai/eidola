# Inference upstream

The model itself does not run inside the Eidola server. It runs in a separate confidential-compute deployment operated by an inference provider (currently [Tinfoil](https://tinfoil.sh)), with its own attestation chain that the Eidola server verifies on every outbound connection.

This page explains what runs where, what the user is trusting at this layer, and how the trust is anchored.

## Where the model runs

The upstream inference provider:

- Runs an OpenAI-compatible API in a confidential-compute enclave (currently AMD SEV-SNP; Intel TDX support is tracked).
- Publishes signed measurements of the running enclave through Sigstore (Fulcio identity + Rekor inclusion), tied to a public source repository.
- Serves its TLS endpoint from inside the enclave with attestation encoded in the certificate SANs, the same construction the Eidola server itself uses.

The Eidola server is a *client* of this enclave. It verifies the upstream's attestation on every TCP+TLS connection it opens to the inference endpoint, using the same `tinfoil-verifier` crate the Eidola client uses to verify the Eidola server.

## What the user is trusting at this layer

The user is trusting, in addition to the layers covered in [client.md](client.md) and [server.md](server.md):

1. **That the model itself runs in confidential compute.** This is verifiable: the inference upstream attests to a measurement that the Eidola server checks against the set of measurements it currently trusts (resolved at runtime — see below).
2. **That the model code's published measurements match the published source.** The inference provider publishes signed measurements via Sigstore against a specific source repository (e.g. `tinfoilsh/confidential-model-router`). The Eidola server verifies this provenance itself: it resolves the latest release's measurement and verifies its Sigstore attestation end-to-end before trusting it.
3. **That the upstream provider's confidential-compute deployment is genuine.** This is the same trust as for the Eidola server's own enclave — ultimately rooted in the hardware vendor (AMD, Intel, or NVIDIA) and its attestation chain.

## What pins the upstream measurement

The set of allowed upstream enclave measurements is resolved **at runtime**, not baked into the binary. `inference.tinfoil.sh` is a *router* enclave that reverse-proxies to separate per-model GPU enclaves, and the router trusts those downstream enclaves via "the latest Sigstore-signed release of the repo" — so statically pinning the router's measurement (and gating changes on a human PR) bought little rigor over the upstream provider's own trust model while causing a fail-closed outage every time the provider shipped a router release. Instead, `crates/eidola-server/src/upstream_trust` matches the provider's actual model: on boot and every ~10 minutes it resolves the provider's latest release and verifies its Sigstore attestation end-to-end before trusting the measurement:

- Fetches the latest release tag, its subject digest (`tinfoil.hash`), and the GitHub artifact attestation (a Sigstore `dsse`/in-toto bundle).
- Verifies the Fulcio certificate chain to the pinned Sigstore trusted root (`releases/trust/sigstore-trusted-root.json`, the sole `releases/` input the server's `build.rs` embeds), the signing identity (GitHub Actions OIDC issuer + the expected repository **and exact tag**), the DSSE signature, and the Rekor transparency-log entry (SET + inclusion proof).
- Reads the measurement from the signed `snp-tdx-multiplatform/v1` predicate and confirms the subject digest matches `tinfoil.hash`.

The allowed set is a rolling window of the two most recently resolved measurements (so an in-progress rolling deploy still attests). This holds from the very first boot: bootstrap resolves the latest release (the fatal readiness gate) and also folds in the immediately-previous **published** release (best-effort — a missing or unverifiable previous release just logs a warning and boots latest-only), so a cold start landing mid-deploy attests the still-draining old router enclave rather than aborting startup. A resolution or verification failure never clears or widens trust — the server keeps its current set, and if it *can't* resolve a verified measurement at boot it refuses to start (there is no static fallback). This whole subsystem is transitional: when Eidola self-hosts inference it is replaced by a statically pinned measurement set.

## What the user is *not* trusting

- **Eidola is not trusting the upstream provider's policy.** The trust is in the running code's measurement, not in any contractual or operational commitment from the provider. A measurement is only trusted once its Sigstore provenance verifies against the expected repository identity; Eidola's server refuses to connect to any enclave whose measurement it hasn't resolved and verified this way.
- **Eidola is not trusting the upstream provider's operators with cleartext inference data.** Cleartext inference data is *necessarily* visible to the enclave performing the inference — that's how the model reads your prompt and generates a response. The trust boundary at this layer is the enclave itself, not the provider's operations team: the same confidential-compute properties that seal the Eidola server enclave against Eidola's operators seal the inference enclave against the upstream provider's operators.

## Per-connection verification

The Eidola server's outbound HTTPS client (constructed by `tinfoil-verifier::attesting_client`) re-verifies the upstream enclave on every new TCP+TLS handshake. The mechanics are the same as for the client→server path, because they use the same crate:

- Inline `GET /.well-known/tinfoil-attestation?nonce=<hex>` (a fresh random nonce per handshake) over the same TCP+TLS connection that will carry the application request.
- Freshness check (echoed nonce matches) and document-signature check against the embedded TLS cert.
- AMD VCEK chain verification, SEV-SNP / TDX report verification, TCB policy enforcement.
- Measurement check against the runtime-resolved allowed set (see [What pins the upstream measurement](#what-pins-the-upstream-measurement)).
- Binding of the report's `REPORT_DATA` to `sha256(tls_key_fp ‖ hpke_key ‖ nonce ‖ …)`, where `tls_key_fp == sha256(SPKI(peer_cert))`.

A failed attestation rejects the request before any inference data crosses the wire.

## Why a separate enclave at all

A reasonable question: why does the model run in a *different* enclave from the Eidola server? The full answer is partly structural and partly transitional.

**Structurally**, confidential-compute infrastructure for serving large language models requires specialized hardware (GPUs with NVIDIA confidential compute) and operational expertise that dedicated inference providers can supply most cleanly. Eidola's role is the privacy and account layer around the inference, not the inference itself.

**Transitionally**, the upstream-provider model has the appealing property that the user's trust chain at the inference layer ends at a measurement signed against the *upstream's* source — which that source can be audited against independently of Eidola.

Two caveats apply to that second framing today:

- Tinfoil's release process is robust — signed measurements, Sigstore provenance, public source — but it does not yet match Eidola's: in particular, Tinfoil's builds are not fully source-bootstrapped reproducible in the StageX sense, and release attestation rides on GitHub's CI attestations rather than per-release human attestations under named legal identities. So "independent" is true at the boundary (different code, different signers) but the audit surface on the upstream side is shaped differently than ours. And because the router forwards to per-model enclaves it trusts at *latest-signed-release* (which Eidola does not independently attest), the code actually running inference is trusted at Tinfoil's bar, not re-derived by Eidola — the full shape of this is catalogued in [gaps.md § Inference upstream](gaps.md#inference-upstream).
- The long-term intent is to end the inference-layer trust chain where the rest of Eidola's does, by either of two paths (either suffices): **self-hosting inference** on GPU-enabled Tinfoil containers that Eidola controls, built through Eidola's own reproducible + human-attested flow so Eidola owns and statically pins the measurement (the goal regardless, but gated on GPU cost that is hard to justify pre-demand); or **Tinfoil adopting static-pinned upstreams** — if the router pins and publishes the specific downstream model-enclave measurements it forwards to, the downstream-review gap closes without Eidola self-hosting, and continuing to use Tinfoil's inference becomes the cheaper preferred long-term solution. Both are roadmap, not the current release.

The cost of the current split is one additional verification step (Eidola server → upstream) on each inference, which adds a small per-handshake latency cost on top of the connection-pooled normal request path.

The cvmimage / OVMF non-determinism caveat moved to [gaps.md#build-chain-opacity](gaps.md#build-chain-opacity), since it cuts across the whole server-side trust chain (not just inference upstream).
