# Inference Upstream

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

1. **That the model itself runs in confidential compute.** This is verifiable: the inference upstream attests to a measurement that the Eidola server checks against a pinned set of allowed measurements (`releases/trust/tinfoil-enclaves.json`).
2. **That the model code's published measurements match the published source.** The inference provider publishes signed measurements via Sigstore against a specific source repository (e.g. `tinfoilsh/confidential-model-router`). The Eidola server could in principle re-verify this provenance; today it relies on the pinned-measurement list and Sigstore verification on measurement updates.
3. **That the upstream provider's confidential-compute deployment is genuine.** This is the same trust as for the Eidola server's own enclave — ultimately rooted in the hardware vendor (AMD, Intel, or NVIDIA) and its attestation chain.

## What pins the upstream measurement

The list of allowed upstream enclave measurements lives in `releases/trust/tinfoil-enclaves.json`. It is a build input to the Eidola server: `crates/eidola-server/build.rs` reads it and generates a static `ALLOWED` slice into the server binary. The server refuses to connect to any upstream enclave whose measurement is not in this slice.

The file is updated by the `.github/workflows/update-measurements.yml` workflow, which:

- Pulls the latest measurements from the upstream's published release feed.
- Verifies the provenance via Sigstore (Fulcio + Rekor) against the expected repository identity.
- Opens a PR that adds the new measurement and removes any measurements older than the rolling-deploy window.

A new upstream measurement does not silently become trusted: it goes through the same review-and-merge process as any other source change, and the resulting Eidola server *build* embeds the new list.

## What the user is *not* trusting

- **Eidola is not trusting the upstream provider's policy.** The trust is in the running code's measurement, not in any contractual or operational commitment from the provider. If the provider were to ship a build that violates its claimed properties, the *measurement would change* and Eidola's server would refuse to connect until that measurement was reviewed and added to the allowed set.
- **Eidola is not trusting the upstream provider's operators with cleartext inference data.** Cleartext inference data is *necessarily* visible to the enclave performing the inference — that's how the model reads your prompt and generates a response. The trust boundary at this layer is the enclave itself, not the provider's operations team: the same confidential-compute properties that seal the Eidola server enclave against Eidola's operators seal the inference enclave against the upstream provider's operators.

## Per-connection verification

The Eidola server's outbound HTTPS client (constructed by `tinfoil-verifier::attesting_client`) re-verifies the upstream enclave on every new TCP+TLS handshake. The mechanics are the same as for the client→server path, because they use the same crate:

- Inline `GET /.well-known/tinfoil-attestation?v=3` over the same TCP+TLS connection that will carry the application request.
- AMD VCEK chain verification, SEV-SNP / TDX report verification, TCB policy enforcement.
- Measurement check against the pinned allowed set.
- Binding of `report_data[0..32]` to `sha256(SPKI(peer_cert))`.

A failed attestation rejects the request before any inference data crosses the wire.

## Why a separate enclave at all

A reasonable question: why does the model run in a *different* enclave from the Eidola server? The full answer is partly structural and partly transitional.

**Structurally**, confidential-compute infrastructure for serving large language models requires specialized hardware (GPUs with NVIDIA confidential compute) and operational expertise that dedicated inference providers can supply most cleanly. Eidola's role is the privacy and account layer around the inference, not the inference itself.

**Transitionally**, the upstream-provider model has the appealing property that the user's trust chain at the inference layer ends at a measurement signed against the *upstream's* source — which that source can be audited against independently of Eidola.

Two caveats apply to that second framing today:

- Tinfoil's release process is robust — signed measurements, Sigstore provenance, public source — but it does not yet match Eidola's: in particular, Tinfoil's builds are not fully source-bootstrapped reproducible in the StageX sense, and release attestation rides on GitHub's CI attestations rather than per-release human attestations under named legal identities. So "independent" is true at the boundary (different code, different signers) but the audit surface on the upstream side is shaped differently than ours.
- The planned future state is to bring the inference pipeline into this repo (still running on Tinfoil's infrastructure), so the same release-attestation discipline applies end-to-end. That is on the roadmap, not in the current release.

The cost of the current split is one additional verification step (Eidola server → upstream) on each inference, which adds a small per-handshake latency cost on top of the connection-pooled normal request path.

The cvmimage / OVMF non-determinism caveat moved to [gaps.md#build-chain-opacity](gaps.md#build-chain-opacity), since it cuts across the whole server-side trust chain (not just inference upstream).
