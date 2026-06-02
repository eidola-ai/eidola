# Inference Upstream

The model itself does not run inside the Eidola server. It runs in a
separate confidential-compute deployment operated by an inference
provider (currently [Tinfoil](https://tinfoil.sh)), with its own
attestation chain that the Eidola server verifies on every
outbound connection.

This page explains what runs where, what the user is trusting at this
layer, and how the trust is anchored.

## Where the model runs

The upstream inference provider:

- Runs an OpenAI-compatible API in a confidential-compute enclave
  (currently AMD SEV-SNP; Intel TDX support is tracked).
- Publishes signed measurements of the running enclave through
  Sigstore (Fulcio identity + Rekor inclusion), tied to a public
  source repository.
- Serves its TLS endpoint from inside the enclave with attestation
  encoded in the certificate SANs, the same construction the Eidola
  server itself uses.

The Eidola server is a *client* of this enclave. It verifies the
upstream's attestation on every TCP+TLS connection it opens to the
inference endpoint, using the same `tinfoil-verifier` crate the
Eidola client uses to verify the Eidola server.

## What the user is trusting at this layer

The user is trusting, in addition to the layers covered in
[client.md](client.md) and [server.md](server.md):

1. **That the model itself runs in confidential compute.** This is
   verifiable: the inference upstream attests to a measurement that
   the Eidola server checks against a pinned set of allowed
   measurements (`releases/trust/tinfoil-enclaves.json`).
2. **That the model code's published measurements match the
   published source.** The inference provider publishes signed
   measurements via Sigstore against a specific source repository
   (e.g. `tinfoilsh/confidential-model-router`). The Eidola server
   could in principle re-verify this provenance; today it relies on
   the pinned-measurement list and Sigstore verification on
   measurement updates.
3. **That the upstream provider's confidential-compute deployment is
   genuine.** This is the same trust as for the Eidola server's own
   enclave — ultimately rooted in the hardware vendor (AMD, Intel,
   or NVIDIA) and its attestation chain.

## What pins the upstream measurement

The list of allowed upstream enclave measurements lives in
`releases/trust/tinfoil-enclaves.json`. It is a build input to the
Eidola server: `crates/eidola-server/build.rs` reads it and generates
a static `ALLOWED` slice into the server binary. The server refuses
to connect to any upstream enclave whose measurement is not in this
slice.

The file is updated by the
`.github/workflows/update-measurements.yml` workflow, which:

- Pulls the latest measurements from the upstream's published
  release feed.
- Verifies the provenance via Sigstore (Fulcio + Rekor) against the
  expected repository identity.
- Opens a PR that adds the new measurement and removes any
  measurements older than the rolling-deploy window.

A new upstream measurement does not silently become trusted: it goes
through the same review-and-merge process as any other source change,
and the resulting Eidola server *build* embeds the new list.

## What the user is *not* trusting

- **Eidola is not trusting the upstream provider's policy.** The
  trust is in the running code's measurement, not in any
  contractual or operational commitment from the provider. If the
  provider were to ship a build that violates its claimed
  properties, the *measurement would change* and Eidola's server
  would refuse to connect until that measurement was reviewed and
  added to the allowed set.
- **Eidola is not trusting the upstream provider with cleartext
  inference data.** The cleartext is necessarily visible to the
  enclave performing inference — there is no way to run a model
  without it. But that visibility is bounded to a verified enclave
  running attested code, not to the provider's operations team.

EDIT: The above piece needs to be worded better; it reads at first
as if no cleartext is sent to them (although it's clarified later).

## Per-connection verification

The Eidola server's outbound HTTPS client (constructed by
`tinfoil-verifier::attesting_client`) re-verifies the upstream
enclave on every new TCP+TLS handshake. The mechanics are the same
as for the client→server path, because they use the same crate:

- Inline `GET /.well-known/tinfoil-attestation?v=3` over the same
  TCP+TLS connection that will carry the application request.
- AMD VCEK chain verification, SEV-SNP / TDX report verification,
  TCB policy enforcement.
- Measurement check against the pinned allowed set.
- Binding of `report_data[0..32]` to `sha256(SPKI(peer_cert))`.

A failed attestation rejects the request before any inference data
crosses the wire.

## Why a separate enclave at all

A reasonable question: why does the model run in a *different*
enclave from the Eidola server? Two reasons:

1. **Concentration of capability.** Confidential-compute infrastructure
   for serving large language models requires specialized hardware
   (GPUs with NVIDIA confidential compute) and operational expertise
   that is currently provided most cleanly by dedicated inference
   providers. Eidola's role is the privacy *and account* layer
   around the inference, not the inference itself.
2. **Independent verification.** Putting the model in a separately-attested
   enclave means the user's chain of trust ends at the model
   provider's signed measurement, which the model provider's source
   code can be audited against independently of Eidola's code. This
   is a stronger property than "Eidola's operators promise the model
   is doing what it says."

The cost is one additional verification step (Eidola server → upstream)
on each inference, which adds a small per-handshake latency cost on
top of the connection-pooled normal request path.

EDIT: I don't think the "Independent verification" piece is particularly
strong. It's definitely something, and they have far more users than us
(we have zero), but I still don't think it's at the critical mass for
this to be a real selling point.Tinfoil's measures are extremely robust,
but they don't currently have fully deterministic source-bootstrapped
builds or quite the same human attestation process (as a side-effect of
the former), instead relying on GitHub's CI attestations for provenance.
We should note these caveats. I do plan on a future state where the entire
inference pipeline exists in this repo (still running in tinfoil's
infrastructure), but we aren't there today.

Edit: It's also true that cvmimage and ovmf lack fully deterministic builds,
and while their contents are pinned by the server's measurement, we're
trusting the build pipeline more than we ideally would. This probably belongs
in "gaps" or similar, as it's not part of the inference upstream. But worth
noting.
