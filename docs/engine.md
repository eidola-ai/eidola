# Inference engine

The model itself does not run inside the Eidola server. It runs on Eidola's own inference engine (`eidola-server-engine`), in separate confidential-compute deployments on GPU hosts, which the Eidola server attests on every outbound connection against deployments pinned in its own measured build.

This page explains what runs where, what the user is trusting at this layer, and how the trust is anchored.

## Where the model runs

An engine deployment:

- Runs Eidola's engine in an Intel TDX confidential VM with NVIDIA confidential-computing GPUs, deployed as a [Tinfoil](https://tinfoil.sh) Container, the same platform that hosts the Eidola server. The VM boots the GPU variant of the platform's firmware-less IGVM image, a release line the platform still marks pre-release. That image carries the NVIDIA driver inside its measurement, and its boot checks the GPUs' device IDs, runs NVIDIA's attestation verifier on each of them inside the VM, and opens them to work only after it has passed for exactly the number of GPUs the configuration attaches. The engine then refuses to start unless the driver reports its GPUs in production confidential-computing mode, with DevTools mode off.
- Is described by a committed configuration, `deploy/engine/<model-id>/<variant>/tinfoil-config.yml`: the engine image by digest, the read-only weights pack, the engine's whole environment, and the hash of the one secret it takes (the token the Eidola server presents). The configuration's SHA-256 is part of the VM's launch identity (MRCONFIGID), and the guest refuses to boot under any other configuration.
- Serves its TLS endpoint from inside the VM and exposes the same self-contained v3 attestation envelope the Eidola server uses, binding a fresh challenge, the endpoint's TLS SPKI and the GPU evidence into a hardware report.
- Serves exactly one model, from weights it hashes when it starts. Every request names the weights hash it expects, and a node serving other weights refuses the request before it is admitted.

The Eidola server is a *client* of these deployments. It verifies each one's attestation on every TCP+TLS connection it opens, using the same `tinfoil-verifier` crate the Eidola client uses to verify the Eidola server.

**No deployment is pinned yet.** Until the first one is measured and pinned, the server sells no models: `/v1/models` lists nothing, and every chat request is refused as an unknown model. A catalog row, its price and its pin land together, and the server's build refuses a row without a pin or a pin without a row.

## What pins the engine

The engine deployments a server build accepts are a build input, `releases/trust/engine-enclaves.json`, compiled into the server and therefore bound into its own enclave measurement, which the client pins. For each model it lists every accepted deployment's:

- **Launch identity**: MRTD, which the TDX module measures over every page of the platform's release image, and MRCONFIGID, the SHA-256 of the deployment's committed configuration. `measure-enclave` recomputes MRTD from the platform's published image, checked against its release manifest, and the configuration hash from the committed bytes.
- **Machine policy and machines**, for the report fields an image cannot fix: the TDX module, exact TD attributes (debug off), the TCB floors and the QE vendor, and the individual hosts allowed, by the PPID in each host's Intel-signed PCK certificate. These come from the hosting platform's Sigstore-signed platform endorsements, which map each of its machines to a named policy: a deployment names one policy of one release, that release is committed with its hash, and its signature is verified offline. The PCK platform flags, which the endorsements do not carry, are stated by the deployment. See [gaps.md § TDX acceptance](gaps.md#tdx-acceptance) for the full list and what it leaves.
- **GPU count**: how many NVIDIA GPU evidence items each handshake must carry.
- **Weights**: the engine's weights hash, and the repository and revision the weights were taken from.

Every server build re-derives that file from the committed configurations in both directions: a pin that disagrees with the configuration it names or with the platform endorsements its deployment names, or a committed deployment nobody pinned, fails the build. No build reads the network: the endorsements are verified against the committed Sigstore trusted root, and a newer release is adopted only by committing it (`just engine-platform-endorsements`), which a scheduled check prompts. The same build check holds each configuration to what an engine needs (one container, pinned by digest, with exactly the engine's measured environment and no other secret).

The set is **same-sha**: a server build accepts exactly the deployments committed in its own tree. Nothing widens it at runtime, there is no list of earlier measurements, and nothing is fetched to decide what to trust. Rolling a deployment forward is therefore ordered (new engines, then a server pinning them, then the client pinning that server), and is described in [`releases/README.md`](../releases/README.md#rolling-engine-deployments).

Where engines are reachable is operational data the server reads from its database, and it is availability only: an address is used only if it names a deployment this build pins, and the attestation on every connection to it decides whether it is that deployment.

## What the user is trusting at this layer

The user is trusting, in addition to the layers covered in [client.md](client.md) and [server.md](server.md):

1. **That the model runs in confidential compute.** This is verifiable: each deployment attests a launch identity that the Eidola server checks against the pins in its own build, and the server's build is in turn pinned by the client.
2. **That the deployment's confidential-compute hardware is genuine.** The same trust as for the Eidola server's own enclave, rooted in the hardware vendors (Intel for the CPU, NVIDIA for the GPUs) and their attestation chains.
3. **That the GPUs are in confidential-computing mode.** The Eidola server counts the per-handshake GPU evidence and checks each item's nonce label, which the measured attestation shim writes beside NVIDIA's signed report, inside the section the CPU quote's signed report data covers. It does not check the nonce inside NVIDIA's report, or appraise the evidence against NVIDIA's certificate chain or reference measurements. What vouches for the GPUs is the measured guest, at boot: the platform's GPU image, which opens the GPUs only after NVIDIA's verifier has passed for each of them, and the engine's own check, before it opens a GPU, that the driver reports confidential computing on, DevTools mode off, the production environment and the GPUs accepting work ([privacy-guarantees.md §3.5](privacy-guarantees.md#3-content)).
4. **That the platform's endorsements of its hosts are sound.** Which machines an engine may run on, and the TDX module and TCB floors they must meet, are the platform's own signed statement, as of the release committed in the server's tree. Eidola verifies that the platform signed it, not that its choices are right, and a newer release takes effect only once it is committed ([gaps.md § TDX acceptance](gaps.md#tdx-acceptance)).
5. **That the platform's boot image is what its release says.** MRTD is recomputed from the platform's published image. The platform builds that image reproducibly with Nix from pinned inputs, which include binary Ubuntu packages and NVIDIA's driver, CUDA and firmware; it is not source-bootstrapped, Eidola has not rebuilt it, and its release line is still a pre-release ([gaps.md § Build chain opacity](gaps.md#build-chain-opacity)).

## What the user is *not* trusting

- **Eidola does not trust a provider's release process for the inference code.** The engine is Eidola's code, in this repository, and the deployment it runs in is pinned by measurement in the server's own build. No enclave on the inference path is attested by anyone but Eidola.
- **Eidola does not trust the hosting platform's operators with cleartext inference data.** Cleartext is *necessarily* visible to the VM performing the inference; that is how the model reads a prompt. The trust boundary at this layer is the confidential VM and its GPUs: the same properties that seal the Eidola server enclave against its operators seal the engine against the host's.

## Per-connection verification

The Eidola server's outbound HTTPS client for each engine (`tinfoil-verifier::attesting_client`, built over that engine's model's pins) re-verifies the deployment on every new TCP+TLS handshake. The mechanics are those of the client→server path, on the TDX branch:

- Inline `GET /.well-known/tinfoil-attestation?nonce=<hex>` (a fresh random nonce per handshake) over the same TCP+TLS connection that will carry the application request.
- Strict envelope and hash checks, nonce freshness, and endorsed TLS SPKI comparison with the peer certificate.
- TDX quote verification to the pinned Intel root with the Intel collateral the document carries, entirely offline, then the pinned launch identity (MRTD, MRCONFIGID, zero RTMRs), machine policy, and the host's PPID in the pinned allowlist.
- Binding of the quote's `REPORT_DATA` to the verifier nonce and the exact endorsed-section hashes, then exactly the pinned number of GPU evidence items, each labelled with this handshake's nonce.

A failed attestation rejects the request before any inference data crosses the wire. Before an engine takes traffic, and periodically after, the server also asks it (over an attested connection) which model and weights it serves, and refuses to route to one that reports anything but the pinned model, the pinned weights hash and read-only verified weights storage.
