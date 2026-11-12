# Inference engine

Models do not run inside the gateway. They run on Eidola's own inference engine (`eidola-server-engine`), in separate confidential-compute deployments on GPU hosts. The gateway attests each deployment on every outbound connection, against pins in its own measured build.

This page explains what runs where, how that layer is pinned, and what the user is trusting at it.

## Where the model runs

An engine deployment:

- Runs Eidola's engine in an Intel TDX confidential VM with NVIDIA confidential-computing GPUs, deployed as a [Tinfoil](https://tinfoil.sh) Container, the same platform that hosts the gateway. The VM boots the GPU variant of the platform's firmware-less IGVM image, a release line the platform still marks pre-release. That image carries the NVIDIA driver inside its measurement, and its boot checks the GPUs' device IDs, runs NVIDIA's attestation verifier on each of them inside the VM, and opens them to work only after it has passed for exactly the number of GPUs the configuration attaches. The engine then refuses to start unless the driver reports its GPUs in production confidential-computing mode, with DevTools mode off.
- Is described by a committed configuration, `deploy/engine/<name>/<variant>/tinfoil-config.yml`: the engine image by digest (built reproducibly from this repository; see [gaps.md § Build chain opacity](gaps.md#build-chain-opacity) for what its toolchain leaves opaque), the read-only weights pack, the engine's whole environment, and the hash of the one secret it takes (the token the gateway presents). The configuration's SHA-256 is part of the VM's launch identity (MRCONFIGID), and the guest refuses to boot under any other configuration.
- Serves its TLS endpoint from inside the VM and exposes the same self-contained v3 attestation envelope the gateway uses, binding a fresh challenge, the endpoint's TLS SPKI and the GPU evidence into a hardware report.
- Serves exactly one model, from weights it hashes when it starts, and is named by that hash. Every request names the weights hash it expects, and a node serving other weights refuses the request before it is admitted. The hash, and how to recompute it from the published files, is specified in [Model identity](model-identity.md).

The gateway is a *client* of these deployments. It verifies each one's attestation on every TCP+TLS connection it opens, using the same `tinfoil-verifier` crate the Eidola client uses to verify the gateway.

**One deployment is pinned, and none is running.** The catalog lists one model, MiMo-V2.6-Flash, and the source tree commits and pins one deployment of it, `deploy/engine/mimo-v2.6-flash/b300x1/` (one NVIDIA B300). That deployment has not been launched. A gateway built from this tree lists the model on `/v1/models`, and refuses every request for it with `503`, refunding all but the request fee, because no running engine is placed for it. A catalog row, its price and its pin land together, and the gateway's build refuses a row without a pin or a pin without a row.

## What pins the engine

The engine deployments a gateway build accepts are a build input, `releases/trust/engine-enclaves.json`, compiled into the gateway and therefore bound into its own enclave measurement, which the client pins. For each model, keyed by its weights hash, it lists every accepted deployment's:

- **Launch identity**: MRTD, which the TDX module measures over every page of the platform's release image, and MRCONFIGID, the SHA-256 of the deployment's committed configuration. `measure-enclave` recomputes MRTD from the platform's published image, checked against its release manifest, and the configuration hash from the committed bytes.
- **Machine policy and machines**, for the report fields an image cannot fix: the TDX module, exact TD attributes (debug off), the TCB floors and the QE vendor, and the individual hosts allowed, by the PPID in each host's Intel-signed PCK certificate. These come from the hosting platform's Sigstore-signed platform endorsements, which map each of its machines to a named policy: a deployment names one policy of one release, that release is committed with its hash, and its signature is verified offline. The PCK platform flags, which the endorsements do not carry, are stated by the deployment. See [gaps.md § TDX acceptance](gaps.md#tdx-acceptance) for the full list and what it leaves.
- **GPU count**: how many NVIDIA GPU evidence items each handshake must carry.
- **Weights**: the engine's weights hash, the model architecture, and the repository and revision the weights were taken from. The gateway's catalog (one file per weights hash under `models/`) is bound to these pins one to one in the same build.

Every server build re-derives that file from the committed configurations in both directions: a pin that disagrees with the configuration it names, with the platform endorsements its deployment names, or with the MRTD the platform's release manifest states for the image the configuration names (the manifest is committed too, pinned by the configuration), or a committed deployment nobody pinned, fails the build. Before a release reaches the main branch, the image itself is downloaded and measured again, and the result must equal the committed pins. No build reads the network: the endorsements are verified against the committed Sigstore trusted root, and a newer release is adopted only by committing it (`just engine-platform-endorsements`), which a scheduled check prompts. The same build check holds each configuration to what an engine needs (one container, pinned by digest, with exactly the engine's measured environment and no other secret).

The set is **same-sha**: a gateway build accepts exactly the deployments committed in its own tree. Nothing widens it at runtime, there is no list of earlier measurements, and nothing is fetched to decide what to trust. Rolling a deployment forward is therefore ordered (new engines, then a gateway pinning them, then the client pinning that gateway), and is described in [`releases/README.md`](../releases/README.md#rolling-engine-deployments).

Where engines are reachable is operational data the gateway reads from its database, and it is availability only: an address is used only if it names a deployment this build pins, and the attestation on every connection to it decides whether it is that deployment.

## What the user is trusting at this layer

The user is trusting, in addition to the layers covered in [client.md](client.md) and [gateway.md](gateway.md):

1. **That the model runs in confidential compute.** This is verifiable: each deployment attests a launch identity that the gateway checks against the pins in its own build, and the gateway's build is in turn pinned by the client.
2. **That the deployment's confidential-compute hardware is genuine.** The same trust as for the gateway's own enclave, rooted in the hardware vendors (Intel for the CPU, NVIDIA for the GPUs) and their attestation chains.
3. **That the GPUs are in confidential-computing mode.** The gateway counts the per-handshake GPU evidence and checks each item's nonce label, which the measured attestation shim writes beside NVIDIA's signed report, inside the section the CPU quote's signed report data covers. It does not check the nonce inside NVIDIA's report, or appraise the evidence against NVIDIA's certificate chain or reference measurements. What vouches for the GPUs is the measured guest, at boot: the platform's GPU image, which opens the GPUs only after NVIDIA's verifier has passed for each of them, and the engine's own check, before it opens a GPU, that the driver reports confidential computing on, DevTools mode off, the production environment and the GPUs accepting work ([privacy-guarantees.md §3.5](privacy-guarantees.md#3-content)).
4. **That the platform's endorsements of its hosts are sound.** Which machines an engine may run on, and the TDX module and TCB floors they must meet, are the platform's own signed statement, as of the release committed in the gateway's tree. Eidola verifies that the platform signed it, not that its choices are right, and a newer release takes effect only once it is committed ([gaps.md § TDX acceptance](gaps.md#tdx-acceptance)).
5. **That the platform's boot image is what its release says.** MRTD is recomputed from the platform's published image. The platform builds that image reproducibly with Nix from pinned inputs, which include binary Ubuntu packages and NVIDIA's driver, CUDA and firmware; it is not source-bootstrapped, and its release line is still a pre-release ([gaps.md § Build chain opacity](gaps.md#build-chain-opacity)).

## What the user is *not* trusting

- **A third party's inference code or release process.** The engine is Eidola's code, in this repository, and each deployment is pinned by measurement in the gateway's own build. Eidola attests every enclave on the inference path itself.
- **The hosting platform's operators, with cleartext.** Cleartext is necessarily visible inside the VM performing the inference. The boundary at this layer is the confidential VM and its GPUs, which are sealed against the host's operators the same way the gateway's enclave is.

## Per-connection verification

The gateway's outbound HTTPS client for each engine (`tinfoil-verifier::attesting_client`, built over that engine's model's pins) re-verifies the deployment on every new TCP+TLS handshake. The mechanics are those of the client→server path, on the TDX branch:

- Inline `GET /.well-known/tinfoil-attestation?nonce=<hex>` (a fresh random nonce per handshake) over the same TCP+TLS connection that will carry the application request.
- Strict envelope and hash checks, nonce freshness, and endorsed TLS SPKI comparison with the peer certificate.
- TDX quote verification to the pinned Intel root with the Intel collateral the document carries, entirely offline, then the pinned launch identity (MRTD, MRCONFIGID, zero RTMRs), machine policy, and the host's PPID in the pinned allowlist.
- Binding of the quote's `REPORT_DATA` to the verifier nonce and the exact endorsed-section hashes, then exactly the pinned number of GPU evidence items, each labelled with this handshake's nonce.

A failed attestation rejects the request before any inference data crosses the wire. Before an engine takes traffic, and periodically after, the gateway also asks it (over an attested connection) which model and weights it serves, and refuses to route to one that reports anything but the pinned model, the pinned weights hash and read-only verified weights storage.

## Response shape

The engine answers every chat request as a stream (the gateway asks for one whatever its client asked for, and builds a client's single answer from it), written to the gateway as frames of one fixed size at a fixed interval, from the moment the stream opens until it ends, and pads every other response to a size bucket, as the gateway does for its client ([gateway.md § Response shape](gateway.md#response-shape)). The host and the network between the enclaves see how long a response ran, not the size or timing of its tokens. What framing cannot hide is the engine's own activity: the host observes the confidential VM's scheduling and its devices' work, whose timing follows the computation.

## When an engine stops

An engine that fails stops serving: every request in flight ends with an error, the gateway settles each on what was generated and stops routing to the engine, and the process exits for the platform to restart it. That includes a computation step that never returns (a GPU driver call that hangs, say): a watchdog independent of the stuck step notices it once it has run far longer than the slowest legitimate step, and does the same. What such an exit cannot do is clean up after the stuck step: the engine normally zeroes the GPU memory a request used once it ends, and a forced exit leaves that memory, the requests in flight included, to the GPU driver as the process ends.
