# Engine deployments

One directory per Eidola-hosted engine deployment: `deploy/engine/<name>/<variant>/`, holding the deployment's `tinfoil-config.yml` and its `deployment.json` (the weights' architecture and provenance, the GPU count, and the machines it runs on). `<name>` is for people (the model's family, say); a deployment is identified by the weights it serves: its node's `ENGINE_MODEL_ID` must be its `ENGINE_WEIGHTS_SHA256`, it is pinned under that hash in `releases/trust/engine-enclaves.json`, and the hash must have a served row in the model catalog (`models/`, see its README).

`deployment.json` names the GPU its devices run on (`"gpu": "NVIDIA B200"`, an `engine_deployment::SupportedGpu` by the name its driver reports, whose memory the device fit is held to) and the weights' architecture, an implemented `eidola_common::architecture::Architecture` (`"architecture": "MimoV26Flash"`): the resource checks (KV bytes per position, the drafter, the executor's scratch) read that architecture's shape, and the draft width is held to its MTP layers. A stub architecture (`MimoV26Pro` today) is refused.

A deployment's one container is the node's release image, `ghcr.io/eidola-ai/eidola-server-engine@sha256:<digest>` (built by `oci/eidola-server-engine/Containerfile`; `just engine-image` prints the digest a source tree builds to), with `ENGINE_KERNELS_DIR: "/opt/eidola/kernels"`, where that image ships the GPU kernels.

The node reads every variable under the prefix `EIDOLA_ENV_PREFIX` names (read bare; it must match `^[A-Z][A-Z0-9_]*_$`), with no fallback to the bare name, as the gateway and the billing binary read their own. Secrets on the container platform are organisation-scoped and arrive under their own names, so a deployment's env must declare a prefix, and every other name in it carries that prefix: the committed configs set `EIDOLA_ENV_PREFIX: "ENGINE_"`, their env is `ENGINE_MODEL_ID`, `ENGINE_WEIGHTS_DIR`, …, and their one secret is `ENGINE_GATEWAY_TOKEN`, the token the gateway's `GATEWAY_ENGINE_TOKEN` carries. The token is not measured, so supplying or rotating it changes no pin. `just run engine-server` sets no prefix, so it uses the names as written.

A deployment runs the `cuda` executor on the platform release's GPU image (`cvm-version: X+gpu@sha256:<that image's release manifest>`; the CPU image refuses GPUs) with `ENGINE_GPU_CC: "required"`, so the node checks its GPUs' confidential-computing mode before opening one; `not-required` is for unpinned development boots on a GPU without confidential computing, and the gateway's build refuses to pin it.

Two deployments are committed, both pinned and not launched: `mimo-v2.6-flash/b300x1/`, MiMo-V2.6-Flash on one B300, and `qwen3.8-27b/b200x1/`, Qwen3.8-27B on one B200 (its machines the platform endorsements' `tdx-b200-1-5-20-prod`). Each one's weights pack is the modelwrap schema 1 pack of its catalog row's `repo@revision` (reproducible: `modelwrap --schema 1 <repo>@<revision>` rebuilds the reference from the repository).

Every deployment committed here must be pinned in `releases/trust/engine-enclaves.json`, and every pin there must match the files here: the gateway's build checks both directions and refuses to build otherwise. Layout, the pin's derivation, and the release-time measurement are in `crates/measure-enclave/AGENTS.md`.

## The image: platform release manifests

A pin's MRTD is the measurement of the platform image its config's `cvm-version` names. The release manifest that `cvm-version` pins by hash is vendored verbatim at `releases/trust/cvm-releases/v<tag>/tinfoil-inference-v<version>-manifest.json`, where `<version>` is the config's (`0.15.4-rc.2+gpu`) and `<tag>` the release it belongs to (`0.15.4-rc.2`). Every gateway build requires the manifest to be there, to hash to the config's pin, to describe that release, to state a launch with no measured boot (format 1, zero RTMRs), and to state exactly the pin's MRTD; anything else fails the build, naming the deployment. `measure-engine-enclaves` reads the same file by the same rule and checks that the image it names measures to it, and the `main` branch's artifacts workflow reruns that measurement from the published image on every run.

Choosing a release for a deployment (a new `cvm-version`) therefore vendors its manifest. It is a release asset of `tinfoilsh/cvmimage`; download it into an empty directory and compare its hash with the pin before copying it in (the build compares it again):

```bash
V=0.15.4-rc.2 F="tinfoil-inference-v$V+gpu-manifest.json"
curl -sfL -A curl/8 -o "$F" "https://github.com/tinfoilsh/cvmimage/releases/download/v$V/tinfoil-inference-v$V%2Bgpu-manifest.json"
shasum -a 256 "$F"   # must equal the @sha256: in cvm-version
mkdir -p releases/trust/cvm-releases/v$V && cp "$F" releases/trust/cvm-releases/v$V/
```

Then fetch the image the manifest names (`tinfoil-tdx-v<version>.igvm` from `https://images.tinfoil.sh/cvm/`, `+` written `%2B`) and re-measure. A vendored manifest no deployment names fails the gateway's build, so remove it in the change that moves the last deployment off it. Rebuilding the release from source is a separate check, in `docs/contributing/rebuilding-platform-images.md`.

## The machines: platform endorsements

A deployment's Intel TDX machine policy (TDX module, TD attributes, XFAM, TCB floors, QE vendor) and the hosts it may run on (by PPID) are never written by hand. The platform provider publishes them as signed platform endorsements, `platform-vX.Y.Z` releases of `tinfoilsh/cvmimage`, and `deployment.json` names one policy of one release, with the hash of that release's file, plus the three PCK platform flags the endorsements do not carry:

```json
"tdx": {
  "platform_endorsements": {
    "release": "platform-v0.1.11",
    "sha256": "ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9548",
    "policy": "tdx-b300-prod"
  },
  "dynamic_platform": "true",
  "cached_keys": "true",
  "smt_enabled": "false"
}
```

The release is vendored under `releases/trust/platform-endorsements/<tag>/` (`platform-endorsements.json` and its Sigstore bundle, `attestation.sigstore.json`). `measure-enclave` and every gateway build verify it offline against the committed `releases/trust/sigstore-trusted-root.json` and derive the pin's policy and PPID allowlist from it; nothing in any build reads the network.

**Adopting a release** is the one step that downloads anything:

```bash
just engine-platform-endorsements platform-vX.Y.Z
```

It fetches the release's `platform-endorsements.json` and GitHub's attestations for its digest into an empty temporary directory, verifies them exactly as the build does (the provider's `.github/workflows/platform-release.yml` at exactly that tag, on a GitHub-hosted runner, logged in Rekor, about a file with that hash), and writes only what verified, then prints the file's hash and its TDX policies. Review the diff against the release it replaces (floors raised or lowered, machines added or retired), point each deployment's `deployment.json` at the new release and hash, re-measure (`measure-engine-enclaves`), and remove a vendored release no deployment names any more, except the newest, which stays as the baseline the drift check compares against. The gateway's build holds all of this: every TDX deployment names the newest vendored release, with no exception, and no vendored release but the newest is one nothing names. So the order of operations matters when a new release would cut a running deployment off (a machine it retires, a TCB floor its host does not meet):

1. Move that deployment to machines the new release accepts, or retire it: its directory, its pin, and its placement go first.
2. Then vendor the release and move every remaining deployment to it, in one change.

The scheduled `platform-endorsements-drift` workflow reports when the provider has published a release newer than the newest vendored one, or than one a deployment names.
