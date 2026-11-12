# Engine deployments

One directory per Eidola-hosted engine deployment: `deploy/engine/<name>/<variant>/`, holding the deployment's `tinfoil-config.yml` and its `deployment.json` (the weights' architecture and provenance, the GPU count, and the machines it runs on). `<name>` is for people (the model's family, say); a deployment is identified by the weights it serves: its node's `EIDOLA_ENGINE_MODEL_ID` must be its `EIDOLA_ENGINE_WEIGHTS_SHA256`, it is pinned under that hash in `releases/trust/engine-enclaves.json`, and the hash must have a served row in the model catalog (`models/`, see its README).

`deployment.json` names the weights' architecture, an implemented `eidola_common::architecture::Architecture` (`"architecture": "MimoV26Flash"`): the resource checks (KV bytes per position, the drafter, the executor's scratch) read that architecture's shape, and the draft width is held to its MTP layers. A stub architecture (`MimoV26Pro` today) is refused.

A deployment's one container is the node's release image, `ghcr.io/eidola-ai/eidola-server-engine@sha256:<digest>` (built by `oci/eidola-server-engine/Containerfile`; `just engine-image` prints the digest a source tree builds to), with `EIDOLA_ENGINE_KERNELS_DIR: "/opt/eidola/kernels"`, where that image ships the GPU kernels.

A deployment runs the `cuda` executor on the platform release's GPU image (`cvm-version: X+gpu@sha256:<that image's release manifest>`; the CPU image refuses GPUs) with `EIDOLA_ENGINE_GPU_CC: "required"`, so the node checks its GPUs' confidential-computing mode before opening one; `not-required` is for unpinned development boots on a GPU without confidential computing, and the gateway's build refuses to pin it.

One deployment is committed: `mimo-v2.6-flash/b300x1/`, MiMo-V2.6-Flash on one B300. It is pinned and has not been launched. Its weights pack is the modelwrap schema 1 pack of the catalog row's `repo@revision`, and its `GATEWAY_TOKEN_HASH` is the hash of a placeholder token that was discarded, so the release that deploys it replaces that hash with the hash of the gateway's real `ENGINE_TOKEN` and re-measures.

Every deployment committed here must be pinned in `releases/trust/engine-enclaves.json`, and every pin there must match the files here: the gateway's build checks both directions and refuses to build otherwise. Layout, the pin's derivation, and the release-time measurement are in `crates/measure-enclave/AGENTS.md`.

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

It fetches the release's `platform-endorsements.json` and GitHub's attestations for its digest into an empty temporary directory, verifies them exactly as the build does (the provider's `.github/workflows/platform-release.yml` at exactly that tag, on a GitHub-hosted runner, logged in Rekor, about a file with that hash), and writes only what verified, then prints the file's hash and its TDX policies. Review the diff against the release it replaces (floors raised or lowered, machines added or retired), point each deployment's `deployment.json` at the new release and hash, re-measure (`measure-engine-enclaves`), and remove a vendored release no deployment names any more. The scheduled `platform-endorsements-drift` workflow reports when the provider has published a release newer than the newest vendored one, or than one a deployment names.
