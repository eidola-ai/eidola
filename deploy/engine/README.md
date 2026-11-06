# Engine deployments

One directory per Eidola-hosted engine deployment: `deploy/engine/<model-id>/<variant>/`, holding the deployment's `tinfoil-config.yml` and its `deployment.json` (weights provenance, GPU count, Intel TDX machine policy).

A deployment runs the `cuda` executor on the platform release's GPU image (`cvm-version: X+gpu@sha256:<that image's release manifest>`; the CPU image refuses GPUs) with `EIDOLA_ENGINE_GPU_CC: "required"`, so the node checks its GPUs' confidential-computing mode before opening one; `not-required` is for unpinned development boots on a GPU without confidential computing, and the gateway's build refuses to pin it.

Every deployment committed here must be pinned in `releases/trust/engine-enclaves.json`, and every pin there must match the files here: the gateway's build checks both directions and refuses to build otherwise. Layout, the pin's derivation, and the release-time measurement are in `crates/measure-enclave/AGENTS.md`.
