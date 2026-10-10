# Engine deployments

One directory per Eidola-hosted engine deployment: `deploy/engine/<model-id>/<variant>/`, holding the deployment's `tinfoil-config.yml` and its `deployment.json` (weights provenance, GPU count, Intel TDX machine policy).

Every deployment committed here must be pinned in `releases/trust/engine-enclaves.json`, and every pin there must match the files here: the gateway's build checks both directions and refuses to build otherwise. Layout, the pin's derivation, and the release-time measurement are in `crates/measure-enclave/AGENTS.md`.
