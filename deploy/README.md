# Deployments

Each file here describes something Eidola runs, and every enclave's file is measured: its SHA-256 is part of the launch identity its clients pin, so the bytes here are what runs, and a change to one is a new identity, released like any other.

- `gateway/tinfoil-config.yml` is the gateway's Tinfoil Container configuration. Its SHA-256 is the gateway's HOST_DATA, recorded in `releases/trust/server-enclave.json` and compiled into every client (`crates/measure-enclave/AGENTS.md`). It is never a build input of the gateway's own image: the config names that image by digest.
- `engine/<name>/<variant>/` is one inference engine deployment, pinned by the gateway's own build (`engine/README.md`).

A config says what an enclave runs and how it is set up. It names the secrets the enclave is given, but never their values: those are credentials for services outside the enclave, trusted as their operator is, and supplied at deployment (`crates/eidola-server-gateway/AGENTS.md` → Tinfoil Containers / TEE integration).

Deploying is not part of a release and is not described here. A release publishes the gateway's config; an operator deploys those exact bytes with the platform's CLI (`tinfoil container create <name> --config <file>`), and a client accepts the result only if it attests to the identity that release measured.
