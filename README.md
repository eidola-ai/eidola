# Eidola

This is an early work in progress. This repository is intended as a monorepo for [Eidola](https://www.eidola.ai), and will include many different kinds of files, from functional code to configuration to documentation.

Over the coming months, its shape will change substantially as we port several external proofs-of-concept into a coherent, maintainable single source of truth.

We plan to release the functional components of this repo under appropriate open source licenses, but have not finalized the details.

## Contributing

Pull requests are checked for CLA coverage. Every Git author and committer
email in a PR must match an entry in `CLA-SIGNERS.txt` for the current hash of
`CLA-INDIVIDUAL.md` or `CLA-CORPORATE.md`.

To sign, add the appropriate entry to `CLA-SIGNERS.txt` in a commit to this
repository. The signer entry plus the relevant Git history are the signature
record; there is no separate PDF or email flow.

## Developing

**Prerequisites:** `rustup`, `just`, `docker`

The Rust toolchain version is pinned in `rust-toolchain.toml` and installed automatically by rustup. Run `just` to see all available recipes.

All Rust workspace packages now live under `crates/`, including the code generation binaries (`generate-openapi`, `shared-typegen`, and `uniffi-bindgen-swift`) and operational utilities such as `tinfoil-shim-mock`, `hash-secret`, and `measure-enclave`.

### Server

The server requires environment variables to work correctly. See .env.example.
For local development, `DATABASE_URL=postgres://eidola@localhost/eidola` uses the plain Postgres
container from `compose.yaml`. For production, you can point `DATABASE_URL` at an external Postgres,
set `DATABASE_PASSWORD` as a secret, and optionally provide `DATABASE_SSL_CERT` with the PEM-encoded
root CA certificate if the database does not chain to the default WebPKI roots.

For a complete postgres, server, and stripe webhook forwarding run:

```bash
just dev
```

To iterate more quickly while building locally:

```bash
# Start backing services (postgres + simulator)
just services

# Set environment variables
set -a; source .env; set +a

# Run the server on the host machine with cargo
cargo run -p eidola-server

# -- OR --

# Run and automatically recompile/reload the server on 
# the host machine with bacon
bacon run-long -- -p eidola-server
```

### CLI

To run the CLI against a local development stack:

1. **Start the stack:** `just dev` (starts Postgres, Server, and the Hardware Shim).
2. **Trust the Mock Root CA:**
   The shim generates a persistent Root CA in `./.dev-certs/ark.pem`.
   - **macOS (terminal):** `security add-trusted-cert -r trustAsRoot -p ssl -k ~/Library/Keychains/login.keychain-db .dev-certs/ark.pem`
   - **macOS (UI):** Open Keychain Access, drag `ark.pem` into your login keychain, double-click it, and set Trust to "Always Trust".
   - **Linux:** `sudo cp .dev-certs/ark.pem /usr/local/share/ca-certificates/eidola-dev.crt && sudo update-ca-certificates`
3. **Configure the CLI:**
   ```bash
   cargo run -p eidola-cli -- configure \
     --base-url https://localhost:8443 \
     --hardware-root-ca .dev-certs/ark.pem \
     --trust-measurement 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
   ```
   The all-zeros measurement matches the mock shim's default. The shim includes its full certificate chain (ARK + ASK) in the attestation response, so only the root CA is needed in the client config.

   On macOS, the CLI's configuration is stored in `~/Library/Application Support/eidola/config.toml`.

Consider installing [bacon](https://github.com/Canop/bacon) (`cargo install bacon`) for convenience.

See more available commands:

```bash
just --help
```

**Updating generated files:**
If you change Rust APIs or types, update the committed Swift bindings or OpenAPI spec:
```bash
just update-bindings      # UniFFI Swift bindings + Crux types
just update-openapi       # OpenAPI spec
just update-xcframework   # XCFramework (dev, native arch only)
```

To refresh `artifact-manifest.json` for the OCI images, macOS app/CLI, and enclave measurements, run:

```bash
just update-manifest
```

This uses the pinned amd64 BuildKit builder configuration for the OCI images, the local Nix macOS builds for the app and CLI, and the `measure-enclave` binary to compute SEV-SNP and TDX measurements from `tinfoil-config.yml`. It currently needs to run on macOS.

To compute enclave measurements independently (without rebuilding images):

```bash
just measure
```
