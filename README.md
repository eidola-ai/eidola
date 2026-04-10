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

There are two supported development workflows:

**1. Full container stack** — postgres, server, shim, and stripe-cli all run
inside docker (detached). Use this when you want a one-shot reproducible
environment and don't mind rebuilding the server image:

```bash
just dev
docker compose logs -f       # follow logs
just down                    # stop everything
```

**2. Host-mode server** — postgres, the tinfoil shim mock, and stripe-cli run
in containers; the server runs on the host with cargo. The shim is configured
to forward to `host.docker.internal:8080`, so requests from the shim and
Stripe webhooks all flow into the cargo-built server. This is the recommended
inner loop while iterating on `eidola-server`:

```bash
# Bring up postgres + shim + stripe-cli; captures the Stripe webhook secret
# and writes .env.local. Stripe forwarding is enabled iff STRIPE_API_KEY is
# set in your environment or .env.
just services

# Load env vars (the .env.local override sets BIND_ADDR=0.0.0.0:8080 and
# STRIPE_WEBHOOK_SECRET so the host server is reachable from the shim).
set -a; source .env; source .env.local; set +a

# Run the server on the host machine with cargo
cargo run -p eidola-server

# -- OR --

# Run and automatically recompile/reload the server on the host machine with bacon
bacon run-long -- -p eidola-server

# When you're done:
just down
```

Both modes expose the same endpoints externally: postgres on `localhost:5432`,
the shim on `https://localhost:8443`, and (in container mode) the server on
`http://localhost:8080`. CLI configuration is identical across the two modes.

### CLI

To run the CLI against a local development stack:

1. **Start the stack:** `just dev` (starts Postgres, Server, and the Hardware Shim).
2. **Trust the Mock TLS Root:**
   On its first boot the shim generates two independent root CAs in `./.dev-certs/`:
   - `tls-ca.pem` + `tls-ca.key` — the **TLS** trust anchor (RSA PKCS#1 v1.5). This is what you trust in your OS keychain.
   - `ark.pem` + `ark.key` + `ask.pem` + `ask.key` — the **SEV-SNP attestation** chain (RSA-PSS, required by AMD's attestation format). These are *not* TLS roots and should not go in your keychain; they are passed to the CLI via `--hardware-root-ca` / `--hardware-intermediate-ca` instead.

   The split exists because Apple's Security framework does not support RSA-PSS in chain validation, so the SEV-SNP chain (which the `sev` crate requires to be PSS) cannot double as the TLS trust anchor. The shim reuses all six files on every subsequent boot, so once you trust `tls-ca.pem` your OS keychain entry survives shim restarts. The trust root flows from the filesystem only — never from the shim's API. To rotate the roots, delete `.dev-certs/` and re-run the steps below.
   - **macOS (terminal):** `sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain .dev-certs/tls-ca.pem`
   - **macOS (UI):** Open Keychain Access, drag `tls-ca.pem` into the **System** keychain, double-click it, expand **Trust**, and set "When using this certificate" to **Always Trust**.
   - **Linux:** `sudo cp .dev-certs/tls-ca.pem /usr/local/share/ca-certificates/eidola-dev.crt && sudo update-ca-certificates`
3. **Configure the CLI:**
   ```bash
   cargo run -p eidola-cli -- configure \
     --base-url https://localhost:8443 \
     --hardware-root-ca .dev-certs/ark.pem \
     --hardware-intermediate-ca .dev-certs/ask.pem \
     --trust-measurement 000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000:000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000:000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
   ```
   `--trust-measurement` takes a `<snp>:<rtmr1>:<rtmr2>` triple — three 96-char hex strings separated by colons — since each Tinfoil release ships paired AMD SEV-SNP and Intel TDX measurements. The mock shim advertises all-zeros for every field, so the dev triple is just three zero blocks. Both `--hardware-root-ca` and `--hardware-intermediate-ca` are required when pointing the CLI at the local mock shim — without ASK, the verifier falls back to AMD's production Genoa ASK, which obviously isn't signed by your local mock ARK and the chain fails to verify. (If you ever rotate `.dev-certs/`, re-run this `configure` command to refresh the embedded certs.)

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
