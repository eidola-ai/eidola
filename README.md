# Eidolons

## Developing

**Prerequisites:** `rustup`, `just`, `docker`

The Rust toolchain version is pinned in `rust-toolchain.toml` and installed automatically by rustup. Run `just` to see all available recipes.

All Rust workspace packages now live under `crates/`, including the code generation binaries (`generate-openapi`, `shared-typegen`, and `uniffi-bindgen-swift`) and operational utilities such as `tinfoil-shim-mock` and `hash-secret`.

### Server

The server requires environment variables to work correctly. See .env.example.

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
cargo run -p eidolons-server
```rust
# -- OR --

# Run and automatically recompile/reload the server on 
# the host machine with bacon
bacon run-long -- -p eidolons-server
```

### CLI

To run the CLI against a local development stack:

1. **Start the stack:** `just dev` (starts Postgres, Server, and the Hardware Shim).
2. **Trust the Mock Root CA:**
   - The shim generates a persistent Root CA in `./.dev-certs/ark.pem`.
   - **macOS:** Open Keychain Access, drag `ark.pem` into "System", double-click it, and set Trust to "Always Trust".
   - **Linux:** `sudo cp .dev-certs/ark.pem /usr/local/share/ca-certificates/eidolons-dev.crt && sudo update-ca-certificates`
3. **Configure the CLI:**
   ```bash
   # Set the base origin (API and attestation are derived automatically)
   cargo run -p eidolons-cli -- configure --base-url https://localhost:8443
   ```
4. **Add the Hardware Root to `config.toml`:**
   Open `~/Library/Application Support/eidolons/config.toml` (or equivalent) and paste the contents of `.dev-certs/ark.pem` into the `hardware_root_ca` field:
   ```toml
   base_url = "https://localhost:8443"
   hardware_root_ca = """
   -----BEGIN CERTIFICATE-----
   ... (contents of .dev-certs/ark.pem) ...
   -----END CERTIFICATE-----
   """
   ```

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

To refresh `artifact-manifest.json` for the OCI images, run:

```bash
just update-manifest
```

This uses the pinned amd64 BuildKit builder configuration that CI uses for reproducible digest verification.
