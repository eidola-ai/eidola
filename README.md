# Eidolons

## Developing

**Prerequisites:** `rustup`, `just`, `docker`

The Rust toolchain version is pinned in `rust-toolchain.toml` and installed automatically by rustup. Run `just` to see all available recipes.

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

# -- OR --

# Run and automatically recompile/reload the server on 
# the host machine with bacon
bacon run-long -- -p eidolons-server
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
