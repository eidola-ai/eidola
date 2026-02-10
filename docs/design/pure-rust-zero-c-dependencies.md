# Pure Rust, Zero C Dependencies

## Context

Cross-compilation is a first-class requirement. The project targets four
platform/architecture combinations today (macOS arm64/x86_64, Linux musl
arm64/x86_64) with iOS, Android, and WebAssembly planned. The build runs on
macOS and must produce static Linux binaries without a C cross-compiler
toolchain.

Most Rust crates that provide cryptography, TLS, or ML inference depend on C
or assembly code. The two dominant TLS backends in the Rust ecosystem — `ring`
and `aws-lc-rs` — both require a C compiler and platform-specific assembly.
Similarly, popular ML inference engines (llama.cpp, ONNX Runtime) are C++
libraries with Rust bindings.

These C dependencies create compounding problems:

1. **Cross-compilation requires a C cross-toolchain.** Building for
   `aarch64-unknown-linux-musl` from macOS needs a musl-targeting C compiler,
   sysroot, and linker. This is fragile and hard to reproduce.
2. **Nix hermetic builds break.** C dependencies often expect system headers
   or pkg-config, which conflicts with Nix's sandboxed environment.
3. **Static linking becomes difficult.** Mixing static Rust with dynamic C
   libraries creates deployment complications, especially for musl targets.
4. **Audit surface increases.** C code requires separate tooling for memory
   safety analysis and is harder to audit than Rust.

## Decision

Eliminate all C and assembly dependencies from the project. Every dependency
must be pure Rust, compilable by `rustc` alone.

### TLS: `rustls` with `rustls-rustcrypto`

Use `rustls` for TLS with the `rustls-rustcrypto` crypto provider instead of
the default `ring` or `aws-lc-rs` backends. `rustls-rustcrypto` implements
cryptographic primitives in pure Rust via the RustCrypto project crates.

```toml
# Cargo.toml pattern used in both server and perception crates
reqwest = { version = "0.12", default-features = false, features = [
    "rustls-tls-webpki-roots-no-provider",
] }
rustls = { version = "0.23", default-features = false, features = ["ring"] }
rustls-rustcrypto = "0.0.2-alpha"
```

The `ring` feature on `rustls` here refers to the API shape, not the `ring`
crate — `rustls-rustcrypto` provides the actual implementation.

Dependencies that would transitively pull in C-based TLS are configured to
avoid it. For example, `hf-hub` defaults to `ureq` which hardcodes `ring`:

```toml
# Disable ureq (hardcodes ring), use tokio backend instead
hf-hub = { version = "0.4", default-features = false, features = ["tokio"] }
```

### CA Certificates: `webpki-roots`

Embed Mozilla's CA certificate bundle at compile time via the `webpki-roots`
crate rather than depending on system certificate stores. This makes binaries
fully self-contained — no runtime dependency on `/etc/ssl/certs` or the macOS
Keychain.

### ML Inference: Burn

Use the Burn deep learning framework for on-device inference rather than C++
engines. Burn is pure Rust with pluggable backends (WGPU for GPU, NdArray for
CPU). Model architectures are implemented directly in Rust. See
[On-Device Inference with Burn](on-device-inference-with-burn.md) for the full ML framework
decision.

### Linking: `rust-lld`

For Linux cross-compilation, use `rust-lld` (the linker bundled with the Rust
toolchain) rather than requiring a system cross-linker. This is configured
automatically in `flake.nix` for musl targets.

## Consequences

**Benefits:**

- Any target in `rust-toolchain.toml` can be built from any host with only
  `rustc` and `cargo`. No C compiler, sysroot, or platform SDK needed (except
  for the macOS app shell, which requires Xcode by nature).
- Nix builds are fully hermetic — no `pkgsCross` or `buildInputs` for C
  libraries.
- Static musl binaries are trivially produced, enabling distroless OCI images.
- The entire dependency tree is auditable as Rust code.

**Trade-offs we accept:**

- `rustls-rustcrypto` is alpha-quality (v0.0.2-alpha). The RustCrypto
  implementations have not received the same level of scrutiny as `ring`'s
  formally verified assembly. We accept this because the project is not yet
  handling production secrets, and the ecosystem is maturing rapidly.
- Performance of pure-Rust crypto may be lower than optimized assembly in
  `ring`. For a proxy server, TLS handshake overhead is negligible compared
  to upstream API latency. For ML inference, crypto is not in the hot path.
- Some popular crates are unusable because they transitively depend on C code.
  This narrows the dependency pool and occasionally requires workarounds
  (e.g., disabling default features, choosing less common alternatives).
- Embedded CA certificates must be updated by rebuilding. The binary won't
  automatically pick up system CA changes (this is also a feature — no
  surprise trust store mutations).

## Future Considerations

- **`rustls-rustcrypto` stabilization.** When it reaches 1.0, the alpha-quality
  concern goes away. Monitor the RustCrypto project's progress.
- **FIPS compliance.** If ever required, `rustls` supports pluggable providers.
  A FIPS-certified provider could be swapped in without changing application
  code, though it would likely reintroduce C dependencies for that specific
  build target.
- **WebAssembly.** The pure-Rust stack should compile to WASM without
  modification, which is not possible with C dependencies.
