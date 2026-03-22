# tinfoil-verifier

Verifies [Tinfoil](https://tinfoil.sh) enclave attestation and pins TLS connections to the attested certificate. This ensures the eidolons server only sends inference traffic to a genuine AMD SEV-SNP enclave running expected code.

## What it does

`verify_and_pin()` performs a single end-to-end verification on startup:

1. **Fetch** the attestation bundle from Tinfoil's ATC service
2. **Verify the VCEK certificate chain** (embedded AMD Genoa ARK → ASK → VCEK) via RSA-PSS(SHA-384)
3. **Verify the attestation report signature** (ECDSA-P384) against the VCEK public key
4. **Validate TCB policy** — minimum firmware versions (`bl >= 0x07`, `snp >= 0x0e`, `ucode >= 0x48`)
5. **Check the code measurement** against a hardcoded allowlist
6. **Cross-check the enclave TLS certificate** against `report_data[0..32]` (SHA-256 of SPKI)
7. **Return a pinned `reqwest::Client`** that rejects any server whose public key fingerprint doesn't match

Steps 2–3 are delegated to the [`sev`](https://crates.io/crates/sev) crate (virtee/sev) with the `crypto_nossl` feature for pure-Rust cryptography.

## Usage

```rust
// A rustls CryptoProvider must be installed first.
let (client, verification) = tinfoil_verifier::verify_and_pin(
    &["<hex-encoded measurement>", "<previous deployment>"],
    None, // default ATC endpoint
).await?;

// `client` will only connect to the verified enclave.
let response = client.post("https://inference.tinfoil.sh/v1/chat/completions")
    .json(&request)
    .send()
    .await?;
```

## Testing

```sh
cargo test -p tinfoil-verifier                   # unit tests (cert chain verification)
cargo test -p tinfoil-verifier -- --ignored  # live test against atc.tinfoil.sh
```
