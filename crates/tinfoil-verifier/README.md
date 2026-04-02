# tinfoil-verifier

Verifies [Tinfoil](https://tinfoil.sh) enclave attestation and pins TLS connections to the attested certificate. This ensures the eidola server only sends inference traffic to a genuine TEE enclave running expected code.

Supports both **AMD SEV-SNP** and **Intel TDX** enclaves.

## What it does

`attesting_client()` performs a single end-to-end verification on startup and returns a client that re-verifies attestation on each new TLS connection:

### AMD SEV-SNP
1. **Fetch** the attestation bundle from Tinfoil's ATC service or the server's well-known endpoint
2. **Verify the VCEK certificate chain** (embedded AMD Genoa ARK → ASK → VCEK) via RSA-PSS(SHA-384)
3. **Verify the attestation report signature** (ECDSA-P384) against the VCEK public key
4. **Validate TCB policy** — minimum firmware versions (`bl >= 0x07`, `snp >= 0x0e`, `ucode >= 0x48`)
5. **Check the code measurement** (48-byte launch digest) against the allowlist
6. **Cross-check the enclave TLS certificate** against `report_data[0..32]` (SHA-256 of SPKI)

### Intel TDX
1. **Fetch** the attestation document from Tinfoil's ATC service or the server's well-known endpoint
2. **Fetch collateral** (TCB info, QE identity, CRLs) from Intel's Provisioning Certification Service
3. **Verify the TDX Quote V4** signature against Intel's SGX Provisioning Root CA
4. **Validate TCB policy** and TDX module identity via Intel-signed collateral
5. **Check the code measurement** (RTMR1 || RTMR2, 96 bytes concatenated) against the allowlist
6. **Cross-check the enclave TLS certificate** against `report_data[0..32]` (SHA-256 of SPKI)

SEV-SNP verification is delegated to the [`sev`](https://crates.io/crates/sev) crate (virtee/sev) with the `crypto_nossl` feature. TDX verification is delegated to [`dcap-qvl`](https://crates.io/crates/dcap-qvl) with the `rustcrypto` backend. Both are pure Rust — no OpenSSL.

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
