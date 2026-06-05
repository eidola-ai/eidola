# tinfoil-verifier

Verifies [Tinfoil](https://tinfoil.sh) enclave attestation and pins TLS connections to the attested certificate. This ensures the eidola server only sends inference traffic to a genuine TEE enclave running expected code.

Supports both **AMD SEV-SNP** and **Intel TDX** enclaves.

## What it does

`attesting_client()` returns a client that re-verifies attestation on each new TLS connection. On every handshake it generates a fresh random nonce, fetches `GET /.well-known/tinfoil-attestation?nonce=<hex>` inline over the same connection, and verifies:

SEV-SNP verification is delegated to the [`sev`](https://crates.io/crates/sev) crate (virtee/sev) with the `crypto_nossl` feature. TDX verification is delegated to [`dcap-qvl`](https://crates.io/crates/dcap-qvl) with the `rustcrypto` backend. Both are pure Rust — no OpenSSL.

### AMD SEV-SNP

1. **Fetch** the fresh nonce-bound document inline over the attested connection (backfilling the VCEK from Tinfoil's ATC service, which the document omits)
2. **Check freshness** — the echoed nonce equals the one sent
3. **Check the document signature** — ECDSA (P-384 prod / P-256 mock, SHA-256) by the embedded TLS leaf cert, which must match the peer cert
4. **Verify the VCEK certificate chain** (AMD Genoa ARK → ASK → VCEK) via RSA-PSS(SHA-384) and the report signature (ECDSA-P384) against the VCEK
5. **Validate TCB policy** — minimum firmware versions (`bl >= 0x07`, `snp >= 0x0e`, `ucode >= 0x48`)
6. **Check the code measurement** (48-byte launch digest) against the allowlist
7. **Cross-check `REPORT_DATA`** — equals `SHA-256(tls_key_fp ‖ hpke_key ‖ nonce ‖ …)`, where `tls_key_fp == sha256(SPKI(peer_cert))`. This binds the nonce, TLS key, and HPKE key to the AMD-signed report.

### Intel TDX

1. **Fetch** the fresh nonce-bound document inline (steps 2–3 as above)
2. **Fetch collateral** (TCB info, QE identity, CRLs) from Intel's Provisioning Certification Service
3. **Verify the TDX Quote V4** signature against Intel's SGX Provisioning Root CA
4. **Validate TCB policy** and TDX module identity via Intel-signed collateral
5. **Check the code measurement** (RTMR1 || RTMR2, 96 bytes concatenated) against the allowlist
6. **Cross-check `REPORT_DATA`** as in the SEV-SNP path

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
