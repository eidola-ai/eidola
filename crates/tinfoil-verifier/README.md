# tinfoil-verifier

Builds a `reqwest::Client` that verifies Tinfoil's self-contained v3 enclave attestation before using each new TLS connection.

For every handshake the connector sends a fresh nonce to `/.well-known/tinfoil-attestation` over the same connection, strictly parses the returned envelope, verifies the endorsed-section hashes and TLS SPKI binding, enforces report field hygiene (version ≥ 3, DEBUG/MIGRATE_MA guest-policy bits off, VCEK-signed, no ID-block — fields the launch measurement does not cover), requires the document-carried AMD ASK/ARK chain to match the pinned chain, verifies the SEV-SNP VCEK and report, verifies the complete carried AMD CRL's ARK identity/signature, signed half-open validity interval, and ASK/VCEK revocation state, enforces TCB and measurement policy, and compares the report's signed `REPORT_DATA` with the envelope's domain-separated recomputation. TDX presentations currently fail closed because the required MRTD/RTMR0 measurement policy is not implemented.

The v3 flow makes no ATC, AMD KDS, or Intel PCS request. Vendor collateral travels in the envelope as untrusted transport and is accepted only after its signature, profile, and validity checks pass. A malicious relay cannot change AMD's signed CRL or extend its validity interval, but it can replay an older AMD-signed CRL until that interval ends.

```rust
let client = tinfoil_verifier::attesting_client(
    tinfoil_verifier::AttestingClientConfig {
        allowed_measurements: &allowed_measurements,
        inference_base_url: "https://inference.tinfoil.sh/v1",
        tls_roots,
        trusted_ark_der: None,
        trusted_ask_der: None,
        snp_min_tcb: None,
        snp_observer: None,
        attestation_observer: None,
    },
)
.await?;
```

Run unit and offline contract tests with `cargo test -p tinfoil-verifier`. Ignored tests exercise the production endpoint; `mock_attesting_client_e2e` runs the full connector against `tinfoil-shim-mock` when `MOCK_URL` and `MOCK_CERT_DIR` are set.
