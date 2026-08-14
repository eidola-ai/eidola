# tinfoil-verifier — Agent Development Guide

Exposes `attesting_client()`, a `reqwest::Client` that re-verifies enclave attestation on every new TCP+TLS handshake. Construction performs no network I/O and there is no verified-fingerprint cache: the first real request is the first attestation, every subsequent new connection re-attests independently, and policy changes take effect on the next handshake. Callers wanting startup readiness issue a trivial request immediately after construction.

## Per-handshake flow

`AttestingConnectorLayer` wraps reqwest's connector. After each TLS handshake it generates a random 32-byte nonce and sends `GET /.well-known/tinfoil-attestation?nonce=<hex>` over the **same** HTTP/1.1 connection that will carry application traffic. Tinfoil's self-contained v3 envelope carries a fresh CPU quote, two base64-encoded endorsed sections, and vendor-signed collateral.

`bundle.rs` mirrors Tinfoil's strict envelope contract:

1. Reject unknown or duplicate JSON members recursively; require lowercase fixed-width hex, canonical standard base64, and unique item/collateral IDs.
2. Hash the exact base64-decoded `crypto_material` and `device_evidence` bytes — never a re-serialization — and compare both hashes with `cpu_evidence.endorsed`.
3. Recompute `REPORT_DATA[0:32] = SHA-256("https://tinfoil.sh/report-data/v1" ‖ nonce ‖ crypto_material_hash ‖ device_evidence_hash)` with the remaining 32 bytes zero; compare it with `challenge.report_data`.
4. Require conventional `tls` (SPKI SHA-256) and `hpke` (X25519) crypto items, then compare the TLS fingerprint with the peer certificate from the completed handshake.
5. For SEV-SNP, require the report to be exactly 1184 bytes and pass field hygiene (`sevsnp::check_report_hygiene`): version ≥ 3, guest policy with DEBUG and MIGRATE_MA off, VCEK-signed (`SIGNING_KEY=0`) with `MASK_CHIP_KEY` clear, and no ID-block launch. The guest policy is **not** part of the launch measurement — a relaunch of the pinned image with DEBUG=1 keeps the same measurement while giving the hypervisor `SNP_DBG_DECRYPT` access — so it must be policed here.
6. Require CPU-subject `amd-vcek/v1` and `amd-crl/v1` endorsement collateral. Decode exactly ASK then ARK from the carried KDS chain and require byte identity with the pinned AMD chain (or explicit mock chain), verify ARK → ASK → VCEK and the report signature, then verify the carried CRL entirely offline. It must be a complete direct v2 CRL (not delta, indirect, or scope-restricted), carry a CRL number and an AKI matching the pinned ARK's SKI, use exact supported signature parameters, contain no unsupported critical extensions, satisfy `thisUpdate <= now < nextUpdate`, and omit both ASK/VCEK serials. Authenticity comes from AMD's ARK signature, not the relay. A malicious relay can replay an older AMD-signed CRL only for the remainder of its signed validity interval; the verifier deliberately makes no AMD KDS request.
7. Enforce the SEV-SNP rollback/floor policy, match the launch measurement, and finally require the hardware report's signed `REPORT_DATA` to equal the recomputed envelope value.

Parsing and challenge checking alone authenticate nothing. Only the signed hardware report makes the endorsed sections trustworthy. There is no document-level ECDSA signature in v3 and no ATC, AMD KDS, or Intel PCS network fallback. Collateral is untrusted transport authenticated by vendor signatures. TDX envelopes are refused with `Error::TdxNotAccepted` until MRTD/RTMR0 policy exists; any future implementation must consume the document-carried Intel PCS collateral offline.

ALPN is pinned to HTTP/1.1 so the inline attestation and application request share one connection lifecycle. Pooled keepalive requests inherit the binding established for that connection. TLS roots are caller-supplied and authenticate only the inference endpoint; hardware roots are separate and never enter WebPKI. Keep the implementation pure Rust.

## Threat-model note

The nonce defeats stale-document replay and the quote authenticates the peer certificate's long-lived SPKI. It does **not** bind the live TLS session. An attacker holding an exfiltrated TLS private key can terminate TLS outside the enclave while relaying a fresh nonce-bound document from a genuine enclave. Closing that gap requires channel binding, such as committing TLS exporter material to `REPORT_DATA`; Tinfoil's current v3 implementation does not do this.
