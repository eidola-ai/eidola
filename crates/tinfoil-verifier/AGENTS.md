# tinfoil-verifier — Agent Development Guide

Exposes `attesting_client()`, which returns a `reqwest::Client` that re-verifies enclave attestation on every new TCP+TLS handshake. There is no startup bootstrap and no verified-fingerprint cache: `attesting_client()` performs no network I/O, the *first* real request is also the first attestation, and every subsequent new TLS handshake re-attests independently. Policy changes (TCB floor, allowed measurements) take effect on the next handshake without a restart. Callers wanting fail-fast-at-startup issue one throwaway request immediately after construction as the readiness check (the eidola server hits `{base}/models`).

## Per-handshake flow

The client's connector layer (`AttestingConnectorLayer`) wraps reqwest's inner connector. On every new TCP+TLS handshake, after TLS completes, the connector generates a fresh random 32-byte nonce and issues an inline HTTP/1.1 `GET /.well-known/tinfoil-attestation?nonce=<hex>` over the **same** connection (using `httparse` for response parsing and `hyper-util::TokioIo` to bridge the I/O traits). The enclave responds with a *freshly collected* hardware report whose `REPORT_DATA` is `SHA-256(tls_key_fp ‖ hpke_key ‖ nonce ‖ gpu_hash ‖ nvswitch_hash)` (the inference router is CPU-only, so GPU/NVSwitch hashes are empty), plus the PEM TLS leaf cert and an ECDSA signature over the document. The fresh document does **not** carry the VCEK, so the connector consults Tinfoil's ATC service (`POST /attestation` with `{enclaveUrl, repo}`) over a side channel to backfill it (the shim mock self-carries the VCEK, so tests need no ATC).

It then verifies, in `bundle.rs` + `attesting_client.rs`:

1. Echoed nonce equals the one sent (freshness).
2. `report_data.tls_key_fp == sha256(SPKI(peer_cert))` and the document's embedded cert matches the peer cert.
3. The document's ECDSA signature against that cert (P-384 production, P-256 mock, both SHA-256-prehash — signature reconstructed by blanking the `signature` value in the raw served bytes so unknown fields survive).
4. The AMD VCEK chain (ARK → ASK → VCEK, RSA-PSS SHA-384).
5. SEV-SNP report signature. (TDX presentations are refused outright at the platform dispatch — `Error::TdxNotAccepted` — because MRTD/RTMR0 are not yet policy-checked and RTMR1/RTMR2 alone are guest-replayable; the TDX verification plumbing stays wired for the re-enable path.)
6. TCB policy (bl≥0x07, snp≥0x0e, ucode≥0x48).
7. Measurement against the caller-supplied allowed set (the client's pinned server-enclave measurement on the client→server path; the runtime-resolved set from the server's `upstream_trust` on the server→upstream path).
8. The hardware report's `REPORT_DATA` equals the recomputed hash — which authenticates every claimed `report_data` field against the AMD/Intel signature.

Only then is the connection yielded to hyper. ALPN is pinned to `http/1.1` so the inline attestation request and the application request share one HTTP lifecycle. Pooled keepalive requests don't re-trigger the connector and inherit the binding to the TLS key attested at connection establishment. The same-connection guarantee makes this safe behind load balancers: whatever backend the LB routes you to is the backend you attest.

## Trust and fallback details

- The enclave's own fresh nonce document is the source of truth. ATC is the single fallback target (today: the VCEK); the legacy static `?v=3` / v2 documents are no longer used. Once Tinfoil folds the VCEK into the document, the ATC path can be removed entirely.
- AMD KDS CRLs are fetched in production for revocation checks, gated on `trusted_ark_der.is_none()` — test deployments supplying a custom mock ARK skip CRL fetching (AMD KDS has no entries for mock chips and the CRL signature couldn't verify against the mock ARK). `trusted_ark_der` / `trusted_ask_der` feed only the SEV-SNP chain verifier; they are **not** added to any TLS root store.
- TLS trust roots are supplied by the caller via `AttestingClientConfig::tls_roots` and used for **all** outbound HTTPS (attested endpoint, ATC, KDS CRLs). The crate intentionally depends on neither `webpki-roots` nor `rustls-native-certs`: the server (`FROM scratch`, no system store) supplies `webpki-roots`; the CLI and macOS app supply `rustls-native-certs` so developers can install local dev CAs (e.g. the shim mock's `tls-ca.pem`) in their OS keychain.
- Pure-Rust deps only (`sev`, `x509-cert`, `der`, `tower`, `hyper`, `hyper-util`, `httparse`) — no OpenSSL.

## Threat-model notes

The per-handshake nonce guarantees *freshness* — a stale or captured document can't be replayed against a different nonce, and a live, genuine CC machine currently holds the cert key. It does **not** by itself defeat exfiltration of the TLS key: the report binds the long-term TLS *key* (cert SPKI), not the live TLS *session*, so an attacker holding the stolen key could actively MITM (deriving session keys despite TLS 1.3 forward secrecy) and relay a fresh nonce-bound report from the enclave's public endpoint, passing every check. Closing that requires channel binding (a TLS-exporter value in `report_data`); today it rests on the key staying sealed in the enclave.
