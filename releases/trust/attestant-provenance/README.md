# Attestant key provenance (informational)

This directory holds **optional, informational** evidence that each pinned attestant signing key is a real, policy-constrained *hardware* key — for example, a YubiKey-PIV attestation certificate stating that the key was generated on-device, is non-exportable, and requires a PIN and a physical touch for every signature.

**Nothing here is part of trust evaluation.** No client, no updater, and no `build.rs` reads these files. The authoritative trust input remains the `trusted_attestant_fingerprints` list in [`../trust-constants.json`](../trust-constants.json). This evidence exists purely so an external auditor can independently confirm, offline, that a pinned fingerprint corresponds to genuine hardware — turning "the maintainer says they used a hardware key" into something cryptographically checkable.

Because it is informational, the format is deliberately loose and **manufacturer-neutral**: a YubiKey, a TPM, a different SmartCard, or a cloud KMS each commits whatever attestation *its* platform produces, in the same bundle shape. Eidola does not commit to a single key vendor.

## Layout

```text
attestant-provenance/
  README.md                    # this file
  <attestant-id>/
    key-attestation.pem        # attestation cert whose subjectPublicKey IS the signing key
    intermediate.pem           # the cert that signed it (device/vendor intermediate)
    meta.json                  # the human-readable binding + how to verify
  yubico-piv-root.pem          # vendor root(s), reference-only (see note below)
```

Each `<attestant-id>/` holds the evidence for that human's **current** signing key — one fingerprint per human at a time. This directory mirrors the *current* `trusted_attestant_fingerprints`: on rotation the bundle is updated in place (the same human, a new key), and the retired key's evidence remains in git history — the same current-state-in-tree model the rest of `releases/trust/` uses. (A human can hold more than one fingerprint *over time*, never more than one at once; that one-at-a-time invariant is what keeps `MIN_HUMAN_ATTESTATIONS` counting distinct *humans*.)

`meta.json` shape (only `pinned_fingerprint_sha256` is machine-checked; the rest is context):

```json
{
  "attestant_id": "your-name",
  "pinned_fingerprint_sha256": "<sha256 of the key's SubjectPublicKeyInfo, hex>",
  "algorithm": "ECDSA-P256",
  "manufacturer": "Yubico",
  "product": "YubiKey 5C",
  "serial": "42342556",
  "firmware": "5.7.x",
  "key_generation": "on-device, non-exportable",
  "pin_policy": "ALWAYS",
  "touch_policy": "ALWAYS",
  "evidence": ["key-attestation.pem", "intermediate.pem"],
  "chains_to": "Yubico PIV Root CA Serial 263751 (yubico-piv-root.pem)"
}
```

The `pinned_fingerprint_sha256` is `sha256(SubjectPublicKeyInfo DER)` of the signing key — the **same** value pinned in `trusted_attestant_fingerprints` and the **same** value `release-tool pkcs11 list` prints. It is computed over the public key embedded in `key-attestation.pem`, so the cert and the pin are directly comparable.

## Tooling

- `cargo run -p release-tool -- provenance capture --attestant-id <id>` (or `just release-provenance-capture <id>`) — YubiKey convenience: runs `ykman` to write `key-attestation.pem` + `intermediate.pem`, then fills `meta.json` from the attestation cert itself — `serial`, `firmware`, and `pin_policy` / `touch_policy` all live in the Yubico attestation extensions (`1.3.6.1.4.1.41482.3.*`), so nothing is left as a `TODO` (confirm the derived `product` string reads sensibly). Non-YubiKey attestants populate the bundle by hand.
- `cargo run -p release-tool -- provenance enrich [--attestant-id <id>]` (or `just release-provenance-enrich`) — (re)derive those `meta.json` fields from a bundle's committed `key-attestation.pem`, **offline, with no device or `ykman`**. Useful to refresh a hand-built bundle or one whose cert was added manually. It merges over any existing `meta.json`, so operator-added context is preserved; only fields the cert authoritatively provides change.
- `cargo run -p release-tool -- provenance check` (or `just release-provenance-check`) — vendor-neutral, CI-friendly: for every bundle, recomputes the certificate's public-key fingerprint and asserts it equals the bundle's claimed `pinned_fingerprint_sha256`, and that the fingerprint is still in the current trusted set. A bundle whose fingerprint is no longer pinned **fails** as a stale leftover (it should be removed on rotation). It does **not** do chain validation (that stays as the documented `openssl` recipe below, so the tool never hardcodes a vendor's CA).

## Verifying as an external auditor

```bash
cd <attestant-id>/

# 1. The attested key matches the pinned fingerprint:
openssl x509 -in key-attestation.pem -pubkey -noout \
  | openssl pkey -pubin -outform DER | shasum -a 256
#    → compare to the matching entry in ../trust-constants.json

# 2. The attestation chains to the vendor root, and read the policies:
openssl verify -CAfile ../yubico-piv-root.pem -untrusted intermediate.pem key-attestation.pem
openssl x509 -in key-attestation.pem -text -noout | grep -A2 '1.3.6.1.4.1.41482.3.8'
#    (Yubico PIV extension 1.3.6.1.4.1.41482.3.8 encodes PIN policy + touch policy;
#     .7 = serial, .3 = firmware version)
```

## Conventions

- **In lockstep with the trusted set.** This directory reflects the *currently* pinned keys, not their history (history is in git, as with the rest of `releases/trust/`). Add a key's bundle in the same release that pins its fingerprint; remove a bundle in the same release that unpins it. `provenance check` enforces this — a bundle for a fingerprint that is no longer pinned fails as a stale leftover. To audit a past release, check out its tag: the directory at that commit holds the evidence for the key(s) trusted then.
- **Vendor roots are reference-only.** Any `*-root.pem` committed here is for offline auditor convenience (repos outlive URLs). It is **not** a trust anchor for anything Eidola verifies — do not reference it from any file a `build.rs` reads. Record where it came from and its sha256 in commit history.
- The device serial is disclosed by the attestation cert. For a project trust root that is acceptable (and aids accountability); note it if that is a concern for a given attestant.
