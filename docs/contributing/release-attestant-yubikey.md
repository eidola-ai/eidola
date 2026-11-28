# Provisioning a YubiKey as a release-attestant key

This runbook is for a release officer provisioning a fresh YubiKey 5 (such as a 5C) as a human release-attestation signing key — a key whose SHA-256 SPKI fingerprint is pinned in `releases/trust/trust-constants.json` (`trusted_attestant_fingerprints`) and which signs each release attestation via `just release-attest`.

For how human attestations fit into Eidola's trust root, see [`docs/releases.md`](../releases.md#the-compiled-in-trust-root). For rotating a key that is already in service, see [`releases/README.md`](../../releases/README.md#rotating-an-attestant-key).

> [!IMPORTANT]
> **This is the human-presence key and must require physical touch.** Do not confuse it with the separate unattended [Apple CI signing YubiKey](release-apple-signing-yubikey.md), which must **not** require touch.

## Why a YubiKey, and the four key decisions

The attestant key authorizes software updates for every client. Generating it on a purpose-built hardware token — where the private key is non-exportable and requires both a PIN and a physical touch for every signature — is materially stronger than a software key derived on a general-purpose computer, where the private key sits in RAM or on disk and can be exfiltrated if the machine is compromised.

Four decisions govern provisioning, and two of them are **immutable** once the key is generated:

- **Slot `9c` ("Digital Signature").** The PIV standard mandates that slot `9c` requires the PIN on *every* private-key operation. The other slots are wrong for this role: `9a` (Authentication) caches the PIN for the whole session, `9e` (Card Authentication) requires no PIN at all, and `9d` (Key Management) is for decryption. Slot `9c` enforces "one PIN entry → one signature" in hardware.
- **Algorithm `ECCP256` (NIST P-256).** The client verifier (`verify_blob_signature_with_spki`) accepts only ECDSA-P256, ECDSA-P384, or Ed25519. P-256 works on every YubiKey 5 firmware and is the best-tested path through cosign's PKCS#11 layer. (Ed25519 in PIV requires firmware 5.7+ and is less battle-tested through `libykcs11`.)
- **PIN policy `ALWAYS` (immutable).** Default for slot `9c`, set explicitly here: requires the PIN on every signature.
- **Touch policy `ALWAYS` (immutable).** Requires a physical touch on every signature so malware that scrapes the PIN still cannot sign without a human present. Because `cosign sign-blob` makes one signature per invocation, `ALWAYS` costs one touch per attestation.

Always generate the key **on the device** — never import one.

## Prerequisites

### macOS

```bash
# ykman for provisioning; yubico-piv-tool provides libykcs11 for cosign.
brew install ykman yubico-piv-tool

ykman piv info   # sanity check; record the firmware version and serial
```

The PKCS#11 library lives at `/opt/homebrew/lib/libykcs11.dylib` on Apple Silicon (`/usr/local/lib/libykcs11.dylib` on Intel).

### Linux

Install `yubikey-manager` (`ykman`), `yubico-piv-tool` (provides `libykcs11.so`), and `pcscd` through your distro package manager, and ensure the `pcscd` service is running before running `ykman piv info`.

### Build `cosign` with PKCS#11 support

PKCS#11 support in `cosign` is a compile-time build tag. Prebuilt binaries from Homebrew and GitHub Releases omit `pivkey` / `pkcs11key`, so signing with a `pkcs11:` URI fails with `This cosign was not built with pkcs11-tool support!`. Build `cosign` from source with cgo and those tags enabled:

```bash
# Requires a C toolchain (on macOS: Xcode Command Line Tools).
# This is the `go install` equivalent of cosign's `cosign-pivkey-pkcs11key` Makefile target.
CGO_ENABLED=1 go install -tags=pivkey,pkcs11key github.com/sigstore/cosign/v2/cmd/cosign@latest
```

Make sure `$(go env GOPATH)/bin` precedes any Homebrew/system `cosign` on your `PATH` (confirm with `which cosign`). *(Note: `cosign-pivkey-pkcs11key` is only a Makefile target name, not a Go package path. Cloud KMS URIs such as `awskms:`, `gcpkms:`, `azurekms:`, and `hashivault:` work with stock `cosign` if you ever use KMS instead of PKCS#11.)*

## Provisioning steps

### 1. Change factory credentials

Factory defaults are PIN `123456`, PUK `12345678`, and a well-known management key — change all three before generating keys:

```bash
# Optional: set retry counts FIRST — `set-retries` resets PIN and PUK to defaults.
# ykman piv access set-retries 3 3

ykman piv access change-pin    # 6–8 chars; entered per release signature
ykman piv access change-puk    # unblocks the PIN after too many wrong tries — store safely

# Generate a random management key and protect it behind the PIN:
ykman piv access change-management-key --generate --protect
```

Store the PUK as carefully as the PIN: if the PIN locks (default 3 failed attempts) and the PUK is lost, the PIV application cannot be unblocked without resetting the whole device and losing the key.

### 2. Generate the signing key on-device

```bash
ykman piv keys generate \
  --algorithm ECCP256 \
  --pin-policy ALWAYS \
  --touch-policy ALWAYS \
  9c attestant-pub.pem
```

`attestant-pub.pem` contains only the **public** key — the private key never leaves the token.

### 3. Generate a self-signed certificate into slot `9c`

`libykcs11` only exposes a slot's key as a PKCS#11 object when a certificate is stored alongside it:

```bash
ykman piv certificates generate \
  --subject "CN=Eidola Release Attestant" \
  9c attestant-pub.pem
```

### 4. Discover the PKCS#11 URI and fingerprint

Use `release-tool`'s built-in enumerator. It reads only public objects on the token, so it **never prompts for or prints a PIN**. It prints both the cosign `--key` URI (identifying the token by stable label rather than a volatile `slot-id`) and the SHA-256 SPKI fingerprint reconstructed identically to what `release-tool attest` computes at sign time:

```bash
just release-list-keys
```

Example output:

```text
  YubiKey PIV #12345678  (serial 12345678)
    algorithm   : ECDSA-P256
    label       : Private key for Digital Signature
    id          : 02
    fingerprint : 4e1c0a8f93b2d7e65a0f1c8b4d9e2a73f6051b8c2d4e7a9f0b3c6d8e1a2f4b7c
    uri         : pkcs11:token=YubiKey%20PIV%20%2312345678;id=%02;type=private?module-path=/opt/homebrew/lib/libykcs11.dylib
```

Select the **Digital Signature** entry (`id 02`, `Private key for Digital Signature`) — that is the slot `9c` key you just generated. Ignore the second entry (`id 19`, `Private key for PIV Attestation`): that is factory slot `F9`, which cannot sign releases and is used only to vouch for hardware provenance in Step 6. Any key with an algorithm the client verifier rejects is flagged with its fingerprint omitted.

The printed `uri` contains no secret and is safe to persist as `EIDOLA_ATTESTANT_COSIGN_KEY`.

> [!WARNING]
> **Avoid `cosign pkcs11-tool list-keys-uris`.** It prints your PIV PIN in plaintext inside every URI (`…&pin-value=<PIN>`) and identifies tokens by a volatile `slot-id` that breaks cosign's interactive PIN prompt. If you ever run it and the PIN appears in a terminal log, paste, or screen share, treat the PIN as exposed and rotate it immediately with `ykman piv access change-pin`.

### 5. Pin the fingerprint

Add the `fingerprint` from Step 4 to `trusted_attestant_fingerprints` in `releases/trust/trust-constants.json`. `crates/eidola-app-core/build.rs` embeds that file at compile time, and clients reject any attestation whose signing-key fingerprint is not in the pinned set.

- **Adding a new attestant (or rotating an active key):** Append the new fingerprint while **keeping** any currently active fingerprints during the overlap window, then follow the two-release procedure in [`releases/README.md`](../../releases/README.md#rotating-an-attestant-key). Never delete an active key in the same release that introduces its replacement.

If you kept `attestant-pub.pem` from Step 2, you can double-check the fingerprint offline without touching the device:

```bash
openssl pkey -pubin -in attestant-pub.pem -outform DER | shasum -a 256 | awk '{print $1}'
```

### 6. Capture on-device hardware provenance

Factory slot `F9` can issue a Yubico-signed attestation certificate proving that your slot `9c` key was generated on-hardware and recording its `pin-policy` and `touch-policy`. Committing this bundle lets external auditors verify the key genuinely lives on a hardware token rather than taking it on faith. **No client or build step consults this evidence** — it is strictly auditor-facing (see [`releases/trust/attestant-provenance/README.md`](../../releases/trust/attestant-provenance/README.md)).

```bash
just release-provenance-capture your-name   # captures the cert bundle + fills meta.json
just release-provenance-check               # confirms the cert matches the pinned fingerprint
```

Inspect the derived `product` string in `meta.json` to confirm it reads sensibly. (`just release-provenance-enrich` re-derives metadata fields offline from a committed cert if you ever rebuild a bundle.) `release-provenance-check` asserts that the cert's public-key fingerprint matches `trust-constants.json` so provenance cannot silently drift from the pinned key; verifying the chain to Yubico's root CA stays a documented `openssl` step in that README.

## Signing releases with the key

Export your attestant identity once (for example, in your shell profile or `.envrc`), using the `uri` from Step 4:

```bash
# Carries no secret — safe to persist.
export EIDOLA_ATTESTANT_COSIGN_KEY='pkcs11:token=YubiKey%20PIV%20%2312345678;id=%02;type=private?module-path=/opt/homebrew/lib/libykcs11.dylib'
export EIDOLA_ATTESTANT_ID='your-name'
export EIDOLA_ATTESTANT_NAME='Your Name'
export EIDOLA_ATTESTANT_JURISDICTION='the State of California, United States'
```

For a release tag that CI has already built and signed:

```bash
just release-verify vX.Y.Z   # fetch & verify CI-signed manifest; inspect diff against prior release
just release-attest vX.Y.Z   # render claims, prompt to affirm each, sign with the YubiKey
```

When `EIDOLA_ATTESTANT_COSIGN_KEY` is a `pkcs11:` URI and `COSIGN_PKCS11_PIN` is unset, `release-attest` prompts for your PIN once (without echo) and passes it to its `cosign` subprocesses. After you review and affirm each claim by typing `yes`, `cosign sign-blob` asks for a physical touch on the YubiKey. (`COSIGN_PASSWORD` is unused for PKCS#11 keys; `release-tool` also checks the public-key algorithm and warns if the fingerprint is not yet pinned.)

All four values can also be passed per invocation instead of via environment variables:

```bash
just release-attest vX.Y.Z \
  --cosign-key 'pkcs11:token=YubiKey%20PIV%20%2312345678;id=%02;type=private?module-path=/opt/homebrew/lib/libykcs11.dylib' \
  --attestant-id your-name \
  --attestant-name "Your Name" \
  --jurisdiction "the State of California, United States"
```

## Operational notes

- **Record device metadata outside the repo:** YubiKey serial (from `ykman piv info`), firmware version, pinned fingerprint, and role (primary / backup), so key rotations stay unambiguous.
- **Interactive use only:** Never wire a human attestant key into unattended CI, and never weaken its PIN or touch policy to "make automation easier." CI signs `artifact-manifest.json` separately via keyless Fulcio/OIDC.
- **Firmware is fixed:** YubiKey firmware cannot be updated in place; simply record the version at provisioning time. Any YubiKey 5 series device supports `ECCP256` in PIV slot `9c`.
