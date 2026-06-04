# Provisioning a YubiKey as the release attestant key

This guide is for a release engineer setting up a fresh YubiKey 5 (e.g. a 5C) as a human release-attestation signing key — a key whose sha256 SPKI fingerprint is pinned in `releases/trust/trust-constants.json` (`trusted_attestant_fingerprints`) and which signs each release attestation via `just release-attest`.

For the conceptual model of what the attestant key is and how it fits the trust root, see [`../trust-root.md`](../trust-root.md). For the rotation procedure once a key is already in service, see [`../../releases/README.md`](../../releases/README.md#rotating-an-attestant-key). This document covers the one-time hardware provisioning that those assume has already happened.

## Why a YubiKey, and the key decisions

The attestant key authorizes software updates for every client. Holding it on a purpose-built device — where the private key is generated on-chip, is non-exportable, and requires a PIN (and a physical touch) for each signature — is materially stronger than a `cosign generate-key-pair` key derived on a general-purpose computer, where the private key exists in memory and on disk and can be exfiltrated by any process that compromises the machine.

Four decisions matter, and two of them are immutable once the key is generated:

- **Slot `9c` ("Digital Signature").** This is not arbitrary. The PIV standard mandates that slot 9c checks the PIN on *every* private-key operation. The alternatives are wrong for this use case: `9a` (Authentication) caches the PIN for the whole session, so one PIN unlocks unlimited signatures; `9e` (Card Authentication) requires no PIN at all; `9d` (Key Management) is for decryption. Slot 9c gives you exactly "one PIN entry → one signature," enforced by hardware.
- **Algorithm `ECCP256` (NIST P-256).** The updater's `verify_blob_signature_with_spki` accepts only ECDSA-P256, ECDSA-P384, or Ed25519. P-256 is supported on every YubiKey 5 firmware and is the best-tested path through cosign's PKCS#11 layer. Ed25519 in PIV exists only on firmware 5.7+ and is less battle-tested through `libykcs11`; prefer P-256 unless you have a specific reason.
- **PIN policy `ALWAYS` (immutable after generation).** The default for slot 9c, but set it explicitly. Requires the PIN for every signature.
- **Touch policy `ALWAYS` (immutable after generation).** Requires a physical touch for every signature, so that malware which has scraped the PIN still cannot sign without a human present at the device. Because `cosign sign-blob` performs a single signature, `ALWAYS` costs exactly one touch per release.

Generate the key **on the device** and never import — that is the entire point. PIV private keys are non-exportable by design, so on-device generation means the private key has never existed in a computer's memory.

## Prerequisites (macOS)

```bash
# ykman for provisioning; yubico-piv-tool provides libykcs11 for cosign.
brew install ykman yubico-piv-tool

ykman piv info   # sanity check; record the firmware version and serial
```

The PKCS#11 module is at `/opt/homebrew/lib/libykcs11.dylib` on Apple Silicon (`/usr/local/lib/libykcs11.dylib` on Intel).

**`cosign` must be built with PKCS#11 support.** PKCS#11 in cosign is a build-time option. The `cosign` distributed via Homebrew and the GitHub releases is compiled without the `pivkey` / `pkcs11key` build tags, so `cosign sign-blob` against a `pkcs11:` key (the signing recipe below) fails with `This cosign was not built with pkcs11-tool support!`. Build cosign from source with those tags:

```bash
# pkcs11key links a C PKCS#11 library via cgo, so a C toolchain is required
# (on macOS: Xcode command line tools). This is the `go install` equivalent of
# cosign's `cosign-pivkey-pkcs11key` Makefile target.
CGO_ENABLED=1 go install -tags=pivkey,pkcs11key github.com/sigstore/cosign/v2/cmd/cosign@latest
```

Note that `cosign-pivkey-pkcs11key` is only the *name of cosign's Makefile target* — there is no `cmd/cosign-pivkey-pkcs11key` package to install; the target simply rebuilds the ordinary `./cmd/cosign` with those tags and cgo enabled. The command above puts a PKCS#11-capable `cosign` in `$(go env GOPATH)/bin`; make sure that directory precedes the Homebrew `cosign` on your `PATH` (`which cosign` to confirm). A cosign-supported KMS URI (`awskms:`, `gcpkms:`, `azurekms:`, `hashivault:`) works with the stock binary if you would rather not build from source. This is the same caveat documented on the `release-attest` recipe in the justfile.

## Provisioning steps

### 1. Set credentials

Factory defaults are PIN `123456`, PUK `12345678`, and a well-known management key — all must change.

```bash
# Optional: set retry counts FIRST — set-retries resets PIN and PUK to defaults.
# ykman piv access set-retries 3 3

ykman piv access change-pin    # 6–8 chars; this is what you type per signature
ykman piv access change-puk    # unblocks the PIN after too many wrong tries — store it safely

# Generate a random management key and seal it behind the PIN so you don't track it separately:
ykman piv access change-management-key --generate --protect
```

Store the PUK as carefully as the PIN: if the PIN locks (default 3 tries) and the PUK is lost, the slot is unrecoverable.

### 2. Generate the signing key on-device

```bash
ykman piv keys generate \
  --algorithm ECCP256 \
  --pin-policy ALWAYS \
  --touch-policy ALWAYS \
  9c attestant-pub.pem
```

`attestant-pub.pem` is the **public** key — the only material that leaves the device.

### 3. Generate a self-signed certificate into the slot

`libykcs11` only exposes a slot's key as a PKCS#11 object if a certificate is present alongside it, so this step is required for cosign to see the key.

```bash
ykman piv certificates generate \
  --subject "CN=Eidola Release Attestant" \
  9c attestant-pub.pem
```

### 4. Discover the URI and fingerprint

Use `release-tool`'s built-in enumerator. It reads only public objects on the device, so it **never prompts for or prints a PIN**. It emits both the cosign `--key` URI (token identified by stable label, no `slot-id`) and the **sha256 SPKI fingerprint** to pin — reconstructed from the public key on the device, byte-identical to what `attest` derives at sign time:

```bash
just release-list-keys
```

```text
  YubiKey PIV #37842605  (serial 37842605)
    algorithm   : ECDSA-P256
    label       : Private key for Digital Signature
    id          : 02
    fingerprint : 4e1c0a8f93b2d7e65a0f1c8b4d9e2a73f6051b8c2d4e7a9f0b3c6d8e1a2f4b7c9
    uri         : pkcs11:token=YubiKey%20PIV%20%2342342556;id=%02;type=private?module-path=/opt/homebrew/lib/libykcs11.dylib
```

You want the **Digital Signature** entry — `id 02`, label `Private key for Digital Signature` — the slot 9c key you just generated. A second entry, `id 19` / `Private key for PIV Attestation`, will also appear: that is PIV slot `F9`, the factory-provisioned attestation key (it chains to a Yubico CA and only vouches that *other* on-device keys were hardware-generated). You cannot sign with it — ignore it here, though it is useful as evidence (see [Capturing on-device proof](#capturing-on-device-proof-optional)). The command flags any key whose algorithm the updater would reject and omits a fingerprint for it.

The printed `uri` is the value you use as `--cosign-key` / `EIDOLA_ATTESTANT_COSIGN_KEY`; it contains no secret, so it is safe to persist. The printed `fingerprint` is what you pin in the next step.

> **Why not `cosign pkcs11-tool list-keys-uris`?** It works, but it bakes your PIN into every URI it prints (`…&pin-value=<PIN>`, in plaintext — surprising, since `ykman` masks the PIN on entry) and identifies the token by a volatile `slot-id` that breaks cosign's own PIN prompt (it dies in `GetTokenInfo` before prompting — which is also why `release-attest` supplies the PIN via `COSIGN_PKCS11_PIN` rather than relying on that prompt). If you ever do run it and the PIN ends up somewhere observable (a paste, a screen share, a recording), treat the PIN as disclosed and rotate it: `ykman piv access change-pin`.

### 5. Pin the fingerprint

Replace the existing entry in `releases/trust/trust-constants.json` (`trusted_attestant_fingerprints`) with the `fingerprint` from step 4. `crates/eidola-app-core/build.rs` reads this file at compile time and bakes the value into `TRUSTED_ATTESTANT_FINGERPRINTS`; the runtime updater rejects any attestation whose signing-key fingerprint is not in that set. This is the genesis pinning — a direct replacement, not an overlap rotation, because the previous (software) key never signed a shipped release.

If you kept `attestant-pub.pem` from step 2, you can confirm you pinned the right key without touching the device — both should print the same hash:

```bash
openssl pkey -pubin -in attestant-pub.pem -outform DER | shasum -a 256 | awk '{print $1}'
```

## Using the key with the attestation script

The release engineer's workflow is two recipes. Set the attestant identity once (preferably in your shell profile or `.envrc`), pointing `EIDOLA_ATTESTANT_COSIGN_KEY` at the URI from step 4:

```bash
# This URI carries no secret, so it is safe to persist.
export EIDOLA_ATTESTANT_COSIGN_KEY='pkcs11:token=YubiKey%20PIV%20%2342342556;id=%02;type=private?module-path=/opt/homebrew/lib/libykcs11.dylib'
export EIDOLA_ATTESTANT_ID='your-name'
export EIDOLA_ATTESTANT_NAME='Your Name'
export EIDOLA_ATTESTANT_JURISDICTION='the State of California, United States'
```

Then, for a tag CI has already built and signed:

```bash
just release-verify vX.Y.Z   # fetch + verify the CI-signed manifest, diff against the prior release
just release-attest vX.Y.Z   # render each claim, prompt to affirm, then sign with the YubiKey
```

When the key is a `pkcs11:` URI and `COSIGN_PKCS11_PIN` is not already set, `release-attest` **prompts for the PIN once** (no echo) and holds it in its environment for the cosign child processes — so no manual `read`/`export` dance is needed. (You may still export it ahead of time if you prefer; the prompt is skipped when it is already set.)

`release-attest` renders each attestation claim, prompts you to type `yes` to affirm it, then invokes `cosign sign-blob --key <ref>` — at which point the YubiKey requires a physical touch (the PIN is supplied non-interactively from the value entered above). No `COSIGN_PASSWORD` is needed for a PKCS#11 key (that variable is only for passphrase-encrypted local PEM keys); `release-tool` detects the `pkcs11:` URI and skips the passphrase check. It also fetches the public key, validates the algorithm is one the client accepts, and warns if the fingerprint is not yet pinned.

Per-invocation overrides are forwarded to `release-tool attest`, so you can pass `--cosign-key` / `--attestant-id` / `--attestant-name` / `--jurisdiction` on the command line instead of via the environment:

```bash
just release-attest vX.Y.Z \
  --cosign-key 'pkcs11:token=YubiKey%20PIV%20%2342342556;id=%02;type=private?module-path=/opt/homebrew/lib/libykcs11.dylib' \
  --attestant-id your-name \
  --attestant-name "Your Name" \
  --jurisdiction "the State of California, United States"
```

## Operational notes

- **Record the device metadata outside the repo:** YubiKey serial (from `ykman piv info`) → pinned fingerprint → role (primary / backup). You will want this when a device's fingerprint eventually rotates out.
- **This key is for the interactive human attestation step only.** It must never be wired into unattended CI — CI signs `artifact-manifest.json` keylessly via Fulcio/OIDC, a separate mechanism. Do not enable PIN caching or touch caching to "make automation easier."
- **Firmware is fixed.** A YubiKey's firmware cannot be updated; just note the version. Any YubiKey 5C performs ECCP256.

### Capturing on-device proof (optional)

The factory attestation key in slot F9 (the second object from step 4) can produce a Yubico-signed certificate proving that your 9c key was hardware-generated and stating its `pin-policy` / `touch-policy`. For a trust-root bootstrapping ceremony this is worth archiving alongside the device metadata:

```bash
ykman piv keys attest 9c slot9c-attestation.pem            # Yubico-signed cert describing the 9c key
ykman piv certificates export f9 yubico-attestation-ca.pem # the intermediate that signed it
```

These let a future auditor verify the genesis attestant key really lived on a YubiKey with PIN-per-signature enforced, rather than taking it on faith.
