# Apps & platform verification

[Verifiable releases](releases.md) explains how a release becomes trustworthy: reproducible builds, a CI-signed manifest, and named human attestations on a public transparency log. That still leaves the last mile — **how do you know the app sitting on *your* device actually matches that verified release?**

Every operating system puts different rules between public source code and a runnable app. Linux lets you hash the exact file you install. macOS requires Apple's non-reproducible code signature and notarization ticket so Gatekeeper will open the app. Mobile app stores go further — re-signing, repackaging, or encrypting every download themselves.

This page walks through how Eidola handles each platform, what you can verify independently, and where each platform forces a limit.

## Desktop and command line

### Linux

Linux gives the cleanest verification path because no third-party signing wrapper stands between the reproducible build and the file you install. Every shipped executable — the GUI, the bundled `eidola` command-line tool, and the `llama-server` local-model runtime — is copied directly into the measured package rather than loosely referenced by path, so any substitution changes the hash.

- **Static Linux CLI (`amd64` and `arm64`).** A single self-contained static binary built with a source-bootstrapped toolchain (StageX) and recorded directly by SHA-256 in the manifest.
- **Debian / Ubuntu package (`.deb`).** Built reproducibly from source. The file you download in your browser *is* the exact byte stream recorded in `artifact-manifest.json` — a single `sha256sum` checks the whole installable. It links against your system's own graphics and system libraries (your operating system is already in your trusted base), and updates arrive through Eidola's own verified update flow rather than a third-party apt repository.
- **Nix package.** Available for NixOS and Nix users who want to install from or rebuild the flake directly. (Unlike the `.deb`, the Nix archive resolves its libraries from `/nix/store`, so it is meant for Nix systems and reproducible rebuilds rather than standalone extraction.)

### macOS and Apple signing

macOS introduces a tension between reproducibility and usability. To open smoothly on a Mac without triggering Gatekeeper warnings, an app must be signed with an **Apple Developer ID** certificate and **notarized** by Apple. Signing modifies the executables inside the `.app` bundle using private keys and attaches a notarization ticket issued by Apple — which means a signed macOS download can never be reproduced byte-for-byte from source code alone.

Eidola resolves this by keeping two layers cleanly separate:

1. **The reproducible unsigned app.** Built deterministically from public source with Nix, packed into a canonical archive, and recorded in the CI-signed `artifact-manifest.json`. Anyone on a Mac can rebuild this exact unsigned app from source without holding any signing keys.
2. **Apple's detached signature and notarization material.** Rather than putting key-dependent hashes into the reproducible manifest (which would break determinism), Eidola publishes the detached Apple signature material alongside the signed shipping zip. Both hashes — along with Eidola's Apple Team ID and signing identifier — are recorded inside the **human release attestation**. Before an attestant can sign a release, Eidola's release tooling reconstructs the signed app from the unsigned build plus the detached signatures and refuses to proceed unless the result matches the shipping download byte for byte.

This gives you two ways to verify a macOS download:

- **Quick check:** hash the downloaded zip and compare it against the signed hash in the verified human attestation.
- **Full reconstruction from source:** rebuild the unsigned app from the release commit, apply the published detached signature bundle (`just verify-apple`), re-zip with the deterministic packer (`just pack-shipping-zip`), and confirm you land on the exact bytes of the shipping download.

**What Apple sees and controls.** Notarizing a release sends a copy of the binary to Apple for scanning and ties the signature to Eidola's Apple developer team identity. Apple retains the ability to revoke Developer ID certificates or notarization tickets for Gatekeeper. This does *not* give Apple a way to silently modify Eidola's reproducible code — and anyone who wants to bypass Apple's signing layer altogether can build and run the unsigned app directly from source.

## Mobile apps

Phones operate under a completely different constraint than desktops: neither the Apple App Store nor Google Play distributes the exact bytes a developer builds, and store review gates every binary release. Shipping an ordinary native app through app stores would break reproducible verification and delay security updates behind store review.

### The bundle and the native shell

To keep core security properties verifiable under app-store rules, Eidola splits the mobile app into two pieces with very different jobs:

- **The bundle** is the real application: Eidola's shared Rust core compiled to WebAssembly, running inside the phone's system webview. Every security-sensitive decision happens here — TLS encryption, hardware enclave attestation against the compiled-in trust root, the anonymous wallet, local database management, and release verification. The bundle builds reproducibly from source (using the same source-bootstrapped StageX toolchain as our server images), and every file in it is covered by a single **bundle digest**.
- **The native shell** is a tiny platform wrapper written directly against system frameworks with zero third-party libraries. It embeds a complete bundle so the app works from first launch without downloading code, serves the bundle to the webview from a private origin, carries only *already-encrypted* raw TCP bytes (it never terminates TLS and never sees prompts or answers in the clear), keeps the app's private files on disk, opens checkout and policy links strictly in the system browser, and draws a native **Bundle digest** sheet when you long-press the app icon on your home screen — outside the webview's control so the bundle cannot spoof it.

### Verifying on Android vs. iOS

| Piece | Android | iOS |
| --- | --- | --- |
| **The app bundle** | Built reproducibly; running digest shown natively by the shell | The same |
| **The native shell** | Unsigned APK reproduces byte for byte; signed APK strips back to it; Play installs compare file by file | **Cannot be verified** (encrypted by Apple) |
| **OS and system webview** | Trusted, as the desktop app trusts its operating system | The same |

- **Checking the bundle (Android and iOS).** Long-press the Eidola icon on your home screen and choose **Bundle digest**. The native sheet displays the running bundle's version, whether it is confirmed or booting on trial, and its SHA-256 digest — which you can compare directly against any verified release's `artifact-manifest.json`, or reproduce from source with Docker (`just mobile-bundle-release`). That sheet is only as trustworthy as the native shell drawing it.
- **Verifying the Android shell.** The unsigned Android APK builds reproducibly from source on both macOS and Linux. Alongside Google Play, every release publishes a directly signed APK whose signature block can be stripped to recover the exact unsigned APK recorded in `artifact-manifest.json` (with the signed APK's own hash and signing certificate fingerprint bound in the human attestation). Google Play rebuilds and re-signs apps with Google's own key, so while a Play install cannot match byte for byte, it can still be checked file by file to confirm that its compiled code (`classes.dex`) and embedded bundle match the published build unchanged.
- **Verifying the iOS shell: blocked by Apple.** Apple re-signs and encrypts every binary distributed through the App Store, so **nobody — Eidola included — can compare an installed iOS shell with a build from source**. While the shell's public source is tiny and never sees network plaintext, an unverified native shell still controls the app's local files, its webview, and the digest sheet it draws — the same reach iOS and WebKit already have over every app on the phone. On iOS, you are trusting Apple's App Store delivery and Eidola's upload of the shell for that device ([Threat model](threat-model.md#residual-trust)). A compromised shell affects only the phone it runs on; it cannot make an enclave attest falsely or compromise any other user.

### Mobile updates and storage safety

- **Attested bundle updates.** Opening the Updates screen runs the exact same release verification as desktop — checking CI's manifest signature, human attestations, Rekor inclusion proofs, and pinned claim templates. Nothing downloads until you tap **Install update**. The app hashes the downloaded archive and every extracted file against the manifest before handing them to the shell, and verifies the digest the shell computes over what it stored. Activation is provisional: if a newly installed bundle ever fails to boot and confirm itself, the shell automatically falls back to the last confirmed bundle on the next launch.
- **Crash-safe local storage.** Mobile webviews can evict browser storage at any time, so Eidola keeps nothing durable in webview storage. The core streams every database write through the shell's private app storage (excluded from cloud/device backups so a restored backup can never resurrect already-spent wallet tokens) and enforces a strict durability barrier: no issuance or spend request leaves the phone until its local recovery record is synced to disk.

## Checking a download yourself

Always start from a **verified release** (valid CI signature plus verified human attestations; see [Verifiable releases](releases.md)). Copying a hash off an unverified webpage proves nothing.

- **Linux `.deb` or static CLI:** run `sha256sum` on the downloaded file and compare directly with its entry in `artifact-manifest.json`.
- **macOS `.zip`:** run `shasum -a 256` and compare with the shipped-artifact hash in the verified human attestation — or run `just verify-apple <unsigned.tar.gz> <sigbundle.zip>` followed by `just pack-shipping-zip` to verify the full link back to source.
- **Android APK:** strip the signature block (`python3 mobile/android/tools/strip_apk_signature.py`) and compare the `sha256sum` with the unsigned APK in `artifact-manifest.json`, or check the signed APK's hash and certificate fingerprint against the human attestation.
- **Mobile bundle:** long-press the Eidola icon → **Bundle digest** and compare with the bundle digest in `artifact-manifest.json`.
- **Server and engine container images:** compare the registry digest (`crane digest`) against the OCI digest recorded in `artifact-manifest.json`.
