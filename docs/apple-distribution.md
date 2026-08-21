# Apple distribution

macOS is the one platform where a **release** currently grows an **envelope** on top of the measured **archive**. The vocabulary is defined in [verification.md](verification.md); this page is Apple-specific: what Apple sees, what the envelope contains, and why those bytes never enter `artifact-manifest.json`.

No Apple-signed Eidola **installable** is published yet. Until then there is no Gatekeeper zip and no Apple-verification command to run. The unsigned **payload** can already be built from source (`nix build .#eidola-gui-macos-universal`) and packed (`nix build .#eidola-gui-macos-universal-archive`).

## Two layers

The **unsigned, reproducible payload** is the trust object: here, "unsigned" means not Developer ID-signed. Its Mach-O executables can still carry the ad-hoc signatures Nix adds during the build. Those bytes are the **archive** whose `archiveSha256` (and `narHash`) sit in the deterministic, CI-signed artifact manifest.

Apple signing and notarization are the **envelope**: a compatibility wrapper for Gatekeeper, hardened runtime, and macOS capabilities. They do not replace the trust object. The **installable** is `apply(archive, envelope)` — a new byte sequence. `codesign` rewrites Mach-Os; the unmodified archive is not sitting inside the signed zip.

## What Apple receives and controls

For each notarized release, Eidola sends Apple a copy of the installable. Apple scans it, issues a notarization ticket if it accepts it, and makes that ticket available to Gatekeeper. Apple therefore sees every notarized release binary. This is an intentional disclosure, not an inference from our privacy architecture: see Apple’s [notarization overview](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution).

Apple controls important distribution-trust decisions. It documents that a developer can work with Apple to revoke notarization tickets for unauthorized software, and it says an app signed with a revoked Developer ID certificate can no longer be installed or launched. The code-signing identity carries an Apple Team ID, which connects the signature to the Apple developer team and thus exposes the legal identity Apple enrolled for distribution. See Apple’s [notarization overview](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution), [Developer ID guidance](https://developer.apple.com/support/developer-id/), and [account-role documentation](https://developer.apple.com/help/app-store-connect/manage-your-team/overview-of-accounts-and-roles/).

This does not give Apple a way to alter the reproducible payload silently. Envelope hashes live in the human attestation, not in `artifact-manifest.json`. A user who does not want the Apple wrapper can build and run the unsigned payload instead, subject to the normal macOS behaviour for unsigned software.

## What a signed macOS release will contain

Three related objects:

- The unsigned **archive** (flake-built `.tar.gz` of the `.app` payload), whose `archiveSha256` is in `artifact-manifest.json`.
- A detached Apple-signature **envelope**, containing the code-signature material and notarization ticket needed to reconstruct the installable.
- The signed, notarized, stapled **installable** intended for normal browser download.

The human release attestation binds the hashes of the installable and the detached envelope, plus the expected Apple Team ID and signing identifier. The artifact manifest must never carry those Apple-dependent values; it remains reproducible and key-independent. For the rest of the release trust chain, see [Releases](releases.md) and [Trust root](trust-root.md).

The stapled ticket lets Gatekeeper find the notarization result when a Mac is offline. Apple documents that an unstapled distribution can be blocked while offline. [Customizing the notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow) describes the ticket and stapling flow.

## Verify a download

Once a release publishes the Apple assets above, follow [verification.md](verification.md#checking-a-download). The one-line check of the browser zip is against the **attested installable** hash, not against `archiveSha256`. Binding that zip to the git ref is `apply(archive, envelope) = installable` (the designed path; `just verify-apple`) or a true inverse of `codesign` — not ignoring extra files in the zip.

Independent rebuild of the payload:

```bash
nix build .#eidola-gui-macos-universal
nix build .#eidola-gui-macos-universal-archive
```

Compare `narHash` / `archiveSha256` with the CI-signed `artifact-manifest.json`.
