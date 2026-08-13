# Apple distribution

Eidola’s macOS distribution has two layers with deliberately different jobs. The **unsigned, reproducible build is the primary trust object**: here, “unsigned” means not Developer ID-signed. Its Mach-O executables can still carry the ad-hoc signatures Nix adds during the build. The build is produced from source and its hash is recorded in the deterministic, CI-signed artifact manifest. Apple signing and notarization are a compatibility wrapper for Gatekeeper, hardened runtime, and macOS capabilities; they do not replace that trust object.

No Apple-signed Eidola release is published yet. This page describes the format and checks that will accompany the first one. Until then, there is no download or Apple-verification command to run.

## What Apple receives and controls

For each notarized release, Eidola sends Apple a copy of the release binary. Apple scans it, issues a notarization ticket if it accepts it, and makes that ticket available to Gatekeeper. Apple therefore sees every notarized release binary. This is an intentional disclosure, not an inference from our privacy architecture: see Apple’s [notarization overview](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution).

Apple controls important distribution-trust decisions. It documents that a developer can work with Apple to revoke notarization tickets for unauthorized software, and it says an app signed with a revoked Developer ID certificate can no longer be installed or launched. The code-signing identity carries an Apple Team ID, which connects the signature to the Apple developer team and thus exposes the legal identity Apple enrolled for distribution. See Apple’s [notarization overview](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution), [Developer ID guidance](https://developer.apple.com/support/developer-id/), and [account-role documentation](https://developer.apple.com/help/app-store-connect/manage-your-team/overview-of-accounts-and-roles/).

This does not give Apple a way to alter the reproducible build silently. The non-deterministic Apple material is published separately from the deterministic artifact manifest, and the published signed app must be reconstructible from the reproducible app plus that material. A user who does not want the Apple wrapper can build and run the reproducible unsigned app instead, subject to the normal macOS behaviour for unsigned software.

## What a signed macOS release will contain

A release will publish three related objects:

- The unsigned macOS app archive, whose SHA-256 is in `artifact-manifest.json`.
- A detached Apple-signature bundle, containing the code-signature material and notarization ticket needed to reconstruct the signed app.
- The signed, notarized, stapled archive intended for normal browser download.

The human release attestation binds the hashes of the signed archive and the detached bundle, plus the expected Apple Team ID and signing identifier. The artifact manifest must never carry those Apple-dependent values; it remains reproducible and key-independent. For the rest of the release trust chain, see [Releases](releases.md) and [Trust root](trust-root.md).

The stapled ticket lets Gatekeeper find the notarization result when a Mac is offline. Apple documents that an unstapled distribution can be blocked while offline. [Customizing the notarization workflow](https://developer.apple.com/documentation/security/customizing-the-notarization-workflow) describes the ticket and stapling flow.

## Verify your download

This section applies once a release publishes the Apple assets above. Start with the browser-download archive and the release’s verified human attestation. The first check is intentionally just one command:

```bash
shasum -a 256 Eidola-X.Y.Z-macos-universal.zip
```

Compare its 64-character digest with that release’s attested shipped-archive hash. A mismatch means do not open the archive. The attestation itself must be verified through the release process described in [Releases](releases.md); copying a hash from an unverified page does not establish trust.

For a deeper check, obtain the unsigned archive and detached signature bundle named by that same verified attestation. The cross-platform apply-and-inspect verifier is not published yet, so this page deliberately does not invent a command for it. Its release documentation will show how to reconstruct the signed app and confirm both the archive hash and the Apple identity.

For an independent rebuild, the reproducible macOS app can be built from the release source today:

```bash
nix build .#eidola-gui-macos-universal
```

Compare the resulting build identity with the CI-signed `artifact-manifest.json`. The release tooling will publish the concise, cross-platform comparison command when the unsigned shipping archive is introduced; today no such archive is published.
