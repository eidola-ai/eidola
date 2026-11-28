# Provisioning a YubiKey for Apple CI signing

> [!CAUTION]
> **This key is NOT the human release-attestation key.** Their policies are intentionally inverted: the [human release-attestation YubiKey](release-attestant-yubikey.md) requires physical touch for every signature, while this Apple code-signing key must **not** require touch because CI signs macOS releases unattended. Never reuse one device for both roles or copy policies between them.

This runbook defines the custody boundary, Apple certificate requirements, and operational hand-off for the hardware token that holds Eidola's Apple Developer ID Application signing key. For how Apple's non-reproducible signature is bound back to Eidola's reproducible unsigned macOS build, see [`docs/apps.md`](../apps.md#macos-and-apple-signing).

## Role separation at a glance

| Property | Human release-attestation key | Apple CI signing key (this doc) |
| --- | --- | --- |
| **Purpose** | Signs human release attestations affirming build reproduction, diff review, and freedom from coercion | Signs and notarizes macOS `.app` releases for Gatekeeper |
| **Device custody** | Held personally by each release officer | Connected to the self-hosted macOS CI runner |
| **Touch policy** | `ALWAYS` (physical presence required) | `NEVER` (unattended CI) |
| **PIN custody** | Known only to the officer; prompted interactively | Stored only in GitHub's protected Apple-signing environment |
| **Runbook** | [`release-attestant-yubikey.md`](release-attestant-yubikey.md) | This document |

Possession of the CI runner or the token alone must not be sufficient to sign: the private key stays non-exportable on the YubiKey, while its PIV PIN lives only as a secret in GitHub's protected release-tag signing environment.

## Provisioning boundary

> [!NOTE]
> **Status — intentionally stops before key-generation commands.** The exact PIV slot, PIN policy, and noninteractive CryptoTokenKit unlock mechanism have not yet been selected and empirically verified on the actual macOS runner. Because PIV key policies are immutable once generated, do not provision the token until those choices are approved and tested on the target runner.

Fixed requirements that apply regardless of final slot selection:

- **On-token generation only.** Generate the private key on the YubiKey; host tooling constructs the PKCS#10 certificate signing request (CSR) and the on-token key signs it. Never generate a private key in Keychain Access, import one into the token, export a `.p12`, or place a private key in a repository, runner config, or GitHub secret. Apple documents smart-card / hardware-token support for `codesign` in [TN3161: Inside Code Signing: Certificates](https://developer.apple.com/documentation/technotes/tn3161-inside-code-signing-certificates).
- **Touch policy `NEVER`.** Unattended CI cannot satisfy physical presence. Touch policy must be set at key-generation time; see Yubico's [PIV policy guide](https://docs.yubico.com/yesdk/users-manual/application-piv/pin-touch-policies.html) and [`ykman` PIV reference](https://docs.yubico.com/software/yubikey/tools/ykman/PIV_Commands.html).
- **Environment-scoped PIN.** The PIV PIN lives only as a secret in GitHub's protected Apple-signing environment. Never log it, pass it on a command line, or persist it on the runner.

## Certificate issuance and installation

1. The Apple Developer Program **Account Holder** (the role Apple requires for Developer ID Application certificates) submits the CSR through Apple's [Developer ID certificate procedure](https://developer.apple.com/help/account/certificates/create-developer-id-certificates/).
2. Import the resulting public certificate into the same YubiKey slot as the matching on-token private key. Never use a certificate whose private key was generated off-token.
3. Confirm macOS exposes a complete Developer ID identity through CryptoTokenKit.
4. Before enabling release signing in CI, test signing a non-release fixture on the actual runner and confirm `codesign` uses the hardware-token identity rather than a file-based key. *(The CI workflow does not yet implement this path, so no workflow command is prescribed here.)*

## Environment and runner hand-off

- Create the protected Apple-signing GitHub environment before enabling signing, and store the PIV PIN secret only there.
- Restrict the environment to release tags and the dedicated signing workflow — never expose it to general `main` builds, PRs, or nightlies.
- Ensure jobs never leave the PIN cached or written on the runner after completion.
- Keep Apple signing assets and key-dependent claims strictly outside `artifact-manifest.json` (enforced by `scripts/check-manifest-determinism.sh`; see [`.github/AGENTS.md`](../../.github/AGENTS.md)).

Record the following custody metadata **outside the repository** (keep PUK and management-key recovery material in the organization's approved secret store, separate from the runner):

- Token serial number and firmware version
- Key role and PIV slot
- Certificate SHA-1 identity, Apple Team ID, and expiry date
- Responsible Apple Account Holder and GitHub environment owners

## Renewal, revocation, and timestamps

- **Calendar the expiry well in advance** of the five-year Developer ID certificate term. Renewal uses a fresh on-token key, host-generated CSR, new Apple certificate, runner verification, and a controlled identity switch — never overwrite an active identity blindly.
- **Always include Apple's secure timestamp.** Apple requires a secure timestamp for notarization, and it allows users to keep running releases signed while the certificate was valid even after the certificate later expires.
- **Revocation is different from expiry:** a revoked Developer ID certificate prevents affected apps from installing or launching, and new releases require a current certificate.

References: Apple's [notarization requirements](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution), [Developer ID certificate-expiry guidance](https://developer.apple.com/support/developer-id/), and [TN3161](https://developer.apple.com/documentation/technotes/tn3161-inside-code-signing-certificates).
