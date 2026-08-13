# Provisioning a YubiKey for Apple CI signing

This key is **not** the human release-attestation key. The policies are intentionally inverted: the human-attestation key requires a physical touch for every signature, while this Apple code-signing key must not require touch because CI signs releases unattended. Do not substitute one key for the other or copy their policies.

The Apple key exists to keep the Developer ID private key non-exportable while still requiring the protected release environment to authorize its use. The YubiKey stays connected to the self-hosted macOS runner; its PIV PIN is stored only as a secret in GitHub’s protected Apple-signing environment. Possession of the runner or token alone must not be sufficient to sign.

## Provisioning boundary

Generate the private key on the YubiKey. Host tooling constructs and writes the PKCS#10 certificate signing request, and the on-token key signs it. Do not generate a private key in Keychain Access, import one into the token, export a `.p12`, or place a private key in a repository, runner configuration, or GitHub secret. Apple documents that `codesign` can use identities held by a smart card or other hardware token; the private key stays on that token. See [TN3161: Inside Code Signing: Certificates](https://developer.apple.com/documentation/technotes/tn3161-inside-code-signing-certificates).

The exact PIV slot, PIN policy, and unattended unlock mechanism have not yet been selected. They are security-significant and immutable or operationally binding, so this runbook intentionally stops before a token-generation command. Do not provision the key until those choices have been approved and tested on the intended runner.

The following requirements are already fixed:

- The private key is generated and retained on-token. Host tooling constructs the CSR and the token key signs it.
- The touch policy is `never`; unattended CI cannot satisfy physical presence.
- The PIV PIN is a GitHub environment secret, available only to the protected Apple-signing workflow. Never log it, put it in a command line, or save it on the runner.
- A human release-attestation key continues to use touch; it is a separate device and role.

Yubico documents that touch policy must be chosen when a PIV key is generated or imported, and that `Never` means no touch is required. Its [PIV policy guide](https://docs.yubico.com/yesdk/users-manual/application-piv/pin-touch-policies.html) and [`ykman` PIV reference](https://docs.yubico.com/software/yubikey/tools/ykman/PIV_Commands.html) are the command reference for the approved procedure.

## Certificate issuance and installation

The Apple Developer Program Account Holder creates the Developer ID Application certificate from the CSR. Apple lists the Account Holder as the required role for this certificate type. The resulting certificate is public material; import it into the same token slot as the matching on-token key, then confirm macOS can see a complete Developer ID identity through CryptoTokenKit. Do not use a certificate whose private key was generated elsewhere.

Follow Apple’s current [Developer ID certificate procedure](https://developer.apple.com/help/account/certificates/create-developer-id-certificates/) for the portal steps. Before enabling a release workflow, test signing a non-release fixture on the actual runner and verify that `codesign` uses the hardware-token identity rather than a file-based key. The current workflow does not yet implement this path, so no CI command is prescribed here.

## Environment and runner hand-off

Create the protected Apple-signing environment before enabling signing. Its PIN secret belongs only there. Restrict the environment to release tags and its signing workflow; do not make the secret available to general `main` builds or nightlies. The runner may host the token, but it must not retain the PIN after a job finishes.

Record outside the repository: token serial, firmware version, key role, certificate SHA-1 identity, Apple Team ID, certificate expiry, the responsible Account Holder, and the environment’s owners. Keep the PUK and management-key recovery material in the organization’s approved secret store, separate from the runner.

## Renewal, revocation, and timestamps

Put the Developer ID certificate expiry on a calendar well before its five-year term ends. Renewal is a new on-token key, a host-generated CSR signed by that key, and certificate issuance, followed by a runner test and a controlled identity change; do not replace the existing identity blindly. Apple says a secure timestamp allows users to continue running an app signed while its Developer ID certificate was valid, but a revoked Developer ID certificate is different: Apple says affected apps can no longer be installed or launched. New releases require a current certificate.

The release signer must include Apple’s secure timestamp. Apple requires it for notarization, and it is what preserves the signing-time validity of releases after certificate expiry. See Apple’s [notarization requirements](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution),
[certificate-expiry guidance](https://developer.apple.com/support/developer-id/),
and [TN3161](https://developer.apple.com/documentation/technotes/tn3161-inside-code-signing-certificates).
