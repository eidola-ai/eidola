# Verification

How a user, an engineer, or CI checks that a download is the software a given git ref claims to be. The [releases](releases.md) page is *why* a release is trustworthy (CI signature plus human attestation). This page is the vocabulary and the hashes.

## Glossary

A **release** publishes **artifacts**. Each artifact has a measured **payload**, serialized as an **archive**. Some platforms wrap that archive in an **envelope** to produce an **installable**.

| Term | Meaning |
| --- | --- |
| **Release** | The unit of trust: everything published for one git ref — every artifact, `artifact-manifest.json`, human attestations, and the paired server enclave. See [releases.md](releases.md). |
| **Artifact** | One named, measured output of a release. The keys in `artifact-manifest.json` (`eidola-gui-macos-universal`, `eidola-server`, …). An artifact is the tuple of payload + archive, and where applicable envelope + installable. |
| **Payload** | The functional file tree: binaries, the `llama-server` sidecar, plist/icons, the embedded trust root. Not Developer ID signatures. Not Mesa on the current Nix Linux GUI (host GPU stack). *Trust object* is the threat-model synonym. |
| **Archive** | A canonical byte-stream of the payload. For Nix artifacts, a gzip'd ustar produced inside the flake (`nix build .#…-archive`); its SHA-256 is `archiveSha256`. For OCI artifacts, the image; its digest is `digest`. This is the hash a user can `shasum` / `sha256sum` without Nix. |
| **Envelope** | Host-required, non-reproducible material applied to an archive. Today: Apple Developer ID, notarization, staple. Never recorded in the CI-signed manifest. |
| **Installable** | What a normal user actually fetches (Gatekeeper zip, `docker pull`, a future Flatpak). Archive composed with envelope. When the envelope is empty, installable = archive. |

**Artifact** is not a stage; it is the name of one of those tuples in the manifest.

Attestations and `artifact-manifest.json` live **beside** the installable on the GitHub release. They are not stuffed inside the payload (the cli must not COPY the manifest that records its own digest). The **trust root** compiled into the payload is a different object: pins for the next verification, not a hash of this artifact.

## What the manifest records

`artifact-manifest.json` is committed at the git ref, signed by CI, and what the engineer attests they reproduced. `schema_version` is `2`.

| Artifact `type` | Identity of the payload | Identity of the archive | Envelope / installable |
| --- | --- | --- | --- |
| `oci` | Layers of the image | `digest` (`sha256:` + hex) — this *is* the archive | Empty. `docker pull` is the installable. |
| `nix` | `narHash` (Nix SRI of the store-path NAR) — rebuild/debug checkpoint | `archiveSha256` (`sha256:` + hex of the flake-built `.tar.gz`) | macOS: Apple material, hashes in the **human attestation** only. Current Linux Nix GUI: empty envelope. |

`narHash` is kept because it is free and isolates "the packer changed" from "the payload changed." It is not the user-facing check. Two serializations of the same tree: if they disagree, the archive derivation is impure.

Anything **copied into** the payload is in both hashes. A store-path *reference* (today: Nix Mesa ICDs via `VK_ADD_DRIVER_FILES`) is not. Functional bytes that ship inside the installable must be in the payload; that is why the Linux Nix GUI **copies** `llama-server` next to the wrapper rather than pointing at another derivation.

## Checking a download

Start from a **verified** release (manifest signature + human attestations, [releases.md](releases.md)). Copying a hash off an unverified page does not establish trust.

**OCI.** `crane digest` / registry digest against `digest`.

**Nix archive, empty envelope (Linux GUI today, unsigned macOS tree).** Hash the published `.tar.gz`:

```bash
sha256sum eidola-gui-linux-amd64.tar.gz
# macOS: shasum -a 256 eidola-gui-macos-universal.tar.gz
```

Compare to that artifact’s `archiveSha256` in the verified manifest.

**Rebuild.** `nix build .#eidola-gui-linux` (or the macOS universal attr) and compare `narHash`. `nix build .#eidola-gui-linux-archive` and compare `archiveSha256`.

**macOS installable (Developer ID zip), once published.** The browser file is not the archive. `codesign` rewrites Mach-Os; the unmodified archive is not a subset of the zip. The first-line check is `shasum` of that zip against the **attested shipped-installable** hash. Binding it to the git ref is one of:

1. **Forward (the designed path).** `apply(archive, envelope) = installable`, then hash the archive against the manifest. `just verify-apple` is this direction.
2. **Reverse.** Invert the Mach-O edits and drop the envelope until the unmodified archive reappears, then hash it. That is not "skip `_CodeSignature`." Load commands and `__LINKEDIT` already changed.

"Unsigned" means not Developer ID-signed. Nix ad-hoc signatures on the payload are part of the archive, not of the envelope.

Apple-specific disclosure, ticket stapling, and Team ID are in [apple-distribution.md](apple-distribution.md).

## Linux GPU stack

The current Nix Linux GUI still *references* nixpkgs Mesa so a Nix glibc binary can load Vulkan ICDs on Ubuntu. Those driver bytes are not in the archive. A Flatpak (Freedesktop GL runtime) or a distro `.deb` (host `mesa-vulkan-drivers`) is how an Ubuntu-shaped installable leaves Mesa out of *our* payload entirely; that work is tracked separately.

Server/CLI Linux images stay OCI: they run in an untrusted environment, so the image is the whole functional closure we can measure.
