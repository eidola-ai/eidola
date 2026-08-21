# Verification

How a user, an engineer, or CI checks that a download is the software a given git ref claims to be. The [releases](releases.md) page is *why* a release is trustworthy (CI signature plus human attestation). This page is the vocabulary and the hashes.

## Glossary

A **release** publishes **artifacts**. Each artifact has a measured **payload**, serialized as an **archive**. Some platforms wrap that archive in an **envelope** to produce an **installable**.

| Term | Meaning |
| --- | --- |
| **Release** | The unit of trust: everything published for one git ref — every artifact, `artifact-manifest.json`, human attestations, and the paired server enclave. See [releases.md](releases.md). |
| **Artifact** | One named, measured output of a release. The keys in `artifact-manifest.json` (`eidola-gui-macos-universal`, `eidola-server`, …). An artifact is the tuple of payload + archive, and where applicable envelope + installable. |
| **Payload** | The functional file tree: binaries, the `llama-server` sidecar, plist/icons, the embedded trust root. Every executable that ships is **copied in**, never referenced. Not Developer ID signatures. Not Mesa on the Nix Linux GUI (host GPU stack). *Trust object* is the threat-model synonym. |
| **Archive** | A canonical byte-stream of the payload. For Nix artifacts, a gzip'd POSIX/pax tar produced inside the flake (`nix build .#…-archive`); its SHA-256 is `archiveSha256`. For OCI artifacts, the image; its digest is `digest`. This is the hash a user can `shasum` / `sha256sum` without Nix. |
| **Envelope** | Host-required, non-reproducible material applied to an archive. Today: Apple Developer ID, notarization, staple. Never recorded in the CI-signed manifest. |
| **Installable** | What a normal user actually fetches and can run (Gatekeeper zip, `docker pull`, a future Flatpak). Archive composed with envelope. When the envelope is empty, installable = archive — **provided the archive is self-contained**; see [the Linux caveat](#the-linux-nix-archive-is-not-a-standalone-installable). |

**Artifact** is not a stage; it is the name of one of those tuples in the manifest.

Attestations and `artifact-manifest.json` live **beside** the installable on the GitHub release. They are not stuffed inside the payload (the cli must not COPY the manifest that records its own digest). The **trust root** compiled into the payload is a different object: pins for the next verification, not a hash of this artifact.

## What the manifest records

`artifact-manifest.json` is committed at the git ref, signed by CI, and what the engineer attests they reproduced. `schema_version` is `2`.

| Artifact `type` | Identity of the payload | Identity of the archive | Envelope / installable |
| --- | --- | --- | --- |
| `oci` | Layers of the image | `digest` (`sha256:` + hex) — this *is* the archive | Empty. `docker pull` is the installable. |
| `nix` | `narHash` (Nix SRI of the store-path NAR) — rebuild/debug checkpoint | `archiveSha256` (`sha256:` + hex of the flake-built `.tar.gz`) | macOS: Apple material, hashes in the **human attestation** only. Current Linux Nix GUI: empty envelope. |

`narHash` is kept because it is free and isolates "the packer changed" from "the payload changed." It is not the user-facing check. Two serializations of the same tree: if they disagree, the archive derivation is impure. That split is load-bearing in one routine case: the archive is gzip-compressed, so the pinned **gzip version** is an input to `archiveSha256`. A `flake.lock` bump can move the archive hash with a byte-identical payload — `narHash` holding steady while `archiveSha256` moves is that, not tampering.

### Copied, not referenced

Anything **copied into** the payload is in both hashes. A store-path *reference* is not.

This is not a stylistic preference. Nix store paths are **input-addressed**: the path is a function of the derivation's inputs, not of its output bytes. A payload that names a binary by store path therefore binds the *build recipe* but not the *result* — a non-reproducible build, or a compromised builder, yields the same path and so the same `narHash` and `archiveSha256`, while the user's machine resolves whatever bytes its substituter holds for that path. Copying moves the bytes into the measured tree, where a substitution changes the hash.

So every executable that ships is copied: the macOS `.app` copies `llama-server` into `Contents/MacOS/`, and the Linux Nix GUI copies **both** the GUI binary (to `bin/.eidola-gui-wrapped`, kept a sibling of the sidecar) and `llama-server` into `$out/bin`. The only deliberate references left are the ones we have decided *not* to measure — the nixpkgs Mesa ICDs reached via `VK_ADD_DRIVER_FILES`, because the device's GPU stack is pre-trusted.

## Checking a download

Start from a **verified** release (manifest signature + human attestations, [releases.md](releases.md)). Copying a hash off an unverified page does not establish trust.

**OCI.** `crane digest` / registry digest against `digest`.

**Nix archive, empty envelope (Linux GUI, unsigned macOS tree).** `archiveSha256` is designed to be the check a user can run with nothing but `sha256sum`:

```bash
sha256sum eidola-gui-linux-amd64.tar.gz
# macOS: shasum -a 256 eidola-gui-macos-universal.tar.gz
```

**Not yet available.** The release pipeline builds each archive to record its `archiveSha256` and does not currently publish the file: the archives are produced by the artifact workflow, while release assets are attached by the tagged release workflow, and nothing carries them across. Until that is wired up, the manifest's `archiveSha256` is verifiable only by rebuilding — which is the next check, and the stronger one. Publishing the archives as release assets is tracked as part of the Linux packaging work.

**Rebuild — the check that works today.** Build the artifact and its archive from the release's git ref and compare both hashes to the verified manifest:

```bash
nix build .#eidola-gui-linux          # compare narHash
nix build .#eidola-gui-linux-archive  # compare archiveSha256
```

The macOS universal attrs (`.#eidola-cli-macos-universal`, `.#eidola-gui-macos-universal`, and their `-archive` variants) work the same way, on a Mac.

**macOS installable (Developer ID zip), once published.** The browser file is not the archive. `codesign` rewrites Mach-Os; the unmodified archive is not a subset of the zip. The first-line check is `shasum` of that zip against the **attested shipped-installable** hash. Binding it to the git ref is one of:

1. **Forward (the designed path).** `apply(archive, envelope) = installable`, then hash the archive against the manifest. `just verify-apple` is this direction: it takes the flake's gzip'd POSIX tar (`nix build .#eidola-gui-macos-universal-archive`) — the exact bytes `archiveSha256` covers — plus the detached signature material, and reconstructs the installable in a temporary directory.
2. **Reverse.** Invert the Mach-O edits and drop the envelope until the unmodified archive reappears, then hash it. That is not "skip `_CodeSignature`." Load commands and `__LINKEDIT` already changed.

"Unsigned" means not Developer ID-signed. Nix ad-hoc signatures on the payload are part of the archive, not of the envelope.

Apple-specific disclosure, ticket stapling, and Team ID are in [apple-distribution.md](apple-distribution.md).

## The Linux Nix archive is not a standalone installable

`eidola-gui-linux-amd64` is a **Nix artifact**, and its archive is a serialization of a Nix store path. The payload now contains the real GUI binary and the real sidecar, so the hash means what it should — but the tree is still Nix-shaped: `bin/eidola-gui` is a wrapper script with a `/nix/store/…` interpreter, and the GUI binary's `RUNPATH` resolves Wayland, libxkbcommon, fontconfig, freetype, and the Vulkan loader out of the store. Extracted onto a machine with no Nix store, it will not start.

That is the correct artifact for NixOS, for `nix profile install`, and for anyone reproducing the build. It is **not** the download to hand a Ubuntu or Fedora user, and this page does not pretend otherwise: for that artifact, `archiveSha256` is a verification identity and a Nix-ecosystem installable, not a browser download that runs.

Closing that gap is packaging work, not hashing work — a second, Freedesktop-runtime installable whose payload is self-contained. It is tracked separately and does not change any rule on this page.

## Linux GPU stack

The Nix Linux GUI *references* nixpkgs Mesa so a Nix glibc binary can load Vulkan ICDs on a non-Nix host. Those driver bytes are not in the archive. A Flatpak (Freedesktop GL runtime) is how a distro-shaped installable leaves Mesa out of *our* payload entirely — and, unlike bundled Mesa, gives proprietary-driver users a matched GPU stack instead of a glibc mismatch.

Server/CLI Linux images stay OCI: they run in an untrusted environment, so the image is the whole functional closure we can measure.
