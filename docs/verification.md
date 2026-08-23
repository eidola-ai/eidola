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
| **Installable** | What a normal user actually fetches and can run (Gatekeeper zip, `apt install ./eidola…deb`, `docker pull`). Archive composed with envelope. When the envelope is empty, installable = archive — **provided the archive is self-contained**, which the Debian package is and the Nix archive is not; see [the Linux installables](#the-two-linux-installables). |

**Artifact** is not a stage; it is the name of one of those tuples in the manifest.

Attestations and `artifact-manifest.json` live **beside** the installable on the GitHub release. They are not stuffed inside the payload (the cli must not COPY the manifest that records its own digest). The **trust root** compiled into the payload is a different object: pins for the next verification, not a hash of this artifact.

## What the manifest records

`artifact-manifest.json` is committed at the git ref, signed by CI, and what the engineer attests they reproduced. `schema_version` is `2`; clients also accept `3`, which adds the macOS unsigned shipping zip and the Debian packages as `file` rows and narrows the Nix Linux installable's key from `eidola-gui-linux-amd64` to `eidola-gui-linux-nix-amd64` (accept-before-emit — see [`releases/README.md`](../releases/README.md#rotating-document-schema-versions)).

| Artifact `type` | Identity of the payload | Identity of the archive | Envelope / installable |
| --- | --- | --- | --- |
| `oci` | Layers of the image | `digest` (`sha256:` + hex) — this *is* the archive | Empty. `docker pull` is the installable. |
| `nix` | `narHash` (Nix SRI of the store-path NAR) — rebuild/debug checkpoint | `archiveSha256` (`sha256:` + hex of the flake-built `.tar.gz`) | macOS: Apple material, hashes in the **human attestation** only. Current Linux Nix GUI: empty envelope. |
| `file` (schema 3) | — | `sha256` (`sha256:` + hex) of one published file | The `.deb`: empty envelope, so this file *is* the installable. The macOS unsigned shipping zip: the *container* an installable takes, before any envelope. |

Every value in this file is a function of source. Nothing key-dependent may enter it — no signed-artifact hash, no detached-bundle hash, no Team ID, no ticket; those live in the human attestation, which is signed and non-deterministic already. That is enforced rather than remembered: `scripts/check-manifest-determinism.sh` (run by `just check` and by the `Rust checks` workflow) validates the document's envelope and holds each artifact type to an exact field list (so an unrecognized field is rejected whether or not it names a key), rejects any key naming signing material, and asserts over `.github/workflows/artifacts.yml` that no ancestor of the job assembling the manifest is a signing job and that no job computing part of it holds the signing environment.

`narHash` is kept because it is free and isolates "the packer changed" from "the payload changed." It is not the user-facing check. Two serializations of the same tree: if they disagree, the archive derivation is impure. That split is load-bearing in one routine case: the archive is gzip-compressed, so the pinned **gzip version** is an input to `archiveSha256`. A `flake.lock` bump can move the archive hash with a byte-identical payload — `narHash` holding steady while `archiveSha256` moves is that, not tampering.

### Copied, not referenced

Anything **copied into** the payload is in both hashes. A store-path *reference* is not.

This is not a stylistic preference. Nix store paths are **input-addressed**: the path is a function of the derivation's inputs, not of its output bytes. A payload that names a binary by store path therefore binds the *build recipe* but not the *result* — a non-reproducible build, or a compromised builder, yields the same path and so the same `narHash` and `archiveSha256`, while the user's machine resolves whatever bytes its substituter holds for that path. Copying moves the bytes into the measured tree, where a substitution changes the hash.

So every executable that ships is copied: the macOS `.app` copies `llama-server` into `Contents/MacOS/`, and the Linux Nix GUI copies **both** the GUI binary (to `bin/.eidola-gui-wrapped`, kept a sibling of the sidecar) and `llama-server` into `$out/bin`. The only deliberate references left are the ones we have decided *not* to measure — the nixpkgs Mesa ICDs reached via `VK_ADD_DRIVER_FILES`, because the device's GPU stack is pre-trusted.

## Checking a download

Start from a **verified** release (manifest signature + human attestations, [releases.md](releases.md)). Copying a hash off an unverified page does not establish trust.

**OCI.** `crane digest` / registry digest against `digest`.

**The measured files are attached to the release.** From the next release tag on, the release workflow fetches the exact files the artifact workflow built and measured, re-hashes each one against the signed manifest, and refuses to attach anything that does not match — the identity across those two workflows is checked, not assumed. Each asset is named after the manifest key that records it, so there is no guessing which row covers which download, and a recorded artifact with no published file fails the release too. One recorded file is deliberately *not* attached: the unsigned macOS shipping zip, because the macOS installable is the Developer ID-signed one.

**Hash a download.** `archiveSha256` and a `file` row's `sha256` are designed to be the check a user can run with nothing but `sha256sum`:

```bash
sha256sum eidola-gui-linux-deb-amd64.deb        # `file` row: sha256
sha256sum eidola-gui-linux-nix-amd64.tar.gz     # `nix` row: archiveSha256
# macOS: shasum -a 256 eidola-gui-macos-universal.tar.gz
```

(Before manifest schema 3 the Nix Linux row and its asset are named `eidola-gui-linux-amd64`, and no `.deb` is recorded.)

**Rebuild — the stronger check.** Build the artifact and its archive from the release's git ref and compare both hashes to the verified manifest:

```bash
nix build .#eidola-gui-linux-nix          # compare narHash
nix build .#eidola-gui-linux-nix-archive  # compare archiveSha256
nix build .#eidola-linux-deb              # compare the .deb's sha256
```

The macOS universal attrs (`.#eidola-cli-macos-universal`, `.#eidola-gui-macos-universal`, and their `-archive` variants) work the same way, on a Mac.

**macOS installable (Developer ID zip), once published.** The browser file is not the archive. `codesign` rewrites Mach-Os; the unmodified archive is not a subset of the zip. The first-line check is `shasum` of that zip against the **attested shipped-installable** hash. Binding it to the git ref is one of:

1. **Forward (the designed path).** `apply(archive, envelope) = installable`, then hash the archive against the manifest. `just verify-apple` is this direction: it takes the flake's gzip'd POSIX tar (`nix build .#eidola-gui-macos-universal-archive`) — the exact bytes `archiveSha256` covers — plus the detached signature material, and reconstructs the installable in a temporary directory.
2. **Reverse.** Invert the Mach-O edits and drop the envelope until the unmodified archive reappears, then hash it. That is not "skip `_CodeSignature`." Load commands and `__LINKEDIT` already changed.

"Unsigned" means not Developer ID-signed. Nix ad-hoc signatures on the payload are part of the archive, not of the envelope.

**The container is part of the check.** The forward direction reconstructs a *tree*, but the file a browser downloaded is a zip — so comparing them requires re-zipping, and a zip of the same tree is only the same file if the recipe is. That recipe is published as a script, not left to a release runbook: `just pack-shipping-zip <tree> <out.zip>` packs any directory — including the one the forward check reconstructs — and the flake's `.#eidola-gui-macos-universal-zip` runs that same script over the Nix-built `.app`, so there is one recipe rather than two that resemble each other. It packs with Info-ZIP (never `ditto`, which is macOS-only and stamps wall-clock time), symlinks stored as symlinks, mtimes pinned to `SOURCE_DATE_EPOCH`, and entries ordered by a sorted `find`. Two runs over the same payload produce the same file, modes are normalized so the packer's umask cannot reach the hash, and nothing in it is macOS-only — POSIX shell and Info-ZIP, the same contract `verify-apple` states — so a verifier who re-zips a reconstructed tree on Linux lands on the same bytes, which is what makes the last step of the comparison meaningful. CI builds it on every macOS run and publishes it as a workflow artifact; from manifest schema 3 its `sha256` is recorded as `eidola-gui-macos-universal-zip`.

Apple-specific disclosure, ticket stapling, and Team ID are in [apple-distribution.md](apple-distribution.md).

## The two Linux installables

Linux has two artifacts from one build, and they are installables in different senses.

**The Debian package** (`eidola-gui-linux-deb-amd64`, `…-arm64`) is the download to hand a Ubuntu or Debian user: `apt install ./eidola-gui-linux-deb-amd64.deb`, which resolves the declared dependencies from the user's own distro repositories. The `.deb` is byte-reproducible, so it needs no archive/envelope indirection — the file a browser downloads *is* the byte stream the manifest hashes, and `sha256sum` on it is the whole check. There is no repository of ours anywhere in that path; updates continue to arrive only through the client's own attestation-verified flow.

Its payload is the same release binary and the same `llama-server` sidecar, with the two pieces of post-link metadata that bind an ELF to the Nix store removed, so the host's own glibc, Wayland, Vulkan loader, and Mesa or proprietary NVIDIA drivers are what load. **Those are outside the payload for the same reason macOS's are: the device's own operating system is pre-trusted.** No new builder or distributor of our bytes appears — the distro supplies the userland it already supplies to everything else on that machine.

**The Nix archive** (`eidola-gui-linux-nix-amd64`) is *not* a standalone installable. Its archive is a serialization of a Nix store path: the payload contains the real GUI binary and the real sidecar, so the hash means what it should, but the tree is Nix-shaped — `bin/eidola-gui` is a wrapper script with a `/nix/store/…` interpreter, and the GUI binary's `RUNPATH` resolves Wayland, libxkbcommon and the Vulkan loader out of the store. Extracted onto a machine with no Nix store, it will not start.

That is the correct artifact for NixOS, for `nix profile install`, and for anyone reproducing the build, and this page does not pretend otherwise: for that artifact, `archiveSha256` is a verification identity and a Nix-ecosystem installable, not a browser download that runs.

## Linux GPU stack

The Nix Linux installable *references* nixpkgs Mesa so a Nix glibc binary can load Vulkan ICDs on a non-Nix host. Those driver bytes are not in the archive. The Debian package needs no such reference and carries no GPU bytes at all: its binary runs on the host's glibc, so the host's Vulkan loader and its Mesa or proprietary NVIDIA drivers load into the ABI they were built for.

Server/CLI Linux images stay OCI: they run in an untrusted environment, so the image is the whole functional closure we can measure.
