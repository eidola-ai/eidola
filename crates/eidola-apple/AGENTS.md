# Apple detached-signature contract

This crate is the cross-platform contract between the release signing side and every consumer that reconstructs or audits a signed macOS app. It owns the placement record, detachment, in-place application, and read-only signature inspection. It must remain pure Rust and must never invoke `codesign`, `signapple`, or any other macOS tool.

## Placement invariant

`eidola-placement.json` records the signed output's per-slice fat placement, Mach header flags, `__LINKEDIT` fields, `LC_CODE_SIGNATURE` fields, superblob hash, file length, and file hash. Application writes those recorded target values; it does not derive or normalize Apple's layout. Before allocating the output, reconstruction first proves the actual input is canonically packed through its file end, then validates the measured target rule in the same table order: the first slice follows the fat table at its power-of-two alignment, every later slice follows the preceding body at its alignment, and the last body ends the file. A target alignment may increase for one architecture, as Apple's x86_64 normalization does, but it may not exceed the maximum exponent physically established anywhere in the input fat table. Alignment shifts and arithmetic are checked. Thin output is one slice at offset zero. The record may not introduce arbitrary gaps or trailing bytes.

Every input slice must already carry `LC_CODE_SIGNATURE` at the recorded load-command position and signature data offset. Application rewrites that command and refuses an unsigned slice by bundle-relative path and architecture. It never inserts a load command.

Application validates the exact unsigned regular-file set and hashes, the detached archive root (one placement record plus exactly the recorded app tree), the exact detached regular-file set implied by the record, reconstructed Mach-Os, sealed-resource hashes, and every plain-file removal/write leaf type before modifying the bundle. Callers pass the archive root — the directory holding the placement record — and nothing outside the two supplied roots is read; a caller holding the recorded app directory resolves the root itself rather than have reconstruction walk up to it. A changed, missing, unexpected, or type-incompatible input must be refused by bundle-relative path, and a failed validation must leave the input untouched. Detach uses the same in-memory reconstruction seam to prove each unsigned Mach-O plus extracted blobs reproduces the signed target before it clears or writes its destination. Paths from the record are relative and normalized. Every existing component and leaf under the supplied bundle and detached roots is checked with symlink-aware metadata before access and again before mutation; symbolic links are refused rather than followed outside either root.

After content validation, application prepares the complete mutation set before its first content write: Mach-O and existing recorded-file leaves become owner-writable, creation and removal parents become owner-writable and searchable, missing signing directories are created, and directories scheduled for recursive removal are prepared throughout. A preparation failure may leave permission changes or empty signing directories, but must not leave partially reconstructed file contents.

Callers provide privately staged roots that are not concurrently modified. The symlink checks protect against static untrusted archive paths; they do not claim to defend against a same-privilege process racing validation and mutation.

The all-or-nothing contract covers validation, not the write phase: once content writes begin, a failing write leaves a partially reconstructed bundle. That is correct for the temporary-directory verifier flow, which discards the tree; a caller reconstructing a bundle it means to keep stages a private copy and promotes it only on success.

## Detached layout

The detached tree mirrors signapple's layout:

```text
eidola-placement.json
Eidola.app/Contents/MacOS/Eidola.x86_64sign
Eidola.app/Contents/MacOS/Eidola.arm64sign
Eidola.app/Contents/MacOS/llama-server.arm64sign
Eidola.app/Contents/_CodeSignature/CodeResources
Eidola.app/Contents/CodeResources
```

The final path is optional and carries a stapled notarization ticket. The bundle seal is optional in the format but expected for a distributable app. Every application implementation removes an omitted seal or ticket, so material from a reused input cannot survive the record; an omitted seal removes only `_CodeSignature/CodeResources`, preserving any other exact-bound input in that directory. Regeneration clears material from a previous detach so removed slices, executables, seals, and tickets cannot survive as stale input. A reused destination must be empty or contain exactly one parseable previous record and its named app tree; unrelated entries and destinations with missing or corrupt records are refused unchanged. Both the current material root and any different root named by an old placement record are resolved and refused if they overlap either source before cleanup.

## Inspection

Inspection structurally parses the main executable named by `Contents/Info.plist` and requires all of its slices to agree on identifier, Team ID, runtime flag, and entitlements. It does not authenticate the CMS signer or establish Apple trust. Release verification composes these parsed claims with the separately authenticated Eidola release attestation, whose hashes bind the unsigned input, detached material, and shipped output and whose expected identity is compared with the claims. The entitlements digest is SHA-256 over the entitlement blob payload, excluding its generic eight-byte magic/length header. Ticket presence means `Contents/CodeResources` exists.

## Regression gates

- The committed synthetic universal fixture must reconstruct byte-for-byte on every platform.
- The wrong-build and mutated-superblob cases must fail with the bundle-relative Mach-O path.
- Unexpected detached root entries, inner regular files, and symbolic links must fail before mutation.
- The unsigned x86_64 fixture must fail with that path and architecture.
- The recorded x86_64 `__LINKEDIT` `vmsize` is asserted directly; never replace it with inferred alignment arithmetic.
- External-sentinel tests cover Mach-O, superblob, seal, detach, and inspect symlink traversal. Expanding the filesystem surface requires extending that table.
- The recorded app directory is not a detached root: passing it must fail with the record missing under it.
- signapple remains a macOS CI-only independent check under the deliberately narrow cases where its known placement arithmetic matches Apple's output.
- `scripts/apple-roundtrip.sh` (`just apple-roundtrip`) grades this crate — through `release-tool apple detach|apply` — against a real universal `.app`, including a padded-entitlements sweep that resizes the replacing signature across a 16 KiB `__LINKEDIT` boundary. It is the only gate that runs the shipping detach and apply on a real bundle, so a change to either belongs in a run of it, not only in the synthetic fixture. Its grading instruments (`scripts/macho_facts.py`, `scripts/apple_linkedit_diff.py`) stay read-only parsers, deliberately not this crate: no Python on any path that writes bytes, and no implementation grading itself.
- `scripts/test-verify-apple.sh` (`just test-verify-apple`) gates the public verifier `scripts/verify-apple.sh` over both containers it reads and runs its whole body **twice, under two userlands**: the host's own tools, then GNU tar plus mawk. That second pass is the point — the verifier is POSIX shell aimed at people on GNU/Linux, and a developer's Mac hides the differences that matter there (bsdtar defers directory modes to the end of extraction where GNU tar applies them on creation, and BSD awk is laxer than mawk about `[[:cntrl:]]` and writes to `/dev/stderr`). A GNU host runs the second pass directly, with one symlink pinning `awk` to `mawk`; elsewhere `nix shell --inputs-from .` supplies the tools from this repo's flake lock. Without either it skips loudly, which is a developer-machine concession only: CI sets `EIDOLA_TEST_REQUIRE_GNU_USERLAND` so a skip there is a failure. Set `EIDOLA_TEST_USERLAND` to `host` or `gnu` to run a single pass.
