# Apple round-trip fixtures

Committed inputs for the golden `apply` test that `crates/eidola-apple` (task 55 Wave 3) codes against. They exist so that test runs in plain `cargo test` on **any** platform: reproducing them needs macOS and `codesign`, but consuming them needs neither. If `apply` ever starts requiring a macOS tool, the Linux `rust-checks` job goes red — which is the property the detached-signature design was chosen for.

Measured and generated on macOS 26.5.2 (25F84), Xcode CLT 26.6.0.0.1781586589. Signature layout is a moving target across macOS releases; regenerate and re-measure rather than hand-edit, and record the new version here.

## `synthetic-universal/`

A two-slice (x86_64 + arm64) universal Mach-O, small enough to commit whole, carrying the one structural case the real artifact also exhibits: `codesign` rounds `__LINKEDIT`'s `vmsize` up to **16 KiB on every slice**, while signapple rounds to the slice's own page size (4 KiB on x86_64). See `../../../work/reference/55-apple-signing/round-trip.md` for the measurement.

| Path | What it is |
|---|---|
| `unsettled.macho` | as `lipo` emits it: x86_64 slice aligned 2^12, `__LINKEDIT` vmsize unrounded, no signature on that slice. Present so the "`__LINKEDIT` case" can be a one-field assertion against a pre-settling input |
| `settled/Fixture.app/` | after one full `codesign` ad-hoc cycle: slice alignment normalized to 2^14, vmsize 16 KiB-rounded, no bundle-level seal. This is `apply`'s input |
| `signed/Fixture.app/` | the golden: `settled` re-signed ad-hoc with `--options runtime --entitlements`, so the replacing signature is a **different size** than the settled one. That size change is what exposes the vmsize divergence; a same-size replacement hides it |
| `detached/Fixture.app/` | the per-slice superblobs and the sealed `CodeResources`, in signapple's layout |
| `detached/eidola-placement.json` | the placement record — input/output hashes per Mach-O |
| `facts.json` | `scripts/macho_facts.py` output for all three Mach-Os, so a test can assert one named field rather than diff whole files |
| `Info.plist`, `ent.plist` | the bundle identity and the entitlements used to grow the signature; kept so the fixture can be regenerated |

The test is `apply(settled/Fixture.app, detached/Fixture.app) == signed/Fixture.app`, byte for byte.

It is a `.app` rather than a bare Mach-O for one reason: `signapple apply` can only reach a fat Mach-O through the bundle path (see the last section), so the differential half of the test needs a bundle. `unsettled.macho` is the one bare file, and it is only ever read, never applied to.

Regenerate (macOS, no signing identity needed):

```sh
cd scripts/fixtures/apple-roundtrip/synthetic-universal
printf 'int main(void){return 0;}\n' > /tmp/t.c
clang -arch arm64 -o /tmp/a /tmp/t.c && clang -arch x86_64 -o /tmp/x /tmp/t.c
lipo -create /tmp/a /tmp/x -output unsettled.macho && chmod +w unsettled.macho

mkdir -p settled/Fixture.app/Contents/MacOS signed/Fixture.app/Contents/MacOS
cp Info.plist settled/Fixture.app/Contents/Info.plist
cp unsettled.macho settled/Fixture.app/Contents/MacOS/Fixture
codesign --force --sign - settled/Fixture.app
codesign --remove-signature settled/Fixture.app/Contents/MacOS/Fixture
rm -rf settled/Fixture.app/Contents/_CodeSignature
codesign --force --sign - settled/Fixture.app
rm -rf settled/Fixture.app/Contents/_CodeSignature

cp -R settled/Fixture.app/. signed/Fixture.app/
codesign --force --sign - --options runtime --entitlements ent.plist signed/Fixture.app
python3 ../../../apple-detach.py signed/Fixture.app detached settled/Fixture.app
```

`facts.json` is `scripts/macho_facts.py` over the three Mach-Os, keyed by path.

## `llama-server/`

The **real** bundled sidecar's superblob, not the binary: 13 MB of llama.cpp is not a fixture, and the signature is the part `apply` has to place. Paired with the placement facts that say where it goes, so a test can assert the write landed at the right offset without carrying the payload.

| File | What it is |
|---|---|
| `llama-server.arm64sign` | the superblob `codesign` produced for the sidecar inside the universal `.app` |
| `facts.json` | `scripts/macho_facts.py` for the sidecar before and after signing: `__LINKEDIT` sizing, the `LC_CODE_SIGNATURE` offsets, and the file hashes |

The sidecar is **arm64-only by decision** even inside the universal app (see `AGENTS.md`), so there is one slice here and that is correct, not an omission.

## Note on signapple as the differential implementation

`signapple apply` can only reach a *fat* Mach-O through the bundle path: given a bare universal binary and a single `.arch sign` file it refuses ("Cannot attach single architecture signature to universal binary"), and given a directory it derives the architecture from the directory's extension, which has none. So the differential check in Wave 3 has to hand signapple a bundle directory — which is why the synthetic fixture is one. `eidola-apple`'s own `apply` is under no such constraint and should accept a bare Mach-O too; `unsettled.macho` is there for that.

## Note on the Nix build cache

`flake.nix`'s `filteredSrc` drops non-Rust *files* but keeps *directories* (crane's filter has to, in order to descend). So this directory tree perturbs the filtered source hash even though none of its files survive the filter, and adding it cost one full rebuild of the macOS artifacts. It does **not** move `artifact-manifest.json` — that records the output's `narHash`, and an empty source directory cannot reach the output. Measured in `work/reference/55-apple-signing/round-trip.md` §5.4, which also names the durable fix.
