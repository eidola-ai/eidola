# Apple round-trip fixtures

Committed inputs for the golden `apply` test that the planned `crates/eidola-apple` will code against. They exist so that test runs in plain `cargo test` on **any** platform: reproducing them needs macOS and `codesign`, but consuming them needs neither. If `apply` ever starts requiring a macOS tool, the Linux `rust-checks` job goes red — which is the property the detached-signature design was chosen for.

`round-trip.md` beside them is the measurement they encode — its result and verdict, and the document the harness and the classifier cite by section.

Measured and generated on macOS 26.5.2 (25F84), Xcode CLT 26.6.0.0.1781586589. Signature layout is a moving target across macOS releases; regenerate and re-measure rather than hand-edit, and record the new version here.

## `synthetic-universal/`

A two-slice (x86_64 + arm64) universal Mach-O, small enough to commit whole, carrying the one structural case the real artifact also exhibits: `codesign` rounds `__LINKEDIT`'s `vmsize` up to **16 KiB on every slice**, while signapple rounds to the slice's own page size (4 KiB on x86_64). See `round-trip.md` beside this file for the measurement.

| Path | What it is |
|---|---|
| `unsettled.macho` | as `lipo` emits it: x86_64 slice aligned 2^12, `__LINKEDIT` vmsize unrounded, and **no `LC_CODE_SIGNATURE` at all on that slice**. Present so the "`__LINKEDIT` case" can be a one-field assertion against a pre-settling input — and so the one input a placement-driven `apply` cannot reach is committed too (see the last section) |
| `settled/Fixture.app/` | after one full `codesign` ad-hoc cycle: slice alignment normalized to 2^14, vmsize 16 KiB-rounded, no bundle-level seal. This is `apply`'s input |
| `signed/Fixture.app/` | the golden: `settled` re-signed ad-hoc with `--options runtime --entitlements`, so the replacing signature is a **different size** than the settled one. That size change is what exposes the vmsize divergence; a same-size replacement hides it |
| `detached/Fixture.app/` | the per-slice superblobs and the sealed `CodeResources`, in signapple's layout |
| `detached/eidola-placement.json` | the placement record — input/output hashes per Mach-O, plus the signed artifact's per-slice structural facts (fat offset/size/align, `__LINKEDIT` sizing and field offsets, superblob placement and hash), which are what let `apply` land on `codesign`'s layout without deriving it |
| `facts.json` | `scripts/macho_facts.py` output for all three Mach-Os, so a test can assert one named field rather than diff whole files |
| `Info.plist`, `ent.plist` | the bundle identity and the entitlements used to grow the signature; kept so the fixture can be regenerated |

The test is `apply(settled/Fixture.app, detached/Fixture.app) == signed/Fixture.app`, byte for byte. `scripts/apple-place.py` already satisfies it from the placement record alone, so the fixture is a golden the implementation can be held to rather than a hope.

It is a `.app` rather than a bare Mach-O for one reason: `signapple apply` can only reach a fat Mach-O through the bundle path (see the last section), so the differential half of the test needs a bundle. `unsettled.macho` is the one bare file, and it is only ever read, never applied to.

## Note on the one input placement cannot reach

Placing a recorded layout *rewrites* load commands; it never inserts one. So the input must already carry an `LC_CODE_SIGNATURE` per slice, at the same offset the record names — which is exactly what the real artifact has, because `autoSignDarwinBinariesHook` ad-hoc signs every slice during the Nix build. `unsettled.macho`'s x86_64 slice has never been signed at all, and `apple-place.py` refuses it by name rather than producing a wrong file:

```text
x86_64: input slice carries no LC_CODE_SIGNATURE; placement rewrites that
load command, it does not insert one
```

Keep that case in the tree: it is the boundary of the shipped design, and `eidola-apple` should either refuse it the same way or handle it deliberately.

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

python3 -c '
import json, sys
sys.path.insert(0, "../../..")
from macho_facts import facts
out = {}
for p in ["unsettled.macho",
          "settled/Fixture.app/Contents/MacOS/Fixture",
          "signed/Fixture.app/Contents/MacOS/Fixture"]:
    f = facts(p)
    del f["path"]
    out[p] = f
print(json.dumps(out, indent=2, sort_keys=True))' > facts.json
```

The last step is not optional. `facts.json` records hashes, sizes, offsets and signature metadata for **these exact three Mach-Os**, and the golden test reads it as the description of the binaries beside it. A compiler or `codesign` change moves the binaries, so regenerating without rebuilding `facts.json` commits a fixture tree that disagrees with itself — and the disagreement is silent, because the stale file still parses.

Re-running the recipe is otherwise safe on an existing tree: `apple-detach.py` clears the bundle it is about to write (and the one a previous `eidola-placement.json` names) before regenerating, so a slice, executable, seal or ticket the new input no longer has cannot survive as a stale `.archsign` for `signapple apply` to consume.

## `llama-server/`

The **real** bundled sidecar's superblob, not the binary: 13 MB of llama.cpp is not a fixture, and the signature is the part `apply` has to place. Paired with the placement facts that say where it goes, so a test can assert the write landed at the right offset without carrying the payload.

| File | What it is |
|---|---|
| `llama-server.arm64sign` | the superblob `codesign` produced for the sidecar inside the universal `.app` |
| `facts.json` | `scripts/macho_facts.py` for the sidecar before and after signing: `__LINKEDIT` sizing, the `LC_CODE_SIGNATURE` offsets, and the file hashes |

The sidecar is **arm64-only by decision** even inside the universal app (see `AGENTS.md`), so there is one slice here and that is correct, not an omission.

## Note on signapple as the differential implementation

`signapple apply` can only reach a *fat* Mach-O through the bundle path: given a bare universal binary and a single `.arch sign` file it refuses ("Cannot attach single architecture signature to universal binary"), and given a directory it derives the architecture from the directory's extension, which has none. So the differential check has to hand signapple a bundle directory — which is why the synthetic fixture is one. `eidola-apple`'s own `apply` is under no such constraint and should accept a bare Mach-O too; `unsettled.macho` is there for that.

## Note on the Nix build cache

`flake.nix`'s `filteredSrc` drops non-Rust *files* but keeps *directories* (crane's filter has to, in order to descend). So this directory tree perturbs the filtered source hash even though none of its files survive the filter, and adding it cost one full rebuild of the macOS artifacts. It does **not** move `artifact-manifest.json` — that records the output's `narHash`, and an empty source directory cannot reach the output. Measured in `round-trip.md` §5.4, which also names the durable fix.
