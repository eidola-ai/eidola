# Task 55 Wave 2 — the universal round-trip spike

Status: **measured**. Written 2026-08-12 on branch `apple-roundtrip` (cut from `origin/next` at `e4ee69f8`, which carries Wave 1's bundle shape: sidecar at `Contents/MacOS/llama-server`, `CFBundleIdentifier = ai.eidola.app.macos`).

This is the go/no-go gate the implementation spec's §3 Wave 2 defines. It answers the four questions on the real `nix build .#eidola-gui-macos-universal` artifact, tests the designed mitigation, and states a verdict.

Task 55's implementation spec and Wave 1's findings are planning documents kept outside the repository; citations to them below are provenance, not links. Nothing here depends on reading them: every fact is measured in this document, and the material Wave 3 needs is committed beside it.

**Measurement environment** — signature formats move, so these facts are dated:

| | |
|---|---|
| macOS | 26.5.2 (build 25F84) |
| Xcode Command Line Tools | 26.6.0.0.1781586589 (`codesign` from CLT) |
| Host | Apple Silicon, 16 KiB pages |
| nixpkgs | `b77b3de8775677f84492abe84635f87b0e153f0f` (nixos-25.11) |
| signapple | `achow101/signapple@3fab3bb5`, packaged as `packages.signapple` |
| Identity for the larger-signature case | `Apple Development: Michael Marcacci (8AMGFBK943)` — no Developer ID certificate exists yet |

Reproduce with `just apple-roundtrip [path/to/Eidola.app]` — `scripts/apple-roundtrip.sh`, plus `scripts/apple-detach.py` and `scripts/macho_facts.py`, all committed in this repository (paths here are relative to the repository root).

---

## Verdict

**GO** — the round trip is byte-exact, on both slices and the sidecar, and survives `codesign --verify --deep --strict`.

On the artifact **as built today**, `apply` does *not* reconstruct the signed bundle: the diff is large and structural, not one byte. Put it through one full `codesign` ad-hoc cycle first — the spec's designed mitigation — and the round trip becomes **byte-exact on both slices of the main binary, on the arm64-only sidecar, and across the whole bundle tree**, with `codesign --verify --deep --strict` passing. Measured end to end on the real 80 MB universal `.app`, with an ad-hoc signature and again with a real Apple identity.

Three conditions. Two are corrections of mechanism; the third is a genuine choice that belongs to Mike.

1. **Two spec mechanisms need correcting** (§1, §5.1): signapple cannot detach a signature it did not create (so `detach` is ours, not signapple's), and signapple's `apply` needs a **one-line fork** to stay byte-equal to `codesign` at signature sizes other than the one measured (§3.3). `eidola-apple::apply` — the implementation that actually ships — is unaffected: it must match `codesign`, and `codesign`'s output is deterministic.
2. **The artifact must reach `apply` in a state `apply` can hit exactly** — either settled in the build, or handled by a placement-record-driven `apply`. Both work; see (3).
3. **Settling inside the derivation has an obstacle the spec did not anticipate**: `codesign` is **not available in the Nix build sandbox** (measured, §4.1), and admitting it would make the recorded `narHash` a function of the host's macOS version — a reproducibility regression. The only closure-resident alternative, `rcodesign` 0.29.0, produces a third layout that matches neither. So the choice between *settling in the build* and *teaching `apply` to land on the unsettled artifact from published placement facts* is open, and it trades reproducibility purity against how much of the independent-checker property survives (§4.2). **That trade is Mike's call**, and it is the one thing in this wave that is.

**Nothing here is a NO-GO.** Every divergence found is in signapple's arithmetic, is one named field or one named behaviour, is deterministic, and is reproduced by committed fixtures. The property the design rests on — that `apply(unsigned, detached)` equals the shipped artifact byte-for-byte — holds.

**One residue the settling does not remove**, reported per the spec's stop-and-report rule: at signature sizes where `round_up(filesize, 4096) != round_up(filesize, 16384)`, upstream signapple writes `__LINKEDIT` `vmsize` **`0x75000` where `codesign` writes `0x78000`, at file offset `0x47f0`, on the x86_64 slice only** (measured on the real artifact with padded entitlements; see §3.3). It does not appear at the ad-hoc size or at the Apple Development size — which is luck, not a guarantee, and exactly why it is written down here. It is a defect in the *independent checker*, not in the shipped `apply`, and the fix is one line. **No normalizer was written.**

---

## 1. Two facts about signapple that change the pipeline's shape

Both were found on first contact and neither is in the research. They are stated first because the rest of the measurement is arranged around them.

### 1.1 signapple cannot detach a signature it did not create

There is no `signapple detach` subcommand. `--detach <dir>` is a **flag on `sign`**, and `sign` takes a mandatory PKCS#12 archive:

```text
sign_subparser.add_argument(
    "keypath",
    help="Path to the PKCS#12 archive containing the certificate and private key to sign with")
```

A PKCS#12 archive contains an exportable private key. The decided key custody (spec §1, §8) is a **non-exportable key on a YubiKey PIV token**, which can never produce one. So the spec's Wave 7 sequence — `codesign -s <cert>` against the CryptoTokenKit token, then "`signapple ... --detach`" — **cannot be executed as written.**

The fix is small and does not touch the decision: detaching is a *read* operation. A detached signature file is exactly the bytes `LC_CODE_SIGNATURE` points at, `[dataoff, dataoff + datasize)` within each slice — verified by inspection of `sign.py`, which serializes the blob and pads it to `sig_cmd.datasize` before writing. `scripts/apple-detach.py` lifts them out of a `codesign`-signed bundle into signapple's exact layout, and `signapple apply` consumes the result unchanged. Wave 3's `eidola-apple` should carry `detach` alongside `apply` for the same reason.

**signapple therefore stays the independent `apply` implementation — the role §2.2 wants it for — and is not the signer or the detacher.**

### 1.2 signapple does not sign or seal a second Mach-O in `Contents/MacOS/`

`_build_resources` (`sign.py:899-904`) skips the whole `Contents/MacOS` directory, and `_setup_code_signers` only ever constructs signers for the bundle's main executable. Bitcoin Core's bundle has exactly one binary in `Contents/MacOS`, so this was never exercised.

Wave 1 put the sidecar at `Contents/MacOS/llama-server`. Measured consequence of signing that bundle *with signapple*:

```text
codesign --verify --deep --strict:
  a sealed resource is missing or invalid
  file added: .../Contents/MacOS/llama-server
```

Real `codesign` seals it (its `rules2` marks `MacOS/` as nested code and it recurses). signapple does not, so its `CodeResources` omits the sidecar and the bundle fails strict verification.

This is **not** an argument against Wave 1's layout, because signapple is not the signer. `codesign` signs; `signapple apply` reattaches. And `apply` handles the sidecar correctly — it globs the detached tree and builds a `CodeSigner` per sig file, so a second binary is just another entry. Measured: on a settled artifact the sidecar round-trips byte-exactly (§4).

---

## 2. The measurements

The artifact: `/nix/store/04gsmkyblbhn2b4djy1rh8dhw8zg1ggr-eidola-gui-macos-universal-1.0`, `narHash sha256-rYyjQOiLaiQNML7b14QECTbRtl3SOagqlkKzFgdGmeo=`. `Contents/MacOS/Eidola` is 80,403,904 bytes, x86_64 + arm64. `Contents/MacOS/llama-server` is 13,459,696 bytes, **thin arm64** — deliberate, and measured as found.

Questions (a), (b) and (d) are below; question (c), the central one, gets §3 to itself.

### (a) Is whole-bundle ad-hoc signing byte-deterministic? — **yes**

Signed inside-out (`llama-server` first, then the bundle; never `--deep`), twice, from the same input:

| | |
|---|---|
| main binary, two independent signings | **identical** |
| sidecar, two independent signings | **identical** |
| `Contents/_CodeSignature/CodeResources` | **identical** |
| whole bundle tree (`diff -r`) | **identical** |

This is the load-bearing one: it means the shipped artifact is a deterministic function of the unsigned artifact plus the key, so an `apply` that reproduces it exists to be written.

*Aside on `--deep`, since the question invites it:* not used for any measurement here. It is deprecated, signs outside-in, and would have hidden the fact in §1.2 by re-signing the sidecar as a side effect. The measured order is the one Wave 7 records.

### (b) Is `detach` → `signapple apply` byte-identical to the signed bundle?

**On the artifact as built: no, and not narrowly.**

| | signed by `codesign` | after `signapple apply` |
|---|---|---|
| `Eidola` file size | 80,229,632 | 80,213,248 |
| x86_64 slice offset / align | `0x4000` / 2^14 | `0x1000` / 2^12 |
| arm64 `__LINKEDIT` filesize | 193,792 | 400,832 |
| `llama-server` | — | 7 bytes differ, first at `0x7f2` — the arm64 `vmsize` itself |

Three distinct causes, all in signapple:

1. **Fat-header alignment.** `codesign` normalizes the x86_64 slice's `align` from 2^12 to 2^14 and moves it from `0x1000` to `0x4000`. signapple preserves what it finds. The 16,384-byte file-size difference is exactly this.
2. **Shrinking signatures.** `sign.py:704` only adjusts `__LINKEDIT` `if end_diff > 0`. When the replacing signature is *smaller* than the one already there, signapple writes the new blob and the new file length but leaves `filesize`/`vmsize` describing the old, larger segment. Since the CodeDirectory hashes the load commands, that alone invalidates the seal. This is why the sidecar fails: sigtool's signature is 105,600 bytes and `codesign`'s is 44,384.
3. **vmsize rounding granularity** — §3.3, latent here, masked by (2).

**On the settled artifact: yes, exactly** — see §4.

`apply` is at least **deterministic**: two runs from the same detached bundle produced identical output, main binary and sidecar.

Also measured, and worth knowing before Wave 3 trusts elfesteem: signapple prints `WARNING: Part of the file was not parsed: 256484 bytes` for the sidecar and `14719` / `14200` for the two main slices. It round-trips those bytes correctly regardless, but its Mach-O model is not complete.

### (d) Does the result survive `codesign --verify --deep --strict`? — **yes, on the settled path**

| Bundle | Result |
|---|---|
| `codesign`-signed (ad-hoc), settled or not | **passes** |
| `signapple apply` output, artifact as built | fails: `invalid signature (code or signature have been modified) / In architecture: arm64` |
| `signapple apply` output, **settled** artifact | **passes** |
| `signapple apply` output, settled + real Apple Development identity + `--options runtime` | **passes** |

The failure is not a separate finding: a signature seals the load commands, so any of the §2(b) divergences is necessarily also a verification failure.

---

## 3. `__LINKEDIT` vmsize — the central question, answered

The research expected "a one-byte first-transition wrinkle". The real artifact has a **bigger and differently-shaped** version of that, plus the one-byte wrinkle underneath it. Three separate facts, kept apart because they have different fixes.

### 3.1 The first transition out of sigtool's state moves a lot more than one byte

Per-slice `__LINKEDIT` `vmsize`, at the file offsets `scripts/macho_facts.py` reports:

| Stage | `Eidola` x86_64 | `Eidola` arm64 | `llama-server` arm64 |
|---|---|---|---|
| as built (`autoSignDarwinBinariesHook` / sigtool) | `0x6f000` @ `0x17f0` | `0x64000` @ `0x27ec7a0` | `0x35c000` @ `0x7f0` |
| after `codesign` cycle 1 | `0x74000` @ `0x47f0` | `0x30000` @ `0x27f47a0` | `0x34c000` @ `0x7f0` |
| after `codesign` cycle 2 | `0x74000` | `0x30000` | `0x34c000` |
| after sign then `--remove-signature` | `0x74000` | `0x30000` | `0x34c000` |

The sidecar's `0x35c000 → 0x34c000` is exactly the one-byte move Wave 1 recorded on the standalone binary. But the main binary's arm64 slice moves `0x64000 → 0x30000` — a **shrink of 212 KiB**, not a byte.

The cause is not alignment at all: **sigtool signs with 4 KiB code pages and `codesign` uses 16 KiB**, so `codesign`'s superblob has a quarter the hashes. The sidecar's signature goes 105,600 → 44,384 bytes; the main binary's arm64 slice 300,032 → 92,992. The artifact *loses* 61,216 bytes on signing. Every downstream surprise in §2(b) follows from a signature getting smaller, which is the case signapple does not handle.

### 3.2 One cycle is enough — signing is idempotent from there

Cycle 2 is byte-identical to cycle 1, on the main binary and on the sidecar. So "settled" is a real state and it is one cycle away, which is what makes §4 cheap. `--remove-signature` does **not** restore the as-built values (`0x74000`, not `0x6f000`), confirming the research's finding that strip is not an exact inverse across the first transition — relevant only to Option A, which we are not doing.

### 3.3 The genuine residue: 16 KiB vs 4 KiB rounding, x86_64 slice, one field

`codesign` on macOS 26 rounds `__LINKEDIT` `vmsize` up to **16 KiB on every slice**. Verified it is not a universal-binary special case: a **thin x86_64** binary with `__LINKEDIT` filesize 18,512 gets `vmsize 0x8000` = `round_up(18512, 16384)`, not `0x5000`.

signapple rounds to the slice's **code-hash page size** (`sign.py:706`, `round_up(linkedit_seg.filesize, cs.page_size)`), which is 4 KiB on x86_64 and 16 KiB on arm64. So the two agree on arm64 always, and on x86_64 only when the 4 KiB-rounded value happens to already be 16 KiB-aligned.

Measured on the **real settled artifact**, sweeping padded entitlements (keyless — plain `codesign --force --sign -`):

| entitlements pad | x86_64 `__LINKEDIT` vmsize (`codesign`) | `apply` result |
|---|---|---|
| 0 | `0x74000` | identical |
| 800 | `0x74000` | identical |
| 1600 | `0x74000` | identical |
| **2400** | **`0x78000`** | **`vmsize 0x78000` vs `0x75000` at file offset `0x47f0`** |

And with a real Apple Development identity plus `--options runtime` (signature 176 bytes larger per slice): **identical**, because 471,264 happens to round to `0x74000` under both rules.

So the residue is **latent, not absent**. Whether it appears is a function of the signature's size modulo 16 KiB. A Developer ID signature will be a different size again, so this must not be left to chance.

**Exact statement, per the stop-and-report rule:** field `__LINKEDIT` `vmsize` (8 bytes, little endian) of the `LC_SEGMENT_64` load command, **file offset `0x47f0`**, **x86_64 slice only**, of `Contents/MacOS/Eidola` in the settled universal artifact. `codesign` writes `0x78000`; upstream signapple writes `0x75000`.

At that pad the x86_64 `__LINKEDIT` `filesize` is `0x74520`, and the rule above predicts both values from it exactly: `round_up(0x74520, 0x4000) = 0x78000` and `round_up(0x74520, 0x1000) = 0x75000`. The harness asserts that rather than allow-listing the field. `apple_linkedit_diff.py` admits the divergence only on an **x86_64** slice — signapple's `PAGE_SIZES` is keyed by cputype, so arm64 and arm64e round to 16 KiB like `codesign` and can never diverge — and only when the two values are exactly those two roundings. An arm64 `vmsize` that moves, or any other value in the permitted field, is graded `other` and fails. `apple-roundtrip.sh` likewise reads whether a swept signature size actually crosses a 16 KiB boundary off the signed artifact instead of inferring it from the verdict, and **fails** if no size in the sweep crosses one, so the latent case cannot go untested while the run still reports an exact round trip.

The fix is one line in the signapple fork (§5.2). Confirmed empirically on the real artifact at the size that triggers it: with the patched `apply`, the main binary, the sidecar and the whole tree are **identical**, and `codesign --verify --deep --strict` passes. **No normalizer was written**, per instruction — and none is wanted: the shipped `apply` is ours and must simply do what `codesign` does.

---

## 4. The designed mitigation: settling inside the build

The spec's designed fix, run exactly as specified — one full `codesign` ad-hoc cycle (sign → `--remove-signature` → sign), leaving no bundle-level seal. Run here on the host; §4.1 is about whether it can move into the derivation:

```sh
codesign --force --sign - "$APP/Contents/MacOS/llama-server"
codesign --force --sign - "$APP"
codesign --remove-signature "$APP/Contents/MacOS/Eidola" "$APP/Contents/MacOS/llama-server"
rm -rf "$APP/Contents/_CodeSignature"
codesign --force --sign - "$APP/Contents/MacOS/llama-server"
codesign --force --sign - "$APP"
rm -rf "$APP/Contents/_CodeSignature"
```

**It works, and it removes all three §2(b) causes at once**, because after it the artifact already has `codesign`'s slice alignment, `codesign`'s 16 KiB-derived signature sizes, and `codesign`'s `__LINKEDIT` sizing — so the shipping signature neither shrinks the segment nor re-rounds it.

Measured on the real universal artifact, settled, then signed → detached → applied:

| Replacing signature | main binary | sidecar | whole tree | `--verify --deep --strict` |
|---|---|---|---|---|
| `codesign` ad-hoc | **identical** | **identical** | **identical** | **passes** |
| `Apple Development` + `--options runtime` | **identical** | **identical** | **identical** | **passes** |
| ad-hoc + padded entitlements crossing a 16 KiB boundary | one field (§3.3) | identical | — | — |
| …the same, with the patched signapple | **identical** | **identical** | **identical** | **passes** |

The last two rows are why §5.2's fork is a condition of the GO rather than a nicety.

### 4.1 The obstacle the spec did not anticipate: `codesign` is not in the build

Settling *works*. Putting it **inside the Nix derivation** is not free, and the reason is measurable rather than stylistic:

```text
$ nix-build -E 'pkgs.runCommand "codesign-probe" {} "... /usr/bin/codesign ..."'
codesign ABSENT
```

`sandbox = true` on this machine, with `sandbox-paths = /System/Library/Frameworks /System/Library/PrivateFrameworks /bin/bash /bin/sh /private/tmp /private/var/tmp /usr/lib` — `/usr/bin` is not on it. `autoSignDarwinBinariesHook` gets away with signing because `sigtool` ships *in* the closure; `codesign` does not.

Adding `/usr/bin/codesign` to `sandbox-paths` would make the recorded `narHash` a function of the **host's macOS version**, since `codesign`'s output is exactly what moves between releases (the 16 KiB page-size choice in §3.3 is one such policy). For a build whose whole point is that the hash is a function of source, that is a real regression, not a purity quibble.

The obvious closure-resident substitute does not match. `rcodesign` 0.29.0 (nixpkgs `rcodesign`, the `apple-codesign` crate) ad-hoc signing the same artifact:

| | `codesign` | `rcodesign` 0.29.0 |
|---|---|---|
| `Eidola` size | 80,229,632 | 80,424,384 |
| x86_64 signature | 342,688 | 329,728 |
| x86_64 `__LINKEDIT` vmsize | `0x74000` | `0x70000` |
| arm64 slice offset | `0x27f4000` | `0x27f0000` |

It does normalize the fat alignment to 2^14 (unlike signapple), but it uses 4 KiB code pages, so its output is its own third layout.

### 4.2 Therefore: two viable paths, and the choice is not Wave 2's

Both reach byte-exactness; they differ in what they cost and in how much independent checking survives.

**Path A — settle in the build** (the spec's design). Needs a closure-resident signer whose output is `codesign`-identical. None exists today; the candidates are impurely admitting `codesign` (reproducibility regression, above) or writing the ad-hoc signer into `eidola-apple` and calling it from the derivation. Moves the `narHash` once. Its payoff: on a settled artifact, signapple — with the one-line fix — reproduces the shipped bundle exactly, so the independent check the whole Option-B design was chosen for stays intact.

**Path B — make `apply` handle the unsettled artifact.** No build change, no `narHash` move, no new host dependency. `eidola-apple::apply` does not have to *derive* `codesign`'s arithmetic: `detach` already walks the signed bundle, so `eidola-placement.json` can carry the **target structural facts** — per slice, the fat offset and align, and `__LINKEDIT`'s `fileoff`/`filesize`/`vmsize` — and `apply` writes what the record says and then checks `output_sha256`. That is robust to any future `codesign` behaviour change by construction, because the facts are published data rather than reimplemented policy. Its cost: signapple can no longer reproduce the shipped bundle at all (it would need the same record), so the differential test degrades to "signapple agrees on the arm64 slice and on settled inputs".

**Recommendation, for the record and not as a decision:** Path B, with Path A available later if the independent-checker property is judged worth a closure-resident signer. Path B is the one that keeps the reproducible build a pure function of source, which is the property everything else in this repo is arranged around. But this trades away part of the "anyone can check us with an independent implementation" argument that motivated Option B over Option A, so **it is Mike's call, not an implementation detail.**

Either way the GO stands: byte-exactness is demonstrated, and neither path requires inventing anything.

### 4.3 What settling costs, measured

- **The `narHash` moves once** (Path A only). Must land before any Apple hash is attested.
- **The artifact gets ~61 KB smaller**: main binary 80,403,904 → 80,229,632, sidecar 13,459,696 → 13,398,480, because `codesign`'s 16 KiB code pages need a quarter of sigtool's hashes.
- **It does not touch the trust story.** The bytes whose hash is recorded are still produced by the build from source, with no key involved.

---

## 5. What Wave 3 and Wave 7 must carry forward

### 5.1 Amendments to the spec that follow from §1

These are corrections to mechanism, not to any decision.

- **Spec §3 Wave 7, "signing order":** the sequence `codesign` → `notarytool` → `stapler` → "`signapple ... --detach`" must become `codesign` → `notarytool` → `stapler` → `release-tool apple detach` (ours). signapple cannot detach a signature it did not create, and it cannot use a non-exportable token key. Nothing about the key-custody decision changes.
- **Spec §3 Wave 3, `release-tool apple detach` "shells to signapple":** it cannot. `detach` is ours, in `eidola-apple`, alongside `apply` and `inspect`. It is the easier half — a read of `LC_CODE_SIGNATURE` per slice plus two file copies — and making it ours also makes `detach` runnable on Linux, which `codesign`-based detaching never would be.
- **Spec §2.2, "signapple's on-disk layout verbatim so `signapple apply` remains an *independent* implementation":** still the right goal, and the layout is unchanged. But see §5.2 — as of `3fab3bb5` signapple's `apply` is not byte-equal to `codesign` on a universal binary, so the differential test needs the fork.
- **Spec §2.2, `eidola-placement.json`:** should carry the **target structural facts** per slice — fat offset and align, and `__LINKEDIT`'s `fileoff`/`filesize`/`vmsize` — not just the input and output hashes. Under Path B (§4.2) that is what makes `apply` exact without reimplementing `codesign`'s policy; under Path A it is a cheap cross-check that turns a wrong output into a named field rather than a mismatched final hash. `scripts/apple-detach.py` already emits the record; extending it is a few lines.
- **Spec §3 Wave 3, "differential vs. signapple":** must run against a **bundle directory**. `signapple apply` cannot reach a fat Mach-O any other way (it refuses a single `.arch sign` against a universal binary, and derives the architecture from the *directory's* extension when the target is a bare file, which yields `KeyError: ''`).

### 5.2 The signapple fork, and its one carried commit

Spec §7 anticipates a fork ("if a patch is ever needed"). It is needed, and it is one line — `sign.py:706`:

```python
# upstream
linkedit_seg.vmsize = round_up(linkedit_seg.filesize, cs.page_size)   # 0x1000 on x86_64
# eidola
linkedit_seg.vmsize = round_up(linkedit_seg.filesize, 0x4000)         # what codesign does
```

`cs.page_size` is the **code-hash page size** (4 KiB on x86_64, 16 KiB on arm64) and is correct in its other two uses — the CodeDirectory's hash count and its `pageSize` field. It is the wrong quantity for a *segment* size. `codesign` on macOS 26 rounds `__LINKEDIT` vmsize to 16 KiB on every slice, including a **thin x86_64** binary (measured), so this is not a universal-binary special case; it is Apple aligning segments to the largest supported page.

Removal trigger for the carried commit: upstream accepting the same change, or Apple changing the granularity (at which point the fixtures go red first, which is the point of committing them).

**Without the fork, `eidola-apple::apply` is still correct** — it must match `codesign`, because `codesign`'s output is by definition the shipped artifact. The fork is what keeps the *independent check* honest. Wave 3 must not "fix" the disagreement by making our `apply` match signapple.

### 5.3 Wave 3 test material

Committed beside this document (see `README.md`):

- `synthetic-universal/` — a two-slice universal `.app`, `settled` + `signed` + `detached`, sized so the replacing signature differs from the settled one. That size change is what exposes the vmsize divergence; a same-size replacement hides it, which is exactly the trap §4 describes.
- `llama-server/` — the real sidecar's superblob plus placement facts, not the 13 MB binary.
- `facts.json` beside each, so the spec's "`__LINKEDIT` case as a one-field assertion" is a one-field assertion and not a whole-file diff.

### 5.4 A trap found while committing the fixtures: new directories bust the macOS build cache

Not an Apple fact, but it bit this wave and it will bite Wave 4, so it is recorded here.

`filteredSrc` in `flake.nix` falls through to `craneLib.filterCargoSources` for anything it has no rule for. That filter drops non-Rust *files* but keeps *directories* (it has to, in order to descend). So committing `scripts/fixtures/apple-roundtrip/**` — whose every file the filter discards — still changes the filtered source tree, because the empty directory skeleton survives.

Isolated by evaluating the same tree three ways:

| Tree | `eidola-gui-macos-universal` derivation |
|---|---|
| pristine `e4ee69f8` | `4165sbmgjgz33nkzadw99jrs010xnwmy` |
| + `scripts/*.py`, `scripts/*.sh` (files only) | `4165sbmgjgz33nkzadw99jrs010xnwmy` — unchanged |
| + `scripts/fixtures/**` (new directories) | `fpzi7fjab330mws5kklc7bfl52vc5azd` — **changed** |

Loose files are free; a new directory anywhere in the filtered area is not.

**It does not change `artifact-manifest.json` — measured, not assumed.** The changed derivation `fpzi7fjab330mws5kklc7bfl52vc5azd` was built to completion and its output hashed:

```text
narHash: sha256-rYyjQOiLaiQNML7b14QECTbRtl3SOagqlkKzFgdGmeo=
```

Byte-for-byte the same `narHash` as the pristine build from `4165sbmgjgz33nkzadw99jrs010xnwmy`. The manifest records the output's contents, and an empty source directory cannot reach the output.

So the whole cost is a **cache miss**: one full recompile of the macOS artifacts, once, with no trust-path consequence. Worth knowing rather than worth blocking on.

Two things follow:

- Spec §3 Wave 4 plans `scripts/fixtures/manifest-determinism/`, which will do the same thing again.
- The durable fix is one filter rule — exclude `scripts/` (and any other non-source top-level tree) from `filteredSrc` outright. It is a small change with a wide blast radius (it moves the filtered source hash for *every* Nix build, so it costs one rebuild to adopt), and it is not Wave 2's to make. Worth doing once, deliberately, rather than paying a cache miss per fixture directory.

### 5.5 Still unknown, and still needing a certificate

Unchanged from Wave 1's open facts; nothing here settles them:

- Whether a **Developer ID Application** signature is deterministic without `--timestamp` (measured only with Apple Development).
- Whether the notarization ticket travels in the detached bundle and `apply` reproduces the **stapled** bundle. `apply` copies non-signature files verbatim, so `Contents/CodeResources` will land; that it is excluded from the seal is still documentation-derived, not observed.
- Whether Apple's notary service accepts the relocated sidecar in `Contents/MacOS/`.
- Whether `codesign` drives the PIV token unattended with the PIN supplied non-interactively.
