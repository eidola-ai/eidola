# Apple detached-signature round trip — measurements and verdict

Status: **measured**. Written 2026-08-12 on branch `apple-roundtrip` (cut from `origin/next` at `e4ee69f8`, which carries the bundle-shape change that put the sidecar at `Contents/MacOS/llama-server` — PR #293 — and `CFBundleIdentifier = ai.eidola.app.macos`).

This is the go/no-go gate for the detached-signature design: does a signature lifted off a `codesign`-signed universal bundle reattach, byte for byte, to the unsigned build? It answers the questions below on the real `nix build .#eidola-gui-macos-universal` artifact, measures both candidate paths to byte-exactness, and states a verdict. Everything here is measured on this repository's own artifacts and is self-contained.

**Measurement environment** — signature formats move, so these facts are dated:

| | |
|---|---|
| macOS | 26.5.2 (build 25F84) |
| Xcode Command Line Tools | 26.6.0.0.1781586589 (`codesign` from CLT) |
| Host | Apple Silicon, 16 KiB pages |
| nixpkgs | `b77b3de8775677f84492abe84635f87b0e153f0f` (nixos-25.11) |
| signapple | `achow101/signapple@3fab3bb5`, packaged as `packages.signapple` |
| Identity for the larger-signature case | `Apple Development: Michael Marcacci (8AMGFBK943)` — no Developer ID certificate exists yet |

Reproduce with `just apple-roundtrip [path/to/Eidola.app]` — `scripts/apple-roundtrip.sh`, plus `scripts/apple-detach.py`, `scripts/apple-place.py` and `scripts/macho_facts.py`, all committed in this repository (paths here are relative to the repository root).

> The measurement below is kept as written. The two Python mutation scripts it names have since been subsumed by the `eidola-apple` crate and deleted; the harness drives `release-tool apple detach|apply` instead. Read every mention of them as naming the crate, and see the addendum at the end for what changed.

---

## Verdict

**GO** — the round trip is byte-exact, on both slices and the sidecar, and survives `codesign --verify --deep --strict`. It is byte-exact by **two independent routes**, and the one the design selected is the one measured on the artifact *as built*.

Both routes are run end to end by `scripts/apple-roundtrip.sh` on the real 80 MB universal `.app`:

| Route | What `apply` receives | Result |
|---|---|---|
| **Placement-driven** (§4.2, **selected**) | the artifact **as built** — sigtool's signatures, sigtool's fat alignment, sigtool's `__LINKEDIT` sizing | **byte-exact** on both slices, the sidecar and the whole tree; `--verify --deep --strict` passes. Holds at two different replacing-signature sizes, including one that crosses the 16 KiB boundary of §3.3 |
| **Settling** (§4) | the artifact after one full `codesign` ad-hoc cycle | **byte-exact** on both slices, the sidecar and the whole tree; `--verify --deep --strict` passes. Measured with an ad-hoc signature and again with a real Apple identity |

What does *not* work is handing `apply` the as-built artifact and expecting it to derive `codesign`'s layout: signapple does exactly that and the diff is large and structural, not one byte (§2(b)). The placement record is what removes the deriving.

Three conditions. Two are corrections of mechanism; the third was a genuine choice, now decided *and* demonstrated.

1. **Two mechanisms the design assumed need correcting** (§1, §5.1): signapple cannot detach a signature it did not create (so `detach` is ours, not signapple's), and signapple's `apply` needs a **one-line fork** to stay byte-equal to `codesign` at signature sizes other than the one measured (§3.3). `eidola-apple::apply` — the implementation that actually ships — is unaffected: it must match `codesign`, and `codesign`'s output is deterministic.
2. **The artifact must reach `apply` in a state `apply` can hit exactly** — either settled in the build, or handled by a placement-record-driven `apply`. Both are measured byte-exact; the second is the one we do — see (3).
3. **Settling inside the derivation has an obstacle**: `codesign` is **not available in the Nix build sandbox** (measured, §4.1), and admitting it would make the recorded `narHash` a function of the host's macOS version — a reproducibility regression. The only closure-resident alternative, `rcodesign` 0.29.0, produces a third layout that matches neither. **Decided: the build does not settle.** The detached bundle publishes the per-slice structural facts and `apply` lands on the unsettled artifact from them — **measured, not projected** (§4.2) — which keeps the `narHash` a pure function of source at the price of part of the independent-checker property.

**Nothing here is a NO-GO.** Every divergence found is in signapple's arithmetic, is one named field or one named behaviour, is deterministic, and is reproduced by committed fixtures. The property the design rests on — that `apply(unsigned, detached)` equals the shipped artifact byte-for-byte — holds, on the unsettled artifact the pipeline will actually produce.

**One precondition the placement route carries, found by measuring it** (§4.2): placement *rewrites* load commands, it never inserts one, so every slice of the input must already carry an `LC_CODE_SIGNATURE` at the offset the record names. The Nix build satisfies this — `autoSignDarwinBinariesHook` ad-hoc signs every slice — and a slice that does not is refused by name rather than mis-placed. It is a condition on the input, not a limit on the design.

**One residue the settling does not remove:** at signature sizes where `round_up(filesize, 4096) != round_up(filesize, 16384)`, upstream signapple writes `__LINKEDIT` `vmsize` **`0x75000` where `codesign` writes `0x78000`, at file offset `0x47f0`, on the x86_64 slice only** (measured on the real artifact with padded entitlements; see §3.3). It does not appear at the ad-hoc size or at the Apple Development size — which is luck, not a guarantee, and exactly why it is written down here. It is a defect in the *independent checker*, not in the shipped `apply`, and the fix is one line. **No normalizer was written.**

---

## 1. Two facts about signapple that change the pipeline's shape

Both were found on first contact and neither was anticipated. They are stated first because the rest of the measurement is arranged around them.

### 1.1 signapple cannot detach a signature it did not create

There is no `signapple detach` subcommand. `--detach <dir>` is a **flag on `sign`**, and `sign` takes a mandatory PKCS#12 archive:

```text
sign_subparser.add_argument(
    "keypath",
    help="Path to the PKCS#12 archive containing the certificate and private key to sign with")
```

A PKCS#12 archive contains an exportable private key. The decided key custody is a **non-exportable key on a YubiKey PIV token**, which can never produce one. So the planned signing sequence — `codesign -s <cert>` against the CryptoTokenKit token, then "`signapple ... --detach`" — **cannot be executed as written.**

The fix is small and does not touch the decision: detaching is a *read* operation. A detached signature file is exactly the bytes `LC_CODE_SIGNATURE` points at, `[dataoff, dataoff + datasize)` within each slice — verified by inspection of `sign.py`, which serializes the blob and pads it to `sig_cmd.datasize` before writing. `scripts/apple-detach.py` lifts them out of a `codesign`-signed bundle into signapple's exact layout, and `signapple apply` consumes the result unchanged. The shipping `eidola-apple` crate carries `detach` alongside `apply` for the same reason.

**signapple therefore stays the independent `apply` implementation — the role the design wants it for — and is not the signer or the detacher.**

### 1.2 signapple does not sign or seal a second Mach-O in `Contents/MacOS/`

`_build_resources` (`sign.py:899-904`) skips the whole `Contents/MacOS` directory, and `_setup_code_signers` only ever constructs signers for the bundle's main executable. Bitcoin Core's bundle has exactly one binary in `Contents/MacOS`, so this was never exercised.

The bundle shape puts the sidecar at `Contents/MacOS/llama-server` (PR #293). Measured consequence of signing that bundle *with signapple*:

```text
codesign --verify --deep --strict:
  a sealed resource is missing or invalid
  file added: .../Contents/MacOS/llama-server
```

Real `codesign` seals it (its `rules2` marks `MacOS/` as nested code and it recurses). signapple does not, so its `CodeResources` omits the sidecar and the bundle fails strict verification.

This is **not** an argument against that layout, because signapple is not the signer. `codesign` signs; `signapple apply` reattaches. And `apply` handles the sidecar correctly — it globs the detached tree and builds a `CodeSigner` per sig file, so a second binary is just another entry. Measured: on a settled artifact the sidecar round-trips byte-exactly (§4).

---

## 2. The measurements

The artifact: `/nix/store/04gsmkyblbhn2b4djy1rh8dhw8zg1ggr-eidola-gui-macos-universal-1.0`, `narHash sha256-rYyjQOiLaiQNML7b14QECTbRtl3SOagqlkKzFgdGmeo=`. `Contents/MacOS/Eidola` is 80,403,904 bytes, x86_64 + arm64. `Contents/MacOS/llama-server` is 13,459,696 bytes, **thin arm64** — deliberate, and measured as found.

Questions (a), (b) and (d) are below; question (c), the central one, gets §3 to itself, and (e) — the placement-driven route on the artifact as built — is §4.2.

### (a) Is whole-bundle ad-hoc signing byte-deterministic? — **yes**

Signed inside-out (`llama-server` first, then the bundle; never `--deep`), twice, from the same input:

| | |
|---|---|
| main binary, two independent signings | **identical** |
| sidecar, two independent signings | **identical** |
| `Contents/_CodeSignature/CodeResources` | **identical** |
| whole bundle tree (`diff -r`) | **identical** |

This is the load-bearing one: it means the shipped artifact is a deterministic function of the unsigned artifact plus the key, so an `apply` that reproduces it exists to be written.

*Aside on `--deep`, since the question invites it:* not used for any measurement here. It is deprecated, signs outside-in, and would have hidden the fact in §1.2 by re-signing the sidecar as a side effect. The measured order is the one the CI signing job must use: inside-out, the nested Mach-O first.

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

**On the settled artifact: yes, exactly** — see §4. And the as-built artifact is not out of reach either; it is out of reach *for an `apply` that derives the layout*. Driven by the placement record it is byte-exact as built — §4.2.

`apply` is at least **deterministic**: two runs from the same detached bundle produced identical output, main binary and sidecar.

Also measured, and worth knowing before `eidola-apple` trusts elfesteem: signapple prints `WARNING: Part of the file was not parsed: 256484 bytes` for the sidecar and `14719` / `14200` for the two main slices. It round-trips those bytes correctly regardless, but its Mach-O model is not complete.

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

A one-byte first-transition wrinkle was anticipated. The real artifact has a **bigger and differently-shaped** version of that, plus the one-byte wrinkle underneath it. Three separate facts, kept apart because they have different fixes.

### 3.1 The first transition out of sigtool's state moves a lot more than one byte

Per-slice `__LINKEDIT` `vmsize`, at the file offsets `scripts/macho_facts.py` reports:

| Stage | `Eidola` x86_64 | `Eidola` arm64 | `llama-server` arm64 |
|---|---|---|---|
| as built (`autoSignDarwinBinariesHook` / sigtool) | `0x6f000` @ `0x17f0` | `0x64000` @ `0x27ec7a0` | `0x35c000` @ `0x7f0` |
| after `codesign` cycle 1 | `0x74000` @ `0x47f0` | `0x30000` @ `0x27f47a0` | `0x34c000` @ `0x7f0` |
| after `codesign` cycle 2 | `0x74000` | `0x30000` | `0x34c000` |
| after sign then `--remove-signature` | `0x74000` | `0x30000` | `0x34c000` |

The sidecar's `0x35c000 → 0x34c000` is the small first-transition move that was anticipated. But the main binary's arm64 slice moves `0x64000 → 0x30000` — a **shrink of 212 KiB**, not a byte.

The cause is not alignment at all: **sigtool signs with 4 KiB code pages and `codesign` uses 16 KiB**, so `codesign`'s superblob has a quarter the hashes. The sidecar's signature goes 105,600 → 44,384 bytes; the main binary's arm64 slice 300,032 → 92,992. The artifact *loses* 61,216 bytes on signing. Every downstream surprise in §2(b) follows from a signature getting smaller, which is the case signapple does not handle.

### 3.2 One cycle is enough — signing is idempotent from there

Cycle 2 is byte-identical to cycle 1, on the main binary and on the sidecar. So "settled" is a real state and it is one cycle away, which is what makes §4 cheap. `--remove-signature` does **not** restore the as-built values (`0x74000`, not `0x6f000`), confirming that strip is not an exact inverse across the first transition — which matters only to the rejected alternative of verifying by stripping the shipped signature back off, rather than by re-applying a published one.

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

**Exact statement:** field `__LINKEDIT` `vmsize` (8 bytes, little endian) of the `LC_SEGMENT_64` load command, **file offset `0x47f0`**, **x86_64 slice only**, of `Contents/MacOS/Eidola` in the settled universal artifact. `codesign` writes `0x78000`; upstream signapple writes `0x75000`.

At that pad the x86_64 `__LINKEDIT` `filesize` is `0x74520`, and the rule above predicts both values from it exactly: `round_up(0x74520, 0x4000) = 0x78000` and `round_up(0x74520, 0x1000) = 0x75000`. The harness asserts that rather than allow-listing the field. `apple_linkedit_diff.py` admits the divergence only on an **x86_64** slice — signapple's `PAGE_SIZES` is keyed by cputype, so arm64 and arm64e round to 16 KiB like `codesign` and can never diverge — and only when the two values are exactly those two roundings. An arm64 `vmsize` that moves, or any other value in the permitted field, is graded `other` and fails. `apple-roundtrip.sh` likewise reads whether a swept signature size actually crosses a 16 KiB boundary off the signed artifact instead of inferring it from the verdict, and **fails** if no size in the sweep crosses one, so the latent case cannot go untested while the run still reports an exact round trip. At the size it does reach it grades the whole bundle, not just the main binary: the arm64-only sidecar — whose signature the padding also resizes, and which has no legitimate divergence at any size because both implementations round an arm64 `__LINKEDIT` to 16 KiB — every other file in the tree, and the applied bundle's own `codesign --verify --deep --strict`.

The fix is one line in the signapple fork (§5.2). Confirmed empirically on the real artifact at the size that triggers it: with the patched `apply`, the main binary, the sidecar and the whole tree are **identical**, and `codesign --verify --deep --strict` passes. **No normalizer was written**, and none is wanted: the shipped `apply` is ours and must simply do what `codesign` does.

---

## 4. The designed mitigation: settling inside the build

The designed fix, run exactly as designed — one full `codesign` ad-hoc cycle (sign → `--remove-signature` → sign), leaving no bundle-level seal. Run here on the host; §4.1 is about whether it can move into the derivation:

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
| ad-hoc + padded entitlements crossing a 16 KiB boundary | one field (§3.3) | identical | that one file only | fails on the x86_64 slice, from that field |
| …the same, with the patched signapple | **identical** | **identical** | **identical** | **passes** |

The last two rows are why §5.2's fork is a condition of the GO rather than a nicety.

### 4.1 The obstacle: `codesign` is not in the build

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

### 4.2 Two viable paths, and the decision

Both reach byte-exactness; they differ in what they cost and in how much independent checking survives. Both were measured, and both sets of measurements are kept here because they are the evidence for the choice.

**Path A — settle in the build.** Needs a closure-resident signer whose output is `codesign`-identical. None exists today; the candidates are impurely admitting `codesign` (reproducibility regression, above) or writing the ad-hoc signer into `eidola-apple` and calling it from the derivation. Moves the `narHash` once. Its payoff: on a settled artifact, signapple — with the one-line fix — reproduces the shipped bundle exactly, so the independent check the detached-signature design was chosen for stays intact.

**Path B — make `apply` handle the unsettled artifact.** No build change, no `narHash` move, no new host dependency. `eidola-apple::apply` does not have to *derive* `codesign`'s arithmetic: `detach` already walks the signed bundle, so `eidola-placement.json` carries the **target structural facts** — per slice, the fat offset/size/align, `__LINKEDIT`'s `fileoff`/`filesize`/`vmsize` with the file offset of each field, and the superblob's `dataoff`/`datasize`/hash — and `apply` writes what the record says and then checks `output_sha256`. That is robust to any future `codesign` behaviour change by construction, because the facts are published data rather than reimplemented policy. Its cost: signapple can no longer reproduce the shipped bundle at all (it would need the same record), so the differential test degrades to "signapple agrees on the arm64 slice and on settled inputs".

**Decided: Path B.** Nothing settles inside the derivation, so the recorded `narHash` stays a pure function of source — the property everything else in this repository is arranged around — and `apply` is driven by the placement record. The cost is taken knowingly: part of the "anyone can check us with an independent implementation" argument that motivated publishing a detached signature at all goes with it. Path A remains available later if that property is judged worth a closure-resident signer, and §4.3 is what it would cost.

Either way the GO stands. Both paths are measured, and neither requires inventing anything.

#### Path B, measured

`scripts/apple-place.py` is the placement-driven `apply`, written as a measurement so the selected path is demonstrated rather than projected. It is not the shipping implementation — `eidola-apple::apply` is — but it consumes the same record `apple-detach.py` emits and it works on the artifact **as built**, with nothing settled. Section (e) of `scripts/apple-roundtrip.sh` grades it.

What it writes, and nothing else: the fat header rebuilt at the recorded per-slice offset/size/align (cputype and cpusubtype come from the input, which signing never moves); `__LINKEDIT`'s `vmsize`/`fileoff`/`filesize` at the recorded field offsets; `LC_CODE_SIGNATURE`'s `dataoff`/`datasize`; the mach header's flags; and the superblob at `dataoff`. Everything below `dataoff` is copied from the input slice untouched. **Nothing is rounded**, so the 16 KiB-vs-4 KiB question of §3.3 never reaches this path — that is Path B's whole claim, and it is the reason the second row below exists.

Measured on the real universal artifact, as built, signed → detached → placed:

| Replacing signature | main binary | sidecar | whole tree | `--verify --deep --strict` |
|---|---|---|---|---|
| `codesign` ad-hoc | **identical** | **identical** | **identical** | **passes** |
| ad-hoc + padded entitlements + `--options runtime`, at a size where `round_up(filesize, 4096) != round_up(filesize, 16384)` on **both** slices | **identical** | **identical** | **identical** | **passes** |

The second row is the size at which upstream signapple gets `__LINKEDIT` `vmsize` wrong (§3.3). Placement is unaffected, as designed — it copies `0x78000` out of the record instead of computing it.

**The precondition, and it is a real one.** Placement rewrites load commands; it never inserts one. So each input slice must already carry an `LC_CODE_SIGNATURE`, at the same in-slice offset the record names, with the same `dataoff`. Three things were checked rather than assumed:

- the real artifact satisfies it — `autoSignDarwinBinariesHook` ad-hoc signs **every** slice during the Nix build, and signing moves neither the load-command layout nor `dataoff` (measured: `LC_CODE_SIGNATURE` at `+0x10e8` on x86_64 and `+0x1098` on arm64, `dataoff` `0x2799590` and `0x24789c0`, identical before and after `codesign` — only `datasize` moves, `325664 → 342688` and `300032 → 92992`);
- a slice that does **not** satisfy it is refused by name, not mis-placed. `synthetic-universal/unsettled.macho`'s x86_64 slice has never been signed at all — `lipo` output straight from `clang`, no `LC_CODE_SIGNATURE` on that slice — and `apple-place.py` stops with `input slice carries no LC_CODE_SIGNATURE; placement rewrites that load command, it does not insert one`;
- the committed synthetic fixture is a passing golden for the same code path: `place(settled/Fixture.app, detached/Fixture.app)` equals `signed/Fixture.app` byte for byte, on a two-slice universal bundle whose replacing signature is a different size than the one it replaces.

So `eidola-apple::apply` inherits a bounded job — write recorded integers into known offsets and append a blob — plus one precondition it must state and check. It does **not** inherit any of `codesign`'s policy.

### 4.3 What settling costs, measured

- **The `narHash` moves once** (Path A only). Must land before any Apple hash is attested.
- **The artifact gets ~61 KB smaller**: main binary 80,403,904 → 80,229,632, sidecar 13,459,696 → 13,398,480, because `codesign`'s 16 KiB code pages need a quarter of sigtool's hashes.
- **It does not touch the trust story.** The bytes whose hash is recorded are still produced by the build from source, with no key involved.

---

## 5. What the implementation must carry forward

### 5.1 Corrections to the planned mechanism that follow from §1

These are corrections to mechanism, not to any decision.

- **Signing order:** the planned sequence `codesign` → `notarytool` → `stapler` → "`signapple ... --detach`" must become `codesign` → `notarytool` → `stapler` → `release-tool apple detach` (ours). signapple cannot detach a signature it did not create, and it cannot use a non-exportable token key. Nothing about the key-custody decision changes.
- **`release-tool apple detach` cannot shell out to signapple.** `detach` is ours, in `eidola-apple`, alongside `apply` and `inspect`. It is the easier half — a read of `LC_CODE_SIGNATURE` per slice plus two file copies — and making it ours also makes `detach` runnable on Linux, which `codesign`-based detaching never would be.
- **Keeping signapple's on-disk layout verbatim, so `signapple apply` remains an *independent* implementation:** still the right goal, and the layout is unchanged. But see §5.2 — as of `3fab3bb5` signapple's `apply` is not byte-equal to `codesign` on a universal binary, so the differential test needs the fork.
- **`eidola-placement.json`** carries the **target structural facts** per slice — fat offset/size/align, `__LINKEDIT`'s `fileoff`/`filesize`/`vmsize` with the file offset of each field, and the superblob's `dataoff`/`datasize`/hash — not just the input and output hashes. That is what makes `apply` exact on the unsettled artifact without reimplementing `codesign`'s policy (§4.2). `scripts/apple-detach.py` emits it and `scripts/apple-place.py` consumes it, both measured; `eidola-apple` keeps the same record contents and gates the `LC_CODE_SIGNATURE` precondition with the committed unsigned-slice fixture.
- **The differential test against signapple** must run against a **bundle directory**. `signapple apply` cannot reach a fat Mach-O any other way (it refuses a single `.arch sign` against a universal binary, and derives the architecture from the *directory's* extension when the target is a bare file, which yields `KeyError: ''`).

### 5.2 The signapple fork, and its one carried commit

`.github/AGENTS.md` anticipates a fork ("if a patch is ever needed"). It is needed, and it is one line — `sign.py:706`:

```python
# upstream
linkedit_seg.vmsize = round_up(linkedit_seg.filesize, cs.page_size)   # 0x1000 on x86_64
# eidola
linkedit_seg.vmsize = round_up(linkedit_seg.filesize, 0x4000)         # what codesign does
```

`cs.page_size` is the **code-hash page size** (4 KiB on x86_64, 16 KiB on arm64) and is correct in its other two uses — the CodeDirectory's hash count and its `pageSize` field. It is the wrong quantity for a *segment* size. `codesign` on macOS 26 rounds `__LINKEDIT` vmsize to 16 KiB on every slice, including a **thin x86_64** binary (measured), so this is not a universal-binary special case; it is Apple aligning segments to the largest supported page.

Removal trigger for the carried commit: upstream accepting the same change, or Apple changing the granularity (at which point the fixtures go red first, which is the point of committing them).

**Without the fork, `eidola-apple::apply` is still correct** — it must match `codesign`, because `codesign`'s output is by definition the shipped artifact. The fork is what keeps the *independent check* honest. `eidola-apple` must not "fix" the disagreement by making our `apply` match signapple.

### 5.3 Test material for the implementation

Committed beside this document (see `README.md`):

- `synthetic-universal/` — a two-slice universal `.app`, `settled` + `signed` + `detached`, sized so the replacing signature differs from the settled one. That size change is what exposes the vmsize divergence; a same-size replacement hides it, which is exactly the trap §4 describes. `detached/eidola-placement.json` carries the structural facts, so this doubles as the golden for the placement-driven `apply` — `apple-place.py` already passes it.
- `unsettled.macho` inside it — the input placement **cannot** reach, because its x86_64 slice carries no `LC_CODE_SIGNATURE` to rewrite. Committed so the precondition of §4.2 has a test case and not just a sentence.
- `llama-server/` — the real sidecar's superblob plus placement facts, not the 13 MB binary.
- `facts.json` beside each, so the `__LINKEDIT` case is a one-field assertion and not a whole-file diff. Regenerating the Mach-Os without regenerating `facts.json` commits a fixture tree that disagrees with itself, so the README's recipe rebuilds it in the same block.

### 5.4 A trap found while committing the fixtures: new directories bust the macOS build cache

Not an Apple fact, but it bit this change and it will bite the next fixture tree, so it is recorded here.

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

- The planned `scripts/fixtures/manifest-determinism/` tree will do the same thing again.
- The durable fix is one filter rule — exclude `scripts/` (and any other non-source top-level tree) from `filteredSrc` outright. It is a small change with a wide blast radius (it moves the filtered source hash for *every* Nix build, so it costs one rebuild to adopt), and it is not this change's to make. Worth doing once, deliberately, rather than paying a cache miss per fixture directory.

### 5.5 Still unknown, and still needing a certificate

Open before this measurement and still open; nothing here settles them:

- Whether a **Developer ID Application** signature is deterministic without `--timestamp` (measured only with Apple Development).
- Whether the notarization ticket travels in the detached bundle and `apply` reproduces the **stapled** bundle. `apply` copies non-signature files verbatim, so `Contents/CodeResources` will land; that it is excluded from the seal is still documentation-derived, not observed.
- Whether Apple's notary service accepts the relocated sidecar in `Contents/MacOS/`.
- Whether `codesign` drives the PIV token unattended with the PIN supplied non-interactively.

---

## Addendum, 2026-08-14: the crate subsumed the measurement scripts

Everything above is the measurement as it was made, and it stays as written — it describes what was measured, with the instruments that existed then. What has changed since is which implementation the harness drives.

`scripts/apple-detach.py` and `scripts/apple-place.py` were scaffolding: written to demonstrate the placement route before `eidola-apple` existed, and never independent oracles, because ground truth throughout is `codesign`'s output bytes. The shipping crate now carries both operations, so `scripts/apple-roundtrip.sh` calls `release-tool apple detach` and `release-tool apple apply`, and the two scripts, their tests, and their duplicated hardening are deleted. Every reference to them above should be read as naming the crate's `detach` and `apply`.

Two consequences worth stating, because they are what the swap bought and what it cost:

- **The shipping implementation is now the one graded against real bundles.** Section (e)'s placement-driven round trip, the boundary sweep in (b++), and the detach feeding signapple in (b) all run the crate. Before this, the crate only ever touched the committed synthetic fixture.
- **The Python that remains reads and never writes.** `scripts/macho_facts.py` and `scripts/apple_linkedit_diff.py` are the instruments the divergences are graded with; making the crate its own classifier would collapse the measurement into the implementation. The rule is the boundary: no Python on any path that writes bytes.

The equivalence was checked before the swap, not assumed: `release-tool apple detach` over the fixture inputs emits a placement record semantically identical to the committed Python-emitted one, differing only in JSON key order (`serde` struct order rather than Python's sorted keys) — the same 2,177 bytes, field for field. `detached/eidola-placement.json` was regenerated once from the crate so the committed record is what the shipping detacher emits, and `README.md`'s recipe now says so.

The independent checks are untouched: `codesign --verify --deep --strict` and the narrow signapple differential (§5.2) both survive the retirement, because neither was ever Python of ours.

One interface change went with it. `eidola_apple::apply` previously accepted either the detached root or the app directory inside it, walking *up* to the parent to find `eidola-placement.json` — reading outside the path it was handed. It now requires the root; `scripts/verify-apple.sh` already located both trees and passes the root, and `scripts/apple-roundtrip.sh` does the same. `signapple apply`, which can only reach a fat Mach-O through the bundle path, still takes the app directory — that is its interface, not ours.
