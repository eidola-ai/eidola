#!/usr/bin/env bash
# The Apple detached-signature round trip, as a re-runnable check.
#
# Answers, on a real `.#eidola-gui-macos-universal` bundle:
#   (a) is whole-bundle ad-hoc signing byte-deterministic?
#   (b) is signapple detach -> apply byte-identical to the signed bundle,
#       on both slices of the main binary and on the arm64-only sidecar?
#   (c) does the artifact's `__LINKEDIT` vmsize settle, or does the first
#       transition out of sigtool's linker-signed state move it?
#   (d) does the result survive `codesign --verify --deep --strict`?
#   (e) does a placement-driven apply reach the signed bundle from the
#       artifact *as built*, with nothing settled?
#
# Written result and verdict: scripts/fixtures/apple-roundtrip/round-trip.md.
#
# Usage:  scripts/apple-roundtrip.sh [path/to/Eidola.app]
# With no argument it runs `nix build .#eidola-gui-macos-universal` first,
# which is a 30-90 minute cold build.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACTS="$REPO_ROOT/scripts/macho_facts.py"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "apple-roundtrip: macOS only (needs codesign)." >&2
  exit 1
fi

# Detaching and placing are the shipping crate, driven through its CLI, so the
# implementation that ships is the one graded against real bundles here.
# `macho_facts.py` and `apple_linkedit_diff.py` stay Python: they only read and
# classify, and using the implementation as its own instrument would make the
# measurement circular.
apple_tool() { (cd "$REPO_ROOT" && cargo run -q -p release-tool -- apple "$@"); }
echo "==> cargo build -p release-tool"
(cd "$REPO_ROOT" && cargo build -q -p release-tool)

APP_SRC="${1:-}"
if [[ -z "$APP_SRC" ]]; then
  echo "==> nix build .#eidola-gui-macos-universal (this is slow)"
  nix build "$REPO_ROOT#eidola-gui-macos-universal" --no-link --print-out-paths \
    >"${TMPDIR:-/tmp}/apple-roundtrip-out"
  APP_SRC="$(cat "${TMPDIR:-/tmp}/apple-roundtrip-out")/Eidola.app"
fi
[[ -d "$APP_SRC" ]] || { echo "not a bundle: $APP_SRC" >&2; exit 1; }
APP_NAME="$(basename "$APP_SRC")"

SIGNAPPLE="$(nix build "$REPO_ROOT#signapple" --no-link --print-out-paths)/bin/signapple"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/apple-roundtrip.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
echo "==> work dir: $WORK"
echo "==> bundle:   $APP_SRC"
echo "==> macOS:    $(sw_vers -productVersion) ($(sw_vers -buildVersion)), CLT $(pkgutil --pkg-info=com.apple.pkg.CLTools_Executables | awk '/version:/{print $2}')"

MAIN="Contents/MacOS/Eidola"
SIDE="Contents/MacOS/llama-server"

FAILURES=0
check() { # check <name> <expected: same|differ> <a> <b>
  local name="$1" want="$2" a="$3" b="$4" got
  if cmp -s "$a" "$b"; then got=same; else got=differ; fi
  if [[ "$got" == "$want" ]]; then
    printf '    PASS  %-52s (%s)\n' "$name" "$got"
  else
    printf '    FAIL  %-52s (want %s, got %s)\n' "$name" "$want" "$got"
    FAILURES=$((FAILURES + 1))
  fi
}

# Whole-tree comparison that composes with classify() below. A file whose
# byte difference classify() already graded as the documented divergence is
# named here as permitted, so the same bytes are not counted a second time as
# an unrecognized regression — otherwise a bundle carrying only the known
# divergence could never report it, because the tree would fail for it.
# Everything else still fails: a differing file that was not classified, a
# file present on one side only, a permitted file that differs in the *other*
# bundle's copy. The verdict lands in TREE_KIND (same|known|differ) rather
# than a return code, so a `set -e` caller can read it without guarding.
TREE_KIND=""
tree_check() { # tree_check <label> <a> <b> [<permitted relative path>...]
  local label="$1" a="$2" b="$3"; shift 3
  local out line rel permitted=0 unexpected=""
  out="$(diff -rq "$a" "$b" 2>&1)" || true
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    for rel in "$@"; do
      if [[ "$line" == "Files $a/$rel and $b/$rel differ" ]]; then
        permitted=$((permitted + 1))
        continue 2
      fi
    done
    unexpected+="$line"$'\n'
  done <<<"$out"
  if [[ -n "$unexpected" ]]; then
    TREE_KIND=differ
    printf '    FAIL  %-52s (differ)\n' "$label"
    printf '%s' "$unexpected" | sed 's/^/          /'
    FAILURES=$((FAILURES + 1))
  elif [[ "$permitted" -gt 0 ]]; then
    TREE_KIND=known
    printf '    KNOWN %-52s (only the classified divergence)\n' "$label"
  else
    TREE_KIND=same
    printf '    PASS  %-52s (same)\n' "$label"
  fi
}

# Each stage gets its own directory holding a copy of the bundle under its
# real name: signapple only treats a directory as a bundle when it ends in
# `.app`, so the basename has to survive the copy.
stage() { # stage <name> -> path to $WORK/<name>/<bundle>.app
  local name="$1"
  rm -rf "${WORK:?}/$name"
  mkdir -p "$WORK/$name"
  cp -R "$APP_SRC" "$WORK/$name/$APP_NAME"
  chmod -R u+w "$WORK/$name/$APP_NAME"
  # Nix store copies can carry the provenance xattr; codesign would seal a
  # difference that has nothing to do with the signature.
  xattr -cr "$WORK/$name/$APP_NAME" 2>/dev/null || true
  echo "$WORK/$name/$APP_NAME"
}

# Sign inside-out, never --deep: the nested Mach-O first, then the bundle.
# This is the order the CI signing job must use, so it is the order measured
# here.
sign_adhoc() {
  local app="$1"
  codesign --force --sign - "$app/$SIDE"
  codesign --force --sign - "$app"
}

# One full codesign ad-hoc cycle, leaving no bundle-level seal. This is the
# mitigation §4 of round-trip.md measures: as built, the artifact carries
# sigtool's linker-signed signatures, whose 4 KiB code pages make them ~2.4x
# larger than codesign's, so the first codesign *shrinks* __LINKEDIT — the
# case signapple's apply does not handle.
#
# Run here, not in the derivation: codesign is absent from the Nix build
# sandbox (measured), and admitting it would tie the recorded narHash to the
# host's macOS version. The shipped pipeline therefore does not settle at all
# — `apply` lands on the unsettled artifact from the placement record's
# structural facts (round-trip.md §4.2). Settling is still measured, because
# it is what lets signapple, the independent checker, reproduce the bundle.
settle() {
  local app="$1"
  sign_adhoc "$app"
  codesign --remove-signature "$app/$MAIN" "$app/$SIDE"
  rm -rf "$app/Contents/_CodeSignature"
  sign_adhoc "$app"
  rm -rf "$app/Contents/_CodeSignature"
}

# Entitlements whose length is a knob: padding them grows every signature in
# the bundle, which is how a replacing signature of a *different* size than
# the one already there gets exercised without a signing identity.
ent_plist() { # ent_plist <pad bytes> -> writes $WORK/ent.plist
  python3 -c "
import sys
open(sys.argv[1], 'w').write(
  '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n'
  '<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" '
  '\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n'
  '<plist version=\"1.0\"><dict>\n'
  '<key>com.apple.security.cs.allow-jit</key><true/>\n'
  '<key>com.apple.security.application-groups</key><string>' + 'A' * int(sys.argv[2]) + '</string>\n'
  '</dict></plist>\n')" "$WORK/ent.plist" "$1"
}

vmsizes() { # vmsizes <mach-o> -> "arch=vmsize@field-offset" per slice
  python3 "$FACTS" "$1" | python3 -c '
import json, sys
out = []
for s in json.load(sys.stdin)["slices"]:
    le = s["linkedit"]
    out.append("%s=%#x@%#x" % (s["arch"], le["vmsize"], le["vmsize_field_offset"]))
print(" ".join(out))'
}

echo
echo "=== (a) ad-hoc bundle signing determinism ==="
S1="$(stage S1)"; sign_adhoc "$S1"
S2="$(stage S2)"; sign_adhoc "$S2"
check "main binary, two independent signings" same "$S1/$MAIN" "$S2/$MAIN"
check "sidecar, two independent signings"     same "$S1/$SIDE" "$S2/$SIDE"
check "CodeResources, two independent signings" same \
  "$S1/Contents/_CodeSignature/CodeResources" "$S2/Contents/_CodeSignature/CodeResources"
tree_check "whole bundle tree" "$S1" "$S2"

echo
echo "=== (c) __LINKEDIT vmsize across signing transitions ==="
U="$(stage U)"
printf '    %-34s main: %s\n' "as built (sigtool ad-hoc)" "$(vmsizes "$U/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$U/$SIDE")"
printf '    %-34s main: %s\n' "after codesign cycle 1" "$(vmsizes "$S1/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$S1/$SIDE")"
S1B="$(stage S1B)"; sign_adhoc "$S1B"; sign_adhoc "$S1B"
printf '    %-34s main: %s\n' "after codesign cycle 2" "$(vmsizes "$S1B/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$S1B/$SIDE")"
check "cycle 2 == cycle 1 (signing is settled)" same "$S1B/$MAIN" "$S1/$MAIN"
check "cycle 2 == cycle 1, sidecar"             same "$S1B/$SIDE" "$S1/$SIDE"
R="$(stage R)"; sign_adhoc "$R"
codesign --remove-signature "$R/$MAIN" "$R/$SIDE"
printf '    %-34s main: %s\n' "after sign then --remove-signature" "$(vmsizes "$R/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$R/$SIDE")"

echo
echo "=== (b) detach -> signapple apply ==="
# `signapple sign --detach` is not the detach side here: it takes a PKCS#12,
# which a non-exportable Developer ID key on a hardware token can never
# supply, and it does not sign or seal a second Mach-O in Contents/MacOS.
# So `codesign` signs (S1, above) and `release-tool apple detach` lifts the
# superblobs into signapple's layout; signapple's `apply` stays the
# independent implementation. This is exactly the equation the design
# rests on: apply(unsigned, detached) == shipped.
apple_tool detach "$S1" "$U" "$WORK/detached" >/dev/null
A="$(stage A)"
"$SIGNAPPLE" apply --no-verify "$A" "$WORK/detached/$APP_NAME" >/dev/null

# Upstream signapple diverges from codesign on exactly one field of the
# x86_64 slice (see apple_linkedit_diff.py). Report that state by name
# rather than as a bare "differ": a diff that has grown beyond it is the
# regression worth failing. Graded only on the settled artifact below: on
# the as-built one the divergence is expected and is itself the finding, so
# it is reported without failing the run.
#
# The verdict lands in CLASSIFY_KIND for the caller to fold into *its own*
# artifact's exactness. Byte-exactness is a property of one bundle, not of
# the run, and (d) below gates a real verification failure on it — so a
# single shared flag would let one section's expected divergence excuse
# another section's genuine break.
CLASSIFY_KIND=""
KNOWN_DIVERGENCE=0
classify() { # classify <graded|info> <label> <signed> <applied>; sets CLASSIFY_KIND
  local grade="$1" label="$2" out kind
  out="$(python3 "$REPO_ROOT/scripts/apple_linkedit_diff.py" "$3" "$4" 2>&1)" || true
  kind="$(printf '%s\n' "$out" | head -1)"
  CLASSIFY_KIND="$kind"
  case "$kind" in
    identical)
      printf '    PASS  %-52s (identical)\n' "$label" ;;
    linkedit-vmsize-only)
      KNOWN_DIVERGENCE=1
      printf '    KNOWN %-52s (documented signapple divergence)\n' "$label"
      printf '%s\n' "$out" | tail -n +2 | sed 's/^/        /' ;;
    *)
      if [[ "$grade" == graded ]]; then
        printf '    FAIL  %-52s\n' "$label"; FAILURES=$((FAILURES + 1))
      else
        printf '    NOTE  %-52s (expected before settling)\n' "$label"
      fi
      printf '%s\n' "$out" | sed 's/^/          /' ;;
  esac
}
diff_kind() { # diff_kind <signed> <applied> -> identical|linkedit-vmsize-only|other
  python3 "$REPO_ROOT/scripts/apple_linkedit_diff.py" "$1" "$2" 2>&1 | head -1 || true
}
classify info "main binary (both slices), apply == signed" "$S1/$MAIN" "$A/$MAIN"
classify info "sidecar (arm64-only), apply == signed"      "$S1/$SIDE" "$A/$SIDE"
check "_CodeSignature/CodeResources"               same \
  "$S1/Contents/_CodeSignature/CodeResources" "$A/Contents/_CodeSignature/CodeResources"
printf '    %-34s main: %s\n' "after apply" "$(vmsizes "$A/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$A/$SIDE")"

A2="$(stage A2)"
"$SIGNAPPLE" apply --no-verify "$A2" "$WORK/detached/$APP_NAME" >/dev/null
check "apply is deterministic (main)"    same "$A/$MAIN" "$A2/$MAIN"
check "apply is deterministic (sidecar)" same "$A/$SIDE" "$A2/$SIDE"

echo
echo "=== (b+) the same round trip on a settled artifact ==="
# The mitigation, end to end: settle, then sign/detach/apply as above. This
# is the configuration in which signapple reproduces the bundle exactly.
SB="$(stage SB)"; settle "$SB"
printf '    %-34s main: %s\n' "settled" "$(vmsizes "$SB/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$SB/$SIDE")"
SS="$WORK/SS/$APP_NAME"; rm -rf "$WORK/SS"; mkdir -p "$WORK/SS"; cp -R "$SB" "$SS"
SA_="$WORK/SA/$APP_NAME"; rm -rf "$WORK/SA"; mkdir -p "$WORK/SA"; cp -R "$SB" "$SA_"
chmod -R u+w "$SS" "$SA_"
sign_adhoc "$SS"
apple_tool detach "$SS" "$SB" "$WORK/detached-settled" >/dev/null
"$SIGNAPPLE" apply --no-verify "$SA_" "$WORK/detached-settled/$APP_NAME" >/dev/null
# Exactness of one applied bundle — this is what (d) may excuse a verification
# failure with, and it is a property of that bundle alone. Two round trips are
# graded this way (the settled one here, the boundary case in (b++)), so
# neither can excuse the other. GRADE_KNOWN carries the classification forward
# to the tree comparison: only a file the classifier named as the documented
# divergence may differ there.
GRADE_EXACT=1
GRADE_KNOWN=()
grade() { # grade <signed> <applied> <relative path> <label>
  classify graded "$4" "$1/$3" "$2/$3"
  case "$CLASSIFY_KIND" in
    identical) return 0 ;;
    linkedit-vmsize-only) GRADE_KNOWN+=("$3") ;;
  esac
  GRADE_EXACT=0
}
# Both executables and then everything else, so a round trip that mishandles
# only the sidecar cannot pass on the strength of an exact main binary.
grade_bundle() { # grade_bundle <label prefix> <signed> <applied>; sets GRADE_EXACT
  GRADE_EXACT=1
  GRADE_KNOWN=()
  grade "$2" "$3" "$MAIN" "$1main binary (both slices), apply == signed"
  grade "$2" "$3" "$SIDE" "$1sidecar (arm64-only), apply == signed"
  tree_check "$1whole bundle tree" "$2" "$3" ${GRADE_KNOWN[@]+"${GRADE_KNOWN[@]}"}
  [[ "$TREE_KIND" == same ]] || GRADE_EXACT=0
}
grade_bundle "" "$SS" "$SA_"
SETTLED_EXACT="$GRADE_EXACT"

echo
echo "=== (b++) the latent case: a replacing signature of a different size ==="
# (b+) replaces an ad-hoc signature with an ad-hoc signature of the same
# size, so __LINKEDIT never has to be re-rounded and codesign's 16 KiB
# granularity never disagrees with signapple's 4 KiB one on the x86_64
# slice. A Developer ID signature will be a different size. Sweep padded
# entitlements (keyless, so this runs anywhere) until the sum crosses a
# 16 KiB boundary, and report what happens there.
#
# Whether a boundary was crossed is read off the signed artifact rather than
# inferred from the classifier's verdict, so "we never exercised the case"
# and "we exercised it and apply agreed" stay distinguishable. The sweep
# spans more than 16 KiB of signature growth, so some size must cross; if
# none does, the harness has stopped testing what it advertises and says so
# instead of reporting an exact round trip. Nothing else here consumes
# scripts/fixtures/apple-roundtrip/ — the synthetic fixture is the crate's
# committed golden, not a substitute for exercising the case on a real bundle.
boundary_state() { # boundary_state <mach-o> -> crossed|aligned|absent
  python3 "$FACTS" "$1" | python3 -c '
import json, sys
def round_up(v, p): return (v + p - 1) // p * p
state = "absent"
for s in json.load(sys.stdin)["slices"]:
    if s["arch"] != "x86_64":
        continue
    f = s["linkedit"]["filesize"]
    state = "crossed" if round_up(f, 0x4000) != round_up(f, 0x1000) else "aligned"
print(state)'
}

BOUNDARY=unexercised
BOUNDARY_EXACT=1
BOUNDARY_STAGE=""
BOUNDARY_PAD=""
for pad in 0 800 1600 2400 3200 4000 4800 5600 6400 7200 8000 8800 9600 10400; do
  # One pair of stage directories, reused: each iteration is independent, and
  # the bundle is 93 MB.
  PS="$WORK/BS/$APP_NAME"; rm -rf "$WORK/BS"; mkdir -p "$WORK/BS"
  cp -R "$SB" "$PS"; chmod -R u+w "$PS"
  PA="$WORK/BA/$APP_NAME"; rm -rf "$WORK/BA"; mkdir -p "$WORK/BA"
  cp -R "$SB" "$PA"; chmod -R u+w "$PA"
  ent_plist "$pad"
  codesign --force --sign - --options runtime --entitlements "$WORK/ent.plist" "$PS/$SIDE" 2>/dev/null
  codesign --force --sign - --options runtime --entitlements "$WORK/ent.plist" "$PS" 2>/dev/null
  apple_tool detach "$PS" "$SB" "$WORK/det-boundary" >/dev/null
  "$SIGNAPPLE" apply --no-verify "$PA" "$WORK/det-boundary/$APP_NAME" >/dev/null 2>&1
  # The padding grows the sidecar's signature too, so every iteration is also
  # a resized-signature case for it. The sidecar is arm64-only and both
  # implementations round an arm64 __LINKEDIT to 16 KiB, so it has no
  # legitimate divergence at any size: a sidecar that is not identical here is
  # a failure, whatever the main binary did. The same goes for the rest of the
  # tree, which is why an unclean one ends the sweep instead of being skipped.
  main_kind="$(diff_kind "$PS/$MAIN" "$PA/$MAIN")"
  side_kind="$(diff_kind "$PS/$SIDE" "$PA/$SIDE")"
  crossed="$(boundary_state "$PS/$MAIN")"
  tree=same; diff -rq "$PS" "$PA" >/dev/null 2>&1 || tree=differs
  printf '    entitlements pad %-5s main: %-24s %-8s -> %s (sidecar %s)\n' \
    "$pad" "$(vmsizes "$PS/$MAIN")" "$crossed" "$main_kind" "$side_kind"
  if [[ "$main_kind" == identical && "$side_kind" == identical && "$tree" == same ]]; then
    # Agreeing on everything while the two roundings land on the same value is
    # the expected non-case; agreeing while they do not is a documented fact
    # that has stopped being true, and must be re-measured, not skipped.
    [[ "$crossed" != crossed ]] || { BOUNDARY=silent; break; }
    continue
  fi
  # Something in the bundle moved: this iteration is the case, and the whole
  # bundle gets graded on it — including the applied bundle's own
  # verification, in (d) below.
  BOUNDARY=crossed
  [[ "$main_kind" == linkedit-vmsize-only && "$side_kind" == identical ]] || BOUNDARY=broken
  grade_bundle "boundary " "$PS" "$PA"
  BOUNDARY_EXACT="$GRADE_EXACT"
  BOUNDARY_STAGE=BA
  BOUNDARY_PAD="$pad"
  break
done
case "$BOUNDARY" in
  # crossed: the case was exercised and graded above.
  # broken:  graded above too, and already counted as a failure there.
  crossed | broken) ;;
  silent)
    printf '    FAIL  %-52s\n' "boundary case exercised the documented divergence"
    printf '        %s\n' "a 16 KiB boundary was crossed and apply matched anyway;" \
      "round-trip.md §3.3 no longer describes signapple — re-measure"
    FAILURES=$((FAILURES + 1)) ;;
  *)
    printf '    FAIL  %-52s\n' "boundary case reached"
    printf '        %s\n' "no padded entitlement size crossed a 16 KiB boundary," \
      "so the latent case went untested; widen the sweep"
    FAILURES=$((FAILURES + 1)) ;;
esac

echo
echo "=== (e) placement-driven apply on the artifact as built ==="
# The design the round trip settled on (round-trip.md §4.2, Path B): nothing
# settles inside the derivation, so `apply` receives the artifact carrying
# sigtool's signatures, sigtool's fat alignment and sigtool's __LINKEDIT
# sizing, and has to land on codesign's layout from the placement record
# alone. `release-tool apple apply` is `eidola-apple::apply`, the shipping
# implementation, so this section grades it against a real bundle.
#
# This is the same equation as (b), with the record standing in for the
# settling, and it is graded the same way:
#     place(as built, detached) == the codesign-signed bundle
# $WORK/detached was taken from S1, which is the as-built artifact signed, so
# U is exactly the input the record names.
P="$(stage P)"
apple_tool apply "$P" "$WORK/detached" >/dev/null
grade_bundle "placement " "$S1" "$P"
PLACEMENT_EXACT="$GRADE_EXACT"

# Again at the replacing-signature size the sweep found, where signapple's
# rounding diverges from codesign's. The record carries codesign's value
# instead of deriving it, so signature size is not supposed to reach this
# path at all; that is a claim about the design, so it is measured rather
# than asserted. The pad comes from the sweep, so if the sweep stopped
# finding a boundary this stops silently exercising a case it names.
PLACEMENT_BOUNDARY_EXACT=1
PLACEMENT_BOUNDARY_STAGE=""
if [[ -n "$BOUNDARY_PAD" ]]; then
  PBS="$(stage PBS)"
  ent_plist "$BOUNDARY_PAD"
  codesign --force --sign - --options runtime --entitlements "$WORK/ent.plist" "$PBS/$SIDE" 2>/dev/null
  codesign --force --sign - --options runtime --entitlements "$WORK/ent.plist" "$PBS" 2>/dev/null
  printf '    %-34s main: %s (%s)\n' "pad $BOUNDARY_PAD, signed as built" \
    "$(vmsizes "$PBS/$MAIN")" "$(boundary_state "$PBS/$MAIN")"
  apple_tool detach "$PBS" "$U" "$WORK/det-placement" >/dev/null
  PB="$(stage PB)"
  apple_tool apply "$PB" "$WORK/det-placement" >/dev/null
  grade_bundle "placement, resized signature " "$PBS" "$PB"
  PLACEMENT_BOUNDARY_EXACT="$GRADE_EXACT"
  PLACEMENT_BOUNDARY_STAGE=PB
else
  printf '    SKIP  %-52s (no size from the sweep)\n' "placement at a resized signature"
fi

echo
echo "=== (d) codesign --verify --deep --strict ==="
# The second argument is *this bundle's* exactness, never the run's: a
# signature seals the load commands, so a bundle that did not round-trip
# byte-exactly is necessarily also a verification failure and that is not a
# separate finding — but a bundle that did round-trip exactly has no such
# excuse and any failure is real.
verify_bundle() { # verify_bundle <stage> <round-tripped byte-exact: yes|no>
  local name="$1" exact="$2"
  if codesign --verify --deep --strict --verbose=2 "$WORK/$name/$APP_NAME" >"$WORK/$name.verify" 2>&1; then
    printf '    PASS  %-52s\n' "$name verifies"
  elif [[ "$exact" == no ]]; then
    printf '    KNOWN %-52s (follows from that bundle'\''s divergence)\n' "$name verifies"
    sed 's/^/        /' "$WORK/$name.verify"
  else
    printf '    FAIL  %-52s\n' "$name verifies"; FAILURES=$((FAILURES + 1))
    sed 's/^/          /' "$WORK/$name.verify"
  fi
}
# S1 is signed by codesign itself, so it must verify unconditionally.
verify_bundle S1 yes
if [[ "$SETTLED_EXACT" -eq 1 ]]; then verify_bundle SA yes; else verify_bundle SA no; fi
# BA is the sweep's applied bundle, when the sweep reached a case to grade:
# the round trip at a replacing signature of a different size has to survive
# verification too, not just compare equal.
if [[ -n "$BOUNDARY_STAGE" ]]; then
  if [[ "$BOUNDARY_EXACT" -eq 1 ]]; then verify_bundle "$BOUNDARY_STAGE" yes
  else verify_bundle "$BOUNDARY_STAGE" no; fi
fi
# P and PB are the placement-driven bundles: signed material written onto the
# artifact as built. Comparing equal is most of the claim, but the signature
# seals the load commands this path rewrites, so verification is the half that
# would catch a record that happened to reproduce a hash the wrong way.
if [[ "$PLACEMENT_EXACT" -eq 1 ]]; then verify_bundle P yes; else verify_bundle P no; fi
if [[ -n "$PLACEMENT_BOUNDARY_STAGE" ]]; then
  if [[ "$PLACEMENT_BOUNDARY_EXACT" -eq 1 ]]; then verify_bundle "$PLACEMENT_BOUNDARY_STAGE" yes
  else verify_bundle "$PLACEMENT_BOUNDARY_STAGE" no; fi
fi

echo
# Two verdicts, because they are two round trips: the shipped path is the
# placement-driven one on the artifact as built, and the settled one is what
# keeps signapple usable as an independent checker.
if [[ "$FAILURES" -ne 0 ]]; then
  echo "apple-roundtrip: $FAILURES check(s) failed"
  exit "$FAILURES"
fi
if [[ "$PLACEMENT_EXACT" -eq 1 && "$PLACEMENT_BOUNDARY_EXACT" -eq 1 ]]; then
  echo "apple-roundtrip: the placement-driven round trip is byte-exact on the"
  echo "                 artifact as built — nothing settled, both signature sizes"
fi
if [[ "$SETTLED_EXACT" -eq 1 ]]; then
  echo "apple-roundtrip: the settled round trip is byte-exact"
  if [[ "$KNOWN_DIVERGENCE" -eq 1 ]]; then
    echo "                 at a signature size crossing a 16 KiB boundary, upstream"
    echo "                 signapple still writes the documented __LINKEDIT vmsize"
    echo "                 (one line in the signapple fork)"
  fi
else
  echo "apple-roundtrip: the settled round trip carries the documented __LINKEDIT"
  echo "                 vmsize divergence (one line in the signapple fork)"
fi
exit 0
