#!/usr/bin/env bash
# Task 55 Wave 2 — the universal round-trip spike, as a re-runnable check.
#
# Answers, on a real `.#eidola-gui-macos-universal` bundle:
#   (a) is whole-bundle ad-hoc signing byte-deterministic?
#   (b) is signapple detach -> apply byte-identical to the signed bundle,
#       on both slices of the main binary and on the arm64-only sidecar?
#   (c) does the artifact's `__LINKEDIT` vmsize settle, or does the first
#       transition out of sigtool's linker-signed state move it?
#   (d) does the result survive `codesign --verify --deep --strict`?
#
# Written result and verdict: work/reference/55-apple-signing/round-trip.md.
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
# This is the order Wave 7's CI job uses, so it is the order measured here.
sign_adhoc() {
  local app="$1"
  codesign --force --sign - "$app/$SIDE"
  codesign --force --sign - "$app"
}

# One full codesign ad-hoc cycle, leaving no bundle-level seal. This is the
# mitigation the spec designed and §4 of round-trip.md measures: as built,
# the artifact carries sigtool's linker-signed signatures, whose 4 KiB code
# pages make them ~2.4x larger than codesign's, so the first codesign
# *shrinks* __LINKEDIT — the case signapple's apply does not handle.
#
# Run here, not in the derivation: codesign is absent from the Nix build
# sandbox (measured), and admitting it would tie the recorded narHash to the
# host's macOS version. Whether the build settles or `apply` learns to land
# on the unsettled artifact is an open decision — round-trip.md §4.2.
settle() {
  local app="$1"
  sign_adhoc "$app"
  codesign --remove-signature "$app/$MAIN" "$app/$SIDE"
  rm -rf "$app/Contents/_CodeSignature"
  sign_adhoc "$app"
  rm -rf "$app/Contents/_CodeSignature"
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
if diff -r "$S1" "$S2" >/dev/null 2>&1; then
  printf '    PASS  %-52s (same)\n' "whole bundle tree"
else
  printf '    FAIL  %-52s (differ)\n' "whole bundle tree"; FAILURES=$((FAILURES + 1))
fi

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
# So `codesign` signs (S1, above) and scripts/apple-detach.py lifts the
# superblobs into signapple's layout; signapple's `apply` stays the
# independent implementation. This is exactly the equation the design
# rests on: apply(unsigned, detached) == shipped.
python3 "$REPO_ROOT/scripts/apple-detach.py" "$S1" "$WORK/detached" "$U" >/dev/null
A="$(stage A)"
"$SIGNAPPLE" apply --no-verify "$A" "$WORK/detached/$APP_NAME" >/dev/null

# Upstream signapple diverges from codesign on exactly one field (see
# apple_linkedit_diff.py). Report that state by name rather than as a bare
# "differ": a diff that has grown beyond it is the regression worth failing.
# Graded only on the settled artifact below: on the as-built one the
# divergence is expected and is itself the finding, so it is reported
# without failing the run.
ROUNDTRIP_EXACT=1
classify() { # classify <graded|info> <label> <a> <b>
  local grade="$1" label="$2" out kind
  out="$(python3 "$REPO_ROOT/scripts/apple_linkedit_diff.py" "$3" "$4" 2>&1)" || true
  kind="$(printf '%s\n' "$out" | head -1)"
  case "$kind" in
    identical)
      printf '    PASS  %-52s (identical)\n' "$label" ;;
    linkedit-vmsize-only)
      ROUNDTRIP_EXACT=0
      printf '    KNOWN %-52s (documented signapple divergence)\n' "$label"
      printf '%s\n' "$out" | tail -n +2 | sed 's/^/        /' ;;
    *)
      ROUNDTRIP_EXACT=0
      if [[ "$grade" == graded ]]; then
        printf '    FAIL  %-52s\n' "$label"; FAILURES=$((FAILURES + 1))
      else
        printf '    NOTE  %-52s (expected before settling)\n' "$label"
      fi
      printf '%s\n' "$out" | sed 's/^/          /' ;;
  esac
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
# is the configuration the design would actually ship.
SB="$(stage SB)"; settle "$SB"
printf '    %-34s main: %s\n' "settled" "$(vmsizes "$SB/$MAIN")"
printf '    %-34s side: %s\n' "" "$(vmsizes "$SB/$SIDE")"
SS="$WORK/SS/$APP_NAME"; rm -rf "$WORK/SS"; mkdir -p "$WORK/SS"; cp -R "$SB" "$SS"
SA_="$WORK/SA/$APP_NAME"; rm -rf "$WORK/SA"; mkdir -p "$WORK/SA"; cp -R "$SB" "$SA_"
chmod -R u+w "$SS" "$SA_"
sign_adhoc "$SS"
python3 "$REPO_ROOT/scripts/apple-detach.py" "$SS" "$WORK/detached-settled" "$SB" >/dev/null
"$SIGNAPPLE" apply --no-verify "$SA_" "$WORK/detached-settled/$APP_NAME" >/dev/null
ROUNDTRIP_EXACT=1
classify graded "main binary (both slices), apply == signed" "$SS/$MAIN" "$SA_/$MAIN"
classify graded "sidecar (arm64-only), apply == signed"      "$SS/$SIDE" "$SA_/$SIDE"
if diff -r "$SS" "$SA_" >/dev/null 2>&1; then
  printf '    PASS  %-52s (same)\n' "whole bundle tree"
else
  printf '    FAIL  %-52s (differ)\n' "whole bundle tree"; FAILURES=$((FAILURES + 1))
fi

echo
echo "=== (b++) the latent case: a replacing signature of a different size ==="
# (b+) replaces an ad-hoc signature with an ad-hoc signature of the same
# size, so __LINKEDIT never has to be re-rounded and codesign's 16 KiB
# granularity never disagrees with signapple's 4 KiB one on the x86_64
# slice. A Developer ID signature will be a different size. Sweep padded
# entitlements (keyless, so this runs anywhere) until the sum crosses a
# 16 KiB boundary, and report what happens there.
found=no
for pad in 0 800 1600 2400 3200; do
  PS="$WORK/P$pad/$APP_NAME"; rm -rf "$WORK/P$pad"; mkdir -p "$WORK/P$pad"
  cp -R "$SB" "$PS"; chmod -R u+w "$PS"
  PA="$WORK/Q$pad/$APP_NAME"; rm -rf "$WORK/Q$pad"; mkdir -p "$WORK/Q$pad"
  cp -R "$SB" "$PA"; chmod -R u+w "$PA"
  python3 -c "
import sys
open(sys.argv[1], 'w').write(
  '<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n'
  '<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" '
  '\"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n'
  '<plist version=\"1.0\"><dict>\n'
  '<key>com.apple.security.cs.allow-jit</key><true/>\n'
  '<key>com.apple.security.application-groups</key><string>' + 'A' * int(sys.argv[2]) + '</string>\n'
  '</dict></plist>\n')" "$WORK/ent.plist" "$pad"
  codesign --force --sign - --options runtime --entitlements "$WORK/ent.plist" "$PS/$SIDE" 2>/dev/null
  codesign --force --sign - --options runtime --entitlements "$WORK/ent.plist" "$PS" 2>/dev/null
  python3 "$REPO_ROOT/scripts/apple-detach.py" "$PS" "$WORK/det$pad" "$SB" >/dev/null
  "$SIGNAPPLE" apply --no-verify "$PA" "$WORK/det$pad/$APP_NAME" >/dev/null 2>&1
  kind="$(python3 "$REPO_ROOT/scripts/apple_linkedit_diff.py" "$PS/$MAIN" "$PA/$MAIN" 2>&1 | head -1)" || true
  printf '    entitlements pad %-5s main: %-24s -> %s\n' "$pad" "$(vmsizes "$PS/$MAIN")" "$kind"
  if [[ "$kind" == "linkedit-vmsize-only" ]]; then
    found=yes
    classify graded "boundary case, apply == signed" "$PS/$MAIN" "$PA/$MAIN"
    break
  elif [[ "$kind" != "identical" ]]; then
    classify graded "boundary case, apply == signed" "$PS/$MAIN" "$PA/$MAIN"
    break
  fi
done
[[ "$found" == yes ]] || printf '    %s\n' "no boundary crossed at these sizes; the case is covered by scripts/fixtures/apple-roundtrip/"

echo
echo "=== (d) codesign --verify --deep --strict ==="
verify_bundle() { # verify_bundle <stage> <required: yes|if-exact>
  local name="$1" required="$2"
  if codesign --verify --deep --strict --verbose=2 "$WORK/$name/$APP_NAME" >"$WORK/$name.verify" 2>&1; then
    printf '    PASS  %-52s\n' "$name verifies"
  elif [[ "$required" == "if-exact" && "$ROUNDTRIP_EXACT" -eq 0 ]]; then
    # A signature seals the load commands, so the vmsize divergence above is
    # necessarily also a verification failure. Not a separate finding.
    printf '    KNOWN %-52s (follows from the divergence above)\n' "$name verifies"
    sed 's/^/        /' "$WORK/$name.verify"
  else
    printf '    FAIL  %-52s\n' "$name verifies"; FAILURES=$((FAILURES + 1))
    sed 's/^/          /' "$WORK/$name.verify"
  fi
}
verify_bundle S1 yes
verify_bundle SA if-exact

echo
if [[ "$FAILURES" -ne 0 ]]; then
  echo "apple-roundtrip: $FAILURES check(s) failed"
elif [[ "$ROUNDTRIP_EXACT" -eq 1 ]]; then
  echo "apple-roundtrip: round trip is byte-exact"
else
  echo "apple-roundtrip: round trip is byte-exact except the documented"
  echo "                 __LINKEDIT vmsize divergence (one line in the signapple fork)"
fi
exit "$FAILURES"
