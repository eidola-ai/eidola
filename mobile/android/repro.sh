#!/usr/bin/env bash
# The reproducibility check: two clean release builds must give the same unsigned APK and the same
# App Bundle, stripping the signature from a signed APK (a throwaway key, made here) must give the
# unsigned one back, and the universal APK bundletool generates from the App Bundle, as Play does,
# must carry the APK's code and bundle byte for byte.
#   repro.sh [--bundle DIR]
# Prints both SHA-256s. Without --bundle it embeds the contract test page, and both must also equal
# reproducible.sha256, which a Linux CI build checks as well; update that file when the shell, its
# build or the contract test page changes (the build prints the new values).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/tools.sh"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
keytool -genkeypair -keystore "$T/key.p12" -storetype PKCS12 -storepass android -alias shell \
  -keyalg EC -groupname secp256r1 -validity 1 -dname "CN=throwaway" >/dev/null 2>&1
ARGS=("$@")
(($# == 0)) && ARGS=(--bundle "$HERE/../contract-test")
for x in a b; do
  KEYSTORE="$T/key.p12" "$HERE/build.sh" "${ARGS[@]}" --aab --out "$T/$x" >/dev/null
  python3 "$HERE/tools/strip_apk_signature.py" "$T/$x/eidola-shell-signed.apk" "$T/$x/stripped.apk"
done
cmp "$T/a/eidola-shell-unsigned.apk" "$T/b/eidola-shell-unsigned.apk"
cmp "$T/a/stripped.apk" "$T/a/eidola-shell-unsigned.apk"
cmp "$T/b/stripped.apk" "$T/a/eidola-shell-unsigned.apk"
cmp -s "$T/a/eidola-shell-signed.apk" "$T/b/eidola-shell-signed.apk" && { echo "signed APKs identical: signing was not exercised" >&2; exit 1; }
cmp "$T/a/eidola-shell-unsigned.aab" "$T/b/eidola-shell-unsigned.aab"
# What Play generates from the bundle differs from the APK in its manifest and resource table
# (bundletool records the split configuration there) and in its signature; the code and the app
# bundle must be the same bytes.
java -jar "$BUNDLETOOL" build-apks --bundle "$T/a/eidola-shell-unsigned.aab" --output "$T/universal.apks" \
  --mode=universal --ks "$T/key.p12" --ks-pass pass:android --ks-key-alias shell 2>/dev/null
unzip -q "$T/universal.apks" universal.apk -d "$T"
python3 "$HERE/tools/compare_apk_contents.py" "$T/a/eidola-shell-unsigned.apk" "$T/universal.apk" \
  --require classes.dex --require assets/ >"$T/compare.txt" || { cat "$T/compare.txt" >&2; exit 1; }
GOT="$(sha256 "$T/a/eidola-shell-unsigned.apk" | cut -d' ' -f1)"
GOT_AAB="$(sha256 "$T/a/eidola-shell-unsigned.aab" | cut -d' ' -f1)"
echo "unsigned APK reproduced, signature strips back to it: $GOT"
echo "App Bundle reproduced, its universal APK carries the same code and bundle: $GOT_AAB"
if (($# == 0)); then
  for pair in "eidola-shell-unsigned.apk $GOT" "eidola-shell-unsigned.aab $GOT_AAB"; do
    set -- $pair
    WANT="$(awk -v f="$1" '$2 == f { print $1 }' "$HERE/reproducible.sha256")"
    [[ "$2" == "$WANT" ]] || { echo "$1: expected ${WANT:-nothing} (reproducible.sha256), got $2" >&2; exit 1; }
  done
  echo "both match reproducible.sha256"
fi
