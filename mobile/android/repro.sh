#!/usr/bin/env bash
# The reproducibility check: two clean release builds must give the same unsigned APK, and
# stripping the signature from a signed one (a throwaway key, made here) must give it back.
#   repro.sh [--bundle DIR]
# Prints the unsigned APK's SHA-256. Without --bundle it embeds the contract test page, and the hash
# must also equal reproducible.sha256, which a Linux CI build checks as well; update that file when
# the shell, its build or the contract test page changes (the build prints the new value).
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
  KEYSTORE="$T/key.p12" "$HERE/build.sh" "${ARGS[@]}" --out "$T/$x" >/dev/null
  python3 "$HERE/tools/strip_apk_signature.py" "$T/$x/eidola-shell-signed.apk" "$T/$x/stripped.apk"
done
cmp "$T/a/eidola-shell-unsigned.apk" "$T/b/eidola-shell-unsigned.apk"
cmp "$T/a/stripped.apk" "$T/a/eidola-shell-unsigned.apk"
cmp "$T/b/stripped.apk" "$T/a/eidola-shell-unsigned.apk"
cmp -s "$T/a/eidola-shell-signed.apk" "$T/b/eidola-shell-signed.apk" && { echo "signed APKs identical: signing was not exercised" >&2; exit 1; }
GOT="$(sha256 "$T/a/eidola-shell-unsigned.apk" | cut -d' ' -f1)"
echo "unsigned APK reproduced, signature strips back to it: $GOT"
if (($# == 0)); then
  WANT="$(cut -d' ' -f1 "$HERE/reproducible.sha256")"
  [[ "$GOT" == "$WANT" ]] || { echo "expected $WANT (reproducible.sha256)" >&2; exit 1; }
  echo "matches reproducible.sha256"
fi
