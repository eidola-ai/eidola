#!/usr/bin/env bash
# Builds the Android shell with the SDK's own tools and pinned kotlinc; no Gradle.
#   build.sh [--bundle DIR] [--debug] [--aab] [--out DIR]
# --bundle  the bundle to embed (default: the app, built by ../build-bundle.sh; the tests embed
#           ../contract-test)
# --debug   a debuggable APK as ai.eidola.mobile.dev (console in logcat, inspectable WebView);
#           the default, release, is ai.eidola.mobile
# --aab     also the App Bundle Play takes, OUT/eidola-shell-unsigned.aab: the same compiled shell and
#           bundle, laid out as a bundletool module (a release build only; Play signs what it
#           delivers, and the upload key signs this file at release time, outside the build)
# --out     output directory (default build/<release|debug>)
# Writes OUT/eidola-shell-unsigned.apk, the reproducible artifact, and, when KEYSTORE names a
# PKCS#12 key store (password KS_PASS, default "android"), OUT/eidola-shell-signed.apk.
set -euo pipefail
export TZ=UTC LC_ALL=C LANG=C
HERE="$(cd "$(dirname "$0")" && pwd)"
BUNDLE=
VARIANT=release
AAB=false
OUT=
while (($#)); do
  case "$1" in
    --bundle) BUNDLE="$(cd "$2" && pwd)"; shift 2 ;;
    --debug) VARIANT=debug; shift ;;
    --aab) AAB=true; shift ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ $AAB == true && $VARIANT == debug ]] && { echo "--aab builds the release variant only" >&2; exit 2; }
OUT="$(mkdir -p "${OUT:-$HERE/build/$VARIANT}" && cd "${OUT:-$HERE/build/$VARIANT}" && pwd)"
MIN_SDK=30
source "$HERE/tools.sh"
if [[ -z "$BUNDLE" ]]; then
  "$HERE/../build-bundle.sh" >&2
  BUNDLE="$HERE/../build/bundle"
fi
test -f "$BUNDLE/index.html" || { echo "$BUNDLE has no index.html" >&2; exit 1; }
# The digest tool refuses a file the shell would not serve; it prints the digest the shell will show.
DIGEST="$(python3 "$HERE/../tools/bundle_digest.py" "$BUNDLE")"

W="$OUT/stage"
rm -rf "$W" && mkdir -p "$W/dex"

# 1. The binary manifest and resources. A debug build is debuggable and installs beside a release one, as
#    ai.eidola.mobile.dev; the release package is the manifest's own, ai.eidola.mobile.
DEBUG_MODE=()
[[ $VARIANT == debug ]] && DEBUG_MODE=(--debug-mode --rename-manifest-package ai.eidola.mobile.dev)
#    The only resources are the launcher icon's, generated from brand/ (just update-brand).
mkdir -p "$W/res"
for f in $(cd "$HERE/res" && find . -type f | LC_ALL=C sort); do
  "$BT/aapt2" compile -o "$W/res" "$HERE/res/$f"
done
"$BT/aapt2" link -o "$W/base.apk" --manifest "$HERE/AndroidManifest.xml" -I "$ANDROID_JAR" \
  --min-sdk-version "$MIN_SDK" --target-sdk-version 36 --no-compress "${DEBUG_MODE[@]}" \
  $(find "$W/res" -name '*.flat' | LC_ALL=C sort)

# 2. Kotlin to classes, against android.jar only (no JDK classes).
"$KOTLIN_HOME/bin/kotlinc" -no-jdk -no-reflect -jvm-target 17 -cp "$ANDROID_JAR" -d "$W/classes.jar" \
  -Xno-param-assertions -Xno-call-assertions -Xno-receiver-assertions \
  $(find "$HERE/src" -name '*.kt' | sort) 2>&1 | grep -v '^warning: classpath entry points to a non-existent location' >&2 || true
test -f "$W/classes.jar"

# 3. R8: our classes plus the Kotlin standard library, shrunk to what the shell uses, to dex.
java -cp "$BT/lib/d8.jar" com.android.tools.r8.R8 --release --min-api "$MIN_SDK" --lib "$ANDROID_JAR" \
  --pg-conf "$HERE/keep.pro" --output "$W/dex" "$W/classes.jar" "$KOTLIN_HOME/lib/kotlin-stdlib.jar"

# 4. A deterministic zip: sorted, stored, one fixed timestamp.
java "$HERE/tools/ApkZip.java" "$W/unaligned.apk" "$W/base.apk" \
  "classes.dex=$W/dex/classes.dex" "assets/bundle/=$BUNDLE"

# 5. Align (16 KiB pages for any native library; there are none).
"$BT/zipalign" -f -P 16 4 "$W/unaligned.apk" "$OUT/eidola-shell-unsigned.apk"

# 6. Sign, v2 and v3 only, preserving alignment so tools/strip_apk_signature.py restores step 5.
if [[ -n "${KEYSTORE:-}" ]]; then
  "$BT/apksigner" sign --ks "$KEYSTORE" --ks-pass pass:"${KS_PASS:-android}" --min-sdk-version "$MIN_SDK" \
    --v1-signing-enabled false --v2-signing-enabled true --v3-signing-enabled true --alignment-preserved \
    --out "$OUT/eidola-shell-signed.apk" "$OUT/eidola-shell-unsigned.apk"
fi
# 7. The App Bundle: the same manifest and resources linked as protocol buffers, which is the form a
#    bundle carries, and the module zip (manifest, resource table, dex, the bundle as assets) that
#    bundletool wraps. Nothing is compiled twice: the dex and the bundle are the APK's own files.
if $AAB; then
  #  --exclude-sources: the proto resource table would otherwise record each resource's absolute source
  #  path, which is the checkout's location (the binary table the APK carries has no such field).
  "$BT/aapt2" link --proto-format --exclude-sources -o "$W/base-proto.apk" \
    --manifest "$HERE/AndroidManifest.xml" -I "$ANDROID_JAR" --min-sdk-version "$MIN_SDK" \
    --target-sdk-version 36 --no-compress \
    $(find "$W/res" -name '*.flat' | LC_ALL=C sort)
  python3 "$HERE/tools/aab_module.py" "$W/base.zip" "$W/base-proto.apk" "$W/dex/classes.dex" "$BUNDLE"
  rm -f "$W/deflated.aab"
  java -jar "$BUNDLETOOL" build-bundle --modules="$W/base.zip" --output="$W/deflated.aab"
  #  bundletool's entries are deflated by the JDK's zlib; stored, they depend on nothing but their contents.
  python3 "$HERE/tools/zip_stored.py" "$W/deflated.aab" "$OUT/eidola-shell-unsigned.aab"
fi
echo "bundle: $BUNDLE"
echo "digest: $DIGEST"
sha256 "$OUT"/eidola-shell-*.apk
if $AAB; then sha256 "$OUT/eidola-shell-unsigned.aab"; fi
