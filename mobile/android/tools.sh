# Sourced by build.sh. Fetches the pinned build tools into a cache and, on every run, checks each
# archive by SHA-256 before it is unpacked afresh (tools/fetch.sh); nothing else is downloaded,
# and the local SDK is not used. (Google's repository manifest lists the archives' SHA-1s; those
# match too.)
#
#   build-tools 36.1.0   aapt2, d8/R8, zipalign, apksigner (per host OS; Linux is x86_64)
#   platform android-36  r02, android.jar
#   kotlinc 2.3.10       the compiler and the Kotlin standard library R8 shrinks into the APK
#   bundletool 1.18.3    the App Bundle (AAB) Play takes, built from the same compiled shell (--aab)
#
# A JDK (17 or newer) must be on PATH: it runs kotlinc, R8, the zip writer and apksigner.
# The APK's bytes do not depend on which JDK (see AGENTS.md -> Reproducibility).

ANDROID_TOOLS="${EIDOLA_ANDROID_TOOLS:-${XDG_CACHE_HOME:-$HOME/.cache}/eidola/android-tools}"
GOOGLE=https://dl.google.com/android/repository
case "$(uname -s)" in
  Darwin) BT_ZIP=build-tools_r36.1_macosx.zip BT_SHA=cc56be1d4ea95041f32f4ebf3a4f5c6b990c7749d706852395006b5654eb82ff ;;
  Linux) BT_ZIP=build-tools_r36.1_linux.zip BT_SHA=a7b5889e4a79fcf3b0976bef40d401f4240fb1eed891d9d91169da1111e11d78 ;;
  *) echo "unsupported host $(uname -s)" >&2; exit 1 ;;
esac

source "$(dirname "${BASH_SOURCE[0]}")/tools/fetch.sh"

BT="$(fetch "$GOOGLE/$BT_ZIP" "$BT_SHA")/android-16"
ANDROID_JAR="$(fetch "$GOOGLE/platform-36_r02.zip" 37607369a28c5b640b3a7998868d45898ebcb777565a0e85f9acf36f29631d2e)/android-36/android.jar"
KOTLIN_HOME="$(fetch https://github.com/JetBrains/kotlin/releases/download/v2.3.10/kotlin-compiler-2.3.10.zip \
  c8d546f9ff433b529fb0ad43feceb39831040cae2ca8d17e7df46364368c9a9e)/kotlinc"
# The same SHA-256 GitHub records for the release asset.
BUNDLETOOL="$(fetch https://github.com/google/bundletool/releases/download/1.18.3/bundletool-all-1.18.3.jar \
  a099cfa1543f55593bc2ed16a70a7c67fe54b1747bb7301f37fdfd6d91028e29 bundletool.jar)/bundletool.jar"
