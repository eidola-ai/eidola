#!/usr/bin/env bash
# tcp.connect's dialling on a host JVM, with the pinned kotlinc (tools.sh): every address a name
# resolves to is tried within the one deadline, a refusing or silent first address leaves time for
# a working one, and an abandoned connect dials nothing more (tests/ConnectTest.kt). The code under
# test (src/ai/eidola/shell/Connect.kt) uses java.net alone, so it compiles here as it does against
# android.jar; nothing from this run reaches the APK.
#   test-connect.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/tools.sh"
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
"$KOTLIN_HOME/bin/kotlinc" -jvm-target 17 -d "$T/test.jar" \
  "$HERE/src/ai/eidola/shell/Connect.kt" "$HERE/tests/ConnectTest.kt" 2>&1 \
  | grep -v '^warning: classpath entry points to a non-existent location' >&2 || true
test -f "$T/test.jar"
java -cp "$T/test.jar:$KOTLIN_HOME/lib/kotlin-stdlib.jar" ai.eidola.shell.ConnectTestKt
