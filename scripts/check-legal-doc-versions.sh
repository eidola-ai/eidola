#!/usr/bin/env bash
# Enforce the legal-document versioning contract (see AGENTS.md,
# "terms-acceptance"): the published legal documents are accepted by the
# SHA-256 of their exact bytes, ordered by a front-matter `version`
# integer that the server's acceptance gate compares with `>=`. That
# ordering is only trustworthy if the bytes can never change without the
# version incrementing — which is exactly what this script checks:
#
#   * every legal document must declare `version = N` (N >= 1)
#   * an unchanged file necessarily keeps its version (the version is part
#     of the hashed bytes)
#   * a changed file must go from version N to exactly N + 1 — including
#     content reverts, which must re-publish under a fresh version rather
#     than resurrect an old hash with ambiguous ordering
#   * a document new in this change (or gaining a version for the first
#     time) must start at version 1
#   * legal documents must never be deleted
#
# Usage: check-legal-doc-versions.sh <base-ref-or-sha>
set -euo pipefail

base="${1:?usage: check-legal-doc-versions.sh <base-ref-or-sha>}"

DOCS=(www/pages/terms.md www/pages/privacy.md)

version_of() {
    # Front-matter `version = N` (the terms_feed parser's narrow contract).
    sed -n 's/^version[[:space:]]*=[[:space:]]*\([0-9][0-9]*\)[[:space:]]*$/\1/p' | head -1
}

fail=0
for f in "${DOCS[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f is missing — published legal documents must not be deleted"
        fail=1
        continue
    fi

    new_v=$(version_of < "$f")
    if [ -z "$new_v" ]; then
        echo "ERROR: $f has no front-matter 'version = N'"
        fail=1
        continue
    fi

    if ! git cat-file -e "$base:$f" 2> /dev/null; then
        # New document: versions start at 1.
        if [ "$new_v" != "1" ]; then
            echo "ERROR: $f is new and must declare 'version = 1' (found $new_v)"
            fail=1
        fi
        continue
    fi

    # Compare blob hashes, not shell strings: command substitution strips
    # trailing newlines, and a newline-only change still changes the
    # acceptance SHA-256, so it must increment the version too.
    if [ "$(git rev-parse "$base:$f")" = "$(git hash-object -- "$f")" ]; then
        continue
    fi

    old_v=$(git show "$base:$f" | version_of)
    if [ -z "$old_v" ]; then
        # First versioned revision of a previously unversioned document.
        if [ "$new_v" != "1" ]; then
            echo "ERROR: $f gained versioning and must start at 'version = 1' (found $new_v)"
            fail=1
        fi
        continue
    fi

    if [ "$new_v" -ne $((old_v + 1)) ]; then
        echo "ERROR: $f changed (its acceptance hash changed) but version went $old_v -> $new_v; it must be $((old_v + 1))"
        fail=1
    fi
done

if [ "$fail" -eq 0 ]; then
    echo "Legal document versioning OK."
fi
exit "$fail"
