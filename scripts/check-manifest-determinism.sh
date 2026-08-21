#!/usr/bin/env bash
# Enforce, structurally, that `artifact-manifest.json` can never record a
# key-dependent value.
#
# The manifest is the one document anyone can regenerate byte-identically
# from source on a machine that holds no keys, and `release-tool` refuses
# to attest a release whose published manifest differs from the committed
# one. That property is what the whole verification story rests on, so it
# is checked rather than remembered. Apple envelope material — the signed
# installable's hash, the detached signature bundle's hash, the Team ID,
# the signing identifier, the notarization ticket — belongs one layer up,
# in the human attestation, which is signed and already non-deterministic
# (docs/verification.md, docs/trust-root.md).
#
# Three assertions:
#
#   1. Every artifact entry declares a known `type` and carries exactly the
#      fields that type allows. A new field cannot appear unnoticed.
#   2. No key anywhere in the document names signing material.
#   3. Over `.github/workflows/artifacts.yml`: the job that assembles the
#      manifest does not depend on a signing job, and no job that produces
#      a manifest partial is granted the signing environment. This is the
#      one that makes the invariant structural instead of lexical — a
#      manifest field cannot depend on a key that never reaches the job
#      computing it.
#
# With no arguments the committed manifest and workflow are checked and the
# bad fixtures beside this script are then required to FAIL — the check's
# own teeth, run everywhere the check runs. `--manifest` / `--workflow`
# check a specific file instead and skip the self-test.
#
# Usage:
#   scripts/check-manifest-determinism.sh
#   scripts/check-manifest-determinism.sh --manifest PATH [--workflow PATH]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_DIR="$REPO_ROOT/scripts/fixtures/manifest-determinism"

MANIFEST=""
WORKFLOW=""
SELF_TEST=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      SELF_TEST=0
      shift 2
      ;;
    --workflow)
      WORKFLOW="$2"
      SELF_TEST=0
      shift 2
      ;;
    -h | --help)
      sed -n '2,30p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

violations=0

fail() {
  echo "  VIOLATION: $*" >&2
  violations=$((violations + 1))
}

# Report each non-empty line of a violation report, optionally wrapped in a
# message ($2 with %s standing for the line).
report_lines() {
  local out="$1" template="${2:-%s}" line
  while IFS= read -r line; do
    if [[ -n "$line" ]]; then
      # shellcheck disable=SC2059 # the template is a literal in this file
      fail "$(printf "$template" "$line")"
    fi
  done <<< "$out"
}

# ── 1 + 2: the manifest document ─────────────────────────────────────────
check_manifest() {
  local file="$1" out

  if [[ ! -f "$file" ]]; then
    echo "error: no such manifest: $file" >&2
    exit 2
  fi

  # Field allow-list per artifact type. `oci` records the image digest;
  # `nix` records the store-path checkpoint plus the archive hash a user
  # can check without Nix; `file` records the sha256 of a single published
  # file. Key sets are compared for equality, so both an unknown field and
  # a dropped one are violations.
  out="$(
    jq -r '
      def allowed:
        {
          "oci":  ["digest", "platform", "type"],
          "nix":  ["archiveSha256", "narHash", "platform", "type"],
          "file": ["platform", "sha256", "type"]
        };
      (.artifacts // {}) | to_entries[]
      | .key as $name
      | .value as $entry
      | (($entry | objects | .type) // "«missing»") as $type
      | if ($entry | type) != "object" then
          "artifacts.\($name): not an object"
        elif (allowed | has($type) | not) then
          "artifacts.\($name): unknown type \"\($type)\" (allowed: oci, nix, file)"
        elif (($entry | keys) != (allowed[$type] | sort)) then
          "artifacts.\($name): type \"\($type)\" allows exactly \(allowed[$type] | sort | join(", ")) — found \($entry | keys | join(", "))"
        else
          empty
        end
    ' "$file"
  )"
  report_lines "$out"

  # No key anywhere may name signing material, at any depth. Artifact names
  # are keys too, so `eidola-gui-macos-signed-zip` trips this as readily as
  # a `team_id` field would.
  out="$(
    jq -r '
      [paths | .[] | select(type == "string")]
      | unique[]
      | select(test("sign|notar|ticket|team|cert|staple|apple"; "i"))
    ' "$file"
  )"
  report_lines "$out" \
    'key "%s" names signing material — Apple envelope hashes and identities belong in the human attestation, not the manifest'
}

# ── 3: the job graph ─────────────────────────────────────────────────────
# A signing job holds a key; a manifest job must not be able to see one.
# Parsed structurally from the workflow rather than grepped: what matters
# is which job `needs` which, and which job is granted the signing
# environment, not whether the word "sign" appears somewhere.
check_workflow() {
  local file="$1" out

  if [[ ! -f "$file" ]]; then
    echo "error: no such workflow: $file" >&2
    exit 2
  fi

  out="$(
    awk '
      # ── collect: job name -> needs, environment, produces-partial ──
      /^[^ #]/ { in_jobs = ($0 ~ /^jobs:/); next }

      in_jobs && /^  [A-Za-z0-9_-]+:[ \t]*$/ {
        job = $0
        sub(/^  /, "", job)
        sub(/:[ \t]*$/, "", job)
        jobs[++njobs] = job
        collecting_needs = 0
        next
      }

      job == "" { next }

      # `needs:` is either inline (`needs: oci`, `needs: [a, b]`) or a
      # block list of `- name` lines that follow it.
      /^    needs:/ {
        rest = $0
        sub(/^    needs:[ \t]*/, "", rest)
        gsub(/[][,]/, " ", rest)
        if (rest ~ /[^ \t]/) {
          n = split(rest, parts, /[ \t]+/)
          for (i = 1; i <= n; i++)
            if (parts[i] != "") needs[job] = needs[job] " " parts[i]
          collecting_needs = 0
        } else {
          collecting_needs = 1
        }
        next
      }
      collecting_needs && /^      - / {
        dep = $0
        sub(/^      -[ \t]*/, "", dep)
        needs[job] = needs[job] " " dep
        next
      }
      collecting_needs && /^    [A-Za-z]/ { collecting_needs = 0 }

      /^    environment:/ {
        env = $0
        sub(/^    environment:[ \t]*/, "", env)
        environment[job] = environment[job] " " env
        next
      }

      # A job "produces a manifest partial" if it runs the manifest script
      # or exports the partial as a job output.
      /artifact-manifest\.sh/ || /artifact_manifest:/ { partial[job] = 1 }

      END {
        # A signing job is one granted the Apple signing environment. The
        # name is checked too, so a signing job that has not yet been given
        # its environment (or was renamed) is still caught.
        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          if (environment[j] ~ /apple-signing/ || j ~ /sign/) signer[j] = 1
        }

        found = 0
        for (i = 1; i <= njobs; i++) if (jobs[i] == "artifact-manifest") found = 1
        if (!found)
          print "no `artifact-manifest` job in this workflow — the job-graph assertion cannot be evaluated"

        n = split(needs["artifact-manifest"], deps, /[ \t]+/)
        for (i = 1; i <= n; i++) {
          d = deps[i]
          if (d != "" && signer[d])
            print "the `artifact-manifest` job needs `" d "`, a signing job — a key-dependent value could then reach the manifest"
        }

        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          if (partial[j] && environment[j] ~ /apple-signing/)
            print "job `" j "` produces a manifest partial and is granted the apple-signing environment"
        }
      }
    ' "$file"
  )"
  report_lines "$out"
}

# ── run ──────────────────────────────────────────────────────────────────
if [[ -n "$MANIFEST" || -n "$WORKFLOW" ]]; then
  if [[ -n "$MANIFEST" ]]; then check_manifest "$MANIFEST"; fi
  if [[ -n "$WORKFLOW" ]]; then check_workflow "$WORKFLOW"; fi
else
  echo "Checking artifact-manifest.json + artifacts.yml for key-dependent material..."
  check_manifest "$REPO_ROOT/artifact-manifest.json"
  check_workflow "$REPO_ROOT/.github/workflows/artifacts.yml"
fi

if [[ "$violations" -gt 0 ]]; then
  echo "FAIL: $violations determinism violation(s)." >&2
  exit 1
fi

if [[ "$SELF_TEST" -eq 1 ]]; then
  echo "Self-test: every bad fixture must be rejected..."
  rc=0
  for fixture in "$FIXTURE_DIR"/bad-*.json; do
    [[ -e "$fixture" ]] || continue
    if "${BASH_SOURCE[0]}" --manifest "$fixture" > /dev/null 2>&1; then
      echo "  FAIL: $(basename "$fixture") was accepted" >&2
      rc=1
    else
      echo "  rejected: $(basename "$fixture")"
    fi
  done
  for fixture in "$FIXTURE_DIR"/bad-*.yml; do
    [[ -e "$fixture" ]] || continue
    if "${BASH_SOURCE[0]}" --workflow "$fixture" > /dev/null 2>&1; then
      echo "  FAIL: $(basename "$fixture") was accepted" >&2
      rc=1
    else
      echo "  rejected: $(basename "$fixture")"
    fi
  done
  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL: the check has lost its teeth — a fixture it must reject passed." >&2
    exit 1
  fi
fi

echo "OK: no key-dependent material can reach artifact-manifest.json."
