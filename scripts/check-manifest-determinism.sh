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
# Four assertions:
#
#   0. The file is one document: no repeated members. Parsers keep the last
#      of a duplicate and drop the rest, so a file with two `artifacts`
#      members would be *checked* as one reading and *signed* as the whole
#      bytes. Nothing below means anything until this holds.
#   1. The document's envelope and every artifact entry carry exactly the
#      fields they are allowed. A new field cannot appear unnoticed — and
#      determinism dies to any unvalidated field, not only to key-shaped
#      ones.
#   2. No key anywhere in the document names signing material.
#   3. Over `.github/workflows/artifacts.yml`: no ancestor of the job that
#      assembles the manifest is a signing job, and no job producing a
#      manifest partial is granted the signing environment. This is the one
#      that makes the invariant structural instead of lexical — a manifest
#      field cannot depend on a key that never reaches the job computing
#      it.
#
# What this is not: assertion 3 reads the workflow with a scanner, not a
# YAML parser, because a real one is not available here without adding a
# dependency to a check whose whole value is being cheap. It reads job
# keys, `needs:` and `environment:` in the forms GitHub accepts for them —
# bare or quoted scalars, flow sequences on one line or several, block
# lists, block scalars, trailing comments, any letter case for an
# environment name — and normalizes them before comparing. It assumes the
# two-space job indentation this repo's workflows use; a workflow indented
# otherwise reports "no artifact-manifest job" and fails loudly rather than
# passing quietly, which is the behavior every unparsed shape should have.
#
# Two backstops hold past its edges. A dependency whose *name* says signing
# counts as a signing job even when this parse never saw its definition. And
# this check is defense in depth, not the authority: the expensive
# `Build & Verify Artifacts` chain regenerates the manifest on a machine
# with no key and requires byte equality with the committed one, which no
# amount of clever YAML can talk its way past.
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
      sed -n '2,58p' "${BASH_SOURCE[0]}"
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

# ── 0 + 1 + 2: the manifest document ─────────────────────────────────────
check_manifest() {
  local file="$1" out

  if [[ ! -f "$file" ]]; then
    echo "error: no such manifest: $file" >&2
    exit 2
  fi

  # Before anything reads the document: is there one document to read?
  # JSON permits repeated members, and `jq` — like most parsers — keeps the
  # last. A file with two `artifacts` members would therefore be *checked*
  # as the second while CI *signs* the bytes containing both, and every
  # assertion below would be answering a question about a document nobody
  # published. Python's `json` with `object_pairs_hook` sees the repetition;
  # jq structurally cannot, since duplicates are gone before any filter
  # runs. Python is admissible here for the same reason it is elsewhere in
  # `scripts/`: this reads and reports, and writes no byte anyone ships.
  if ! command -v python3 > /dev/null 2>&1; then
    echo "error: python3 is required (duplicate-member detection)" >&2
    exit 2
  fi
  out="$(
    python3 -c '
import json
import sys

def duplicates(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            print(
                f"duplicate member \"{key}\" — parsers keep one and drop the "
                "other, so the document checked here is not the document "
                "whose bytes get signed"
            )
        seen.add(key)
    return dict(pairs)

try:
    with open(sys.argv[1], "rb") as handle:
        json.load(handle, object_pairs_hook=duplicates)
except ValueError as exc:
    print(f"not parseable as JSON: {exc}")
' "$file"
  )"
  report_lines "$out"
  # A document that does not parse has no shape to check; stop here rather
  # than let every later filter repeat the same complaint in jq's voice.
  if [[ "$out" == *"not parseable as JSON"* ]]; then
    return
  fi

  # The envelope first: the document's own shape. Checked before the
  # entries because a nonce, a run ID or a build timestamp does not have to
  # look like signing material to break determinism — it only has to be a
  # field nobody validated. Key sets are compared for equality at every
  # level, so an added field and a dropped one are both violations, and an
  # empty `artifacts` object is not a vacuous pass.
  out="$(
    jq -r '
      def keyset($who; $want):
        if (keys != ($want | sort)) then
          "\($who): allows exactly \($want | sort | join(", ")) — found \(keys | join(", "))"
        else empty end;
      if type != "object" then
        "manifest: not a JSON object"
      else
        keyset("manifest"; ["artifacts", "enclave", "schema_version"]),
        # A positive integer, per docs/trust-root.md — and specifically what
        # the client parses it as. The verifier reads it with `as_u64()`, so
        # 2.5 or -3 is malformed *there* and reads as a claims change; this
        # gate has to reject what that parser would reject, or a release
        # passes here and alarms every client.
        (if (.schema_version | type) != "number" then
           "manifest.schema_version: not a number"
         elif (.schema_version | floor) != .schema_version then
           "manifest.schema_version: \(.schema_version) is not an integer"
         elif .schema_version < 1 then
           "manifest.schema_version: \(.schema_version) is not a positive integer"
         else empty end),
        (if (.artifacts | type) != "object" then
           "manifest.artifacts: not an object"
         elif (.artifacts | length) == 0 then
           "manifest.artifacts: empty — a manifest that records nothing proves nothing"
         else empty end),
        (if (.enclave | type) != "object" then
           "manifest.enclave: not an object"
         else
           (.enclave | keyset("enclave"; ["cmdline", "snp_measurement", "tdx_measurement"])),
           (if (.enclave.tdx_measurement | type) != "object" then
              "enclave.tdx_measurement: not an object"
            else
              (.enclave.tdx_measurement | keyset("enclave.tdx_measurement"; ["rtmr1", "rtmr2"]))
            end)
         end)
      end
    ' "$file"
  )"
  report_lines "$out"

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
      (if type == "object" then (.artifacts // {}) else {} end)
      | objects
      | to_entries[]
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
      # ── the scanner ─────────────────────────────────────────────────
      # Not a YAML parser. It reads the constructs the assertions need —
      # job keys, `needs:`, `environment:` — in the forms GitHub Actions
      # accepts for them: quoted or bare scalars, flow sequences (on one
      # line or several), block lists, block scalars, trailing comments,
      # and any letter case for an environment name. Values are collected
      # by indentation and normalized once, here, so a new spelling of the
      # same thing is a normalization change rather than a new rule.
      BEGIN { SQ = sprintf("%c", 39) }

      # A `#` opens a comment at line start or after whitespace — and only
      # there, so `a#b` stays the name `a#b`.
      function decomment(s) {
        sub(/^[ \t]*#.*$/, "", s)
        sub(/[ \t]+#.*$/, "", s)
        return s
      }

      function unquote(s,   first, last) {
        gsub(/^[ \t]+/, "", s)
        gsub(/[ \t]+$/, "", s)
        first = substr(s, 1, 1)
        last = substr(s, length(s), 1)
        if ((first == "\"" || first == SQ) && last == first && length(s) > 1)
          s = substr(s, 2, length(s) - 2)
        return s
      }

      # Collected `needs:` text -> the job names in it. Flow punctuation is
      # whitespace; a `-` is a list bullet; `>`/`|` with their chomping and
      # indentation indicators are block-scalar markers, never job names.
      function job_names(s,   n, i, t, out) {
        gsub(/[][,]/, " ", s)
        n = split(s, parts, /[ \t]+/)
        out = ""
        for (i = 1; i <= n; i++) {
          t = unquote(parts[i])
          if (t == "" || t == "-") continue
          if (t ~ /^[>|][-+0-9]*$/) continue
          out = out " " t
        }
        return out
      }

      # ── collect: job name -> needs, environment, produces-partial ──
      /^[^ #]/ { in_jobs = ($0 ~ /^jobs:/); next }

      in_jobs && /^  [^ \t#-][^:]*:[ \t]*(#.*)?$/ {
        job = decomment($0)
        sub(/^  /, "", job)
        sub(/:[ \t]*$/, "", job)
        job = unquote(job)
        jobs[++njobs] = job
        collecting_needs = 0
        collecting_env = 0
        next
      }

      job == "" { next }

      # Any new key at job level closes whatever block was being collected.
      /^    [A-Za-z0-9_-]+:/ { collecting_needs = 0; collecting_env = 0 }

      # `needs:` and `environment:` are both collected as *text*: the value
      # on the key line plus every line indented under it. What form that
      # text took — inline, list, flow sequence, block scalar — is the
      # normalizer'"'"'s problem, not the scanner'"'"'s.
      /^    needs:/ {
        value = decomment($0)
        sub(/^    needs:[ \t]*/, "", value)
        needs_raw[job] = needs_raw[job] " " value
        collecting_needs = 1
        next
      }
      collecting_needs && /^      / {
        needs_raw[job] = needs_raw[job] " " decomment($0)
        next
      }
      /^    environment:/ {
        env = decomment($0)
        sub(/^    environment:[ \t]*/, "", env)
        environment[job] = environment[job] " " env
        collecting_env = 1
        next
      }
      collecting_env && /^      / {
        environment[job] = environment[job] " " decomment($0)
        next
      }

      # A job "produces a manifest partial" if it runs the manifest script
      # or exports the partial as a job output.
      /artifact-manifest\.sh/ || /artifact_manifest:/ { partial[job] = 1 }

      END {
        # A signing job is one granted the Apple signing environment. The
        # name is checked too, so a signing job that has not yet been given
        # its environment (or was renamed) is still caught.
        # Environment names are case-insensitive to GitHub, so they are
        # matched that way here; `Apple-Signing` is the same protected
        # environment as `apple-signing`.
        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          needs[j] = job_names(needs_raw[j])
          if (tolower(environment[j]) ~ /apple-signing/ || tolower(j) ~ /sign/) signer[j] = 1
        }

        found = 0
        for (i = 1; i <= njobs; i++) if (jobs[i] == "artifact-manifest") found = 1
        if (!found)
          print "no `artifact-manifest` job in this workflow — the job-graph assertion cannot be evaluated"

        # Walk *every* ancestor, not just the direct `needs:`. A signing
        # job two hops up hands its outputs down the chain just as well as
        # one hop up, and the intermediate job is under no obligation to
        # drop them.
        head = 1
        tail = 1
        queue[1] = "artifact-manifest"
        seen["artifact-manifest"] = 1
        chain["artifact-manifest"] = "artifact-manifest"
        while (head <= tail) {
          cur = queue[head++]
          n = split(needs[cur], deps, /[ \t]+/)
          for (i = 1; i <= n; i++) {
            d = deps[i]
            if (d == "" || seen[d]) continue
            seen[d] = 1
            chain[d] = chain[cur] " -> " d
            queue[++tail] = d
            # `signer[d]` covers jobs this file defines; the name test
            # also covers a dependency on a job this parse never saw — a
            # reusable workflow, or a job whose key it could not read.
            if (signer[d] || tolower(d) ~ /sign/)
              print "the `artifact-manifest` job depends on signing job `" d "` (" chain[d] ") — a key-dependent value could then reach the manifest"
          }
        }

        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          if (partial[j] && tolower(environment[j]) ~ /apple-signing/)
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
  manifest_fixtures=0
  workflow_fixtures=0

  for fixture in "$FIXTURE_DIR"/bad-*.json; do
    [[ -e "$fixture" ]] || continue
    manifest_fixtures=$((manifest_fixtures + 1))
    if "${BASH_SOURCE[0]}" --manifest "$fixture" > /dev/null 2>&1; then
      echo "  FAIL: $(basename "$fixture") was accepted" >&2
      rc=1
    else
      echo "  rejected: $(basename "$fixture")"
    fi
  done
  for fixture in "$FIXTURE_DIR"/bad-*.yml; do
    [[ -e "$fixture" ]] || continue
    workflow_fixtures=$((workflow_fixtures + 1))
    if "${BASH_SOURCE[0]}" --workflow "$fixture" > /dev/null 2>&1; then
      echo "  FAIL: $(basename "$fixture") was accepted" >&2
      rc=1
    else
      echo "  rejected: $(basename "$fixture")"
    fi
  done

  # Counted per class, because an empty glob is silence: deleting every
  # workflow fixture would otherwise retire half the self-test while the
  # command still reported success.
  if [[ "$manifest_fixtures" -eq 0 ]]; then
    echo "FAIL: no bad-*.json fixture in $FIXTURE_DIR — the document assertions are unproven" >&2
    rc=1
  fi
  if [[ "$workflow_fixtures" -eq 0 ]]; then
    echo "FAIL: no bad-*.yml fixture in $FIXTURE_DIR — the job-graph assertion is unproven" >&2
    rc=1
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "FAIL: the check has lost its teeth — a fixture it must reject passed, or a fixture class is gone." >&2
    exit 1
  fi
  echo "  ($manifest_fixtures manifest fixtures, $workflow_fixtures workflow fixtures)"
fi

echo "OK: no key-dependent material can reach artifact-manifest.json."
