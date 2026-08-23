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
#   0. The file is one document, and one every reader agrees on: no
#      repeated members (parsers keep the last and drop the rest, so a file
#      with two `artifacts` members would be *checked* as one reading and
#      *signed* as the whole bytes) and no `NaN`/`Infinity` (Python accepts
#      them, JSON has no such literal, and the client's parser refuses the
#      whole document). Nothing below means anything until this holds.
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
# The manifest half is read by python3 (stdlib only, read-only) rather than
# jq: `schema_version` is a u64 to the client, jq numbers are doubles, and
# repeated members are resolved before any filter runs — both are checks a
# filter language cannot express. It is also the interpreter every platform
# here already ships, so this gate adds nothing to a fresh checkout's
# prerequisites.
#
# Assertion 3's scanner is built as two pipelines rather than a set of
# patterns, because patterns compose badly: every scalar it reads — a job
# name, a `needs:` entry, an `environment:` name, however it was spelled and
# whichever extraction path it arrived by — passes one classifier before
# anything is matched against it, and every job body passes one header
# reader (name, then anchor, then body form) before its shape is decided.
# A pairing nobody anticipated meets the refusal its parts would have met
# individually, instead of falling between two rules. Both graders also
# announce that they ran: an empty report has to be distinguishable from a
# scanner that died, and this awk exits 0 on a syntax error on some
# platforms.
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
# What it will not do is guess. A YAML alias or anchor in one of those
# fields, an escape inside a quoted scalar there, a merge key in a job
# body, or a job body written as a flow mapping is a violation rather than
# a resolution attempt: GitHub resolves aliases (since September 2025), this
# does not, and a scanner that guessed would be asserting something it
# cannot know. The same answer covers the two other things it cannot read.
# An `environment:` computed by a `${{ }}` expression is refused on any job
# that can reach the manifest — GitHub evaluates those, this does not —
# except expressions pinned in the check as read and shown harmless, which
# is what makes adding one a review moment. And a job-level `uses:` in the
# manifest job's ancestry is refused, because the called workflow's
# environments and outputs are in a file this is not reading; nothing here
# uses one today, and following calls is a real parser's job.
#
# `needs:` takes no expressions — it is absent from GitHub's
# context-availability table, since the dependency graph must be known
# before any context exists — so there is nothing to refuse there.
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

  # One grader, one language. The document is read by Python rather than
  # jq for a reason the checks below depend on: `schema_version` is a u64
  # to the client (`as_u64()`), and jq numbers are IEEE-754 doubles, so
  # `1e100` and `2^64` are integral there and out of range here — the gate
  # would pass a manifest every client reads as malformed. Python integers
  # are arbitrary precision, so the range check is exact. It also sees
  # repeated members, which no filter language can: duplicates are resolved
  # before a filter ever runs. Python is admissible for the same reason it
  # is elsewhere in `scripts/`: this reads and reports, and writes no byte
  # anyone ships. (It is also the interpreter every platform here already
  # has, which keeps `just check` from needing anything extra.)
  if ! command -v python3 > /dev/null 2>&1; then
    echo "error: python3 is required (manifest validation)" >&2
    exit 2
  fi
  if ! out="$(
    python3 -c '
import json
import re
import sys

U64_MAX = 2 ** 64 - 1
SIGNING = re.compile("sign|notar|ticket|team|cert|staple|apple", re.IGNORECASE)
ALLOWED = {
    "oci": ["digest", "platform", "type"],
    "nix": ["archiveSha256", "narHash", "platform", "type"],
    "file": ["platform", "sha256", "type"],
}

violations = []
duplicates = []
constants = []


def note_constant(text):
    # NaN / Infinity / -Infinity: Python accepts them, JSON does not have
    # them, and the client parses these bytes with serde_json, which
    # refuses the whole document. Same theme as duplicate members — the
    # thing checked here would not be the thing anyone can read.
    constants.append(text)
    return 0.0


def note_duplicates(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return dict(pairs)


def keyset(who, obj, want):
    if sorted(obj) != sorted(want):
        violations.append(
            "{}: allows exactly {} — found {}".format(
                who, ", ".join(sorted(want)), ", ".join(sorted(obj)) or "nothing"
            )
        )


def walk_keys(node, out):
    if isinstance(node, dict):
        for key, value in node.items():
            out.add(key)
            walk_keys(value, out)
    elif isinstance(node, list):
        for item in node:
            walk_keys(item, out)


try:
    with open(sys.argv[1], "rb") as handle:
        manifest = json.load(
            handle, object_pairs_hook=note_duplicates, parse_constant=note_constant
        )
except ValueError as exc:
    print("not parseable as JSON: {}".format(exc))
    sys.exit(0)

for text in constants:
    print(
        "non-standard JSON constant `{}` — JSON has no such literal, and the "
        "client reads these bytes with a parser that refuses the whole "
        "document".format(text)
    )

for key in duplicates:
    print(
        "duplicate member \"{}\" — parsers keep one and drop the other, so "
        "the document checked here is not the document whose bytes get "
        "signed".format(key)
    )

if not isinstance(manifest, dict):
    print("manifest: not a JSON object")
    sys.exit(0)

# The envelope: a nonce, a run ID or a build timestamp does not have to look
# like signing material to break determinism — it only has to be a field
# nobody validated. Key sets are equalities at every level, so an added
# field and a dropped one are both violations.
keyset("manifest", manifest, ["artifacts", "enclave", "schema_version"])

version = manifest.get("schema_version")
if isinstance(version, bool) or not isinstance(version, int):
    violations.append(
        "manifest.schema_version: {} is not an integer".format(json.dumps(version))
    )
elif version < 1:
    violations.append(
        "manifest.schema_version: {} is not a positive integer".format(version)
    )
elif version > U64_MAX:
    violations.append(
        "manifest.schema_version: {} exceeds the u64 the client parses it "
        "as".format(version)
    )

artifacts = manifest.get("artifacts")
if not isinstance(artifacts, dict):
    violations.append("manifest.artifacts: not an object")
    artifacts = {}
elif not artifacts:
    violations.append(
        "manifest.artifacts: empty — a manifest that records nothing proves nothing"
    )

enclave = manifest.get("enclave")
if not isinstance(enclave, dict):
    violations.append("manifest.enclave: not an object")
else:
    keyset("enclave", enclave, ["cmdline", "snp_measurement", "tdx_measurement"])
    tdx = enclave.get("tdx_measurement")
    if not isinstance(tdx, dict):
        violations.append("enclave.tdx_measurement: not an object")
    else:
        keyset("enclave.tdx_measurement", tdx, ["rtmr1", "rtmr2"])

# Field allow-list per artifact type. `oci` records the image digest; `nix`
# records the store-path checkpoint plus the archive hash a user can check
# without Nix; `file` records the sha256 of one published file.
for name, entry in artifacts.items():
    if not isinstance(entry, dict):
        violations.append("artifacts.{}: not an object".format(name))
        continue
    kind = entry.get("type")
    if kind not in ALLOWED:
        violations.append(
            "artifacts.{}: unknown type {} (allowed: oci, nix, file)".format(
                name, json.dumps(kind)
            )
        )
        continue
    if sorted(entry) != sorted(ALLOWED[kind]):
        violations.append(
            "artifacts.{}: type \"{}\" allows exactly {} — found {}".format(
                name, kind, ", ".join(sorted(ALLOWED[kind])), ", ".join(sorted(entry))
            )
        )

# No key anywhere may name signing material, at any depth. Artifact names
# are keys too, so `eidola-gui-macos-signed-zip` trips this as readily as a
# `team_id` field would.
keys = set()
walk_keys(manifest, keys)
for key in sorted(keys):
    if SIGNING.search(key):
        violations.append(
            "key \"{}\" names signing material — Apple envelope hashes and "
            "identities belong in the human attestation, not the "
            "manifest".format(key)
        )

for line in violations:
    print(line)

# Same liveness marker as the workflow scanner, for the same reason.
print("@graded")
' "$file"
  )"; then
    echo "error: the manifest grader failed to run over $file" >&2
    exit 2
  fi
  if [[ "$out" != *"@graded"* ]]; then
    echo "error: the manifest grader did not run to completion over $file" >&2
    exit 2
  fi
  report_lines "$(printf '%s\n' "$out" | grep -v '^@graded$')"
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

  # A scanner that fails to run prints nothing, and nothing is what a clean
  # file prints too. The status is checked so a broken scanner fails the
  # gate instead of passing it — the one failure mode a check like this
  # must not have.
  if ! out="$(
    awk '
      # ── the scanner ─────────────────────────────────────────────────
      # Not a YAML parser. It reads the constructs the assertions need —
      # job keys, `needs:`, `environment:` — in the forms GitHub Actions
      # accepts for them, and refuses the ones it cannot read.
      #
      # It is a pipeline rather than a set of patterns, because patterns
      # compose badly: every scalar reaches `classify_scalar` before
      # anything is matched against it, and every job body reaches the same
      # header reader before its form is classified. A construct arriving
      # by a path nobody anticipated — an escape inside a mapping, an
      # anchor in front of a flow body — meets the same refusal as the form
      # that was anticipated, because there is one place where a value
      # becomes readable and one place where a body becomes a form.
      BEGIN {
        SQ = sprintf("%c", 39)
        # `environment:` accepts expressions (GitHub'"'"'s context-availability
        # table lists it), and this scanner evaluates none. Rather than
        # guess what one resolves to, expressions are refused on the jobs
        # that could feed the manifest — except those pinned here, which
        # have been read and shown to produce no signing environment.
        # Adding one is deliberately a review moment: it widens the set of
        # environments a manifest-producing job may receive.
        PERMITTED_ENV[1] = "${{ (github.event_name == " SQ "push" SQ \
          " && github.ref == " SQ "refs/heads/main" SQ ") && " \
          SQ "cachix-write" SQ " || " SQ SQ " }}"
        N_PERMITTED_ENV = 1
      }

      function normalize_ws(s) {
        gsub(/[ \t]+/, " ", s)
        gsub(/^ | $/, "", s)
        return s
      }

      function permitted_env(s,   i) {
        s = normalize_ws(s)
        for (i = 1; i <= N_PERMITTED_ENV; i++)
          if (s == PERMITTED_ENV[i]) return 1
        return 0
      }

      # A `#` opens a comment at line start or after whitespace — and only
      # there, so `a#b` stays the name `a#b`.
      function decomment(s) {
        sub(/^[ \t]*#.*$/, "", s)
        sub(/[ \t]+#.*$/, "", s)
        return s
      }

      # ── the one gate every scalar passes ────────────────────────────
      # Returns "" for a scalar this scanner can read, leaving its plain
      # text in SCALAR; otherwise the reason it cannot. An alias or anchor
      # names something defined elsewhere; an escape means GitHub decodes a
      # spelling this does not. Neither is guessed at.
      function classify_scalar(tok,   first, last) {
        tok = normalize_ws(tok)
        SCALAR = ""
        if (tok == "") return ""
        first = substr(tok, 1, 1)
        if (first == "*" || first == "&") return "a YAML alias or anchor"
        if (first == "\"" && index(tok, "\\") > 0)
          return "an escape inside a quoted scalar"
        if (first == SQ && index(substr(tok, 2, length(tok) - 2), SQ SQ) > 0)
          return "an escape inside a quoted scalar"
        last = substr(tok, length(tok), 1)
        if ((first == "\"" || first == SQ) && last == first && length(tok) > 1)
          tok = substr(tok, 2, length(tok) - 2)
        SCALAR = tok
        return ""
      }

      # ── the one tokenizer every collected value passes ──────────────
      # Splits a value into scalars, keeping a quoted scalar whole even
      # when it holds spaces or flow punctuation. Fills TOK[1..NTOK] and
      # returns 0, or 1 if a quote never closed — itself a value this
      # scanner cannot read.
      function tokenize(s,   i, c, n, cur, q) {
        NTOK = 0
        n = length(s)
        cur = ""
        q = ""
        for (i = 1; i <= n; i++) {
          c = substr(s, i, 1)
          if (q != "") {
            cur = cur c
            if (c == q) {
              # A doubled quote inside a single-quoted scalar is YAML'"'"'s one
              # escape, not the end of the scalar. Closing the token here
              # would split one scalar into two and lose whatever it named
              # — so the run stays whole, and the classifier refuses it for
              # carrying an escape.
              if (q == SQ && substr(s, i + 1, 1) == SQ) { cur = cur SQ; i++; continue }
              TOK[++NTOK] = cur
              cur = ""
              q = ""
            }
            continue
          }
          if (c == "\"" || c == SQ) {
            if (cur != "") TOK[++NTOK] = cur
            cur = c
            q = c
            continue
          }
          if (c == " " || c == "\t" || c == "[" || c == "]" || c == "{" || c == "}" || c == ",") {
            if (cur != "") { TOK[++NTOK] = cur; cur = "" }
            continue
          }
          cur = cur c
        }
        if (q != "") return 1
        if (cur != "") TOK[++NTOK] = cur
        return 0
      }

      # ── one reader for every collected field ────────────────────────
      # `needs:` and `environment:` are collected as text — the value on
      # the key line plus everything indented under it — and become
      # readable only here. Which spelling the text used (inline, list,
      # flow sequence, block scalar, nested mapping) is the tokenizer'"'"'s
      # problem; whether each scalar can be read at all is the
      # classifier'"'"'s; nothing downstream sees anything else.
      function read_field(jb, field, blob,   i, t, reason, out) {
        blob = normalize_ws(blob)
        if (blob == "") return
        if (index(blob, "${{") > 0) { expression[jb, field] = 1; return }
        if (tokenize(blob) != 0) {
          refusal[jb, field] = "an unterminated quoted scalar"
          return
        }
        out = ""
        for (i = 1; i <= NTOK; i++) {
          t = TOK[i]
          # Structure, not scalars: list bullets, block-scalar markers with
          # their chomping and indentation indicators, and the keys of a
          # nested mapping (`name:`, `url:`) — a job ID cannot hold a
          # colon, so a trailing one always marks a key.
          if (t == "-") continue
          if (t ~ /^[>|][-+0-9]*$/) continue
          if (t ~ /:$/) continue
          reason = classify_scalar(t)
          if (reason != "") { refusal[jb, field] = reason; continue }
          if (SCALAR != "") out = out " " SCALAR
        }
        values[jb, field] = out
      }

      # Reads a mapping key up to its first colon outside quotes, so a
      # quoted key holding one stays whole. Sets KEY and REST.
      function split_key(line,   i, c, n, q) {
        n = length(line)
        q = ""
        for (i = 1; i <= n; i++) {
          c = substr(line, i, 1)
          if (q != "") { if (c == q) q = ""; continue }
          if (c == "\"" || c == SQ) { q = c; continue }
          if (c == ":") { KEY = substr(line, 1, i - 1); REST = substr(line, i + 1); return 1 }
        }
        return 0
      }

      # ── collect ─────────────────────────────────────────────────────
      /^[^ #]/ { in_jobs = ($0 ~ /^jobs:/); next }

      # One job-header reader. The name goes through the scalar gate; the
      # body form is classified only after any anchor, quoting and comment
      # have been taken off, so anchor-plus-flow and quote-plus-flow are
      # handled by the order of these steps rather than by a pattern for
      # each pairing.
      in_jobs && /^  [^ \t#]/ {
        line = decomment($0)
        sub(/^  /, "", line)
        if (!split_key(line)) next
        reason = classify_scalar(KEY)
        job = (reason == "" ? SCALAR : normalize_ws(KEY))
        jobs[++njobs] = job
        if (reason != "") refusal[job, "name"] = reason
        rest = normalize_ws(REST)
        # An anchor names the body; it does not hide it. Taken off before
        # the form is read, so an anchored body is read like any other.
        sub(/^&[A-Za-z0-9_.-]+[ \t]*/, "", rest)
        if (rest == "") body[job] = "block"
        else if (substr(rest, 1, 1) == "{") body[job] = "flow"
        else { body[job] = "other"; body_text[job] = rest }
        collecting_needs = 0
        collecting_env = 0
        next
      }

      job == "" { next }

      # One reader for job-level keys, the same one the header uses. YAML
      # allows whitespace before a colon and quotes around a key, so the
      # key is read rather than pattern-matched — `needs :`, `"needs":` and
      # `needs:` are one key, and any job-level key at all closes whatever
      # block was being collected.
      /^    [^ \t#]/ {
        line = decomment($0)
        sub(/^    /, "", line)
        if (split_key(line)) {
          key = KEY
          if (classify_scalar(key) == "") key = SCALAR
          else key = normalize_ws(key)
          value = normalize_ws(REST)
          collecting_needs = 0
          collecting_env = 0
          if (key == "needs") {
            raw[job, "needs"] = raw[job, "needs"] " " value
            collecting_needs = 1
          } else if (key == "environment") {
            raw[job, "environment"] = raw[job, "environment"] " " value
            collecting_env = 1
          } else if (key == "uses") {
            raw[job, "uses"] = value
          } else if (key == "<<") {
            # A merge key splices another mapping into this job, which can
            # carry an `environment:` that never appears here as a key.
            # Recorded, not resolved.
            merge_key[job] = 1
          }
          next
        }
      }

      collecting_needs && /^      / {
        raw[job, "needs"] = raw[job, "needs"] " " decomment($0)
        next
      }
      collecting_env && /^      / {
        raw[job, "environment"] = raw[job, "environment"] " " decomment($0)
        next
      }

      # A job "produces a manifest partial" if it runs the manifest script
      # or exports the partial as a job output.
      /artifact-manifest\.sh/ || /artifact_manifest:/ { partial[job] = 1 }

      END {
        # Everything collected becomes readable here, through the one
        # reader, before a single conclusion is drawn from it.
        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          read_field(j, "needs", raw[j, "needs"])
          read_field(j, "environment", raw[j, "environment"])
          read_field(j, "uses", raw[j, "uses"])
        }

        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          if (refusal[j, "name"] != "")
            print "job `" j "` writes its name with " refusal[j, "name"] " — this scanner resolves and decodes nothing, so the name it compares would not be the name GitHub reads"
          if (refusal[j, "needs"] != "")
            print "job `" j "` writes a `needs:` entry with " refusal[j, "needs"] " — this scanner resolves and decodes nothing, so the dependency must be written plainly"
          if (refusal[j, "environment"] != "")
            print "job `" j "` writes its `environment:` with " refusal[j, "environment"] " — this scanner resolves and decodes nothing, so the environment must be written plainly"
          if (merge_key[j])
            print "job `" j "` uses a merge key (`<<:`) — it can splice in an `environment:` this scanner never sees, so it is refused rather than merged"
        }

        # A signing job is one granted the Apple signing environment.
        # Environment names are case-insensitive to GitHub, so they are
        # matched that way here. The job name is checked too, so a signing
        # job not yet given its environment (or renamed) is still caught.
        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          needs[j] = values[j, "needs"]
          if (tolower(values[j, "environment"]) ~ /apple-signing/ || tolower(j) ~ /sign/) signer[j] = 1
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
          if (partial[j] && tolower(values[j, "environment"]) ~ /apple-signing/)
            print "job `" j "` produces a manifest partial and is granted the apple-signing environment"
        }

        # Scoped to the jobs whose body could matter: an ancestor of the
        # manifest job (seen by the walk above) or a job computing part of
        # it. An unrelated job may be written however it likes.
        for (i = 1; i <= njobs; i++) {
          j = jobs[i]
          if (!seen[j] && !partial[j]) continue
          if (expression[j, "environment"] && !permitted_env(raw[j, "environment"]))
            print "job `" j "` computes its `environment:` with an expression this check does not evaluate — the environment a job feeding the manifest receives has to be readable from the file, so pin the expression in the check once it is shown it cannot produce a signing environment"
          if (!seen[j]) continue
          if (body[j] == "flow")
            print "job `" j "` is written as a flow mapping and the manifest job depends on it — this scanner reads block-form job bodies, so its environment and outputs are unread"
          if (body[j] == "other")
            print "job `" j "` has a body this scanner cannot read (`" body_text[j] "`) and the manifest job depends on it"
          if (values[j, "uses"] != "" || expression[j, "uses"] || refusal[j, "uses"] != "")
            print "job `" j "` calls another workflow (`uses:" values[j, "uses"] "`) and the manifest job depends on it — this scanner does not follow calls, so the called workflow'"'"'s environments and outputs are unread"
        }
        # Announced last so the caller can tell "nothing to report" from
        # "never ran": this awk exits 0 on a syntax error on some
        # platforms, and an empty report is exactly what a clean file
        # produces.
        print "@scanned " njobs " jobs"
      }
    ' "$file"
  )"; then
    echo "error: the workflow scanner failed to run over $file" >&2
    exit 2
  fi
  if [[ "$out" != *"@scanned "* ]]; then
    echo "error: the workflow scanner did not run to completion over $file" >&2
    exit 2
  fi
  report_lines "$(printf '%s\n' "$out" | grep -v '^@scanned ')"
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
