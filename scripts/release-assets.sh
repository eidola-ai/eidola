#!/usr/bin/env bash
# Decide which files a tagged release publishes, and prove each one is the
# file the signed manifest says it is.
#
# `artifact-manifest.json` records a hash for every artifact, and
# `docs/verification.md` promises that hash is checkable with nothing but
# `sha256sum`. That promise is only real if the file is published *and* is
# the file the hash covers. The two are produced by different workflows —
# `artifacts.yml` builds and measures, `tinfoil-build.yml` attaches to the
# tagged release — so identity across that seam is checked here rather than
# assumed. Whatever this prints is what gets uploaded; anything it cannot
# account for fails the release.
#
# The manifest is the authority, not the directory: every row that records a
# user-checkable file hash must have its file present and matching. A file
# in the directory that no row covers is simply not published — it has no
# attested identity, and an unattested asset beside attested ones is worse
# than a missing one.
#
# Which rows are publishable is a table, not a guess:
#
#   * `nix` rows carry `archiveSha256`, the hash of the flake-built
#     `.tar.gz`. Published as `<key>.tar.gz`.
#   * `file` rows naming a Debian package carry the `.deb`'s own `sha256` —
#     the deb *is* the byte stream, with no archive/envelope indirection.
#     Published as `<key>.deb`.
#   * `eidola-gui-macos-universal-zip` is deliberately **not** published.
#     It is the *unsigned* container, recorded so a verifier can reproduce
#     the recipe; the macOS installable is the Developer ID-signed zip,
#     whose hash lives in the human attestation (docs/verification.md).
#   * `oci` rows are pulled from a registry by digest, not downloaded.
#
# Any other row shape is a hard failure. A new artifact class must be a
# decision about publication, made by whoever adds it, rather than an
# asset that quietly never appears.
#
# Published names are derived from the manifest key so a downloaded file
# and the row a user checks it against cannot be mismatched by eye. The
# same rule names the files `scripts/artifact-manifest.sh --artifact-dir`
# writes, which is what makes this comparison a real check rather than a
# rename.
#
# The second command guards the other end of the same path: the publisher
# has to still be waiting when the workflow that builds these files
# finishes. See `wait-budget` below for why that is derived rather than
# declared.
#
# Usage:
#   scripts/release-assets.sh verify --manifest PATH --dir DIR [--list PATH]
#   scripts/release-assets.sh wait-budget --artifacts-workflow PATH \
#       --release-workflow PATH [--job NAME]
#
#   --list PATH   Write the verified files' paths, one per line, to PATH
#                 (they are printed to stdout regardless).
#   --job NAME    The waiting job in the release workflow
#                 (default: publish-assets).

set -euo pipefail

MANIFEST=""
DIR=""
LIST=""
ARTIFACTS_WORKFLOW=""
RELEASE_WORKFLOW=""
PUBLISH_JOB="publish-assets"

usage() {
  sed -n '2,58p' "${BASH_SOURCE[0]}"
}

COMMAND="${1:-}"
case "$COMMAND" in
  verify | wait-budget) shift ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "error: expected the 'verify' or 'wait-budget' command" >&2
    usage >&2
    exit 2
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --dir)
      DIR="$2"
      shift 2
      ;;
    --list)
      LIST="$2"
      shift 2
      ;;
    --artifacts-workflow)
      ARTIFACTS_WORKFLOW="$2"
      shift 2
      ;;
    --release-workflow)
      RELEASE_WORKFLOW="$2"
      shift 2
      ;;
    --job)
      PUBLISH_JOB="$2"
      shift 2
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if ! command -v python3 > /dev/null 2>&1; then
  echo "error: python3 is required" >&2
  exit 2
fi

# ── wait-budget ──────────────────────────────────────────────────────────
# How long may the publisher wait for the workflow that builds what it
# publishes? Long enough for that workflow to finish, which is not a number
# anyone should be maintaining by hand — it is a property of the producer's
# own declared timeouts, and it moves whenever one of them does.
#
# So the publisher declares no clock at all: it polls until the run reaches
# a conclusion, bounded only by its own `timeout-minutes`. This command is
# what keeps that one declared number honest, by computing the producer's
# longest declared path (each job's `timeout-minutes` plus the longest path
# through its `needs`) and requiring the publisher's timeout to exceed it
# with room for queueing. Run on every PR, so a timeout bump in one file
# fails in the other immediately rather than at release time.
#
# Not a YAML parser, and it does not pretend to be: it reads job keys,
# `needs:` and `timeout-minutes:` in the plain forms this repo's workflows
# use, and refuses anything else rather than guessing — the same doctrine
# and the same two-space job indentation `scripts/check-manifest-determinism.sh`
# states. A shape it cannot read fails here, on a PR, where it is cheap.
if [[ "$COMMAND" == "wait-budget" ]]; then
  if [[ -z "$ARTIFACTS_WORKFLOW" || -z "$RELEASE_WORKFLOW" ]]; then
    echo "error: --artifacts-workflow and --release-workflow are required" >&2
    exit 2
  fi
  # shellcheck disable=SC2016 # the grader is Python source, not shell
  python3 -c '
import re
import sys

producer_path, publisher_path, publisher_job = sys.argv[1:4]

# Queue time is real and no `timeout-minutes` covers it: a runner may not be
# free the moment a job becomes eligible. This is the margin for that.
QUEUE_SLACK_MINUTES = 60
# GitHub kills a hosted-runner job at six hours whatever it declares, so a
# larger number is a promise that will not be kept.
HOSTED_JOB_LIMIT_MINUTES = 360
# What a job that declares no timeout gets.
DEFAULT_JOB_TIMEOUT_MINUTES = 360

JOB_KEY = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$")
FIELD = re.compile(r"^    (needs|timeout-minutes):(.*)$")
BLOCK_ITEM = re.compile(r"^      - (.*)$")
UNREADABLE = re.compile(r"[*&\"\x27{}]|\$\{\{")


def die(message):
    print("error: {}".format(message), file=sys.stderr)
    sys.exit(2)


def strip_comment(text):
    text = re.sub(r"^\s*#.*$", "", text)
    return re.sub(r"\s+#.*$", "", text).strip()


def read_jobs(path):
    """{job: (timeout_minutes, [needs])} from a workflow file."""
    jobs = {}
    current = None
    pending = None  # a `needs:` block list still being collected
    in_jobs = False
    for raw in open(path):
        line = raw.rstrip("\n")
        if line.startswith("jobs:"):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if line and not line.startswith(" "):
            break  # left the jobs mapping

        item = BLOCK_ITEM.match(line)
        if pending is not None and item:
            value = strip_comment(item.group(1))
            if UNREADABLE.search(value):
                die("{}: unreadable `needs` entry {!r}".format(path, value))
            jobs[pending][1].append(value)
            continue
        if pending is not None and (JOB_KEY.match(line) or FIELD.match(line)):
            pending = None

        key = JOB_KEY.match(line)
        if key:
            current = key.group(1)
            jobs.setdefault(current, [None, []])
            continue

        field = FIELD.match(line)
        if not field or current is None:
            continue
        name, value = field.group(1), strip_comment(field.group(2))
        if name == "timeout-minutes":
            if not value.isdigit():
                die("{}: job {!r} has a `timeout-minutes` this cannot read: "
                    "{!r}".format(path, current, value))
            jobs[current][0] = int(value)
        else:
            if value == "":
                pending = current  # block list on the following lines
                continue
            if UNREADABLE.search(value.replace("[", "").replace("]", "")):
                die("{}: job {!r} has a `needs` this cannot read: "
                    "{!r}".format(path, current, value))
            jobs[current][1].extend(
                part.strip() for part in value.strip("[]").split(",") if part.strip()
            )
    return {name: (timeout, needs) for name, (timeout, needs) in jobs.items()}


def longest_path(jobs):
    """Minutes until the whole run can still legitimately be running."""
    memo = {}

    def cost(name, seen):
        if name in memo:
            return memo[name]
        if name in seen:
            die("cycle through job {!r}".format(name))
        timeout, needs = jobs.get(name, (None, []))
        if name not in jobs:
            die("job {!r} is needed but not defined".format(name))
        upstream = max(
            [cost(dep, seen | {name}) for dep in needs] or [0]
        )
        memo[name] = (timeout or DEFAULT_JOB_TIMEOUT_MINUTES) + upstream
        return memo[name]

    return max(cost(name, frozenset()) for name in jobs)


producer_jobs = read_jobs(producer_path)
if not producer_jobs:
    die("{}: no jobs found (is the job indentation two spaces?)".format(producer_path))
publisher_jobs = read_jobs(publisher_path)
if publisher_job not in publisher_jobs:
    die("{}: no job {!r}".format(publisher_path, publisher_job))

needed = longest_path(producer_jobs) + QUEUE_SLACK_MINUTES
declared = publisher_jobs[publisher_job][0] or DEFAULT_JOB_TIMEOUT_MINUTES

print("producer longest declared path: {} min (+{} queue slack = {})".format(
    needed - QUEUE_SLACK_MINUTES, QUEUE_SLACK_MINUTES, needed))
print("publisher {!r} timeout-minutes: {}".format(publisher_job, declared))

if declared < needed:
    die(
        "{} job {!r} would give up after {} min, but {} can still be "
        "legitimately running at {} min. Raise its timeout-minutes to at "
        "least {}.".format(
            publisher_path, publisher_job, declared, producer_path, needed, needed
        )
    )
if declared > HOSTED_JOB_LIMIT_MINUTES:
    die(
        "{} job {!r} declares {} min, but GitHub stops a hosted-runner job "
        "at {} min — the extra patience is fiction. Shorten the producer, "
        "or move this job off a hosted runner.".format(
            publisher_path, publisher_job, declared, HOSTED_JOB_LIMIT_MINUTES
        )
    )
print("OK: the publisher outlasts the workflow it waits on.")
' "$ARTIFACTS_WORKFLOW" "$RELEASE_WORKFLOW" "$PUBLISH_JOB"
  exit $?
fi

if [[ -z "$MANIFEST" || -z "$DIR" ]]; then
  echo "error: --manifest and --dir are required" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "error: no such manifest: $MANIFEST" >&2
  exit 2
fi
if [[ ! -d "$DIR" ]]; then
  echo "error: no such directory: $DIR" >&2
  exit 2
fi
# One grader, as in scripts/check-manifest-determinism.sh, and for the same
# reasons: it reads and reports, it hashes with the stdlib, and it is the
# interpreter every platform here already has. It announces that it ran, so
# an empty report cannot be mistaken for a dead script.
out="$(
  # shellcheck disable=SC2016 # the grader is Python source, not shell
  python3 -c '
import hashlib
import json
import os
import sys

manifest_path, directory = sys.argv[1], sys.argv[2]

with open(manifest_path, "rb") as handle:
    manifest = json.load(handle)

problems = []
published = []

NOT_PUBLISHED = {
    "eidola-gui-macos-universal-zip": (
        "the unsigned macOS container; the installable is the Developer "
        "ID-signed zip, whose hash lives in the human attestation"
    ),
}


def expected(name, entry):
    """(filename, `sha256:`-prefixed hash) for a publishable row, or None."""
    kind = entry.get("type")
    if kind == "oci":
        return None
    if name in NOT_PUBLISHED:
        return None
    if kind == "nix":
        return name + ".tar.gz", entry.get("archiveSha256")
    if kind == "file" and "-deb-" in name:
        return name + ".deb", entry.get("sha256")
    problems.append(
        "artifacts.{}: type {} is not in the publication table — a new "
        "artifact class needs a deliberate decision about whether a "
        "release publishes it (scripts/release-assets.sh)".format(
            name, json.dumps(kind)
        )
    )
    return None


artifacts = manifest.get("artifacts")
if not isinstance(artifacts, dict) or not artifacts:
    problems.append("manifest records no artifacts")
    artifacts = {}

for name in sorted(artifacts):
    entry = artifacts[name]
    if not isinstance(entry, dict):
        problems.append("artifacts.{}: not an object".format(name))
        continue
    plan = expected(name, entry)
    if plan is None:
        continue
    filename, want = plan
    if not isinstance(want, str) or not want.startswith("sha256:"):
        problems.append(
            "artifacts.{}: no sha256 to check the published file "
            "against".format(name)
        )
        continue
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        problems.append(
            "artifacts.{}: the manifest records this artifact, but {} is "
            "not in the release inputs — the signed manifest would point "
            "at a file nobody can download".format(name, filename)
        )
        continue
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    got = "sha256:" + digest.hexdigest()
    if got != want:
        problems.append(
            "artifacts.{}: {} hashes to {}, but the manifest records {} — "
            "the file built is not the file measured".format(
                name, filename, got, want
            )
        )
        continue
    published.append(path)

for line in problems:
    print("PROBLEM " + line)
for path in published:
    print("PUBLISH " + path)
print("@graded")
' "$MANIFEST" "$DIR"
)"

if [[ "$out" != *"@graded"* ]]; then
  echo "error: the release-asset grader did not run to completion" >&2
  exit 2
fi

problems=0
files=()
while IFS= read -r line; do
  case "$line" in
    "PROBLEM "*)
      echo "  ${line#PROBLEM }" >&2
      problems=$((problems + 1))
      ;;
    "PUBLISH "*) files+=("${line#PUBLISH }") ;;
  esac
done <<< "$out"

if [[ "$problems" -gt 0 ]]; then
  echo "error: $problems release asset(s) could not be verified against $MANIFEST" >&2
  exit 1
fi
if [[ "${#files[@]}" -eq 0 ]]; then
  echo "error: no release assets to publish — the manifest records none, which cannot be right" >&2
  exit 1
fi

printf '%s\n' "${files[@]}"
if [[ -n "$LIST" ]]; then
  printf '%s\n' "${files[@]}" > "$LIST"
fi
echo "Verified ${#files[@]} release asset(s) against $MANIFEST." >&2
