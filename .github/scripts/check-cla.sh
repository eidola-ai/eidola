#!/usr/bin/env bash
# CLA verification script.
# Validates that every commit author/committer in a PR has signed the
# current version of the CLA, using CLA-SIGNERS.txt as the source of truth.
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────

CLA_INDIVIDUAL="CLA-INDIVIDUAL.md"
CLA_CORPORATE="CLA-CORPORATE.md"
SIGNERS_FILE="CLA-SIGNERS.txt"

BASE_REF="${1:-main}"

# Emails that bypass the CLA check entirely.
EXEMPT_BOTS="
dependabot[bot]@users.noreply.github.com
github-actions[bot]@users.noreply.github.com
41898282+github-actions[bot]@users.noreply.github.com
noreply@github.com
"

# ── Helpers ────────────────────────────────────────────────────────────────────

die() { printf '\033[1;31mError:\033[0m %s\n' "$1" >&2; exit 1; }
info() { printf '\033[1;34m▸\033[0m %s\n' "$1"; }
pass() { printf '\033[1;32m✓\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31m✗\033[0m %s\n' "$1"; }

sha256_file() {
  # Portable: works on both macOS (shasum) and Linux (sha256sum).
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

is_bot() {
  echo "$EXEMPT_BOTS" | grep -qFx "$1"
}

# ── Compute current CLA hashes (from the proposed branch state) ────────────────

[[ -f "$CLA_INDIVIDUAL" ]] || die "$CLA_INDIVIDUAL not found"
[[ -f "$CLA_CORPORATE" ]]  || die "$CLA_CORPORATE not found"
[[ -f "$SIGNERS_FILE" ]]   || die "$SIGNERS_FILE not found"

HASH_INDIVIDUAL="$(sha256_file "$CLA_INDIVIDUAL")"
HASH_CORPORATE="$(sha256_file "$CLA_CORPORATE")"

info "CLA-INDIVIDUAL hash: ${HASH_INDIVIDUAL:0:16}…"
info "CLA-CORPORATE  hash: ${HASH_CORPORATE:0:16}…"

# ── Extract unique commit emails ───────────────────────────────────────────────

MERGE_BASE="$(git merge-base "$BASE_REF" HEAD)"

EMAILS="$(git log "$MERGE_BASE"..HEAD --format='%ae%n%ce' | tr '[:upper:]' '[:lower:]' | sort -uf)"

if [[ -z "$EMAILS" ]]; then
  info "No commits found in diff — nothing to check."
  exit 0
fi

info "Emails to verify: $(echo $EMAILS | tr '\n' ' ')"

# ── Pre-filter CLA-SIGNERS.txt into individual and corporate lists ─────────────
# Strip comments/blanks, then split by hash match.

ACTIVE_ENTRIES="$(grep -v '^\s*#' "$SIGNERS_FILE" | grep -v '^\s*$' || true)"

# Individual: lines whose hash matches CLA-INDIVIDUAL, field 2 is the pattern.
INDIVIDUAL_PATTERNS="$(echo "$ACTIVE_ENTRIES" | awk -v h="$HASH_INDIVIDUAL" '$1 == h {print tolower($2)}' || true)"

# Corporate: lines whose hash matches CLA-CORPORATE, field 2 is the @domain pattern.
CORPORATE_PATTERNS="$(echo "$ACTIVE_ENTRIES" | awk -v h="$HASH_CORPORATE" '$1 == h {print tolower($2)}' || true)"

# ── Validate each email ───────────────────────────────────────────────────────

FAILED=0
MISSING_EMAILS=""

for email in $EMAILS; do
  # Bot exemption
  if is_bot "$email"; then
    pass "$email (bot — exempt)"
    continue
  fi

  # Exact individual match
  if echo "$INDIVIDUAL_PATTERNS" | grep -qFx "$email"; then
    pass "$email (individual CLA)"
    continue
  fi

  # Corporate domain match: extract @domain and check
  domain="@${email#*@}"
  if echo "$CORPORATE_PATTERNS" | grep -qFx "$domain"; then
    pass "$email (corporate CLA via $domain)"
    continue
  fi

  # No match — record failure
  fail "$email — CLA signature not found"
  MISSING_EMAILS="$MISSING_EMAILS $email"
  FAILED=1
done

# ── Report ─────────────────────────────────────────────────────────────────────

if [[ $FAILED -ne 0 ]]; then
  echo ""
  echo "══════════════════════════════════════════════════════════════"
  echo " CLA CHECK FAILED"
  echo "══════════════════════════════════════════════════════════════"
  echo ""
  echo "The following email(s) have not signed the current CLA:"
  echo ""
  for email in $MISSING_EMAILS; do
    echo "  $email"
  done
  echo ""
  echo "To sign, add the following line(s) to $SIGNERS_FILE in this"
  echo "branch and push a new commit:"
  echo ""
  for email in $MISSING_EMAILS; do
    echo "  ${HASH_INDIVIDUAL} ${email} ${email} Your Name"
  done
  echo ""
  echo "Format: <cla-hash> <git-email-pattern> <signer-email> <signer-name>"
  echo "See CLA-INDIVIDUAL.md (or CLA-CORPORATE.md for organizations)."
  echo "══════════════════════════════════════════════════════════════"
  exit 1
fi

echo ""
pass "All commit emails have valid CLA signatures."
