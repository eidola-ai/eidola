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
49699333+dependabot[bot]@users.noreply.github.com
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
HASH_INDIVIDUAL="${HASH_INDIVIDUAL:0:8}"
HASH_CORPORATE="$(sha256_file "$CLA_CORPORATE")"
HASH_CORPORATE="${HASH_CORPORATE:0:8}"

info "CLA-INDIVIDUAL hash: ${HASH_INDIVIDUAL}"
info "CLA-CORPORATE  hash: ${HASH_CORPORATE}"

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

# ── PR comment helper ─────────────────────────────────────────────────────────

COMMENT_MARKER="<!-- cla-check -->"

post_pr_comment() {
  # Only post when running in a GitHub Actions PR context.
  if [[ -z "${PR_NUMBER:-}" || -z "${GITHUB_REPOSITORY:-}" ]]; then
    return
  fi

  local body="$1"
  body="${COMMENT_MARKER}
${body}"

  # Minimize any previous CLA comments so the latest result is prominent.
  local old_ids
  old_ids="$(gh api "repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
    --jq "[.[] | select(.body | startswith(\"${COMMENT_MARKER}\")) | .id] | .[]" \
    2>/dev/null || true)"

  for id in $old_ids; do
    # Minimize via GraphQL — the REST API doesn't support hiding comments.
    local node_id
    node_id="$(gh api "repos/${GITHUB_REPOSITORY}/issues/comments/${id}" \
      --jq '.node_id' 2>/dev/null || true)"
    if [[ -n "$node_id" ]]; then
      gh api graphql -f query='
        mutation($id: ID!) {
          minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) {
            clientMutationId
          }
        }' -f id="$node_id" >/dev/null 2>&1 || true
    fi
  done

  gh pr comment "$PR_NUMBER" --repo "$GITHUB_REPOSITORY" --body "$body" >/dev/null 2>&1 || true
}

# ── Report ─────────────────────────────────────────────────────────────────────

if [[ $FAILED -ne 0 ]]; then
  MESSAGE="## CLA Check Failed

The following commit email(s) have not signed the current CLA:

$(for email in $MISSING_EMAILS; do echo "- \`$email\`"; done)

**To sign the CLA as an individual,** add a line to \`$SIGNERS_FILE\` in this branch and push:

\`\`\`
$(for email in $MISSING_EMAILS; do echo "${HASH_INDIVIDUAL} ${email} ${email} Your Name"; done)
\`\`\`

**To sign the CLA for your company,** add a line with your corporate domain instead:

\`\`\`
$(for email in $MISSING_EMAILS; do domain="@${email#*@}"; echo "${HASH_CORPORATE} ${domain} ${email} Your Name | Company Name"; done)
\`\`\`

See [CLA-INDIVIDUAL.md](../blob/main/CLA-INDIVIDUAL.md) or [CLA-CORPORATE.md](../blob/main/CLA-CORPORATE.md) for full details."

  echo "$MESSAGE"
  post_pr_comment "$MESSAGE"
  exit 1
fi

MESSAGE="## CLA Check Passed

All commit emails have valid CLA signatures."

echo "$MESSAGE"
post_pr_comment "$MESSAGE"
