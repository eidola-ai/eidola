#!/usr/bin/env bash
#
# Rekor SSH-format `rekord` smoke test.
#
# Purpose
# -------
# Validate the protocol assumptions in the human attestation flow:
# that rekor.sigstore.dev accepts an SSH-signed `rekord` v0.0.1 entry
# (with `signature.format=ssh`), and that the response shape matches
# what `crates/eidola-app-core/src/updater/human_attestation.rs`
# expects to parse — `body`, `logIndex`, `logID`, `integratedTime`,
# `verification.signedEntryTimestamp`, and `verification.inclusionProof`.
#
# History — why `rekord` and not `hashedrekord`
# ---------------------------------------------
# An earlier version of this script attempted a `hashedrekord` v0.0.1
# entry with an OpenSSH public key. That doesn't work: Rekor's
# `hashedrekord` schema hardcodes x509 PKI (PEM-encoded leaf cert).
# SSH-format keys are routed through the polymorphic `rekord` entry
# with `signature.format=ssh`. The mistake is preserved here as a
# lesson: SSH lives in `rekord`, not `hashedrekord`.
#
# Two further forced choices documented inline below:
#   - Rekor REJECTS hash-only data on rekord+ssh (`missing data content`)
#     — `data.content` (full bytes, base64) is required at POST time.
#     Rekor then canonicalizes it down to `data.hash` in the persisted
#     entry, so the public log record is still hash-only.
#   - Rekor REJECTS any SSH-SIG namespace other than `"file"`
#     (`invalid signature namespace: <ns>`). The signature must be
#     produced with `ssh-keygen -Y sign -n file`; this matches the
#     hardcoded constant in `release-tool`'s `attest.rs` and in
#     `eidola-app-core`'s `human_attestation::SSH_SIG_NAMESPACE`.
#
# WARNING — public transparency log
# ---------------------------------
# Rekor entries are *immutable and publicly visible*. Running this script
# (without --dry-run) publishes a synthetic entry to the production
# Sigstore log under a freshly-generated throwaway test key. The entry
# itself is harmless, but the entry will live in the public Merkle tree
# forever and the fact that this project ran a smoke test on a given day
# becomes part of public record. The key never touches any hardware or
# long-lived secret — it's generated, used, and discarded in this run.
#
# Usage
# -----
#   scripts/rekor-smoke-test.sh --dry-run   # build the entry, print it, do NOT POST
#   scripts/rekor-smoke-test.sh             # actually POST to rekor (irreversible)
#
# Environment overrides
# ---------------------
#   REKOR_URL   — defaults to https://rekor.sigstore.dev
#
# Requires: ssh-keygen, curl, jq, base64, mktemp, and one of
#           sha256sum / shasum.

set -euo pipefail

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  "") ;;
  *) echo "unknown arg: $1 (use --dry-run or no args)" >&2; exit 2 ;;
esac

REKOR_URL="${REKOR_URL:-https://rekor.sigstore.dev}"
# Rekor's SSH PKI verifier pins the SSH-SIG namespace to "file"
# (sigstore/rekor `pkg/pki/ssh/sign.go`). Any other namespace is
# rejected at POST time with HTTP 400.
NAMESPACE="file"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
echo "working dir: $WORK_DIR"

# ---------------------------------------------------------------------------
# 1. Throwaway ed25519 keypair (NOT a hardware key; discarded at exit).
# ---------------------------------------------------------------------------
KEY="$WORK_DIR/smoke_key"
ssh-keygen -t ed25519 -N "" \
  -C "eidola-rekor-smoke-test $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -f "$KEY" >/dev/null
FPR="$(ssh-keygen -lf "$KEY.pub" | awk '{print $2}')"
echo "✓ throwaway ed25519 key fingerprint: $FPR"

# ---------------------------------------------------------------------------
# 2. Synthetic payload + SHA-256.
# ---------------------------------------------------------------------------
PAYLOAD="$WORK_DIR/payload.txt"
printf 'eidola rekor smoke test: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$PAYLOAD"
if command -v sha256sum >/dev/null 2>&1; then
  DIGEST="$(sha256sum "$PAYLOAD" | awk '{print $1}')"
else
  DIGEST="$(shasum -a 256 "$PAYLOAD" | awk '{print $1}')"
fi
echo "✓ payload sha256: $DIGEST"

# ---------------------------------------------------------------------------
# 3. Sign with `ssh-keygen -Y sign -n file`. The namespace is forced by
#    Rekor (see header); the release-tool's `attest.rs` uses the same
#    constant.
# ---------------------------------------------------------------------------
ssh-keygen -Y sign -f "$KEY" -n "$NAMESPACE" "$PAYLOAD" >/dev/null
SIG_FILE="$PAYLOAD.sig"
echo "✓ ssh signature: $SIG_FILE"

# ---------------------------------------------------------------------------
# 4. Build rekord v0.0.1 ProposedEntry with `format: "ssh"`.
#    - `signature.content` is base64 of the PEM-armored SSH-SIG file.
#    - `signature.publicKey.content` is base64 of the OpenSSH single-line
#      `.pub` ("ssh-ed25519 AAAA... comment") format.
#    - `data.content` is base64 of the full payload. Rekor will
#      canonicalize this away to `data.hash` in the stored body, but
#      it's required at POST time — a hash-only entry is rejected.
#    `tr -d '\n'` handles GNU vs BSD base64 differences (no `-w0` on macOS).
# ---------------------------------------------------------------------------
SIG_B64="$(base64 < "$SIG_FILE" | tr -d '\n')"
PUBKEY_B64="$(base64 < "$KEY.pub" | tr -d '\n')"
DATA_B64="$(base64 < "$PAYLOAD" | tr -d '\n')"

ENTRY="$WORK_DIR/entry.json"
jq -n \
  --arg sig "$SIG_B64" \
  --arg pubkey "$PUBKEY_B64" \
  --arg data "$DATA_B64" \
  '{
     kind: "rekord",
     apiVersion: "0.0.1",
     spec: {
       data: { content: $data },
       signature: {
         format: "ssh",
         content: $sig,
         publicKey: { content: $pubkey }
       }
     }
   }' > "$ENTRY"
echo "✓ rekord+ssh entry built"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "=== DRY RUN — entry that WOULD be POSTed ==="
  jq . "$ENTRY"
  echo
  echo "Re-run without --dry-run to actually publish to $REKOR_URL"
  exit 0
fi

# ---------------------------------------------------------------------------
# 5. POST to Rekor. From here on the side effect is irreversible.
# ---------------------------------------------------------------------------
echo
echo "=== POSTing to $REKOR_URL/api/v1/log/entries ==="
RESPONSE="$WORK_DIR/response.json"
HTTP_STATUS="$(curl -sS -o "$RESPONSE" -w '%{http_code}' \
  -X POST \
  -H 'Content-Type: application/json' \
  --data @"$ENTRY" \
  "$REKOR_URL/api/v1/log/entries")"

if [[ "$HTTP_STATUS" != "201" ]]; then
  echo "✗ rekor returned HTTP $HTTP_STATUS:" >&2
  cat "$RESPONSE" >&2
  exit 1
fi
echo "✓ rekor accepted entry (HTTP 201)"

# ---------------------------------------------------------------------------
# 6. Parse response and confirm every field the verifier consumes is present.
# ---------------------------------------------------------------------------
UUID="$(jq -r 'keys[0]' "$RESPONSE")"
LOG_INDEX="$(jq -r ".\"$UUID\".logIndex" "$RESPONSE")"
LOG_ID="$(jq -r ".\"$UUID\".logID" "$RESPONSE")"
INTEGRATED_TIME="$(jq -r ".\"$UUID\".integratedTime" "$RESPONSE")"
BODY_B64="$(jq -r ".\"$UUID\".body" "$RESPONSE")"
SET_B64="$(jq -r ".\"$UUID\".verification.signedEntryTimestamp // empty" "$RESPONSE")"
HAS_INCLUSION_PROOF="$(jq -r ".\"$UUID\".verification.inclusionProof != null" "$RESPONSE")"

echo
echo "=== rekor response ==="
printf '  uuid:                   %s\n' "$UUID"
printf '  logIndex:               %s\n' "$LOG_INDEX"
printf '  logID:                  %s\n' "$LOG_ID"
printf '  integratedTime:         %s\n' "$INTEGRATED_TIME"
printf '  body (b64) bytes:       %s\n' "${#BODY_B64}"
printf '  SET (b64) bytes:        %s\n' "${#SET_B64}"
printf '  inclusionProof present: %s\n' "$HAS_INCLUSION_PROOF"

if [[ -z "$SET_B64" ]]; then
  echo "✗ no signedEntryTimestamp on the response — verifier expects this" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 7. Decode the body and confirm rekor canonicalized `data.content` away,
#    leaving only `data.hash`. This is the privacy property the producer
#    side relies on — the engineer's full attestation prose stays out of
#    the public transparency log.
# ---------------------------------------------------------------------------
BODY_JSON="$(echo "$BODY_B64" | base64 -d)"
STORED_KIND="$(echo "$BODY_JSON" | jq -r '.kind')"
STORED_FORMAT="$(echo "$BODY_JSON" | jq -r '.spec.signature.format')"
STORED_HAS_HASH="$(echo "$BODY_JSON" | jq -r '.spec.data.hash.value != null')"
STORED_HAS_CONTENT="$(echo "$BODY_JSON" | jq -r '.spec.data.content != null')"
STORED_HASH="$(echo "$BODY_JSON" | jq -r '.spec.data.hash.value // empty')"
STORED_PUBKEY="$(echo "$BODY_JSON" | jq -r '.spec.signature.publicKey.content')"
STORED_SIG="$(echo "$BODY_JSON" | jq -r '.spec.signature.content')"

echo
echo "=== canonicalized body checks ==="
printf '  kind:                   %s   (expected: rekord)\n' "$STORED_KIND"
printf '  signature.format:       %s    (expected: ssh)\n' "$STORED_FORMAT"
printf '  data.hash present:      %s\n' "$STORED_HAS_HASH"
printf '  data.content present:   %s  (expected: false — rekor strips it)\n' "$STORED_HAS_CONTENT"

if [[ "$STORED_KIND" != "rekord" ]] \
  || [[ "$STORED_FORMAT" != "ssh" ]] \
  || [[ "$STORED_HAS_HASH" != "true" ]] \
  || [[ "$STORED_HAS_CONTENT" != "false" ]]; then
  echo "✗ canonicalized body does not match expectations" >&2
  exit 1
fi
if [[ "$STORED_HASH" != "$DIGEST" ]]; then
  echo "✗ canonicalized data.hash.value ($STORED_HASH) ≠ sha256(payload) ($DIGEST)" >&2
  exit 1
fi
echo "✓ data.hash.value matches sha256(payload)"

if [[ "$STORED_PUBKEY" == "$PUBKEY_B64" ]]; then
  echo "✓ stored publicKey.content matches submitted (no canonicalization)"
else
  echo "⚠ stored publicKey.content DIFFERS — rekor canonicalized it:"
  echo "    sent (len ${#PUBKEY_B64}):   $PUBKEY_B64"
  echo "    stored (len ${#STORED_PUBKEY}): $STORED_PUBKEY"
  echo "  → verifier's body-hash check must hash the STORED form, not the submitted form."
fi

if [[ "$STORED_SIG" == "$SIG_B64" ]]; then
  echo "✓ stored signature.content matches submitted"
else
  echo "⚠ stored signature.content DIFFERS — rekor canonicalized it"
fi

# ---------------------------------------------------------------------------
# 8. Surfaces for human inspection.
# ---------------------------------------------------------------------------
echo
echo "view on sigstore search:"
echo "  https://search.sigstore.dev/?logIndex=$LOG_INDEX"
echo
echo "fetch raw:"
echo "  curl $REKOR_URL/api/v1/log/entries/$UUID | jq ."
