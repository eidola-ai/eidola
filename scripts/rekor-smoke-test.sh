#!/usr/bin/env bash
#
# Rekor `hashedrekord` smoke test via `cosign sign-blob`.
#
# Purpose
# -------
# Validate the protocol assumptions in the human attestation flow:
# that `cosign sign-blob` produces a Sigstore Bundle v0.3 with a
# `hashedrekord` v0.0.1 tlog entry that
# `crates/eidola-app-core/src/updater/human_attestation.rs` can verify.
# Confirms:
#
#   - Rekor still accepts cosign sign-blob entries (i.e. the production
#     Rekor v1 instance + cosign client are in working order).
#   - The bundle's `mediaType`, `verificationMaterial.publicKey.hint`
#     (cosign emits `base64(sha256(SPKI DER))`), and tlog kindVersion
#     match what our verifier expects.
#   - The rekor body's `signature.publicKey.content` is the PEM
#     `-----BEGIN PUBLIC KEY-----` SPKI block, and `data.hash.value`
#     equals sha256(blob) as a hex string.
#
# History — why cosign+hashedrekord and not the older flows
# ---------------------------------------------------------
# Earlier iterations of this script tried (1) `hashedrekord` v0.0.1
# with an OpenSSH public key — rejected because hashedrekord hardcodes
# x509 PKI — and (2) `rekord` v0.0.1 with `signature.format=ssh` and
# SSH-SIG signatures — worked, but Rekor v2 drops `rekord` entirely
# along with all non-x509 PKIs. Cosign-emitted `hashedrekord` with a
# PKIX SubjectPublicKeyInfo (ECDSA or Ed25519) is the entry shape that
# survives v2, and it's also what `cosign sign-blob --key …` produces
# whether the key is a local PEM, a PKCS#11 URI (YubiKey-PIV), or any
# of cosign's KMS URIs.
#
# WARNING — public transparency log
# ---------------------------------
# Rekor entries are *immutable and publicly visible*. Running this
# script (without --dry-run) publishes a synthetic entry to the
# production Sigstore log under a freshly-generated throwaway test
# keypair. The entry is harmless, but it will live in the public Merkle
# tree forever and the fact that this project ran a smoke test on a
# given day becomes part of public record. The key never touches any
# hardware or long-lived secret — it's generated, used, and discarded
# in this run.
#
# Usage
# -----
#   scripts/rekor-smoke-test.sh --dry-run   # build the bundle locally, do NOT POST
#   scripts/rekor-smoke-test.sh             # actually sign and POST (irreversible)
#
# Requires: cosign, jq, openssl (for PEM→DER verification), and one of
#           sha256sum / shasum.

set -euo pipefail

DRY_RUN=0
case "${1:-}" in
  --dry-run) DRY_RUN=1 ;;
  "") ;;
  *) echo "unknown arg: $1 (use --dry-run or no args)" >&2; exit 2 ;;
esac

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
echo "working dir: $WORK_DIR"

# ---------------------------------------------------------------------------
# 1. Throwaway P-256 cosign keypair (NOT a hardware key; discarded at exit).
# ---------------------------------------------------------------------------
# `cosign generate-key-pair` writes `cosign.key` (encrypted PEM) and
# `cosign.pub` (PEM SPKI) to the cwd. COSIGN_PASSWORD="" means no
# passphrase, which is fine for a 60-second smoke test.
( cd "$WORK_DIR" && COSIGN_PASSWORD="" cosign generate-key-pair >/dev/null )
echo "✓ throwaway P-256 keypair generated"

# Independent fingerprint computation:
# `cosign public-key` prints the PEM SPKI; sha256 of its DER form
# (openssl pkey decodes PEM→DER) is the canonical attestant fingerprint
# that our verifier matches against TRUSTED_ATTESTANT_FINGERPRINTS.
SPKI_DER_HEX=$(openssl pkey -pubin -in "$WORK_DIR/cosign.pub" -outform DER | xxd -p | tr -d '\n')
FP_HEX=$(printf '%s' "$SPKI_DER_HEX" | xxd -r -p | shasum -a 256 | awk '{print $1}')
echo "✓ attestant fingerprint (sha256 of SPKI DER): $FP_HEX"

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

BUNDLE="$WORK_DIR/payload.bundle.json"

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "=== DRY RUN — would run cosign sign-blob ==="
  echo "  cosign sign-blob --yes --key $WORK_DIR/cosign.key \\"
  echo "    --bundle $BUNDLE $PAYLOAD"
  echo
  echo "Re-run without --dry-run to actually sign and publish to Rekor."
  exit 0
fi

# ---------------------------------------------------------------------------
# 3. Sign and upload. cosign POSTs to Rekor and writes the v0.3 bundle.
#    --yes skips its "are you sure?" prompt.
# ---------------------------------------------------------------------------
echo
echo "=== cosign sign-blob ==="
COSIGN_PASSWORD="" cosign sign-blob \
  --yes \
  --key "$WORK_DIR/cosign.key" \
  --bundle "$BUNDLE" \
  "$PAYLOAD" >/dev/null
echo "✓ cosign sign-blob succeeded; bundle → $BUNDLE"

# ---------------------------------------------------------------------------
# 4. Parse the bundle and confirm every field the verifier expects.
# ---------------------------------------------------------------------------
MEDIA_TYPE=$(jq -r '.mediaType' "$BUNDLE")
HAS_PK=$(jq -r '.verificationMaterial.publicKey != null' "$BUNDLE")
HAS_CERT=$(jq -r '.verificationMaterial.certificate != null' "$BUNDLE")
HINT=$(jq -r '.verificationMaterial.publicKey.hint' "$BUNDLE")
LOG_INDEX=$(jq -r '.verificationMaterial.tlogEntries[0].logIndex' "$BUNDLE")
LOG_ID=$(jq -r '.verificationMaterial.tlogEntries[0].logId.keyId' "$BUNDLE")
KIND=$(jq -r '.verificationMaterial.tlogEntries[0].kindVersion.kind' "$BUNDLE")
VERSION=$(jq -r '.verificationMaterial.tlogEntries[0].kindVersion.version' "$BUNDLE")
HAS_SET=$(jq -r '.verificationMaterial.tlogEntries[0].inclusionPromise.signedEntryTimestamp != null' "$BUNDLE")
HAS_PROOF=$(jq -r '.verificationMaterial.tlogEntries[0].inclusionProof != null' "$BUNDLE")
MSG_DIGEST_B64=$(jq -r '.messageSignature.messageDigest.digest' "$BUNDLE")
MSG_DIGEST_HEX=$(printf '%s' "$MSG_DIGEST_B64" | base64 -d | xxd -p | tr -d '\n')
SIG_LEN=$(jq -r '.messageSignature.signature' "$BUNDLE" | base64 -d | wc -c | tr -d ' ')

echo
echo "=== bundle structural checks ==="
printf '  mediaType:                    %s\n' "$MEDIA_TYPE"
printf '  verificationMaterial.publicKey present:  %s   (expected: true)\n' "$HAS_PK"
printf '  verificationMaterial.certificate present: %s  (expected: false)\n' "$HAS_CERT"
printf '  publicKey.hint:               %s\n' "$HINT"
printf '  tlogEntries[0].logIndex:      %s\n' "$LOG_INDEX"
printf '  tlogEntries[0].kindVersion:   %s %s   (expected: hashedrekord 0.0.1)\n' "$KIND" "$VERSION"
printf '  inclusionPromise.SET present: %s\n' "$HAS_SET"
printf '  inclusionProof present:       %s\n' "$HAS_PROOF"
printf '  messageDigest:                %s   (expected: %s)\n' "$MSG_DIGEST_HEX" "$DIGEST"
printf '  signature bytes:              %s\n' "$SIG_LEN"

FAIL=0
[[ "$MEDIA_TYPE" == application/vnd.dev.sigstore.bundle* ]] || { echo "✗ mediaType wrong" >&2; FAIL=1; }
[[ "$HAS_PK" == "true"  ]] || { echo "✗ publicKey arm missing"  >&2; FAIL=1; }
[[ "$HAS_CERT" == "false" ]] || { echo "✗ unexpected certificate arm" >&2; FAIL=1; }
[[ "$KIND" == "hashedrekord" && "$VERSION" == "0.0.1" ]] || { echo "✗ wrong kindVersion"  >&2; FAIL=1; }
[[ "$HAS_SET" == "true" ]] || { echo "✗ missing signedEntryTimestamp" >&2; FAIL=1; }
[[ "$HAS_PROOF" == "true" ]] || { echo "✗ missing inclusionProof"     >&2; FAIL=1; }
[[ "$MSG_DIGEST_HEX" == "$DIGEST" ]] || { echo "✗ messageDigest ≠ sha256(payload)" >&2; FAIL=1; }

# Confirm publicKey.hint equals base64(sha256(SPKI DER)) — the cosign
# convention our verifier accepts.
FP_BYTES_B64=$(printf '%s' "$FP_HEX" | xxd -r -p | base64)
if [[ "$HINT" == "$FP_BYTES_B64" ]]; then
  echo "✓ publicKey.hint matches base64(sha256(SPKI DER))"
else
  echo "✗ publicKey.hint ($HINT) ≠ base64 fingerprint ($FP_BYTES_B64)" >&2
  FAIL=1
fi

# Decode the canonicalized body and confirm:
#   - kind/apiVersion/data.hash match
#   - publicKey.content is a PEM PUBLIC KEY block (PKIX SPKI), matching cosign.pub
BODY_JSON=$(jq -r '.verificationMaterial.tlogEntries[0].canonicalizedBody' "$BUNDLE" | base64 -d)
BODY_KIND=$(echo "$BODY_JSON" | jq -r '.kind')
BODY_VERSION=$(echo "$BODY_JSON" | jq -r '.apiVersion')
BODY_HASH=$(echo "$BODY_JSON" | jq -r '.spec.data.hash.value')
BODY_PUBKEY_PEM=$(echo "$BODY_JSON" | jq -r '.spec.signature.publicKey.content' | base64 -d)

[[ "$BODY_KIND" == "hashedrekord" && "$BODY_VERSION" == "0.0.1" ]] \
  || { echo "✗ body kind/apiVersion wrong" >&2; FAIL=1; }
[[ "$BODY_HASH" == "$DIGEST" ]] \
  || { echo "✗ body data.hash.value ($BODY_HASH) ≠ sha256(payload) ($DIGEST)" >&2; FAIL=1; }
if echo "$BODY_PUBKEY_PEM" | grep -q '^-----BEGIN PUBLIC KEY-----'; then
  echo "✓ rekor body publicKey.content is a PEM PUBLIC KEY block"
else
  echo "✗ rekor body publicKey.content is not a PEM PUBLIC KEY block" >&2
  FAIL=1
fi
if [[ "$BODY_PUBKEY_PEM" == "$(cat "$WORK_DIR/cosign.pub")" ]]; then
  echo "✓ rekor body publicKey.content equals cosign.pub byte-for-byte"
else
  echo "⚠ rekor body publicKey.content differs from cosign.pub (rekor may have canonicalized)"
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo "✗ smoke test FAILED — see warnings above" >&2
  exit 1
fi
echo "✓ smoke test PASSED — cosign output matches the shape our verifier expects"

# ---------------------------------------------------------------------------
# 5. Surfaces for human inspection.
# ---------------------------------------------------------------------------
echo
echo "view on sigstore search:"
echo "  https://search.sigstore.dev/?logIndex=$LOG_INDEX"
echo
echo "bundle:"
echo "  $BUNDLE"
