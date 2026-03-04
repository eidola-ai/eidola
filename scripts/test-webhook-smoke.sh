#!/usr/bin/env bash
# E2E webhook smoke tests.
#
# Validates the full pipeline: Stripe CLI → HTTP → server → signature
# verification → event parsing → handler dispatch → DB lookup.
#
# All triggers create fixture objects with Stripe-generated customer IDs that
# don't exist in our DB, so every handler hits the "orphan customer" path.
# Assertions verify: events received (log grep), no crashes/panics, server
# stays healthy. The unit integration tests already cover happy-path ledger
# mutations.
#
# Requires: STRIPE_API_KEY (sk_test_...) set in environment or .env.

set -euo pipefail

# ── Cleanup on exit ──────────────────────────────────────────────────────────

cleanup() {
    echo "==> Cleaning up..."
    docker compose --profile test down --volumes --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── Preflight ────────────────────────────────────────────────────────────────

if [ -z "${STRIPE_API_KEY:-}" ]; then
    echo "ERROR: STRIPE_API_KEY is not set." >&2
    echo "Usage: STRIPE_API_KEY=sk_test_xxx just test-webhook-smoke" >&2
    exit 1
fi

# ── Build images ─────────────────────────────────────────────────────────────

echo "==> Building images (docker-dev profile)..."
CARGO_PROFILE=docker-dev docker buildx bake

# ── Start postgres, wait healthy, apply schema ───────────────────────────────

echo "==> Starting postgres..."
docker compose up -d postgres
echo "==> Waiting for postgres to be healthy..."
for i in $(seq 1 30); do
    if docker compose exec postgres pg_isready -U eidolons >/dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: postgres did not become healthy in 30s" >&2
        exit 1
    fi
    sleep 1
done

echo "==> Applying schema..."
docker compose exec postgres psql -U eidolons -d eidolons -f /schema/schema.sql -q

# ── Capture webhook secret ───────────────────────────────────────────────────

echo "==> Capturing Stripe webhook secret..."
STRIPE_WEBHOOK_SECRET=$(
    docker compose run --rm --no-deps stripe-cli listen --print-secret 2>/dev/null
)

if [ -z "$STRIPE_WEBHOOK_SECRET" ]; then
    echo "ERROR: failed to capture webhook secret" >&2
    exit 1
fi

export STRIPE_WEBHOOK_SECRET
echo "    secret: ${STRIPE_WEBHOOK_SECRET:0:12}..."

# ── Start server + stripe-cli ────────────────────────────────────────────────

echo "==> Starting server and stripe-cli..."
docker compose --profile test up -d server stripe-cli

# ── Poll /health ─────────────────────────────────────────────────────────────

echo "==> Waiting for server /health..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
        echo "    server healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: server did not become healthy in 30s" >&2
        docker compose --profile test logs server
        exit 1
    fi
    sleep 1
done

# ── Wait for stripe-cli "Ready!" ────────────────────────────────────────────

echo "==> Waiting for stripe-cli to be ready..."
for i in $(seq 1 30); do
    if docker compose --profile test logs stripe-cli 2>&1 | grep -q "Ready!"; then
        echo "    stripe-cli ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: stripe-cli did not become ready in 30s" >&2
        docker compose --profile test logs stripe-cli
        exit 1
    fi
    sleep 1
done

# ── Trigger events ───────────────────────────────────────────────────────────

EVENTS=(
    "checkout.session.completed"
    "invoice.paid"
    "charge.refunded"
    "charge.dispute.created"
)

echo "==> Triggering ${#EVENTS[@]} events..."
for event in "${EVENTS[@]}"; do
    echo "    trigger: $event"
    docker compose run --rm --no-deps stripe-cli trigger "$event" 2>&1 | tail -1
    sleep 2
done

# ── Wait for delivery/processing ─────────────────────────────────────────────

echo "==> Waiting 5s for delivery/processing..."
sleep 5

# ── Assertions ───────────────────────────────────────────────────────────────

echo "==> Running assertions..."
FAILED=0
SERVER_LOGS=$(docker compose --profile test logs server 2>&1)

# Server is still running
if ! curl -sf http://localhost:8080/health >/dev/null 2>&1; then
    echo "FAIL: server is no longer healthy"
    FAILED=1
fi

# Each event type appears in server logs
for event in "${EVENTS[@]}"; do
    if echo "$SERVER_LOGS" | grep -q "webhook: received $event"; then
        echo "  OK: $event received"
    else
        echo "FAIL: $event not found in server logs"
        FAILED=1
    fi
done

# No panics
if echo "$SERVER_LOGS" | grep -qi "panic"; then
    echo "FAIL: panic detected in server logs"
    FAILED=1
else
    echo "  OK: no panics"
fi

# ── Result ───────────────────────────────────────────────────────────────────

echo ""
if [ "$FAILED" -eq 0 ]; then
    echo "All webhook smoke tests passed."
else
    echo "Some webhook smoke tests FAILED." >&2
    echo ""
    echo "==> Server logs:"
    echo "$SERVER_LOGS"
    exit 1
fi
