#!/usr/bin/env bash
# E2E webhook smoke tests.
#
# Validates Stripe CLI → receiver → durable inbox → independent worker.
#
# All triggers create fixture objects with Stripe-generated customer IDs that
# don't exist in our DB, so jobs remain visible for review. Assertions inspect
# durable inbox state and receiver health. Local fixture-driven integration
# tests cover ledger mutations without contacting Stripe.
#
# Requires: STRIPE_API_KEY (sk_test_...) set in environment or .env.

set -euo pipefail

# ── Cleanup on exit ──────────────────────────────────────────────────────────

cleanup() {
    echo "==> Cleaning up..."
    docker compose --profile billing --profile stripe down --volumes --remove-orphans 2>/dev/null || true
}

# ── Preflight ────────────────────────────────────────────────────────────────

if [ -z "${STRIPE_API_KEY:-}" ]; then
    echo "ERROR: STRIPE_API_KEY is not set." >&2
    echo "Usage: STRIPE_API_KEY=sk_test_xxx just test-webhook-smoke" >&2
    exit 1
fi
case "$STRIPE_API_KEY" in
    sk_test_* | rk_test_*) ;;
    *) echo "ERROR: this smoke test requires a Stripe test-mode key." >&2; exit 1 ;;
esac
# Never tear down the developer's existing Compose project.
export COMPOSE_PROJECT_NAME="eidola-webhook-smoke-$$"
trap cleanup EXIT

# ── Build images ─────────────────────────────────────────────────────────────

echo "==> Building images (docker-dev profile)..."
CARGO_PROFILE=docker-dev docker buildx bake postgres billing stripe-cli

# ── Start postgres, wait healthy, apply schema ───────────────────────────────

echo "==> Starting postgres..."
docker compose up -d postgres
echo "==> Waiting for postgres to be healthy..."
for i in $(seq 1 30); do
    if docker compose exec postgres pg_isready -U eidola >/dev/null 2>&1; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: postgres did not become healthy in 30s" >&2
        exit 1
    fi
    sleep 1
done

echo "==> Applying schema (if not already present)..."
SCHEMA_PRESENT=$(
    docker compose exec -T postgres psql -U eidola -d eidola -tAc \
        "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='account'" \
        2>/dev/null || true
)
if [ -z "$SCHEMA_PRESENT" ]; then
    docker compose exec -T postgres psql -U eidola -d eidola \
        -v ON_ERROR_STOP=1 -f /docker-entrypoint-initdb.d/schema.sql -q
else
    echo "    schema already applied; skipping"
fi

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
echo "    webhook signing secret captured"

# ── Start receiver, worker and stripe-cli ────────────────────────────────────

echo "==> Starting billing processes and stripe-cli..."
docker compose --profile billing --profile stripe up -d billing-receiver billing-worker stripe-cli

# ── Poll receiver health ────────────────────────────────────────────────────

echo "==> Waiting for receiver /healthz..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8081/healthz >/dev/null 2>&1; then
        echo "    receiver healthy"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: receiver did not become healthy in 30s" >&2
        docker compose --profile billing --profile stripe logs billing-receiver
        exit 1
    fi
    sleep 1
done

# ── Wait for stripe-cli "Ready!" ────────────────────────────────────────────

echo "==> Waiting for stripe-cli to be ready..."
for i in $(seq 1 30); do
    if docker compose --profile billing --profile stripe logs stripe-cli 2>&1 | grep -q "Ready!"; then
        echo "    stripe-cli ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: stripe-cli did not become ready in 30s" >&2
        docker compose --profile billing --profile stripe logs stripe-cli
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
BILLING_LOGS=$(docker compose --profile billing --profile stripe logs billing-receiver billing-worker 2>&1)

# Receiver is still running
if ! curl -sf http://localhost:8081/healthz >/dev/null 2>&1; then
    echo "FAIL: receiver is no longer healthy"
    FAILED=1
fi

# Every fixture event is durable and has reached the separate worker. These
# event names are fixed literals above, never user/provider input in SQL.
for event in "${EVENTS[@]}"; do
    observed=$(docker compose exec -T postgres psql -U eidola -d eidola -tAc \
        "SELECT EXISTS(SELECT 1 FROM billing_event WHERE kind='$event' AND attempts>0 AND last_error='requires_review' AND completed_at IS NULL)")
    if [ "$observed" = "t" ]; then
        echo "  OK: $event durably received and processed for review"
    else
        echo "FAIL: $event has not reached review in the durable inbox"
        FAILED=1
    fi
done

# No panics
if echo "$BILLING_LOGS" | grep -qi "panic"; then
    echo "FAIL: panic detected in billing logs"
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
    echo "==> Billing logs:"
    echo "$BILLING_LOGS"
    exit 1
fi
