#!/usr/bin/env bash
# Start the full stack with Stripe webhook forwarding for manual testing.
#
# Captures the webhook signing secret from stripe-cli, then starts all
# services in the foreground. Ctrl-C tears everything down.
#
# Requires: STRIPE_API_KEY (sk_test_...) set in environment or .env.

set -euo pipefail

# ── Cleanup on exit ──────────────────────────────────────────────────────────

cleanup() {
    echo ""
    echo "==> Cleaning up..."
    # docker compose --profile test down --volumes --remove-orphans 2>/dev/null || true
    docker compose --profile test down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── Preflight ────────────────────────────────────────────────────────────────

if [ -z "${STRIPE_API_KEY:-}" ]; then
    echo "ERROR: STRIPE_API_KEY is not set." >&2
    echo "Usage: STRIPE_API_KEY=sk_test_xxx just dev-stripe" >&2
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
docker compose exec postgres psql -U eidolons -d eidolons -f /docker-entrypoint-initdb.d/schema.sql -q

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

# ── Start everything in foreground ───────────────────────────────────────────

echo "==> Starting server and stripe-cli (Ctrl-C to stop)..."
docker compose --profile test up server stripe-cli
