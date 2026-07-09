#!/usr/bin/env bash
# Bring up the local dev stack.
#
# Usage:
#   scripts/dev.sh [--container|--host] [--inference]
#
# Modes:
#   --container  (default)  eidola-server runs inside docker. Shim forwards
#                           to the in-network `server` container.
#   --host                  eidola-server runs on the host with cargo. Shim
#                           forwards to host.docker.internal:8080. The script
#                           writes `.env.local` with BIND_ADDR (and the
#                           captured Stripe webhook secret) for the host
#                           server to source.
#
# Options:
#   --inference             Also run the self-hosted eidola-inference
#                           container (llama.cpp serving the dev Gemma model;
#                           first start downloads + hash-verifies ~3.3 GiB of
#                           weights into the `models` volume) and point the
#                           server at it via EIDOLA_INFERENCE_URL.
#
# In both modes the script:
#   - builds the images it needs
#   - starts postgres and applies schema.sql (idempotent — pass through
#     `just db-reset` if you want a clean DB)
#   - if STRIPE_API_KEY is set, captures the Stripe webhook secret and starts
#     stripe-cli forwarding to the shim; otherwise stripe-cli is skipped
#   - starts everything detached and prints next-step instructions
#
# Stop with `just down`.

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Parse args ───────────────────────────────────────────────────────────────

MODE="container"
INFERENCE=0
for arg in "$@"; do
    case "$arg" in
        --container) MODE="container" ;;
        --host) MODE="host" ;;
        --inference) INFERENCE=1 ;;
        -h | --help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $arg" >&2
            echo "Usage: $0 [--container|--host] [--inference]" >&2
            exit 1
            ;;
    esac
done

# ── Build images ─────────────────────────────────────────────────────────────
#
# Use bake (not `docker compose build`) so the amd64 platform pinning in
# docker-bake.hcl is honored — that keeps the build silent on arm64 hosts,
# whose only option for these images is amd64 emulation anyway.

BAKE_TARGETS=(postgres shim)
if [ "$MODE" = "container" ]; then
    BAKE_TARGETS+=(server)
fi
if [ "$INFERENCE" = "1" ]; then
    BAKE_TARGETS+=(inference)
fi
if [ -n "${STRIPE_API_KEY:-}" ]; then
    BAKE_TARGETS+=(stripe-cli)
fi

echo "==> Building images: ${BAKE_TARGETS[*]}..."
CARGO_PROFILE="${CARGO_PROFILE:-docker-dev}" docker buildx bake "${BAKE_TARGETS[@]}"

# ── Postgres + schema ────────────────────────────────────────────────────────

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
    echo "    schema already applied; skipping (run \`just db-reset\` to recreate)"
fi

# ── Stripe webhook secret (optional) ─────────────────────────────────────────

STRIPE_WEBHOOK_SECRET=""
if [ -n "${STRIPE_API_KEY:-}" ]; then
    echo "==> Capturing Stripe webhook secret..."
    STRIPE_WEBHOOK_SECRET=$(
        docker compose run --rm --no-deps stripe-cli listen --print-secret 2>/dev/null
    )
    if [ -z "$STRIPE_WEBHOOK_SECRET" ]; then
        echo "ERROR: failed to capture webhook secret" >&2
        exit 1
    fi
    echo "    secret: ${STRIPE_WEBHOOK_SECRET:0:12}..."
else
    echo "==> STRIPE_API_KEY not set; skipping stripe-cli."
fi

# ── Compute mode-specific config ─────────────────────────────────────────────

UP_SERVICES=(shim)
UP_PROFILES=()
EIDOLA_INFERENCE_URL=""
if [ "$MODE" = "container" ]; then
    SHIM_UPSTREAM_URL="http://server:8080"
    UP_SERVICES+=(server)
    UP_PROFILES+=(--profile server)
    if [ "$INFERENCE" = "1" ]; then
        # In-network address; the containerized server reaches the inference
        # service by compose DNS name.
        EIDOLA_INFERENCE_URL="http://inference:8081/v1"
    fi
else
    SHIM_UPSTREAM_URL="http://host.docker.internal:8080"
    if [ "$INFERENCE" = "1" ]; then
        # The host-mode server reaches the inference container through its
        # published port.
        EIDOLA_INFERENCE_URL="http://localhost:8081/v1"
    fi
fi
if [ "$INFERENCE" = "1" ]; then
    UP_SERVICES+=(inference)
    UP_PROFILES+=(--profile inference)
fi
if [ -n "$STRIPE_WEBHOOK_SECRET" ]; then
    UP_SERVICES+=(stripe-cli)
    UP_PROFILES+=(--profile stripe)
fi

# ── Bring up services (detached) ─────────────────────────────────────────────

echo "==> Starting ${UP_SERVICES[*]} (mode: $MODE, detached)..."
SHIM_UPSTREAM_URL="$SHIM_UPSTREAM_URL" \
STRIPE_WEBHOOK_SECRET="$STRIPE_WEBHOOK_SECRET" \
EIDOLA_INFERENCE_URL="$EIDOLA_INFERENCE_URL" \
    docker compose ${UP_PROFILES[@]+"${UP_PROFILES[@]}"} up -d "${UP_SERVICES[@]}"

# ── Host-mode finalization: write .env.local for cargo ───────────────────────

if [ "$MODE" = "host" ]; then
    {
        echo "# Generated by scripts/dev.sh --host — host-mode dev overrides."
        echo "# Source after .env: \`set -a; source .env; source .env.local; set +a\`"
        echo "BIND_ADDR=0.0.0.0:8080"
        if [ -n "$STRIPE_WEBHOOK_SECRET" ]; then
            echo "STRIPE_WEBHOOK_SECRET=$STRIPE_WEBHOOK_SECRET"
        fi
        if [ -n "$EIDOLA_INFERENCE_URL" ]; then
            echo "EIDOLA_INFERENCE_URL=$EIDOLA_INFERENCE_URL"
            echo "EIDOLA_INFERENCE_MODEL=gemma4-e2b"
            echo "EIDOLA_INFERENCE_MODEL_NAME=Gemma 4 E2B"
            echo "EIDOLA_INFERENCE_CONTEXT_LENGTH=8192"
        fi
    } > .env.local
fi

# ── Done ─────────────────────────────────────────────────────────────────────

cat <<EOF

==> Stack is running (detached).

    postgres : localhost:5432
    shim     : https://localhost:8443  (-> $SHIM_UPSTREAM_URL)
EOF
[ "$MODE" = "container" ] && echo "    server   : http://localhost:8080  (in docker)"
[ "$INFERENCE" = "1" ] && echo "    inference: http://localhost:8081  (llama-server; first boot fetches + verifies weights)"
[ -n "$STRIPE_WEBHOOK_SECRET" ] && echo "    stripe-cli: forwarding to https://shim:8443/v1/webhooks/stripe"
echo ""

if [ "$MODE" = "host" ]; then
    cat <<EOF
To run the server on the host:

    set -a; source .env; source .env.local; set +a
    cargo run -p eidola-server

EOF
fi

cat <<EOF
To follow logs:    docker compose logs -f
To stop the stack: just down
EOF
