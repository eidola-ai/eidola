#!/usr/bin/env bash
# The account side of the local dev stack, for the phone interface's UI tests: the real gateway
# (cargo-built, on this machine) over the compose stack's TLS-only PostgreSQL, behind an attesting
# tinfoil-shim-mock, with tools/billing_edge_stub.py between the two standing in for the payment
# processor (the gateway runs without a Stripe key). The terms gate is on, pinned to this checkout's
# www/pages/terms.md and privacy.md, so the first run's consent page has documents to show.
#   account_stack.sh up DIR     starts it; writes DIR/eidola.json (what a development bundle is told
#                               to trust: the shim's URL, its hardware CAs, its TLS root, its
#                               measurement) and DIR/pids; logs to DIR
#   account_stack.sh down DIR   stops what `up` started (and the database container)
# The shim's certificate also names 10.0.2.2, the address the Android emulator reaches this machine
# at. Needs docker (the database) and the database image `just services` builds.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CMD="${1:-}"
DIR="${2:-}"
[[ ( "$CMD" == up || "$CMD" == down ) && -n "$DIR" ]] || { echo "usage: account_stack.sh up|down DIR" >&2; exit 2; }
mkdir -p "$DIR" && DIR="$(cd "$DIR" && pwd)"
GATEWAY_PORT=18080
EDGE_PORT=18081
SHIM_PORT=18446
MEASUREMENT="$(printf '5e%.0s' $(seq 48))"
HOST_DATA="$(printf '0%.0s' $(seq 64))"

if [[ "$CMD" == down ]]; then
  [[ -f "$DIR/pids" ]] && while read -r pid; do kill "$pid" 2>/dev/null || true; done <"$DIR/pids"
  rm -f "$DIR/pids"
  (cd "$ROOT" && docker compose stop postgres >/dev/null 2>&1) || true
  exit 0
fi

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
wait_port() { for _ in $(seq 120); do nc -z 127.0.0.1 "$1" 2>/dev/null && return 0; sleep 0.5; done; echo "nothing on $1" >&2; return 1; }

# The database: TLS only, its root trusted by the gateway.
(cd "$ROOT" && ./scripts/dev-postgres-tls.sh >/dev/null && docker compose up -d postgres >/dev/null)
for i in $(seq 60); do
  (cd "$ROOT" && docker compose exec -T postgres pg_isready -U eidola >/dev/null 2>&1) && break
  [[ $i == 60 ]] && { echo "postgres did not start" >&2; exit 1; }
  sleep 1
done
PRESENT="$(cd "$ROOT" && docker compose exec -T postgres psql -U eidola -d eidola -tAc \
  "SELECT 1 FROM information_schema.tables WHERE table_schema='public' AND table_name='account'" 2>/dev/null || true)"
if [[ -z "$PRESENT" ]]; then
  (cd "$ROOT" && docker compose exec -T postgres psql -U eidola -d eidola -v ON_ERROR_STOP=1 \
    -f /docker-entrypoint-initdb.d/schema.sql -q)
fi

(cd "$ROOT" && cargo build -q -p eidola-server-gateway -p tinfoil-shim-mock)
BIN="${CARGO_TARGET_DIR:-$ROOT/target}/debug"
: >"$DIR/pids"
env -i PATH="$PATH" HOME="$HOME" \
  DATABASE_URL="postgres://eidola@localhost/eidola?sslmode=verify-full" \
  DATABASE_SSL_CERT="$(cat "$ROOT/.dev-certs/postgres/ca.pem")" \
  CREDENTIAL_MASTER_KEY="$(printf '0%.0s' $(seq 64))" \
  BIND_ADDR="127.0.0.1:$GATEWAY_PORT" \
  ENGINE_TOKEN=dev-engine-token-not-a-secret \
  TERMS_OF_SERVICE_SHA256="$(sha256 "$ROOT/www/pages/terms.md" | cut -d' ' -f1)" \
  TERMS_OF_SERVICE_URL=https://www.eidola.ai/terms/ \
  PRIVACY_POLICY_SHA256="$(sha256 "$ROOT/www/pages/privacy.md" | cut -d' ' -f1)" \
  PRIVACY_POLICY_URL=https://www.eidola.ai/privacy/ \
  "$BIN/eidola-server-gateway" >"$DIR/gateway.log" 2>&1 &
echo $! >>"$DIR/pids"
wait_port "$GATEWAY_PORT"
python3 "$HERE/billing_edge_stub.py" --port "$EDGE_PORT" --upstream "http://127.0.0.1:$GATEWAY_PORT" \
  >"$DIR/edge.log" 2>&1 &
echo $! >>"$DIR/pids"
wait_port "$EDGE_PORT"
mkdir -p "$DIR/certs"
CERT_DIR="$DIR/certs" LISTEN_ADDR="127.0.0.1:$SHIM_PORT" UPSTREAM_URL="http://127.0.0.1:$EDGE_PORT" \
  DEV_MEASUREMENT="$MEASUREMENT" EXTRA_TLS_NAMES=10.0.2.2 "$BIN/tinfoil-shim-mock" >"$DIR/shim.log" 2>&1 &
echo $! >>"$DIR/pids"
wait_port "$SHIM_PORT"

python3 - "$DIR" "$SHIM_PORT" "$MEASUREMENT" "$HOST_DATA" >"$DIR/eidola.json" <<'PY'
import json, sys
d, port, measurement, host_data = sys.argv[1:]
pem = lambda name: open(f"{d}/certs/{name}").read()
print(json.dumps({
    "port": int(port), "measurement": measurement, "host_data": host_data, "guest_policy": "0x30133",
    "ark_pem": pem("ark.pem"), "ask_pem": pem("ask.pem"), "tls_ca_pem": pem("tls-ca.pem"),
}, indent=1))
PY
echo "account stack: shim https://127.0.0.1:$SHIM_PORT -> edge :$EDGE_PORT -> gateway :$GATEWAY_PORT; $DIR/eidola.json"
