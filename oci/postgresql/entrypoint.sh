#!/bin/sh
set -e

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
POSTGRES_USER="${POSTGRES_USER:-eidola}"
POSTGRES_DB="${POSTGRES_DB:-eidola}"
TLS_DIR="${TLS_DIR:-/tls}"

# Serve only TLS over the network: the certificate, its key and the
# hba rules come from scripts/dev-postgres-tls.sh, mounted at $TLS_DIR.
for f in server.pem server.key pg_hba.conf; do
  if [ ! -f "$TLS_DIR/$f" ]; then
    echo "missing $TLS_DIR/$f: run scripts/dev-postgres-tls.sh and mount its directory at $TLS_DIR" >&2
    exit 1
  fi
done
# The server refuses a key anyone but its owner can read.
install -m 600 "$TLS_DIR/server.key" /run/postgresql/server.key

# Initialize data directory if not already initialized
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "Initializing PostgreSQL data directory..."
  initdb -D "$PGDATA" --username="$POSTGRES_USER" --auth=trust --no-locale --encoding=UTF8

  # Listen on all interfaces
  sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PGDATA/postgresql.conf"

  # Start temporarily to create database and run init scripts
  pg_ctl -D "$PGDATA" -w start -o "-p 5432"

  # initdb only creates "postgres", "template0", "template1"
  if [ "$POSTGRES_DB" != "postgres" ]; then
    createdb -U "$POSTGRES_USER" "$POSTGRES_DB"
  fi

  # Run init scripts in Postgres' default public schema.
  for f in /docker-entrypoint-initdb.d/*.sql; do
    if [ -f "$f" ]; then
      echo "Running init script: $f"
      psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$f"
    fi
  done

  pg_ctl -D "$PGDATA" -w stop
  echo "PostgreSQL initialized."
fi

exec postgres -D "$PGDATA" \
  -c ssl=on \
  -c ssl_min_protocol_version=TLSv1.3 \
  -c ssl_cert_file="$TLS_DIR/server.pem" \
  -c ssl_key_file=/run/postgresql/server.key \
  -c hba_file="$TLS_DIR/pg_hba.conf"
