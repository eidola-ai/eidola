#!/bin/sh
set -e

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
POSTGRES_USER="${POSTGRES_USER:-eidola}"
POSTGRES_DB="${POSTGRES_DB:-eidola}"

# Initialize data directory if not already initialized
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo "Initializing PostgreSQL data directory..."
  initdb -D "$PGDATA" --username="$POSTGRES_USER" --auth=trust --no-locale --encoding=UTF8

  # Allow connections from any host (for docker networking)
  echo "host all all 0.0.0.0/0 trust" >> "$PGDATA/pg_hba.conf"
  echo "host all all ::/0 trust" >> "$PGDATA/pg_hba.conf"

  # Listen on all interfaces
  sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" "$PGDATA/postgresql.conf"

  # Start temporarily to create database and run init scripts
  pg_ctl -D "$PGDATA" -w start -o "-p 5432"

  # initdb only creates "postgres", "template0", "template1"
  if [ "$POSTGRES_DB" != "postgres" ]; then
    createdb -U "$POSTGRES_USER" "$POSTGRES_DB"
  fi

  # Create the eidola schema and run init scripts within it
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "CREATE SCHEMA IF NOT EXISTS eidola;"
  for f in /docker-entrypoint-initdb.d/*.sql; do
    if [ -f "$f" ]; then
      echo "Running init script: $f"
      PGOPTIONS="-c search_path=eidola" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f "$f"
    fi
  done

  pg_ctl -D "$PGDATA" -w stop
  echo "PostgreSQL initialized."
fi

exec postgres -D "$PGDATA"
