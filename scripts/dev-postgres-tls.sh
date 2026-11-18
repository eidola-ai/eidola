#!/usr/bin/env bash
# Make the TLS a local PostgreSQL serves, and the root its clients trust.
#
# Usage: scripts/dev-postgres-tls.sh [dir]     (default: .dev-certs/postgres)
#
# Every database connection Eidola makes is verified TLS, development and
# tests included (crates/eidola-billing/src/connection.rs). A local database
# has no certificate from a public CA, so this makes a private root for it:
#
#   ca.pem       the root clients trust: the value of DATABASE_SSL_CERT
#   server.pem   the server's certificate, for localhost, postgres (the
#                compose service), 127.0.0.1 and ::1, signed by that root
#   server.key   its key. World-readable, so a container's server user can
#                copy it into place: it guards a disposable local database
#                and nothing else
#   pg_hba.conf  socket connections and TLS connections only, so the server
#                refuses plaintext TCP as well
#
# The root's key is discarded once the server's certificate is signed, so
# nothing can mint another certificate this root vouches for. An existing set
# is kept until its certificate is within 30 days of expiring; delete the
# directory to make a new one. Needs only `openssl` (OpenSSL or LibreSSL).

set -euo pipefail
cd "$(dirname "$0")/.."

dir="${1:-.dev-certs/postgres}"

if [ -f "$dir/ca.pem" ] && [ -f "$dir/server.pem" ] && [ -f "$dir/server.key" ] &&
    [ -f "$dir/pg_hba.conf" ] &&
    openssl x509 -checkend 2592000 -noout -in "$dir/server.pem" >/dev/null; then
    exit 0
fi

mkdir -p "$dir"
staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT

cat >"$staging/ext.cnf" <<'EOF'
[req]
distinguished_name = dn
[dn]
[root]
basicConstraints = critical, CA:TRUE, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
[server]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = serverAuth
subjectAltName = DNS:localhost, DNS:postgres, IP:127.0.0.1, IP:0:0:0:0:0:0:0:1
authorityKeyIdentifier = keyid
EOF

openssl ecparam -name prime256v1 -genkey -noout -out "$staging/ca.key"
openssl req -x509 -new -sha256 -days 3650 -key "$staging/ca.key" \
    -subj "/CN=Eidola local PostgreSQL root" \
    -config "$staging/ext.cnf" -extensions root -out "$staging/ca.pem"

openssl ecparam -name prime256v1 -genkey -noout -out "$staging/server.key"
openssl req -new -sha256 -key "$staging/server.key" -subj "/CN=localhost" \
    -config "$staging/ext.cnf" -out "$staging/server.csr"
openssl x509 -req -sha256 -days 825 -in "$staging/server.csr" \
    -CA "$staging/ca.pem" -CAkey "$staging/ca.key" -set_serial "0x$(openssl rand -hex 16)" \
    -extfile "$staging/ext.cnf" -extensions server -out "$staging/server.pem" 2>/dev/null

cat >"$staging/pg_hba.conf" <<'EOF'
# TYPE   DATABASE  USER  ADDRESS  METHOD
local    all       all            trust
hostssl  all       all   all      trust
EOF

chmod 644 "$staging/ca.pem" "$staging/server.pem" "$staging/server.key" "$staging/pg_hba.conf"
mv "$staging/ca.pem" "$staging/server.pem" "$staging/server.key" "$staging/pg_hba.conf" "$dir/"
echo "wrote $dir (DATABASE_SSL_CERT is the contents of $dir/ca.pem)"
