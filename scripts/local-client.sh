#!/usr/bin/env bash
# Point the local Eidola client at the local dev stack — or put it back on
# the built-in trust-root pins.
#
# Usage:
#   scripts/local-client.sh enable     # write the dev overrides
#   scripts/local-client.sh disable    # remove them (back to the built-in pins)
#   scripts/local-client.sh status     # print the client's current trust bundle
#
# Wrapped by `just client-local` / `just client-reset`.
#
# What this touches: nothing but the `eidola` backend row in *your* local
# database — `eidola.db` in whatever data directory the CLI resolves for this
# platform (`eidola` prints its config path) — via the sanctioned `eidola
# configure` surface. Each of the four values it
# writes — base URL, ARK, ASK, trusted measurement — is a per-column
# override of the trust-root pin compiled into the binary; `disable` clears
# every one back to NULL, which *is* the pin. No release behavior is
# involved and no verification is weakened: the client still runs the full
# per-handshake SEV-SNP attestation, just against the mock shim's chain.
#
# The GUI and the CLI share one profile (both resolve `dirs::config_dir()` /
# `dirs::data_dir()`), so this configures both at once — and because the
# local database is single-writer, any running Eidola must be quit first.
#
# The one step this script does *not* perform is trusting the mock TLS root
# in your OS trust store: that needs `sudo`, and the exact invocation is
# left for a human to run. It is printed when it's needed, and the
# ready-to-uncomment lines are in `trust_ca_note` / `untrust_ca_note` below.
#
# Settings (all optional), read from the environment first and then from
# `.env` (see `dotenv_get`). They fall into two deliberately different
# classes:
#
#   DEV_MEASUREMENT       the measurement the shim advertises — literally the
#                         same scalar on both sides, so one value has to
#                         govern both. `compose.yaml` forwards it to the shim
#                         container and this script trusts it on the client;
#                         set it once, in the environment or `.env`, and
#                         `just dev` + `just client-local` agree.
#                                                  (default: 48 zero bytes)
#
#   EIDOLA_DEV_CERT_DIR   where the shim's cert set already is (host path)
#                                                  (default: .dev-certs)
#   EIDOLA_DEV_BASE_URL   where the shim already listens
#                                                  (default: https://localhost:8443)
#
#                         These two are *client-side only* and deliberately
#                         not consumed by compose: they exist to point the
#                         client at a stack that is not the default compose
#                         one — a shim someone else runs, one on another
#                         port, or (the common case) a client in one worktree
#                         talking to the stack in another checkout. They have
#                         no compose counterpart to follow: the shim's
#                         `CERT_DIR` is a *container* path whose host side is
#                         the bind-mount source, and the client-visible URL
#                         is the published port mapping. Moving the stack
#                         itself means editing that volume/ports pair in
#                         `compose.yaml` — at which point you point the
#                         client at the result with these.
#
# Only `enable` reads DEV_MEASUREMENT. `disable` drops the whole measurement
# override list unconditionally, so reverting can never depend on reproducing
# the environment that configured it — or on `.env` being well-formed.

set -euo pipefail
# `BASH_SOURCE`, not `$0`: everything below resolves paths relative to the
# repo root, and `$0` is the *interpreter* when this file is sourced (which
# the tests do, to exercise the functions directly).
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Read one key out of `.env`, without executing it. Compose reads `.env` from
# the project directory automatically, and the justfile's `set dotenv-load`
# gives every recipe the same view — so a value that lives there reaches both
# `just dev` (the shim) and `just client-local` (the client). Running this
# script directly has to see it too, or the shim advertises one measurement
# while the client trusts another. Environment wins over `.env`, matching
# what both compose and just do.
#
# Deliberately not `source .env`: that executes arbitrary shell, and this
# file holds real secrets (TINFOIL_API_KEY, STRIPE_API_KEY) that would then
# be exported into every process this script spawns.
#
# The supported subset of compose's `.env` grammar, verified against
# `docker compose config`: `export` prefix, single or double quotes, an
# inline `#` comment after whitespace (or after a closing quote), and a bare
# `#` inside an unquoted value staying literal. The one thing compose does
# that this deliberately does not is interpolate `${VAR}` — reimplementing
# that is reimplementing compose — so a value containing `${` warns and is
# passed through literally, where the caller's validation rejects it loudly.
dotenv_get() {
    local key="$1" line
    [ -f .env ] || return 0
    line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?${key}[[:space:]]*=" .env | tail -n 1 || true)"
    [ -n "$line" ] || return 0
    line="${line#*=}"
    line="${line%$'\r'}"                       # tolerate CRLF
    line="${line#"${line%%[![:space:]]*}"}"    # trim leading space
    if [[ "$line" =~ ^\"([^\"]*)\" ]] || [[ "$line" =~ ^\'([^\']*)\' ]]; then
        # Quoted: the quoted span is the value; anything after it is comment.
        line="${BASH_REMATCH[1]}"
    else
        line="${line%%[[:space:]]#*}"          # ` # comment` — `a#b` is literal
        line="${line%"${line##*[![:space:]]}"}" # trim trailing space
    fi
    if [[ "$line" == *'${'* ]]; then
        echo "WARNING: $key in .env contains \${...}, which this script does not" >&2
        echo "         interpolate (compose does). Export the variable instead." >&2
    fi
    printf '%s' "$line"
}

# Environment, then `.env`, then the built-in default.
setting() {
    local name="$1" default="$2" value="${!1:-}"
    [ -n "$value" ] || value="$(dotenv_get "$name")"
    printf '%s' "${value:-$default}"
}

CERT_DIR="$(setting EIDOLA_DEV_CERT_DIR .dev-certs)"
BASE_URL="$(setting EIDOLA_DEV_BASE_URL https://localhost:8443)"

# 48 zero bytes as hex — what tinfoil-shim-mock advertises unless DEV_MEASUREMENT
# is set on the shim. All three fields (SNP, TDX rtmr1, TDX rtmr2) carry it.
ZERO_MEASUREMENT="000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
MEASUREMENT="$(setting DEV_MEASUREMENT "$ZERO_MEASUREMENT")"
MEASUREMENT_TRIPLE="$MEASUREMENT:$MEASUREMENT:$MEASUREMENT"

TLS_CA="$CERT_DIR/tls-ca.pem"
ARK="$CERT_DIR/ark.pem"
ASK="$CERT_DIR/ask.pem"
# Where the Linux instructions file the mock TLS root.
LINUX_CA_PATH="/usr/local/share/ca-certificates/eidola-dev.crt"

cli() {
    cargo run -q -p eidola-cli -- "$@"
}

# The host OS, as a function so tests can stub it.
platform() {
    uname -s
}

# ── Mock TLS root: report, never mutate ──────────────────────────────────────
#
# `tls-ca.pem` is the shim's *TLS* anchor (RSA PKCS#1 v1.5) — deliberately a
# different identity from the SEV-SNP ARK/ASK, which stay out of the OS trust
# store (Apple's Security framework can't chain-validate RSA-PSS). Trusting it
# is a privileged, system-wide change, so this script only tells you the
# command; the mutation itself is the commented block in each function.

# Best-effort, read-only: is *this* mock TLS root already trusted?
# Prints `trusted`, `untrusted`, or `unknown`.
#
# Both branches answer for the current `tls-ca.pem`, not for "some eidola dev
# cert once existed" — deleting `.dev-certs/` mints a new root, and a check
# that ignored which cert is installed would suppress the trust command and
# leave TLS failing before attestation ever runs. On macOS that falls out of
# `verify-cert`, which evaluates this cert against the store. On Linux the
# installed file is a verbatim copy of the source, so comparing bytes is the
# equivalent question (and needs no openssl).
ca_trust_state() {
    if [ ! -f "$TLS_CA" ]; then
        echo "unknown"
        return
    fi
    case "$(platform)" in
        Darwin)
            # `verify-cert` evaluates trust; it writes nothing.
            if security verify-cert -c "$TLS_CA" -p ssl >/dev/null 2>&1; then
                echo "trusted"
            else
                echo "untrusted"
            fi
            ;;
        Linux)
            if [ -f "$LINUX_CA_PATH" ] && cmp -s "$TLS_CA" "$LINUX_CA_PATH"; then
                echo "trusted"
            else
                echo "untrusted"
            fi
            ;;
        *) echo "unknown" ;;
    esac
}

trust_ca_note() {
    local state
    state="$(ca_trust_state)"
    if [ "$state" = "trusted" ]; then
        echo "==> Mock TLS root already trusted ($TLS_CA)."
        return
    fi
    echo "==> Mock TLS root is NOT trusted by this machine ($state)."
    echo "    Run this once by hand (it needs sudo and changes the system trust"
    echo "    store). Note what you are agreeing to: this root's private key is a"
    echo "    file in $CERT_DIR, and trusting it machine-wide means anyone who"
    echo "    can read that file can mint a certificate this machine accepts for"
    echo "    any host. \`just client-reset\` prints the command to undo it."
    case "$(platform)" in
        Darwin)
            echo ""
            echo "    sudo security add-trusted-cert -d -r trustRoot \\"
            echo "        -k /Library/Keychains/System.keychain $TLS_CA"
            echo ""
            ;;
        Linux)
            echo ""
            echo "    sudo cp $TLS_CA $LINUX_CA_PATH && sudo update-ca-certificates"
            echo ""
            ;;
        *)
            echo "    (unrecognized platform — see the CLI section of README.md)"
            ;;
    esac

    # KEYCHAIN MUTATION — deliberately left commented out. It needs sudo,
    # rewrites the system-wide trust store, and has NOT been executed or
    # validated by the author of this script. Uncomment (and delete the
    # printing above) once you have run it by hand and are happy with the
    # exact invocation.
    #
    # case "$(platform)" in
    #     Darwin)
    #         sudo security add-trusted-cert -d -r trustRoot \
    #             -k /Library/Keychains/System.keychain "$TLS_CA"
    #         ;;
    #     Linux)
    #         sudo cp "$TLS_CA" "$LINUX_CA_PATH" && sudo update-ca-certificates
    #         ;;
    # esac
}

untrust_ca_note() {
    if [ "$(ca_trust_state)" != "trusted" ]; then
        return
    fi
    echo "==> The mock TLS root is still trusted by this machine."
    echo "    Worth removing when you are done with local dev: its private key"
    echo "    sits in $CERT_DIR, and whoever can read that key can mint a"
    echo "    certificate for ANY host this machine will then accept. The shim"
    echo "    keeps the key owner-only, but the trust itself is machine-wide"
    echo "    and outlives the stack. To remove it:"
    case "$(platform)" in
        Darwin)
            echo ""
            echo "    sudo security remove-trusted-cert -d $TLS_CA"
            echo ""
            ;;
        Linux)
            echo ""
            echo "    sudo rm $LINUX_CA_PATH && sudo update-ca-certificates --fresh"
            echo ""
            ;;
    esac

    # KEYCHAIN MUTATION — see the note in `trust_ca_note`. Same reasoning,
    # same status: unvalidated, so left commented out.
    #
    # case "$(platform)" in
    #     Darwin)
    #         sudo security remove-trusted-cert -d "$TLS_CA"
    #         ;;
    #     Linux)
    #         sudo rm "$LINUX_CA_PATH" && sudo update-ca-certificates --fresh
    #         ;;
    # esac
}

# ── Commands ─────────────────────────────────────────────────────────────────

# `-s`, not `-f`: an empty cert file is the failure mode of a half-finished
# copy or a mount that didn't land, and it must not reach `configure` (where
# blank input used to read as clear-to-pin). The CLI rejects blank contents
# too — this just fails earlier, with the hint that names the cause.
require_certs() {
    # All three, including the TLS root: the shim mints them in order (ARK,
    # ASK, then TLS-CA), so an interrupted first boot leaves a partial set —
    # and a set without `tls-ca.pem` used to pass, commit every override, and
    # leave the client redirected to a shim whose chain nothing can verify,
    # with the trust command naming a file that isn't there.
    local missing=()
    local f
    for f in "$ARK" "$ASK" "$TLS_CA"; do
        [ -s "$f" ] || missing+=("$f")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: missing or empty in $CERT_DIR:" >&2
        for f in "${missing[@]}"; do
            echo "         $(basename "$f")" >&2
        done
        echo "       The mock shim mints its cert set on first boot — start the" >&2
        echo "       stack once with \`just dev\` (or \`just services\`), then retry." >&2
        echo "       (Delete .dev-certs/ to have it mint a fresh set.)" >&2
        exit 1
    fi
}

# ── Server-bound profile state ───────────────────────────────────────────────
#
# Only the connection + trust bundle is per-column overridable; the account
# (config.toml) and the wallet (local DB) are not, and neither recipe moves
# them. That is not fixable from a script — a profile carries one account and
# one credential set — so the honest thing is to say precisely what stays
# behind and how to get out of each consequence.
#
# Both reads below are local (config.toml + local DB) with no network, so the
# pre-flight works with no stack running — verified by running them against an
# unreachable base URL.

# The config file, taken from the CLI's own answer rather than guessed.
# `dirs::config_dir()` resolves somewhere different on every platform, and
# this file holds the only copy of an account secret the user is about to be
# told to destroy — naming the wrong one sends a Linux user to `account
# reset` without ever finding their id/secret. Falls back to naming the
# command when the CLI reports no resolvable path (`None`).
config_path_from_status() {
    local path
    path="$(printf '%s\n' "$1" | sed -n 's/^config path: Some("\(.*\)")$/\1/p')"
    if [ -n "$path" ]; then
        printf '%s' "$path"
    else
        printf '%s' '(run `eidola` — its "config path" line names the file)'
    fi
}

# Prints "<in-flight> <active>" — the two sections of `wallet credentials
# list`, counted.
credential_counts() {
    cli wallet credentials list 2>/dev/null | awk '
        /^in-flight credentials:/ { s = "f"; next }
        /^active credentials:/    { s = "a"; next }
        /^[[:space:]]*$/          { s = "";  next }
        s == "f" { f++ }
        s == "a" { a++ }
        END { printf "%d %d", f + 0, a + 0 }'
}

# Pre-flight for `enable`: what this profile is about to carry into a stack
# that did not issue it. Silent on a clean profile.
warn_server_bound_state() {
    local in_flight active status
    read -r in_flight active <<<"$(credential_counts)"
    status="$(cli)"

    if printf '%s\n' "$status" | grep -q '^account_id: <set>'; then
        # Quoted heredocs (the literal backticks are prose, not commands),
        # with the resolved path echoed between them.
        cat <<'EOF'
==> WARNING: this profile has an account, and it belongs to the server you
    are leaving. The local stack has never heard of it.

    `eidola account create` REFUSES while one is configured ("account
    credentials already configured — reset first"), so making an account on
    the local stack means running `eidola account reset` first — and that
    DISCARDS the id and secret from config.toml. Copy them somewhere first:

EOF
        echo "        $(config_path_from_status "$status")"
        cat <<'EOF'

    `eidola account configure --id <id> --secret <secret>` puts them back.

EOF
    fi

    if [ "${active:-0}" -gt 0 ]; then
        cat <<EOF
==> WARNING: $active active credential(s) in this wallet were issued by the
    server you are leaving. Credential selection has no issuer filter, so a
    turn against the local stack can pick one, mark it in-flight, and then
    fail — the local issuer cannot verify a credential it never minted.

    Such a credential is parked, not lost: after \`just client-reset\` run
        eidola wallet credentials recover
    which replays the stored spend proof to the server that issued it.

EOF
    fi
}

# Closing note for `disable`: anything the local stack left in flight can be
# settled now that the pinned service is back.
note_in_flight_credentials() {
    local in_flight active
    read -r in_flight active <<<"$(credential_counts)"
    [ "${in_flight:-0}" -gt 0 ] || return 0
    cat <<EOF
==> $in_flight credential(s) are still in flight. You are back on the pinned
    service, so if they were issued by it, settle them now:

        eidola wallet credentials recover

EOF
}

warn_if_shim_down() {
    command -v curl >/dev/null 2>&1 || return 0
    # -k: liveness only. Whether the chain verifies is exactly what the OS
    # trust check above answers, and what the client itself will enforce.
    if ! curl -sk --max-time 2 -o /dev/null "$BASE_URL/.well-known/tinfoil-attestation"; then
        echo "==> WARNING: nothing answered at $BASE_URL — start the stack with \`just dev\`."
    fi
}

cmd_enable() {
    require_certs
    # Belt-and-braces over the CLI's own parse-before-write validation: the
    # measurement can now arrive from `.env`, and naming the source beats a
    # bare parse error. A shape this rejects would also panic the shim.
    if ! printf '%s' "$MEASUREMENT" | grep -qE '^[0-9a-fA-F]{96}$'; then
        echo "ERROR: DEV_MEASUREMENT must be 96 hex characters (48 bytes), got:" >&2
        echo "       $MEASUREMENT" >&2
        echo "       (checked the environment, then .env)" >&2
        echo "       If that value came from .env, note this script does not" >&2
        echo "       interpolate \${...} the way compose does — export the" >&2
        echo "       variable instead." >&2
        exit 1
    fi
    echo "==> Pointing the local client at $BASE_URL"
    # Before the first write, so there is something to abort on.
    warn_server_bound_state
    trust_ca_note
    cli configure \
        --base-url "$BASE_URL" \
        --hardware-root-ca "$ARK" \
        --hardware-intermediate-ca "$ASK" \
        --trust-measurement "$MEASUREMENT_TRIPLE"
    echo ""
    warn_if_shim_down
    cmd_status
    cat <<EOF

==> The GUI and the CLI share this profile, so both now talk to the local
    stack. Undo with \`just client-reset\`.

    Only the connection + trust bundle moved. If this profile has no account
    yet, \`eidola account create --accept-terms\` makes one on the local
    stack; if it has one, see the warning above — it is not the local
    stack's, and creating a local one is destructive to it.
EOF
}

cmd_disable() {
    echo "==> Reverting the local client to the built-in trust-root pins"
    # Every flag clears a column to NULL unconditionally, so the revert is
    # complete whatever `enable` wrote and whatever is in the environment now
    # — including a DEV_MEASUREMENT that was set then and isn't set here.
    # (Untrusting by key instead would leave the column non-NULL, and a
    # non-NULL list that omits the pinned measurement rejects the real
    # enclave: a "reset" that silently breaks production.) Idempotent:
    # clearing an absent override is a no-op the CLI reports honestly.
    cli configure \
        --clear-base-url \
        --clear-hardware-root-ca \
        --clear-hardware-intermediate-ca \
        --clear-trusted-measurements
    echo ""
    note_in_flight_credentials
    untrust_ca_note
    cmd_status
}

cmd_status() {
    echo "==> Client trust bundle:"
    cli | sed 's/^/    /'
}

# Guarded so the functions above can be sourced and exercised directly
# (`platform` and `LINUX_CA_PATH` are the stubbing seams).
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        enable) cmd_enable ;;
        disable) cmd_disable ;;
        status) cmd_status ;;
        -h | --help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "ERROR: expected one of: enable, disable, status" >&2
            echo "Usage: $0 {enable|disable|status}" >&2
            exit 1
            ;;
    esac
fi
