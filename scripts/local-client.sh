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
# database (`~/Library/Application Support/eidola/eidola.db` on macOS), via
# the sanctioned `eidola configure` surface. Each of the four values it
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
# Environment overrides (all optional):
#   EIDOLA_DEV_CERT_DIR   shim cert directory      (default: .dev-certs)
#   EIDOLA_DEV_BASE_URL   shim URL                 (default: https://localhost:8443)
#   DEV_MEASUREMENT       measurement the shim advertises — the same variable
#                         the shim itself reads, forwarded to the container by
#                         compose.yaml, so one value governs both sides
#                                                  (default: 48 zero bytes)
#
# Only `enable` reads DEV_MEASUREMENT. `disable` drops the whole measurement
# override list unconditionally, so reverting can never depend on reproducing
# the environment that configured it.

set -euo pipefail
cd "$(dirname "$0")/.."

CERT_DIR="${EIDOLA_DEV_CERT_DIR:-.dev-certs}"
BASE_URL="${EIDOLA_DEV_BASE_URL:-https://localhost:8443}"

# 48 zero bytes as hex — what tinfoil-shim-mock advertises unless DEV_MEASUREMENT
# is set on the shim. All three fields (SNP, TDX rtmr1, TDX rtmr2) carry it.
ZERO_MEASUREMENT="000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
MEASUREMENT="${DEV_MEASUREMENT:-$ZERO_MEASUREMENT}"
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
    echo "    Run this once by hand (it needs sudo and changes the system trust store):"
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
    echo "    Leaving it in place is harmless (it only signs the local shim's"
    echo "    certificate), but to remove it:"
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

require_certs() {
    if [ ! -f "$ARK" ] || [ ! -f "$ASK" ]; then
        echo "ERROR: $ARK / $ASK not found." >&2
        echo "       The mock shim mints its cert set on first boot — start the" >&2
        echo "       stack once with \`just dev\` (or \`just services\`), then retry." >&2
        exit 1
    fi
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
    echo "==> Pointing the local client at $BASE_URL"
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

    Accounts and wallet credentials are per-server: the account in your
    config.toml was issued by whichever server you were pointed at. Create
    one on the local stack with \`eidola account create --accept-terms\`,
    and expect the reverse mismatch after \`just client-reset\`.
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
