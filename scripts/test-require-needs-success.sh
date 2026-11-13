#!/usr/bin/env bash
# Every case scripts/require-needs-success.py decides: only "all needed jobs
# succeeded" passes. Python 3 standard library only; no network.
set -euo pipefail

script="$(cd "$(dirname "$0")" && pwd)/require-needs-success.py"
failures=0

expect() {
  local want="$1" name="$2" json="$3" got=pass
  NEEDS_JSON="$json" python3 -I "$script" >/dev/null 2>&1 || got=fail
  if [ "$got" = "$want" ]; then
    echo "ok: $name ($got)"
  else
    echo "FAIL: $name: expected $want, got $got" >&2
    failures=$((failures + 1))
  fi
}

ok='{"result":"success","outputs":{}}'
expect pass "every need succeeded" "{\"guard\":$ok,\"oci\":$ok,\"apple\":$ok}"
expect fail "one need failed" "{\"guard\":$ok,\"oci\":{\"result\":\"failure\",\"outputs\":{}}}"
expect fail "one need was skipped" "{\"guard\":$ok,\"apple\":{\"result\":\"skipped\",\"outputs\":{}}}"
expect fail "one need was cancelled" "{\"guard\":$ok,\"oci\":{\"result\":\"cancelled\",\"outputs\":{}}}"
expect fail "the guard failed and everything after it was skipped" \
  '{"guard":{"result":"failure","outputs":{}},"oci":{"result":"skipped","outputs":{}}}'
expect fail "a need has no result" "{\"guard\":$ok,\"oci\":{\"outputs\":{}}}"
expect fail "the context is empty" '{}'
expect fail "the context is not an object" '[]'
expect fail "the context is not JSON" 'not json'
expect fail "the context is missing" ''

if [ "$failures" -ne 0 ]; then
  echo "$failures case(s) failed" >&2
  exit 1
fi
