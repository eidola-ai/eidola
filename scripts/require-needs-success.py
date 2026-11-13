#!/usr/bin/env python3
"""Fail unless every job a workflow job needs succeeded.

A job whose `needs` did not all succeed is skipped by default, and GitHub
counts a skipped job as satisfying a required status check, as does
`gh pr checks`. A job that must give a verdict therefore runs
`if: ${{ !cancelled() }}` and calls this first, with its `needs` context:

    env:
      NEEDS_JSON: ${{ toJSON(needs) }}
    run: python3 scripts/require-needs-success.py

Exit 0 only if the context names at least one job and every job's `result`
is `success`; otherwise print each job that did not succeed and exit 1. A
malformed or empty context is a failure, never a pass.

Python 3 standard library only.
"""

import json
import os
import sys


def verdict(raw):
    """Return (ok, lines) for a `needs` context serialized as JSON."""
    try:
        needs = json.loads(raw)
    except (TypeError, ValueError) as err:
        return False, [f"the needs context is not JSON: {err}"]
    if not isinstance(needs, dict) or not needs:
        return False, ["the needs context names no job"]
    lines = []
    ok = True
    for job in sorted(needs):
        entry = needs[job]
        result = entry.get("result") if isinstance(entry, dict) else None
        if result == "success":
            lines.append(f"ok: {job}: success")
        else:
            ok = False
            lines.append(f"error: {job}: {result if isinstance(result, str) else 'no result'}")
    return ok, lines


def main():
    ok, lines = verdict(os.environ.get("NEEDS_JSON"))
    for line in lines:
        print(line, file=sys.stdout if line.startswith("ok:") else sys.stderr)
    if not ok:
        print("error: a job this one needs did not succeed; this check cannot pass", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
