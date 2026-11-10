# Private billing tool

This binary links `eidola-billing`, connects directly to PostgreSQL and Stripe, and runs with the caller's credentials. It is never an administrative gateway endpoint. Keep the CLI a presentation layer over the shared billing service; do not add another balance mutation path here.

- Keep database and Stripe credentials in the environment, never command-line arguments, persisted support records, or output.
- Refunds name an explicit durable operation UUID. Reusing it means retrying the same request, not authorizing a second refund. Keep normal refunds and cash-only goodwill refunds visibly distinct.
- A preview makes no mutations. Execution repeats validation under the shared locks. Output is JSON for both human operators and private automation.
- Reconciliation reads authoritative provider state and applies recorded intents; it never treats a refund observation as permission to cancel account credits.
- `README.md` is the operator playbook. Keep it aligned with the shared service's failure states, including uncertain outcomes and expired reservations.

Check billing changes against [anonymous credits](../../docs/anonymous-credits.md).
