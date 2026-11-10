# Private billing tool

This binary links `eidola-billing`, connects directly to PostgreSQL and Stripe, and runs with the caller's credentials. It is never an administrative gateway endpoint. Keep the CLI a presentation layer over the shared billing service; do not add another balance mutation path here. The `serve-webhooks` and `work` commands are separately deployable roles in the same binary/image; neither starts the other.

- Keep database and Stripe credentials in the environment, never command-line arguments, persisted support records, or output.
- Refunds name an explicit durable operation UUID. Reusing it means retrying the same request, not authorizing a second refund. Keep normal refunds and cash-only goodwill refunds visibly distinct.
- A preview makes no mutations. Execution repeats validation under the shared locks. Output is JSON for both human operators and private automation.
- Reconciliation reads authoritative provider state and applies recorded intents; it never treats a refund observation as permission to cancel account credits.
- `README.md` is the operator playbook. Keep it aligned with the shared service's failure states, including uncertain outcomes and expired reservations.
- `runtime.rs` owns only HTTP ingress, process lifecycle and the database-work retry loop. Stripe raw-body verification and the minimal event parser live in `eidola-billing::webhook`; there is one signature implementation, including multi-`v1` secret rotation. The receiver requires database access and its signing secret, never a Stripe API key; workers require the provider key and open no listener. Store only event identities, never webhook payloads.
- Receiver acknowledgment follows durable enqueue. The public `/healthz` checks database access and ingress compatibility, while worker execution checks the independent execution protocol. Do not infer financial completion from receiver health or successful delivery. Unknown job types/versions remain visible for review.
- SIGTERM/SIGINT interrupt initial protocol admission too, including a stalled database connection/query; poll shutdown while awaiting that check. After startup, stop new work and allow an active request/batch a bounded 30-second grace. Cancellation must drop database guards and leave pending work recoverable. The worker timer scans durable work only, never Stripe accounts. Logs contain fixed categories/counts, not account/provider identifiers or response bodies.
- `tests/receiver.rs` exercises real HTTP delivery and PostgreSQL durability, duplicates/conflicts, write failures, malformed/oversized input, rotation, protocol refusal and shutdown during a blocked startup protocol read for both runtime roles. Use a disposable `BILLING_TEST_DATABASE_URL` for its ignored tests. `runtime::tests` checks batch shutdown behavior; shared billing tests own multi-worker and financial race coverage.

Check billing changes against [anonymous credits](../../docs/anonymous-credits.md).
