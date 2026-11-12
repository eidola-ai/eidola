# Billing support playbook

`eidola-billing` provides private operator commands and separately runnable webhook receiver and worker roles. It connects directly to PostgreSQL and Stripe using the credentials of the human or workload running it. The gateway has no administrative refund or adjustment endpoint. Its account-holder repair endpoint records a durable request scoped to the authenticated account; workers reconcile that account and its stored Stripe customer.

The [terms of service](../../www/pages/terms.md) govern eligibility. This playbook describes how to carry out that policy; it does not narrow legal remedies or replace case-specific judgment.

## Setup and authority

Build with `cargo build -p eidola-billing-cli`. The binary is `eidola-billing`. Supply `DATABASE_URL`, optional `DATABASE_PASSWORD` and `DATABASE_SSL_CERT`, and `STRIPE_API_KEY` through the environment or the workload's secret facility. Database TLS uses the same verification rules as the gateway. Never put secrets in arguments, support notes, screenshots, or output files.

The database records its authenticated `current_user` for billing operations. Use distinct database and Stripe credentials for different operators/workloads; a typed operator name is not an authorization mechanism. Access control and deployment of this private tool are outside the public gateway.

Configure the Stripe billing portal for cancellation and payment-method management, without plan switching or prorations. Automatic grants support the fixed-price subscription periods and one-time purchases created by our checkout. A changed price, proration, or unsupported payment requires investigation; the CLI cannot refund or grant credits for a payment without a verified purchase record. Establish its provenance before an audited correction or separately recorded payment remedy, and reconcile any provider refund. Do not invent a credit amount to make an unsupported payment fit.

Commands return JSON. Account and payment references in that output are private support data. Keep case references concise; do not copy customer messages or payment details into reasons. No command inspects, identifies, redeems, or revokes ACTs.

## Receiver and worker processes

Run the two roles independently, with as many replicas of each as needed:

```sh
eidola-billing serve-webhooks --bind 0.0.0.0:8080
eidola-billing work --batch-size 32 --retry-interval-secs 5
```

The receiver needs the database settings above and `STRIPE_WEBHOOK_SECRET`. It does **not** need `STRIPE_API_KEY`, and does not execute reconciliation or start workers. Direct Stripe webhooks to `POST /v1/webhooks/stripe` on this stable deployment. It serves plain HTTP behind the deployment's HTTPS termination, including a Tinfoil shim if desired. `GET /healthz` checks database connectivity and the ingress protocol. The receiver limits request bodies to 2 MiB, authenticates the exact raw bytes with at most five minutes of clock skew in either direction, and stores only the supported event's ID, type, and object ID. Keep the host clock synchronized. It acknowledges a supported event only after the database commits it; a failed write returns a non-success response for Stripe to retry. Unsupported event types are acknowledged without storing them, including events whose objects have no ID.

The worker needs the database settings and `STRIPE_API_KEY`, but no webhook secret. It opens no HTTP listener. Its timer retries pending database work, including webhook observations, account repair requests, and interrupted refunds/closures; it never periodically scans Stripe accounts. Each process handles one batch at a time. Scale worker containers for parallel accounts. Database locks coordinate duplicate deliveries, multiple workers, and direct support commands; durable operation identities and Stripe idempotency remain necessary when a process loses a provider response.

Both roles handle SIGTERM and SIGINT. Receivers stop accepting connections and allow active requests up to 30 seconds to finish. Workers stop taking new batches and allow an active batch up to 30 seconds to finish. After that, incomplete work remains durable and its database locks are released when the process exits. Set the container termination grace longer than 30 seconds. Runtime logs report fixed failure categories/counts to stderr without event bodies, account references, or credentials.

The image recipe is [oci/eidola-billing/Containerfile](../../oci/eidola-billing/Containerfile). Build from the repository root:

```sh
docker buildx build --platform linux/amd64 --load \
  -f oci/eidola-billing/Containerfile -t eidola-billing .
docker run --rm --env-file /path/to/receiver.env -p 8080:8080 \
  eidola-billing serve-webhooks --bind 0.0.0.0:8080
docker run --rm --env-file /path/to/worker.env \
  eidola-billing work --batch-size 32 --retry-interval-secs 5
```

The same non-root, static image runs either role or an operator command; no role starts implicitly. Keep secret files outside the repository. Production hostname, TLS, credentials, replication, monitoring, and deployment manifests belong to the operator's deployment configuration, which may live in a private repository. Monitor pending-work age and repeated failures in `billing_event`, and alert on aging `submitting` refunds well before their 23-hour safe retry window closes; a healthy receiver confirms durable delivery, not that the worker has completed the financial effect.

Ingress and financial execution have separate database protocol versions. A receiver or gateway may continue recording v1 requests while compatible workers evolve. Workers and direct operator commands refuse incompatible execution versions before performing financial work. An incompatible change requires stopping and draining old workers/operator processes before changing the execution version/schema and starting the new build; do not change that version underneath an admitted operation. Unsupported job kinds or versions remain pending for review rather than being silently marked complete.

## Inspect and repair

```sh
eidola-billing inspect --account "$BILLING_ACCOUNT_ID"
eidola-billing reconcile --account "$BILLING_ACCOUNT_ID"
eidola-billing drain-events --limit 32
```

`inspect` reads local records. `reconcile` fetches current Stripe state, verifies purchase provenance, and atomically applies that account's purchases, refunds, and disputes. It also resumes already-approved refund operations and account closures. It cannot choose a new discretionary credit effect.

The stable receiver durably records verified webhook identities before acknowledging them. Separately deployed workers drain this database work queue, including interrupted internal operations. Provider facts are fetched through the same reconciliation path. Events can arrive twice or out of order without creating a second business operation.

Account holders request repair through authenticated `POST /v1/account/reconcile` and read its status through `GET /v1/account/reconcile`; neither requires accepting new terms or a configured gateway Stripe client. The response tracks requested and completed generations so a request arriving during a running repair remains pending. A completed report records observations and unresolved review items; it is not a promise that a pending Stripe refund has settled. Direct `eidola-billing reconcile` remains available to an operator under the same account lock.

Read `review` items in the result. Unknown grant terms, unsupported payment funding, or an ambiguous old refund attempt require investigation. Do not guess cash value from credit face value. An unexplained external refund is recorded without canceling credits; decide any remedy explicitly.

## Refund an eligible balance

1. Establish the account and purchase from the authenticated account or payment evidence. A payment receipt can support cancellation/refund without restoring a lost account secret. Verify evidence before acting; an account ID alone is not proof of entitlement.
2. Review the remaining unconverted credits, purchase price, expiration, prior refunds, disputes, and the time the request was received. For multiple purchases, each is valued at its own price.
3. Generate an operation UUID and retain it with the case. Quote the request, then execute that exact request using the same UUID.

```sh
eidola-billing new-operation
eidola-billing refund --account "$BILLING_ACCOUNT_ID" \
  --purchase "$BILLING_PURCHASE_ID" --operation "$BILLING_OPERATION_ID" \
  --credits 1000000 --reason 'case 123: unused balance requested'
eidola-billing refund --account "$BILLING_ACCOUNT_ID" \
  --purchase "$BILLING_PURCHASE_ID" --operation "$BILLING_OPERATION_ID" \
  --credits 1000000 --reason 'case 123: unused balance requested' --execute
```

The quote makes no mutations and can become stale. Execution rechecks current state, reserves the exact credits in the database, then calls Stripe, then records the result. The reservation prevents concurrent conversion. Rounding the cash amount never consumes extra credits. A repeated operation UUID with different instructions is refused.

For a delayed support request whose eligible credits expired before an operator could reserve them, do not reject the request merely because time passed. Review the receipt timestamp and account ledger, excluding credits converted after the request. Use an explicitly valued cash-only remedy for credits that have already expired; record the case and basis. This is a policy remedy when owed, not necessarily goodwill. The tool does not reconstruct a backdated refund request automatically.

## Cash-only exceptions and remedies

```sh
eidola-billing refund --account "$BILLING_ACCOUNT_ID" \
  --purchase "$BILLING_PURCHASE_ID" --operation "$BILLING_OPERATION_ID" \
  --cash-minor 500 --reason 'case 124: reviewed cash-only remedy'
```

Add `--execute` after reviewing the quote. `--cash-minor` is expressed in the original payment currency's smallest unit. It cannot be combined with `--credits`, and it causes no credit deduction. Use it for approved goodwill, non-delivery remedies, or other reviewed exceptions. A cash-only refund still counts toward the amount already returned for that payment.

Examples needing judgment include a subscription that renewed while its holder reviewed new terms, a timely refund request handled after expiration, and a valid service-failure claim despite ACT conversion. An ACT conversion is not proof that usable service was delivered. Preserve mandatory remedies; do not promise recovery or tracing of the customer's anonymous tokens.

## Interrupted or failed refunds

```sh
eidola-billing resume-refund --operation "$BILLING_OPERATION_ID"
```

Use the same operation after a timeout, lost response, process crash, or uncertain provider error. Reconciliation first searches for the refund under its durable operation metadata. A safe retry uses the same Stripe idempotency key. An uncertain attempt older than the conservative retry window is refused for review; generating a new UUID is not a remedy for uncertainty.

A rejected retry does not prove that the original request failed. For example, a changed API key may allow reads but refuse refund creation while Stripe is still recovering the first request. The operation stays `submitting`; do not release its credits, replace the attempt, or arrange an alternate payout until the original outcome is established.

If the retry window expires with no refund object, preserve the hold and escalate to Stripe using the operation's idempotency key (`eidola/refund/v1/<operation UUID>`) and request records. An empty refund list is not proof that an earlier request cannot still complete. A crash before the POST, or a database failure after receiving a rejection, can also leave this conservative state. There is deliberately no force-fail command: resolve the provider outcome first, then arrange a reviewed, audited repair under the account's normal locks if no provider object can resolve it. Never bypass the hold merely to finish closure.

| State | Operator action |
| --- | --- |
| `prepared` | Resume to send the authorized refund, or release it if it should not proceed. |
| `submitting` | Outcome is uncertain. Resume/reconcile; do not release or create an unrelated replacement. |
| `pending`, `requires_action` | Follow the provider's pending/action state. Webhooks reconcile the result; credits remain reserved. |
| `succeeded` | The recorded cash return and credit effect are complete. |
| `failed`, `canceled` | Keep the reservation while choosing a remedy. Retry with a new attempt ID, arrange an alternate payout, or release eligible credits. |
| `released` | Credits were restored. This operation cannot send money later. |
| `superseded` | A replacement attempt owns the same reservation. Follow the replacement operation. |

```sh
eidola-billing retry-refund --operation "$BILLING_OPERATION_ID" \
  --new-operation "$BILLING_REPLACEMENT_ID" --reason 'case 123: definitive failure, retry'
eidola-billing release-refund --operation "$BILLING_OPERATION_ID"
```

A replacement is allowed only after definitive failure/cancellation and transfers the existing reservation; it does not reserve or debit credits again. This preserves a timely request even after the original credits expire. Releasing an expired hold is refused: restoring already-expired credits would erase the customer's frozen refund entitlement.

If the original payment method cannot receive the refund, arrange an approved alternate payout and retain its evidence. Then record the completed payout:

```sh
eidola-billing confirm-external-refund --operation "$BILLING_OPERATION_ID" \
  --reference 'case 123: alternate payout reference'
```

This command records money already returned; it does not send money. It is refused for an ambiguous in-flight Stripe attempt. Establish that there is no competing payout before paying by another route. The external return remains part of the payment's refund accounting.

Later observations of the failed Stripe refund cannot undo the alternate settlement. An unexpected additional Stripe return is recorded as separate cash and flagged for review. Releasing a hold or recording an alternate payout on a closing account also queues another closure pass, so no webhook or repeated close command is needed.

## Subscriptions and account closure

```sh
eidola-billing cancel-subscriptions --account "$BILLING_ACCOUNT_ID"
eidola-billing close --account "$BILLING_ACCOUNT_ID" --reason 'case 125: customer closure'
eidola-billing close --account "$BILLING_ACCOUNT_ID" --reason 'case 125: customer closure' --execute
```

Cancellation stops renewals; granted credits keep their expiry. It does not implicitly refund them. Close without `--execute` displays the current records. Executing closure freezes new purchases/conversions first, preserves the eligible balance at closure, cancels subscriptions, and refunds that balance after prior returns and chargebacks. Pending refunds leave closure resumable. Repeating the command, account repair, or a queued job resumes the recorded closure.

A checkout already underway can finish after closure. Reconciliation records that payment and refunds its credits instead of leaving a closed account with unusable paid value. Closing an account never revokes previously issued ACTs. Deleting credentials on a device is not account closure or cancellation.

Reviewing new terms does not itself cancel a subscription or authorize closure. The existing terms and renewal authorization continue until acceptance or cancellation; handle exceptional refund requests individually.

## Chargebacks and externally initiated refunds

No operator command should mimic a chargeback by manually debiting credits. Reconcile the authoritative dispute. Inquiries move no funds. A real reversal deducts its full cash value, using the original purchase prices of affected credits, including across purchases; any uncovered debt persists separately from credit expiry. Prior refunds and reversals are accounted for once. A win undoes the recorded effect with the original expiration dates.

A dispute alone does not establish fraud or close the account. Any remaining positive spendable balance stays usable; already-issued ACTs cannot be traced or clawed back. Payment blocking belongs in Stripe and privately authorized support actions. Determine fraud or termination independently of ledger repair.

If someone refunds through the Stripe dashboard, reconcile it. The cash record must agree with Stripe, but there is no inferred permission to cancel credits. The result is flagged for review. Use the internal tool for future refunds so the intended credit effect is recorded before the cash request.
