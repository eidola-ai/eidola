# Billing support playbook

`eidola-billing` is a private operator tool. It connects directly to PostgreSQL and Stripe using the credentials of the human or workload running it. The gateway has no administrative refund or adjustment endpoint. Its account-holder repair endpoint invokes the same reconciliation code, scoped to the authenticated account and its stored Stripe customer.

The [terms of service](../../www/pages/terms.md) govern eligibility. This playbook describes how to carry out that policy; it does not narrow legal remedies or replace case-specific judgment.

## Setup and authority

Build with `cargo build -p eidola-billing-cli`. The binary is `eidola-billing`. Supply `DATABASE_URL`, optional `DATABASE_PASSWORD` and `DATABASE_SSL_CERT`, and `STRIPE_API_KEY` through the environment or the workload's secret facility. Database TLS uses the same verification rules as the gateway. Never put secrets in arguments, support notes, screenshots, or output files.

The database records its authenticated `current_user` for billing operations. Use distinct database and Stripe credentials for different operators/workloads; a typed operator name is not an authorization mechanism. Access control and deployment of this private tool are outside the public gateway.

Commands return JSON. Account and payment references in that output are private support data. Keep case references concise; do not copy customer messages or payment details into reasons. No command inspects, identifies, redeems, or revokes ACTs.

## Inspect and repair

```sh
eidola-billing inspect --account "$BILLING_ACCOUNT_ID"
eidola-billing reconcile --account "$BILLING_ACCOUNT_ID"
eidola-billing drain-events --limit 32
```

`inspect` reads local records. `reconcile` fetches current Stripe state, verifies purchase provenance, and atomically applies that account's purchases, refunds, and disputes. It also resumes already-approved refund operations and account closures. It cannot choose a new discretionary credit effect.

The gateway durably records verified webhook identities before acknowledging them. Its worker drains this same database work queue, including interrupted internal operations. Provider facts are fetched through the same reconciliation path; there is no independent periodic Stripe scan. Events can arrive twice or out of order without creating a second business operation.

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
