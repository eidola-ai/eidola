# Billing — shared account operations

This crate owns the account billing behavior used by the gateway and the private `eidola-billing-cli`. Neither surface has its own refund or chargeback algorithm. Read [anonymous-credit doctrine](../../docs/anonymous-credits.md) for every change: these tables belong to the linked account surface. No ACT spend, nullifier, recovery key, inference request or client spend trail belongs here.

## Boundaries

- `stripe.rs` owns the pinned provider transport and normalized observations. A webhook is a wakeup, not an instruction to credit or debit its payload.
- `service.rs` owns provider reads and the durable prepare → submit → observe workflow. The CLI connects directly to PostgreSQL and Stripe with its caller's credentials. The gateway exposes only account-scoped reconciliation.
- `store.rs` owns transactions and `schema/billing.sql` owns allocation functions. The latter is embedded verbatim between the shared-billing markers in the gateway's fresh baseline, `../eidola-server-gateway/schema/schema.sql`. `gateway_and_library_share_exact_billing_schema` checks the copies and that there is exactly one debit and balance function. Update both copies together. There is no historical-data migration requirement for this pre-production baseline; do not add a lossy backfill or a compatibility mutation path.

## Concurrency and identity

Every balance mutation takes the same account row lock as credential issuance. `record_credential_issuance` answers its stored idempotent response first, then checks the issuer key and account state under that lock. Closing is irreversible: a new issuance returns `closed`, while retries of paid issuances still recover.

Purchases are identified by their immutable provider source, never an event ID. Each purchase records the actual amount paid, currency, credits, expiry and charge/invoice/PaymentIntent references. Equal expiration dates do not merge purchase prices. Ledger foreign keys bind an allocation to its purchase, account and original expiry; payment effects also name a durable billing operation.

Provider reads and their application use `ReconciliationGuard`. It holds a session advisory lock on a detached connection; snapshot transactions commit on that same connection. A lost lock connection cannot apply a stale snapshot through another one. Dropping the guard closes the connection instead of returning a held lock to the pool, and a process-wide semaphore bounds the detached connections. Production provider observations go through guard methods. The standalone store mutation helpers are useful for explicitly supplied administrative provenance and database tests, not a substitute for the provider read lock.

`apply_observations` commits a related snapshot atomically. A purchased grant must never become issuable in an intermediate commit before its known dispute is deducted. Unknown financial objects are retained without guessing credit value; unrelated verified purchases can still reconcile.

## Refunds

Ordinary refunds reserve exact eligible credits before contacting Stripe. A goodwill refund has an explicit cash amount and zero credit effect. Provider refunds without an internal operation are cash observations only: they never invent an account debit. All observed cash returns still reduce the money that can be refunded again and the loss attributable to a chargeback.

Cash rounding is cumulative within a purchase, capped by the remaining payment. It does not cancel extra credits to obtain a whole cent. Canceling remaining credits after a previous cash-only refund can consequently return zero cash; that operation completes locally and must not call Stripe with zero.

The operation's first submission time is immutable. Safe retries compare it with the database clock, never the operator device's clock. A lost response leaves an ambiguous submission, never a releasable failure. A definitive failure retains its hold. `retry_refund` transfers that entitlement to a new operation and Stripe idempotency key without releasing credits, even after the original expiry. Released/superseded attempts remain recordable when Stripe repeats their failed or canceled state. An alternate payout is an explicit operator attestation with a unique reference; it is forbidden while a Stripe submission is ambiguous.

Do not release an expired hold into already-expired credits and silently erase the customer's frozen refund entitlement. Complete it by another attempt or an explicitly recorded alternate payout. Delayed support requests that were not recorded while eligible require receipt/ledger review and the discretionary path.

## Disputes and closure

Chargebacks collect cash value at each affected purchase's actual price, in the same currency. Uncovered value is durable monetary debt, never an at-par or expiring credit overdraft. Free credits have no cash value and remain usable. `available_balance` reports a negative credit equivalent only when no positive credits remain; the API also reports currency debt explicitly.

Only successful refunds reduce a chargeback's target. Pending refunds reserve cash against new refund requests but cannot expose credits before cash returns. Targets are capped by the purchase price after prior returns and other disputes. A win undoes recorded allocations with their original expiries. Restored paid credits first satisfy any other outstanding debt under the same account lock. Integer credit allocation rounds down, retaining any fractional monetary residue.

Closure freezes eligibility when requested and stops new checkout/issuance. Provider cancellation, refunds and closure finalization remain resumable. A paid invoice can legitimately be collected after its service period ends: keep that actual expiry and cash provenance. A payment collected after closure is eligible for the closure refund even if its service period ended earlier.

Webhook events and local refund/closure intents are durable work items with bounded retry backoff. Creating a hold or beginning closure creates its local work item in the same transaction, so a process crash needs no new webhook to resume the operation. Retry work does not independently poll Stripe accounts.

## Validation

`tests/store.rs` exercises real PostgreSQL in unique schemas of the explicitly supplied disposable `BILLING_TEST_DATABASE_URL`; run its ignored tests too. Cases cover purchase identity, price/expiry provenance, refund versus issuance, failed and expired holds, alternate payouts, external refunds, cumulative cash rounding, partial refund/dispute overlap, negative debt, free credits, crossed disputes, out-of-order purchase discovery, atomic rollback, closure and guard cancellation. `tests/service.rs` exercises the provider workflow with local HTTP fixtures. The gateway tests cover the authenticated public boundary.

Never substitute unit arithmetic tests for these database races. Never use real Stripe credentials or a production database for fixture-driven tests.
