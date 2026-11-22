//! Billing's instruments, recorded through the global meter provider.
//!
//! They are no-ops unless the process installed a meter provider: the billing
//! receiver and worker install one when OTLP export is configured, the
//! gateway installs its own (and so reports the Stripe calls it makes under
//! its service name), and a support command installs none.
//!
//! Every label value comes from a fixed set in this crate. No account,
//! customer, subscription, price, product, event, charge or refund
//! identifier, no per-object amount and no Stripe-authored text is ever a
//! label: Stripe's own values pass through an allow-list ([`currency_label`],
//! [`job_type_label`], the Stripe operation and error-type lists) with a
//! catch-all for anything else. Money appears only as totals summed over
//! every purchase in an export interval.

use std::sync::LazyLock;
use std::time::Duration;

use opentelemetry::KeyValue;
use opentelemetry::metrics::{Counter, Gauge, Histogram};

use crate::service::{BILLING_EVENT_TYPES, LOCAL_JOB_KINDS};
use crate::store::{Applied, Backlog};

fn meter() -> opentelemetry::metrics::Meter {
    opentelemetry::global::meter("eidola-billing")
}

/// Second-scale buckets for a Stripe call: its client timeout is 30 s. The
/// SDK's default ladder is millisecond-scaled and would put every call in its
/// first bucket.
const STRIPE_LATENCY_BUCKETS: &[f64] = &[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0];

/// Currencies reported by name; any other is `other`.
const CURRENCIES: &[&str] = &["usd", "eur", "gbp", "cad", "aud"];

/// A currency as a label: its code when allow-listed, else `other`.
pub fn currency_label(currency: &str) -> &'static str {
    CURRENCIES
        .iter()
        .copied()
        .find(|known| *known == currency)
        .unwrap_or("other")
}

/// An inbox kind as a label: a supported Stripe event type or local job
/// kind, else `unsupported`.
pub fn job_type_label(kind: &str) -> &'static str {
    BILLING_EVENT_TYPES
        .iter()
        .chain(LOCAL_JOB_KINDS)
        .copied()
        .find(|known| *known == kind)
        .unwrap_or("unsupported")
}

/// Stripe API calls, by `operation` (`stripe::STRIPE_OPERATIONS`), `status`
/// (`2xx`; `400`, `401`, `402`, `403`, `404`, `409` or `429`; else `4xx`,
/// `5xx` or `other`; `timeout` or `network` when no response arrived) and
/// `error_type` (`none` for a success or no response, else Stripe's error
/// type resolved against a fixed list, `other` or `unparseable`). One per
/// HTTP request, so a paginated list counts each page.
static STRIPE_REQUESTS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("stripe.requests")
        .with_description("Stripe API requests, by operation, status and error type")
        .build()
});

/// Stripe API call latency, by `operation` and `status`.
static STRIPE_REQUEST_DURATION: LazyLock<Histogram<f64>> = LazyLock::new(|| {
    meter()
        .f64_histogram("stripe.request.duration")
        .with_description("Duration of Stripe API requests")
        .with_unit("s")
        .with_boundaries(STRIPE_LATENCY_BUCKETS.to_vec())
        .build()
});

pub(crate) fn stripe_request(
    operation: &'static str,
    status: &'static str,
    error_type: &'static str,
    elapsed: Duration,
) {
    STRIPE_REQUESTS.add(
        1,
        &[
            KeyValue::new("operation", operation),
            KeyValue::new("status", status),
            KeyValue::new("error_type", error_type),
        ],
    );
    STRIPE_REQUEST_DURATION.record(
        elapsed.as_secs_f64(),
        &[
            KeyValue::new("operation", operation),
            KeyValue::new("status", status),
        ],
    );
}

/// Inbox jobs a worker claimed and ran, by `job_type` ([`job_type_label`])
/// and `outcome`: `completed`, or the failure category the job is left
/// pending with (`requires_review`, `provider_unavailable`,
/// `provider_rejected`, `database_unavailable`, `conflict`, `not_found`,
/// `invalid`, `parse`, `unsupported_kind`, `unsupported_protocol`).
static JOBS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.jobs")
        .with_description("Billing inbox jobs run, by type and outcome")
        .build()
});

/// Jobs run again after an earlier attempt failed, by `job_type`.
static JOB_RETRIES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.jobs.retries")
        .with_description("Billing inbox jobs run again after a failed attempt")
        .build()
});

pub(crate) fn job(kind: &str, attempts: i32, outcome: &'static str) {
    let job_type = KeyValue::new("job_type", job_type_label(kind));
    if attempts > 0 {
        JOB_RETRIES.add(1, std::slice::from_ref(&job_type));
    }
    JOBS.add(1, &[job_type, KeyValue::new("outcome", outcome)]);
}

/// Account reconciliations (a full read of one account's Stripe state and
/// its application), by `outcome`: `ok` or the failure's class.
static RECONCILIATIONS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.reconciliations")
        .with_description("Account reconciliations, by outcome")
        .build()
});

pub(crate) fn reconciliation(outcome: &'static str) {
    RECONCILIATIONS.add(1, &[KeyValue::new("outcome", outcome)]);
}

/// Purchases granted for the first time, by `sale` (`one_time`,
/// `subscription_start` or `subscription_renewal`) and `currency`
/// ([`currency_label`]).
static PURCHASES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.purchases")
        .with_description("Purchases granted, by sale and currency")
        .build()
});

/// The amounts paid for those purchases, summed, in the currency's minor
/// unit, by `sale` and `currency`. Read it per currency: `other` sums
/// unlike currencies.
static PURCHASE_AMOUNT: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.purchases.amount")
        .with_description("Amount paid for purchases granted, in minor units")
        .with_unit("{minor_unit}")
        .build()
});

/// Refunds whose Stripe status is new or changed, by `status` (`pending`,
/// `requires_action`, `succeeded`, `failed`, `canceled`, or `rejected` for
/// a first submission Stripe refused outright) and `origin` (`operation`, a
/// refund we submitted, or `external`, one made outside this tool).
static REFUNDS: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.refunds")
        .with_description("Refund status changes, by status and origin")
        .build()
});

pub(crate) fn refund_rejected() {
    REFUNDS.add(
        1,
        &[
            KeyValue::new("status", "rejected"),
            KeyValue::new("origin", "operation"),
        ],
    );
}

/// Dispute events, by `outcome`: `opened`, or how it closed (`won`, `lost`,
/// `warning_closed`).
static DISPUTES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.disputes")
        .with_description("Disputes opened and closed, by outcome")
        .build()
});

/// Count what one commit changed. `renewal` says which purchases are a
/// subscription's later periods.
pub(crate) fn applied(applied: &Applied, renewal: impl Fn(&str) -> bool) {
    for purchase in &applied.purchases {
        let sale = match purchase.kind.as_str() {
            "purchase" => "one_time",
            _ if renewal(&purchase.id) => "subscription_renewal",
            _ => "subscription_start",
        };
        let labels = [
            KeyValue::new("sale", sale),
            KeyValue::new("currency", currency_label(&purchase.currency)),
        ];
        PURCHASES.add(1, &labels);
        PURCHASE_AMOUNT.add(purchase.paid_minor.max(0) as u64, &labels);
    }
    for (status, ours) in &applied.refunds {
        REFUNDS.add(
            1,
            &[
                KeyValue::new("status", *status),
                KeyValue::new("origin", if *ours { "operation" } else { "external" }),
            ],
        );
    }
    for outcome in &applied.disputes {
        DISPUTES.add(1, &[KeyValue::new("outcome", *outcome)]);
    }
}

/// Account closures, by `stage`: `requested` (the account stopped taking
/// purchases and conversions) or `closed` (every refund it owed settled).
static CLOSURES: LazyLock<Counter<u64>> = LazyLock::new(|| {
    meter()
        .u64_counter("billing.account.closures")
        .with_description("Account closures, by stage")
        .build()
});

pub(crate) fn closure(stage: &'static str) {
    CLOSURES.add(1, &[KeyValue::new("stage", stage)]);
}

/// Pending inbox rows, by `job_type`.
static WORK_PENDING: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    meter()
        .u64_gauge("billing.work.pending")
        .with_description("Pending billing inbox rows, by job type")
        .build()
});

/// Pending inbox rows that have failed at least once, by `job_type`.
static WORK_FAILING: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    meter()
        .u64_gauge("billing.work.failing")
        .with_description("Pending billing inbox rows that have failed, by job type")
        .build()
});

/// The oldest pending inbox row's age, by `job_type`, from when the row was
/// first received. Account repair and closure rows are reused for later
/// requests, so for those two it is the age of the account's first request.
static WORK_OLDEST_AGE: LazyLock<Gauge<f64>> = LazyLock::new(|| {
    meter()
        .f64_gauge("billing.work.oldest_age")
        .with_description("Age of the oldest pending billing inbox row, by job type")
        .with_unit("s")
        .build()
});

/// Refund operations not yet settled, by `state` (`prepared`, `submitting`,
/// `pending`, `requires_action`).
static REFUNDS_OPEN: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    meter()
        .u64_gauge("billing.refunds.open")
        .with_description("Unsettled refund operations, by state")
        .build()
});

/// The oldest `submitting` refund's age since its first submission. Its
/// safe retry window closes at 23 hours; alert well before.
static REFUND_SUBMITTING_OLDEST_AGE: LazyLock<Gauge<f64>> = LazyLock::new(|| {
    meter()
        .f64_gauge("billing.refunds.submitting.oldest_age")
        .with_description("Age of the oldest refund whose submission outcome is unknown")
        .with_unit("s")
        .build()
});

/// Items awaiting support review, by `source`: `unmatched_refund` and
/// `unmatched_dispute` (provider money this ledger could not attribute), and
/// `job` (pending work whose last failure needs review).
static REVIEW_OPEN: LazyLock<Gauge<u64>> = LazyLock::new(|| {
    meter()
        .u64_gauge("billing.review.open")
        .with_description("Billing items awaiting support review, by source")
        .build()
});

/// Record the backlog gauges, every label value included, zeros for what is
/// absent, so a cleared backlog reads zero rather than its last value.
pub fn backlog(backlog: &Backlog) {
    let mut work: Vec<(&'static str, u64, u64, f64)> = BILLING_EVENT_TYPES
        .iter()
        .chain(LOCAL_JOB_KINDS)
        .chain(&["unsupported"])
        .map(|kind| (*kind, 0, 0, 0.0))
        .collect();
    for (kind, pending, failing, age) in &backlog.work {
        let label = job_type_label(kind);
        if let Some(row) = work.iter_mut().find(|row| row.0 == label) {
            row.1 += pending;
            row.2 += failing;
            row.3 = row.3.max(*age);
        }
    }
    for (job_type, pending, failing, age) in work {
        let labels = [KeyValue::new("job_type", job_type)];
        WORK_PENDING.record(pending, &labels);
        WORK_FAILING.record(failing, &labels);
        WORK_OLDEST_AGE.record(age, &labels);
    }

    let mut submitting_age = 0.0;
    for state in ["prepared", "submitting", "pending", "requires_action"] {
        let (count, age) = backlog
            .refunds
            .iter()
            .find(|(s, _, _)| s == state)
            .map_or((0, 0.0), |(_, count, age)| (*count, *age));
        if state == "submitting" {
            submitting_age = age;
        }
        REFUNDS_OPEN.record(count, &[KeyValue::new("state", state)]);
    }
    REFUND_SUBMITTING_OLDEST_AGE.record(submitting_age, &[]);

    for (kind, source) in [
        ("refund", "unmatched_refund"),
        ("dispute", "unmatched_dispute"),
    ] {
        let count = backlog
            .unmatched
            .iter()
            .filter(|(k, _)| k == kind)
            .map(|(_, count)| count)
            .sum();
        REVIEW_OPEN.record(count, &[KeyValue::new("source", source)]);
    }
    REVIEW_OPEN.record(
        backlog.jobs_needing_review,
        &[KeyValue::new("source", "job")],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_values_resolve_against_fixed_lists() {
        assert_eq!(currency_label("usd"), "usd");
        assert_eq!(currency_label("xyz"), "other");
        assert_eq!(currency_label("USD"), "other");
        assert_eq!(job_type_label("v1.refund.updated"), "v1.refund.updated");
        assert_eq!(
            job_type_label("billing.account.reconcile"),
            "billing.account.reconcile"
        );
        assert_eq!(job_type_label("v1.customer.updated"), "unsupported");
        assert_eq!(job_type_label("evt_123"), "unsupported");
    }
}
