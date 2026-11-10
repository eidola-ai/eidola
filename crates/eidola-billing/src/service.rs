//! The one workflow used by webhook delivery, account repair and support tools.
//!
//! Provider facts are fetched under an account reconciliation lock and applied
//! atomically. A refund's durable reservation precedes its network request.

use std::time::{Duration, SystemTime};

use deadpool_postgres::Pool;
use serde::Serialize;
use uuid::Uuid;

use crate::BillingError;
use crate::store::{
    DisputeObservation, Purchase, ReconciliationGuard, RefundObservation, RefundOperation,
    RefundRequest, Store,
};
use crate::stripe::{RefundSnapshot, ReviewItem, StripeClient};

/// A margin inside Stripe's minimum 24-hour idempotency retention. Once this
/// expires, absence from a list is not permission to create another refund.
const SAFE_RESUBMIT_WINDOW: Duration = Duration::from_secs(23 * 60 * 60);

/// Stripe notifications that trigger the same authoritative reconciliation.
/// Internal operation jobs deliberately do not appear in this public allowlist.
pub const BILLING_EVENT_TYPES: &[&str] = &[
    "checkout.session.completed",
    "checkout.session.async_payment_succeeded",
    "checkout.session.async_payment_failed",
    "invoice.paid",
    "invoice.payment_succeeded",
    "charge.refunded",
    "charge.succeeded",
    "refund.created",
    "refund.updated",
    "refund.failed",
    "charge.dispute.created",
    "charge.dispute.updated",
    "charge.dispute.closed",
    "charge.dispute.funds_withdrawn",
    "charge.dispute.funds_reinstated",
];

#[derive(Debug, Default, Serialize)]
pub struct ReconcileReport {
    pub purchases: usize,
    pub refunds: usize,
    pub disputes: usize,
    pub review: Vec<ReviewItem>,
}

#[derive(Debug, Default, Serialize)]
pub struct DrainReport {
    pub processed: usize,
    pub completed: usize,
    pub failed: usize,
    pub review: usize,
}

pub struct BillingService {
    store: Store,
    stripe: StripeClient,
}

impl BillingService {
    pub fn new(pool: Pool, stripe: StripeClient) -> Self {
        Self {
            store: Store::new(pool),
            stripe,
        }
    }

    pub fn store(&self) -> &Store {
        &self.store
    }

    /// No provider IDs or amounts come from the account holder. The caller
    /// supplies only the authenticated account UUID; its customer comes from
    /// our database, and all financial facts come from Stripe.
    pub async fn reconcile_account(
        &self,
        account_id: Uuid,
    ) -> Result<ReconcileReport, BillingError> {
        let mut guard = self.store.lock_reconciliation(account_id).await?;
        let result = self.reconcile_locked(account_id, &mut guard, true).await;
        finish_locked(result, guard.release().await)
    }

    async fn reconcile_locked(
        &self,
        account_id: Uuid,
        guard: &mut ReconciliationGuard,
        resume_work: bool,
    ) -> Result<ReconcileReport, BillingError> {
        let Some(customer) = self.store.customer_id(account_id).await? else {
            return Ok(ReconcileReport::default());
        };
        let snapshot = self.stripe.fetch_account_snapshot(&customer).await?;
        let purchases: Vec<_> = snapshot
            .purchases
            .into_iter()
            .map(|p| Purchase {
                id: p.id,
                account_id,
                charge_id: p.charge_id,
                payment_intent: p.payment_intent,
                invoice_id: p.invoice_id,
                currency: p.currency,
                paid_minor: p.paid_minor,
                credits: p.credits,
                expires_at: p.expires_at,
                purchased_at: p.purchased_at,
                kind: p.kind,
            })
            .collect();
        let refunds: Vec<_> = snapshot.refunds.iter().map(refund_observation).collect();
        let disputes: Vec<_> = snapshot
            .disputes
            .into_iter()
            .map(|d| DisputeObservation {
                id: d.id,
                charge_id: d.charge_id,
                amount_minor: d.amount,
                currency: d.currency,
                status: d.status,
            })
            .collect();
        // One commit prevents issuance between recognizing a purchase and
        // recognizing its already-known chargeback. Unknown financial graphs
        // fail closed; they must never become a grant with its debit omitted.
        guard
            .apply_observations(account_id, &purchases, &refunds, &disputes)
            .await?;
        let mut report = ReconcileReport {
            purchases: purchases.len(),
            refunds: refunds.len(),
            disputes: disputes.len(),
            review: snapshot.review,
        };
        for refund in &refunds {
            if refund.operation_id.is_none() {
                report.review.push(ReviewItem {
                    object_id: refund.id.clone(),
                    reason: "external_refund_no_credit_effect".into(),
                });
            }
        }
        report
            .review
            .extend(
                self.store
                    .review_issues(account_id)
                    .await?
                    .into_iter()
                    .map(|issue| ReviewItem {
                        object_id: issue.id,
                        reason: issue.reason,
                    }),
            );
        if !resume_work {
            return Ok(report);
        }
        // Repair may finish an already-approved operation whose worker died
        // before its first provider call. It can never choose a new cash amount
        // or credit effect, both of which are fixed in the durable intent.
        for operation in self.store.list_refunds(account_id).await? {
            if matches!(operation.state.as_str(), "prepared" | "submitting") {
                let id = operation.id;
                match self.submit_reserved_locked(operation, guard).await {
                    Ok(_) => {}
                    Err(BillingError::RequiresReview(_)) => report.review.push(ReviewItem {
                        object_id: id.to_string(),
                        reason: "refund_attempt_requires_review".into(),
                    }),
                    Err(error) => return Err(error),
                }
            }
        }
        let state = self.store.account_state(account_id).await?;
        if state == "closing" || state == "closed" {
            // A checkout begun before closure can finish afterwards. The
            // resulting payment remains a recorded obligation to refund.
            self.finish_closure_locked(account_id, &mut report, guard)
                .await?;
        }
        Ok(report)
    }

    /// Reserve first, send second, reconcile third. Callers choose and persist
    /// the operation UUID before their first call; a repeated UUID must carry
    /// exactly the same request, including its explicit credit effect.
    pub async fn submit_refund(
        &self,
        request: &RefundRequest,
    ) -> Result<RefundOperation, BillingError> {
        let mut guard = self.store.lock_reconciliation(request.account_id).await?;
        let result = async {
            self.reconcile_locked(request.account_id, &mut guard, false)
                .await?;
            let operation = self.store.prepare_refund(request).await?;
            self.submit_reserved_locked(operation, &mut guard).await
        }
        .await;
        finish_locked(result, guard.release().await)
    }

    pub async fn resume_refund(&self, id: Uuid) -> Result<RefundOperation, BillingError> {
        let operation = self.store.get_refund(id).await?;
        let mut guard = self.store.lock_reconciliation(operation.account_id).await?;
        let result = async {
            self.reconcile_locked(operation.account_id, &mut guard, false)
                .await?;
            self.submit_reserved_locked(self.store.get_refund(id).await?, &mut guard)
                .await
        }
        .await;
        finish_locked(result, guard.release().await)
    }

    async fn submit_reserved_locked(
        &self,
        mut operation: RefundOperation,
        guard: &mut ReconciliationGuard,
    ) -> Result<RefundOperation, BillingError> {
        if operation.amount_minor == 0 || terminal_refund(&operation.state) {
            return Ok(operation);
        }

        // Look for a completed request whose response or database update was
        // lost. Metadata names our durable intent; the Stripe refund ID is
        // subsequently bound to that exact amount, currency and charge.
        let observations = self.stripe.list_refunds(&operation.charge_id).await?;
        let matches: Vec<_> = observations
            .iter()
            .filter(|r| r.operation_id == Some(operation.id))
            .collect();
        if matches.len() > 1 {
            return Err(BillingError::RequiresReview(
                "multiple refunds name one operation".into(),
            ));
        }
        if let Some(refund) = matches.first() {
            guard
                .apply_observations(
                    operation.account_id,
                    &[],
                    &[refund_observation(refund)],
                    &[],
                )
                .await?;
            return self.store.get_refund(operation.id).await;
        }
        if operation.refund_id.is_some()
            || !matches!(operation.state.as_str(), "prepared" | "submitting")
        {
            return Err(BillingError::RequiresReview(
                "recorded refund is absent from the complete provider list".into(),
            ));
        }
        if !can_resubmit(operation.submitted_at, self.store.current_time().await?) {
            return Err(BillingError::RequiresReview(
                "uncertain refund is outside the safe idempotency retry window".into(),
            ));
        }
        operation = self.store.mark_submitted(operation.id).await?;
        let key = refund_idempotency_key(operation.id);
        // Capture the monotonic anchor before reading the database clock: a
        // task paused after this point must not start an old-key request when
        // it wakes, and an operator device's clock must not extend the window.
        let anchor = tokio::time::Instant::now();
        let age = self
            .store
            .current_time()
            .await?
            .duration_since(operation.submitted_at.ok_or_else(|| {
                BillingError::RequiresReview("refund has no submission timestamp".into())
            })?)
            .map_err(|_| BillingError::RequiresReview("refund clock moved backwards".into()))?;
        let remaining = SAFE_RESUBMIT_WINDOW
            .checked_sub(age)
            .filter(|d| !d.is_zero())
            .ok_or_else(|| BillingError::RequiresReview("refund retry window elapsed".into()))?;
        let response = tokio::time::timeout_at(
            anchor + remaining.min(Duration::from_secs(30)),
            self.stripe.create_refund(
                &operation.charge_id,
                operation.amount_minor,
                operation.id,
                &key,
            ),
        )
        .await
        .map_err(|_| BillingError::Network("refund submission outcome is unknown".into()))?;
        match response {
            Ok(refund) => {
                guard
                    .apply_observations(
                        operation.account_id,
                        &[],
                        &[refund_observation(&refund)],
                        &[],
                    )
                    .await?;
                self.store.get_refund(operation.id).await
            }
            Err(error @ BillingError::ProviderRejected(_)) => {
                // A definitive rejection is distinct from a timeout or a 5xx.
                // Failed holds remain reserved until the operator chooses a
                // remedy; no network error automatically restores credits.
                guard.mark_refund_rejected(operation.id).await?;
                Err(error)
            }
            Err(error) => Err(error),
        }
    }

    pub async fn release_refund(&self, id: Uuid) -> Result<RefundOperation, BillingError> {
        let operation = self.store.get_refund(id).await?;
        let mut guard = self.store.lock_reconciliation(operation.account_id).await?;
        let result = async {
            // Reconcile first so a delayed success cannot be released based on
            // stale local state. Store also refuses ambiguous or expired holds.
            self.reconcile_locked(operation.account_id, &mut guard, false)
                .await?;
            self.store.release_refund(id).await?;
            self.store.get_refund(id).await
        }
        .await;
        finish_locked(result, guard.release().await)
    }

    /// A new attempt after a definitive failure keeps the original held
    /// entitlement, even if its account-credit expiry has since passed.
    pub async fn retry_refund(
        &self,
        id: Uuid,
        new_id: Uuid,
        reason: &str,
    ) -> Result<RefundOperation, BillingError> {
        let operation = self.store.get_refund(id).await?;
        let mut guard = self.store.lock_reconciliation(operation.account_id).await?;
        let result = async {
            self.reconcile_locked(operation.account_id, &mut guard, false)
                .await?;
            let replacement = self.store.retry_refund(id, new_id, reason).await?;
            self.submit_reserved_locked(replacement, &mut guard).await
        }
        .await;
        finish_locked(result, guard.release().await)
    }

    /// Record an alternate payout the operator has already completed. This
    /// does not send money; its reference is an audit record, not API authority.
    pub async fn confirm_external_refund(
        &self,
        id: Uuid,
        reference: &str,
    ) -> Result<RefundOperation, BillingError> {
        let operation = self.store.get_refund(id).await?;
        let mut guard = self.store.lock_reconciliation(operation.account_id).await?;
        let result = async {
            self.reconcile_locked(operation.account_id, &mut guard, false)
                .await?;
            self.store.confirm_external_refund(id, reference).await?;
            self.store.get_refund(id).await
        }
        .await;
        finish_locked(result, guard.release().await)
    }

    pub async fn cancel_subscriptions(&self, account_id: Uuid) -> Result<usize, BillingError> {
        let guard = self.store.lock_reconciliation(account_id).await?;
        let result = self.cancel_subscriptions_locked(account_id).await;
        finish_locked(result, guard.release().await)
    }

    async fn cancel_subscriptions_locked(&self, account_id: Uuid) -> Result<usize, BillingError> {
        let Some(customer) = self.store.customer_id(account_id).await? else {
            return Ok(0);
        };
        let subscriptions = self.stripe.list_subscriptions(&customer).await?;
        let mut canceled = 0;
        for subscription in subscriptions {
            if !matches!(
                subscription.status.as_str(),
                "canceled" | "incomplete_expired"
            ) {
                self.stripe.cancel_subscription(&subscription.id).await?;
                canceled += 1;
            }
        }
        Ok(canceled)
    }

    /// Closure is resumable. Freeze new purchases/conversions before any
    /// upstream calls, stop renewals, then refund the eligible net balance.
    pub async fn close_account(
        &self,
        account_id: Uuid,
        reason: &str,
    ) -> Result<ReconcileReport, BillingError> {
        if reason.trim().is_empty() {
            return Err(BillingError::Invalid("closure requires a reason".into()));
        }
        let mut guard = self.store.lock_reconciliation(account_id).await?;
        let result = async {
            self.store
                .begin_close_with_reason(account_id, reason)
                .await?;
            self.cancel_subscriptions_locked(account_id).await?;
            let mut report = self.reconcile_locked(account_id, &mut guard, true).await?;
            // Accounts with no payment relationship still close durably.
            if self.store.customer_id(account_id).await?.is_none() {
                self.finish_closure_locked(account_id, &mut report, &mut guard)
                    .await?;
            }
            Ok(report)
        }
        .await;
        finish_locked(result, guard.release().await)
    }

    async fn finish_closure_locked(
        &self,
        account_id: Uuid,
        report: &mut ReconcileReport,
        guard: &mut ReconciliationGuard,
    ) -> Result<(), BillingError> {
        // Resuming through a webhook or account repair must finish cancellation
        // too: the original closer may have stopped immediately after freezing.
        self.cancel_subscriptions_locked(account_id).await?;
        let mut awaiting = false;
        for operation in self.store.list_refunds(account_id).await? {
            if matches!(
                operation.state.as_str(),
                "released" | "succeeded" | "superseded"
            ) {
                continue;
            }
            let operation = self.submit_reserved_locked(operation, guard).await?;
            if operation.state != "succeeded" && operation.state != "released" {
                awaiting = true;
            }
        }
        // Reserving each lot removes it from subsequent lists. If interrupted
        // between lots, existing reservations are resumed above before another
        // is created, so no random UUID is used twice for the same balance.
        for (purchase, credits) in self.store.list_refundable_purchases(account_id).await? {
            let request = RefundRequest {
                id: Uuid::new_v4(),
                account_id,
                purchase_id: purchase.id,
                credits,
                cash_minor: None,
                reason: "account closure".into(),
            };
            let operation = self.store.prepare_refund(&request).await?;
            let operation = self.submit_reserved_locked(operation, guard).await?;
            awaiting |= operation.state != "succeeded";
        }
        if awaiting {
            report.review.push(ReviewItem {
                object_id: account_id.to_string(),
                reason: "closure_awaiting_refund_settlement".into(),
            });
        } else {
            self.store.finalize_close(account_id).await?;
        }
        Ok(())
    }

    /// Retry durable webhook work only. There is no periodic Stripe scan.
    pub async fn drain_events(&self, limit: i64) -> Result<DrainReport, BillingError> {
        if !(1..=100).contains(&limit) {
            return Err(BillingError::Invalid("event limit must be 1..=100".into()));
        }
        let events = self.store.pending_events(limit).await?;
        let mut report = DrainReport::default();
        for event in events {
            report.processed += 1;
            if matches!(
                event.kind.as_str(),
                "billing.refund.resume" | "billing.close.resume"
            ) {
                let result = async {
                    let id = Uuid::parse_str(&event.object_id).map_err(|_| {
                        BillingError::RequiresReview("invalid internal billing job".into())
                    })?;
                    if event.kind == "billing.refund.resume" {
                        self.resume_refund(id).await?;
                        Ok(0)
                    } else {
                        if !matches!(
                            self.store.account_state(id).await?.as_str(),
                            "closing" | "closed"
                        ) {
                            return Err(BillingError::RequiresReview(
                                "closure was not requested".into(),
                            ));
                        }
                        Ok(self
                            .close_account(id, "resume recorded account closure")
                            .await?
                            .review
                            .len())
                    }
                }
                .await;
                match result {
                    Ok(review) => {
                        self.store.complete_event(&event.id).await?;
                        report.completed += 1;
                        report.review += review;
                    }
                    Err(error) => {
                        self.store
                            .fail_event(&event.id, failure_class(&error))
                            .await?;
                        report.failed += 1;
                    }
                }
                continue;
            }
            let Some(kind) = event_object_kind(&event.kind) else {
                self.store.complete_event(&event.id).await?;
                report.completed += 1;
                continue;
            };
            let result = async {
                let customer = self
                    .stripe
                    .event_customer(kind, &event.object_id)
                    .await?
                    .ok_or_else(|| BillingError::RequiresReview("event has no customer".into()))?;
                let account_id = self
                    .store
                    .account_for_customer(&customer)
                    .await?
                    .ok_or_else(|| BillingError::RequiresReview("customer is not linked".into()))?;
                self.reconcile_account(account_id).await
            }
            .await;
            match result {
                Ok(result) => {
                    report.review += result.review.len();
                    self.store.complete_event(&event.id).await?;
                    report.completed += 1;
                }
                Err(error) => {
                    self.store
                        .fail_event(&event.id, failure_class(&error))
                        .await?;
                    report.failed += 1;
                }
            }
        }
        Ok(report)
    }
}

fn finish_locked<T>(
    result: Result<T, BillingError>,
    released: Result<(), BillingError>,
) -> Result<T, BillingError> {
    match result {
        Ok(value) => released.map(|()| value),
        Err(error) => Err(error),
    }
}

fn refund_observation(refund: &RefundSnapshot) -> RefundObservation {
    RefundObservation {
        id: refund.id.clone(),
        charge_id: refund.charge_id.clone(),
        amount_minor: refund.amount,
        currency: refund.currency.clone(),
        status: refund.status.clone(),
        operation_id: refund.operation_id,
    }
}

fn refund_idempotency_key(id: Uuid) -> String {
    format!("eidola/refund/v1/{id}")
}

fn can_resubmit(submitted_at: Option<SystemTime>, now: SystemTime) -> bool {
    submitted_at.is_none_or(|submitted| {
        now.duration_since(submitted)
            .is_ok_and(|age| age < SAFE_RESUBMIT_WINDOW)
    })
}

fn terminal_refund(state: &str) -> bool {
    matches!(
        state,
        "succeeded" | "failed" | "canceled" | "released" | "superseded"
    )
}

/// The webhook entry point can use this exact allowlist before enqueuing.
pub fn event_object_kind(kind: &str) -> Option<&'static str> {
    if !BILLING_EVENT_TYPES.contains(&kind) {
        return None;
    }
    match kind {
        "checkout.session.completed"
        | "checkout.session.async_payment_succeeded"
        | "checkout.session.async_payment_failed" => Some("checkout.session"),
        "invoice.paid" | "invoice.payment_succeeded" => Some("invoice"),
        "charge.refunded" | "charge.succeeded" => Some("charge"),
        "refund.created" | "refund.updated" | "refund.failed" => Some("refund"),
        "charge.dispute.created"
        | "charge.dispute.updated"
        | "charge.dispute.closed"
        | "charge.dispute.funds_withdrawn"
        | "charge.dispute.funds_reinstated" => Some("dispute"),
        _ => None,
    }
}

fn failure_class(error: &BillingError) -> &'static str {
    match error {
        BillingError::Invalid(_) => "invalid",
        BillingError::Conflict(_) => "conflict",
        BillingError::NotFound(_) => "not_found",
        BillingError::RequiresReview(_) => "requires_review",
        BillingError::Database(_) => "database_unavailable",
        BillingError::Network(_) => "provider_unavailable",
        BillingError::Parse(_) => "parse",
        BillingError::ServiceUnavailable(_) => "provider_unavailable",
        BillingError::ProviderRejected(_) => "provider_rejected",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncertain_refunds_never_get_a_new_idempotency_window() {
        let first = SystemTime::UNIX_EPOCH + Duration::from_secs(100_000);
        assert!(can_resubmit(None, first));
        assert!(can_resubmit(Some(first), first + Duration::from_secs(60)));
        assert!(!can_resubmit(Some(first), first + SAFE_RESUBMIT_WINDOW));
        assert!(!can_resubmit(Some(first), first - Duration::from_secs(1)));
        let id = Uuid::new_v4();
        assert_eq!(refund_idempotency_key(id), refund_idempotency_key(id));
    }

    #[test]
    fn terminal_failures_are_never_new_attempts_under_an_old_key() {
        for state in ["succeeded", "failed", "canceled", "released"] {
            assert!(terminal_refund(state));
        }
        assert!(!terminal_refund("submitting"));
        assert!(!terminal_refund("pending"));
    }

    #[test]
    fn every_cash_event_reaches_reconciliation_without_an_inferred_effect() {
        assert_eq!(event_object_kind("refund.failed"), Some("refund"));
        assert_eq!(event_object_kind("charge.refunded"), Some("charge"));
        assert_eq!(event_object_kind("charge.dispute.closed"), Some("dispute"));
        assert_eq!(event_object_kind("charge.succeeded"), Some("charge"));
        assert_eq!(event_object_kind("customer.updated"), None);
    }
}
