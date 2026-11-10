//! Thin Stripe API client.
//!
//! Shared by the gateway and the private support CLI. No gateway privileges
//! or anonymous credential state are needed here.
//! Uses form-encoded bodies (Stripe's native format) and Bearer auth.

use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{BillingError as ServerError, error::parse_error_summary, tls_config};

/// The Stripe API version every request pins, via the `Stripe-Version`
/// header set once as a default on the client.
///
/// **Pinned deliberately.** Without this header a request is served at the
/// *account's* default version — a dashboard setting, changeable by a human
/// who is not looking at this repository, that silently rewrites what a
/// sealed, attested server binary does. Everything else about this server's
/// behavior is a function of its measured source; an unpinned API version
/// was the one input that was not. It had already cost us: Basil moved the
/// subscription billing period onto subscription items (2025-03-31), the
/// response kept deserializing because the field was optional, and the
/// app's "current billing period ends …" line quietly never rendered again.
///
/// `2025-03-31.basil` is the oldest version carrying the shapes below, which
/// maximizes the runway before Stripe retires it.
///
/// **Bumping this is a review task, not a version bump.** Re-read the
/// intervening changelogs against every shape this file parses:
/// [`Subscription`] (+ [`SubscriptionItem`]), [`StripePrice`],
/// [`StripeProduct`], [`CheckoutSession`], [`CheckoutLineItem`],
/// [`Customer`], [`PortalSession`]. Note that this header pins **only what
/// this client reads**: webhook payloads are rendered at the version
/// configured on the webhook endpoint in Stripe's dashboard, so
/// `webhook.rs`'s `serde_json::Value` lookups are versioned separately and
/// are not covered by this constant.
pub const STRIPE_API_VERSION: &str = "2025-03-31.basil";

/// Minimal Stripe subscription representation.
#[derive(Debug, Deserialize)]
pub struct Subscription {
    pub id: String,
    pub status: String,
    /// The subscription's items, which is where the billing period lives.
    /// Included in list responses by default.
    pub items: SubscriptionItems,
}

#[derive(Debug, Deserialize)]
pub struct SubscriptionItems {
    pub data: Vec<SubscriptionItem>,
}

/// One line of a subscription. Since Basil this is what carries the billing
/// period; the subscription itself no longer has one.
#[derive(Debug, Deserialize)]
pub struct SubscriptionItem {
    /// **Required on purpose.** Optional here is what made the last shape
    /// change silent — a removed field and a genuinely absent value became
    /// the same `None`. Pinned to [`STRIPE_API_VERSION`], this field is
    /// guaranteed present, so demanding it turns any future move into a
    /// loud parse failure in dev and CI instead of a feature that quietly
    /// stops rendering in production.
    pub current_period_end: i64,
}

/// The Stripe subscription statuses Eidola treats as a subscription being
/// **in force** — the account has a live subscription relationship, so it
/// may not start a second one and the app offers management rather than
/// plans.
///
/// - `active` — paid and current.
/// - `trialing` — entitling; billing has not started yet.
/// - `past_due` — Stripe is still retrying payment and keeps the
///   subscription alive; the right move is fixing the card in the billing
///   portal, not buying a second subscription.
///
/// Everything else is not in force: `incomplete` (a first payment that
/// never succeeded; Stripe expires it within a day), `incomplete_expired`,
/// `unpaid`, `paused`, and `canceled`.
///
/// One list, one meaning: both the subscription read and the checkout
/// guard resolve through this predicate, so the app can never be told
/// "no subscription" by one and "already subscribed" by the other.
pub const IN_FORCE_SUBSCRIPTION_STATUSES: &[&str] = &["active", "trialing", "past_due"];

/// Page size for the subscription walk — Stripe's maximum, so the answer
/// takes one round trip for any customer who is not pathological.
const SUBSCRIPTION_PAGE_SIZE: &str = "100";

/// How far the walk will go before refusing to answer. At 100 per page this
/// is a thousand non-canceled subscriptions on one customer, which our own
/// checkout guard makes unreachable; the cap exists so a third party can
/// never hold a request open indefinitely, not as a budget we expect to
/// spend.
const MAX_SUBSCRIPTION_PAGES: usize = 10;

impl Subscription {
    /// Whether this subscription is in force — see
    /// [`IN_FORCE_SUBSCRIPTION_STATUSES`].
    pub fn is_in_force(&self) -> bool {
        IN_FORCE_SUBSCRIPTION_STATUSES.contains(&self.status.as_str())
    }

    /// When the current billing period ends, as a Unix timestamp.
    ///
    /// Items can bill on different intervals, so there is not always one
    /// period; the **soonest** boundary is reported, because that is the
    /// next time anything happens to this subscription — and for Eidola's
    /// single-price subscriptions it is simply that item's period. `None`
    /// only if Stripe returned a subscription with no items at all.
    pub fn current_period_end(&self) -> Option<i64> {
        self.items.data.iter().map(|i| i.current_period_end).min()
    }
}

/// Stripe list response wrapper. `has_more` is what makes a walk over a
/// paginated list terminate on Stripe's own word rather than on a guess
/// about how full a page looks.
#[derive(Debug, Deserialize)]
struct ListResponse<T> {
    pub data: Vec<T>,
    /// **Required on purpose**, exactly like [`SubscriptionItem::current_period_end`]
    /// and for the same reason. Defaulted, a response that omits it — a
    /// shape change, a malformed intermediary — reads as `false`, which the
    /// walk takes for "the whole list has been seen": an incomplete walk
    /// becomes a confident "no subscription", and the checkout guard sells
    /// a second subscription over a live one. A default here is precisely
    /// the guess this field exists to replace, so its absence fails the
    /// parse and the request refuses instead.
    pub has_more: bool,
}

/// Stripe checkout session (only the URL field we need).
#[derive(Debug, Deserialize)]
struct CheckoutSession {
    pub url: Option<String>,
}

/// Stripe customer (only the ID field we need).
#[derive(Debug, Deserialize)]
struct Customer {
    pub id: String,
}

/// Stripe billing portal session.
#[derive(Debug, Deserialize)]
struct PortalSession {
    pub url: String,
}

/// Stripe API error response.
#[derive(Debug, Deserialize)]
struct StripeErrorResponse {
    pub error: StripeErrorBody,
}

#[derive(Debug, Deserialize)]
struct StripeErrorBody {
    /// Stripe's enumerable error classifier (`type` in the JSON). The
    /// free-text `message` field is deliberately not modeled — see
    /// [`stripe_error`].
    #[serde(rename = "type", default)]
    pub error_type: Option<String>,
}

/// The one field of a Stripe Charge the dispute handlers need.
///
/// Required but nullable, deliberately: a Charge always carries `customer`
/// (`null` when the charge was made without one), so an omitted key is not
/// a charge without a customer but a response this code does not
/// understand. Defaulting it to `None` would have ended the dispute as
/// `orphan_customer` — a 200 that stops Stripe's redelivery — on evidence
/// that was never there; a parse error takes the retryable branch instead.
/// (serde treats a bare `Option` field as missing-means-`None`; the
/// `deserialize_with` is what makes the key itself required.)
#[derive(Debug, Deserialize)]
struct StripeChargeCustomer {
    #[serde(deserialize_with = "Option::deserialize")]
    customer: Option<String>,
}

/// Stripe product (expanded from a price).
#[derive(Debug, Deserialize)]
pub struct StripeProduct {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Stripe recurring billing info on a price.
#[derive(Debug, Deserialize)]
pub struct StripeRecurring {
    pub interval: String,
    pub interval_count: i64,
}

/// Stripe price with expanded product (returned by list).
#[derive(Debug, Deserialize)]
pub struct StripePrice {
    pub id: String,
    pub active: bool,
    pub currency: String,
    #[serde(default)]
    pub unit_amount: Option<i64>,
    #[serde(rename = "type")]
    pub price_type: String,
    #[serde(default)]
    pub recurring: Option<StripeRecurring>,
    pub product: StripeProduct,
    #[serde(default)]
    pub lookup_key: Option<String>,
}

/// Minimal price representation (just enough to determine checkout mode).
#[derive(Debug, Deserialize)]
pub struct StripePriceMinimal {
    #[serde(default)]
    pub recurring: Option<StripeRecurring>,
}

/// A price nested inside a checkout line item.
#[derive(Debug, Deserialize)]
pub struct CheckoutLineItemPrice {
    /// Product ID (string when not expanded).
    pub product: String,
}

/// A single line item from a checkout session.
#[derive(Debug, Deserialize)]
pub struct CheckoutLineItem {
    pub id: String,
    pub price: CheckoutLineItemPrice,
}

/// Parameters for creating a Stripe Checkout Session.
pub struct CheckoutParams<'a> {
    pub customer_id: &'a str,
    pub price_id: &'a str,
    pub mode: &'a str,
    pub success_url: &'a str,
    pub cancel_url: &'a str,
    pub client_reference_id: Option<&'a str>,
    /// Disclosure text rendered next to Checkout's pay button
    /// (`custom_text[submit][message]`) — e.g. the credit-expiry terms.
    pub submit_note: Option<&'a str>,
    /// Customer-visible description stamped on the PaymentIntent (payment
    /// mode) or Subscription (subscription mode). Stripe renders it on
    /// email receipts and invoices — the receipt-side expiry disclosure.
    pub description: Option<&'a str>,
    /// The agreed grant, captured when checkout is created. Reconciliation
    /// never consults mutable Product metadata to value an old purchase.
    pub credits: i64,
}

/// Current provider facts, read by webhook handling and account repair alike.
/// Unsupported historical purchases are review items, not invented grants.
#[derive(Debug, Default)]
pub struct AccountSnapshot {
    pub purchases: Vec<PurchaseSnapshot>,
    pub refunds: Vec<RefundSnapshot>,
    pub disputes: Vec<DisputeSnapshot>,
    pub review: Vec<ReviewItem>,
}

#[derive(Debug, Clone)]
pub struct PurchaseSnapshot {
    pub id: String,
    pub charge_id: Option<String>,
    pub payment_intent: Option<String>,
    pub invoice_id: Option<String>,
    pub currency: String,
    pub paid_minor: i64,
    pub credits: i64,
    pub expires_at: SystemTime,
    pub purchased_at: SystemTime,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RefundSnapshot {
    pub id: String,
    pub charge_id: String,
    pub amount: i64,
    pub currency: String,
    pub status: String,
    pub operation_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DisputeSnapshot {
    pub id: String,
    pub charge_id: String,
    pub amount: i64,
    pub currency: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewItem {
    pub object_id: String,
    /// A fixed reason, never Stripe's free text or payment details.
    pub reason: String,
}

/// Stripe expandable references work both before and after expansion.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StripeId {
    Id(String),
    Expanded { id: String },
}

impl StripeId {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Id(id) | Self::Expanded { id } => id,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct PaidCheckout {
    pub id: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub customer: Option<StripeId>,
    pub mode: String,
    pub payment_status: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub status: Option<String>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub currency: Option<String>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub amount_total: Option<i64>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub payment_intent: Option<StripeId>,
    pub metadata: HashMap<String, String>,
    pub created: i64,
}

#[derive(Debug, Deserialize)]
pub struct PaidInvoice {
    pub id: String,
    pub customer: StripeId,
    pub status: String,
    pub amount_paid: i64,
    pub total: i64,
    pub starting_balance: i64,
    pub currency: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub billing_reason: Option<String>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub parent: Option<InvoiceParent>,
    pub lines: InvoiceLines,
    pub status_transitions: InvoiceTransitions,
}

#[derive(Debug, Deserialize)]
pub struct InvoiceParent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub subscription_details: Option<InvoiceSubscription>,
}

#[derive(Debug, Deserialize)]
pub struct InvoiceSubscription {
    pub subscription: StripeId,
    /// Stripe snapshots subscription metadata when the invoice is created.
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct InvoiceTransitions {
    #[serde(deserialize_with = "Option::deserialize")]
    pub paid_at: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct InvoiceLines {
    pub data: Vec<InvoiceLine>,
    pub has_more: bool,
}

#[derive(Debug, Deserialize)]
pub struct InvoiceLine {
    pub id: String,
    pub period: InvoicePeriod,
    #[serde(deserialize_with = "Option::deserialize")]
    pub pricing: Option<InvoicePricing>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub quantity: Option<i64>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub parent: Option<InvoiceLineParent>,
}

#[derive(Debug, Deserialize)]
pub struct InvoiceLineParent {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub subscription_item_details: Option<SubscriptionLineDetails>,
}

#[derive(Debug, Deserialize)]
pub struct SubscriptionLineDetails {
    pub proration: bool,
}

#[derive(Debug, Deserialize)]
pub struct InvoicePeriod {
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Deserialize)]
pub struct InvoicePricing {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub price_details: Option<InvoicePriceDetails>,
}

#[derive(Debug, Deserialize)]
pub struct InvoicePriceDetails {
    pub price: StripeId,
}

#[derive(Debug, Deserialize)]
pub struct InvoicePayment {
    pub id: String,
    pub invoice: StripeId,
    pub status: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub amount_paid: Option<i64>,
    pub currency: String,
    pub payment: InvoicePaymentSource,
}

#[derive(Debug, Deserialize)]
pub struct InvoicePaymentSource {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub payment_intent: Option<StripeId>,
}

#[derive(Debug, Deserialize)]
pub struct StripeCharge {
    pub id: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub customer: Option<StripeId>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub payment_intent: Option<StripeId>,
    pub amount: i64,
    pub amount_refunded: i64,
    pub currency: String,
    pub paid: bool,
    pub captured: bool,
    pub status: String,
    pub created: i64,
}

#[derive(Debug, Deserialize)]
pub struct StripePaymentIntent {
    pub id: String,
    #[serde(deserialize_with = "Option::deserialize")]
    pub customer: Option<StripeId>,
    #[serde(deserialize_with = "Option::deserialize")]
    pub latest_charge: Option<StripeId>,
    pub amount_received: i64,
    pub currency: String,
    pub status: String,
}

#[derive(Debug, Deserialize)]
struct StripeRefund {
    id: String,
    charge: StripeId,
    amount: i64,
    currency: String,
    status: String,
    metadata: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct StripeDispute {
    id: String,
    charge: StripeId,
    amount: i64,
    currency: String,
    status: String,
}

trait ObjectId {
    fn object_id(&self) -> &str;
}

macro_rules! object_id {
    ($($t:ty),+ $(,)?) => { $(impl ObjectId for $t {
        fn object_id(&self) -> &str { &self.id }
    })+ };
}

object_id!(
    PaidCheckout,
    PaidInvoice,
    InvoiceLine,
    InvoicePayment,
    StripeCharge,
    StripeRefund,
    StripeDispute,
    StripePrice,
    CheckoutLineItem,
    Subscription
);

#[derive(Clone)]
pub struct StripeClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl StripeClient {
    pub fn new(api_key: String) -> Self {
        // `Stripe-Version` rides as a client default rather than on each
        // call site, so a request added later cannot forget it and fall
        // back to the account's mutable default version. See
        // [`STRIPE_API_VERSION`].
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "Stripe-Version",
            reqwest::header::HeaderValue::from_static(STRIPE_API_VERSION),
        );

        let client = reqwest::Client::builder()
            .tls_backend_preconfigured(tls_config())
            .default_headers(headers)
            .connect_timeout(Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            api_key,
            base_url: "https://api.stripe.com/v1".to_string(),
        }
    }

    /// Create a Stripe customer linked to an account.
    pub async fn create_customer(&self, account_id: Uuid) -> Result<String, ServerError> {
        let response = self
            .client
            .post(format!("{}/customers", self.base_url))
            .bearer_auth(&self.api_key)
            .header("Idempotency-Key", format!("eidola-customer-{account_id}"))
            .form(&[("metadata[account_id]", account_id.to_string())])
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let customer: Customer = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!("stripe customer: {}", parse_error_summary(&e)))
        })?;

        Ok(customer.id)
    }

    /// The customer's in-force subscription, if they have one — the only
    /// question either caller asks, and therefore the only one this client
    /// answers.
    ///
    /// **The list is walked, not sampled.** Stripe's list endpoint takes at
    /// most one `status` value, so it cannot express the in-force set; the
    /// filter has to run here, which means a single page is an answer only
    /// when it is the whole list. A customer with a page full of newer
    /// `incomplete` checkout attempts ahead of one older `active`
    /// subscription would otherwise read as unsubscribed — and since the
    /// checkout guard asks this same question, they could then open a
    /// second paid subscription over a live one.
    ///
    /// Returning `Option<Subscription>` rather than the list is what keeps
    /// that unrepresentable: there is no truncatable list for a caller to
    /// read the in-force rule off, so `Subscription::is_in_force` is
    /// applied in exactly one place.
    pub async fn in_force_subscription(
        &self,
        customer_id: &str,
    ) -> Result<Option<Subscription>, ServerError> {
        let mut starting_after: Option<String> = None;

        for _ in 0..MAX_SUBSCRIPTION_PAGES {
            let page = self
                .subscription_page(customer_id, starting_after.as_deref())
                .await?;

            // Stripe pages by the id of the last item seen, so the cursor is
            // read before the page is consumed by the search.
            let cursor = page.data.last().map(|s| s.id.clone());

            if let Some(sub) = page.data.into_iter().find(|s| s.is_in_force()) {
                return Ok(Some(sub));
            }

            match (page.has_more, cursor) {
                // The whole list has been seen and none of it is in force.
                (false, _) => return Ok(None),
                // More pages, and somewhere to resume from.
                (true, Some(id)) => starting_after = Some(id),
                // "More results exist" handed back with nothing to page
                // from. The walk cannot continue and it has not finished —
                // so this is an incomplete walk, and answering `None` about
                // one is the very claim this method exists to stop making.
                // It refuses, exactly as the page cap below does. Stripe
                // does not do this; the arm exists so a surprise cannot
                // spin, and stopping the spin is not a licence to guess.
                (true, None) => {
                    return Err(ServerError::ServiceUnavailable(
                        "stripe: a page of a customer's subscriptions reported \
                         more results but carried none"
                            .to_string(),
                    ));
                }
            }
        }

        // The cap is not a page budget we are willing to answer inside — a
        // walk that ran out of pages has seen only a prefix, and "no
        // subscription" said about a prefix is exactly the claim this method
        // exists to stop making. Refusing means the client shows "couldn't
        // check" and checkout declines to sell a second subscription it
        // cannot rule out.
        Err(ServerError::ServiceUnavailable(format!(
            "stripe: a customer's subscriptions did not fit in \
             {MAX_SUBSCRIPTION_PAGES} pages"
        )))
    }

    /// One page of a customer's subscriptions, in Stripe's own order.
    /// Private: a page is not an answer (see [`Self::in_force_subscription`]).
    async fn subscription_page(
        &self,
        customer_id: &str,
        starting_after: Option<&str>,
    ) -> Result<ListResponse<Subscription>, ServerError> {
        // No `status`: Stripe's default for this endpoint is every
        // subscription that has not been canceled, which is the smallest set
        // that can still contain an in-force one. Asking for `status=all`
        // would pad the pages with a customer's whole cancellation history
        // and make the walk longer for no answer it could change.
        let mut query: Vec<(&str, &str)> =
            vec![("customer", customer_id), ("limit", SUBSCRIPTION_PAGE_SIZE)];
        if let Some(cursor) = starting_after {
            query.push(("starting_after", cursor));
        }

        let response = self
            .client
            .get(format!("{}/subscriptions", self.base_url))
            .bearer_auth(&self.api_key)
            .query(&query)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!("stripe subscriptions: {}", parse_error_summary(&e)))
        })
    }

    /// List active prices with expanded product info.
    pub async fn list_prices(&self) -> Result<Vec<StripePrice>, ServerError> {
        self.list_all(
            "/prices",
            &[("active", "true"), ("expand[]", "data.product")],
        )
        .await
    }

    /// Fetch a single price to determine its type (recurring vs one-time).
    pub async fn get_price(&self, price_id: &str) -> Result<StripePriceMinimal, ServerError> {
        validate_id(price_id)?;
        let response = self
            .client
            .get(format!("{}/prices/{}", self.base_url, price_id))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe price: {}", parse_error_summary(&e))))
    }

    /// Fetch a single product by ID.
    pub async fn get_product(&self, product_id: &str) -> Result<StripeProduct, ServerError> {
        validate_id(product_id)?;
        let response = self
            .client
            .get(format!("{}/products/{}", self.base_url, product_id))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe product: {}", parse_error_summary(&e))))
    }

    /// The customer a charge was made for, if any. A Dispute carries only
    /// the disputed charge's id — no customer — so this is how a
    /// `charge.dispute.*` event reaches the account it concerns.
    pub async fn get_charge_customer(
        &self,
        charge_id: &str,
    ) -> Result<Option<String>, ServerError> {
        validate_id(charge_id)?;
        let response = self
            .client
            .get(format!("{}/charges/{}", self.base_url, charge_id))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let charge: StripeChargeCustomer = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!("stripe charge: {}", parse_error_summary(&e)))
        })?;
        Ok(charge.customer)
    }

    /// Point the client at a stub server.
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Create a Stripe Checkout Session and return the checkout URL.
    pub async fn create_checkout_session(
        &self,
        params: &CheckoutParams<'_>,
    ) -> Result<String, ServerError> {
        if params.credits <= 0 || !matches!(params.mode, "payment" | "subscription") {
            return Err(ServerError::Invalid("invalid checkout grant".into()));
        }
        validate_id(params.price_id)?;
        validate_id(params.customer_id)?;
        let credits = params.credits.to_string();
        let expiry = if params.mode == "subscription" {
            "billing_period"
        } else {
            "purchase_year"
        };
        let mut form: Vec<(&str, &str)> = vec![
            ("customer", params.customer_id),
            ("mode", params.mode),
            ("line_items[0][price]", params.price_id),
            ("line_items[0][quantity]", "1"),
            ("success_url", params.success_url),
            ("cancel_url", params.cancel_url),
            ("metadata[eidola_schema]", "1"),
            ("metadata[eidola_credits]", &credits),
            ("metadata[eidola_expiry]", expiry),
            ("metadata[eidola_price]", params.price_id),
        ];

        // Session metadata is not copied automatically. Stamp the same grant
        // on the payment/subscription; Stripe then snapshots subscription
        // metadata onto each new invoice's parent.subscription_details.
        if params.mode == "subscription" {
            form.extend([
                ("subscription_data[metadata][eidola_schema]", "1"),
                ("subscription_data[metadata][eidola_credits]", &credits),
                ("subscription_data[metadata][eidola_expiry]", expiry),
                ("subscription_data[metadata][eidola_price]", params.price_id),
            ]);
        } else {
            form.extend([
                ("payment_intent_data[metadata][eidola_schema]", "1"),
                ("payment_intent_data[metadata][eidola_credits]", &credits),
                ("payment_intent_data[metadata][eidola_expiry]", expiry),
                (
                    "payment_intent_data[metadata][eidola_price]",
                    params.price_id,
                ),
            ]);
        }

        if let Some(ref_id) = params.client_reference_id {
            form.push(("client_reference_id", ref_id));
        }

        if let Some(note) = params.submit_note {
            form.push(("custom_text[submit][message]", note));
        }

        if let Some(desc) = params.description {
            let key = if params.mode == "subscription" {
                "subscription_data[description]"
            } else {
                "payment_intent_data[description]"
            };
            form.push((key, desc));
        }

        let response = self
            .client
            .post(format!("{}/checkout/sessions", self.base_url))
            .bearer_auth(&self.api_key)
            .form(&form)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let session: CheckoutSession = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!("stripe checkout: {}", parse_error_summary(&e)))
        })?;

        session
            .url
            .ok_or_else(|| ServerError::Parse("stripe checkout session missing url".to_string()))
    }

    /// List line items for a checkout session (price expanded to get product ID).
    pub async fn list_checkout_line_items(
        &self,
        session_id: &str,
    ) -> Result<Vec<CheckoutLineItem>, ServerError> {
        validate_id(session_id)?;
        self.list_all(
            &format!("/checkout/sessions/{session_id}/line_items"),
            &[("expand[]", "data.price")],
        )
        .await
    }

    /// Create a Stripe billing portal session and return the portal URL.
    pub async fn create_portal_session(
        &self,
        customer_id: &str,
        return_url: &str,
    ) -> Result<String, ServerError> {
        let response = self
            .client
            .post(format!("{}/billing_portal/sessions", self.base_url))
            .bearer_auth(&self.api_key)
            .form(&[("customer", customer_id), ("return_url", return_url)])
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let session: PortalSession = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!("stripe portal: {}", parse_error_summary(&e)))
        })?;

        Ok(session.url)
    }
}

impl StripeClient {
    async fn response<T: serde::de::DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, ServerError> {
        let response = request
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;
        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e.without_url())))?;
        if !status.is_success() {
            // Only an explicit provider rejection can settle a write as
            // rejected. Timeouts, 409/429 and 5xx leave its outcome unknown.
            let invalid_request = serde_json::from_slice::<StripeErrorResponse>(&body)
                .ok()
                .and_then(|e| e.error.error_type)
                .as_deref()
                == Some("invalid_request_error");
            if matches!(status.as_u16(), 401 | 403 | 404)
                || (status.as_u16() == 400 && invalid_request)
            {
                return Err(ServerError::ProviderRejected(format!(
                    "stripe HTTP {}",
                    status.as_u16()
                )));
            }
            return Err(stripe_error(&body));
        }
        serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe: {}", parse_error_summary(&e))))
    }

    async fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, ServerError> {
        self.response(
            self.client
                .get(format!("{}{}", self.base_url, path))
                .bearer_auth(&self.api_key),
        )
        .await
    }

    /// A list is complete or it fails. Repeated cursors, empty nonterminal
    /// pages and the safety bound never become a partial authoritative answer.
    async fn list_all<T: serde::de::DeserializeOwned + ObjectId>(
        &self,
        path: &str,
        filters: &[(&str, &str)],
    ) -> Result<Vec<T>, ServerError> {
        let mut rows = Vec::new();
        let mut cursor: Option<String> = None;
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            let mut query = filters.to_vec();
            query.push(("limit", "100"));
            if let Some(id) = cursor.as_deref() {
                query.push(("starting_after", id));
            }
            let page: ListResponse<T> = self
                .response(
                    self.client
                        .get(format!("{}{}", self.base_url, path))
                        .bearer_auth(&self.api_key)
                        .query(&query),
                )
                .await?;
            let next = page.data.last().map(|item| item.object_id().to_owned());
            for item in &page.data {
                if !seen.insert(item.object_id().to_owned()) {
                    return Err(ServerError::ServiceUnavailable(
                        "stripe pagination repeated an object".into(),
                    ));
                }
            }
            rows.extend(page.data);
            if !page.has_more {
                return Ok(rows);
            }
            cursor = Some(next.ok_or_else(|| {
                ServerError::ServiceUnavailable(
                    "stripe pagination reported more results with no cursor".into(),
                )
            })?);
        }
        Err(ServerError::ServiceUnavailable(
            "stripe pagination safety bound reached".into(),
        ))
    }

    pub async fn get_price_expanded(&self, id: &str) -> Result<StripePrice, ServerError> {
        validate_id(id)?;
        self.response(
            self.client
                .get(format!("{}/prices/{id}", self.base_url))
                .bearer_auth(&self.api_key)
                .query(&[("expand[]", "product")]),
        )
        .await
    }

    pub async fn get_checkout(&self, id: &str) -> Result<PaidCheckout, ServerError> {
        validate_id(id)?;
        self.get(&format!("/checkout/sessions/{id}")).await
    }

    pub async fn get_invoice(&self, id: &str) -> Result<PaidInvoice, ServerError> {
        validate_id(id)?;
        self.get(&format!("/invoices/{id}")).await
    }

    pub async fn get_charge(&self, id: &str) -> Result<StripeCharge, ServerError> {
        validate_id(id)?;
        self.get(&format!("/charges/{id}")).await
    }

    pub async fn get_payment_intent(&self, id: &str) -> Result<StripePaymentIntent, ServerError> {
        validate_id(id)?;
        self.get(&format!("/payment_intents/{id}")).await
    }

    pub async fn list_paid_checkouts(
        &self,
        customer: &str,
    ) -> Result<Vec<PaidCheckout>, ServerError> {
        validate_id(customer)?;
        Ok(self
            .list_all::<PaidCheckout>(
                "/checkout/sessions",
                &[("customer", customer), ("status", "complete")],
            )
            .await?
            .into_iter()
            .filter(|s| {
                s.mode == "payment"
                    && matches!(s.payment_status.as_str(), "paid" | "no_payment_required")
            })
            .collect())
    }

    pub async fn list_paid_invoices(
        &self,
        customer: &str,
    ) -> Result<Vec<PaidInvoice>, ServerError> {
        validate_id(customer)?;
        self.list_all("/invoices", &[("customer", customer), ("status", "paid")])
            .await
    }

    pub async fn list_invoice_payments(
        &self,
        invoice: &str,
    ) -> Result<Vec<InvoicePayment>, ServerError> {
        validate_id(invoice)?;
        self.list_all(
            "/invoice_payments",
            &[("invoice", invoice), ("status", "paid")],
        )
        .await
    }

    pub async fn list_subscriptions(
        &self,
        customer: &str,
    ) -> Result<Vec<Subscription>, ServerError> {
        validate_id(customer)?;
        self.list_all(
            "/subscriptions",
            &[("customer", customer), ("status", "all")],
        )
        .await
    }

    pub async fn list_refunds(&self, charge: &str) -> Result<Vec<RefundSnapshot>, ServerError> {
        validate_id(charge)?;
        self.list_all::<StripeRefund>("/refunds", &[("charge", charge)])
            .await?
            .into_iter()
            .map(|refund| {
                if refund.charge.as_str() != charge {
                    return Err(malformed("refund charge mismatch"));
                }
                refund_snapshot(refund)
            })
            .collect()
    }

    pub async fn list_disputes(&self, charge: &str) -> Result<Vec<DisputeSnapshot>, ServerError> {
        validate_id(charge)?;
        self.list_all::<StripeDispute>("/disputes", &[("charge", charge)])
            .await?
            .into_iter()
            .map(|dispute| {
                if dispute.charge.as_str() != charge {
                    return Err(malformed("dispute charge mismatch"));
                }
                dispute_snapshot(dispute)
            })
            .collect()
    }

    /// Caller owns the durable intent and must recover a lost response before
    /// creating another refund. Stripe's idempotency retention is finite.
    pub async fn create_refund(
        &self,
        charge: &str,
        amount_minor: i64,
        operation_id: Uuid,
        idempotency_key: &str,
    ) -> Result<RefundSnapshot, ServerError> {
        validate_id(charge)?;
        if amount_minor <= 0 || idempotency_key.is_empty() || idempotency_key.len() > 255 {
            return Err(ServerError::Invalid("invalid refund intent".into()));
        }
        let amount = amount_minor.to_string();
        let operation = operation_id.to_string();
        let refund: StripeRefund = self
            .response(
                self.client
                    .post(format!("{}/refunds", self.base_url))
                    .bearer_auth(&self.api_key)
                    .header("Idempotency-Key", idempotency_key)
                    .form(&[
                        ("charge", charge),
                        ("amount", &amount),
                        ("metadata[eidola_operation]", &operation),
                    ]),
            )
            .await?;
        let refund = refund_snapshot(refund)?;
        if refund.charge_id != charge
            || refund.amount != amount_minor
            || refund.operation_id != Some(operation_id)
        {
            return Err(malformed("refund response does not match its intent"));
        }
        Ok(refund)
    }

    pub async fn cancel_subscription(&self, id: &str) -> Result<(), ServerError> {
        validate_id(id)?;
        // DELETE is idempotent by definition; Stripe ignores idempotency keys
        // on it. Do not prorate or generate another invoice during closure.
        let _: serde_json::Value = self
            .response(
                self.client
                    .delete(format!("{}/subscriptions/{id}", self.base_url))
                    .bearer_auth(&self.api_key)
                    .form(&[("invoice_now", "false"), ("prorate", "false")]),
            )
            .await?;
        Ok(())
    }

    /// Resolve only account-surface provider objects. The caller derives the
    /// account from the stored customer binding, never from webhook metadata.
    pub async fn event_customer(
        &self,
        object_kind: &str,
        id: &str,
    ) -> Result<Option<String>, ServerError> {
        validate_id(id)?;
        let customer = match object_kind {
            "charge" => self.get_charge_customer(id).await?,
            "checkout.session" => self
                .get_checkout(id)
                .await?
                .customer
                .map(|v| v.as_str().to_owned()),
            "invoice" => Some(self.get_invoice(id).await?.customer.as_str().to_owned()),
            "payment_intent" => self
                .get_payment_intent(id)
                .await?
                .customer
                .map(|v| v.as_str().to_owned()),
            "refund" => {
                let refund: StripeRefund = self.get(&format!("/refunds/{id}")).await?;
                self.get_charge_customer(refund.charge.as_str()).await?
            }
            "dispute" => {
                let dispute: StripeDispute = self.get(&format!("/disputes/{id}")).await?;
                self.get_charge_customer(dispute.charge.as_str()).await?
            }
            _ => {
                return Err(ServerError::Invalid(
                    "unsupported billing event object".into(),
                ));
            }
        };
        Ok(customer)
    }

    /// Fresh current state through the same provider reader for webhooks and
    /// explicit repair. No balances are inferred from an event's arrival order.
    pub async fn fetch_account_snapshot(
        &self,
        customer: &str,
    ) -> Result<AccountSnapshot, ServerError> {
        validate_id(customer)?;
        let mut snapshot = AccountSnapshot::default();
        let charges: Vec<StripeCharge> =
            self.list_all("/charges", &[("customer", customer)]).await?;
        let mut charge_map = HashMap::new();
        for charge in charges {
            if charge.customer.as_ref().map(StripeId::as_str) != Some(customer) {
                return Err(malformed("charge customer mismatch"));
            }
            validate_money(charge.amount, &charge.currency)?;
            // Collect every financial observation, even for a purchase whose
            // historical grant needs review. An unexplained refund must not
            // disappear merely because we cannot reconstruct its credit lot.
            snapshot
                .refunds
                .extend(self.list_refunds(&charge.id).await?);
            snapshot
                .disputes
                .extend(self.list_disputes(&charge.id).await?);
            charge_map.insert(charge.id.clone(), charge);
        }
        for checkout in self.list_paid_checkouts(customer).await? {
            let id = checkout.id.clone();
            match self
                .checkout_purchase(customer, checkout, &charge_map)
                .await
            {
                Ok(purchase) => snapshot.purchases.push(purchase),
                Err(ServerError::RequiresReview(reason)) => snapshot.review.push(ReviewItem {
                    object_id: id,
                    reason,
                }),
                Err(error) => return Err(error),
            }
        }
        for invoice in self.list_paid_invoices(customer).await? {
            if !matches!(
                invoice.billing_reason.as_deref(),
                Some("subscription_create" | "subscription_cycle")
            ) {
                continue;
            }
            let id = invoice.id.clone();
            match self.invoice_purchase(customer, invoice, &charge_map).await {
                Ok(purchase) => {
                    if purchase.expires_at <= purchase.purchased_at {
                        snapshot.review.push(ReviewItem {
                            object_id: purchase.id.clone(),
                            reason: "paid_after_service_period".into(),
                        });
                    }
                    snapshot.purchases.push(purchase);
                }
                Err(ServerError::RequiresReview(reason)) => snapshot.review.push(ReviewItem {
                    object_id: id,
                    reason,
                }),
                Err(error) => return Err(error),
            }
        }
        // One charge funding multiple purchases is outside this sale model.
        // Detect it before a caller applies any of this snapshot.
        let mut charge_counts: HashMap<String, usize> = HashMap::new();
        for purchase in &snapshot.purchases {
            if let Some(charge) = &purchase.charge_id {
                *charge_counts.entry(charge.clone()).or_default() += 1;
            }
        }
        snapshot.purchases.retain(|purchase| {
            if purchase
                .charge_id
                .as_ref()
                .is_some_and(|id| charge_counts[id] > 1)
            {
                snapshot.review.push(ReviewItem {
                    object_id: purchase.id.clone(),
                    reason: "shared_payment_requires_review".into(),
                });
                false
            } else {
                true
            }
        });
        let recognized: std::collections::HashSet<_> = snapshot
            .purchases
            .iter()
            .filter_map(|purchase| purchase.charge_id.as_deref())
            .collect();
        for charge in charge_map.values() {
            if charge.paid
                && charge.captured
                && charge.status == "succeeded"
                && !recognized.contains(charge.id.as_str())
            {
                snapshot.review.push(ReviewItem {
                    object_id: charge.id.clone(),
                    reason: "unrecognized_payment_requires_review".into(),
                });
            }
        }
        Ok(snapshot)
    }

    async fn payment_charge<'a>(
        &self,
        customer: &str,
        payment_intent: &str,
        amount: i64,
        currency: &str,
        charges: &'a HashMap<String, StripeCharge>,
    ) -> Result<&'a StripeCharge, ServerError> {
        let intent = self.get_payment_intent(payment_intent).await?;
        if intent.customer.as_ref().map(StripeId::as_str) != Some(customer)
            || intent.id != payment_intent
            || intent.status != "succeeded"
            || intent.amount_received != amount
            || intent.currency != currency
        {
            return Err(review("payment_amount_or_owner_requires_review"));
        }
        let charge_id = intent
            .latest_charge
            .as_ref()
            .ok_or_else(|| review("payment_charge_missing"))?
            .as_str();
        let charge = charges
            .get(charge_id)
            .ok_or_else(|| review("payment_charge_missing"))?;
        if !charge.paid
            || !charge.captured
            || charge.status != "succeeded"
            || charge.amount != amount
            || charge.currency != currency
            || charge.payment_intent.as_ref().map(StripeId::as_str) != Some(payment_intent)
        {
            return Err(review("payment_amount_or_owner_requires_review"));
        }
        Ok(charge)
    }

    async fn checkout_purchase(
        &self,
        customer: &str,
        checkout: PaidCheckout,
        charges: &HashMap<String, StripeCharge>,
    ) -> Result<PurchaseSnapshot, ServerError> {
        if checkout.customer.as_ref().map(StripeId::as_str) != Some(customer)
            || checkout.status.as_deref() != Some("complete")
        {
            return Err(malformed("checkout owner or status mismatch"));
        }
        let (credits, _) = grant_terms(&checkout.metadata, "purchase_year")?;
        let currency = checkout
            .currency
            .ok_or_else(|| review("purchase_currency_missing"))?;
        let paid_minor = checkout
            .amount_total
            .ok_or_else(|| review("purchase_amount_missing"))?;
        validate_money(paid_minor, &currency)?;
        let (charge_id, payment_intent, purchased_at) = if paid_minor == 0 {
            if checkout.payment_status != "no_payment_required" {
                return Err(review("zero_payment_requires_review"));
            }
            (None, None, timestamp(checkout.created)?)
        } else {
            if checkout.payment_status != "paid" {
                return Err(review("payment_not_paid"));
            }
            let intent = checkout
                .payment_intent
                .as_ref()
                .ok_or_else(|| review("payment_intent_missing"))?
                .as_str();
            let charge = self
                .payment_charge(customer, intent, paid_minor, &currency, charges)
                .await?;
            (
                Some(charge.id.clone()),
                Some(intent.to_owned()),
                timestamp(charge.created)?,
            )
        };
        let expires_at = purchased_at
            .checked_add(Duration::from_secs(365 * 24 * 3600))
            .ok_or_else(|| malformed("purchase expiry overflow"))?;
        Ok(PurchaseSnapshot {
            id: checkout.id,
            charge_id,
            payment_intent,
            invoice_id: None,
            currency,
            paid_minor,
            credits,
            expires_at,
            purchased_at,
            kind: "purchase".into(),
        })
    }

    async fn invoice_purchase(
        &self,
        customer: &str,
        invoice: PaidInvoice,
        charges: &HashMap<String, StripeCharge>,
    ) -> Result<PurchaseSnapshot, ServerError> {
        if invoice.customer.as_str() != customer || invoice.status != "paid" {
            return Err(malformed("invoice owner or status mismatch"));
        }
        validate_money(invoice.amount_paid, &invoice.currency)?;
        // A zero amount_paid is not necessarily a free grant: customer
        // balance can settle a positive invoice without any new charge. That
        // funding graph needs a reviewed import, not invented zero cash cost.
        if invoice.total != invoice.amount_paid || invoice.starting_balance != 0 {
            return Err(review("invoice_balance_funding_requires_review"));
        }
        let parent = invoice
            .parent
            .as_ref()
            .ok_or_else(|| review("subscription_snapshot_missing"))?;
        if parent.kind != "subscription_details" {
            return Err(review("subscription_snapshot_missing"));
        }
        let subscription = parent
            .subscription_details
            .as_ref()
            .ok_or_else(|| review("subscription_snapshot_missing"))?;
        let (credits, price) = grant_terms(&subscription.metadata, "billing_period")?;
        let lines = if invoice.lines.has_more {
            self.list_all::<InvoiceLine>(&format!("/invoices/{}/lines", invoice.id), &[])
                .await?
        } else {
            invoice.lines.data
        };
        if lines.len() != 1 {
            return Err(review("invoice_lines_require_review"));
        }
        let line = &lines[0];
        let matching_price = line
            .pricing
            .as_ref()
            .filter(|v| v.kind == "price_details")
            .and_then(|v| v.price_details.as_ref())
            .map(|v| v.price.as_str())
            == Some(price);
        let ordinary_line = line
            .parent
            .as_ref()
            .filter(|p| p.kind == "subscription_item_details")
            .and_then(|p| p.subscription_item_details.as_ref())
            .is_some_and(|v| !v.proration);
        if !matching_price
            || !ordinary_line
            || line.quantity != Some(1)
            || line.period.end <= line.period.start
        {
            return Err(review("invoice_lines_require_review"));
        }
        let purchased_at = timestamp(
            invoice
                .status_transitions
                .paid_at
                .ok_or_else(|| review("invoice_paid_time_missing"))?,
        )?;
        let expires_at = timestamp(line.period.end)?;
        let (charge_id, payment_intent) = if invoice.amount_paid == 0 {
            (None, None)
        } else {
            let payments = self.list_invoice_payments(&invoice.id).await?;
            if payments.len() != 1 {
                return Err(review("invoice_payments_require_review"));
            }
            let payment = &payments[0];
            if payment.invoice.as_str() != invoice.id
                || payment.status != "paid"
                || payment.amount_paid != Some(invoice.amount_paid)
                || payment.currency != invoice.currency
                || payment.payment.kind != "payment_intent"
            {
                return Err(review("invoice_payments_require_review"));
            }
            let intent = payment
                .payment
                .payment_intent
                .as_ref()
                .ok_or_else(|| review("payment_intent_missing"))?
                .as_str();
            let charge = self
                .payment_charge(
                    customer,
                    intent,
                    invoice.amount_paid,
                    &invoice.currency,
                    charges,
                )
                .await?;
            (Some(charge.id.clone()), Some(intent.to_owned()))
        };
        Ok(PurchaseSnapshot {
            id: invoice.id.clone(),
            charge_id,
            payment_intent,
            invoice_id: Some(invoice.id),
            currency: invoice.currency,
            paid_minor: invoice.amount_paid,
            credits,
            expires_at,
            purchased_at,
            kind: "subscription_renewal".into(),
        })
    }
}

fn validate_id(id: &str) -> Result<(), ServerError> {
    if id.is_empty()
        || id.len() > 255
        || !id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return Err(ServerError::Invalid(
            "invalid Stripe object identifier".into(),
        ));
    }
    Ok(())
}

fn validate_money(amount: i64, currency: &str) -> Result<(), ServerError> {
    if amount < 0 || currency.len() != 3 || !currency.bytes().all(|b| b.is_ascii_lowercase()) {
        return Err(malformed("invalid payment amount or currency"));
    }
    Ok(())
}

fn timestamp(value: i64) -> Result<SystemTime, ServerError> {
    let value: u64 = value
        .try_into()
        .map_err(|_| malformed("invalid provider timestamp"))?;
    UNIX_EPOCH
        .checked_add(Duration::from_secs(value))
        .ok_or_else(|| malformed("invalid provider timestamp"))
}

fn malformed(reason: &str) -> ServerError {
    ServerError::Parse(reason.into())
}
fn review(reason: &str) -> ServerError {
    ServerError::RequiresReview(reason.into())
}

fn grant_terms<'a>(
    metadata: &'a HashMap<String, String>,
    expiry: &str,
) -> Result<(i64, &'a str), ServerError> {
    if metadata.get("eidola_schema").map(String::as_str) != Some("1")
        || metadata.get("eidola_expiry").map(String::as_str) != Some(expiry)
    {
        return Err(review("grant_snapshot_missing"));
    }
    let credits = metadata
        .get("eidola_credits")
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| review("grant_snapshot_invalid"))?;
    let price = metadata
        .get("eidola_price")
        .filter(|v| !v.is_empty())
        .ok_or_else(|| review("grant_snapshot_invalid"))?;
    Ok((credits, price))
}

fn refund_snapshot(refund: StripeRefund) -> Result<RefundSnapshot, ServerError> {
    validate_money(refund.amount, &refund.currency)?;
    if refund.amount == 0
        || !matches!(
            refund.status.as_str(),
            "pending" | "requires_action" | "succeeded" | "failed" | "canceled"
        )
    {
        return Err(malformed("unknown refund status or zero refund"));
    }
    let operation_id = refund
        .metadata
        .get("eidola_operation")
        .and_then(|v| Uuid::parse_str(v).ok());
    Ok(RefundSnapshot {
        id: refund.id,
        charge_id: refund.charge.as_str().to_owned(),
        amount: refund.amount,
        currency: refund.currency,
        status: refund.status,
        operation_id,
    })
}

fn dispute_snapshot(dispute: StripeDispute) -> Result<DisputeSnapshot, ServerError> {
    validate_money(dispute.amount, &dispute.currency)?;
    if dispute.amount == 0
        || !matches!(
            dispute.status.as_str(),
            "needs_response"
                | "under_review"
                | "won"
                | "lost"
                | "warning_needs_response"
                | "warning_under_review"
                | "warning_closed"
        )
    {
        return Err(malformed("unknown dispute status or zero dispute"));
    }
    Ok(DisputeSnapshot {
        id: dispute.id,
        charge_id: dispute.charge.as_str().to_owned(),
        amount: dispute.amount,
        currency: dispute.currency,
        status: dispute.status,
    })
}

/// Parse a Stripe error response body into a ServerError.
///
/// The body is Stripe-authored: its `message` is free text that can quote
/// ids and customer-facing detail, and this error's `Display` reaches the
/// logs — so only the fixed-list resolution of Stripe's `type` classifier
/// is carried, never the raw body. Operators diagnose the specifics in
/// the Stripe dashboard, which holds the authoritative request log anyway.
fn stripe_error(body: &[u8]) -> ServerError {
    const STRIPE_ERROR_TYPES: &[&str] = &[
        "api_error",
        "card_error",
        "idempotency_error",
        "invalid_request_error",
    ];
    match serde_json::from_slice::<StripeErrorResponse>(body) {
        Ok(err) => {
            let label = err
                .error
                .error_type
                .as_deref()
                .and_then(|t| STRIPE_ERROR_TYPES.iter().copied().find(|known| *known == t))
                .unwrap_or("other");
            ServerError::Network(format!("stripe error: {label}"))
        }
        Err(_) => ServerError::Network("stripe error: unparseable error body".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sub(status: &str) -> Subscription {
        Subscription {
            id: "sub_1".to_string(),
            status: status.to_string(),
            items: SubscriptionItems { data: Vec::new() },
        }
    }

    #[test]
    fn in_force_covers_paying_trialing_and_retrying_subscriptions() {
        assert!(sub("active").is_in_force());
        assert!(sub("trialing").is_in_force());
        assert!(sub("past_due").is_in_force());
    }

    #[test]
    fn a_subscription_that_never_started_or_has_ended_is_not_in_force() {
        for status in [
            "incomplete",
            "incomplete_expired",
            "canceled",
            "unpaid",
            "paused",
        ] {
            assert!(!sub(status).is_in_force(), "{status} counted as in force");
        }
    }

    #[test]
    fn an_unrecognized_status_is_not_in_force() {
        // Stripe adding a status must not silently entitle an account.
        assert!(!sub("something_new").is_in_force());
    }

    // --- The subscription walk ------------------------------------------
    //
    // Stood up against a local stand-in for Stripe's list endpoint rather
    // than a mocked client: what is under test is the pagination protocol
    // (`limit`, `starting_after`, `has_more`), which only a real request
    // can exercise.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::Router;
    use axum::extract::{Query, State};
    use axum::routing::get;

    /// One page the stand-in will serve, in order.
    #[derive(Clone)]
    struct Page {
        statuses: Vec<&'static str>,
        has_more: bool,
    }

    #[derive(Clone)]
    struct Stub {
        pages: Vec<Page>,
        /// Every request's query string, in order — the pagination protocol
        /// as it actually went over the wire.
        queries: Arc<std::sync::Mutex<Vec<HashMap<String, String>>>>,
        /// The `Stripe-Version` header each request carried, if any.
        versions: Arc<std::sync::Mutex<Vec<Option<String>>>>,
        calls: Arc<AtomicUsize>,
    }

    impl Stub {
        /// The `starting_after` cursor of each request; `None` for a first
        /// page.
        fn cursors(&self) -> Vec<Option<String>> {
            self.queries
                .lock()
                .unwrap()
                .iter()
                .map(|q| q.get("starting_after").cloned())
                .collect()
        }

        fn limits(&self) -> Vec<Option<String>> {
            self.queries
                .lock()
                .unwrap()
                .iter()
                .map(|q| q.get("limit").cloned())
                .collect()
        }
    }

    async fn serve_page(
        State(stub): State<Stub>,
        headers: axum::http::HeaderMap,
        Query(params): Query<HashMap<String, String>>,
    ) -> axum::Json<serde_json::Value> {
        let n = stub.calls.fetch_add(1, Ordering::SeqCst);
        stub.queries.lock().unwrap().push(params);
        stub.versions.lock().unwrap().push(
            headers
                .get("Stripe-Version")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
        );

        // The last page repeats if the client asks past the end, so a client
        // that fails to stop is caught by the page cap rather than by a 404.
        let page = stub
            .pages
            .get(n)
            .cloned()
            .unwrap_or_else(|| stub.pages.last().unwrap().clone());

        // The Basil shape: the billing period lives on the items, and the
        // subscription itself has no `current_period_end` at all.
        let data: Vec<serde_json::Value> = page
            .statuses
            .iter()
            .enumerate()
            .map(|(i, status)| {
                serde_json::json!({
                    "id": format!("sub_{n}_{i}"),
                    "status": status,
                    "items": { "data": [{ "current_period_end": PERIOD_END }] },
                })
            })
            .collect();
        axum::Json(serde_json::json!({ "data": data, "has_more": page.has_more }))
    }

    /// An arbitrary but fixed period end the stand-in stamps on every item.
    const PERIOD_END: i64 = 1_900_000_000;

    /// Start the stand-in and return a client pointed at it.
    async fn stub_stripe(pages: Vec<Page>) -> (StripeClient, Stub) {
        let stub = Stub {
            pages,
            queries: Arc::new(std::sync::Mutex::new(Vec::new())),
            versions: Arc::new(std::sync::Mutex::new(Vec::new())),
            calls: Arc::new(AtomicUsize::new(0)),
        };
        let app = Router::new()
            .route("/subscriptions", get(serve_page))
            .with_state(stub.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        // The production constructor, redirected — so the walk is exercised
        // through the same HTTP client the server actually ships. That
        // client's TLS config wants the process-wide provider `main` installs
        // (idempotent here; several tests may reach this first).
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let mut client = StripeClient::new("sk_test".to_string());
        client.base_url = format!("http://{addr}");
        (client, stub)
    }

    #[tokio::test]
    async fn an_in_force_subscription_behind_a_full_page_is_still_found() {
        // The defect this walk exists for: a page of newer attempts that
        // entitle nothing, with the live subscription behind them. Read as
        // one page, this customer is "not subscribed" — and the checkout
        // guard asks the same question, so they could buy a second one.
        let (client, stub) = stub_stripe(vec![
            Page {
                statuses: vec!["incomplete"; 100],
                has_more: true,
            },
            Page {
                statuses: vec!["active"],
                has_more: false,
            },
        ])
        .await;

        let found = client.in_force_subscription("cus_1").await.unwrap();
        assert_eq!(
            found.map(|s| s.status),
            Some("active".to_string()),
            "the in-force subscription on the second page was missed"
        );
        assert_eq!(stub.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            stub.cursors(),
            vec![None, Some("sub_0_99".to_string())],
            "the second page must resume after the last id of the first"
        );
    }

    #[tokio::test]
    async fn a_customer_with_nothing_in_force_reads_as_none_once_the_list_ends() {
        let (client, stub) = stub_stripe(vec![Page {
            statuses: vec!["incomplete", "incomplete_expired"],
            has_more: false,
        }])
        .await;

        assert!(
            client
                .in_force_subscription("cus_1")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            stub.calls.load(Ordering::SeqCst),
            1,
            "a list that says it has no more pages must not be walked further"
        );
    }

    #[tokio::test]
    async fn the_walk_stops_at_the_first_page_that_answers() {
        let (client, stub) = stub_stripe(vec![
            Page {
                statuses: vec!["canceled", "active"],
                has_more: true,
            },
            Page {
                statuses: vec!["incomplete"],
                has_more: false,
            },
        ])
        .await;

        assert!(
            client
                .in_force_subscription("cus_1")
                .await
                .unwrap()
                .is_some()
        );
        assert_eq!(
            stub.calls.load(Ordering::SeqCst),
            1,
            "an answer found on the first page costs exactly one call"
        );
    }

    #[tokio::test]
    async fn a_walk_that_runs_out_of_pages_refuses_rather_than_saying_no() {
        // Saying "no subscription" about a prefix is the exact claim the
        // walk exists to stop making, so the cap refuses instead.
        let (client, stub) = stub_stripe(vec![Page {
            statuses: vec!["incomplete"],
            has_more: true,
        }])
        .await;

        let err = client.in_force_subscription("cus_1").await.unwrap_err();
        assert!(
            matches!(err, ServerError::ServiceUnavailable(_)),
            "expected a refusal, got {err:?}"
        );
        assert_eq!(stub.calls.load(Ordering::SeqCst), MAX_SUBSCRIPTION_PAGES);
    }

    #[tokio::test]
    async fn a_page_promising_more_but_carrying_none_refuses_too() {
        // The walk cannot continue (no cursor) and has not finished (Stripe
        // says there is more). That is an incomplete walk, not an ended one,
        // and "no subscription" said about an incomplete walk is what lets
        // the checkout guard sell a second subscription over a live one.
        let (client, stub) = stub_stripe(vec![Page {
            statuses: vec![],
            has_more: true,
        }])
        .await;

        let err = client.in_force_subscription("cus_1").await.unwrap_err();
        assert!(
            matches!(err, ServerError::ServiceUnavailable(_)),
            "expected a refusal, got {err:?}"
        );
        // Refusing on the spot: no cursor means nothing to retry with, so it
        // must not burn the page budget either.
        assert_eq!(stub.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn every_request_pins_the_api_version() {
        // Unpinned, a request is served at the account's default version —
        // a dashboard setting that can rewrite this server's behavior
        // without touching its measured source. That is how the billing
        // period went missing.
        let (client, stub) = stub_stripe(vec![Page {
            statuses: vec!["active"],
            has_more: false,
        }])
        .await;

        client.in_force_subscription("cus_1").await.unwrap();
        assert_eq!(
            *stub.versions.lock().unwrap(),
            vec![Some(STRIPE_API_VERSION.to_string())],
            "the Stripe-Version header must ride every request"
        );
    }

    #[tokio::test]
    async fn the_billing_period_is_read_from_the_subscriptions_items() {
        // Since Basil the subscription object has no `current_period_end`;
        // it lives on the items. Read from the subscription, the period is
        // simply never found and the app's "billing period ends …" line
        // never renders.
        let (client, _) = stub_stripe(vec![Page {
            statuses: vec!["active"],
            has_more: false,
        }])
        .await;

        let sub = client
            .in_force_subscription("cus_1")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            sub.current_period_end(),
            Some(PERIOD_END),
            "the billing period must come off the subscription's items"
        );
    }

    #[test]
    fn a_subscription_billing_on_mixed_intervals_reports_the_soonest_boundary() {
        // Items can bill on different intervals, so there is no single
        // period; the next thing that happens is what a reader is owed.
        let mixed = Subscription {
            id: "sub_1".to_string(),
            status: "active".to_string(),
            items: SubscriptionItems {
                data: vec![
                    SubscriptionItem {
                        current_period_end: 2_000,
                    },
                    SubscriptionItem {
                        current_period_end: 1_000,
                    },
                ],
            },
        };
        assert_eq!(mixed.current_period_end(), Some(1_000));
        // No items at all is the only way to have no period.
        assert_eq!(sub("active").current_period_end(), None);
    }

    #[tokio::test]
    async fn a_page_that_omits_has_more_refuses_rather_than_assuming_the_end() {
        // Defaulted to `false`, a missing `has_more` reads as "the whole
        // list has been seen" — so a shape change or a malformed
        // intermediary turns an incomplete walk into a confident "no
        // subscription", and the checkout guard sells a second subscription
        // over a live one. Absence must fail the parse instead.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = Router::new().route(
            "/subscriptions",
            get(|| async {
                // A page of non-entitling subscriptions and no `has_more` —
                // exactly what the old default silently read as "done".
                axum::Json(serde_json::json!({
                    "data": [{
                        "id": "sub_0",
                        "status": "incomplete",
                        "items": { "data": [{ "current_period_end": PERIOD_END }] },
                    }],
                }))
            }),
        );
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let mut client = StripeClient::new("sk_test".to_string());
        client.base_url = format!("http://{addr}");

        let err = client.in_force_subscription("cus_1").await.unwrap_err();
        assert!(
            matches!(err, ServerError::Parse(_)),
            "expected the missing field to fail the parse, got {err:?}"
        );
    }

    #[tokio::test]
    async fn the_walk_asks_for_stripes_largest_page() {
        // A small page size would make the cap bind on customers Stripe
        // itself considers ordinary, and puts the cost back in round trips.
        let (client, stub) = stub_stripe(vec![Page {
            statuses: vec![],
            has_more: false,
        }])
        .await;
        assert!(
            client
                .in_force_subscription("cus_1")
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(stub.limits(), vec![Some("100".to_string())]);
    }

    // --- The charge lookup ----------------------------------------------

    /// Serve one fixed JSON body at `/charges/{id}`.
    async fn stub_charge(body: serde_json::Value) -> StripeClient {
        use axum::routing::get;
        let app = axum::Router::new().route(
            "/charges/{id}",
            get(move || {
                let body = body.clone();
                async move { axum::Json(body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let mut client = StripeClient::new("sk_test".to_string());
        client.base_url = format!("http://{addr}");
        client
    }

    /// `customer: null` is a charge made without a customer, and the dispute
    /// handlers may end on it. A body with no `customer` key at all is not a
    /// Charge as Stripe defines one, and must fail the parse — the handlers'
    /// retryable branch — rather than read as the same orphan.
    #[tokio::test]
    async fn a_charge_without_the_customer_key_is_a_parse_error_not_an_orphan() {
        let explicit_null = stub_charge(serde_json::json!({
            "id": "ch_1", "object": "charge", "customer": null
        }))
        .await;
        assert_eq!(
            explicit_null.get_charge_customer("ch_1").await.unwrap(),
            None
        );

        let omitted = stub_charge(serde_json::json!({ "id": "ch_1", "object": "charge" })).await;
        let err = omitted.get_charge_customer("ch_1").await.unwrap_err();
        assert!(
            matches!(err, ServerError::Parse(_)),
            "expected the omitted field to fail the parse, got {err:?}"
        );
    }
}

#[cfg(test)]
mod reconciliation_tests {
    use super::*;
    use axum::{
        body::Bytes,
        extract::State,
        http::{HeaderMap, Method, StatusCode, Uri},
        response::IntoResponse,
    };
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    #[derive(Debug)]
    struct Request {
        method: String,
        uri: String,
        headers: HeaderMap,
        body: String,
    }

    #[derive(Clone)]
    struct Stub {
        responses: Arc<HashMap<String, (StatusCode, Value)>>,
        requests: Arc<Mutex<Vec<Request>>>,
    }

    async fn serve(
        State(stub): State<Stub>,
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
    ) -> impl IntoResponse {
        let response = stub
            .responses
            .get(&format!("{method} {}", uri.path()))
            .cloned()
            .unwrap_or((
                StatusCode::NOT_FOUND,
                json!({"error":{"type":"invalid_request_error"}}),
            ));
        stub.requests.lock().unwrap().push(Request {
            method: method.to_string(),
            uri: uri.to_string(),
            headers,
            body: String::from_utf8(body.to_vec()).unwrap(),
        });
        (response.0, axum::Json(response.1))
    }

    async fn stub(responses: HashMap<String, (StatusCode, Value)>) -> (StripeClient, Stub) {
        let state = Stub {
            responses: Arc::new(responses),
            requests: Arc::default(),
        };
        let router = axum::Router::new()
            .fallback(serve)
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        (
            StripeClient::new("sk_test".into()).with_base_url(format!("http://{address}")),
            state,
        )
    }

    fn list(data: Vec<Value>) -> Value {
        json!({"data":data,"has_more":false})
    }
    fn metadata(expiry: &str) -> Value {
        json!({"eidola_schema":"1","eidola_credits":"20000000","eidola_expiry":expiry,"eidola_price":"price_1"})
    }
    fn charge() -> Value {
        json!({"id":"ch_1","customer":"cus_1","payment_intent":"pi_1","amount":1500,
            "amount_refunded":0,"currency":"usd","paid":true,"captured":true,"status":"succeeded","created":1700000000})
    }
    fn intent() -> Value {
        json!({"id":"pi_1","customer":{"id":"cus_1"},"latest_charge":{"id":"ch_1"},
            "status":"succeeded","currency":"usd","amount_received":1500})
    }
    fn checkout() -> Value {
        json!({"id":"cs_1","customer":{"id":"cus_1"},"mode":"payment","payment_status":"paid","status":"complete",
            "amount_total":1500,"currency":"usd","payment_intent":"pi_1","metadata":metadata("purchase_year"),"created":1699990000})
    }
    fn invoice() -> Value {
        json!({"id":"in_1","customer":"cus_1","status":"paid","amount_paid":1500,"total":1500,"starting_balance":0,"currency":"usd",
            "billing_reason":"subscription_cycle","parent":{"type":"subscription_details","subscription_details":{
                "subscription":"sub_1","metadata":metadata("billing_period")}},
            "lines":{"has_more":false,"data":[{"id":"il_1","quantity":1,
                "period":{"start":1700000000,"end":1702592000},
                "pricing":{"type":"price_details","price_details":{"price":"price_1"}},
                "parent":{"type":"subscription_item_details","subscription_item_details":{"proration":false}}}]},
            "status_transitions":{"paid_at":1700000100}})
    }
    fn invoice_payment() -> Value {
        json!({"id":"inpay_1","invoice":"in_1","status":"paid","amount_paid":1500,"currency":"usd",
            "payment":{"type":"payment_intent","payment_intent":"pi_1"}})
    }
    fn refund() -> Value {
        json!({"id":"re_1","charge":"ch_1","amount":300,"currency":"usd","status":"succeeded","metadata":{}})
    }
    fn dispute() -> Value {
        json!({"id":"dp_1","charge":"ch_1","amount":1500,"currency":"usd","status":"won"})
    }
    fn responses() -> HashMap<String, (StatusCode, Value)> {
        [
            ("GET /charges", list(vec![charge()])),
            ("GET /checkout/sessions", list(vec![])),
            ("GET /invoices", list(vec![])),
            ("GET /refunds", list(vec![])),
            ("GET /disputes", list(vec![])),
            ("GET /payment_intents/pi_1", intent()),
            ("GET /invoice_payments", list(vec![invoice_payment()])),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), (StatusCode::OK, v)))
        .collect()
    }

    #[tokio::test]
    async fn basil_invoice_uses_paid_payment_mapping_and_snapshotted_grant() {
        let mut replies = responses();
        replies.insert(
            "GET /invoices".into(),
            (StatusCode::OK, list(vec![invoice()])),
        );
        let (stripe, stub) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert!(snapshot.review.is_empty());
        let purchase = &snapshot.purchases[0];
        assert_eq!(purchase.id, "in_1");
        assert_eq!(purchase.charge_id.as_deref(), Some("ch_1"));
        assert_eq!(purchase.payment_intent.as_deref(), Some("pi_1"));
        assert_eq!(
            purchase.paid_minor, 1500,
            "cash paid is independent of compute credit face value"
        );
        assert_eq!(purchase.credits, 20_000_000);
        assert_eq!(
            purchase.expires_at,
            UNIX_EPOCH + Duration::from_secs(1702592000)
        );
        assert!(
            stub.requests
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.uri.contains("invoice_payments?invoice=in_1&status=paid"))
        );
        assert!(
            stub.requests
                .lock()
                .unwrap()
                .iter()
                .all(|r| !r.uri.contains("products"))
        );
    }

    #[tokio::test]
    async fn unsupported_purchase_does_not_hide_refunds_or_won_disputes() {
        let mut old = checkout();
        old["metadata"] = json!({});
        let mut replies = responses();
        replies.insert(
            "GET /checkout/sessions".into(),
            (StatusCode::OK, list(vec![old])),
        );
        replies.insert(
            "GET /refunds".into(),
            (StatusCode::OK, list(vec![refund()])),
        );
        replies.insert(
            "GET /disputes".into(),
            (StatusCode::OK, list(vec![dispute()])),
        );
        let (stripe, _) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert!(snapshot.purchases.is_empty());
        assert_eq!(snapshot.review[0].reason, "grant_snapshot_missing");
        assert_eq!(snapshot.refunds[0].id, "re_1");
        assert_eq!(snapshot.disputes[0].status, "won");
    }

    #[tokio::test]
    async fn unpaid_checkout_is_never_a_grant_and_paid_expiry_uses_charge_time() {
        let mut unpaid = checkout();
        unpaid["id"] = json!("cs_unpaid");
        unpaid["payment_status"] = json!("unpaid");
        let mut replies = responses();
        replies.insert(
            "GET /checkout/sessions".into(),
            (StatusCode::OK, list(vec![unpaid, checkout()])),
        );
        let (stripe, _) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert_eq!(snapshot.purchases.len(), 1);
        assert_eq!(snapshot.purchases[0].id, "cs_1");
        assert_eq!(
            snapshot.purchases[0].expires_at,
            UNIX_EPOCH + Duration::from_secs(1700000000 + 365 * 24 * 3600)
        );
    }

    #[tokio::test]
    async fn invoice_with_multiple_payments_requires_review_without_guessing_a_charge() {
        let mut second = invoice_payment();
        second["id"] = json!("inpay_2");
        let mut replies = responses();
        replies.insert(
            "GET /invoices".into(),
            (StatusCode::OK, list(vec![invoice()])),
        );
        replies.insert(
            "GET /invoice_payments".into(),
            (StatusCode::OK, list(vec![invoice_payment(), second])),
        );
        let (stripe, stub) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert!(snapshot.purchases.is_empty());
        assert_eq!(snapshot.review[0].reason, "invoice_payments_require_review");
        assert!(
            !stub
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.uri.starts_with("/payment_intents/"))
        );
    }

    #[tokio::test]
    async fn same_payment_cannot_fund_two_purchase_grants() {
        let mut replies = responses();
        replies.insert(
            "GET /checkout/sessions".into(),
            (StatusCode::OK, list(vec![checkout()])),
        );
        replies.insert(
            "GET /invoices".into(),
            (StatusCode::OK, list(vec![invoice()])),
        );
        let (stripe, _) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert!(snapshot.purchases.is_empty());
        assert_eq!(
            snapshot
                .review
                .iter()
                .filter(|r| r.reason == "shared_payment_requires_review")
                .count(),
            2
        );
        assert!(
            snapshot.review.iter().any(
                |r| r.object_id == "ch_1" && r.reason == "unrecognized_payment_requires_review"
            )
        );
    }

    #[tokio::test]
    async fn zero_paid_invoice_grants_non_cash_credits_without_a_payment_lookup() {
        let mut free = invoice();
        free["amount_paid"] = json!(0);
        free["total"] = json!(0);
        let mut replies = responses();
        replies.insert("GET /invoices".into(), (StatusCode::OK, list(vec![free])));
        let (stripe, stub) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert_eq!(snapshot.purchases[0].paid_minor, 0);
        assert!(snapshot.purchases[0].charge_id.is_none());
        assert!(
            !stub
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|r| r.uri.starts_with("/invoice_payments"))
        );
    }

    #[tokio::test]
    async fn customer_balance_settlement_is_not_misclassified_as_a_free_grant() {
        let mut funded = invoice();
        funded["amount_paid"] = json!(0);
        funded["starting_balance"] = json!(-1500);
        let mut replies = responses();
        replies.insert("GET /invoices".into(), (StatusCode::OK, list(vec![funded])));
        let (stripe, _) = stub(replies).await;
        let snapshot = stripe.fetch_account_snapshot("cus_1").await.unwrap();
        assert!(snapshot.purchases.is_empty());
        assert_eq!(
            snapshot.review[0].reason,
            "invoice_balance_funding_requires_review"
        );
    }

    #[tokio::test]
    async fn incomplete_or_repeated_pages_do_not_become_authoritative_refund_lists() {
        for body in [
            json!({"data":[],"has_more":true}),
            json!({"data":[]}),
            json!({"data":[refund()],"has_more":true}),
        ] {
            let (stripe, _) = stub(HashMap::from([(
                "GET /refunds".into(),
                (StatusCode::OK, body),
            )]))
            .await;
            assert!(stripe.list_refunds("ch_1").await.is_err());
        }
    }

    #[tokio::test]
    async fn refund_creation_carries_the_durable_operation_and_idempotency_key() {
        let operation = Uuid::new_v4();
        let mut created = refund();
        created["metadata"] = json!({"eidola_operation":operation.to_string()});
        created["status"] = json!("pending");
        let (stripe, stub) = stub(HashMap::from([(
            "POST /refunds".into(),
            (StatusCode::OK, created),
        )]))
        .await;
        let refund = stripe
            .create_refund("ch_1", 300, operation, "durable_attempt_1")
            .await
            .unwrap();
        assert_eq!(refund.operation_id, Some(operation));
        assert_eq!(refund.status, "pending");
        let requests = stub.requests.lock().unwrap();
        assert_eq!(requests[0].method, "POST");
        assert_eq!(requests[0].headers["idempotency-key"], "durable_attempt_1");
        assert_eq!(requests[0].headers["stripe-version"], STRIPE_API_VERSION);
        assert!(requests[0].body.contains("amount=300"));
        assert!(
            requests[0]
                .body
                .contains(&format!("metadata%5Beidola_operation%5D={operation}"))
        );
    }

    #[tokio::test]
    async fn ambiguous_refund_failure_is_never_reported_as_a_definitive_rejection() {
        for (status, kind) in [
            (StatusCode::INTERNAL_SERVER_ERROR, "api_error"),
            (StatusCode::CONFLICT, "invalid_request_error"),
            (StatusCode::TOO_MANY_REQUESTS, "api_error"),
            (StatusCode::BAD_REQUEST, "idempotency_error"),
        ] {
            let (stripe, _) = stub(HashMap::from([(
                "POST /refunds".into(),
                (
                    status,
                    json!({"error":{"type":kind,"message":"secret customer details"}}),
                ),
            )]))
            .await;
            let error = stripe
                .create_refund("ch_1", 300, Uuid::new_v4(), "attempt")
                .await
                .unwrap_err();
            assert!(!matches!(error, ServerError::ProviderRejected(_)));
            assert!(!error.to_string().contains("secret customer details"));
        }
    }

    #[tokio::test]
    async fn checkout_snapshots_terms_on_both_session_and_subscription() {
        let (stripe, stub) = stub(HashMap::from([(
            "POST /checkout/sessions".into(),
            (
                StatusCode::OK,
                json!({"url":"https://checkout.stripe.test/session"}),
            ),
        )]))
        .await;
        stripe
            .create_checkout_session(&CheckoutParams {
                customer_id: "cus_1",
                price_id: "price_1",
                mode: "subscription",
                success_url: "https://example.test/success",
                cancel_url: "https://example.test/cancel",
                client_reference_id: None,
                submit_note: None,
                description: None,
                credits: 20_000_000,
            })
            .await
            .unwrap();
        let requests = stub.requests.lock().unwrap();
        assert!(
            requests[0]
                .body
                .contains("metadata%5Beidola_credits%5D=20000000")
        );
        assert!(
            requests[0]
                .body
                .contains("subscription_data%5Bmetadata%5D%5Beidola_credits%5D=20000000")
        );
        assert!(
            requests[0]
                .body
                .contains("metadata%5Beidola_expiry%5D=billing_period")
        );
    }

    #[test]
    fn provider_ids_cannot_inject_paths_and_unknown_states_do_not_mutate_money() {
        assert!(validate_id("ch_1/../refunds").is_err());
        assert!(validate_id("ch_1?customer=cus_other").is_err());
        let mut value = refund();
        value["status"] = json!("new_provider_state");
        assert!(refund_snapshot(serde_json::from_value(value).unwrap()).is_err());
        let mut value = dispute();
        value["status"] = json!("new_provider_state");
        assert!(dispute_snapshot(serde_json::from_value(value).unwrap()).is_err());
    }
}
