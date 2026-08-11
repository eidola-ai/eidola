//! Thin Stripe API client.
//!
//! Only the endpoints needed for account management are implemented.
//! Uses form-encoded bodies (Stripe's native format) and Bearer auth.

use std::collections::HashMap;

use serde::Deserialize;
use uuid::Uuid;

use crate::error::ServerError;

/// Minimal Stripe subscription representation.
#[derive(Debug, Deserialize)]
pub struct Subscription {
    pub id: String,
    pub status: String,
    #[serde(default)]
    pub current_period_end: Option<i64>,
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
}

/// Stripe list response wrapper. `has_more` is what makes a walk over a
/// paginated list terminate on Stripe's own word rather than on a guess
/// about how full a page looks.
#[derive(Debug, Deserialize)]
struct ListResponse<T> {
    pub data: Vec<T>,
    #[serde(default)]
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
}

pub struct StripeClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl StripeClient {
    pub fn new(api_key: String) -> Self {
        let client = reqwest::Client::builder()
            .tls_backend_preconfigured(crate::tls_config())
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
            ServerError::Parse(format!(
                "stripe customer: {}",
                crate::error::parse_error_summary(&e)
            ))
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
                // "More" after an empty page: there is no cursor to advance
                // on, so treating it as the end is the only non-looping
                // reading. Stripe does not do this; the arm exists so a
                // surprise cannot spin.
                (true, None) => return Ok(None),
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
            ServerError::Parse(format!(
                "stripe subscriptions: {}",
                crate::error::parse_error_summary(&e)
            ))
        })
    }

    /// List active prices with expanded product info.
    pub async fn list_prices(&self) -> Result<Vec<StripePrice>, ServerError> {
        let response = self
            .client
            .get(format!("{}/prices", self.base_url))
            .bearer_auth(&self.api_key)
            .query(&[
                ("active", "true"),
                ("expand[]", "data.product"),
                ("limit", "100"),
            ])
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

        let list: ListResponse<StripePrice> = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!(
                "stripe prices: {}",
                crate::error::parse_error_summary(&e)
            ))
        })?;

        Ok(list.data)
    }

    /// Fetch a single price to determine its type (recurring vs one-time).
    pub async fn get_price(&self, price_id: &str) -> Result<StripePriceMinimal, ServerError> {
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

        serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!(
                "stripe price: {}",
                crate::error::parse_error_summary(&e)
            ))
        })
    }

    /// Fetch a single product by ID.
    pub async fn get_product(&self, product_id: &str) -> Result<StripeProduct, ServerError> {
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

        serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!(
                "stripe product: {}",
                crate::error::parse_error_summary(&e)
            ))
        })
    }

    /// Create a Stripe Checkout Session and return the checkout URL.
    pub async fn create_checkout_session(
        &self,
        params: &CheckoutParams<'_>,
    ) -> Result<String, ServerError> {
        let mut form: Vec<(&str, &str)> = vec![
            ("customer", params.customer_id),
            ("mode", params.mode),
            ("line_items[0][price]", params.price_id),
            ("line_items[0][quantity]", "1"),
            ("success_url", params.success_url),
            ("cancel_url", params.cancel_url),
        ];

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
            ServerError::Parse(format!(
                "stripe checkout: {}",
                crate::error::parse_error_summary(&e)
            ))
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
        let response = self
            .client
            .get(format!(
                "{}/checkout/sessions/{}/line_items",
                self.base_url, session_id
            ))
            .bearer_auth(&self.api_key)
            .query(&[("expand[]", "data.price")])
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

        let list: ListResponse<CheckoutLineItem> = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!(
                "stripe line items: {}",
                crate::error::parse_error_summary(&e)
            ))
        })?;

        Ok(list.data)
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
            ServerError::Parse(format!(
                "stripe portal: {}",
                crate::error::parse_error_summary(&e)
            ))
        })?;

        Ok(session.url)
    }
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
            current_period_end: None,
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
        Query(params): Query<HashMap<String, String>>,
    ) -> axum::Json<serde_json::Value> {
        let n = stub.calls.fetch_add(1, Ordering::SeqCst);
        stub.queries.lock().unwrap().push(params);

        // The last page repeats if the client asks past the end, so a client
        // that fails to stop is caught by the page cap rather than by a 404.
        let page = stub
            .pages
            .get(n)
            .cloned()
            .unwrap_or_else(|| stub.pages.last().unwrap().clone());

        let data: Vec<serde_json::Value> = page
            .statuses
            .iter()
            .enumerate()
            .map(
                |(i, status)| serde_json::json!({ "id": format!("sub_{n}_{i}"), "status": status }),
            )
            .collect();
        axum::Json(serde_json::json!({ "data": data, "has_more": page.has_more }))
    }

    /// Start the stand-in and return a client pointed at it.
    async fn stub_stripe(pages: Vec<Page>) -> (StripeClient, Stub) {
        let stub = Stub {
            pages,
            queries: Arc::new(std::sync::Mutex::new(Vec::new())),
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
}
