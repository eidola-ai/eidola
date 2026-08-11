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

impl Subscription {
    /// Whether this subscription is in force — see
    /// [`IN_FORCE_SUBSCRIPTION_STATUSES`].
    pub fn is_in_force(&self) -> bool {
        IN_FORCE_SUBSCRIPTION_STATUSES.contains(&self.status.as_str())
    }
}

/// Stripe list response wrapper.
#[derive(Debug, Deserialize)]
struct ListResponse<T> {
    pub data: Vec<T>,
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

    /// List a Stripe customer's subscriptions, verbatim.
    ///
    /// Stripe's list endpoint takes at most one `status` value, so it
    /// cannot express the in-force set; callers filter with
    /// [`Subscription::is_in_force`]. Unfiltered, this returns everything
    /// Stripe returns by default — every status except `canceled`.
    pub async fn list_subscriptions(
        &self,
        customer_id: &str,
    ) -> Result<Vec<Subscription>, ServerError> {
        let response = self
            .client
            .get(format!("{}/subscriptions", self.base_url))
            .bearer_auth(&self.api_key)
            .query(&[("customer", customer_id), ("limit", "10")])
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

        let list: ListResponse<Subscription> = serde_json::from_slice(&body).map_err(|e| {
            ServerError::Parse(format!(
                "stripe subscriptions: {}",
                crate::error::parse_error_summary(&e)
            ))
        })?;

        Ok(list.data)
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
}
