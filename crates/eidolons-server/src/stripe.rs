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
    pub message: String,
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
}

pub struct StripeClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

impl StripeClient {
    pub fn new(api_key: String) -> Self {
        let client = reqwest::Client::builder()
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
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let customer: Customer = serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe customer: {}", e)))?;

        Ok(customer.id)
    }

    /// List subscriptions for a Stripe customer.
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
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let list: ListResponse<Subscription> = serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe subscriptions: {}", e)))?;

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
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let list: ListResponse<StripePrice> = serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe prices: {}", e)))?;

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
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe price: {}", e)))
    }

    /// Fetch a single product by ID.
    pub async fn get_product(&self, product_id: &str) -> Result<StripeProduct, ServerError> {
        let response = self
            .client
            .get(format!("{}/products/{}", self.base_url, product_id))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe product: {}", e)))
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

        let response = self
            .client
            .post(format!("{}/checkout/sessions", self.base_url))
            .bearer_auth(&self.api_key)
            .form(&form)
            .send()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let session: CheckoutSession = serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe checkout: {}", e)))?;

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
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let list: ListResponse<CheckoutLineItem> = serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe line items: {}", e)))?;

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
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        let status = response.status();
        let body = response
            .bytes()
            .await
            .map_err(|e| ServerError::Network(format!("stripe: {}", e)))?;

        if !status.is_success() {
            return Err(stripe_error(&body));
        }

        let session: PortalSession = serde_json::from_slice(&body)
            .map_err(|e| ServerError::Parse(format!("stripe portal: {}", e)))?;

        Ok(session.url)
    }
}

/// Parse a Stripe error response body into a ServerError.
fn stripe_error(body: &[u8]) -> ServerError {
    if let Ok(err) = serde_json::from_slice::<StripeErrorResponse>(body) {
        ServerError::Network(format!("stripe error: {}", err.error.message))
    } else {
        ServerError::Network(format!("stripe error: {}", String::from_utf8_lossy(body)))
    }
}
