use crux_core::{
    App, Command,
    macros::effect,
    render::{RenderOperation, render},
};
use crux_http::{
    command::Http,
    protocol::HttpRequest,
};
use serde::{Deserialize, Serialize};

// ── Type alias for the Command used by Http ─────────────────────────────────

type Http_ = Http<Effect, Event>;

// ── Helper: build Basic auth header value ───────────────────────────────────

fn basic_auth(id: &str, secret: &str) -> String {
    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(format!("{id}:{secret}"));
    format!("Basic {encoded}")
}

// ── Events ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum Event {
    // ── Events from the shell ─────────────────────────────────────────
    /// Shell sends config at startup
    Init {
        base_url: String,
        account_id: Option<String>,
        account_secret: Option<String>,
    },
    GetAccount,
    CreateAccount,
    GetPrices,
    Checkout { price_id: String },
    GetBalances,

    // ── Events local to the core ──────────────────────────────────────
    // These never cross the FFI boundary — the core creates them
    // internally when capability responses arrive. Must be grouped at
    // the end so #[serde(skip)] doesn't shift non-skipped indices.
    #[serde(skip)]
    GetAccountDone(crux_http::Result<crux_http::Response<GetAccountBody>>),
    #[serde(skip)]
    CreateAccountDone(crux_http::Result<crux_http::Response<CreateAccountBody>>),
    #[serde(skip)]
    GetPricesDone(crux_http::Result<crux_http::Response<ListPricesBody>>),
    #[serde(skip)]
    CheckoutDone(crux_http::Result<crux_http::Response<CheckoutBody>>),
    #[serde(skip)]
    GetBalancesDone(crux_http::Result<crux_http::Response<BalancesBody>>),
}

// ── Response body types (mirror server API) ─────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct GetAccountBody {
    pub id: String,
    pub stripe_customer_id: Option<String>,
    pub created_at: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CreateAccountBody {
    pub account_id: String,
    pub secret: String,
    pub created_at: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ListPricesBody {
    pub data: Vec<PriceBody>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PriceBody {
    pub id: String,
    pub product_name: String,
    pub product_description: Option<String>,
    pub unit_amount: Option<i64>,
    pub currency: String,
    pub recurring: Option<RecurringBody>,
    pub credits: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RecurringBody {
    pub interval: String,
    pub interval_count: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CheckoutBody {
    pub checkout_url: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BalancesBody {
    pub available: i64,
    pub pools: Vec<BalancePoolBody>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BalancePoolBody {
    pub amount: i64,
    pub source: String,
    pub expires_at: Option<String>,
}


// ── Screen ──────────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    #[default]
    Idle,
    Loading,
    Account,
    AccountCreated,
    Prices,
    Checkout,
    Balances,
    Error,
}

// ── Model ───────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct Model {
    base_url: String,
    account_id: Option<String>,
    account_secret: Option<String>,
    screen: Screen,
    error: Option<String>,
    account: Option<GetAccountBody>,
    created_account: Option<CreateAccountBody>,
    prices: Vec<PriceBody>,
    checkout_url: Option<String>,
    balances: Option<BalancesBody>,
}

// ── ViewModel ───────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Clone, Default, Debug, PartialEq)]
pub struct ViewModel {
    pub screen: Screen,
    pub error: Option<String>,
    // Account
    pub account_id: Option<String>,
    pub account_stripe_customer_id: Option<String>,
    pub account_created_at: Option<String>,
    // Created account
    pub created_account_id: Option<String>,
    pub created_account_secret: Option<String>,
    pub created_account_created_at: Option<String>,
    // Prices
    pub prices: Vec<PriceViewModel>,
    // Checkout
    pub checkout_url: Option<String>,
    // Balances
    pub balances_available: Option<i64>,
    pub balances_pools: Vec<BalancePoolViewModel>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PriceViewModel {
    pub id: String,
    pub product_name: String,
    pub product_description: Option<String>,
    pub unit_amount: Option<i64>,
    pub currency: String,
    pub recurring_interval: Option<String>,
    pub recurring_interval_count: Option<i64>,
    pub credits: i64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct BalancePoolViewModel {
    pub amount: i64,
    pub source: String,
    pub expires_at: Option<String>,
}

// ── Effect ──────────────────────────────────────────────────────────────────

#[effect(typegen)]
pub enum Effect {
    Render(RenderOperation),
    Http(HttpRequest),
}

// ── App ─────────────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct EidolonsApp;

impl App for EidolonsApp {
    type Event = Event;
    type Model = Model;
    type ViewModel = ViewModel;
    type Capabilities = ();
    type Effect = Effect;

    fn update(
        &self,
        event: Self::Event,
        model: &mut Self::Model,
        _caps: &Self::Capabilities,
    ) -> Command<Self::Effect, Self::Event> {
        match event {
            Event::Init {
                base_url,
                account_id,
                account_secret,
            } => {
                model.base_url = base_url;
                model.account_id = account_id;
                model.account_secret = account_secret;
                Command::done()
            }

            // ── GetAccount ──────────────────────────────────────────────
            Event::GetAccount => {
                let (id, secret) = match (&model.account_id, &model.account_secret) {
                    (Some(id), Some(secret)) => (id.clone(), secret.clone()),
                    _ => return set_error(model, "account not configured"),
                };
                model.screen = Screen::Loading;
                Http_::get(format!("{}/v1/account", model.base_url))
                    .header("Authorization", basic_auth(&id, &secret))
                    .expect_json::<GetAccountBody>()
                    .build()
                    .then_send(Event::GetAccountDone)
            }
            Event::GetAccountDone(Ok(mut resp)) => {
                model.account = resp.take_body();
                model.screen = Screen::Account;
                render()
            }
            Event::GetAccountDone(Err(e)) => set_error(model, &e.to_string()),

            // ── CreateAccount ───────────────────────────────────────────
            Event::CreateAccount => {
                model.screen = Screen::Loading;
                Http_::post(format!("{}/v1/account", model.base_url))
                    .expect_json::<CreateAccountBody>()
                    .build()
                    .then_send(Event::CreateAccountDone)
            }
            Event::CreateAccountDone(Ok(mut resp)) => {
                let body = resp.take_body();
                if let Some(ref b) = body {
                    model.account_id = Some(b.account_id.clone());
                    model.account_secret = Some(b.secret.clone());
                }
                model.created_account = body;
                model.screen = Screen::AccountCreated;
                render()
            }
            Event::CreateAccountDone(Err(e)) => set_error(model, &e.to_string()),

            // ── GetPrices ───────────────────────────────────────────────
            Event::GetPrices => {
                model.screen = Screen::Loading;
                Http_::get(format!("{}/v1/prices", model.base_url))
                    .expect_json::<ListPricesBody>()
                    .build()
                    .then_send(Event::GetPricesDone)
            }
            Event::GetPricesDone(Ok(mut resp)) => {
                if let Some(body) = resp.take_body() {
                    model.prices = body.data;
                }
                model.screen = Screen::Prices;
                render()
            }
            Event::GetPricesDone(Err(e)) => set_error(model, &e.to_string()),

            // ── Checkout ────────────────────────────────────────────────
            Event::Checkout { price_id } => {
                let (id, secret) = match (&model.account_id, &model.account_secret) {
                    (Some(id), Some(secret)) => (id.clone(), secret.clone()),
                    _ => return set_error(model, "account not configured"),
                };
                model.screen = Screen::Loading;
                Http_::post(format!("{}/v1/account/checkout", model.base_url))
                    .header("Authorization", basic_auth(&id, &secret))
                    .body_json(&serde_json::json!({ "price_id": price_id }))
                    .expect("serialize checkout body")
                    .expect_json::<CheckoutBody>()
                    .build()
                    .then_send(Event::CheckoutDone)
            }
            Event::CheckoutDone(Ok(mut resp)) => {
                model.checkout_url = resp.take_body().map(|b| b.checkout_url);
                model.screen = Screen::Checkout;
                render()
            }
            Event::CheckoutDone(Err(e)) => set_error(model, &e.to_string()),

            // ── GetBalances ─────────────────────────────────────────────
            Event::GetBalances => {
                let (id, secret) = match (&model.account_id, &model.account_secret) {
                    (Some(id), Some(secret)) => (id.clone(), secret.clone()),
                    _ => return set_error(model, "account not configured"),
                };
                model.screen = Screen::Loading;
                Http_::get(format!("{}/v1/account/balances", model.base_url))
                    .header("Authorization", basic_auth(&id, &secret))
                    .expect_json::<BalancesBody>()
                    .build()
                    .then_send(Event::GetBalancesDone)
            }
            Event::GetBalancesDone(Ok(mut resp)) => {
                model.balances = resp.take_body();
                model.screen = Screen::Balances;
                render()
            }
            Event::GetBalancesDone(Err(e)) => set_error(model, &e.to_string()),
        }
    }

    fn view(&self, model: &Self::Model) -> Self::ViewModel {
        ViewModel {
            screen: model.screen.clone(),
            error: model.error.clone(),
            // Account
            account_id: model.account.as_ref().map(|a| a.id.clone()),
            account_stripe_customer_id: model
                .account
                .as_ref()
                .and_then(|a| a.stripe_customer_id.clone()),
            account_created_at: model.account.as_ref().map(|a| a.created_at.clone()),
            // Created account
            created_account_id: model
                .created_account
                .as_ref()
                .map(|a| a.account_id.clone()),
            created_account_secret: model.created_account.as_ref().map(|a| a.secret.clone()),
            created_account_created_at: model
                .created_account
                .as_ref()
                .map(|a| a.created_at.clone()),
            // Prices
            prices: model
                .prices
                .iter()
                .map(|p| PriceViewModel {
                    id: p.id.clone(),
                    product_name: p.product_name.clone(),
                    product_description: p.product_description.clone(),
                    unit_amount: p.unit_amount,
                    currency: p.currency.clone(),
                    recurring_interval: p.recurring.as_ref().map(|r| r.interval.clone()),
                    recurring_interval_count: p.recurring.as_ref().map(|r| r.interval_count),
                    credits: p.credits,
                })
                .collect(),
            // Checkout
            checkout_url: model.checkout_url.clone(),
            // Balances
            balances_available: model.balances.as_ref().map(|b| b.available),
            balances_pools: model
                .balances
                .as_ref()
                .map(|b| {
                    b.pools
                        .iter()
                        .map(|p| BalancePoolViewModel {
                            amount: p.amount,
                            source: p.source.clone(),
                            expires_at: p.expires_at.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

fn set_error(model: &mut Model, message: &str) -> Command<Effect, Event> {
    model.error = Some(message.to_string());
    model.screen = Screen::Error;
    render()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crux_core::testing::AppTester;
    #[test]
    fn test_init_stores_config() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();

        let _ = app.update(
            Event::Init {
                base_url: "http://localhost:8080".into(),
                account_id: Some("test-id".into()),
                account_secret: Some("test-secret".into()),
            },
            &mut model,
        );

        assert_eq!(model.base_url, "http://localhost:8080");
        assert_eq!(model.account_id.as_deref(), Some("test-id"));
        assert_eq!(model.account_secret.as_deref(), Some("test-secret"));
    }

    #[test]
    fn test_get_account_without_credentials_errors() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();
        model.base_url = "http://localhost:8080".into();

        let cmd = app.update(Event::GetAccount, &mut model);

        assert_eq!(model.screen, Screen::Error);
        assert!(model.error.as_ref().unwrap().contains("not configured"));
        let _ = cmd.expect_one_effect();
    }

    #[test]
    fn test_get_account_emits_http_effect() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();
        model.base_url = "http://localhost:8080".into();
        model.account_id = Some("test-id".into());
        model.account_secret = Some("test-secret".into());

        let cmd = app.update(Event::GetAccount, &mut model);

        assert_eq!(model.screen, Screen::Loading);
        let effect = cmd.expect_one_effect();
        assert!(matches!(effect, Effect::Http(_)));
    }

    #[test]
    fn test_create_account_emits_http_effect() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();
        model.base_url = "http://localhost:8080".into();

        let cmd = app.update(Event::CreateAccount, &mut model);

        assert_eq!(model.screen, Screen::Loading);
        let effect = cmd.expect_one_effect();
        assert!(matches!(effect, Effect::Http(_)));
    }

    #[test]
    fn test_get_account_done_updates_model() {
        let app = AppTester::<EidolonsApp>::default();
        let mut model = Model::default();

        let body = GetAccountBody {
            id: "abc".into(),
            stripe_customer_id: None,
            created_at: "2026-01-01T00:00:00Z".into(),
        };

        let response = crux_http::testing::ResponseBuilder::ok()
            .body(body)
            .build();

        let cmd = app.update(
            Event::GetAccountDone(Ok(response)),
            &mut model,
        );

        assert_eq!(model.screen, Screen::Account);
        assert_eq!(model.account.as_ref().unwrap().id, "abc");
        let _ = cmd.expect_one_effect();
    }

    #[test]
    fn test_view_reflects_account() {
        let app = EidolonsApp;
        let model = Model {
            screen: Screen::Account,
            account: Some(GetAccountBody {
                id: "abc".into(),
                stripe_customer_id: Some("cus_123".into()),
                created_at: "2026-01-01T00:00:00Z".into(),
            }),
            ..Model::default()
        };

        let vm = app.view(&model);
        assert_eq!(vm.screen, Screen::Account);
        assert_eq!(vm.account_id.as_deref(), Some("abc"));
        assert_eq!(
            vm.account_stripe_customer_id.as_deref(),
            Some("cus_123")
        );
    }
}
