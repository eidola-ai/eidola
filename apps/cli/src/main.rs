mod config;
mod db;

use std::io::IsTerminal;
use std::time::{SystemTime, UNIX_EPOCH};

use anonymous_credit_tokens::{IssuanceResponse, Params, PreIssuance, PublicKey, scalar_to_credit};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use clap::{Parser, Subcommand};
use rand_core::OsRng;
use serde::Deserialize;
use uuid::Uuid;

use config::Config;

#[derive(Parser)]
#[command(name = "eidolons", about = "Eidolons CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Set the server base URL
    Configure {
        #[arg(long)]
        base_url: String,
    },
    /// Manage account
    Account {
        #[command(subcommand)]
        command: Option<AccountCommand>,
    },
    /// Manage local wallet
    Wallet {
        #[command(subcommand)]
        command: WalletCommand,
    },
}

#[derive(Subcommand)]
enum AccountCommand {
    /// Create a new account on the server
    Create,
    /// Remove stored account credentials
    Reset,
    /// Set existing account credentials
    Configure {
        #[arg(long)]
        id: String,
        #[arg(long)]
        secret: String,
    },
    /// List available prices
    Prices,
    /// Create a checkout session and open payment link
    Checkout {
        /// Stripe price ID
        price_id: String,
        /// Print URL instead of opening browser
        #[arg(long)]
        no_browser: bool,
    },
    /// Show credit balances
    Balances,
    /// Allocate credits into an anonymous credential
    Allocate {
        /// Number of credits to allocate
        credits: i64,
    },
}

#[derive(Subcommand)]
enum WalletCommand {
    /// Manage credentials
    Credentials {
        #[command(subcommand)]
        command: CredentialsCommand,
    },
}

#[derive(Subcommand)]
enum CredentialsCommand {
    /// List active credentials
    List,
}

#[derive(Deserialize)]
struct CreateAccountResponse {
    account_id: Uuid,
    secret: String,
    created_at: String,
}

#[derive(Deserialize)]
struct GetAccountResponse {
    id: Uuid,
    stripe_customer_id: Option<String>,
    created_at: String,
}

#[derive(Deserialize)]
struct ListPricesResponse {
    data: Vec<PriceResponse>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct PriceResponse {
    id: String,
    product_name: String,
    product_description: Option<String>,
    unit_amount: Option<i64>,
    currency: String,
    #[serde(rename = "type")]
    price_type: String,
    recurring: Option<RecurringResponse>,
    credits: i64,
}

#[derive(Deserialize)]
struct RecurringResponse {
    interval: String,
    interval_count: i64,
}

#[derive(Deserialize)]
struct CheckoutUrlResponse {
    checkout_url: String,
}

#[derive(Deserialize)]
struct BalancesResponse {
    available: i64,
    pools: Vec<BalancePool>,
}

#[derive(Deserialize)]
struct BalancePool {
    amount: i64,
    source: String,
    expires_at: Option<String>,
}

#[derive(Deserialize)]
struct ListKeysResponse {
    data: Vec<IssuerKeyResponse>,
}

#[derive(Deserialize)]
struct IssuerKeyResponse {
    epoch: String,
    public_key: String,
    domain_separator: String,
    #[allow(dead_code)]
    valid_from: String,
    valid_until: String,
    #[allow(dead_code)]
    accept_until: String,
}

#[derive(Deserialize)]
struct IssueCredentialsResponse {
    issuance_response: String,
    epoch: String,
    credits: i64,
    #[allow(dead_code)]
    ledger_entry_id: String,
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs();
    let days = (secs / 86400) as i64;
    let time_of_day = secs % 86400;
    let (hour, min, sec) = (time_of_day / 3600, (time_of_day % 3600) / 60, time_of_day % 60);
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

fn require_base_url(config: &Config) -> Result<&str, String> {
    config
        .base_url
        .as_deref()
        .ok_or_else(|| "base_url not configured. Run: eidolons configure --base_url=<url>".into())
}

fn require_credentials(config: &Config) -> Result<(&str, &str), String> {
    match (&config.account_id, &config.account_secret) {
        (Some(id), Some(secret)) => Ok((id, secret)),
        _ => Err("account not configured. Run: eidolons account create".into()),
    }
}

#[tokio::main]
async fn main() {
    rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider())
        .expect("failed to install rustls crypto provider");

    let cli = Cli::parse();

    if let Err(e) = run(cli).await {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        None => {
            let config = Config::load();
            println!("config path: {:?}", Config::path());
            println!(
                "base_url: {}",
                config.base_url.as_deref().unwrap_or("<not set>")
            );
            println!(
                "account_id: {}",
                config.account_id.as_deref().unwrap_or("<not set>")
            );
            println!(
                "account_secret: {}",
                if config.account_secret.is_some() {
                    "<set>"
                } else {
                    "<not set>"
                }
            );
            Ok(())
        }
        Some(Command::Configure { base_url }) => {
            let mut config = Config::load();
            config.base_url = Some(base_url.clone());
            config.save()?;
            println!("base_url set to {base_url}");
            Ok(())
        }
        Some(Command::Account { command }) => match command {
            None => cmd_account_show().await,
            Some(AccountCommand::Create) => cmd_account_create().await,
            Some(AccountCommand::Reset) => cmd_account_reset(),
            Some(AccountCommand::Configure { id, secret }) => {
                cmd_account_configure(&id, &secret)
            }
            Some(AccountCommand::Prices) => cmd_account_prices().await,
            Some(AccountCommand::Checkout {
                price_id,
                no_browser,
            }) => cmd_account_checkout(&price_id, no_browser).await,
            Some(AccountCommand::Balances) => cmd_account_balances().await,
            Some(AccountCommand::Allocate { credits }) => cmd_account_allocate(credits).await,
        },
        Some(Command::Wallet { command }) => match command {
            WalletCommand::Credentials { command } => match command {
                CredentialsCommand::List => cmd_wallet_credentials_list().await,
            },
        },
    }
}

async fn cmd_account_show() -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let (id, secret) = require_credentials(&config)?;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/v1/account"))
        .basic_auth(id, Some(secret))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let account: GetAccountResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;

    println!("id: {}", account.id);
    if let Some(customer_id) = &account.stripe_customer_id {
        println!("stripe_customer_id: {customer_id}");
    }
    println!("created_at: {}", account.created_at);
    Ok(())
}

async fn cmd_account_create() -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;

    if config.account_id.is_some() || config.account_secret.is_some() {
        return Err(
            "account credentials already configured. Run 'eidolons account reset' first.".into(),
        );
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/account"))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let created: CreateAccountResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;

    let mut config = Config::load();
    config.account_id = Some(created.account_id.to_string());
    config.account_secret = Some(created.secret);
    config.save()?;

    println!("account created");
    println!("id: {}", created.account_id);
    println!("created_at: {}", created.created_at);
    Ok(())
}

fn cmd_account_reset() -> Result<(), String> {
    let mut config = Config::load();
    config.account_id = None;
    config.account_secret = None;
    config.save()?;
    println!("account credentials removed");
    Ok(())
}

fn cmd_account_configure(id: &str, secret: &str) -> Result<(), String> {
    let config = Config::load();

    if config.account_id.is_some() || config.account_secret.is_some() {
        return Err(
            "account credentials already configured. Run 'eidolons account reset' first.".into(),
        );
    }

    let mut config = config;
    config.account_id = Some(id.to_string());
    config.account_secret = Some(secret.to_string());
    config.save()?;
    println!("account configured");
    Ok(())
}

async fn cmd_account_prices() -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/v1/prices"))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let prices: ListPricesResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;

    if prices.data.is_empty() {
        println!("no prices available");
        return Ok(());
    }

    for p in &prices.data {
        let amount = p
            .unit_amount
            .map(|a| format!("{}.{:02} {}", a / 100, a % 100, p.currency.to_uppercase()))
            .unwrap_or_else(|| "free".to_string());

        let recurrence = p
            .recurring
            .as_ref()
            .map(|r| {
                if r.interval_count == 1 {
                    format!("/{}", r.interval)
                } else {
                    format!("/{}x{}", r.interval_count, r.interval)
                }
            })
            .unwrap_or_default();

        println!(
            "{}: {} ({}{}, {} credits)",
            p.id, p.product_name, amount, recurrence, p.credits
        );
        if let Some(desc) = &p.product_description {
            println!("  {desc}");
        }
    }
    Ok(())
}

async fn cmd_account_checkout(price_id: &str, no_browser: bool) -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let (id, secret) = require_credentials(&config)?;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/v1/account/checkout"))
        .basic_auth(id, Some(secret))
        .json(&serde_json::json!({ "price_id": price_id }))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let checkout: CheckoutUrlResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;

    let should_open = !no_browser && std::io::stdout().is_terminal();

    if should_open {
        println!("{}", checkout.checkout_url);
        open::that(&checkout.checkout_url)
            .map_err(|e| format!("failed to open browser: {e}"))?;
    } else {
        println!("{}", checkout.checkout_url);
    }
    Ok(())
}

async fn cmd_account_balances() -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let (id, secret) = require_credentials(&config)?;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{base_url}/v1/account/balances"))
        .basic_auth(id, Some(secret))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let balances: BalancesResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;

    println!("available: {}", balances.available);
    for pool in &balances.pools {
        let expires = pool
            .expires_at
            .as_deref()
            .map(|e| format!(", expires {e}"))
            .unwrap_or_default();
        println!("  {} ({}{})", pool.amount, pool.source, expires);
    }
    Ok(())
}

async fn cmd_account_allocate(credits: i64) -> Result<(), String> {
    if credits <= 0 {
        return Err("credits must be greater than 0".into());
    }

    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let (account_id, secret) = require_credentials(&config)?;
    let client = reqwest::Client::new();

    // 1. Fetch issuer keys from server
    let resp = client
        .get(format!("{base_url}/v1/keys"))
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let keys: ListKeysResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse keys response: {e}"))?;

    let key = keys
        .data
        .first()
        .ok_or("no issuer keys available from server")?;

    // Decode the public key
    let public_key_cbor = URL_SAFE_NO_PAD
        .decode(&key.public_key)
        .map_err(|e| format!("invalid base64 public key: {e}"))?;

    let public_key = PublicKey::from_cbor(&public_key_cbor)
        .map_err(|e| format!("invalid public key CBOR: {e}"))?;

    // Reconstruct params from epoch
    let params = Params::new("eidolons", "inference", "production", &key.epoch);

    // 2. Open database
    let database = db::open().await?;
    let conn = database
        .connect()
        .map_err(|e| format!("failed to connect: {e}"))?;

    // 3. Store issuer key locally
    let domain_separator = &key.domain_separator;
    let params_hash = blake3::hash(domain_separator.as_bytes()).to_hex().to_string();
    let now = now_iso();

    db::upsert_issuer_key(
        &conn,
        &key.epoch,
        &params_hash,
        &public_key_cbor,
        domain_separator.as_bytes(),
        &key.valid_until,
        &now,
    )
    .await?;

    // 4. Create PreIssuance and store in pre_credential (crash-safe checkpoint)
    let pre_issuance = PreIssuance::random(OsRng);
    let pre_issuance_cbor = pre_issuance
        .to_cbor()
        .map_err(|e| format!("failed to encode pre_issuance: {e}"))?;

    let pre_credential_id = Uuid::now_v7().to_string();

    db::insert_pre_credential_issuance(
        &conn,
        &pre_credential_id,
        &key.epoch,
        &pre_issuance_cbor,
        credits,
        &now,
    )
    .await?;

    // 5. Generate issuance request and send to server
    let issuance_request = pre_issuance.request(&params, OsRng);
    let request_cbor = issuance_request
        .to_cbor()
        .map_err(|e| format!("failed to encode issuance request: {e}"))?;

    let resp = client
        .post(format!("{base_url}/v1/account/credentials"))
        .basic_auth(account_id, Some(secret))
        .json(&serde_json::json!({
            "issuance_request": URL_SAFE_NO_PAD.encode(&request_cbor),
            "credits": credits,
        }))
        .send()
        .await
        .map_err(|e| format!("credential issuance request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let issued: IssueCredentialsResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse issuance response: {e}"))?;

    // 6. Decode response and construct CreditToken
    let response_cbor = URL_SAFE_NO_PAD
        .decode(&issued.issuance_response)
        .map_err(|e| format!("invalid base64 issuance response: {e}"))?;

    let issuance_response = IssuanceResponse::from_cbor(&response_cbor)
        .map_err(|e| format!("invalid issuance response CBOR: {e}"))?;

    let credit_token = pre_issuance
        .to_credit_token::<128>(&params, &public_key, &issuance_request, &issuance_response)
        .map_err(|e| format!("failed to construct credit token: {e}"))?;

    // 7. Store the credential
    let token_cbor = credit_token
        .to_cbor()
        .map_err(|e| format!("failed to encode credit token: {e}"))?;

    let nonce_hex = hex_encode(&credit_token.nullifier().to_bytes());
    let token_credits =
        scalar_to_credit::<128>(&credit_token.credits()).map_err(|e| format!("invalid credit amount in token: {e}"))?;

    db::insert_credential(
        &conn,
        &nonce_hex,
        &pre_credential_id,
        &issued.epoch,
        &token_cbor,
        token_credits as i64,
        0,
        &now,
    )
    .await?;

    println!("credential allocated: {nonce_hex}");
    println!("credits: {}", issued.credits);
    println!("epoch: {}", issued.epoch);
    Ok(())
}

async fn cmd_wallet_credentials_list() -> Result<(), String> {
    let database = db::open().await?;
    let conn = database
        .connect()
        .map_err(|e| format!("failed to connect: {e}"))?;

    let credentials = db::list_active_credentials(&conn).await?;

    if credentials.is_empty() {
        println!("no active credentials");
        return Ok(());
    }

    for c in &credentials {
        println!(
            "{}: {} credits (gen {})",
            c.nonce, c.credits, c.generation
        );
    }
    Ok(())
}
