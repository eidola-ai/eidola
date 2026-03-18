mod attestation;
mod config;
mod db;

use std::io::IsTerminal;
use std::time::{SystemTime, UNIX_EPOCH};

use anonymous_credit_tokens::{
    CreditToken, IssuanceResponse, Params, PreIssuance, PublicKey, Refund, credit_to_scalar,
    scalar_to_credit,
};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use clap::{Parser, Subcommand};
use rand_core::OsRng;
use serde::Deserialize;
use sha2::{Digest, Sha256};
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
        base_url: Option<String>,
        /// Path to a PEM file containing the CA certificate to trust
        #[arg(long)]
        ca_cert: Option<String>,
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
    /// Send a chat message
    Chat {
        /// The prompt to send
        prompt: String,
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
    id: String,
    public_key: String,
    domain_separator: String,
    #[allow(dead_code)]
    issue_from: String,
    issue_until: String,
    #[allow(dead_code)]
    accept_until: String,
}

#[derive(Deserialize)]
struct IssueCredentialsResponse {
    issuance_response: String,
    issuer_key_id: String,
    credits: i64,
    #[allow(dead_code)]
    ledger_entry_id: String,
}

#[derive(Deserialize)]
struct ModelPricingInfo {
    per_prompt_token: ScaledPriceInfo,
    per_completion_token: ScaledPriceInfo,
}

#[derive(Deserialize)]
struct ScaledPriceInfo {
    value: u64,
    scale_factor: u64,
}

#[derive(Deserialize)]
struct ModelsResponseInfo {
    data: Vec<ModelListEntry>,
}

#[derive(Deserialize)]
struct ModelListEntry {
    id: String,
    context_length: u64,
    pricing: ModelPricingInfo,
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd-length hex string".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("invalid hex: {e}")))
        .collect()
}

/// ACT token type (draft-schlesinger-privacypass-act-01).
const ACT_TOKEN_TYPE: u16 = 0xE5AD;
const ISSUER_NAME: &str = "eidolons";
const ORIGIN_INFO: &str = "inference";

/// Serialize TokenChallenge per draft-schlesinger-privacypass-act-01 Section 7.
fn serialize_token_challenge() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
    buf.extend_from_slice(&(ISSUER_NAME.len() as u16).to_be_bytes());
    buf.extend_from_slice(ISSUER_NAME.as_bytes());
    buf.push(0); // redemption_context (empty)
    buf.extend_from_slice(&(ORIGIN_INFO.len() as u16).to_be_bytes());
    buf.extend_from_slice(ORIGIN_INFO.as_bytes());
    buf.push(0); // credential_context (empty)
    buf
}

fn compute_challenge_digest() -> [u8; 32] {
    Sha256::digest(serialize_token_challenge()).into()
}

fn now_iso() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before epoch")
        .as_secs();
    let days = (secs / 86400) as i64;
    let time_of_day = secs % 86400;
    let (hour, min, sec) = (
        time_of_day / 3600,
        (time_of_day % 3600) / 60,
        time_of_day % 60,
    );
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

/// Build a reqwest client. When `ca_cert` is configured, only that CA is
/// trusted (no public WebPKI roots) and the RA-TLS attestation verifier is
/// always active — the server's compose_hash must appear in
/// `trusted_compose_hashes` or the connection is refused.
fn build_client(config: &Config) -> Result<reqwest::Client, String> {
    match config.ca_cert.as_deref() {
        Some(pem) => {
            // Build a WebPKI verifier with the pinned CA, then wrap it in
            // our attestation verifier.
            let mut ca_store = rustls::RootCertStore::empty();
            let certs = rustls_pemfile::certs(&mut pem.as_bytes())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("invalid ca_cert PEM: {e}"))?;
            for cert in certs {
                ca_store
                    .add(cert)
                    .map_err(|e| format!("failed to add CA cert: {e}"))?;
            }

            let webpki_verifier =
                rustls::client::WebPkiServerVerifier::builder(std::sync::Arc::new(ca_store))
                    .build()
                    .map_err(|e| format!("failed to build WebPKI verifier: {e}"))?;

            let attest_verifier = std::sync::Arc::new(attestation::AttestationVerifier::new(
                webpki_verifier,
                config.trusted_compose_hashes.clone(),
            ));

            let tls_config = rustls::ClientConfig::builder()
                .dangerous()
                .with_custom_certificate_verifier(attest_verifier)
                .with_no_client_auth();

            reqwest::Client::builder()
                .use_preconfigured_tls(tls_config)
                .build()
                .map_err(|e| format!("failed to build HTTP client: {e}"))
        }
        None => Ok(reqwest::Client::new()),
    }
}

/// Map a reqwest send error to a human-readable message.
/// Detects RA-TLS attestation failures buried inside the TLS layer and
/// surfaces them explicitly instead of printing a generic "error sending
/// request" wrapper.
fn describe_request_error(e: reqwest::Error) -> String {
    // Walk the full error chain so we can detect messages from inner layers
    // (rustls → our AttestationVerifier) that reqwest's Display may truncate.
    let mut chain = format!("{e}");
    {
        let mut source = std::error::Error::source(&e);
        while let Some(err) = source {
            use std::fmt::Write;
            let _ = write!(chain, ": {err}");
            source = err.source();
        }
    }
    if chain.contains("compose_hash") && chain.contains("not in the trusted set") {
        return format!(
            "attestation failed: the server's compose_hash is not in your \
             trusted_compose_hashes list.\n\
             The running server version is not trusted by this client.\n\
             Update trusted_compose_hashes in your config, or verify you are \
             connecting to the correct server.\n\
             (inner: {chain})"
        );
    }
    if chain.contains("missing PHALA_RATLS_ATTESTATION") {
        return format!(
            "attestation failed: the server's TLS certificate does not contain \
             an RA-TLS attestation extension.\n\
             The server may not be running inside a Confidential VM, or the \
             dstack simulator may not support attestation certificates.\n\
             (inner: {chain})"
        );
    }
    if chain.contains("attestation") {
        return format!(
            "attestation failed: could not verify the server's RA-TLS certificate.\n\
             (inner: {chain})"
        );
    }
    format!("request failed: {chain}")
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
            println!(
                "ca_cert: {}",
                if config.ca_cert.is_some() {
                    "<set>"
                } else {
                    "<not set>"
                }
            );
            if config.trusted_compose_hashes.is_empty() {
                println!("trusted_compose_hashes: <none> (all connections will be refused)");
            } else {
                println!(
                    "trusted_compose_hashes: {}",
                    config.trusted_compose_hashes.join(", ")
                );
            }
            Ok(())
        }
        Some(Command::Configure { base_url, ca_cert }) => {
            if base_url.is_none() && ca_cert.is_none() {
                return Err("specify at least one of --base_url or --ca_cert".into());
            }
            let mut config = Config::load();
            if let Some(url) = &base_url {
                config.base_url = Some(url.clone());
                println!("base_url set to {url}");
            }
            if let Some(path) = &ca_cert {
                let pem = std::fs::read_to_string(path)
                    .map_err(|e| format!("failed to read {path}: {e}"))?;
                if !pem.contains("-----BEGIN CERTIFICATE-----") {
                    return Err(format!(
                        "{path} does not appear to contain a PEM certificate"
                    ));
                }
                config.ca_cert = Some(pem);
                println!("ca_cert set from {path}");
            }
            config.save()?;
            Ok(())
        }
        Some(Command::Account { command }) => match command {
            None => cmd_account_show().await,
            Some(AccountCommand::Create) => cmd_account_create().await,
            Some(AccountCommand::Reset) => cmd_account_reset(),
            Some(AccountCommand::Configure { id, secret }) => cmd_account_configure(&id, &secret),
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
        Some(Command::Chat { prompt }) => cmd_chat(&prompt).await,
    }
}

async fn cmd_account_show() -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let (id, secret) = require_credentials(&config)?;

    let client = build_client(&config)?;
    let resp = client
        .get(format!("{base_url}/v1/account"))
        .basic_auth(id, Some(secret))
        .send()
        .await
        .map_err(describe_request_error)?;

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

    let client = build_client(&config)?;
    let resp = client
        .post(format!("{base_url}/v1/account"))
        .send()
        .await
        .map_err(describe_request_error)?;

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

    let client = build_client(&config)?;
    let resp = client
        .get(format!("{base_url}/v1/prices"))
        .send()
        .await
        .map_err(describe_request_error)?;

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

    let client = build_client(&config)?;
    let resp = client
        .post(format!("{base_url}/v1/account/checkout"))
        .basic_auth(id, Some(secret))
        .json(&serde_json::json!({ "price_id": price_id }))
        .send()
        .await
        .map_err(describe_request_error)?;

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
        open::that(&checkout.checkout_url).map_err(|e| format!("failed to open browser: {e}"))?;
    } else {
        println!("{}", checkout.checkout_url);
    }
    Ok(())
}

async fn cmd_account_balances() -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let (id, secret) = require_credentials(&config)?;

    let client = build_client(&config)?;
    let resp = client
        .get(format!("{base_url}/v1/account/balances"))
        .basic_auth(id, Some(secret))
        .send()
        .await
        .map_err(describe_request_error)?;

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
    let client = build_client(&config)?;

    // 1. Fetch issuer keys from server
    let resp = client
        .get(format!("{base_url}/v1/keys"))
        .send()
        .await
        .map_err(describe_request_error)?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("server returned {status}: {body}"));
    }

    let keys: ListKeysResponse = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse keys response: {e}"))?;

    // Filter keys to those matching our expected domain separator.
    let expected_ds = config.domain_separator();
    let key = keys
        .data
        .iter()
        .find(|k| k.domain_separator == expected_ds)
        .ok_or_else(|| {
            let server_ds: Vec<&str> = keys
                .data
                .iter()
                .map(|k| k.domain_separator.as_str())
                .collect();
            format!(
                "no issuer key matches expected domain separator \"{expected_ds}\"\n\
                 server advertised: {server_ds:?}\n\
                 hint: if the server has legitimately changed its domain separator, \
                 update domain_separator in your config"
            )
        })?;

    // Decode the public key
    let public_key_cbor = URL_SAFE_NO_PAD
        .decode(&key.public_key)
        .map_err(|e| format!("invalid base64 public key: {e}"))?;

    let public_key = PublicKey::from_cbor(&public_key_cbor)
        .map_err(|e| format!("invalid public key CBOR: {e}"))?;

    // Reconstruct params from the validated domain separator (ACT-v1:org:service:deployment:version).
    // We already verified key.domain_separator == expected_ds above.
    let ds_parts: Vec<&str> = expected_ds.split(':').collect();
    assert_eq!(
        ds_parts.len(),
        5,
        "compiled-in domain separator has wrong format"
    );
    let params = Params::new(ds_parts[1], ds_parts[2], ds_parts[3], ds_parts[4]);

    // 2. Open database
    let database = db::open().await?;
    let conn = database
        .connect()
        .map_err(|e| format!("failed to connect: {e}"))?;

    // 3. Store issuer key locally
    let domain_separator = &key.domain_separator;
    let params_hash = blake3::hash(domain_separator.as_bytes())
        .to_hex()
        .to_string();
    let now = now_iso();

    db::upsert_issuer_key(
        &conn,
        &key.id,
        &params_hash,
        &public_key_cbor,
        domain_separator.as_bytes(),
        &key.issue_until,
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
        &key.id,
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
        .map_err(describe_request_error)?;

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
    let token_credits = scalar_to_credit::<128>(&credit_token.credits())
        .map_err(|e| format!("invalid credit amount in token: {e}"))?;

    db::insert_credential(
        &conn,
        &nonce_hex,
        &pre_credential_id,
        &issued.issuer_key_id,
        &token_cbor,
        token_credits as i64,
        0,
        &now,
    )
    .await?;

    println!("credential allocated: {nonce_hex}");
    println!("credits: {}", issued.credits);
    println!("issuer_key_id: {}", issued.issuer_key_id);
    Ok(())
}

async fn cmd_chat(prompt: &str) -> Result<(), String> {
    let config = Config::load();
    let base_url = require_base_url(&config)?;
    let client = build_client(&config)?;
    let model_id = "phala/qwen3-vl-30b-a3b-instruct";

    // 1. Fetch model info for pricing
    let resp = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await
        .map_err(describe_request_error)?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("failed to fetch models: {status}: {body}"));
    }
    let models: ModelsResponseInfo = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse models response: {e}"))?;
    let model = models
        .data
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("model not found: {model_id}"))?;

    // 2. Estimate max_completion_tokens: cap at 4096
    let max_completion_tokens = (model.context_length).min(4096) as u32;

    // 3. Calculate worst-case cost (mirrors server logic):
    //    prompt_bytes * prompt_rate + max_completion_tokens * completion_rate
    let sf = model.pricing.per_prompt_token.scale_factor as u128;
    let prompt_bytes = prompt.len() as u128;
    let prompt_rate = model.pricing.per_prompt_token.value as u128;
    let prompt_credits = (prompt_bytes * prompt_rate).div_ceil(sf);
    let completion_rate = model.pricing.per_completion_token.value as u128;
    let completion_credits = (max_completion_tokens as u128 * completion_rate).div_ceil(sf);
    let charge_credits = prompt_credits + completion_credits;

    if charge_credits == 0 {
        return Err("computed charge is zero — model pricing may be missing".into());
    }

    // 4. Open DB and find a credential with enough credits
    let database = db::open().await?;
    let conn = database
        .connect()
        .map_err(|e| format!("failed to connect: {e}"))?;

    let cred = db::find_spendable_credential(&conn, charge_credits as i64)
        .await?
        .ok_or("no credential with sufficient credits found")?;

    // 5. Load credential and key data
    let credit_token = CreditToken::from_cbor(&cred.data)
        .map_err(|e| format!("failed to decode credential: {e}"))?;
    let public_key = PublicKey::from_cbor(&cred.public_key_data)
        .map_err(|e| format!("failed to decode public key: {e}"))?;

    let ds = config.domain_separator();
    let ds_parts: Vec<&str> = ds.split(':').collect();
    assert_eq!(ds_parts.len(), 5, "domain separator has wrong format");
    let params = Params::new(ds_parts[1], ds_parts[2], ds_parts[3], ds_parts[4]);

    // 6. Create spend proof
    let charge_scalar = credit_to_scalar::<128>(charge_credits)
        .map_err(|e| format!("invalid charge amount: {e:?}"))?;
    let (spend_proof, pre_refund) = credit_token
        .prove_spend::<128>(&params, charge_scalar, OsRng)
        .map_err(|e| format!("failed to create spend proof: {e:?}"))?;

    // 7. Checkpoint PreRefund in database (crash-safe)
    let pre_refund_cbor = pre_refund
        .to_cbor()
        .map_err(|e| format!("failed to encode pre_refund: {e}"))?;
    let spend_proof_cbor = spend_proof
        .to_cbor()
        .map_err(|e| format!("failed to encode spend proof: {e}"))?;
    let pre_cred_id = Uuid::now_v7().to_string();
    let now = now_iso();
    db::insert_pre_credential_refund(
        &conn,
        &pre_cred_id,
        &cred.nonce,
        &cred.issuer_key_id,
        &pre_refund_cbor,
        charge_credits as i64,
        &now,
    )
    .await?;

    // 8. Build ACT wire token: token_type || challenge_digest || issuer_key_id || spend_proof_cbor
    let issuer_key_hash = hex_decode(&cred.issuer_key_id)?;
    let challenge_digest = compute_challenge_digest();

    let mut token_bytes = Vec::new();
    token_bytes.extend_from_slice(&ACT_TOKEN_TYPE.to_be_bytes());
    token_bytes.extend_from_slice(&challenge_digest);
    token_bytes.extend_from_slice(&issuer_key_hash);
    token_bytes.extend_from_slice(&spend_proof_cbor);

    let token_b64 = URL_SAFE_NO_PAD.encode(&token_bytes);
    let auth_value = format!("PrivateToken token=\"{token_b64}\"");

    // 9. Send chat completion request
    let resp = client
        .post(format!("{base_url}/v1/chat/completions"))
        .header("Authorization", &auth_value)
        .json(&serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_completion_tokens,
        }))
        .send()
        .await
        .map_err(describe_request_error)?;

    let status = resp.status();
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("failed to parse response: {e}"))?;

    // 10. Extract and process refund if present
    if let Some(refund_obj) = body.get("refund") {
        let refund_b64 = refund_obj
            .get("refund")
            .and_then(|v| v.as_str())
            .ok_or("missing refund data in response")?;
        let refund_key_id = refund_obj
            .get("issuer_key_id")
            .and_then(|v| v.as_str())
            .ok_or("missing issuer_key_id in refund")?;

        let refund_cbor = URL_SAFE_NO_PAD
            .decode(refund_b64)
            .map_err(|e| format!("invalid refund base64: {e}"))?;
        let refund =
            Refund::from_cbor(&refund_cbor).map_err(|e| format!("invalid refund CBOR: {e}"))?;

        let new_token = pre_refund
            .to_credit_token::<128>(&params, &spend_proof, &refund, &public_key)
            .map_err(|e| format!("failed to construct refund credit token: {e:?}"))?;

        let new_token_cbor = new_token
            .to_cbor()
            .map_err(|e| format!("failed to encode new credit token: {e}"))?;
        let new_nonce = hex_encode(&new_token.nullifier().to_bytes());
        let new_credits = scalar_to_credit::<128>(&new_token.credits())
            .map_err(|e| format!("invalid credit amount in refund token: {e}"))?;

        db::insert_credential(
            &conn,
            &new_nonce,
            &pre_cred_id,
            refund_key_id,
            &new_token_cbor,
            new_credits as i64,
            cred.generation + 1,
            &now,
        )
        .await?;
    }

    if !status.is_success() {
        let error_msg = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("server returned {status}: {error_msg}"));
    }

    // 11. Print response content
    if let Some(choices) = body.get("choices").and_then(|c| c.as_array())
        && let Some(content) = choices
            .first()
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
    {
        println!("{content}");
    }

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
        println!("{}: {} credits (gen {})", c.nonce, c.credits, c.generation);
    }
    Ok(())
}
