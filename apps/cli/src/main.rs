mod config;

use clap::{Parser, Subcommand};
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
