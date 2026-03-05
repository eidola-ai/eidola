mod bridge;
mod config;
mod http;
mod render;

use bincode::Options;
use clap::{Parser, Subcommand};
use crux_core::bridge::Request;
use eidolons_shared::{EffectFfi, Event};

use bridge::{bincode_options, get_view, send_event, send_response};
use config::Config;
use http::execute_http;
use render::{render_view, ShellContext};

// ── Clap CLI ────────────────────────────────────────────────────────────────

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
}

// ── Effect processing loop ──────────────────────────────────────────────────

async fn process_effects(
    requests: Vec<Request<EffectFfi>>,
    ctx: &ShellContext,
) -> Result<(), String> {
    for request in requests {
        match request.effect {
            EffectFfi::Render(_) => {
                let vm = get_view();
                render_view(&vm, ctx)?;
            }
            EffectFfi::Http(http_request) => {
                let result = execute_http(http_request).await;

                let result_bytes = bincode_options()
                    .serialize(&result)
                    .expect("serialize HttpResult");

                let new_requests = send_response(request.id.0, &result_bytes);
                Box::pin(process_effects(new_requests, ctx)).await?;
            }
        }
    }
    Ok(())
}

// ── Main ────────────────────────────────────────────────────────────────────

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
        Some(Command::Account { command }) => {
            let config = Config::load();
            let ctx = ShellContext { no_browser: false };

            match command {
                None => {
                    send_init(&config);
                    let requests = send_event(&Event::GetAccount);
                    process_effects(requests, &ctx).await
                }
                Some(AccountCommand::Create) => {
                    if config.account_id.is_some() || config.account_secret.is_some() {
                        return Err(
                            "account credentials already configured. Run 'eidolons account reset' first.".into(),
                        );
                    }
                    send_init(&config);
                    let requests = send_event(&Event::CreateAccount);
                    process_effects(requests, &ctx).await
                }
                Some(AccountCommand::Reset) => {
                    let mut config = config;
                    config.account_id = None;
                    config.account_secret = None;
                    config.save()?;
                    println!("account credentials removed");
                    Ok(())
                }
                Some(AccountCommand::Configure { id, secret }) => {
                    if config.account_id.is_some() || config.account_secret.is_some() {
                        return Err(
                            "account credentials already configured. Run 'eidolons account reset' first.".into(),
                        );
                    }
                    let mut config = config;
                    config.account_id = Some(id);
                    config.account_secret = Some(secret);
                    config.save()?;
                    println!("account configured");
                    Ok(())
                }
                Some(AccountCommand::Prices) => {
                    send_init(&config);
                    let requests = send_event(&Event::GetPrices);
                    process_effects(requests, &ctx).await
                }
                Some(AccountCommand::Checkout {
                    price_id,
                    no_browser,
                }) => {
                    send_init(&config);
                    let ctx = ShellContext { no_browser };
                    let requests = send_event(&Event::Checkout { price_id });
                    process_effects(requests, &ctx).await
                }
                Some(AccountCommand::Balances) => {
                    send_init(&config);
                    let requests = send_event(&Event::GetBalances);
                    process_effects(requests, &ctx).await
                }
            }
        }
    }
}

fn send_init(config: &Config) {
    let base_url = config.base_url.clone().unwrap_or_default();
    let _ = send_event(&Event::Init {
        base_url,
        account_id: config.account_id.clone(),
        account_secret: config.account_secret.clone(),
    });
}
