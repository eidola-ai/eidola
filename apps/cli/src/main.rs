use std::io::{IsTerminal, Write};

use clap::{Parser, Subcommand};
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatStreamEvent, config};

#[derive(Parser)]
#[command(name = "eidola", about = "Eidola CLI")]
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
        /// URL for attestation verification (defaults to Tinfoil ATC)
        #[arg(long)]
        attestation_url: Option<String>,
        /// Path to PEM-encoded SEV-SNP ARK (Root CA) certificate
        #[arg(long)]
        hardware_root_ca: Option<String>,
        /// Path to PEM-encoded SEV-SNP ASK (Intermediate CA) certificate
        #[arg(long)]
        hardware_intermediate_ca: Option<String>,
        /// Add a trusted enclave release: `<snp>:<rtmr1>:<rtmr2>`
        #[arg(long)]
        trust_measurement: Option<String>,
        /// Remove a trusted enclave release by SNP measurement
        #[arg(long)]
        untrust_measurement: Option<String>,
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
        /// Model to use (defaults to first available)
        #[arg(long, short)]
        model: Option<String>,
        /// Continue an existing conversation by space ID
        #[arg(long, short)]
        space: Option<String>,
    },
    /// Manage conversation spaces
    Spaces {
        #[command(subcommand)]
        command: SpacesCommand,
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
    /// Recover stuck (in-flight) credentials
    Recover,
}

#[derive(Subcommand)]
enum SpacesCommand {
    /// List active conversation spaces
    List,
    /// Archive a conversation space
    Archive {
        /// Space ID to archive
        id: String,
    },
}

fn build_core() -> AppCore {
    let config_dir = config::default_config_path()
        .and_then(|p| p.parent().map(|d| d.to_string_lossy().into_owned()))
        .expect("could not determine config directory");
    let data_dir = config::default_data_dir()
        .map(|d| d.to_string_lossy().into_owned())
        .expect("could not determine data directory");
    AppCore::new(config_dir, data_dir)
}

fn main() {
    // Build the core (and its tokio runtime) outside any async context so it
    // can be dropped cleanly when main returns.
    let core = build_core();
    let cli = Cli::parse();

    // Use the core's own runtime to drive the CLI commands.
    let result = core.runtime().block_on(run(&core, cli));

    // Drop core before exiting so its runtime shuts down outside async context.
    drop(core);

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

async fn run(core: &AppCore, cli: Cli) -> Result<(), AppError> {
    match cli.command {
        None => {
            let state = core.config_state();
            println!("config path: {:?}", config::default_config_path());
            println!("base_url: {}", state.base_url);
            println!(
                "account_id: {}",
                if state.has_account {
                    "<set>"
                } else {
                    "<not set>"
                }
            );
            println!(
                "account_secret: {}",
                if state.has_account_secret {
                    "<set>"
                } else {
                    "<not set>"
                }
            );
            println!("trusted_measurements:");
            for m in &state.trusted_measurements {
                println!("  - snp = {}", m.snp);
                println!("    tdx.rtmr1 = {}", m.tdx_rtmr1);
                println!("    tdx.rtmr2 = {}", m.tdx_rtmr2);
            }
            println!(
                "hardware_root_ca: {}",
                if state.has_hardware_root_ca {
                    "<set>"
                } else {
                    "<not set>"
                }
            );
            println!(
                "hardware_intermediate_ca: {}",
                if state.has_hardware_intermediate_ca {
                    "<set>"
                } else {
                    "<not set>"
                }
            );
            println!(
                "attestation_url: {}",
                state.attestation_url.as_deref().unwrap_or("<default ATC>")
            );
            Ok(())
        }
        Some(Command::Configure {
            base_url,
            attestation_url,
            hardware_root_ca,
            hardware_intermediate_ca,
            trust_measurement,
            untrust_measurement,
        }) => {
            if base_url.is_none()
                && attestation_url.is_none()
                && hardware_root_ca.is_none()
                && hardware_intermediate_ca.is_none()
                && trust_measurement.is_none()
                && untrust_measurement.is_none()
            {
                return Err(AppError::Config {
                    message: "specify at least one option (see --help)".into(),
                });
            }
            if let Some(url) = base_url {
                core.set_base_url(url.clone())?;
                println!("base_url set to {url}");
            }
            if let Some(url) = attestation_url {
                core.set_attestation_url(url.clone())?;
                println!("attestation_url set to {url}");
            }
            if let Some(path) = hardware_root_ca {
                let pem = std::fs::read_to_string(&path).map_err(|e| AppError::Config {
                    message: format!("failed to read {path}: {e}"),
                })?;
                core.set_hardware_root_ca(pem)?;
                println!("hardware_root_ca set from {path}");
            }
            if let Some(path) = hardware_intermediate_ca {
                let pem = std::fs::read_to_string(&path).map_err(|e| AppError::Config {
                    message: format!("failed to read {path}: {e}"),
                })?;
                core.set_hardware_intermediate_ca(pem)?;
                println!("hardware_intermediate_ca set from {path}");
            }
            if let Some(spec) = trust_measurement {
                let m = config::parse_trust_measurement(&spec)?;
                let added = core.trust_measurement(
                    m.snp_measurement.clone(),
                    m.tdx_measurement.rtmr1.clone(),
                    m.tdx_measurement.rtmr2.clone(),
                )?;
                if added {
                    println!(
                        "added trusted measurement: snp={}, tdx.rtmr1={}, tdx.rtmr2={}",
                        m.snp_measurement, m.tdx_measurement.rtmr1, m.tdx_measurement.rtmr2,
                    );
                } else {
                    println!("measurement already trusted (snp={})", m.snp_measurement);
                }
            }
            if let Some(spec) = untrust_measurement {
                let key = config::parse_untrust_key(&spec)?;
                let removed = core.untrust_measurement(key.clone())?;
                if removed {
                    println!("removed trusted measurement (snp={key})");
                } else {
                    println!("measurement not found (snp={key})");
                }
            }
            Ok(())
        }
        Some(Command::Account { command }) => match command {
            None => {
                let info = core.account_show().await?;
                println!("id: {}", info.id);
                if let Some(customer_id) = &info.stripe_customer_id {
                    println!("stripe_customer_id: {customer_id}");
                }
                println!("created_at: {}", info.created_at);
                Ok(())
            }
            Some(AccountCommand::Create) => {
                let result = core.account_create().await?;
                println!("account created");
                println!("id: {}", result.id);
                println!("created_at: {}", result.created_at);
                Ok(())
            }
            Some(AccountCommand::Reset) => {
                core.reset_account()?;
                println!("account credentials removed");
                Ok(())
            }
            Some(AccountCommand::Configure { id, secret }) => {
                core.set_account_credentials(id, secret)?;
                println!("account configured");
                Ok(())
            }
            Some(AccountCommand::Prices) => {
                let prices = core.account_prices().await?;
                if prices.is_empty() {
                    println!("no prices available");
                    return Ok(());
                }
                for p in &prices {
                    println!(
                        "{}: {} ({}{}, {} credits)",
                        p.id, p.product_name, p.amount_display, p.recurrence, p.credits
                    );
                    if let Some(desc) = &p.product_description {
                        println!("  {desc}");
                    }
                }
                Ok(())
            }
            Some(AccountCommand::Checkout {
                price_id,
                no_browser,
            }) => {
                let url = core.account_checkout(price_id).await?;
                let should_open = !no_browser && std::io::stdout().is_terminal();
                println!("{url}");
                if should_open {
                    let _ = open::that(&url);
                }
                Ok(())
            }
            Some(AccountCommand::Balances) => {
                let balances = core.account_balances().await?;
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
            Some(AccountCommand::Allocate { credits }) => {
                let result = core.account_allocate(credits).await?;
                println!("credential allocated: {}", result.nonce);
                println!("credits: {}", result.credits);
                println!("issuer_key_id: {}", result.issuer_key_id);
                Ok(())
            }
        },
        Some(Command::Wallet { command }) => match command {
            WalletCommand::Credentials { command } => match command {
                CredentialsCommand::List => {
                    let spending = core.wallet_spending_credentials().await?;
                    if !spending.is_empty() {
                        println!("in-flight credentials:");
                        for c in &spending {
                            println!(
                                "  {}: {} credits, {} charged",
                                c.nonce, c.credits, c.spend_amount
                            );
                        }
                        println!();
                    }
                    let credentials = core.wallet_credentials().await?;
                    if credentials.is_empty() && spending.is_empty() {
                        println!("no credentials");
                        return Ok(());
                    }
                    if !credentials.is_empty() {
                        println!("active credentials:");
                        for c in &credentials {
                            println!(
                                "  {}: {} credits (gen {})",
                                c.nonce, c.credits, c.generation
                            );
                        }
                    }
                    Ok(())
                }
                CredentialsCommand::Recover => {
                    let spending = core.wallet_spending_credentials().await?;
                    if spending.is_empty() {
                        println!("no in-flight credentials");
                        return Ok(());
                    }
                    println!("attempting to recover {} credential(s)...", spending.len());
                    let recovered = core.recover_spending_credentials().await?;
                    if recovered.is_empty() {
                        println!("no credentials could be recovered");
                    } else {
                        println!("recovered {} credential(s):", recovered.len());
                        for nonce in &recovered {
                            println!("  {nonce}");
                        }
                    }
                    Ok(())
                }
            },
        },
        Some(Command::Chat {
            prompt,
            model,
            space,
        }) => {
            let model = model.unwrap_or_else(|| "gemma4-31b".to_string());

            // Stream chunks straight to stdout. Reasoning goes to stderr
            // (dim, prefixed with "thinking: ") so a piped stdout still
            // captures only the final answer text.
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
            let chat_fut = core.chat_stream(prompt, model.clone(), space, tx);

            // Pump events while chat_fut runs. We `tokio::join!` the two
            // halves so events drain in real time rather than only after
            // the request future awaits a yield point.
            let printer = async move {
                let mut stdout = std::io::stdout().lock();
                let mut stderr = std::io::stderr().lock();
                let stderr_is_tty = std::io::stderr().is_terminal();
                let mut in_reasoning = false;
                while let Some(event) = rx.recv().await {
                    match event {
                        ChatStreamEvent::ContentDelta(text) => {
                            if in_reasoning {
                                let _ = writeln!(stderr);
                                if stderr_is_tty {
                                    let _ = write!(stderr, "\x1b[0m");
                                }
                                in_reasoning = false;
                            }
                            let _ = stdout.write_all(text.as_bytes());
                            let _ = stdout.flush();
                        }
                        ChatStreamEvent::ReasoningDelta(text) => {
                            if !in_reasoning {
                                if stderr_is_tty {
                                    let _ = write!(stderr, "\x1b[2mthinking: ");
                                } else {
                                    let _ = write!(stderr, "thinking: ");
                                }
                                in_reasoning = true;
                            }
                            let _ = stderr.write_all(text.as_bytes());
                            let _ = stderr.flush();
                        }
                    }
                }
                if in_reasoning && stderr_is_tty {
                    let _ = write!(stderr, "\x1b[0m");
                }
                let _ = writeln!(stdout);
            };

            let (result, ()) = tokio::join!(chat_fut, printer);
            let result = result?;
            eprintln!(
                "---\nspace: {}  model: {}  tokens: {}/{}",
                result.space_id,
                result.model,
                result.input_tokens.unwrap_or(0),
                result.output_tokens.unwrap_or(0),
            );
            Ok(())
        }
        Some(Command::Spaces { command }) => match command {
            SpacesCommand::List => {
                let spaces = core.list_spaces().await?;
                if spaces.is_empty() {
                    println!("no active spaces");
                    return Ok(());
                }
                for s in &spaces {
                    let title = s.title.as_deref().unwrap_or("<untitled>");
                    println!("{}: {}", s.id, title);
                }
                Ok(())
            }
            SpacesCommand::Archive { id } => {
                let archived = core.archive_space(id.clone()).await?;
                if archived {
                    println!("archived space {id}");
                } else {
                    println!("space not found or already archived: {id}");
                }
                Ok(())
            }
        },
    }
}
