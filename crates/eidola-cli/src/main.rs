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
        /// Model to use (defaults to the configured `default_model`)
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
    /// Manage local inference models (llama.cpp)
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Manage inference backends (where an ask can be routed)
    Backend {
        #[command(subcommand)]
        command: BackendCommand,
    },
    /// Check for and verify newer releases
    Update {
        #[command(subcommand)]
        command: UpdateCommand,
    },
}

#[derive(Subcommand)]
enum UpdateCommand {
    /// Check whether a verified newer release exists (the release marked
    /// `latest`, verified against the embedded trust root) and print the
    /// outcome. States mirror the GUI's Updates window: up to date /
    /// update available / unverifiable (security warning, exit code 2) /
    /// claims changed (side-by-side, exit code 3) / check failed.
    Check,
    /// Run the full self-update verification pipeline (CI Sigstore +
    /// human cosign+Rekor + template equality) against `release.json`.
    /// Prints the verified attestation prose. Does not install — that's
    /// a future `--install` flag once step 5 lands.
    Verify {
        /// Pin the installed version explicitly (default: this binary's
        /// compile-time version). Useful for testing the continuity gate.
        #[arg(long)]
        installed_version: Option<String>,
        /// Pin the installed git commit (default: none, meaning "first
        /// install — bypass continuity"). Useful for testing.
        #[arg(long)]
        installed_git_commit: Option<String>,
        /// (dev only) read release bytes from a local directory instead
        /// of GitHub. The directory must contain `release.json` plus each
        /// referenced asset by URL basename (e.g. `artifact-manifest.json`,
        /// `artifact-manifest.json.sigstore`, `attestation-<id>.json`, ...).
        /// The verifier runs the same crypto checks — only the byte source
        /// changes — so this is the tight dev loop for iterating on the
        /// verifier itself.
        #[arg(long, value_name = "PATH")]
        fixtures_dir: Option<String>,
        /// Print one diagnostic line per pipeline stage to stderr.
        #[arg(long, short = 'v')]
        verbose: bool,
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
enum ModelCommand {
    /// List local models (downloaded / loading / loaded) and the curated
    /// Gemma 4 catalog
    List,
    /// Download a model: a catalog id (see `model list`) or a `.gguf` URL
    /// (direct or a Hugging Face file page). Waits with a progress line.
    Download {
        /// Catalog id (e.g. `gemma-4-e2b`) or URL
        source: String,
    },
    /// Delete a downloaded model
    Delete {
        /// Model id (`local/<slug>`) or bare slug
        id: String,
    },
    /// Load a model: start its llama-server engine and wait until ready
    Load {
        /// Model id (`local/<slug>`) or bare slug
        id: String,
    },
    /// Unload a model, terminating its engine
    Unload {
        /// Model id (`local/<slug>`) or bare slug
        id: String,
    },
}

#[derive(Subcommand)]
enum BackendCommand {
    /// List configured backends
    List,
    /// Add an external backend
    Add {
        #[command(subcommand)]
        kind: BackendAddCommand,
    },
    /// Enable a backend
    Enable {
        /// Backend id
        id: String,
    },
    /// Disable a backend (for `eidola`: run with no account, on-device only)
    Disable {
        /// Backend id
        id: String,
    },
    /// Remove an external backend (its forensic trail is preserved;
    /// re-adding the same id revives it)
    Remove {
        /// Backend id
        id: String,
    },
    /// List the models a backend offers (`model list` covers `local`)
    Models {
        /// Backend id
        id: String,
    },
}

#[derive(Subcommand)]
enum BackendAddCommand {
    /// Any OpenAI-compatible HTTP server (self-hosted vLLM/Ollama/llama.cpp,
    /// or a conventional provider you choose to trust)
    Openai {
        /// Backend id (lowercase letters, digits, hyphens) — models are then
        /// addressed as `<model>@<id>`
        id: String,
        /// Base URL (e.g. http://192.168.1.20:8000)
        #[arg(long)]
        url: String,
        /// API key, sent as a Bearer token
        #[arg(long)]
        api_key: Option<String>,
        /// Display name (defaults to the id)
        #[arg(long)]
        name: Option<String>,
        /// Pin the model list (comma-separated) instead of trusting
        /// GET /v1/models — not every "OpenAI-compatible" server offers it
        #[arg(long, value_delimiter = ',')]
        models: Option<Vec<String>>,
    },
    /// A llama.cpp install whose models you manage yourself: Eidola scans
    /// the directory and starts/stops llama-server engines on demand
    Llamacpp {
        /// Backend id (lowercase letters, digits, hyphens)
        id: String,
        /// Directory of .gguf files (never written to by Eidola)
        #[arg(long)]
        models_dir: String,
        /// Display name (defaults to the id)
        #[arg(long)]
        name: Option<String>,
    },
}

#[derive(Subcommand)]
enum SpacesCommand {
    /// List active conversation spaces
    List {
        /// Include archived spaces
        #[arg(long)]
        archived: bool,
    },
    /// Archive a conversation space
    Archive {
        /// Space ID to archive
        id: String,
    },
    /// Rename a conversation space
    Rename {
        /// Space ID to rename
        id: String,
        /// New title
        title: String,
    },
}

fn build_core() -> AppCore {
    let config_dir = config::default_config_path()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .expect("could not determine config directory");
    let data_dir = config::default_data_dir().expect("could not determine data directory");
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
        // The typed onboarding errors get actionable hints. (Chat
        // auto-provisions credentials from the account balance, so these
        // only fire when the account itself is missing or unfunded.) Look
        // through the `ChatFailed` wrapper that `chat`/`chat_stream` attach
        // once a space is persisted, so a wrapped `NoAccount` /
        // `InsufficientBalance` still routes to its hint.
        match e.root() {
            AppError::NoAccount => {
                eprintln!("hint: run `eidola account create` to create an anonymous account");
            }
            AppError::InsufficientBalance { .. } => {
                eprintln!(
                    "hint: run `eidola account prices`, then \
                     `eidola account checkout <price_id>` to add credit"
                );
            }
            _ => {}
        }
        std::process::exit(1);
    }
}

async fn run(core: &AppCore, cli: Cli) -> Result<(), AppError> {
    match cli.command {
        None => {
            let state = core.config_state();
            println!("config path: {:?}", config::default_config_path());
            println!("base_url: {}", state.base_url);
            println!("default_model: {}", state.default_model);
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
            // No --model flag → the user's configured default (the
            // `default_model` override, falling back to the embedded
            // default).
            let model = model.unwrap_or_else(|| core.config_state().default_model);

            // Engine-served models (the managed `local` store and llamacpp
            // backends) run on an engine owned by *this* process (a `model
            // load` in another CLI invocation died with it), so such a chat
            // auto-loads its engine for the duration of the run.
            let mref = eidola_app_core::parse_model_ref(&model);
            let engine_backed = mref.backend_id == eidola_app_core::LOCAL_BACKEND_ID
                || core.list_backends().await?.iter().any(|b| {
                    b.id == mref.backend_id && b.kind == eidola_app_core::BackendKind::LlamaCpp
                });
            if engine_backed {
                let state = core.local_models_state().await?;
                let in_backend: Vec<&eidola_app_core::LocalModelInfo> =
                    if mref.backend_id == eidola_app_core::LOCAL_BACKEND_ID {
                        state.models.iter().collect()
                    } else {
                        state
                            .external
                            .iter()
                            .find(|b| b.backend_id == mref.backend_id)
                            .map(|b| b.models.iter().collect())
                            .unwrap_or_default()
                    };
                let loaded = in_backend.iter().any(|m| {
                    m.id == model
                        && matches!(m.status, eidola_app_core::LocalModelStatus::Loaded { .. })
                });
                if !loaded {
                    eprintln!("loading {model}…");
                    core.load_local_model(model.clone()).await?;
                }
            }

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
            SpacesCommand::List { archived } => {
                let spaces = core.list_spaces(archived).await?;
                if spaces.is_empty() {
                    println!("no active spaces");
                    return Ok(());
                }
                for s in &spaces {
                    let title = s
                        .title
                        .as_deref()
                        .or(s.snippet.as_deref())
                        .unwrap_or("<untitled>");
                    let marker = if s.archived_at.is_some() {
                        " [archived]"
                    } else {
                        ""
                    };
                    println!("{}: {}{}", s.id, title, marker);
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
            SpacesCommand::Rename { id, title } => {
                core.rename_space(id.clone(), title).await?;
                println!("renamed space {id}");
                Ok(())
            }
        },
        Some(Command::Model { command }) => match command {
            ModelCommand::List => {
                let state = core.local_models_state().await?;
                match &state.engine_path {
                    Some(p) => println!("engine: {p}"),
                    None => println!(
                        "engine: llama-server not found — install llama.cpp \
                         (e.g. `brew install llama.cpp`)"
                    ),
                }
                println!();
                if state.models.is_empty() {
                    println!("no local models yet");
                } else {
                    for m in &state.models {
                        let status = match &m.status {
                            eidola_app_core::LocalModelStatus::Downloading { received, total } => {
                                match total {
                                    Some(t) if *t > 0 => {
                                        format!("downloading {}%", received * 100 / t)
                                    }
                                    _ => "downloading".to_string(),
                                }
                            }
                            eidola_app_core::LocalModelStatus::Available => "available".into(),
                            eidola_app_core::LocalModelStatus::Loading => "loading".into(),
                            eidola_app_core::LocalModelStatus::Loaded { port, .. } => {
                                format!("loaded (127.0.0.1:{port})")
                            }
                        };
                        println!(
                            "{:<40} {:>9}  {}",
                            m.id,
                            m.size_bytes.map(fmt_size).unwrap_or_default(),
                            status
                        );
                        if let Some(err) = &m.last_error {
                            println!("    last error: {}", err.lines().next().unwrap_or(err));
                        }
                    }
                }
                println!("\ncatalog (download with `eidola model download <id>`):");
                for entry in core.local_model_catalog() {
                    let installed = state.models.iter().any(|m| m.file_name == entry.file_name);
                    println!(
                        "{:<18} {:>9}  {}{}",
                        entry.id,
                        fmt_size(entry.size_bytes),
                        entry.description,
                        if installed { "  [installed]" } else { "" }
                    );
                }
                Ok(())
            }
            ModelCommand::Download { source } => {
                // A catalog id resolves to its URL; anything else is a URL.
                let url = core
                    .local_model_catalog()
                    .iter()
                    .find(|c| c.id == source)
                    .map(|c| c.url.to_string())
                    .unwrap_or(source);
                let id = core.download_local_model(url).await?;
                println!("downloading {id}…");
                // The transfer task dies with this process, so wait for it,
                // rendering a progress line.
                let slug = id.strip_prefix("local/").unwrap_or(&id).to_string();
                loop {
                    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                    let state = core.local_models_state().await?;
                    let Some(m) = state.models.iter().find(|m| m.slug == slug) else {
                        return Err(AppError::LocalModel {
                            message: "model disappeared during download".into(),
                        });
                    };
                    match &m.status {
                        eidola_app_core::LocalModelStatus::Downloading { received, total } => {
                            match total {
                                Some(t) if *t > 0 => print!(
                                    "\r{} / {} ({}%)   ",
                                    fmt_size(*received),
                                    fmt_size(*t),
                                    received * 100 / t
                                ),
                                _ => print!("\r{}   ", fmt_size(*received)),
                            }
                            let _ = std::io::stdout().flush();
                        }
                        _ => {
                            println!();
                            match &m.last_error {
                                Some(err) => {
                                    return Err(AppError::LocalModel {
                                        message: format!("download failed: {err}"),
                                    });
                                }
                                None => println!("downloaded {id}"),
                            }
                            break;
                        }
                    }
                }
                Ok(())
            }
            ModelCommand::Delete { id } => {
                core.delete_local_model(id.clone()).await?;
                println!("deleted {id}");
                Ok(())
            }
            ModelCommand::Load { id } => {
                println!("loading {id} (this can take a while for large models)…");
                core.load_local_model(id.clone()).await?;
                let state = core.local_models_state().await?;
                let slug = id.strip_prefix("local/").unwrap_or(&id);
                if let Some(eidola_app_core::LocalModelStatus::Loaded { port, .. }) = state
                    .models
                    .iter()
                    .find(|m| m.slug == slug)
                    .map(|m| m.status.clone())
                {
                    println!("loaded — serving on 127.0.0.1:{port}");
                    println!("chat with it: `eidola chat \"hi\" --model local/{slug}`");
                }
                // The engine is a child of *this* process, so exiting would
                // kill it. Keep serving until Ctrl-C (the GUI, by contrast,
                // holds engines for its whole app lifetime).
                println!("serving — press Ctrl-C to stop");
                let _ = tokio::signal::ctrl_c().await;
                core.unload_local_model(id.clone()).await.ok();
                println!("\nunloaded {id}");
                Ok(())
            }
            ModelCommand::Unload { id } => {
                core.unload_local_model(id.clone()).await?;
                println!("unloaded {id}");
                Ok(())
            }
        },
        Some(Command::Backend { command }) => match command {
            BackendCommand::List => {
                let backends = core.list_backends().await?;
                for b in &backends {
                    let state = if b.enabled { "enabled" } else { "disabled" };
                    println!(
                        "{:<16} {:<9} {:<9} {}",
                        b.id,
                        b.kind.as_str(),
                        state,
                        b.display_name
                    );
                    if let Some(url) = &b.base_url {
                        println!(
                            "    url: {url}{}",
                            if b.has_api_key { "  (api key set)" } else { "" }
                        );
                    }
                    if let Some(dir) = &b.models_dir {
                        println!("    models dir: {dir}");
                    }
                    if let Some(pinned) = &b.model_overrides {
                        println!("    pinned models: {}", pinned.join(", "));
                    }
                }
                Ok(())
            }
            BackendCommand::Add { kind } => {
                let new = match kind {
                    BackendAddCommand::Openai {
                        id,
                        url,
                        api_key,
                        name,
                        models,
                    } => eidola_app_core::NewBackend {
                        id,
                        kind: eidola_app_core::BackendKind::OpenAi,
                        display_name: name.unwrap_or_default(),
                        base_url: Some(url),
                        api_key,
                        models_dir: None,
                        model_overrides: models,
                    },
                    BackendAddCommand::Llamacpp {
                        id,
                        models_dir,
                        name,
                    } => eidola_app_core::NewBackend {
                        id,
                        kind: eidola_app_core::BackendKind::LlamaCpp,
                        display_name: name.unwrap_or_default(),
                        base_url: None,
                        api_key: None,
                        models_dir: Some(models_dir),
                        model_overrides: None,
                    },
                };
                let added = core.add_backend(new).await?;
                println!("added backend `{}` ({})", added.id, added.kind.as_str());
                println!(
                    "address its models as `<model>@{}` (see `eidola backend models {}`)",
                    added.id, added.id
                );
                Ok(())
            }
            BackendCommand::Enable { id } => {
                core.set_backend_enabled(id.clone(), true).await?;
                println!("enabled backend `{id}`");
                Ok(())
            }
            BackendCommand::Disable { id } => {
                core.set_backend_enabled(id.clone(), false).await?;
                println!("disabled backend `{id}`");
                if id == "eidola" {
                    println!(
                        "asks now route only to local / configured backends (no account needed)"
                    );
                }
                Ok(())
            }
            BackendCommand::Remove { id } => {
                core.remove_backend(id.clone()).await?;
                println!("removed backend `{id}`");
                Ok(())
            }
            BackendCommand::Models { id } => {
                let models = core.backend_models(id.clone()).await?;
                if models.is_empty() {
                    println!("no models offered (for engine-backed backends, load one first)");
                    return Ok(());
                }
                for m in &models {
                    if m.context_length > 0 {
                        println!("{:<48} {}-token context", m.id, m.context_length);
                    } else {
                        println!("{}", m.id);
                    }
                }
                Ok(())
            }
        },
        Some(Command::Update {
            command: UpdateCommand::Check,
        }) => {
            let snapshot = core.update_check().await;
            print_update_check(&snapshot);
            match snapshot.result {
                // Distinct exit codes so scripts can route on the two
                // states that need a human decision.
                eidola_app_core::updates::UpdateCheckResult::Unverifiable { .. } => {
                    std::process::exit(2);
                }
                eidola_app_core::updates::UpdateCheckResult::ClaimsChanged { .. } => {
                    std::process::exit(3);
                }
                _ => Ok(()),
            }
        }
        Some(Command::Update {
            command:
                UpdateCommand::Verify {
                    installed_version,
                    installed_git_commit,
                    fixtures_dir,
                    verbose,
                },
        }) => {
            let installed_version =
                installed_version.unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
            let installed_git_commit = installed_git_commit.as_deref();

            eprintln!("Checking for updates...");
            eprintln!(
                "  installed version: {installed_version}{}",
                installed_git_commit
                    .map(|c| format!(" (commit {c})"))
                    .unwrap_or_else(|| " (first-install; no commit pinned)".into())
            );
            if let Some(d) = fixtures_dir.as_deref() {
                eprintln!("  fixtures dir:      {d} (dev mode; no network fetches)");
            }
            eprintln!();

            let fetcher = match fixtures_dir.as_deref() {
                Some(dir) => eidola_app_core::updater::Fetcher::fixtures(dir),
                None => eidola_app_core::updater::Fetcher::network()?,
            };
            let opts = eidola_app_core::updater::VerifyOptions { verbose };
            let summary = eidola_app_core::updater::check_for_update_with(
                &fetcher,
                opts,
                &installed_version,
                installed_git_commit,
            )
            .await?;

            match summary {
                None => {
                    println!("You're already on the latest release.");
                }
                Some(s) => print_release_summary(&s),
            }
            Ok(())
        }
    }
}

/// Human-readable byte size (GB/MB) for model listings.
fn fmt_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1e9)
    } else {
        format!("{:.0} MB", bytes as f64 / 1e6)
    }
}

/// Print one `update check` outcome — the same five states the GUI's
/// Updates window renders, sharing the core types.
fn print_update_check(snapshot: &eidola_app_core::updates::UpdateCheckSnapshot) {
    use eidola_app_core::updates::UpdateCheckResult;

    match &snapshot.result {
        UpdateCheckResult::UpToDate { latest_version } => {
            println!("Eidola {} is up to date.", env!("CARGO_PKG_VERSION"));
            match latest_version {
                Some(v) => println!("latest release: v{v}"),
                None => println!("no release is marked latest yet"),
            }
        }
        UpdateCheckResult::UpdateAvailable { release } => {
            println!(
                "Eidola v{} is available — cryptographically verified.",
                release.version
            );
            if release.claims_accepted {
                println!(
                    "(its claims differ from this build's expectations; you previously chose \
                     to treat it as an update)"
                );
            }
            println!("  signed by: {}", release.ci_identity);
            println!(
                "  rekor:     https://search.sigstore.dev/?logIndex={}",
                release.rekor_log_index
            );
            if let Some(url) = &release.release_url {
                println!("  release:   {url}");
            }
        }
        UpdateCheckResult::Unverifiable {
            version,
            tag,
            reason,
        } => {
            eprintln!("SECURITY WARNING: release v{version} ({tag}) could not be verified.");
            eprintln!();
            eprintln!("  {reason}");
            eprintln!();
            eprintln!(
                "This may be a fake release or a compromised update channel. Do not download \
                 or install it. This warning persists until a later check finds a verifiable \
                 latest release."
            );
        }
        UpdateCheckResult::ClaimsChanged {
            release,
            comparison,
        } => {
            println!(
                "Release v{} verified cryptographically, but its attested claims differ \
                 from what this build expects.",
                release.version
            );
            println!();
            println!("  {:<40} {:<44} attested", "claim", "expected");
            for delta in &comparison.deltas {
                println!(
                    "  {:<40} {:<44} {}",
                    delta.key,
                    delta.expected.as_deref().unwrap_or("—"),
                    delta.attested.as_deref().unwrap_or("—"),
                );
            }
            println!();
            println!(
                "{} of {} expected claims match. This release is NOT treated as an update \
                 until you explicitly accept the change in the app's Updates window \
                 (Eidola menu → Check for Updates…).",
                comparison.expected.len().saturating_sub(
                    comparison
                        .deltas
                        .iter()
                        .filter(|d| d.expected.is_some())
                        .count()
                ),
                comparison.expected.len(),
            );
        }
        UpdateCheckResult::CheckFailed { message } => {
            println!("Couldn't check for updates (offline?).");
            println!("  {message}");
        }
    }
}

fn print_release_summary(s: &eidola_app_core::updater::ReleaseSummary) {
    println!("=== New release available ===");
    println!();
    println!("  version:    {}", s.version);
    println!("  tag:        {}", s.git_tag);
    println!("  git_commit: {}", s.git_commit);
    println!("  released:   {}", s.released_at);
    if let Some(prev) = &s.previous_release {
        println!("  previous:   {} ({})", prev.version, prev.git_commit);
    }
    println!();
    println!(
        "=== Verified human attestations ({} total) ===",
        s.attestations.len()
    );
    for att in &s.attestations {
        println!();
        println!("  attestant: {} <{}>", att.attestant_name, att.attestant_id);
        println!("    jurisdiction:   {}", att.jurisdiction);
        println!("    key fingerprint: sha256:{}", att.fingerprint_hex);
        println!(
            "    rekor entry:    https://search.sigstore.dev/?logIndex={}",
            att.rekor_log_index
        );
        println!("    attested at:    {}", att.attested_at);
        println!();
        println!("  preamble:");
        print_indented(&att.attestant_statement, 4);
        println!();
        println!("  claims:");
        for claim in &att.claims {
            println!("    {}:", claim.claim_id);
            print_indented(&claim.statement, 6);
        }
    }
    println!();
    println!(
        "All cryptographic verification stages passed: CI Sigstore bundle, every human \
         signature + Rekor inclusion, and every signed claim matches its pinned template."
    );
    println!();
    println!(
        "Install is not yet implemented (deferred to step 5). When it lands you'll review \
         the prose above, then approve the install."
    );
}

fn print_indented(text: &str, indent: usize) {
    let pad = " ".repeat(indent);
    // Word-wrap at ~76 columns minus the indent so the output stays
    // readable in a terminal. Falls back gracefully on words longer than
    // the wrap width.
    let wrap = 76usize.saturating_sub(indent);
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > wrap {
                println!("{pad}{line}");
                line.clear();
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            println!("{pad}{line}");
        }
    }
}
