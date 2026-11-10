//! Private support tool. Authority is the caller's database and Stripe access.

use clap::{Args, Parser, Subcommand};
use eidola_billing::BillingError;
use eidola_billing::service::BillingService;
use eidola_billing::store::{RefundOperation, RefundRequest, Store};
use eidola_billing::stripe::StripeClient;
use serde_json::{Value, json};
use std::net::SocketAddr;
use std::time::Duration;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "eidola-billing", about = "Private Eidola billing operations")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run only the stable webhook receiver; needs no Stripe API key.
    ServeWebhooks {
        #[arg(long, default_value = "0.0.0.0:8080")]
        bind: SocketAddr,
    },
    /// Run only the durable billing work loop; opens no HTTP listener.
    Work {
        #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(i64).range(1..=100))]
        batch_size: i64,
        #[arg(long, default_value_t = 5, value_parser = clap::value_parser!(u64).range(1..=3600))]
        retry_interval_secs: u64,
    },
    /// Inspect local billing records; no provider calls or mutations.
    Inspect(Account),
    /// Fetch current Stripe facts and reconcile this account.
    Reconcile(Account),
    /// Retry a bounded batch of durably received webhook events.
    DrainEvents {
        #[arg(long, default_value_t = 32)]
        limit: i64,
    },
    /// Quote a refund; add --execute to reserve credits and submit it.
    Refund(Refund),
    /// Reconcile or safely retry the SAME durable refund operation.
    ResumeRefund(Operation),
    /// Release a prepared or definitively failed refund's held credits.
    ReleaseRefund(Operation),
    /// Retry a definitively failed refund with a new attempt, preserving its hold.
    RetryRefund {
        #[arg(long)]
        operation: Uuid,
        #[arg(long)]
        new_operation: Uuid,
        #[arg(long)]
        reason: String,
    },
    /// Record an alternate payout ALREADY completed; does not send money.
    ConfirmExternalRefund {
        #[arg(long)]
        operation: Uuid,
        #[arg(long)]
        reference: String,
    },
    /// Cancel existing subscriptions; granted credits keep their expiry.
    CancelSubscriptions(Account),
    /// Freeze the account, cancel subscriptions and refund eligible credits.
    Close {
        #[arg(long)]
        account: Uuid,
        #[arg(long)]
        reason: String,
        /// Without this flag, show the current account records only.
        #[arg(long)]
        execute: bool,
    },
    /// Generate an operation UUID to retain with the support case.
    NewOperation,
}

#[derive(Args)]
struct Account {
    #[arg(long)]
    account: Uuid,
}

#[derive(Args)]
struct Operation {
    #[arg(long)]
    operation: Uuid,
}

#[derive(Args)]
struct Refund {
    #[arg(long)]
    account: Uuid,
    #[arg(long)]
    purchase: String,
    /// Reuse this exact UUID and request when retrying.
    #[arg(long)]
    operation: Uuid,
    /// Account credits to cancel; mutually exclusive with --cash-minor.
    #[arg(
        long,
        conflicts_with = "cash_minor",
        required_unless_present = "cash_minor"
    )]
    credits: Option<i64>,
    /// Cash-only refund or remedy in original currency minor units; NO credit effect.
    #[arg(long, conflicts_with = "credits", required_unless_present = "credits")]
    cash_minor: Option<i64>,
    #[arg(long)]
    reason: String,
    #[arg(long)]
    execute: bool,
}

impl Refund {
    fn request(&self) -> Result<RefundRequest, BillingError> {
        if self.credits.is_some_and(|n| n <= 0)
            || self.cash_minor.is_some_and(|n| n <= 0)
            || self.reason.trim().is_empty()
        {
            return Err(BillingError::Invalid(
                "refund requires a positive amount and a reason".into(),
            ));
        }
        Ok(RefundRequest {
            id: self.operation,
            account_id: self.account,
            purchase_id: self.purchase.clone(),
            credits: self.credits.unwrap_or(0),
            cash_minor: self.cash_minor,
            reason: self.reason.clone(),
        })
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let _ = rustls_rustcrypto::provider().install_default();
    if matches!(
        cli.command,
        Command::ServeWebhooks { .. } | Command::Work { .. }
    ) {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(std::io::stderr)
            .init();
    }
    match run(cli).await {
        Ok(value) => println!("{}", serde_json::to_string_pretty(&value).unwrap()),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

async fn run(cli: Cli) -> Result<Value, BillingError> {
    if matches!(cli.command, Command::NewOperation) {
        return Ok(json!({ "operation": Uuid::new_v4() }));
    }
    // Credentials belong to the environment/workload, never argv or output.
    let database_url = required_env("DATABASE_URL")?;
    let password = std::env::var("DATABASE_PASSWORD").ok();
    let cert = std::env::var("DATABASE_SSL_CERT").ok();
    let pool = eidola_billing::connection::create_pool(
        &database_url,
        password.as_deref(),
        cert.as_deref(),
    )
    .map_err(BillingError::Database)?;
    let store = Store::new(pool.clone());
    match &cli.command {
        Command::ServeWebhooks { bind } => {
            let secret = required_env("STRIPE_WEBHOOK_SECRET")?;
            let shutdown = eidola_billing_cli::runtime::shutdown_signal()?;
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .map_err(|_| BillingError::Network("could not bind webhook listener".into()))?;
            eidola_billing_cli::runtime::serve_webhooks(listener, store, secret, shutdown).await?;
            return Ok(json!({"stopped": "webhooks"}));
        }
        Command::Inspect(args) => return inspect(&store, args.account).await,
        Command::Close {
            account,
            execute: false,
            ..
        } => return inspect(&store, *account).await,
        Command::Refund(args) if !args.execute => {
            let operation = store.preview_refund(&args.request()?).await?;
            return Ok(refund_json(&operation));
        }
        _ => {}
    }
    let service = BillingService::new(pool, StripeClient::new(required_env("STRIPE_API_KEY")?));
    match cli.command {
        Command::Work {
            batch_size,
            retry_interval_secs,
        } => {
            let shutdown = eidola_billing_cli::runtime::shutdown_signal()?;
            eidola_billing_cli::runtime::work(
                service,
                batch_size,
                Duration::from_secs(retry_interval_secs),
                shutdown,
            )
            .await?;
            Ok(json!({"stopped": "worker"}))
        }
        Command::Reconcile(args) => {
            serde_json::to_value(service.reconcile_account(args.account).await?)
                .map_err(|_| BillingError::Parse("report serialization failed".into()))
        }
        Command::DrainEvents { limit } => serde_json::to_value(service.drain_events(limit).await?)
            .map_err(|_| BillingError::Parse("report serialization failed".into())),
        Command::Refund(args) => Ok(refund_json(&service.submit_refund(&args.request()?).await?)),
        Command::ResumeRefund(args) => {
            Ok(refund_json(&service.resume_refund(args.operation).await?))
        }
        Command::ReleaseRefund(args) => {
            Ok(refund_json(&service.release_refund(args.operation).await?))
        }
        Command::RetryRefund {
            operation,
            new_operation,
            reason,
        } => Ok(refund_json(
            &service
                .retry_refund(operation, new_operation, &reason)
                .await?,
        )),
        Command::ConfirmExternalRefund {
            operation,
            reference,
        } => Ok(refund_json(
            &service
                .confirm_external_refund(operation, &reference)
                .await?,
        )),
        Command::CancelSubscriptions(args) => Ok(json!({
            "canceled": service.cancel_subscriptions(args.account).await?
        })),
        Command::Close {
            account, reason, ..
        } => serde_json::to_value(service.close_account(account, &reason).await?)
            .map_err(|_| BillingError::Parse("report serialization failed".into())),
        Command::Inspect(_) | Command::NewOperation | Command::ServeWebhooks { .. } => {
            unreachable!("handled before provider access")
        }
    }
}

async fn inspect(store: &Store, account_id: Uuid) -> Result<Value, BillingError> {
    let purchases = store.list_purchases(account_id).await?;
    let eligible: std::collections::HashMap<_, _> = store
        .list_refundable_purchases(account_id)
        .await?
        .into_iter()
        .map(|(purchase, credits)| (purchase.id, credits))
        .collect();
    let refunds = store.list_refunds(account_id).await?;
    let debts = store.debts(account_id).await?;
    Ok(json!({
        "account": account_id,
        "state": store.account_state(account_id).await?,
        "review": store.review_issues(account_id).await?,
        "purchases": purchases.iter().map(|p| json!({
            "id": p.id, "charge": p.charge_id, "invoice": p.invoice_id,
            "payment_intent": p.payment_intent, "currency": p.currency,
            "paid_minor": p.paid_minor, "credits": p.credits,
            "eligible_credits": eligible.get(&p.id).copied().unwrap_or(0),
            "expires_at_unix": unix(p.expires_at), "purchased_at_unix": unix(p.purchased_at)
        })).collect::<Vec<_>>(),
        "refunds": refunds.iter().map(refund_json).collect::<Vec<_>>(),
        "debts": debts.iter().map(|d| json!({"currency":d.currency,"amount_minor":d.amount_minor})).collect::<Vec<_>>()
    }))
}

fn refund_json(op: &RefundOperation) -> Value {
    json!({
        "operation": op.id, "account": op.account_id, "purchase": op.purchase_id,
        "credits_canceled": op.credits, "cash_minor": op.amount_minor,
        "currency": op.currency, "charge": op.charge_id, "state": op.state,
        "refund": op.refund_id, "created_at_unix": unix(op.created_at),
        "first_submitted_at_unix": op.submitted_at.map(unix)
    })
}

fn unix(time: std::time::SystemTime) -> Option<u64> {
    time.duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|t| t.as_secs())
}

fn required_env(name: &str) -> Result<String, BillingError> {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| BillingError::Invalid(format!("{name} is required")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refund_args() -> Vec<&'static str> {
        vec![
            "eidola-billing",
            "refund",
            "--account",
            "11111111-1111-4111-8111-111111111111",
            "--purchase",
            "cs_example",
            "--operation",
            "22222222-2222-4222-8222-222222222222",
            "--reason",
            "customer requested refund",
        ]
    }

    #[test]
    fn cash_only_refunds_cannot_accidentally_cancel_credits() {
        let mut args = refund_args();
        args.extend(["--cash-minor", "500"]);
        let Command::Refund(command) = Cli::try_parse_from(args.clone()).unwrap().command else {
            panic!()
        };
        let request = command.request().unwrap();
        assert_eq!(request.credits, 0);
        assert_eq!(request.cash_minor, Some(500));
        assert!(!command.execute);
        args.extend(["--credits", "1000000"]);
        assert!(Cli::try_parse_from(args).is_err());
    }

    #[test]
    fn normal_refund_requires_an_explicit_persisted_operation_and_credit_amount() {
        assert!(Cli::try_parse_from(refund_args()).is_err());
        let mut args = refund_args();
        args.extend(["--credits", "1000000", "--execute"]);
        let Command::Refund(command) = Cli::try_parse_from(args).unwrap().command else {
            panic!()
        };
        assert!(command.execute);
        assert_eq!(command.request().unwrap().cash_minor, None);
        assert_eq!(command.request().unwrap().credits, 1_000_000);
    }

    #[test]
    fn repair_accepts_no_cash_or_credit_instructions() {
        assert!(
            Cli::try_parse_from([
                "eidola-billing",
                "reconcile",
                "--account",
                "11111111-1111-4111-8111-111111111111",
                "--credits",
                "100"
            ])
            .is_err()
        );
    }

    #[test]
    fn runtime_roles_are_explicit_and_worker_options_are_bounded() {
        assert!(Cli::try_parse_from(["eidola-billing"]).is_err());
        assert!(matches!(
            Cli::try_parse_from([
                "eidola-billing",
                "serve-webhooks",
                "--bind",
                "127.0.0.1:8080"
            ])
            .unwrap()
            .command,
            Command::ServeWebhooks { .. }
        ));
        assert!(matches!(
            Cli::try_parse_from(["eidola-billing", "work"])
                .unwrap()
                .command,
            Command::Work { .. }
        ));
        for (option, value) in [
            ("--batch-size", "0"),
            ("--batch-size", "101"),
            ("--retry-interval-secs", "0"),
            ("--retry-interval-secs", "3601"),
        ] {
            assert!(Cli::try_parse_from(["eidola-billing", "work", option, value]).is_err());
        }
    }
}
