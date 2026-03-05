use std::io::IsTerminal;

use eidolons_shared::{Screen, ViewModel};

use crate::config::Config;

pub struct ShellContext {
    pub no_browser: bool,
}

pub fn render_view(vm: &ViewModel, ctx: &ShellContext) -> Result<(), String> {
    match vm.screen {
        Screen::Idle => {}
        Screen::Loading => {}
        Screen::Account => {
            if let Some(id) = &vm.account_id {
                println!("id: {id}");
            }
            if let Some(cid) = &vm.account_stripe_customer_id {
                println!("stripe_customer_id: {cid}");
            }
            if let Some(at) = &vm.account_created_at {
                println!("created_at: {at}");
            }
        }
        Screen::AccountCreated => {
            println!("account created");
            if let Some(id) = &vm.created_account_id {
                println!("id: {id}");
            }
            if let Some(at) = &vm.created_account_created_at {
                println!("created_at: {at}");
            }
            // Persist credentials to config
            if let (Some(id), Some(secret)) =
                (&vm.created_account_id, &vm.created_account_secret)
            {
                let mut config = Config::load();
                config.account_id = Some(id.clone());
                config.account_secret = Some(secret.clone());
                config.save()?;
            }
        }
        Screen::Prices => {
            if vm.prices.is_empty() {
                println!("no prices available");
            } else {
                for p in &vm.prices {
                    let amount = p
                        .unit_amount
                        .map(|a| {
                            format!(
                                "{}.{:02} {}",
                                a / 100,
                                a % 100,
                                p.currency.to_uppercase()
                            )
                        })
                        .unwrap_or_else(|| "free".to_string());

                    let recurrence = match (&p.recurring_interval, p.recurring_interval_count) {
                        (Some(interval), Some(1)) => format!("/{interval}"),
                        (Some(interval), Some(count)) => format!("/{count}x{interval}"),
                        _ => String::new(),
                    };

                    println!(
                        "{}: {} ({}{}, {} credits)",
                        p.id, p.product_name, amount, recurrence, p.credits
                    );
                    if let Some(desc) = &p.product_description {
                        println!("  {desc}");
                    }
                }
            }
        }
        Screen::Checkout => {
            if let Some(url) = &vm.checkout_url {
                println!("{url}");
                let should_open = !ctx.no_browser && std::io::stdout().is_terminal();
                if should_open {
                    let _ = open::that(url);
                }
            }
        }
        Screen::Balances => {
            if let Some(available) = vm.balances_available {
                println!("available: {available}");
            }
            for pool in &vm.balances_pools {
                let expires = pool
                    .expires_at
                    .as_deref()
                    .map(|e| format!(", expires {e}"))
                    .unwrap_or_default();
                println!("  {} ({}{})", pool.amount, pool.source, expires);
            }
        }
        Screen::Error => {
            if let Some(err) = &vm.error {
                return Err(err.clone());
            }
        }
    }
    Ok(())
}
