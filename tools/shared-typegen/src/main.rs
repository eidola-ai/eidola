use anyhow::Result;
use crux_core::typegen::TypeGen;
use eidolons_shared::{EidolonsApp, Screen};
use std::path::PathBuf;

fn main() -> Result<()> {
    let output_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./generated"));

    eprintln!("Generating Swift types to: {}", output_dir.display());

    let mut typegen = TypeGen::new();

    // Screen has many unit variants — provide samples so serde-reflection
    // can discover all of them before register_app traces the type graph.
    typegen.register_samples(vec![
        Screen::Idle,
        Screen::Loading,
        Screen::Account,
        Screen::AccountCreated,
        Screen::Prices,
        Screen::Checkout,
        Screen::Balances,
        Screen::Error,
    ])?;

    // Register the app to generate types for Event, Effect, ViewModel, etc.
    typegen.register_app::<EidolonsApp>()?;

    // Register crux_http protocol types for shell-side deserialization
    typegen.register_type::<crux_http::protocol::HttpRequest>()?;
    typegen.register_type::<crux_http::protocol::HttpResponse>()?;
    typegen.register_type::<crux_http::protocol::HttpResult>()?;

    // Generate Swift types
    typegen.swift("SharedTypes", &output_dir)?;

    eprintln!("Type generation complete");

    Ok(())
}
