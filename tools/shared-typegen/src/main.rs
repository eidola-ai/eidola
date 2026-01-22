use anyhow::Result;
use crux_core::typegen::TypeGen;
use eidolons_shared::EidolonsApp;
use std::path::PathBuf;

fn main() -> Result<()> {
    let output_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./generated"));

    eprintln!("Generating Swift types to: {}", output_dir.display());

    let mut typegen = TypeGen::new();

    // Register the app to generate types for Event, Effect, ViewModel, etc.
    typegen.register_app::<EidolonsApp>()?;

    // Generate Swift types
    typegen.swift("SharedTypes", &output_dir)?;

    eprintln!("Type generation complete");

    Ok(())
}
