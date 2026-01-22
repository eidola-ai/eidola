use anyhow::Result;
use crux_core::typegen::TypeGen;
use eidolons_shared::{ChatMessage, EidolonsApp, Role};
use std::path::PathBuf;

fn main() -> Result<()> {
    let output_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./generated"));

    eprintln!("Generating Swift types to: {}", output_dir.display());

    let mut typegen = TypeGen::new();

    // Register sample values for types used in ViewModel
    // This is needed for serde-reflection to trace nested enums
    typegen.register_type_with_samples(vec![
        ChatMessage {
            role: Role::User,
            content: "sample".to_string(),
        },
        ChatMessage {
            role: Role::Assistant,
            content: "sample".to_string(),
        },
    ])?;

    // Register the app to generate types for Event, Effect, ViewModel, etc.
    typegen.register_app::<EidolonsApp>()?;

    // Generate Swift types
    typegen.swift("SharedTypes", &output_dir)?;

    eprintln!("Type generation complete");

    Ok(())
}
