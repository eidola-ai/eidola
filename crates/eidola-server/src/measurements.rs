//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.70
/// Built: 2026-03-10T05:43:04Z
/// Artifact digest: sha256:2eeb293f089ad6859fe4c0b7542acf4db7d2f48bc483de5c466dea92cf06965b
/// Rekor log index: 1123037597
/// Sigstore: https://search.sigstore.dev/?logIndex=1123037597
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.70
const CURRENT: &str = "630af778a62ccb543893f27e9669d8b52c73e3829d75fb7efa626af5c108c56a367b5734a62a93ec347111ee064011bd";

pub const ALLOWED: &[&str] = &[CURRENT];
