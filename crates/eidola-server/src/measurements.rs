//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.77
/// Built: 2026-03-16T23:01:15Z
/// Artifact digest: sha256:616af84f7483699b0d08aeb0e31bff42bc38c1b84280045e306b3775f7df175b
/// Rekor log index: 1203584482
/// Sigstore: https://search.sigstore.dev/?logIndex=1203584482
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.77
const CURRENT: &str =
    "0fd23214514ef717881179b954dc06618f6abe13b15cd372e040047e2a28cb7b56d3bb482ededfd12f0f5caf7880ccb4";

/// Previous: v0.0.70
const PREVIOUS: &str =
    "630af778a62ccb543893f27e9669d8b52c73e3829d75fb7efa626af5c108c56a367b5734a62a93ec347111ee064011bd";

pub const ALLOWED: &[&str] = &[
    CURRENT,
    PREVIOUS,
];
