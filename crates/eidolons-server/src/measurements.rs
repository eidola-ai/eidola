//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.74
/// Built: 2026-03-16T23:01:15Z
/// Artifact digest: sha256:20dfa5176ad3a7f154018f121ae829a603d2b26c5125892620eed103c9f3bfa2
/// Rekor log index: 1178675460
/// Sigstore: https://search.sigstore.dev/?logIndex=1178675460
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.74
const CURRENT: &str =
    "cb5f7c127af60268407396a0692fe6c7f669a70288cd7fd56da3a7e140cdd1183b86dfba86ea16ceb7a6d47820d57a1d";

/// Previous: v0.0.70
const PREVIOUS: &str =
    "630af778a62ccb543893f27e9669d8b52c73e3829d75fb7efa626af5c108c56a367b5734a62a93ec347111ee064011bd";

pub const ALLOWED: &[&str] = &[
    CURRENT,
    PREVIOUS,
];
