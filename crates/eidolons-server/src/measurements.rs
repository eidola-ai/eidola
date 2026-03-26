//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.75
/// Built: 2026-03-16T23:01:15Z
/// Artifact digest: sha256:c337d719792307909947c779db80d834d1f73515199436ef3a7b880e60898c92
/// Rekor log index: 1186330805
/// Sigstore: https://search.sigstore.dev/?logIndex=1186330805
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.75
const CURRENT: &str =
    "c4a4c013a36ee9a55b4c25636276dd05c96407c5500a895e2932f9139403470d1116c1df0dbbd3c6891c607c134e0576";

/// Previous: v0.0.70
const PREVIOUS: &str =
    "630af778a62ccb543893f27e9669d8b52c73e3829d75fb7efa626af5c108c56a367b5734a62a93ec347111ee064011bd";

pub const ALLOWED: &[&str] = &[
    CURRENT,
    PREVIOUS,
];
