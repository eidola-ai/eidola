//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.83
/// Built: 2026-04-03T16:16:04Z
/// Artifact digest: sha256:9fab386e3e45627721d5303421e28bb66689780cf7ed477e7e22b26f5ff689b4
/// Rekor log index: 1260499310
/// Sigstore: https://search.sigstore.dev/?logIndex=1260499310
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.83
const CURRENT: &str = "85fc3906b3e2bd2d10e4b3411016c13aec03ebe2e2777159e9abdc46f24d4e47146c92664971d0a38ffdae8276ba80bf";

/// Previous: v0.0.81
const PREVIOUS: &str = "d6848e43be21b268536059930c717abb7004279e860cbbb8f88be8a48d250d972276d936c0896bd157984bbec77d4919";

pub const ALLOWED: &[&str] = &[CURRENT, PREVIOUS];
