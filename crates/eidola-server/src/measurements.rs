//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.78
/// Built: 2026-03-16T23:01:15Z
/// Artifact digest: sha256:07d4f234c4a27efba2375fc2e4bd24dcec20a2c8b3bc4aa93f3cbb09595c7919
/// Rekor log index: 1207551895
/// Sigstore: https://search.sigstore.dev/?logIndex=1207551895
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.78
const CURRENT: &str = "796de90863fcfb5cb9270983f1a7e43c26869191f99ab8301b0f90a1b89e4c2a715d09e26b52624d1585e3cd95bc5679";

/// Previous: v0.0.77
const PREVIOUS: &str = "0fd23214514ef717881179b954dc06618f6abe13b15cd372e040047e2a28cb7b56d3bb482ededfd12f0f5caf7880ccb4";

pub const ALLOWED: &[&str] = &[CURRENT, PREVIOUS];
