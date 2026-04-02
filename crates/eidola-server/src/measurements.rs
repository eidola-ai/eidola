//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! SEV-SNP measurements are 96-char hex strings (48 bytes, single launch digest).
//! TDX measurements are 192-char hex strings (RTMR1 || RTMR2 concatenated, 96 bytes).
//! Both types coexist in the same ALLOWED list — the verifier dispatches based on
//! the attestation document's platform field, not the measurement format.
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.79
/// Built: 2026-03-16T23:01:15Z
/// Artifact digest: sha256:52be78d7a7fbbff988f252d77006e07c6874e7431c4752c01a56e6e9b8e2f853
/// Rekor log index: 1211200183
/// Sigstore: https://search.sigstore.dev/?logIndex=1211200183
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.79
const CURRENT: &str = "2b24e7d18c2c6de912cfe32218bfb1a66d961cc3b06d623651114905bf617de95035ed47330ffa12cfb3847c3a369b37";

/// Previous: v0.0.77
const PREVIOUS: &str = "0fd23214514ef717881179b954dc06618f6abe13b15cd372e040047e2a28cb7b56d3bb482ededfd12f0f5caf7880ccb4";

pub const ALLOWED: &[&str] = &[CURRENT, PREVIOUS];
