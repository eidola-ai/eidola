//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.81
/// Built: 2026-04-03T16:16:04Z
/// Artifact digest: sha256:c0275a148227e6efddf5bfc61052c188ae0bac984bf70e3f4cd75ef7c943203a
/// Rekor log index: 1230171568
/// Sigstore: https://search.sigstore.dev/?logIndex=1230171568
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.81
const CURRENT: &str = "d6848e43be21b268536059930c717abb7004279e860cbbb8f88be8a48d250d972276d936c0896bd157984bbec77d4919";

/// Previous: v0.0.79
const PREVIOUS: &str = "2b24e7d18c2c6de912cfe32218bfb1a66d961cc3b06d623651114905bf617de95035ed47330ffa12cfb3847c3a369b37";

pub const ALLOWED: &[&str] = &[CURRENT, PREVIOUS];
