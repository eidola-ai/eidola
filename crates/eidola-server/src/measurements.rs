//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Source repo: tinfoilsh/confidential-model-router

/// Tinfoil inference enclave v0.0.82
/// Built: 2026-04-03T16:16:04Z
/// Artifact digest: sha256:27e255a1834aee11feca4f13c86625dafb0b98f8395ffff4baf60e93005e704c
/// Rekor log index: 1258002931
/// Sigstore: https://search.sigstore.dev/?logIndex=1258002931
/// GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.82
const CURRENT: &str = "903e7371df42ae7c92d827b5f95acdbad5fd1683e97ca38b7accfe5b19e9354613dc83bbca1eb0b74d5cc06d3c84e62c";

/// Previous: v0.0.81
const PREVIOUS: &str = "d6848e43be21b268536059930c717abb7004279e860cbbb8f88be8a48d250d972276d936c0896bd157984bbec77d4919";

pub const ALLOWED: &[&str] = &[CURRENT, PREVIOUS];
