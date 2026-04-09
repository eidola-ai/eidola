//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Each [`EnclaveMeasurement`] entry pairs the AMD SEV-SNP launch digest with
//! the Intel TDX RTMR1/RTMR2 values for a single Tinfoil release. This mirrors
//! the shape used in Tinfoil's `tinfoil-deployment.json` and our own
//! `artifact-manifest.json`. `tinfoil-verifier` picks the matching field at
//! verification time based on the platform that produced the attestation.
//!
//! Source repo: tinfoilsh/confidential-model-router

use std::sync::LazyLock;

use tinfoil_verifier::{EnclaveMeasurement, TdxMeasurement};

pub static ALLOWED: LazyLock<Vec<EnclaveMeasurement>> = LazyLock::new(|| {
    vec![
        // === CURRENT: v0.0.83 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:9fab386e3e45627721d5303421e28bb66689780cf7ed477e7e22b26f5ff689b4
        // Rekor log index: 1260499310
        // Sigstore: https://search.sigstore.dev/?logIndex=1260499310
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.83
        EnclaveMeasurement {
            snp_measurement:
                "85fc3906b3e2bd2d10e4b3411016c13aec03ebe2e2777159e9abdc46f24d4e47146c92664971d0a38ffdae8276ba80bf"
                    .into(),
            tdx_measurement: TdxMeasurement {
                rtmr1:
                    "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d"
                        .into(),
                rtmr2:
                    "e6bccb0314f2bd5db061625eab7fb0948baabdc2a795cb3503a93394de309d266485364c829cf811470c5c248e93fc56"
                        .into(),
            },
        },
        // === PREVIOUS: v0.0.82 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:27e255a1834aee11feca4f13c86625dafb0b98f8395ffff4baf60e93005e704c
        // Rekor log index: 1258002931
        // Sigstore: https://search.sigstore.dev/?logIndex=1258002931
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.82
        EnclaveMeasurement {
            snp_measurement:
                "903e7371df42ae7c92d827b5f95acdbad5fd1683e97ca38b7accfe5b19e9354613dc83bbca1eb0b74d5cc06d3c84e62c"
                    .into(),
            tdx_measurement: TdxMeasurement {
                rtmr1:
                    "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d"
                        .into(),
                rtmr2:
                    "3101fac8a5c95de61b6e6fe1647957374c3422f20da685086d0d71f510991f698dc80eeb4aeb2871b754d938f271a725"
                        .into(),
            },
        },
    ]
});
