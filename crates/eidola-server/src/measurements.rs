//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! Updated by CI when Tinfoil publishes new enclave builds.
//! At least two entries for rolling deploys (current + previous).
//!
//! Each `EnclaveMeasurement` entry pairs the AMD SEV-SNP launch digest with
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
        // === CURRENT: v0.0.84 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:67c958c281bb7652b72d0363b0c23aaa63f86608f0205a18b777a0c400b3df84
        // Rekor log index: 1268393495
        // Sigstore: https://search.sigstore.dev/?logIndex=1268393495
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.84
        EnclaveMeasurement {
            snp_measurement: "f2214f4e72b9fdd7f2a842020f17d1da4c3e2c324487b9396ad1891c2ebf5c7c25a5a78f4b5c5855136dd619bc77b5b3".into(),
            tdx_measurement: TdxMeasurement {
                rtmr1: "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d".into(),
                rtmr2: "49e134efb1b8415fd5d3b04683a6760558e3a103e0bd516afb86b1475328dd8d3ca009b4be847ea4fd7caef2ada6b421".into(),
            },
        },
        // === PREVIOUS: v0.0.83 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:9fab386e3e45627721d5303421e28bb66689780cf7ed477e7e22b26f5ff689b4
        // Rekor log index: 1260499310
        // Sigstore: https://search.sigstore.dev/?logIndex=1260499310
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.83
        EnclaveMeasurement {
            snp_measurement: "85fc3906b3e2bd2d10e4b3411016c13aec03ebe2e2777159e9abdc46f24d4e47146c92664971d0a38ffdae8276ba80bf".into(),
            tdx_measurement: TdxMeasurement {
                rtmr1: "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d".into(),
                rtmr2: "e6bccb0314f2bd5db061625eab7fb0948baabdc2a795cb3503a93394de309d266485364c829cf811470c5c248e93fc56".into(),
            },
        },
    ]
});
