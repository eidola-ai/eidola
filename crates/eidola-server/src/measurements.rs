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
        // === CURRENT: v0.0.91 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:8283d54c069cb7f2bfff74f5c39ebc7f6012a555bc5504ac4470820888d4d194
        // Rekor log index: 1398535472
        // Sigstore: https://search.sigstore.dev/?logIndex=1398535472
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.91
        EnclaveMeasurement {
            snp_measurement: "64e8af1427c39087d030db14a894d9ed1212611d8abf7245d8da4d425b5871eef5863740ce8e3586d1934e24e5fac959".into(),
            tdx_measurement: TdxMeasurement {
                rtmr1: "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d".into(),
                rtmr2: "e1eb892bf0aa9e60e48f74afadfd0ba456f200a2cd80056b87fde2ad05050bcc4bfb6d0bb49fe92a2b3728791e713358".into(),
            },
        },
        // === PREVIOUS: v0.0.84 ===
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
    ]
});
