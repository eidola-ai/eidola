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
        // === CURRENT: v0.0.102 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:53bb76eb6c26eec26f30e9b14d710c8eceeb75ea0c29c5208b5130dfa2b361e4
        // Rekor log index: 1551094524
        // Sigstore: https://search.sigstore.dev/?logIndex=1551094524
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.102
        EnclaveMeasurement {
            snp_measurement: "0a64f1b64b0b39b86d5b19acf95c746f9b9c99ac3bb4c64bcc976521d97dc46109a965363bc436c3be9b5b3b2c4a9f49".into(),
            tdx_measurement: TdxMeasurement {
                rtmr1: "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d".into(),
                rtmr2: "12498bc720197443c0753d33bc3e21eba5b92fa51f0bce130f4335e925ea4e3a9cbeca728f0c74f869cb27609fdfd0f8".into(),
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
