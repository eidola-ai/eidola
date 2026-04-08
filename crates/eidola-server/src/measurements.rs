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
        // === CURRENT: v0.0.82 ===
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
        // === PREVIOUS: v0.0.81 ===
        // Built: 2026-04-03T16:16:04Z
        // Artifact digest: sha256:c0275a148227e6efddf5bfc61052c188ae0bac984bf70e3f4cd75ef7c943203a
        // Rekor log index: 1230171568
        // Sigstore: https://search.sigstore.dev/?logIndex=1230171568
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.81
        EnclaveMeasurement {
            snp_measurement:
                "d6848e43be21b268536059930c717abb7004279e860cbbb8f88be8a48d250d972276d936c0896bd157984bbec77d4919"
                    .into(),
            tdx_measurement: TdxMeasurement {
                rtmr1:
                    "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d"
                        .into(),
                rtmr2:
                    "34cd93a0c2ea0629323c09145636a25a0ac1ead868ff9337e315fb3ce846763eb5c5c97a4927c34b24bb513e8f74db70"
                        .into(),
            },
        },
    ]
});
