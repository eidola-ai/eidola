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
        // === CURRENT: v0.0.81 ===
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
        // === PREVIOUS: v0.0.79 ===
        // Built: 2026-03-16T23:01:15Z
        // Artifact digest: sha256:52be78d7a7fbbff988f252d77006e07c6874e7431c4752c01a56e6e9b8e2f853
        // Rekor log index: 1211200183
        // Sigstore: https://search.sigstore.dev/?logIndex=1211200183
        // GitHub: https://github.com/tinfoilsh/confidential-model-router/releases/tag/v0.0.79
        EnclaveMeasurement {
            snp_measurement:
                "2b24e7d18c2c6de912cfe32218bfb1a66d961cc3b06d623651114905bf617de95035ed47330ffa12cfb3847c3a369b37"
                    .into(),
            tdx_measurement: TdxMeasurement {
                rtmr1:
                    "727f33421ba6f289eb5e2d4e27391cc8b0fcda181879dba38d87ce789dfad11927380ff32d016b6b7bc8733179b10753"
                        .into(),
                rtmr2:
                    "15c2aeeae78cd68b574ae971be0d9b404653feaee198995ccc972fcb25f267acaa8c1d3a1b052734f4fcad583331b12e"
                        .into(),
            },
        },
    ]
});
