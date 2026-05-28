//! Allowed code measurements for Tinfoil's inference enclave.
//!
//! The `ALLOWED` static is generated at build time by `build.rs` from
//! `releases/trust/tinfoil-enclaves.json` at the workspace root. The JSON
//! file is updated by `.github/workflows/update-measurements.yml` when
//! Tinfoil publishes a new enclave build, and is the place to audit /
//! diff measurement changes — never edit the generated file directly.
//!
//! Each `EnclaveMeasurement` pairs the AMD SEV-SNP launch digest with the
//! Intel TDX RTMR1/RTMR2 values for a single Tinfoil release. The shape
//! mirrors Tinfoil's `tinfoil-deployment.json` and our `artifact-manifest.json`
//! `enclave` block; `tinfoil-verifier` picks the matching field at
//! verification time based on the platform that produced the attestation.

include!(concat!(env!("OUT_DIR"), "/measurements.gen.rs"));
