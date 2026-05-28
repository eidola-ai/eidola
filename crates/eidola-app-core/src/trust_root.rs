//! Pinned trust root for verifying Eidola releases and the paired server
//! enclave.
//!
//! The constants below are generated at build time by `build.rs` from
//! the JSON files under `releases/trust/` + `releases/schema/` at the
//! workspace root (notably `server-enclave.json` for the paired-server
//! measurement). The generated source is at `$OUT_DIR/trust_root.gen.rs`.
//!
//! This is the *default* trust root for the binary. `Config` overrides
//! (`base_url`, `trusted_measurements`) take precedence at runtime so
//! developers can point a build at a local server or alternate enclave.

include!(concat!(env!("OUT_DIR"), "/trust_root.gen.rs"));

use tinfoil_verifier::{EnclaveMeasurement, TdxMeasurement};

/// The pinned server enclave measurement this client release verifies.
pub fn server_measurement() -> EnclaveMeasurement {
    EnclaveMeasurement {
        snp_measurement: SERVER_SNP_MEASUREMENT.to_string(),
        tdx_measurement: TdxMeasurement {
            rtmr1: SERVER_TDX_RTMR1.to_string(),
            rtmr2: SERVER_TDX_RTMR2.to_string(),
        },
    }
}
