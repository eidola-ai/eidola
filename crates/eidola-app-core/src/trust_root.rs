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

use tinfoil_verifier::{AllowedMeasurement, PlatformMeasurement};

/// The pinned server enclave launch identity this client release verifies,
/// on the TEE the release's server runs in (`SERVER_PLATFORM`): on SEV-SNP the
/// launch digest of the release image the server's config names
/// (`SERVER_CVM_VERSION`, image `SERVER_IGVM_SHA256`), the config's hash as
/// HOST_DATA and the image's guest policy; on TDX the image's MRTD, the
/// config's hash in MRCONFIGID and the machine policy. The client pins one
/// TEE: a presentation of the server on the other is refused before it is
/// authenticated.
pub fn server_pin() -> AllowedMeasurement {
    let platform: PlatformMeasurement = serde_json::from_str(SERVER_PIN_JSON)
        .expect("build.rs wrote a pin in the verifier's shape");
    AllowedMeasurement {
        platform,
        expected_gpus: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compiled-in pin is one the attesting client's pin compiler accepts,
    /// on the TEE the release names, and its launch measurement and config
    /// hash are the ones the server URL is derived from.
    #[test]
    fn the_server_pin_compiles_and_names_the_urls_identity() {
        let pin = server_pin();
        tinfoil_verifier::validate_pins(std::slice::from_ref(&pin)).unwrap();
        match &pin.platform {
            PlatformMeasurement::SevSnp(snp) => {
                assert_eq!(SERVER_PLATFORM, "sev_snp");
                assert_eq!(snp.measurement, SERVER_LAUNCH_MEASUREMENT);
                assert_eq!(snp.host_data, SERVER_CONFIG_HASH);
            }
            PlatformMeasurement::Tdx(tdx) => {
                assert_eq!(SERVER_PLATFORM, "tdx");
                assert_eq!(tdx.mrtd, SERVER_LAUNCH_MEASUREMENT);
                assert_eq!(
                    tdx.mrconfigid,
                    format!("{SERVER_CONFIG_HASH}{}", "0".repeat(32))
                );
            }
        }
        assert!(SERVER_URL.contains(SERVER_URL_HASH));
    }
}
