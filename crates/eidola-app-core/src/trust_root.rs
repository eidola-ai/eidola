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

use tinfoil_verifier::SevSnpPin;

/// The pinned server enclave launch identity this client release verifies:
/// the SEV-SNP launch digest of the release image the server's config names
/// (`SERVER_CVM_VERSION`, image `SERVER_IGVM_SHA256`), the config's hash as
/// HOST_DATA, and the image's guest policy. The server is pinned on SEV-SNP
/// only; a TDX presentation of it is refused before it is authenticated.
pub fn server_measurement() -> SevSnpPin {
    SevSnpPin {
        measurement: SERVER_SNP_MEASUREMENT.to_string(),
        host_data: SERVER_SNP_HOST_DATA.to_string(),
        guest_policy: SERVER_SNP_GUEST_POLICY.to_string(),
    }
}
