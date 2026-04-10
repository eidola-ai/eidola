use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tinfoil_verifier::EnclaveMeasurement;

/// The expected domain separator for credential operations.
///
/// This value is checked against the server's advertised domain separator
/// before issuing or spending any credential. An exact match is required.
///
/// This protects against a malicious operator silently changing the domain
/// separator to partition users into smaller anonymity sets (since credentials
/// under different domain separators are cryptographically unlinkable).
/// By hardcoding the expected value in the client and rejecting mismatches,
/// the operator cannot use the domain separator as a covert linking channel.
pub const DEFAULT_DOMAIN_SEPARATOR: &str = "ACT-v1:eidola:inference:production:2026-03-05";

/// Default GitHub source repository the eidola server enclave is attested
/// against. The CLI talks to an eidola-server running inside a Tinfoil
/// Container built from this repo, so this is always the right default
/// for the CLI's bootstrap attestation fetch.
pub const DEFAULT_ATTESTATION_REPO: &str = "eidola-ai/eidola";

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_secret: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_separator: Option<String>,
    /// Trusted enclave measurements.
    ///
    /// Each entry pairs a 96-char SEV-SNP launch digest with the matching
    /// Intel TDX RTMR1/RTMR2 values for a single Tinfoil release. The shape
    /// mirrors `tinfoil-deployment.json` and our `artifact-manifest.json`, so
    /// values can be lifted directly from those documents.
    ///
    /// When non-empty, the CLI verifies the server's Tinfoil attestation
    /// before sending any confidential requests. `tinfoil-verifier` detects
    /// the platform from the attestation document and checks the matching
    /// field of each entry.
    ///
    /// Example `config.toml`:
    ///
    /// ```toml
    /// [[trusted_measurements]]
    /// snp_measurement = "d6848e43..."
    /// tdx_measurement = { rtmr1 = "4f7be532...", rtmr2 = "34cd93a0..." }
    /// ```
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trusted_measurements: Vec<EnclaveMeasurement>,
    /// URL to fetch the server's attestation bundle for verification.
    /// Defaults to the Tinfoil ATC endpoint. Only used when
    /// `trusted_measurements` is non-empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_url: Option<String>,
    /// Source repository used to attest the enclave via the Tinfoil ATC
    /// `POST /attestation` endpoint. Defaults to
    /// [`DEFAULT_ATTESTATION_REPO`] (`eidola-ai/eidola`), since the CLI
    /// always connects to an eidola-server enclave built from this repo.
    /// Override only when pointing the CLI at a different attested
    /// upstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_repo: Option<String>,
    /// Optional PEM-encoded SEV-SNP ARK (Root CA) to use for verification.
    /// If provided, this overrides the built-in AMD Genoa ARK. Only consulted
    /// when verifying AMD SEV-SNP attestations; ignored for Intel TDX.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_root_ca: Option<String>,
    /// Optional PEM-encoded SEV-SNP ASK (Intermediate CA) to use for verification.
    /// If provided, this overrides the built-in AMD Genoa ASK. Only consulted
    /// when verifying AMD SEV-SNP attestations; ignored for Intel TDX.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_intermediate_ca: Option<String>,
}

impl Config {
    /// Returns the domain separator to enforce, falling back to the compiled-in default.
    pub fn domain_separator(&self) -> &str {
        self.domain_separator
            .as_deref()
            .unwrap_or(DEFAULT_DOMAIN_SEPARATOR)
    }

    /// Returns the source repo to attest the upstream enclave against,
    /// falling back to the compiled-in default ([`DEFAULT_ATTESTATION_REPO`]).
    pub fn attestation_repo(&self) -> &str {
        self.attestation_repo
            .as_deref()
            .unwrap_or(DEFAULT_ATTESTATION_REPO)
    }
}

impl Config {
    pub fn path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("eidola").join("config.toml"))
    }

    pub fn load() -> Config {
        let Some(path) = Self::path() else {
            return Config::default();
        };
        let Ok(contents) = fs::read_to_string(&path) else {
            return Config::default();
        };
        toml::from_str(&contents).unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        let path = Self::path().ok_or("could not determine config directory")?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create config directory: {e}"))?;
        }
        let contents =
            toml::to_string_pretty(self).map_err(|e| format!("failed to serialize config: {e}"))?;
        fs::write(&path, contents).map_err(|e| format!("failed to write config: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tinfoil_verifier::TdxMeasurement;

    #[test]
    fn trusted_measurements_round_trip_via_toml() {
        let snp = "d6848e43be21b268536059930c717abb7004279e860cbbb8f88be8a48d250d972276d936c0896bd157984bbec77d4919";
        let rtmr1 = "4f7be53273f4ed3114e7578574f98eec533d5a18484e4e8a5feef1672b4a94e17646e7ab9e1f3c722faea496bac4dc8d";
        let rtmr2 = "34cd93a0c2ea0629323c09145636a25a0ac1ead868ff9337e315fb3ce846763eb5c5c97a4927c34b24bb513e8f74db70";

        let original = Config {
            base_url: Some("https://example.com".into()),
            trusted_measurements: vec![EnclaveMeasurement {
                snp_measurement: snp.into(),
                tdx_measurement: TdxMeasurement {
                    rtmr1: rtmr1.into(),
                    rtmr2: rtmr2.into(),
                },
            }],
            ..Config::default()
        };

        let toml_text = toml::to_string_pretty(&original).expect("serialize");
        let parsed: Config = toml::from_str(&toml_text).expect("deserialize");

        assert_eq!(parsed.trusted_measurements.len(), 1);
        assert_eq!(parsed.trusted_measurements[0].snp_measurement, snp);
        assert_eq!(parsed.trusted_measurements[0].tdx_measurement.rtmr1, rtmr1);
        assert_eq!(parsed.trusted_measurements[0].tdx_measurement.rtmr2, rtmr2);
    }

    /// Hand-written TOML in the manifest shape (matches the example in the
    /// `trusted_measurements` doc comment) must deserialize correctly.
    #[test]
    fn manifest_shaped_toml_deserializes() {
        let text = r#"
base_url = "https://example.com"

[[trusted_measurements]]
snp_measurement = "aa"
tdx_measurement = { rtmr1 = "bb", rtmr2 = "cc" }
"#;
        let cfg: Config = toml::from_str(text).expect("parse");
        assert_eq!(cfg.trusted_measurements.len(), 1);
        assert_eq!(cfg.trusted_measurements[0].snp_measurement, "aa");
        assert_eq!(cfg.trusted_measurements[0].tdx_measurement.rtmr1, "bb");
        assert_eq!(cfg.trusted_measurements[0].tdx_measurement.rtmr2, "cc");
    }
}
