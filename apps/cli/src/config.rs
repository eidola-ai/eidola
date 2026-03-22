use std::fs;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

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
pub const DEFAULT_DOMAIN_SEPARATOR: &str = "ACT-v1:eidolons:inference:production:2026-03-05";

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
    /// Trusted enclave measurements (hex-encoded, SEV-SNP).
    /// When non-empty, the CLI verifies the server's Tinfoil attestation
    /// before sending any confidential requests.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trusted_measurements: Vec<String>,
    /// URL to fetch the server's attestation bundle for verification.
    /// Defaults to the Tinfoil ATC endpoint. Only used when
    /// `trusted_measurements` is non-empty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_url: Option<String>,
    /// Optional PEM-encoded SEV-SNP ARK (Root CA) to use for verification.
    /// If provided, this overrides the built-in AMD Genoa ARK.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_root_ca: Option<String>,
    /// Optional PEM-encoded SEV-SNP ASK (Intermediate CA) to use for verification.
    /// If provided, this overrides the built-in AMD Genoa ASK.
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
}

impl Config {
    pub fn path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("eidolons").join("config.toml"))
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
