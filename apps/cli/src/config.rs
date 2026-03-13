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
    /// PEM-encoded CA certificate for RA-TLS verification.
    /// Typically the dstack App CA (per-app trust anchor), not the
    /// infrastructure Root CA (which would trust all dstack apps).
    /// When set, only this CA is trusted (no public WebPKI roots).
    #[serde(alias = "root_ca", skip_serializing_if = "Option::is_none")]
    pub ca_cert: Option<String>,
    /// Trusted compose hashes for RA-TLS attestation verification.
    /// When `ca_cert` is set, the CLI always verifies the server's attestation
    /// certificate contains a compose_hash in this set. If the list is empty,
    /// no compose_hash can match and every connection is refused.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub trusted_compose_hashes: Vec<String>,
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
