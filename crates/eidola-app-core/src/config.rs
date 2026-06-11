use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tinfoil_verifier::EnclaveMeasurement;

use crate::error::AppError;

/// The expected domain separator for credential operations.
///
/// Checked against the server's advertised domain separator before issuing or
/// spending any credential. An exact match is required to prevent a malicious
/// operator from silently partitioning users into smaller anonymity sets.
pub const DEFAULT_DOMAIN_SEPARATOR: &str = "ACT-v1:eidola:inference:production:2026-03-05";

/// Default GitHub source repository the eidola server enclave is attested
/// against via the Tinfoil ATC `POST /attestation` endpoint.
pub const DEFAULT_ATTESTATION_REPO: &str = "eidola-ai/eidola";

/// Embedded fallback for the inference model used when neither the user's
/// config (`default_model`) nor the caller specifies one.
pub const DEFAULT_MODEL: &str = "gemma4-31b";

/// Returns the default config file path: `<config_dir>/eidola/config.toml`.
pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("eidola").join("config.toml"))
}

/// Returns the default data directory: `<data_dir>/eidola/`.
pub fn default_data_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("eidola"))
}

/// User-facing client config, deserialized from `config.toml`. Fields
/// prefixed with `*_override` carry the user's overrides; the resolved
/// values are exposed through `base_url()` / `trusted_measurements()`,
/// which fall back to the trust-root pin when no override is set.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(rename = "base_url", default, skip_serializing_if = "Option::is_none")]
    pub base_url_override: Option<String>,
    #[serde(
        rename = "default_model",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub default_model_override: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_secret: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_separator: Option<String>,
    #[serde(
        rename = "trusted_measurements",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub trusted_measurements_override: Vec<EnclaveMeasurement>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_repo: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_root_ca: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_intermediate_ca: Option<String>,
    /// Base URL of an alternate update feed (a GitHub-releases-API-shaped
    /// server), for dev/test fixture servers. Same `*_override` pattern as
    /// `base_url`: the resolved endpoint comes from [`Config::update_feed_url`],
    /// which falls back to the trust-root pin.
    #[serde(
        rename = "update_feed",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub update_feed_override: Option<String>,
}

impl Config {
    /// Returns the domain separator to enforce, falling back to the
    /// compiled-in default.
    pub fn domain_separator(&self) -> &str {
        self.domain_separator
            .as_deref()
            .unwrap_or(DEFAULT_DOMAIN_SEPARATOR)
    }

    /// Returns the source repo to attest the upstream enclave against.
    pub fn attestation_repo(&self) -> &str {
        self.attestation_repo
            .as_deref()
            .unwrap_or(DEFAULT_ATTESTATION_REPO)
    }

    /// The inference model to use when the caller doesn't specify one: the
    /// user's `default_model` override if set, otherwise the embedded
    /// fallback ([`DEFAULT_MODEL`]).
    pub fn default_model(&self) -> &str {
        self.default_model_override
            .as_deref()
            .unwrap_or(DEFAULT_MODEL)
    }

    /// The server URL to talk to: the user's `base_url` override if set,
    /// otherwise the trust-root pin baked into this binary.
    pub fn base_url(&self) -> &str {
        self.base_url_override
            .as_deref()
            .unwrap_or(crate::trust_root::SERVER_URL)
    }

    /// The full URL of the latest-release endpoint the update checker
    /// polls: `<update_feed override>/releases/latest` when the override is
    /// set, otherwise the trust-root pin (`UPDATE_DISCOVERY_URL`, the
    /// GitHub `releases/latest` API). The override is a *base* URL so a
    /// dev/test fixture server mounts the same `/releases/latest` path the
    /// real API serves.
    pub fn update_feed_url(&self) -> String {
        match self.update_feed_override.as_deref() {
            Some(base) => format!("{}/releases/latest", base.trim_end_matches('/')),
            None => crate::trust_root::UPDATE_DISCOVERY_URL.to_string(),
        }
    }

    /// The set of enclave measurements the client will accept on TLS
    /// handshake: the user's `trusted_measurements` override list if any,
    /// otherwise the single pinned server measurement.
    pub fn trusted_measurements(&self) -> Vec<EnclaveMeasurement> {
        if self.trusted_measurements_override.is_empty() {
            vec![crate::trust_root::server_measurement()]
        } else {
            self.trusted_measurements_override.clone()
        }
    }

    /// Load config from `path`, returning defaults if the file is missing or
    /// unparseable.
    pub fn load_from(path: &Path) -> Config {
        let Ok(contents) = fs::read_to_string(path) else {
            return Config::default();
        };
        toml::from_str(&contents).unwrap_or_default()
    }

    /// Serialize and write the config to `path`, creating parent directories
    /// as needed.
    pub fn save_to(&self, path: &Path) -> Result<(), AppError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| AppError::Config {
                message: format!("failed to create config directory: {e}"),
            })?;
        }
        let contents = toml::to_string_pretty(self).map_err(|e| AppError::Config {
            message: format!("failed to serialize config: {e}"),
        })?;
        fs::write(path, contents).map_err(|e| AppError::Config {
            message: format!("failed to write config: {e}"),
        })
    }
}

// ---------------------------------------------------------------------------
// Measurement parsing helpers
// ---------------------------------------------------------------------------

/// Parse a `<snp>:<rtmr1>:<rtmr2>` trust spec into an [`EnclaveMeasurement`].
pub fn parse_trust_measurement(spec: &str) -> Result<EnclaveMeasurement, AppError> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 3 {
        return Err(AppError::Config {
            message: "trust_measurement must be `<snp>:<rtmr1>:<rtmr2>` \
                      (three colon-separated 96-char hex strings)"
                .into(),
        });
    }
    let snp = validate_hex_field(parts[0], "snp_measurement")?;
    let rtmr1 = validate_hex_field(parts[1], "tdx.rtmr1")?;
    let rtmr2 = validate_hex_field(parts[2], "tdx.rtmr2")?;
    Ok(EnclaveMeasurement {
        snp_measurement: snp,
        tdx_measurement: tinfoil_verifier::TdxMeasurement { rtmr1, rtmr2 },
    })
}

/// Extract the SNP measurement key from an `--untrust_measurement` argument.
/// Accepts either a bare 96-char SNP measurement or the full
/// `<snp>:<rtmr1>:<rtmr2>` triple.
pub fn parse_untrust_key(spec: &str) -> Result<String, AppError> {
    let snp = match spec.split_once(':') {
        Some((snp, _)) => snp,
        None => spec,
    };
    validate_hex_field(snp, "snp_measurement")
}

fn validate_hex_field(value: &str, field: &str) -> Result<String, AppError> {
    if value.len() != 96 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(AppError::Config {
            message: format!("{field} must be 96 hex characters (48 bytes)"),
        });
    }
    Ok(value.to_ascii_lowercase())
}

/// Parse a PEM or raw base64 DER certificate from a config value.
pub(crate) fn parse_cert_config(
    value: Option<&str>,
    field_name: &str,
) -> Result<Option<Vec<u8>>, AppError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.starts_with("-----BEGIN") {
        use der::DecodePem;
        let cert = x509_cert::Certificate::from_pem(trimmed).map_err(|e| AppError::Config {
            message: format!("failed to parse {field_name} PEM: {e}"),
        })?;
        Ok(Some(der::Encode::to_der(&cert).map_err(|e| {
            AppError::Config {
                message: format!("failed to encode {field_name}: {e}"),
            }
        })?))
    } else {
        use base64::Engine;
        let b64: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
        Ok(Some(
            base64::engine::general_purpose::STANDARD
                .decode(&b64)
                .map_err(|e| AppError::Config {
                    message: format!("failed to decode {field_name} base64: {e}"),
                })?,
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
            base_url_override: Some("https://example.com".into()),
            trusted_measurements_override: vec![EnclaveMeasurement {
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

        assert_eq!(
            parsed.base_url_override.as_deref(),
            Some("https://example.com")
        );
        assert_eq!(parsed.trusted_measurements_override.len(), 1);
        assert_eq!(parsed.trusted_measurements_override[0].snp_measurement, snp);
        assert_eq!(
            parsed.trusted_measurements_override[0]
                .tdx_measurement
                .rtmr1,
            rtmr1
        );
        assert_eq!(
            parsed.trusted_measurements_override[0]
                .tdx_measurement
                .rtmr2,
            rtmr2
        );
    }

    #[test]
    fn manifest_shaped_toml_deserializes() {
        let text = r#"
base_url = "https://example.com"

[[trusted_measurements]]
snp_measurement = "aa"
tdx_measurement = { rtmr1 = "bb", rtmr2 = "cc" }
"#;
        let cfg: Config = toml::from_str(text).expect("parse");
        assert_eq!(
            cfg.base_url_override.as_deref(),
            Some("https://example.com")
        );
        assert_eq!(cfg.trusted_measurements_override.len(), 1);
        assert_eq!(cfg.trusted_measurements_override[0].snp_measurement, "aa");
        assert_eq!(
            cfg.trusted_measurements_override[0].tdx_measurement.rtmr1,
            "bb"
        );
        assert_eq!(
            cfg.trusted_measurements_override[0].tdx_measurement.rtmr2,
            "cc"
        );
    }

    #[test]
    fn parse_trust_measurement_valid() {
        let snp = "a".repeat(96);
        let rtmr1 = "b".repeat(96);
        let rtmr2 = "c".repeat(96);
        let spec = format!("{snp}:{rtmr1}:{rtmr2}");
        let m = parse_trust_measurement(&spec).unwrap();
        assert_eq!(m.snp_measurement, snp);
        assert_eq!(m.tdx_measurement.rtmr1, rtmr1);
        assert_eq!(m.tdx_measurement.rtmr2, rtmr2);
    }

    #[test]
    fn parse_trust_measurement_rejects_bad_length() {
        assert!(parse_trust_measurement("aa:bb:cc").is_err());
    }

    #[test]
    fn parse_untrust_key_bare_and_triple() {
        let snp = "a".repeat(96);
        assert_eq!(parse_untrust_key(&snp).unwrap(), snp);
        let triple = format!("{}:{}:{}", snp, "b".repeat(96), "c".repeat(96));
        assert_eq!(parse_untrust_key(&triple).unwrap(), snp);
    }

    #[test]
    fn default_model_falls_back_to_embedded_value() {
        let cfg = Config::default();
        assert_eq!(cfg.default_model(), DEFAULT_MODEL);
    }

    #[test]
    fn default_model_round_trips_via_toml() {
        let original = Config {
            default_model_override: Some("kimi-k2-6".into()),
            ..Config::default()
        };
        let toml_text = toml::to_string_pretty(&original).expect("serialize");
        assert!(
            toml_text.contains("default_model = \"kimi-k2-6\""),
            "override must serialize under the public `default_model` key: {toml_text}"
        );
        let parsed: Config = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(parsed.default_model(), "kimi-k2-6");

        // Absent key → override stays None and the resolver falls back.
        let parsed: Config = toml::from_str("").expect("deserialize empty");
        assert!(parsed.default_model_override.is_none());
        assert_eq!(parsed.default_model(), DEFAULT_MODEL);
    }

    #[test]
    fn defaults_fall_back_to_trust_root_pin() {
        let cfg = Config::default();
        assert_eq!(cfg.base_url(), crate::trust_root::SERVER_URL);
        let measurements = cfg.trusted_measurements();
        assert_eq!(measurements.len(), 1);
        assert_eq!(
            measurements[0].snp_measurement,
            crate::trust_root::SERVER_SNP_MEASUREMENT
        );
    }

    #[test]
    fn update_feed_url_resolves_override_and_pin() {
        let cfg = Config::default();
        assert_eq!(
            cfg.update_feed_url(),
            crate::trust_root::UPDATE_DISCOVERY_URL
        );

        let cfg = Config {
            update_feed_override: Some("http://127.0.0.1:9999/".into()),
            ..Config::default()
        };
        assert_eq!(
            cfg.update_feed_url(),
            "http://127.0.0.1:9999/releases/latest"
        );
    }

    #[test]
    fn update_feed_override_round_trips_via_toml() {
        let cfg = Config {
            update_feed_override: Some("http://localhost:8123".into()),
            ..Config::default()
        };
        let toml_text = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(toml_text.contains("update_feed"));
        let parsed: Config = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(
            parsed.update_feed_override.as_deref(),
            Some("http://localhost:8123")
        );
    }

    #[test]
    fn overrides_are_preferred() {
        let cfg = Config {
            base_url_override: Some("https://override.example".into()),
            trusted_measurements_override: vec![EnclaveMeasurement {
                snp_measurement: "a".repeat(96),
                tdx_measurement: TdxMeasurement {
                    rtmr1: "b".repeat(96),
                    rtmr2: "c".repeat(96),
                },
            }],
            ..Config::default()
        };
        assert_eq!(cfg.base_url(), "https://override.example");
        let measurements = cfg.trusted_measurements();
        assert_eq!(measurements.len(), 1);
        assert_eq!(measurements[0].snp_measurement, "a".repeat(96));
    }
}
