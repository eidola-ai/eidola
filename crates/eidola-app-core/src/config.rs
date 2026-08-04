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

/// Embedded fallback for the inference model. The `default_model` config key
/// is gone (Participants v1); this is the model the seeded "Default" space
/// template's agent participant carries, and the last-resort fallback when a
/// space or template has no resolvable agent model.
pub const DEFAULT_MODEL: &str = "gemma4-31b";

/// Well-known id of the seeded "Default" space template — the compiled-in
/// value [`Config::default_template`] resolves to when the user has set no
/// `default_template` override. Stable across installs so the seed is
/// idempotent (see `db::ensure_default_participants`) and the config resolver
/// always has a live target.
pub const DEFAULT_TEMPLATE_ID: &str = "00000000-0000-7000-8000-000000000010";

/// The base type-scale factor (`1.0` = the app's designed sizes). Applied as a
/// single multiplier over the whole type ramp — the theme's UI font size (which
/// gpui-component uses as the window `rem_size`, so every `rems()`-relative
/// measurement, line height, and padding rides it) and the prose reading size.
/// Persisted so a user's chosen size survives restarts; adjusted through the
/// GUI's View → Actual Size / Zoom In / Zoom Out.
pub const FONT_SCALE_DEFAULT: f32 = 1.0;
/// The smallest allowed [`Config::font_scale`] — a sane floor so text can't be
/// shrunk to illegibility.
pub const FONT_SCALE_MIN: f32 = 0.8;
/// The largest allowed [`Config::font_scale`] — a ceiling that keeps even the
/// prose ramp's h1 (2.5×) from blowing the window apart, while still doubling
/// the base size for low-vision users.
pub const FONT_SCALE_MAX: f32 = 2.0;
/// The discrete zoom ladder Zoom In / Zoom Out step through (ascending). Chosen
/// so each step is a perceptible but not jarring jump; `1.0` is the anchor
/// Actual Size resets to.
pub const FONT_SCALE_STEPS: &[f32] = &[0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0];

/// Clamp an arbitrary scale into the allowed range (and coerce a non-finite
/// value back to the default), so a hand-edited or corrupt config can never
/// wedge the UI at an unusable size.
pub fn clamp_font_scale(scale: f32) -> f32 {
    if !scale.is_finite() {
        return FONT_SCALE_DEFAULT;
    }
    scale.clamp(FONT_SCALE_MIN, FONT_SCALE_MAX)
}

/// The next ladder step strictly above `current` (Zoom In), saturating at
/// [`FONT_SCALE_MAX`]. Snaps to the ladder from any starting value.
pub fn font_scale_step_up(current: f32) -> f32 {
    let current = clamp_font_scale(current);
    FONT_SCALE_STEPS
        .iter()
        .copied()
        .find(|s| *s > current + 1e-3)
        .unwrap_or(FONT_SCALE_MAX)
}

/// The next ladder step strictly below `current` (Zoom Out), saturating at
/// [`FONT_SCALE_MIN`].
pub fn font_scale_step_down(current: f32) -> f32 {
    let current = clamp_font_scale(current);
    FONT_SCALE_STEPS
        .iter()
        .rev()
        .copied()
        .find(|s| *s < current - 1e-3)
        .unwrap_or(FONT_SCALE_MIN)
}

/// The day/night axis of the circadian theme: which palette family is
/// active. `System` tracks the OS light/dark appearance; `Day`/`Night` pin
/// one family; `Auto` switches on the sun — between (timezone-approximated)
/// sunrise and sunset is day, falling back to a fixed 06:00–18:00 window
/// when the timezone yields no coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppearanceSetting {
    System,
    Day,
    Night,
    #[default]
    Auto,
}

/// The time-of-day axis of the circadian theme. `On` follows the sun: the
/// palette takes on the character of the light at the current hour — bluish
/// around sunrise, neutral at solar noon/midnight, warm orange around
/// sunset — with sunrise/sunset approximated from the system timezone's
/// tzdb coordinates (clock-only fallback when the zone has none). `Off`
/// pins the character to the user's [`LightCharacter`] choice instead.
/// Values are strings (not a bool) so future variants extend rather than
/// break the config key.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeOfDayTint {
    #[default]
    On,
    Off,
}

/// The character of the light the palettes render under — the value the
/// time-of-day axis animates when it is `On`, and the user's fixed choice
/// (config key `light_character`) when it is `Off`. `Neutral` is the
/// untinted anchor palette; the other two are derived from it by the GUI's
/// tint machinery. The aliases accept the pre-rename values (`bluish` /
/// `orange`) so an existing config file keeps parsing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightCharacter {
    /// Dawn / sunrise — the cool blue cast of early light.
    #[serde(alias = "bluish")]
    Cool,
    /// Noon / midnight — the anchor palettes, no cast.
    #[default]
    Neutral,
    /// Sunset / dusk — the warm orange/red of low sun.
    #[serde(alias = "orange")]
    Warm,
}

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
/// values are exposed through resolver methods that fall back to a
/// compiled-in default when no override is set.
///
/// The eidola connection + trust bundle (base URL, trusted measurements,
/// hardware CAs) does **not** live here — it moved to the `eidola` backend
/// row (see `backends.rs` / `AppCore::eidola_trust`). Config is now purely
/// config-backed, resolvable without the database.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {
    /// The UUID of the `space_template` new spaces are instantiated from.
    /// `None` = the seeded "Default" template ([`DEFAULT_TEMPLATE_ID`]).
    /// Replaces the removed `default_model` key (Participants v1).
    #[serde(
        rename = "default_template",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub default_template_override: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub account_secret: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain_separator: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attestation_repo: Option<String>,
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
    /// Explicit path to the `llama-server` binary used for local
    /// inference. `None` = discover it (`PATH`, then the usual install
    /// locations). Unlike the `*_override` pins there is no embedded
    /// fallback value — resolution lives in
    /// `local_models::resolve_engine_path`.
    #[serde(
        rename = "llama_server_path",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub llama_server_path_override: Option<String>,
    /// Circadian theme, day/night axis. `None` = the default (`auto`).
    #[serde(
        rename = "appearance",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub appearance_override: Option<AppearanceSetting>,
    /// Circadian theme, time-of-day axis. `None` = the default (`on`).
    #[serde(
        rename = "time_of_day_tint",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub time_of_day_tint_override: Option<TimeOfDayTint>,
    /// Circadian theme, fixed light character (used when the time-of-day
    /// axis is `off`). `None` = the default (`neutral`).
    #[serde(
        rename = "light_character",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub light_character_override: Option<LightCharacter>,
    /// Base type-scale factor (`1.0` = designed sizes). `None` = the default
    /// ([`FONT_SCALE_DEFAULT`]). The resolver clamps any stored value into
    /// `[FONT_SCALE_MIN, FONT_SCALE_MAX]`.
    #[serde(
        rename = "font_scale",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub font_scale_override: Option<f32>,
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

    /// The space template new spaces are instantiated from: the user's
    /// `default_template` override if set, otherwise the seeded "Default"
    /// template ([`DEFAULT_TEMPLATE_ID`]). Callers should still verify the
    /// resolved id names a *live* (non-removed) template and fall back to the
    /// seeded default if it doesn't (the resolver can't reach the DB).
    pub fn default_template(&self) -> &str {
        self.default_template_override
            .as_deref()
            .unwrap_or(DEFAULT_TEMPLATE_ID)
    }

    /// The circadian day/night axis: the user's `appearance` override if
    /// set, otherwise `auto` (follow the sun).
    pub fn appearance(&self) -> AppearanceSetting {
        self.appearance_override.unwrap_or_default()
    }

    /// The circadian time-of-day axis: the user's `time_of_day_tint`
    /// override if set, otherwise `on`.
    pub fn time_of_day_tint(&self) -> TimeOfDayTint {
        self.time_of_day_tint_override.unwrap_or_default()
    }

    /// The fixed light character used while the time-of-day axis is `off`:
    /// the user's `light_character` override if set, otherwise `neutral`.
    pub fn light_character(&self) -> LightCharacter {
        self.light_character_override.unwrap_or_default()
    }

    /// The resolved base type-scale factor: the user's `font_scale` override
    /// (clamped into the allowed range) if set, otherwise
    /// [`FONT_SCALE_DEFAULT`].
    pub fn font_scale(&self) -> f32 {
        self.font_scale_override
            .map(clamp_font_scale)
            .unwrap_or(FONT_SCALE_DEFAULT)
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

/// Check an attestation (ATC) endpoint override — the same scheme rule the
/// `eidola` row applies to `base_url`, which this key had never had. Blank
/// is accepted here and means *clear* (see `AppCore::set_attestation_url`);
/// everything else must name an http(s) endpoint, because the value is used
/// verbatim as a URL and a typo would otherwise persist and fail at every
/// handshake instead of at the setter.
pub fn validate_attestation_url(url: &str) -> Result<(), AppError> {
    let trimmed = url.trim();
    if trimmed.is_empty() || trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        return Ok(());
    }
    Err(AppError::Config {
        message: "attestation URL must start with http:// or https://".into(),
    })
}

/// Check that a certificate value parses, without writing anything — the
/// pure half of the hardware-CA setters, exposed so a caller applying a
/// batch of trust-bundle changes can validate every input *before* it
/// applies the first one. The setters re-validate on write; this is the
/// same rule read early, not a second one.
pub fn validate_cert_pem(value: &str, field_name: &str) -> Result<(), AppError> {
    parse_cert_config(Some(value), field_name).map(|_| ())
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
    fn default_template_falls_back_to_seeded_id() {
        let cfg = Config::default();
        assert_eq!(cfg.default_template(), DEFAULT_TEMPLATE_ID);
    }

    #[test]
    fn default_template_round_trips_via_toml() {
        let original = Config {
            default_template_override: Some("00000000-0000-7000-8000-0000000000ab".into()),
            ..Config::default()
        };
        let toml_text = toml::to_string_pretty(&original).expect("serialize");
        assert!(
            toml_text.contains("default_template = \"00000000-0000-7000-8000-0000000000ab\""),
            "override must serialize under the public `default_template` key: {toml_text}"
        );
        let parsed: Config = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(
            parsed.default_template(),
            "00000000-0000-7000-8000-0000000000ab"
        );

        // Absent key → override stays None and the resolver falls back.
        let parsed: Config = toml::from_str("").expect("deserialize empty");
        assert!(parsed.default_template_override.is_none());
        assert_eq!(parsed.default_template(), DEFAULT_TEMPLATE_ID);
    }

    #[test]
    fn circadian_settings_default_and_round_trip_via_toml() {
        // Absent keys → the resolvers fall back to the defaults (`auto` —
        // follow the sun — since the local-inference wave flipped it).
        let cfg: Config = toml::from_str("").expect("deserialize empty");
        assert_eq!(cfg.appearance(), AppearanceSetting::Auto);
        assert_eq!(cfg.time_of_day_tint(), TimeOfDayTint::On);
        assert_eq!(cfg.light_character(), LightCharacter::Neutral);

        let original = Config {
            appearance_override: Some(AppearanceSetting::Day),
            time_of_day_tint_override: Some(TimeOfDayTint::Off),
            light_character_override: Some(LightCharacter::Warm),
            ..Config::default()
        };
        let toml_text = toml::to_string_pretty(&original).expect("serialize");
        assert!(
            toml_text.contains("appearance = \"day\""),
            "override must serialize under the public `appearance` key: {toml_text}"
        );
        assert!(
            toml_text.contains("time_of_day_tint = \"off\""),
            "override must serialize under the public `time_of_day_tint` key: {toml_text}"
        );
        assert!(
            toml_text.contains("light_character = \"warm\""),
            "override must serialize under the public `light_character` key: {toml_text}"
        );
        let parsed: Config = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(parsed.appearance(), AppearanceSetting::Day);
        assert_eq!(parsed.time_of_day_tint(), TimeOfDayTint::Off);
        assert_eq!(parsed.light_character(), LightCharacter::Warm);

        // Pre-rename values keep parsing (a stale config must not reset).
        let parsed: Config = toml::from_str("light_character = \"bluish\"").expect("alias parses");
        assert_eq!(parsed.light_character(), LightCharacter::Cool);
        let parsed: Config = toml::from_str("light_character = \"orange\"").expect("alias parses");
        assert_eq!(parsed.light_character(), LightCharacter::Warm);
    }

    #[test]
    fn font_scale_defaults_clamps_and_round_trips() {
        // Absent key → the resolver falls back to the default.
        let cfg: Config = toml::from_str("").expect("deserialize empty");
        assert_eq!(cfg.font_scale(), FONT_SCALE_DEFAULT);

        // A stored value round-trips under the public `font_scale` key.
        let original = Config {
            font_scale_override: Some(1.25),
            ..Config::default()
        };
        let toml_text = toml::to_string_pretty(&original).expect("serialize");
        assert!(
            toml_text.contains("font_scale = 1.25"),
            "override must serialize under `font_scale`: {toml_text}"
        );
        let parsed: Config = toml::from_str(&toml_text).expect("deserialize");
        assert_eq!(parsed.font_scale(), 1.25);

        // Out-of-range and non-finite stored values are clamped/coerced by the
        // resolver rather than trusted verbatim.
        let too_big = Config {
            font_scale_override: Some(99.0),
            ..Config::default()
        };
        assert_eq!(too_big.font_scale(), FONT_SCALE_MAX);
        let too_small = Config {
            font_scale_override: Some(0.01),
            ..Config::default()
        };
        assert_eq!(too_small.font_scale(), FONT_SCALE_MIN);
        assert_eq!(clamp_font_scale(f32::NAN), FONT_SCALE_DEFAULT);
    }

    #[test]
    fn font_scale_ladder_steps_and_saturates() {
        // Stepping up from the anchor lands on the next rung, not an arbitrary
        // delta; stepping down mirrors it.
        assert_eq!(font_scale_step_up(1.0), 1.1);
        assert_eq!(font_scale_step_down(1.0), 0.9);
        // From an off-ladder value it snaps to the nearest rung in the step
        // direction.
        assert_eq!(font_scale_step_up(1.05), 1.1);
        assert_eq!(font_scale_step_down(1.05), 1.0);
        // The ends saturate rather than walking off the ladder.
        assert_eq!(font_scale_step_up(FONT_SCALE_MAX), FONT_SCALE_MAX);
        assert_eq!(font_scale_step_down(FONT_SCALE_MIN), FONT_SCALE_MIN);
        // A full round trip up the ladder ends exactly at the ceiling.
        let mut s = FONT_SCALE_MIN;
        for _ in 0..FONT_SCALE_STEPS.len() {
            s = font_scale_step_up(s);
        }
        assert_eq!(s, FONT_SCALE_MAX);
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
}
