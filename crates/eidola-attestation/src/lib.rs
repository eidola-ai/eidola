//! Shared attestation template loading + rendering.
//!
//! Both the release-tool (signing side) and the client's updater
//! (verifier side) MUST agree on rendering output character-for-character.
//! This crate is the single source of truth; the on-disk templates file
//! [`releases/schema/attestation-templates-v1.0.0.json`] is just data
//! consumed through these functions.
//!
//! ## Loading
//!
//! - Release-tool reads `releases/schema/attestation-templates-v1.0.0.json`
//!   from the working tree at sign time → use [`load_from_path`].
//! - Client verifier reads the build-time-embedded
//!   `eidola_app_core::trust_root::ATTESTATION_TEMPLATES_JSON` constant →
//!   use [`load_from_str`].
//!
//! ## Rendering
//!
//! [`render`] performs literal `{placeholder}` substitution from a
//! `sources` map of dotted JSON paths. The signing side renders to produce
//! the prose that goes into the signed attestation; the verifier
//! re-renders the same template with the same sources and compares to the
//! signed `statement` byte-for-byte. Any deviation = verification fails.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

/// The single schema version this crate understands. Bumping templates
/// (adding/removing/changing a claim) requires shipping a new version
/// alongside the old, then retiring the old over an overlap window.
pub const SUPPORTED_SCHEMA_VERSION: &str = "1.0.0";

#[derive(Debug, Deserialize)]
pub struct Templates {
    pub schema_version: String,
    pub attestant_statement_template: TemplateEntry,
    pub claims: BTreeMap<String, ClaimTemplate>,
}

#[derive(Debug, Deserialize)]
pub struct TemplateEntry {
    pub template: String,
    pub sources: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct ClaimTemplate {
    pub template: String,
    pub sources: BTreeMap<String, String>,
    /// For each substituted placeholder, an optional dotted path the
    /// resolved value must also equal. The verifier walks this; the
    /// signing tool can skip it (template renders fully without these).
    pub cross_checks: BTreeMap<String, String>,
}

/// Parse a templates JSON string. Validates `schema_version`.
pub fn load_from_str(json: &str) -> Result<Templates> {
    let parsed: Templates =
        serde_json::from_str(json).context("parsing attestation templates JSON")?;
    if parsed.schema_version != SUPPORTED_SCHEMA_VERSION {
        bail!(
            "attestation-templates schema_version `{}` not supported (expected `{}`)",
            parsed.schema_version,
            SUPPORTED_SCHEMA_VERSION
        );
    }
    Ok(parsed)
}

/// Convenience: read the file at `path` and parse it.
pub fn load_from_path(path: &Path) -> Result<Templates> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let s = std::str::from_utf8(&bytes)
        .with_context(|| format!("`{}` is not valid UTF-8", path.display()))?;
    load_from_str(s)
}

/// Substitute every `{key}` placeholder in `template` with the value found
/// at `sources[key]`. Sources are dotted paths into `roots`, e.g.
/// `attestation.attestant.name` resolves
/// `roots["attestation"]["attestant"]["name"]`.
///
/// Returns the rendered string plus the resolved `{key → value}` map.
/// The verifier uses the map to populate `claim.fields` when checking
/// attestations that declare them.
pub fn render(
    template: &str,
    sources: &BTreeMap<String, String>,
    roots: &BTreeMap<&str, &Value>,
) -> Result<(String, BTreeMap<String, String>)> {
    let placeholders = extract_placeholders(template);
    let source_keys: std::collections::BTreeSet<_> = sources.keys().collect();
    let placeholder_set: std::collections::BTreeSet<_> = placeholders.iter().collect();
    if source_keys != placeholder_set {
        bail!(
            "template/sources mismatch — template has {:?}, sources declares {:?}",
            placeholder_set,
            source_keys
        );
    }

    let mut values: BTreeMap<String, String> = BTreeMap::new();
    for key in &placeholders {
        let path = sources
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("no source for `{key}`"))?;
        let value = resolve_dotted_path(path, roots)
            .with_context(|| format!("resolving `{path}` for placeholder `{{{key}}}`"))?;
        values.insert(key.clone(), value);
    }

    let mut out = template.to_string();
    for (key, val) in &values {
        out = out.replace(&format!("{{{key}}}"), val);
    }
    Ok((out, values))
}

/// Resolve a dotted path against `roots`. Public so the verifier can use
/// this for cross-checks without re-implementing path resolution.
pub fn resolve_dotted_path(path: &str, roots: &BTreeMap<&str, &Value>) -> Result<String> {
    let mut parts = path.split('.');
    let root_key = parts.next().ok_or_else(|| anyhow::anyhow!("empty path"))?;
    let mut cursor = *roots
        .get(root_key)
        .ok_or_else(|| anyhow::anyhow!("no root named `{root_key}`"))?;
    for part in parts {
        cursor = cursor
            .get(part)
            .ok_or_else(|| anyhow::anyhow!("path `{path}` not found at segment `{part}`"))?;
    }
    cursor
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("`{path}` is not a string"))
}

fn extract_placeholders(template: &str) -> Vec<String> {
    // Match `{identifier}` — alphanumeric + underscore, at least 1 char.
    // Brace-pair only; no nested or escaped braces in our templates.
    let re = Regex::new(r"\{([A-Za-z_][A-Za-z0-9_]*)\}").expect("static regex");
    re.captures_iter(template)
        .map(|c| c[1].to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn roots_with(attestation: Value) -> BTreeMap<&'static str, Value> {
        let mut m = BTreeMap::new();
        m.insert("attestation", attestation);
        m
    }

    #[test]
    fn render_substitutes_all_placeholders() {
        let template = "Hello {name}, you live in {city}.";
        let sources = BTreeMap::from([
            ("name".to_string(), "attestation.attestant.name".to_string()),
            (
                "city".to_string(),
                "attestation.attestant.jurisdiction".to_string(),
            ),
        ]);
        let attestation = json!({
            "attestant": { "name": "Mike", "jurisdiction": "California" }
        });
        let bindings = roots_with(attestation);
        let mut roots: BTreeMap<&str, &Value> = BTreeMap::new();
        for (k, v) in &bindings {
            roots.insert(k, v);
        }
        let (rendered, values) = render(template, &sources, &roots).unwrap();
        assert_eq!(rendered, "Hello Mike, you live in California.");
        assert_eq!(values["name"], "Mike");
        assert_eq!(values["city"], "California");
    }

    #[test]
    fn missing_source_is_rejected() {
        let template = "Hello {name}.";
        let sources = BTreeMap::new();
        let bindings = roots_with(json!({}));
        let mut roots: BTreeMap<&str, &Value> = BTreeMap::new();
        for (k, v) in &bindings {
            roots.insert(k, v);
        }
        assert!(render(template, &sources, &roots).is_err());
    }

    #[test]
    fn extra_source_is_rejected() {
        let template = "Hi.";
        let sources = BTreeMap::from([(
            "extra".to_string(),
            "attestation.attestant.name".to_string(),
        )]);
        let bindings = roots_with(json!({}));
        let mut roots: BTreeMap<&str, &Value> = BTreeMap::new();
        for (k, v) in &bindings {
            roots.insert(k, v);
        }
        assert!(render(template, &sources, &roots).is_err());
    }

    #[test]
    fn unresolved_path_errors_explicitly() {
        let template = "X={x}.";
        let sources = BTreeMap::from([("x".to_string(), "attestation.missing.deeper".to_string())]);
        let bindings = roots_with(json!({ "attestant": {} }));
        let mut roots: BTreeMap<&str, &Value> = BTreeMap::new();
        for (k, v) in &bindings {
            roots.insert(k, v);
        }
        let err = render(template, &sources, &roots).unwrap_err();
        assert!(format!("{err:?}").contains("missing"));
    }

    #[test]
    fn extract_placeholders_dedups() {
        let p = extract_placeholders("{a} and {a} and {b}");
        assert_eq!(p, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn schema_version_mismatch_rejected() {
        let json = r#"{"schema_version":"2.0.0","attestant_statement_template":{"template":"","sources":{}},"claims":{}}"#;
        assert!(load_from_str(json).is_err());
    }

    #[test]
    fn parses_minimal_valid_templates() {
        let json = r#"{
            "schema_version": "1.0.0",
            "attestant_statement_template": {"template": "I", "sources": {}},
            "claims": {}
        }"#;
        let t = load_from_str(json).unwrap();
        assert_eq!(t.schema_version, "1.0.0");
        assert!(t.claims.is_empty());
    }
}
