//! Shared attestation template loading + rendering.
//!
//! Both the release-tool (signing side) and the client's updater
//! (verifier side) MUST agree on rendering output character-for-character.
//! This crate is the single source of truth; the on-disk templates file
//! [`releases/schema/attestation-templates.json`] is just data
//! consumed through these functions.
//!
//! ## Loading
//!
//! - Release-tool reads `releases/schema/attestation-templates.json` as
//!   committed at the *previous* release tag (the copy installed clients
//!   verify against) at sign time → use [`load_from_str`] on the
//!   `git show` output, or [`load_from_path`] for an explicit override.
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

pub mod trust_shapes;
pub use trust_shapes::{
    ArtifactManifestRef, HumanAttestationRef, PreviousRelease, ReleaseIndex, TrustConstants,
};

/// The single schema version this crate understands. It versions the
/// file *format* (the field shape these structs parse), not the claim
/// text: prose changes leave it untouched and roll out via the
/// transition rule (the release that changes templates is attested
/// under the previous release's copy — see `releases/README.md`).
/// A format change bumps this and the file's `schema_version` in the
/// same commit, and must keep the previous release's templates loadable
/// for the transition release's signing pass.
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Templates {
    pub schema_version: u32,
    pub attestant_statement_template: TemplateEntry,
    pub claims: BTreeMap<String, ClaimTemplate>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemplateEntry {
    pub template: String,
    pub sources: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
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
        let json = r#"{"schema_version":99,"attestant_statement_template":{"template":"","sources":{}},"claims":{}}"#;
        assert!(load_from_str(json).is_err());
    }

    #[test]
    fn parses_minimal_valid_templates() {
        let json = r#"{
            "schema_version": 1,
            "attestant_statement_template": {"template": "I", "sources": {}},
            "claims": {}
        }"#;
        let t = load_from_str(json).unwrap();
        assert_eq!(t.schema_version, 1);
        assert!(t.claims.is_empty());
    }

    /// The committed templates file must load through this crate exactly
    /// as the client verifier and release-tool will load it — a
    /// file/loader drift (schema_version, unknown fields) fails here
    /// instead of failing every release verification at runtime.
    ///
    /// The claim-ID list is asserted so that adding, removing, or
    /// renaming a claim is a deliberate event: update this list and
    /// follow the template-change procedure in `releases/README.md`
    /// (the release that changes templates is attested under the
    /// previous release's copy).
    #[test]
    fn committed_templates_file_loads_under_pinned_schema() {
        let json = include_str!("../../../releases/schema/attestation-templates.json");
        let templates = load_from_str(json).expect("committed templates file must load");
        assert_eq!(templates.schema_version, SUPPORTED_SCHEMA_VERSION);
        let claim_ids: Vec<&str> = templates.claims.keys().map(String::as_str).collect();
        assert_eq!(
            claim_ids,
            [
                "code_delivers_guarantees",
                "diff_reviewed",
                "manifest_reproduced",
                "no_coercion",
                "no_compelled_subversion",
                "no_gag_preventing_attestation",
                "no_known_backdoor",
                "privacy_guarantees_not_weakened",
                "signing_freely",
            ],
        );
    }
}
