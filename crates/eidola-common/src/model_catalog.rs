//! The model catalog's source files (`models/<family>/<label>.toml` at the
//! workspace root), parsed and checked: the one reader both the gateway's
//! build (which compiles the catalog in and binds it to its engine pins) and
//! the client's build (which compiles the same catalog in, to tell a user
//! what their model choice resolves to) run, so the two cannot read a file
//! differently.
//!
//! Behind this crate's `catalog` feature, which adds `toml`, `sha2` and
//! `serde` only to the consumers that enable it (the gateway's and the
//! client's build scripts and tests), each of which already depends on them.
//!
//! # What is checked
//!
//! For each file ([`parse`]):
//!
//! - its shape, strictly (unknown members refused at every level; TOML
//!   itself refuses a repeated key);
//! - `sha256` is 64 lowercase hex digits, and **equals the SHA-256 of the
//!   canonical manifest of `files`** ([`crate::weights_manifest`]): every
//!   file the engine's weights hash covers is listed with its size and
//!   SHA-256, and nothing else;
//! - `architecture` names an [`Architecture`] that is **implemented**;
//! - every tag is a tag ([`crate::model_reference::is_tag`]: `[A-Za-z0-9._-]+`,
//!   at most 64 bytes, so never `@` or `:`, the reference syntax's
//!   separators), and none is listed twice;
//! - the source is `<owner>/<name>` at a 40-hex-digit revision;
//! - display and capabilities are complete (a context window, a ceiling
//!   inside it, text in and text out, no repeated modality);
//! - the base price is finite and non-negative, per token or per request; a
//!   per-token row may also state a cached-input price, no higher than its
//!   input price, and **a token-priced row declares `max_output_tokens`**:
//!   it is the top of the ladder of completion ceilings a request is held
//!   at, so a row without one would be held and served at its whole
//!   context window;
//! - the lifecycle is consistent: `deprecated` carries `retire_after`
//!   (`YYYY-MM-DD`), `current` does not.
//!
//! Across files ([`check_catalog`]): no weights hash is listed twice, and
//! **each tag names exactly one hash**.
//!
//! Against the engine pins ([`check_binding`], the gateway's): every row
//! that is not retired has a pin for its hash, with the same architecture
//! and source, every pin has such a row, and no retired row is pinned.

use crate::architecture::Architecture;
use crate::weights_manifest;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Where the catalog files live, relative to the workspace root.
pub const MODELS_ROOT: &str = "models";

pub use crate::model_reference::{MAX_TAG_LEN, is_tag};

/// One checked catalog file.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The file's path, relative to the workspace root.
    pub path: String,
    pub sha256: String,
    pub architecture: Architecture,
    pub tags: Vec<String>,
    pub repo: String,
    pub revision: String,
    /// Sorted by path, as the manifest is.
    pub files: Vec<WeightsFile>,
    pub name: String,
    pub description: String,
    pub context_length: u64,
    pub tool_calling: bool,
    pub reasoning: bool,
    /// `text`, `image` or `audio`.
    pub input_modalities: Vec<String>,
    pub output_modalities: Vec<String>,
    pub max_output_tokens: Option<u64>,
    /// `standard` or `reasoning`.
    pub output_budget_class: String,
    pub price: Price,
    /// What a prompt token declared cached costs, USD per million before the
    /// markup, where the row sells one: only beside a per-token price, and no
    /// higher than its input price. `None` means a request to this model
    /// cannot declare a cached prefix.
    pub cached_input_per_m: Option<f64>,
    pub lifecycle: Lifecycle,
}

/// One file the weights hash covers.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightsFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// A row's base price, USD before the markup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Price {
    PerToken { input_per_m: f64, output_per_m: f64 },
    PerRequest { usd: f64 },
}

/// Whether a row is served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lifecycle {
    /// Served, and what its tags are for.
    Current,
    /// Served until a release retires it, announced for `retire_after`.
    Deprecated { retire_after: String },
    /// Not served and not pinned; kept so a reference to it is answered as
    /// retired rather than unknown.
    Retired { retire_after: Option<String> },
}

impl Lifecycle {
    pub fn is_retired(&self) -> bool {
        matches!(self, Lifecycle::Retired { .. })
    }
}

// The file's typed shape. Every level refuses unknown members.

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    sha256: String,
    architecture: String,
    tags: Vec<String>,
    name: String,
    description: String,
    source: RawSource,
    files: Vec<WeightsFile>,
    capabilities: RawCapabilities,
    price: RawPrice,
    lifecycle: RawLifecycle,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSource {
    repo: String,
    revision: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCapabilities {
    context_length: u64,
    tool_calling: bool,
    reasoning: bool,
    input_modalities: Vec<String>,
    output_modalities: Vec<String>,
    max_output_tokens: Option<u64>,
    output_budget_class: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPrice {
    input_per_m: Option<f64>,
    output_per_m: Option<f64>,
    cached_input_per_m: Option<f64>,
    per_request: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawLifecycle {
    status: String,
    retire_after: Option<String>,
}

const MODALITIES: [&str; 3] = ["text", "image", "audio"];
const BUDGET_CLASSES: [&str; 2] = ["standard", "reasoning"];

/// Whether `path` is a catalog file's path: `models/<family>/<label>.toml`,
/// each component `[a-z0-9][a-z0-9._-]*`.
pub fn is_catalog_path(path: &str) -> bool {
    let Some(rest) = path
        .strip_prefix(MODELS_ROOT)
        .and_then(|r| r.strip_prefix('/'))
    else {
        return false;
    };
    let Some((family, file)) = rest.split_once('/') else {
        return false;
    };
    let Some(label) = file.strip_suffix(".toml") else {
        return false;
    };
    is_component(family) && is_component(label)
}

fn is_component(s: &str) -> bool {
    let mut bytes = s.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && bytes.all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Parse and check one catalog file (`path`, relative to the workspace root).
pub fn parse(path: &str, bytes: &[u8]) -> Result<Row, String> {
    if !is_catalog_path(path) {
        return Err(format!(
            "{path}: a catalog file is {MODELS_ROOT}/<family>/<label>.toml, each component \
             matching [a-z0-9][a-z0-9._-]*"
        ));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    let raw: RawFile = toml::from_str(text).map_err(|e| format!("{path}: {e}"))?;
    let at = |m: &str| format!("{path}: {m}");

    if !weights_manifest::is_lower_hex_sha256(&raw.sha256) {
        return Err(at("sha256 must be 64 lowercase hex digits"));
    }
    let architecture = Architecture::from_name(&raw.architecture).ok_or_else(|| {
        at(&format!(
            "architecture {:?} is not one the engine knows",
            raw.architecture
        ))
    })?;
    if !architecture.is_implemented() {
        return Err(at(&format!(
            "architecture {architecture} is not implemented; no catalog row may name it"
        )));
    }

    for (i, tag) in raw.tags.iter().enumerate() {
        if !is_tag(tag) {
            return Err(at(&format!(
                "tag {tag:?} must match [A-Za-z0-9._-]+ (at most {MAX_TAG_LEN} bytes)"
            )));
        }
        if raw.tags[..i].contains(tag) {
            return Err(at(&format!("tag {tag:?} is listed twice")));
        }
    }

    if !is_repo(&raw.source.repo) {
        return Err(at("source.repo must be <owner>/<name> of [A-Za-z0-9._-]"));
    }
    if !(raw.source.revision.len() == 40
        && raw
            .source
            .revision
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
    {
        return Err(at("source.revision must be 40 lowercase hex digits"));
    }

    // The per-file manifest, and its aggregate.
    let manifest = weights_manifest::canonical_manifest(
        raw.files
            .iter()
            .map(|f| (f.path.as_str(), f.sha256.as_str())),
    )
    .map_err(|e| at(&format!("files: {e}")))?;
    let aggregate = hex(&Sha256::digest(manifest.as_bytes()));
    if aggregate != raw.sha256 {
        return Err(at(&format!(
            "sha256 is {}, but the manifest of files hashes to {aggregate}",
            raw.sha256
        )));
    }
    let mut files = raw.files;
    files.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));

    if raw.name.trim().is_empty() || raw.description.trim().is_empty() {
        return Err(at("name and description must not be empty"));
    }
    let caps = raw.capabilities;
    if caps.context_length == 0 {
        return Err(at("capabilities.context_length must be positive"));
    }
    if let Some(ceiling) = caps.max_output_tokens
        && (ceiling == 0 || ceiling >= caps.context_length)
    {
        return Err(at(
            "capabilities.max_output_tokens must be positive and less than context_length, so a \
             prompt still fits",
        ));
    }
    for (field, list) in [
        ("input_modalities", &caps.input_modalities),
        ("output_modalities", &caps.output_modalities),
    ] {
        for (i, m) in list.iter().enumerate() {
            if !MODALITIES.contains(&m.as_str()) {
                return Err(at(&format!(
                    "capabilities.{field}: {m:?} is not one of {MODALITIES:?}"
                )));
            }
            if list[..i].contains(m) {
                return Err(at(&format!("capabilities.{field}: {m:?} is listed twice")));
            }
        }
        if !list.iter().any(|m| m == "text") {
            return Err(at(&format!(
                "capabilities.{field} must include text: every row is a chat model"
            )));
        }
    }
    if !BUDGET_CLASSES.contains(&caps.output_budget_class.as_str()) {
        return Err(at(&format!(
            "capabilities.output_budget_class must be one of {BUDGET_CLASSES:?}"
        )));
    }

    let usable = |v: f64| v.is_finite() && v >= 0.0;
    let price = match (
        raw.price.input_per_m,
        raw.price.output_per_m,
        raw.price.per_request,
    ) {
        (Some(input), Some(output), None) if usable(input) && usable(output) => Price::PerToken {
            input_per_m: input,
            output_per_m: output,
        },
        (None, None, Some(usd)) if usable(usd) => Price::PerRequest { usd },
        _ => {
            return Err(at(
                "price is input_per_m and output_per_m, or per_request alone, each finite and \
                 non-negative (USD before the markup)",
            ));
        }
    };
    let cached_input_per_m = match (raw.price.cached_input_per_m, price) {
        (None, _) => None,
        (Some(cached), Price::PerToken { input_per_m, .. })
            if usable(cached) && cached <= input_per_m =>
        {
            Some(cached)
        }
        _ => {
            return Err(at(
                "price.cached_input_per_m belongs to a per-token price, finite, non-negative \
                 and no higher than input_per_m",
            ));
        }
    };
    if matches!(price, Price::PerToken { .. }) && caps.max_output_tokens.is_none() {
        return Err(at(
            "a token-priced row declares capabilities.max_output_tokens: it bounds the \
             completion a request is held and served for",
        ));
    }

    let lifecycle = match (raw.lifecycle.status.as_str(), raw.lifecycle.retire_after) {
        ("current", None) => Lifecycle::Current,
        ("current", Some(_)) => {
            return Err(at(
                "lifecycle.retire_after is for a deprecated or retired row",
            ));
        }
        ("deprecated", Some(date)) if is_date(&date) => {
            Lifecycle::Deprecated { retire_after: date }
        }
        ("deprecated", _) => {
            return Err(at(
                "a deprecated row states lifecycle.retire_after as YYYY-MM-DD",
            ));
        }
        ("retired", None) => Lifecycle::Retired { retire_after: None },
        ("retired", Some(date)) if is_date(&date) => Lifecycle::Retired {
            retire_after: Some(date),
        },
        ("retired", Some(_)) => return Err(at("lifecycle.retire_after must be YYYY-MM-DD")),
        _ => {
            return Err(at(
                "lifecycle.status must be current, deprecated or retired",
            ));
        }
    };

    Ok(Row {
        path: path.to_owned(),
        sha256: raw.sha256,
        architecture,
        tags: raw.tags,
        repo: raw.source.repo,
        revision: raw.source.revision,
        files,
        name: raw.name,
        description: raw.description,
        context_length: caps.context_length,
        tool_calling: caps.tool_calling,
        reasoning: caps.reasoning,
        input_modalities: caps.input_modalities,
        output_modalities: caps.output_modalities,
        max_output_tokens: caps.max_output_tokens,
        output_budget_class: caps.output_budget_class,
        price,
        cached_input_per_m,
        lifecycle,
    })
}

/// The rules across files: no hash is listed twice, and each tag names
/// exactly one hash.
pub fn check_catalog(rows: &[Row]) -> Result<(), String> {
    for (i, row) in rows.iter().enumerate() {
        for earlier in &rows[..i] {
            if earlier.sha256 == row.sha256 {
                return Err(format!(
                    "{} and {} both describe weights {}; a model is listed once",
                    earlier.path, row.path, row.sha256
                ));
            }
            if let Some(tag) = row.tags.iter().find(|t| earlier.tags.contains(t)) {
                return Err(format!(
                    "tag {tag:?} names both {} ({}) and {} ({}); a tag names exactly one hash",
                    earlier.sha256, earlier.path, row.sha256, row.path
                ));
            }
        }
    }
    Ok(())
}

/// What the binding reads from one engine pin.
#[derive(Debug, Clone, Copy)]
pub struct PinView<'a> {
    pub sha256: &'a str,
    pub architecture: Architecture,
    pub repo: &'a str,
    pub revision: &'a str,
}

/// The rows and the engine pins bind one to one: every row that is served
/// (current or deprecated) has the pin for its hash, agreeing on the
/// architecture and the source; every pin has a served row; no retired row
/// is pinned.
pub fn check_binding(rows: &[Row], pins: &[PinView<'_>]) -> Result<(), String> {
    for row in rows {
        let pin = pins.iter().find(|p| p.sha256 == row.sha256);
        match (row.lifecycle.is_retired(), pin) {
            (true, Some(_)) => {
                return Err(format!(
                    "{} is retired, but releases/trust/engine-enclaves.json still pins {}",
                    row.path, row.sha256
                ));
            }
            (true, None) => {}
            (false, None) => {
                return Err(format!(
                    "{} is served, but releases/trust/engine-enclaves.json pins no deployment of \
                     {}",
                    row.path, row.sha256
                ));
            }
            (false, Some(pin)) => {
                if pin.architecture != row.architecture {
                    return Err(format!(
                        "{} is {}, but its deployments' deployment.json says {}",
                        row.path, row.architecture, pin.architecture
                    ));
                }
                if pin.repo != row.repo || pin.revision != row.revision {
                    return Err(format!(
                        "{}'s source ({}@{}) differs from its deployments' ({}@{})",
                        row.path, row.repo, row.revision, pin.repo, pin.revision
                    ));
                }
            }
        }
    }
    for pin in pins {
        if !rows
            .iter()
            .any(|r| r.sha256 == pin.sha256 && !r.lifecycle.is_retired())
        {
            return Err(format!(
                "releases/trust/engine-enclaves.json pins weights {}, which no served row in \
                 {MODELS_ROOT}/ describes",
                pin.sha256
            ));
        }
    }
    Ok(())
}

/// `YYYY-MM-DD` with a month and a day that exist (leap years included).
pub fn is_date(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
        return false;
    }
    let num = |r: std::ops::Range<usize>| -> Option<u32> {
        let part = &s[r];
        part.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| part.parse().ok())
            .flatten()
    };
    let (Some(y), Some(m), Some(d)) = (num(0..4), num(5..7), num(8..10)) else {
        return false;
    };
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => return false,
    };
    y >= 2000 && (1..=days).contains(&d)
}

/// `<owner>/<name>`, each non-empty `[A-Za-z0-9._-]` and not `.` or `..`.
fn is_repo(s: &str) -> bool {
    let part = |p: &str| {
        !p.is_empty()
            && p != "."
            && p != ".."
            && p.bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    matches!(s.split_once('/'), Some((owner, name)) if part(owner) && part(name))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
