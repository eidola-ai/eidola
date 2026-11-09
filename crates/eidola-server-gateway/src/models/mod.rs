//! The model catalog: one row per set of weights, keyed by the weights hash,
//! with tags as movable names for a hash.
//!
//! The rows are the hand-written files under `models/` at the workspace root
//! (`models/<family>/<label>.toml`), checked and rendered as Rust by
//! `build.rs` ([`file`]) and compiled in as [`MODEL_CATALOG`]. The build also
//! binds them to the engine pins (`releases/trust/engine-enclaves.json`) one
//! to one by hash, so a gateway that builds serves exactly the rows it pins.
//!
//! **The natural key of a model is its weights hash** (the SHA-256 the
//! engine computes over every file it reads from its weights directory;
//! `docs/model-identity.md`). A tag is a name that resolves to exactly one
//! hash in a given build, and moves only with a release. A client names a
//! model by [`ModelReference`]: `tag` (follow the tag), `tag@sha256:<hex>`
//! (the tag, asserted to resolve to that hash) or `sha256:<hex>` (the hash
//! alone). The gateway resolves the reference to a hash at the edge, and
//! everything after it (routing, placement, billing, logs, telemetry) sees
//! only the hash.

#[cfg(test)]
pub(crate) mod file;

use eidola_common::architecture::Architecture;

use crate::types::{Modality, OutputBudgetClass};

include!(concat!(env!("OUT_DIR"), "/model_catalog.gen.rs"));

/// One catalog row: a set of weights, what they can do, and what we charge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CatalogRow {
    /// The weights hash, lowercase hex: the row's key.
    pub sha256: &'static str,
    pub architecture: Architecture,
    /// The names that resolve to this hash. Each tag names exactly one row.
    pub tags: &'static [&'static str],
    /// The source repository, `<owner>/<name>`.
    pub repo: &'static str,
    /// The source repository's revision (40 hex digits).
    pub revision: &'static str,
    /// Every file the weights hash covers, sorted by path.
    pub files: &'static [WeightsFile],
    pub name: &'static str,
    pub description: &'static str,
    pub context_length: u64,
    /// Whether the model accepts a `tools` request field.
    pub tool_calling: bool,
    /// Whether the model produces reasoning content before its answer.
    pub reasoning: bool,
    pub input_modalities: &'static [Modality],
    pub output_modalities: &'static [Modality],
    /// The largest completion we are prepared to ask this model for.
    ///
    /// **Declared by us, not transcribed.** It is a ceiling we commit to: a
    /// length the model has been seen to reach comfortably, kept well inside
    /// its context window so a long prompt still leaves room to answer.
    /// Budget ladders live *under* this number; none may cross it.
    pub max_output_tokens: Option<u64>,
    /// Which public output-budget ladder the model draws from. Declared
    /// rather than derived from `reasoning`, so a model can be moved between
    /// ladders without the catalog having to lie about what it does.
    pub output_budget_class: OutputBudgetClass,
    /// The base price, USD before the markup. The variant is the row's
    /// pricing shape, whatever the amounts (a zero price keeps its shape).
    pub price: RowPrice,
    pub lifecycle: Lifecycle,
}

/// A row's base price, USD before the markup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowPrice {
    PerToken { input_per_m: f64, output_per_m: f64 },
    PerRequest { usd: f64 },
}

/// One file the weights hash covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightsFile {
    /// The file's name in the weights directory (and in the source
    /// repository at its revision).
    pub path: &'static str,
    pub size: u64,
    /// Lowercase hex.
    pub sha256: &'static str,
}

/// Whether a row is served.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    /// Served.
    Current,
    /// Served, with its retirement announced for `retire_after`
    /// (`YYYY-MM-DD`). A release retires it; the date is a notice, not a
    /// timer.
    Deprecated { retire_after: &'static str },
    /// Not served and not pinned. A reference to it is refused as retired.
    Retired { retire_after: Option<&'static str> },
}

impl Lifecycle {
    pub const fn is_retired(&self) -> bool {
        matches!(self, Lifecycle::Retired { .. })
    }
}

impl CatalogRow {
    /// The reference that names this row and nothing else: its first tag
    /// pinned to its hash, or the hash alone when it has no tag.
    pub fn pinned_reference(&self) -> String {
        match self.tags.first() {
            Some(tag) => format!("{tag}@sha256:{}", self.sha256),
            None => format!("sha256:{}", self.sha256),
        }
    }

    /// The id `/v1/models` lists the row under: its first tag (which
    /// follows), or `sha256:<hex>` when it has none. Never holds `@`, so a
    /// client that appends `@<backend>` to an id still parses it.
    pub fn list_id(&self) -> String {
        match self.tags.first() {
            Some(tag) => (*tag).to_string(),
            None => format!("sha256:{}", self.sha256),
        }
    }
}

/// A client's `model` field, parsed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelReference<'a> {
    /// `tag`: whatever the tag resolves to.
    Follow { tag: &'a str },
    /// `tag@sha256:<hex>`: the tag, which must resolve to `sha256`.
    Pin { tag: &'a str, sha256: &'a str },
    /// `sha256:<hex>`: the weights, under whatever tags.
    Hash { sha256: &'a str },
}

/// The prefix of a digest in a reference.
pub const DIGEST_PREFIX: &str = "sha256:";

impl<'a> ModelReference<'a> {
    /// Parse a reference. Only full digests are accepted: 64 lowercase hex
    /// digits after `sha256:`.
    pub fn parse(text: &'a str) -> Result<Self, ModelRefError> {
        let digest = |d: &'a str| -> Result<&'a str, ModelRefError> {
            d.strip_prefix(DIGEST_PREFIX)
                .filter(|hex| eidola_common::weights_manifest::is_lower_hex_sha256(hex))
                .ok_or(ModelRefError::Malformed)
        };
        if text.starts_with(DIGEST_PREFIX) {
            return Ok(ModelReference::Hash {
                sha256: digest(text)?,
            });
        }
        let (tag, pinned) = match text.split_once('@') {
            Some((tag, rest)) => (tag, Some(digest(rest)?)),
            None => (text, None),
        };
        if !is_tag(tag) {
            return Err(ModelRefError::Malformed);
        }
        Ok(match pinned {
            Some(sha256) => ModelReference::Pin { tag, sha256 },
            None => ModelReference::Follow { tag },
        })
    }
}

/// The tag grammar (`file::is_tag`, the rule the build checks rows with).
pub fn is_tag(tag: &str) -> bool {
    !tag.is_empty()
        && tag.len() <= MAX_TAG_LEN
        && tag
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The longest tag accepted (`file::MAX_TAG_LEN`).
pub const MAX_TAG_LEN: usize = 128;

/// Why a reference was refused. Every message is fixed text, never the
/// reference itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelRefError {
    /// Not `tag`, `tag@sha256:<hex>` or `sha256:<hex>`.
    Malformed,
    /// No row has that tag or hash.
    Unknown,
    /// The row is retired.
    Retired,
    /// `tag@sha256:<hex>` where the tag resolves to another hash.
    Mismatch,
}

impl ModelRefError {
    /// The HTTP status.
    pub fn status(self) -> u16 {
        match self {
            ModelRefError::Malformed => 400,
            ModelRefError::Unknown => 404,
            ModelRefError::Mismatch => 409,
            ModelRefError::Retired => 410,
        }
    }

    /// The OpenAI-style error type.
    pub fn error_type(self) -> &'static str {
        match self {
            ModelRefError::Malformed => "invalid_model_reference",
            ModelRefError::Unknown => "model_not_found",
            ModelRefError::Mismatch => "model_reference_mismatch",
            ModelRefError::Retired => "model_retired",
        }
    }

    /// The message: fixed text.
    pub fn message(self) -> &'static str {
        match self {
            ModelRefError::Malformed => {
                "model must be a tag, tag@sha256:<64 lowercase hex digits>, or \
                 sha256:<64 lowercase hex digits>"
            }
            ModelRefError::Unknown => "unknown model",
            ModelRefError::Mismatch => {
                "the tag does not resolve to that hash; name the hash alone (sha256:<hex>) to \
                 keep these weights, or the tag alone to follow it"
            }
            ModelRefError::Retired => "this model is retired",
        }
    }
}

impl std::fmt::Display for ModelRefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// Resolve `reference` against `rows`: the row it names and the reference
/// a response echoes (`tag@sha256:<hex>`, or `sha256:<hex>` for a hash
/// named alone).
///
/// The rule: a tag resolves to the one row listing it; a pin is accepted
/// only when its tag resolves to exactly that hash **now**; a hash is
/// accepted when a row has it. In every case a retired row is refused. A
/// client that wants specific weights to survive a tag moving names the hash
/// alone.
pub fn resolve<'r>(
    rows: &'r [CatalogRow],
    reference: &str,
) -> Result<(&'r CatalogRow, String), ModelRefError> {
    let by_tag = |tag: &str| rows.iter().find(|r| r.tags.contains(&tag));
    let (row, echo) = match ModelReference::parse(reference)? {
        ModelReference::Follow { tag } => {
            let row = by_tag(tag).ok_or(ModelRefError::Unknown)?;
            (row, format!("{tag}@sha256:{}", row.sha256))
        }
        ModelReference::Pin { tag, sha256 } => {
            let row = by_tag(tag).ok_or(ModelRefError::Unknown)?;
            if row.sha256 != sha256 {
                return Err(if row.lifecycle.is_retired() {
                    ModelRefError::Retired
                } else {
                    ModelRefError::Mismatch
                });
            }
            (row, format!("{tag}@sha256:{sha256}"))
        }
        ModelReference::Hash { sha256 } => {
            let row = rows
                .iter()
                .find(|r| r.sha256 == sha256)
                .ok_or(ModelRefError::Unknown)?;
            (row, format!("{DIGEST_PREFIX}{sha256}"))
        }
    };
    if row.lifecycle.is_retired() {
        return Err(ModelRefError::Retired);
    }
    Ok((row, echo))
}

#[cfg(test)]
mod tests;
