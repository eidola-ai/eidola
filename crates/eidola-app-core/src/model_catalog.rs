//! The Eidola model catalog, compiled in, and what a user's model choice
//! means against it.
//!
//! The gateway identifies a model by its **weights hash** and resolves a
//! reference (`eidola_common::model_reference`): `tag` follows the tag,
//! `tag@sha256:<hex>` pins those weights (the digest is authoritative, the
//! tag a label), `sha256:<hex>` names the weights unlabelled. The gateway
//! resolves without telling anyone that a tag has moved; that is the
//! client's job, and this module does it from the same catalog files the
//! gateway compiles in (`models/` at the workspace root, read by
//! `eidola_common::model_catalog` in `build.rs`), so the client knows
//! without asking anyone:
//!
//! - what a reference resolves to here ([`resolve`]);
//! - the **notices** a choice deserves ([`notices`]): a followed tag that now
//!   names other weights than the last answer came from, a pin whose tag has
//!   moved on to newer weights, weights that are deprecated or retired;
//! - each model's identity ([`identity`]): its hash, source and per-file
//!   manifest, which a user can check independently
//!   ([`MODEL_IDENTITY_DOC`]).
//!
//! Every function takes the rows it reads, so tests run against fixtures
//! (tags moved, weights deprecated and retired) as well as the shipped
//! catalog.

use serde::{Deserialize, Serialize};

use eidola_common::model_reference::{ModelReference, pinned};

include!(concat!(env!("OUT_DIR"), "/model_catalog.gen.rs"));

/// Where the weights hash, and how to recompute it, is specified.
pub const MODEL_IDENTITY_DOC: &str = "https://www.eidola.ai/docs/model-identity/";

/// One catalog row: a set of weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogRow {
    /// The weights hash, lowercase hex.
    pub sha256: &'static str,
    /// The names that resolve to this hash. Each names exactly one row.
    pub tags: &'static [&'static str],
    pub name: &'static str,
    /// The source repository (on Hugging Face), `<owner>/<name>`.
    pub repo: &'static str,
    /// Its revision, a 40-hex-digit commit.
    pub revision: &'static str,
    /// Every file the hash covers, sorted by path.
    pub files: &'static [WeightsFile],
    pub lifecycle: Lifecycle,
}

/// One file the weights hash covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightsFile {
    pub path: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

/// A row's lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Current,
    /// Still served; retirement announced for `retire_after` (`YYYY-MM-DD`).
    Deprecated {
        retire_after: &'static str,
    },
    /// No longer served.
    Retired {
        retire_after: Option<&'static str>,
    },
}

/// What a reference resolves to in a catalog: the row, and the pinned
/// spelling that names exactly it (the tag the reference carried, if any,
/// with the row's hash).
pub fn resolve<'r>(rows: &'r [CatalogRow], reference: &str) -> Option<(&'r CatalogRow, String)> {
    let parsed = ModelReference::parse(reference).ok()?;
    let row = match parsed.digest() {
        Some(sha256) => rows.iter().find(|r| r.sha256 == sha256)?,
        None => {
            let tag = parsed.tag()?;
            rows.iter().find(|r| r.tags.contains(&tag))?
        }
    };
    Some((row, pinned(parsed.tag(), row.sha256)))
}

/// Something a user should be told about their model choice.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelNotice {
    /// A followed tag now names other weights than the last answer came
    /// from: the model was updated by a release.
    Updated {
        tag: String,
        previous_sha256: String,
        current_sha256: String,
    },
    /// A pin's tag has moved on: `tag` now names `newer_sha256`, and the
    /// pinned weights are still what is served.
    NewerAvailable { tag: String, newer_sha256: String },
    /// The weights are deprecated, to retire after `retire_after`
    /// (`YYYY-MM-DD`); `past` once that date has come.
    Retiring { retire_after: String, past: bool },
    /// The weights are retired: the gateway refuses them.
    Retired,
}

/// The notices a choice deserves. A reference this catalog does not know
/// deserves none: the gateway may still serve it (a build older than the
/// catalog it talks to), and if it does not, its refusal names the model.
///
/// `reference` is the Eidola model reference (without `@eidola`);
/// `last_sha256` is the hash the choice's last answer came from, if it has
/// one; `acknowledged_sha256` the weights the user last dismissed an
/// `Updated` notice for, which keeps that notice dismissed while the tag
/// names them; `today` is the current date, `YYYY-MM-DD`.
pub fn notices(
    rows: &[CatalogRow],
    reference: &str,
    last_sha256: Option<&str>,
    acknowledged_sha256: Option<&str>,
    today: &str,
) -> Vec<ModelNotice> {
    let Ok(parsed) = ModelReference::parse(reference) else {
        return Vec::new();
    };
    let Some((row, _)) = resolve(rows, reference) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    match parsed {
        ModelReference::Follow { tag } => {
            if let Some(previous) = last_sha256
                && previous != row.sha256
                && acknowledged_sha256 != Some(row.sha256)
            {
                out.push(ModelNotice::Updated {
                    tag: tag.to_string(),
                    previous_sha256: previous.to_string(),
                    current_sha256: row.sha256.to_string(),
                });
            }
        }
        ModelReference::Pin { tag, sha256 } => {
            if let Some(newer) = rows.iter().find(|r| r.tags.contains(&tag))
                && newer.sha256 != sha256
                && !matches!(newer.lifecycle, Lifecycle::Retired { .. })
            {
                out.push(ModelNotice::NewerAvailable {
                    tag: tag.to_string(),
                    newer_sha256: newer.sha256.to_string(),
                });
            }
        }
        ModelReference::Hash { .. } => {}
    }
    match row.lifecycle {
        Lifecycle::Current => {}
        Lifecycle::Deprecated { retire_after } => out.push(ModelNotice::Retiring {
            retire_after: retire_after.to_string(),
            // Both are `YYYY-MM-DD`, so the byte order is the date order.
            past: today >= retire_after,
        }),
        Lifecycle::Retired { .. } => out.push(ModelNotice::Retired),
    }
    out
}

/// The pin a `NewerAvailable` notice among `notices` offers to switch to:
/// the tag's label with the newer weights, `tag@sha256:<hex>`.
pub fn newer_pin(notices: &[ModelNotice]) -> Option<String> {
    notices.iter().find_map(|n| match n {
        ModelNotice::NewerAvailable { tag, newer_sha256 } => Some(pinned(Some(tag), newer_sha256)),
        _ => None,
    })
}

/// The weights an `Updated` notice among `notices` says the tag names now,
/// which dismissing it records.
pub fn updated_to(notices: &[ModelNotice]) -> Option<&str> {
    notices.iter().find_map(|n| match n {
        ModelNotice::Updated { current_sha256, .. } => Some(current_sha256.as_str()),
        _ => None,
    })
}

/// A model's identity, for a user to see and check independently
/// ([`MODEL_IDENTITY_DOC`] specifies the hash and how to recompute it).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub sha256: String,
    pub tags: Vec<String>,
    pub name: String,
    pub repo: String,
    pub revision: String,
    pub files: Vec<ModelIdentityFile>,
    /// `current`, `deprecated` or `retired`.
    pub lifecycle: String,
    pub retire_after: Option<String>,
    /// The page that specifies the hash.
    pub doc_url: String,
}

/// One file of a model's manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelIdentityFile {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// The identity of what `reference` resolves to, if the catalog knows it.
pub fn identity(rows: &[CatalogRow], reference: &str) -> Option<ModelIdentity> {
    let (row, _) = resolve(rows, reference)?;
    let (lifecycle, retire_after) = match row.lifecycle {
        Lifecycle::Current => ("current", None),
        Lifecycle::Deprecated { retire_after } => ("deprecated", Some(retire_after.to_string())),
        Lifecycle::Retired { retire_after } => ("retired", retire_after.map(str::to_string)),
    };
    Some(ModelIdentity {
        sha256: row.sha256.to_string(),
        tags: row.tags.iter().map(|t| (*t).to_string()).collect(),
        name: row.name.to_string(),
        repo: row.repo.to_string(),
        revision: row.revision.to_string(),
        files: row
            .files
            .iter()
            .map(|f| ModelIdentityFile {
                path: f.path.to_string(),
                size: f.size,
                sha256: f.sha256.to_string(),
            })
            .collect(),
        lifecycle: lifecycle.to_string(),
        retire_after,
        doc_url: MODEL_IDENTITY_DOC.to_string(),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const FLASH_1: &str =
        "1111111111111111111111111111111111111111111111111111111111111111";
    pub(crate) const FLASH_2: &str =
        "2222222222222222222222222222222222222222222222222222222222222222";
    const OLD: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const GONE: &str = "4444444444444444444444444444444444444444444444444444444444444444";

    const FILES: &[WeightsFile] = &[WeightsFile {
        path: "model.safetensors",
        size: 9,
        sha256: "5555555555555555555555555555555555555555555555555555555555555555",
    }];

    const fn row(
        sha256: &'static str,
        tags: &'static [&'static str],
        lifecycle: Lifecycle,
    ) -> CatalogRow {
        CatalogRow {
            sha256,
            tags,
            name: "Fixture",
            repo: "example/fixture",
            revision: "6666666666666666666666666666666666666666",
            files: FILES,
            lifecycle,
        }
    }

    /// `flash` moved from FLASH_1 (now deprecated) to FLASH_2; `old` is
    /// deprecated with its date passed; `gone` is retired.
    pub(crate) const ROWS: &[CatalogRow] = &[
        row(FLASH_2, &["flash", "flash-latest"], Lifecycle::Current),
        row(
            FLASH_1,
            &["flash-previous"],
            Lifecycle::Deprecated {
                retire_after: "2027-06-01",
            },
        ),
        row(
            OLD,
            &["old"],
            Lifecycle::Deprecated {
                retire_after: "2026-01-01",
            },
        ),
        row(GONE, &["gone"], Lifecycle::Retired { retire_after: None }),
    ];

    const TODAY: &str = "2026-11-09";

    #[test]
    fn references_resolve_as_the_gateway_resolves_them() {
        assert_eq!(
            resolve(ROWS, "flash").unwrap().1,
            format!("flash@sha256:{FLASH_2}")
        );
        // The digest is authoritative; the tag is a label.
        let (row, spelled) = resolve(ROWS, &format!("flash@sha256:{FLASH_1}")).unwrap();
        assert_eq!(row.sha256, FLASH_1);
        assert_eq!(spelled, format!("flash@sha256:{FLASH_1}"));
        assert_eq!(
            resolve(ROWS, &format!("sha256:{OLD}")).unwrap().1,
            format!("sha256:{OLD}")
        );
        assert!(resolve(ROWS, "nothing").is_none());
        assert!(resolve(ROWS, "not a reference").is_none());
    }

    #[test]
    fn a_followed_tag_that_moved_says_the_model_was_updated() {
        assert_eq!(notices(ROWS, "flash", Some(FLASH_2), None, TODAY), vec![]);
        assert_eq!(notices(ROWS, "flash", None, None, TODAY), vec![]);
        assert_eq!(
            notices(ROWS, "flash", Some(FLASH_1), None, TODAY),
            vec![ModelNotice::Updated {
                tag: "flash".into(),
                previous_sha256: FLASH_1.into(),
                current_sha256: FLASH_2.into(),
            }]
        );
    }

    #[test]
    fn a_dismissed_update_stays_dismissed_until_the_tag_moves_again() {
        assert_eq!(
            notices(ROWS, "flash", Some(FLASH_1), Some(FLASH_2), TODAY),
            vec![]
        );
        // Dismissed for other weights than the tag names now: shown again.
        assert_eq!(
            notices(ROWS, "flash", Some(FLASH_1), Some(OLD), TODAY).len(),
            1
        );
    }

    /// The pinned spelling of the newer weights a `NewerAvailable` notice
    /// offers, which the one-click switch writes.
    #[test]
    fn a_newer_weights_notice_names_the_pin_to_switch_to() {
        let pin = format!("flash@sha256:{FLASH_1}");
        let n = notices(ROWS, &pin, None, None, TODAY);
        assert_eq!(newer_pin(&n), Some(format!("flash@sha256:{FLASH_2}")));
        assert_eq!(newer_pin(&notices(ROWS, "flash", None, None, TODAY)), None);
    }

    #[test]
    fn a_pin_whose_tag_moved_is_offered_the_newer_model_and_warned_of_retirement() {
        assert_eq!(
            notices(
                ROWS,
                &format!("flash@sha256:{FLASH_1}"),
                Some(FLASH_1),
                None,
                TODAY
            ),
            vec![
                ModelNotice::NewerAvailable {
                    tag: "flash".into(),
                    newer_sha256: FLASH_2.into(),
                },
                ModelNotice::Retiring {
                    retire_after: "2027-06-01".into(),
                    past: false,
                },
            ]
        );
        assert_eq!(
            notices(ROWS, &format!("flash@sha256:{FLASH_2}"), None, None, TODAY),
            vec![]
        );
        assert_eq!(
            notices(ROWS, &format!("sha256:{OLD}"), None, None, TODAY),
            vec![ModelNotice::Retiring {
                retire_after: "2026-01-01".into(),
                past: true,
            }]
        );
        assert_eq!(
            notices(ROWS, "gone", None, None, TODAY),
            vec![ModelNotice::Retired]
        );
        assert_eq!(notices(ROWS, "nothing", None, None, TODAY), vec![]);
        assert_eq!(
            notices(ROWS, "not a reference", Some(FLASH_1), None, TODAY),
            vec![]
        );
    }

    #[test]
    fn a_models_identity_is_its_hash_source_and_manifest() {
        let id = identity(ROWS, "flash").unwrap();
        assert_eq!(id.sha256, FLASH_2);
        assert_eq!(id.tags, ["flash", "flash-latest"]);
        assert_eq!(id.repo, "example/fixture");
        assert_eq!(id.files.len(), 1);
        assert_eq!(id.lifecycle, "current");
        assert_eq!(id.doc_url, MODEL_IDENTITY_DOC);
        assert_eq!(
            identity(ROWS, &format!("sha256:{FLASH_1}"))
                .unwrap()
                .retire_after
                .as_deref(),
            Some("2027-06-01")
        );
    }

    /// The compiled-in catalog is the one the gateway checks: every row
    /// resolves by each of its tags and by its hash.
    #[test]
    fn the_compiled_catalog_resolves_every_row() {
        for row in CATALOG {
            for tag in row.tags {
                assert_eq!(resolve(CATALOG, tag).unwrap().0.sha256, row.sha256);
            }
            assert!(resolve(CATALOG, &format!("sha256:{}", row.sha256)).is_some());
        }
    }
}
