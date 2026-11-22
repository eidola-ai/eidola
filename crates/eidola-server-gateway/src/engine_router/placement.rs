//! Where engines are: the operator-edited `engine_placement` table.
//!
//! **Availability only, never trust.** The table lives in Postgres, outside
//! the enclave, so anything in it may be wrong or hostile. A row can only
//! make an upstream a *candidate*: it is used only when it is enabled, names
//! a model this gateway pins, names one of the deployments this gateway's own
//! tree pins for that model (same-sha pinning: a row for any other deployment,
//! an older or newer engine build included, is ignored by this gateway and
//! may serve another), and has an `https` base URL. Whether a candidate is
//! the engine it claims to be is decided on every TLS handshake, against the
//! model's compiled-in pins, and before it takes traffic by a probe of the
//! weights it reports (`super::probe`). Nothing about an upstream's health is
//! written back: every gateway keeps its own.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;

use crate::db::EnginePlacementRow;

/// A source of placement rows.
pub trait PlacementSource: Send + Sync + 'static {
    fn load(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<EnginePlacementRow>, String>> + Send + '_>>;
}

/// The production source: the gateway's database.
pub struct PostgresPlacement(pub deadpool_postgres::Pool);

impl PlacementSource for PostgresPlacement {
    fn load(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<EnginePlacementRow>, String>> + Send + '_>> {
        Box::pin(async move {
            crate::db::get_engine_placement(&self.0)
                .await
                .map_err(|e| e.to_string())
        })
    }
}

/// A placement row this gateway may use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// The deployment's config hash.
    pub deployment: String,
    /// The engine's base URL, normalized (`https://host[:port][/path]`, no
    /// trailing slash). Requests go to `{base_url}/v1/chat/completions`.
    pub base_url: String,
}

/// Why a row is not used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Skipped {
    Disabled,
    UnpinnedModel,
    UnpinnedDeployment,
    InvalidUrl,
    Duplicate,
}

impl Skipped {
    /// Every reason, in the order the `engine.placement.rows` gauge reports
    /// them after `usable`.
    pub const ALL: [Skipped; 5] = [
        Skipped::Disabled,
        Skipped::UnpinnedModel,
        Skipped::UnpinnedDeployment,
        Skipped::InvalidUrl,
        Skipped::Duplicate,
    ];

    /// The reason's metric label.
    pub fn label(self) -> &'static str {
        match self {
            Skipped::Disabled => "disabled",
            Skipped::UnpinnedModel => "unpinned_model",
            Skipped::UnpinnedDeployment => "unpinned_deployment",
            Skipped::InvalidUrl => "invalid_url",
            Skipped::Duplicate => "duplicate",
        }
    }

    /// The reason, for the operator.
    pub fn reason(self) -> &'static str {
        match self {
            Skipped::Disabled => "the row is disabled",
            Skipped::UnpinnedModel => "this build pins no model with that id",
            Skipped::UnpinnedDeployment => {
                "this build does not pin that deployment for the model (a stale or newer config hash)"
            }
            Skipped::InvalidUrl => "its base URL is not a plain https URL",
            Skipped::Duplicate => "another row already places the model at that URL",
        }
    }
}

/// A row not used, by what it names and why.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SkippedRow {
    /// The row's model id, as the table has it.
    pub model: String,
    /// The row's deployment config hash, as the table has it.
    pub deployment: String,
    pub reason: Skipped,
}

/// The usable rows, per model, in the order given; and every row skipped,
/// with its reason. `accepted(model)` is the set of config hashes this build
/// pins for the model, `None` when it pins no such model.
pub fn admissible<'a>(
    rows: &[EnginePlacementRow],
    accepted: impl Fn(&str) -> Option<&'a [String]>,
) -> (BTreeMap<String, Vec<Placement>>, Vec<SkippedRow>) {
    let mut out: BTreeMap<String, Vec<Placement>> = BTreeMap::new();
    let mut skipped = Vec::new();
    for row in rows {
        let mut skip = |reason| {
            skipped.push(SkippedRow {
                model: row.model_id.clone(),
                deployment: row.deployment.clone(),
                reason,
            })
        };
        if !row.enabled {
            skip(Skipped::Disabled);
            continue;
        }
        let Some(deployments) = accepted(&row.model_id) else {
            skip(Skipped::UnpinnedModel);
            continue;
        };
        if !deployments.contains(&row.deployment) {
            skip(Skipped::UnpinnedDeployment);
            continue;
        }
        let Some(base_url) = normalize_base_url(&row.base_url) else {
            skip(Skipped::InvalidUrl);
            continue;
        };
        let list = out.entry(row.model_id.clone()).or_default();
        if list.iter().any(|p| p.base_url == base_url) {
            skip(Skipped::Duplicate);
            continue;
        }
        list.push(Placement {
            deployment: row.deployment.clone(),
            base_url,
        });
    }
    (out, skipped)
}

/// What the operator has been told about placement, so each condition is
/// reported once per change rather than on every refresh: the rows skipped
/// at the last refresh, the pinned models it left without a usable row, and
/// how many rows it counted by reason (the `engine.placement.rows` gauge).
#[derive(Debug, Default)]
pub struct PlacementReport {
    skipped: BTreeSet<SkippedRow>,
    unplaced: BTreeSet<String>,
    rows: [u64; 1 + Skipped::ALL.len()],
}

/// What changed at one refresh, for the operator.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct PlacementChanges {
    /// Rows skipped now that were not at the last refresh.
    pub newly_skipped: Vec<SkippedRow>,
    /// Pinned models with no usable row now that had one at the last
    /// refresh (or at the first refresh).
    pub newly_unplaced: Vec<String>,
    /// Pinned models with a usable row again.
    pub placed_again: Vec<String>,
}

impl PlacementReport {
    /// Record one refresh's result: the usable rows per model, the rows
    /// skipped, and every pinned model's id. Returns what changed since the
    /// last one. Bounded by the table's row limit, since it keeps only the
    /// last refresh.
    pub fn update<'a>(
        &mut self,
        usable: &BTreeMap<String, Vec<Placement>>,
        skipped: &[SkippedRow],
        pinned: impl Iterator<Item = &'a String>,
    ) -> PlacementChanges {
        let skipped_now: BTreeSet<SkippedRow> = skipped.iter().cloned().collect();
        let unplaced_now: BTreeSet<String> = pinned
            .filter(|model| usable.get(*model).is_none_or(Vec::is_empty))
            .cloned()
            .collect();
        let changes = PlacementChanges {
            newly_skipped: skipped_now.difference(&self.skipped).cloned().collect(),
            newly_unplaced: unplaced_now.difference(&self.unplaced).cloned().collect(),
            placed_again: self.unplaced.difference(&unplaced_now).cloned().collect(),
        };
        let mut rows = [0; 1 + Skipped::ALL.len()];
        rows[0] = usable.values().map(|list| list.len() as u64).sum();
        for row in skipped {
            let at = Skipped::ALL
                .iter()
                .position(|r| *r == row.reason)
                .unwrap_or(0);
            rows[1 + at] += 1;
        }
        self.rows = rows;
        self.skipped = skipped_now;
        self.unplaced = unplaced_now;
        changes
    }

    /// The last refresh's rows by `reason`: `usable`, then each skip reason.
    pub fn rows(&self) -> Vec<(&'static str, u64)> {
        std::iter::once("usable")
            .chain(Skipped::ALL.iter().map(|r| r.label()))
            .zip(self.rows)
            .collect()
    }
}

/// `https://host[:port][/path]` with no credentials, query or fragment, and
/// no trailing slash; anything else is refused. The attesting client verifies
/// TLS and attestation, so a plain `http` URL could never carry a request
/// anyway; refusing it here keeps the row from ever looking usable.
pub fn normalize_base_url(text: &str) -> Option<String> {
    let url = reqwest::Url::parse(text).ok()?;
    if url.scheme() != "https"
        || url.host_str().is_none_or(str::is_empty)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    let mut normalized = url.to_string();
    while normalized.ends_with('/') {
        normalized.pop();
    }
    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PINNED: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn row(model: &str, deployment: &str, url: &str, enabled: bool) -> EnginePlacementRow {
        EnginePlacementRow {
            model_id: model.into(),
            deployment: deployment.into(),
            base_url: url.into(),
            enabled,
        }
    }

    /// A row is used only when it is enabled, names a pinned model and one of
    /// that model's pinned deployments, and has a usable URL; the first of two
    /// rows for one URL wins.
    #[test]
    fn only_enabled_rows_for_pinned_deployments_are_used() {
        let accepted = vec![PINNED.to_string()];
        let lookup = |m: &str| (m == "pinned-model").then_some(accepted.as_slice());
        let rows = [
            row("pinned-model", PINNED, "https://a.test/", true),
            row("pinned-model", PINNED, "https://b.test", false),
            row("pinned-model", OTHER, "https://c.test", true),
            row("other-model", PINNED, "https://d.test", true),
            row("pinned-model", PINNED, "http://e.test", true),
            row("pinned-model", PINNED, "https://a.test", true),
            row("pinned-model", PINNED, "https://f.test:8443/engine/", true),
        ];
        let (usable, skipped) = admissible(&rows, lookup);
        let skipped: Vec<Skipped> = skipped.into_iter().map(|row| row.reason).collect();
        assert_eq!(
            usable["pinned-model"],
            vec![
                Placement {
                    deployment: PINNED.into(),
                    base_url: "https://a.test".into()
                },
                Placement {
                    deployment: PINNED.into(),
                    base_url: "https://f.test:8443/engine".into()
                },
            ]
        );
        assert_eq!(usable.len(), 1);
        assert_eq!(
            skipped,
            vec![
                Skipped::Disabled,
                Skipped::UnpinnedDeployment,
                Skipped::UnpinnedModel,
                Skipped::InvalidUrl,
                Skipped::Duplicate,
            ]
        );
    }

    /// Each condition is reported once per change: a refresh that finds the
    /// same rows reports nothing, a row that starts being skipped is reported
    /// with its reason, and a pinned model is reported when it loses its last
    /// usable row and when it gets one again.
    #[test]
    fn placement_conditions_are_reported_once_per_change() {
        let accepted = vec![PINNED.to_string()];
        let lookup = |m: &str| (m == "pinned-model").then_some(accepted.as_slice());
        let pinned = ["pinned-model".to_string()];
        let mut report = PlacementReport::default();

        let rows = [row("other-model", PINNED, "https://d.test", true)];
        let (usable, skipped) = admissible(&rows, lookup);
        let first = report.update(&usable, &skipped, pinned.iter());
        assert_eq!(
            first,
            PlacementChanges {
                newly_skipped: vec![SkippedRow {
                    model: "other-model".into(),
                    deployment: PINNED.into(),
                    reason: Skipped::UnpinnedModel,
                }],
                newly_unplaced: vec!["pinned-model".into()],
                placed_again: vec![],
            }
        );
        assert_eq!(
            report.update(&usable, &skipped, pinned.iter()),
            PlacementChanges::default(),
            "an unchanged table reports nothing"
        );

        let rows = [
            row("other-model", PINNED, "https://d.test", true),
            row("pinned-model", OTHER, "https://c.test", true),
            row("pinned-model", PINNED, "https://a.test", true),
        ];
        let (usable, skipped) = admissible(&rows, lookup);
        assert_eq!(
            report.update(&usable, &skipped, pinned.iter()),
            PlacementChanges {
                newly_skipped: vec![SkippedRow {
                    model: "pinned-model".into(),
                    deployment: OTHER.into(),
                    reason: Skipped::UnpinnedDeployment,
                }],
                newly_unplaced: vec![],
                placed_again: vec!["pinned-model".into()],
            }
        );
        assert_eq!(
            report.rows(),
            vec![
                ("usable", 1),
                ("disabled", 0),
                ("unpinned_model", 1),
                ("unpinned_deployment", 1),
                ("invalid_url", 0),
                ("duplicate", 0),
            ]
        );
    }

    #[test]
    fn only_a_plain_https_url_is_usable() {
        for refused in [
            "http://a.test",
            "https://user:pw@a.test",
            "https://user@a.test",
            "https://a.test/?q=1",
            "https://a.test/#f",
            "ftp://a.test",
            "a.test",
            "",
            "https://",
        ] {
            assert_eq!(normalize_base_url(refused), None, "{refused}");
        }
        assert_eq!(
            normalize_base_url("https://A.test:443/").as_deref(),
            Some("https://a.test")
        );
        assert_eq!(
            normalize_base_url("https://a.test:8443/x/").as_deref(),
            Some("https://a.test:8443/x")
        );
    }
}
