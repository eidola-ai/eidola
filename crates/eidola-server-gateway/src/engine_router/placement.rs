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

use std::collections::BTreeMap;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Skipped {
    Disabled,
    UnpinnedModel,
    UnpinnedDeployment,
    InvalidUrl,
    Duplicate,
}

/// The usable rows, per model, in the order given; and how many of each kind
/// were skipped. `accepted(model)` is the set of config hashes this build pins
/// for the model, `None` when it pins no such model.
pub fn admissible<'a>(
    rows: &[EnginePlacementRow],
    accepted: impl Fn(&str) -> Option<&'a [String]>,
) -> (BTreeMap<String, Vec<Placement>>, Vec<Skipped>) {
    let mut out: BTreeMap<String, Vec<Placement>> = BTreeMap::new();
    let mut skipped = Vec::new();
    for row in rows {
        if !row.enabled {
            skipped.push(Skipped::Disabled);
            continue;
        }
        let Some(deployments) = accepted(&row.model_id) else {
            skipped.push(Skipped::UnpinnedModel);
            continue;
        };
        if !deployments.contains(&row.deployment) {
            skipped.push(Skipped::UnpinnedDeployment);
            continue;
        }
        let Some(base_url) = normalize_base_url(&row.base_url) else {
            skipped.push(Skipped::InvalidUrl);
            continue;
        };
        let list = out.entry(row.model_id.clone()).or_default();
        if list.iter().any(|p| p.base_url == base_url) {
            skipped.push(Skipped::Duplicate);
            continue;
        }
        list.push(Placement {
            deployment: row.deployment.clone(),
            base_url,
        });
    }
    (out, skipped)
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
