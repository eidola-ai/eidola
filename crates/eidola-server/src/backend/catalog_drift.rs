//! The committed catalog against the list it was transcribed from.
//!
//! `MODEL_CATALOG` hand-transcribes what upstream publishes per model, so every
//! row is a claim that was true on the day it was written. Upstream
//! republishes on its own clock: a row can stop being true with nothing in
//! this tree changing, and no test that compares the catalog to another copy
//! of itself can see that happen — both copies are equally stale. The second
//! copy has to be the live list.
//!
//! So the fetch is one `#[ignore]`d test that a **scheduled** workflow runs
//! (`.github/workflows/catalog-drift.yml`), never the PR gate — the same shape
//! and the same reason as the daily advisory audit: an upstream change is not
//! something a pull request introduced, and failing unrelated work for it is
//! how a check gets muted. `cargo test` therefore stays offline, and what the
//! gate does run is the comparison itself, against the synthetic lists below.
//!
//! The comparison lives here, beside the catalog, so it reads `MODEL_CATALOG`
//! and `NOT_SOLD_UPSTREAM_MODELS` themselves rather than a restatement of them.
//!
//! **Three things are loud, and the omission record is what leaves only
//! three:**
//!
//! 1. A sold row whose upstream value moved (`Drift::Moved`).
//! 2. An upstream id that is neither sold nor recorded as deliberately unsold
//!    (`Drift::Unrecorded`) — "we missed it".
//! 3. A sold row, or an omission record, naming an id upstream no longer
//!    publishes (`Drift::Gone`) — a stale row, or a stale omission.
//!
//! **What is deliberately not compared**, so the check reports decisions
//! rather than differences:
//!
//! - **Cached-input price.** Upstream quotes one for some models and the
//!   pricing contract has no cache-hit term at all (see `MODEL_CATALOG`), so
//!   there is nothing here for it to disagree with.
//! - **`type`.** Upstream calls `gpt-oss-safeguard-120b` a `safety` model and
//!   `voxtral-small-24b` an `audio` one; both serve `/v1/chat/completions` and
//!   both are deliberately sold. The task surface this server can route to is
//!   `endpoints`, so that is what is asserted; `type` is carried only to
//!   describe an unrecorded id in the report.
//! - **Output modalities, names and descriptions.** Upstream publishes no
//!   output-modality list, and the catalog's prose is ours to write.
//! - **What an unsold model's upstream row says.** Every omission reason here
//!   is about a route *this server* does not expose, so no upstream value can
//!   make one wrong; only the id vanishing can.

use std::fmt;

use serde::Deserialize;

use super::{CatalogEntry, MODEL_CATALOG, Modality, NOT_SOLD_UPSTREAM_MODELS, UnsoldModel};

/// The published list the catalog is transcribed from. Public and unkeyed —
/// the scheduled run needs no secret.
const UPSTREAM_MODELS_URL: &str = "https://inference.tinfoil.sh/v1/models";

/// The one inference route this server registers, and therefore the only
/// endpoint that makes an upstream model something a conversation can reach.
const CHAT_ENDPOINT: &str = "/v1/chat/completions";

#[derive(Debug, Deserialize)]
struct UpstreamList {
    data: Vec<UpstreamModel>,
}

/// One model as upstream publishes it.
///
/// Every compared field is **required**: upstream publishes all of them on
/// every model, so an absence is a shape change — and a shape change that
/// defaulted quietly to `false` would either invent drift nobody caused or
/// hide drift that happened. A parse failure is the honest report of "the
/// list no longer looks like this".
///
/// `context_window` is the one field upstream genuinely omits, on models that
/// have none (`doc-upload`, the speech and realtime rows) — all of them
/// recorded as not sold. Absent on a row we *do* sell, it is drift.
///
/// Unknown keys are ignored rather than refused: upstream adding a field is
/// not this check's business, and `cachedInputTokenPricePer1M` already arrives
/// on some rows and is deliberately unread.
#[derive(Debug, Clone, Deserialize)]
struct UpstreamModel {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    context_window: Option<u64>,
    tool_calling: bool,
    reasoning: bool,
    multimodal: bool,
    endpoints: Vec<String>,
    pricing: UpstreamPricing,
}

#[derive(Debug, Clone, Deserialize)]
struct UpstreamPricing {
    #[serde(rename = "inputTokenPricePer1M")]
    input_per_m: f64,
    #[serde(rename = "outputTokenPricePer1M")]
    output_per_m: f64,
    #[serde(rename = "requestPrice")]
    per_request_usd: f64,
}

/// What an id that upstream has stopped publishing was doing here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    Sold,
    RecordedUnsold,
}

/// One disagreement between the catalog and the published list.
#[derive(Debug, PartialEq)]
enum Drift {
    /// A sold row whose upstream value moved out from under it.
    Moved {
        id: String,
        axis: &'static str,
        catalog: String,
        upstream: String,
    },
    /// Upstream publishes an id that is in neither list.
    Unrecorded {
        id: String,
        kind: String,
        endpoints: Vec<String>,
    },
    /// An id this tree names that upstream no longer publishes.
    Gone { id: String, role: Role },
}

impl fmt::Display for Drift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Drift::Moved {
                id,
                axis,
                catalog,
                upstream,
            } => write!(
                f,
                "{id}: {axis} — catalog says {catalog}, upstream publishes {upstream}"
            ),
            Drift::Unrecorded {
                id,
                kind,
                endpoints,
            } => write!(
                f,
                "{id}: published upstream (type {kind}, endpoints {}) but neither sold nor \
                 recorded in NOT_SOLD_UPSTREAM_MODELS",
                endpoints.join(", ")
            ),
            Drift::Gone {
                id,
                role: Role::Sold,
            } => write!(
                f,
                "{id}: sold by MODEL_CATALOG but no longer published upstream"
            ),
            Drift::Gone {
                id,
                role: Role::RecordedUnsold,
            } => write!(
                f,
                "{id}: recorded in NOT_SOLD_UPSTREAM_MODELS but no longer published upstream"
            ),
        }
    }
}

/// Prices are decimal literals on both sides, so equal values land on the same
/// `f64`. The tolerance is here only so a representation difference can never
/// be reported as a price change.
fn same_price(catalog: f64, upstream: f64) -> bool {
    (catalog - upstream).abs() <= 1e-9 * catalog.abs().max(upstream.abs()).max(1.0)
}

/// Diff every pinned row against the published list.
fn drift_against(
    upstream: &[UpstreamModel],
    catalog: &[CatalogEntry],
    unsold: &[UnsoldModel],
) -> Vec<Drift> {
    let mut drifts = Vec::new();

    for entry in catalog {
        let Some(live) = upstream.iter().find(|m| m.id == entry.id) else {
            drifts.push(Drift::Gone {
                id: entry.id.to_string(),
                role: Role::Sold,
            });
            continue;
        };

        let mut moved = |axis: &'static str, catalog: String, upstream: String| {
            drifts.push(Drift::Moved {
                id: entry.id.to_string(),
                axis,
                catalog,
                upstream,
            });
        };

        match live.context_window {
            Some(window) if window == entry.context_length => {}
            Some(window) => moved(
                "context window",
                entry.context_length.to_string(),
                window.to_string(),
            ),
            None => moved(
                "context window",
                entry.context_length.to_string(),
                "none".to_string(),
            ),
        }

        if live.tool_calling != entry.tool_calling {
            moved(
                "tool calling",
                entry.tool_calling.to_string(),
                live.tool_calling.to_string(),
            );
        }

        if live.reasoning != entry.reasoning {
            moved(
                "reasoning",
                entry.reasoning.to_string(),
                live.reasoning.to_string(),
            );
        }

        // Upstream expresses modality as one `multimodal` boolean; the catalog
        // carries explicit lists, so the comparable claim is whether image
        // input is offered.
        let takes_images = entry.input_modalities.contains(&Modality::Image);
        if live.multimodal != takes_images {
            moved(
                "image input",
                takes_images.to_string(),
                live.multimodal.to_string(),
            );
        }

        if !live.endpoints.iter().any(|e| e == CHAT_ENDPOINT) {
            moved(
                "task surface",
                format!("sold, so it must serve {CHAT_ENDPOINT}"),
                live.endpoints.join(", "),
            );
        }

        if !same_price(entry.input_per_m, live.pricing.input_per_m) {
            moved(
                "input price per 1M",
                entry.input_per_m.to_string(),
                live.pricing.input_per_m.to_string(),
            );
        }

        if !same_price(entry.output_per_m, live.pricing.output_per_m) {
            moved(
                "output price per 1M",
                entry.output_per_m.to_string(),
                live.pricing.output_per_m.to_string(),
            );
        }

        if !same_price(entry.per_request_usd, live.pricing.per_request_usd) {
            moved(
                "per-request price",
                entry.per_request_usd.to_string(),
                live.pricing.per_request_usd.to_string(),
            );
        }
    }

    for record in unsold {
        if !upstream.iter().any(|m| m.id == record.id) {
            drifts.push(Drift::Gone {
                id: record.id.to_string(),
                role: Role::RecordedUnsold,
            });
        }
    }

    for live in upstream {
        let sold = catalog.iter().any(|e| e.id == live.id);
        let recorded = unsold.iter().any(|r| r.id == live.id);
        if !sold && !recorded {
            drifts.push(Drift::Unrecorded {
                id: live.id.clone(),
                kind: live.kind.clone(),
                endpoints: live.endpoints.clone(),
            });
        }
    }

    drifts
}

/// The failure message: what drifted, one line each, and where to fix it.
fn report(drifts: &[Drift]) -> String {
    let mut out = format!(
        "the model catalog no longer matches {UPSTREAM_MODELS_URL} ({} difference(s)):\n",
        drifts.len()
    );
    for drift in drifts {
        out.push_str(&format!("  - {drift}\n"));
    }
    out.push_str(
        "\nEvery line is a decision, not a chore: correct the row in MODEL_CATALOG \
         (crates/eidola-server/src/backend.rs), or record the id in \
         NOT_SOLD_UPSTREAM_MODELS with the reason it is not sold.\n",
    );
    out
}

/// The live half. Ignored by default so `cargo test` and every PR gate stay
/// offline; `.github/workflows/catalog-drift.yml` names this test explicitly
/// on a daily schedule.
#[tokio::test]
#[ignore = "requires network access to inference.tinfoil.sh"]
async fn the_live_upstream_list_still_matches_the_catalog() {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let client = reqwest::Client::builder()
        .tls_backend_preconfigured(crate::tls_config())
        .build()
        .expect("build an HTTPS client for the published model list");

    let list: UpstreamList = client
        .get(UPSTREAM_MODELS_URL)
        .send()
        .await
        .expect("fetch the published model list")
        .error_for_status()
        .expect("the published model list answered an error status")
        .json()
        .await
        .expect("the published model list no longer parses as this check expects");

    assert!(
        !list.data.is_empty(),
        "{UPSTREAM_MODELS_URL} published an empty list; \
         comparing against it would pass by vacuity"
    );
    println!(
        "{UPSTREAM_MODELS_URL} published {} models; {} sold, {} recorded as not sold",
        list.data.len(),
        MODEL_CATALOG.len(),
        NOT_SOLD_UPSTREAM_MODELS.len()
    );

    let drifts = drift_against(&list.data, MODEL_CATALOG, NOT_SOLD_UPSTREAM_MODELS);
    assert!(drifts.is_empty(), "{}", report(&drifts));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An upstream row that says exactly what the catalog says about it — the
    /// baseline every fixture below perturbs, built from `MODEL_CATALOG`
    /// rather than typed out, so it cannot drift from the thing it mirrors.
    fn mirror(entry: &CatalogEntry) -> UpstreamModel {
        UpstreamModel {
            id: entry.id.to_string(),
            kind: "chat".to_string(),
            context_window: Some(entry.context_length),
            tool_calling: entry.tool_calling,
            reasoning: entry.reasoning,
            multimodal: entry.input_modalities.contains(&Modality::Image),
            endpoints: vec![CHAT_ENDPOINT.to_string()],
            pricing: UpstreamPricing {
                input_per_m: entry.input_per_m,
                output_per_m: entry.output_per_m,
                per_request_usd: entry.per_request_usd,
            },
        }
    }

    /// A row for an id we deliberately do not sell. Nothing about it is
    /// compared, so only its presence matters.
    fn unsold_row(record: &UnsoldModel) -> UpstreamModel {
        UpstreamModel {
            id: record.id.to_string(),
            kind: "embedding".to_string(),
            context_window: None,
            tool_calling: false,
            reasoning: false,
            multimodal: false,
            endpoints: vec!["/v1/embeddings".to_string()],
            pricing: UpstreamPricing {
                input_per_m: 0.0,
                output_per_m: 0.0,
                per_request_usd: 0.0,
            },
        }
    }

    fn a_faithful_list() -> Vec<UpstreamModel> {
        MODEL_CATALOG
            .iter()
            .map(mirror)
            .chain(NOT_SOLD_UPSTREAM_MODELS.iter().map(unsold_row))
            .collect()
    }

    fn drift(list: &[UpstreamModel]) -> Vec<Drift> {
        drift_against(list, MODEL_CATALOG, NOT_SOLD_UPSTREAM_MODELS)
    }

    /// The positive baseline. A check that reports nothing because it can see
    /// nothing is worse than no check, so every fixture below is read against
    /// this one.
    #[test]
    fn a_list_that_agrees_with_the_catalog_reports_nothing() {
        let list = a_faithful_list();
        assert!(!list.is_empty(), "the fixture itself is empty");
        assert_eq!(drift(&list), Vec::new(), "{}", report(&drift(&list)));
    }

    /// One upstream value moved out from under a row, and the axis that names
    /// it.
    type Perturbation = (&'static str, Box<dyn Fn(&mut UpstreamModel)>);

    /// Class (a): a sold row whose upstream value moved. One perturbation per
    /// axis the catalog transcribes, each expected to name that axis and that
    /// model and nothing else.
    #[test]
    fn a_sold_row_whose_upstream_value_moved_is_reported() {
        let sold = MODEL_CATALOG[0].id;
        let perturbations: Vec<Perturbation> = vec![
            (
                "context window",
                Box::new(|m: &mut UpstreamModel| m.context_window = Some(4_096)),
            ),
            (
                "context window",
                Box::new(|m: &mut UpstreamModel| m.context_window = None),
            ),
            (
                "tool calling",
                Box::new(|m: &mut UpstreamModel| m.tool_calling = !m.tool_calling),
            ),
            (
                "reasoning",
                Box::new(|m: &mut UpstreamModel| m.reasoning = !m.reasoning),
            ),
            (
                "image input",
                Box::new(|m: &mut UpstreamModel| m.multimodal = !m.multimodal),
            ),
            (
                "task surface",
                Box::new(|m: &mut UpstreamModel| m.endpoints = vec!["/v1/responses".to_string()]),
            ),
            (
                "input price per 1M",
                Box::new(|m: &mut UpstreamModel| m.pricing.input_per_m += 0.25),
            ),
            (
                "output price per 1M",
                Box::new(|m: &mut UpstreamModel| m.pricing.output_per_m += 0.25),
            ),
            (
                "per-request price",
                Box::new(|m: &mut UpstreamModel| m.pricing.per_request_usd += 0.01),
            ),
        ];

        for (axis, perturb) in perturbations {
            let mut list = a_faithful_list();
            perturb(&mut list[0]);

            let drifts = drift(&list);
            assert_eq!(
                drifts.len(),
                1,
                "{axis}: expected one difference, got {drifts:?}"
            );
            match &drifts[0] {
                Drift::Moved {
                    id, axis: found, ..
                } => {
                    assert_eq!(id, sold, "{axis}: named the wrong model");
                    assert_eq!(*found, axis, "named the wrong axis");
                }
                other => panic!("{axis}: expected a moved value, got {other:?}"),
            }
            assert!(
                report(&drifts).contains(sold),
                "{axis}: report omits the model"
            );
        }
    }

    /// Class (b): upstream publishes something in neither list. This is the
    /// one the omission record buys — without it every deliberate absence
    /// would land here too, and a check that cries seven times gets muted.
    #[test]
    fn an_upstream_id_that_is_neither_sold_nor_recorded_is_reported() {
        let mut list = a_faithful_list();
        let mut newcomer = mirror(&MODEL_CATALOG[0]);
        newcomer.id = "a-model-nobody-here-has-heard-of".to_string();
        newcomer.kind = "chat".to_string();
        list.push(newcomer);

        let drifts = drift(&list);
        assert_eq!(drifts.len(), 1, "{drifts:?}");
        match &drifts[0] {
            Drift::Unrecorded { id, .. } => {
                assert_eq!(id, "a-model-nobody-here-has-heard-of");
            }
            other => panic!("expected an unrecorded id, got {other:?}"),
        }
        assert!(report(&drifts).contains("a-model-nobody-here-has-heard-of"));
    }

    /// Class (c), first half: a sold row upstream has retired. Selling an id
    /// nobody serves prices a selection that can only fail at the point of
    /// use.
    #[test]
    fn a_sold_id_upstream_no_longer_publishes_is_reported() {
        let retired = MODEL_CATALOG[0].id;
        let list: Vec<UpstreamModel> = a_faithful_list()
            .into_iter()
            .filter(|m| m.id != retired)
            .collect();

        let drifts = drift(&list);
        assert_eq!(drifts.len(), 1, "{drifts:?}");
        assert_eq!(
            drifts[0],
            Drift::Gone {
                id: retired.to_string(),
                role: Role::Sold,
            }
        );
        assert!(report(&drifts).contains(retired));
    }

    /// Class (c), second half: an omission record upstream has retired. A
    /// stale entry here is how the record quietly stops meaning anything —
    /// it would go on excusing an absence nobody is choosing any more.
    #[test]
    fn a_stale_omission_record_is_reported() {
        let retired = NOT_SOLD_UPSTREAM_MODELS[0].id;
        let list: Vec<UpstreamModel> = a_faithful_list()
            .into_iter()
            .filter(|m| m.id != retired)
            .collect();

        let drifts = drift(&list);
        assert_eq!(drifts.len(), 1, "{drifts:?}");
        assert_eq!(
            drifts[0],
            Drift::Gone {
                id: retired.to_string(),
                role: Role::RecordedUnsold,
            }
        );
        assert!(report(&drifts).contains(retired));
    }

    /// A published field this check does not read must not become drift: the
    /// catalog would otherwise have to restate a number the pricing contract
    /// has nowhere to put, and a `type` that says `safety` or `audio` about a
    /// model that chats.
    #[test]
    fn unread_upstream_fields_are_not_drift() {
        let mut list = a_faithful_list();
        list[0].kind = "safety".to_string();
        list[0].endpoints.push("/v1/responses".to_string());
        assert_eq!(drift(&list), Vec::new());

        let with_cached_price = serde_json::json!({
            "id": "x",
            "type": "chat",
            "context_window": 8192,
            "tool_calling": true,
            "reasoning": false,
            "multimodal": false,
            "endpoints": [CHAT_ENDPOINT],
            "pricing": {
                "cachedInputTokenPricePer1M": 0.1,
                "inputTokenPricePer1M": 0.4,
                "outputTokenPricePer1M": 1.0,
                "requestPrice": 0.0,
            },
            "owned_by": "tinfoil",
        });
        let parsed: UpstreamModel = serde_json::from_value(with_cached_price)
            .expect("an upstream row with fields we do not read still parses");
        assert_eq!(parsed.id, "x");
    }

    /// A capability that stopped being published is a shape change, and a
    /// shape change that defaulted to `false` would report drift nobody
    /// caused. Refusing the parse is how it stays honest.
    #[test]
    fn a_row_missing_a_capability_refuses_to_parse() {
        let without_reasoning = serde_json::json!({
            "id": "x",
            "type": "chat",
            "context_window": 8192,
            "tool_calling": true,
            "multimodal": false,
            "endpoints": [CHAT_ENDPOINT],
            "pricing": {
                "inputTokenPricePer1M": 0.4,
                "outputTokenPricePer1M": 1.0,
                "requestPrice": 0.0,
            },
        });
        assert!(serde_json::from_value::<UpstreamModel>(without_reasoning).is_err());
    }
}
