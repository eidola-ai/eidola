//! The model catalog, and the shapes an upstream's answer takes.
//!
//! Every model this gateway sells is served by Eidola's own inference engine,
//! in the deployments `releases/trust/engine-enclaves.json` pins for it, and
//! reached through `engine_router`. The catalog (`crate::models`) is built
//! from `models/` and bound to those pins at compile time; only pricing can
//! be overridden at runtime, via `PRICING_OVERRIDES`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::engine_trust::{self, PinnedModel};
use crate::models::{self, CatalogRow, Lifecycle, MODEL_CATALOG, ModelRefError, RowPrice};
use crate::types::{
    Capability, ChatCompletionChunk, ChatCompletionResponse, EngineDeploymentPin, LifecycleStatus,
    Model, ModelCapabilities, ModelLifecycle, ModelPricing, ModelsResponse,
    PinnedWeightsCapability, PromptCacheCapability, ScaledPrice, WeightsFileEntry,
    WeightsProvenance,
};

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// The type of Trusted Execution Environment backing a model.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum TeeType {
    /// An Eidola inference node in a confidential VM (Intel TDX with NVIDIA
    /// confidential-computing GPUs), attested against this gateway's pins.
    EidolaEngine,
}

/// Metadata about a backend's execution of a request.
#[derive(Debug, Clone)]
pub struct BackendMeta {
    /// The backend provider name (`engine_router::PROVIDER`).
    pub provider: String,

    /// The chat completion ID returned by the backend.
    pub chat_id: Option<String>,

    /// The actual model name used by the backend.
    pub backend_model: String,

    /// Whether this model runs inside a TEE.
    pub tee_type: Option<TeeType>,

    /// Token usage statistics (from the response or final streaming chunk).
    pub usage: Option<crate::types::Usage>,
}

/// A completed (non-streaming) backend response.
pub struct BackendResponse {
    /// The OpenAI-format completion response.
    pub response: ChatCompletionResponse,

    /// Metadata about this execution.
    pub meta: BackendMeta,
}

/// Events emitted by a streaming backend.
///
/// Chunk and Done carry the [`std::time::Instant`] at which the upstream
/// reader took them off the socket. Timing must use that stamp, not the
/// receive time on the handler side: both the backend and downstream
/// channels are bounded, so a slow client backpressures the pipeline and
/// arrival at the handler measures delivery delay, not upstream
/// generation. (A persistently slow client eventually throttles the
/// socket itself via TCP flow control — that residual coupling is
/// inherent to bounded buffering and accepted.)
pub enum BackendStreamEvent {
    /// A content chunk (standard OpenAI format), stamped with its arrival
    /// off the upstream socket.
    Chunk(ChatCompletionChunk, std::time::Instant),

    /// The stream has completed. Carries final metadata, stamped with the
    /// moment the upstream stream ended.
    Done(BackendMeta, std::time::Instant),
}

// ---------------------------------------------------------------------------
// Static model catalog
// ---------------------------------------------------------------------------

/// Default pricing markup factor.
pub const DEFAULT_PRICING_MARKUP: f64 = 1.5;

/// Validate that the pricing markup covers the contract's safe cost factor.
///
/// The shared client/server pricing contract (`eidola-common`) charges the
/// prompt-side byte term at no less than `bytes / N`, where `N =
/// SAFE_COST_FACTOR_NUM / SAFE_COST_FACTOR_DEN` — and since BPE tokenizers
/// never produce more tokens than content bytes, the worst-case
/// actual/charged token ratio on that term is exactly `N`. Break-even on
/// dynamic costs therefore requires `PRICING_MARKUP >= N`; a markup below
/// the factor silently opens a loss window, so startup refuses it outright.
///
/// Future refinement (out of scope for now): when the configured markup
/// exceeds the factor, the server could publish a larger dynamic factor to
/// clients via the `/models` payload instead of both sides relying on the
/// compiled-in constants.
pub fn validate_pricing_markup(markup: f64) -> Result<(), String> {
    let floor =
        eidola_common::SAFE_COST_FACTOR_NUM as f64 / eidola_common::SAFE_COST_FACTOR_DEN as f64;
    // `is_nan` check: a NaN markup must fail too, and `<` alone lets it pass.
    if markup.is_nan() || markup < floor {
        return Err(format!(
            "PRICING_MARKUP ({markup}) must be >= the safe cost factor \
             {}/{} = {floor}: the pricing contract charges prompt bytes at \
             1/{floor} of their worst-case token count, so a lower markup \
             would sell tokens below cost",
            eidola_common::SAFE_COST_FACTOR_NUM,
            eidola_common::SAFE_COST_FACTOR_DEN,
        ));
    }
    Ok(())
}

/// Fixed scale factor for pricing: credits per token = value / PRICING_SCALE_FACTOR.
pub const PRICING_SCALE_FACTOR: u64 = 1_000_000;

/// Convert USD per million tokens to scaled integer credits, applying markup.
///
/// The 1e6 (USD→µ$) and /1e6 (per-M→per-token) factors cancel, leaving:
/// `scaled = usd_per_million * markup * PRICING_SCALE_FACTOR`
fn usd_per_m_to_scaled_credits(usd_per_million: f64, markup: f64) -> u64 {
    (usd_per_million * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

/// Convert USD per request to scaled integer credits, applying markup.
fn usd_per_req_to_scaled_credits(usd_per_request: f64, markup: f64) -> u64 {
    // credits/request = USD * 1e6 (USD→µ$) * markup
    // scaled = credits/request * PRICING_SCALE_FACTOR
    (usd_per_request * 1e6 * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

/// Runtime pricing override for one model, from `PRICING_OVERRIDES`.
///
/// `PRICING_OVERRIDES` is a JSON object **keyed by weights hash** (64
/// lowercase hex digits), e.g. `{"<sha256>": {"input": 4.0, "output": 20.0}}`:
/// a price belongs to the weights being served, so it neither moves with a
/// tag nor follows one onto other weights. Every key must be a hash a served
/// row has, and every member must suit the row's pricing shape; anything
/// else refuses the boot ([`Catalog::new`]), so a stale or mistyped override
/// can never be silently ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PricingOverride {
    /// Override: USD per million input tokens.
    input: Option<f64>,
    /// Override: USD per million output tokens.
    output: Option<f64>,
    /// Override: USD per request.
    request: Option<f64>,
}

/// Parse `PRICING_OVERRIDES` and hold it to `rows`: every key a served row's
/// hash, every value a finite non-negative price of that row's shape.
pub fn parse_pricing_overrides(
    json: &str,
    rows: &[CatalogRow],
) -> Result<HashMap<String, PricingOverride>, String> {
    let overrides: HashMap<String, PricingOverride> = serde_json::from_str(json)
        .map_err(|e| format!("PRICING_OVERRIDES is not a JSON object of overrides: {e}"))?;
    for (key, ovr) in &overrides {
        let row = rows
            .iter()
            .find(|r| r.sha256 == key && !r.lifecycle.is_retired())
            .ok_or_else(|| {
                format!(
                    "PRICING_OVERRIDES names {key:?}, which is not the weights hash of a served \
                     model (keys are weights hashes, 64 lowercase hex digits)"
                )
            })?;
        let per_request = matches!(row.price, RowPrice::PerRequest { .. });
        let shape_ok = if per_request {
            ovr.input.is_none() && ovr.output.is_none() && ovr.request.is_some()
        } else {
            ovr.request.is_none() && (ovr.input.is_some() || ovr.output.is_some())
        };
        if !shape_ok {
            return Err(format!(
                "PRICING_OVERRIDES[{key:?}]: {} is priced {}, so its override takes {}",
                row.sha256,
                if per_request {
                    "per request"
                } else {
                    "per token"
                },
                if per_request {
                    "`request` alone"
                } else {
                    "`input` and/or `output`"
                }
            ));
        }
        for value in [ovr.input, ovr.output, ovr.request].into_iter().flatten() {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!(
                    "PRICING_OVERRIDES[{key:?}]: prices must be finite and non-negative"
                ));
            }
        }
    }
    Ok(overrides)
}

// ---------------------------------------------------------------------------
// The published catalog
// ---------------------------------------------------------------------------

/// The models this gateway sells, as published on `/v1/models`: the served
/// `MODEL_CATALOG` rows (`crate::models`), each with its pin's
/// capabilities and deployments and with prices after markup and any
/// `PRICING_OVERRIDES`.
///
/// **The catalog is a build input, never a boot-time read of anything.** It
/// is what an enclave *publishes* about the models it sells, and clients
/// shape requests from it — whether an agent is offered tools, most of all.
/// Loading it at boot would make that an unmeasured input, so what a client
/// is told a model can do could change without any measurement moving. Only
/// prices can be overridden at runtime.
///
/// **Cached input is not modeled.** The pricing contract (`eidola-common`,
/// and the `ModelPricing` shape clients read from `/v1/models`) has a single
/// prompt-token rate with no cache-hit term. Charging every prompt token at
/// the full input price therefore over-covers a turn whose prefix the engine
/// reused rather than under-covering it, which is the safe direction for both
/// the client's hold and the server's charge; modeling a discount would be a
/// wire-visible contract change on both sides.
pub struct Catalog {
    rows: &'static [CatalogRow],
    /// The published entry of every served row, by weights hash.
    models: Vec<Model>,
}

/// A client's model reference, resolved at the edge.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The published entry: its `sha256` is all that is used downstream.
    pub model: Model,
    /// What a response's `model` echoes: `tag@sha256:<hex>`, or
    /// `sha256:<hex>` for a hash named alone.
    pub echo: String,
}

impl Catalog {
    /// The shipped catalog, at `pricing_markup` (default
    /// [`DEFAULT_PRICING_MARKUP`]) with `pricing_overrides` (the
    /// `PRICING_OVERRIDES` JSON) applied. An override that does not name a
    /// served row by hash, or does not fit its pricing shape, is an error:
    /// the boot refuses it.
    pub fn new(
        pricing_markup: Option<f64>,
        pricing_overrides: Option<&str>,
    ) -> Result<Self, String> {
        Self::build(
            MODEL_CATALOG,
            engine_trust::PINNED_MODELS,
            &|id: &str| {
                engine_trust::accepted_deployments(id)
                    .unwrap_or_else(|e| panic!("the compiled-in pins of {id} do not read: {e}"))
            },
            pricing_markup.unwrap_or(DEFAULT_PRICING_MARKUP),
            pricing_overrides,
        )
    }

    fn build(
        rows: &'static [CatalogRow],
        pins: &[PinnedModel],
        deployments: &dyn Fn(&str) -> Vec<engine_trust::AcceptedDeployment>,
        markup: f64,
        pricing_overrides: Option<&str>,
    ) -> Result<Self, String> {
        let overrides = match pricing_overrides {
            Some(json) => parse_pricing_overrides(json, rows)?,
            None => HashMap::new(),
        };
        let models = rows
            .iter()
            .filter(|row| !row.lifecycle.is_retired())
            .map(|row| {
                let pin = pins
                    .iter()
                    .find(|p| p.id() == row.sha256)
                    .unwrap_or_else(|| panic!("{} has no engine pin", row.sha256));
                published(
                    row,
                    pin,
                    deployments(row.sha256),
                    markup,
                    overrides.get(row.sha256),
                )
            })
            .collect();
        Ok(Self { rows, models })
    }

    /// A catalog over fixture rows and pins, with no deployments.
    #[cfg(test)]
    pub(crate) fn from_parts(
        rows: &'static [CatalogRow],
        pins: &[PinnedModel],
        markup: f64,
        pricing_overrides: Option<&str>,
    ) -> Result<Self, String> {
        Self::build(rows, pins, &|_| Vec::new(), markup, pricing_overrides)
    }

    /// Resolve a client's `model` field (`crate::models::resolve`).
    pub fn resolve(&self, reference: &str) -> Result<ResolvedModel, ModelRefError> {
        let (row, echo) = models::resolve(self.rows, reference)?;
        let model = self
            .models
            .iter()
            .find(|m| m.sha256 == row.sha256)
            .cloned()
            .expect("every served row is published");
        Ok(ResolvedModel { model, echo })
    }

    /// The published list.
    pub fn list(&self) -> ModelsResponse {
        ModelsResponse {
            data: self.models.clone(),
        }
    }
}

/// A served row as `/v1/models` publishes it.
fn published(
    row: &CatalogRow,
    pin: &PinnedModel,
    deployments: Vec<engine_trust::AcceptedDeployment>,
    markup: f64,
    ovr: Option<&PricingOverride>,
) -> Model {
    let pricing = match row.price {
        RowPrice::PerRequest { usd } => {
            let per_req = ovr.and_then(|o| o.request).unwrap_or(usd);
            ModelPricing {
                per_prompt_token: ScaledPrice {
                    value: 0,
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_completion_token: ScaledPrice {
                    value: 0,
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_request: Some(ScaledPrice {
                    value: usd_per_req_to_scaled_credits(per_req, markup),
                    scale_factor: PRICING_SCALE_FACTOR,
                }),
            }
        }
        RowPrice::PerToken {
            input_per_m,
            output_per_m,
        } => {
            let input = ovr.and_then(|o| o.input).unwrap_or(input_per_m);
            let output = ovr.and_then(|o| o.output).unwrap_or(output_per_m);
            ModelPricing {
                per_prompt_token: ScaledPrice {
                    value: usd_per_m_to_scaled_credits(input, markup),
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_completion_token: ScaledPrice {
                    value: usd_per_m_to_scaled_credits(output, markup),
                    scale_factor: PRICING_SCALE_FACTOR,
                },
                per_request: None,
            }
        }
    };
    let (prompt_cache, pinned_weights) = engine_capabilities(pin);
    let lifecycle = match row.lifecycle {
        Lifecycle::Current => ModelLifecycle {
            status: LifecycleStatus::Current,
            retire_after: None,
        },
        Lifecycle::Deprecated { retire_after } => ModelLifecycle {
            status: LifecycleStatus::Deprecated,
            retire_after: Some(retire_after.to_string()),
        },
        Lifecycle::Retired { .. } => unreachable!("a retired row is not published"),
    };
    Model {
        id: row.list_id(),
        sha256: row.sha256.to_string(),
        tags: row.tags.iter().map(|t| (*t).to_string()).collect(),
        architecture: row.architecture.name().to_string(),
        weights: WeightsProvenance {
            repo: row.repo.to_string(),
            revision: row.revision.to_string(),
            files: row
                .files
                .iter()
                .map(|f| WeightsFileEntry {
                    path: f.path.to_string(),
                    size: f.size,
                    sha256: f.sha256.to_string(),
                })
                .collect(),
        },
        lifecycle,
        deployments: deployments
            .into_iter()
            .map(|d| EngineDeploymentPin {
                config_sha256: d.config_sha256,
                measurement: serde_json::to_value(&d.pin).expect("a pin serializes"),
            })
            .collect(),
        name: row.name.to_string(),
        description: row.description.to_string(),
        context_length: row.context_length,
        max_output_tokens: row.max_output_tokens,
        output_budget_class: row.output_budget_class,
        capabilities: ModelCapabilities {
            tool_calling: Capability::new(row.tool_calling),
            reasoning: Capability::new(row.reasoning),
            input_modalities: row.input_modalities.to_vec(),
            output_modalities: row.output_modalities.to_vec(),
            prompt_cache,
            pinned_weights,
        },
        pricing,
    }
}

/// The capabilities a row's pin fixes: prompt-cache retention as the
/// deployments' measured configs set it, and the weights every deployment
/// serves.
fn engine_capabilities(pin: &PinnedModel) -> (PromptCacheCapability, PinnedWeightsCapability) {
    let cache = pin.prompt_cache();
    let prompt_cache = if cache.enabled {
        PromptCacheCapability {
            supported: true,
            idle_ttl_secs: Some(cache.idle_ttl_secs),
            max_age_secs: Some(cache.max_age_secs),
        }
    } else {
        PromptCacheCapability::unsupported()
    };
    let weights = pin.weights();
    let pinned_weights = PinnedWeightsCapability {
        supported: true,
        sha256: Some(weights.sha256.to_string()),
        repo: Some(weights.repo.to_string()),
        revision: Some(weights.revision.to_string()),
    };
    (prompt_cache, pinned_weights)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::engine_trust::{PinnedWeights, PromptCachePolicy};
    use crate::types::{Modality, OutputBudgetClass};

    #[test]
    fn markup_below_safe_cost_factor_is_rejected() {
        // The startup gate: a markup under the contract's safe cost factor
        // (3/2 = 1.5) would sell prompt tokens below cost.
        assert!(validate_pricing_markup(1.2).is_err());
        assert!(validate_pricing_markup(0.0).is_err());
        assert!(validate_pricing_markup(f64::NAN).is_err());
        assert!(validate_pricing_markup(1.5).is_ok());
        assert!(validate_pricing_markup(DEFAULT_PRICING_MARKUP).is_ok());
        assert!(validate_pricing_markup(2.0).is_ok());
    }

    #[test]
    fn test_usd_per_m_to_scaled_credits() {
        // $1.8/M tokens with 1.5x markup.
        // The 1e6 factors (USD→µ$ and /M→/token) cancel:
        // scaled = 1.8 * 1.5 * 1_000_000 = 2_700_000
        assert_eq!(usd_per_m_to_scaled_credits(1.8, 1.5), 2_700_000);

        // $0.15/M with 1.5x markup
        // scaled = 0.15 * 1.5 * 1_000_000 = 225_000
        assert_eq!(usd_per_m_to_scaled_credits(0.15, 1.5), 225_000);

        // Zero price
        assert_eq!(usd_per_m_to_scaled_credits(0.0, 1.5), 0);

        // A rate with no exact binary representation: $0.05/M with 1.5x markup.
        // 0.05 * 1.5 = 0.075; ceil(0.075 * 1e6) = 75_000
        // (may be 75_001 due to f64 representation — ceil rounds up any epsilon)
        let epsilon_rate = usd_per_m_to_scaled_credits(0.05, 1.5);
        assert!(
            epsilon_rate == 75_000 || epsilon_rate == 75_001,
            "got {epsilon_rate}"
        );
    }

    #[test]
    fn test_usd_per_req_to_scaled_credits() {
        // $0.01/req with 1.5x markup
        // credits/req = 0.01 * 1e6 * 1.5 = 15_000
        // scaled = 15_000 * 1e6 = 15_000_000_000
        assert_eq!(usd_per_req_to_scaled_credits(0.01, 1.5), 15_000_000_000);

        // Zero
        assert_eq!(usd_per_req_to_scaled_credits(0.0, 1.5), 0);
    }

    // -----------------------------------------------------------------
    // Fixture rows and pins. Never shipped: the production catalog and
    // its pins are checked by the same functions below, and the fixtures
    // keep every check exercised while the production catalog is empty.
    // -----------------------------------------------------------------

    pub(crate) const ENGINE_SHA: &str =
        "3333333333333333333333333333333333333333333333333333333333333333";
    pub(crate) const STANDARD_SHA: &str =
        "5555555555555555555555555555555555555555555555555555555555555555";
    pub(crate) const RETIRED_SHA: &str =
        "6666666666666666666666666666666666666666666666666666666666666666";
    const PER_REQUEST_SHA: &str =
        "7777777777777777777777777777777777777777777777777777777777777777";

    const fn pin(id: &'static str, cache: bool) -> PinnedModel {
        PinnedModel::fixture(
            id,
            PinnedWeights {
                sha256: id,
                repo: "example/engine-fixture",
                revision: "4444444444444444444444444444444444444444",
            },
            PromptCachePolicy {
                enabled: cache,
                idle_ttl_secs: 900,
                max_age_secs: 7200,
            },
        )
    }

    const ENGINE_PIN: PinnedModel = pin(ENGINE_SHA, true);

    const FILES: &[models::WeightsFile] = &[models::WeightsFile {
        path: "model.safetensors",
        size: 4,
        sha256: "8888888888888888888888888888888888888888888888888888888888888888",
    }];

    const fn row(sha256: &'static str, tags: &'static [&'static str]) -> CatalogRow {
        CatalogRow {
            sha256,
            architecture: eidola_common::architecture::Architecture::MimoV26Flash,
            tags,
            repo: "example/engine-fixture",
            revision: "4444444444444444444444444444444444444444",
            files: FILES,
            name: "Fixture",
            description: "A synthetic row",
            context_length: 8_192,
            tool_calling: true,
            reasoning: true,
            input_modalities: &[Modality::Text],
            output_modalities: &[Modality::Text],
            max_output_tokens: Some(4_096),
            output_budget_class: OutputBudgetClass::Reasoning,
            price: RowPrice::PerToken {
                input_per_m: 1.0,
                output_per_m: 2.0,
            },
            lifecycle: Lifecycle::Current,
        }
    }

    /// A fixture catalog with the variety a real one has: a reasoning model
    /// with a prompt cache, a deprecated standard one without tools or a
    /// cache, and a retired one.
    pub(crate) const FIXTURE_CATALOG: &[CatalogRow] = &[
        CatalogRow {
            name: "Engine Fixture",
            description: "A synthetic reasoning model with tool calling and image input",
            context_length: 1_048_576,
            input_modalities: &[Modality::Text, Modality::Image],
            max_output_tokens: Some(32_768),
            price: RowPrice::PerToken {
                input_per_m: 1.80,
                output_per_m: 5.75,
            },
            ..row(ENGINE_SHA, &["engine-fixture", "engine-latest"])
        },
        CatalogRow {
            name: "Standard Fixture",
            description: "A synthetic model that answers directly and takes no tools",
            context_length: 32_768,
            tool_calling: false,
            reasoning: false,
            output_budget_class: OutputBudgetClass::Standard,
            price: RowPrice::PerToken {
                input_per_m: 0.20,
                output_per_m: 0.60,
            },
            lifecycle: Lifecycle::Deprecated {
                retire_after: "2027-01-01",
            },
            ..row(STANDARD_SHA, &["standard-fixture"])
        },
        CatalogRow {
            lifecycle: Lifecycle::Retired { retire_after: None },
            ..row(RETIRED_SHA, &["retired-fixture"])
        },
    ];
    pub(crate) const FIXTURE_PINS: &[PinnedModel] = &[ENGINE_PIN, pin(STANDARD_SHA, false)];

    fn fixture(markup: f64, overrides: Option<&str>) -> Catalog {
        Catalog::from_parts(FIXTURE_CATALOG, FIXTURE_PINS, markup, overrides).unwrap()
    }

    /// The production catalog and the fixture one, each with its pins: every
    /// check below runs over both.
    fn catalogs() -> [(&'static [CatalogRow], Catalog); 2] {
        [
            (
                MODEL_CATALOG,
                Catalog::new(Some(DEFAULT_PRICING_MARKUP), None).unwrap(),
            ),
            (FIXTURE_CATALOG, fixture(DEFAULT_PRICING_MARKUP, None)),
        ]
    }

    fn served(rows: &[CatalogRow]) -> impl Iterator<Item = &CatalogRow> {
        rows.iter().filter(|r| !r.lifecycle.is_retired())
    }

    /// Every served row reaches the wire **whole**: its hash, tags, source and
    /// manifest, lifecycle, window, capabilities, declared output ceiling, and
    /// prices multiplied by the markup and by nothing else. A retired row is
    /// not listed.
    ///
    /// The table this walks is the catalog itself rather than a second
    /// hand-written copy of the same numbers beside it. Two static copies in
    /// one file agree by construction: they catch a stray edit to one of
    /// them and nothing else.
    #[test]
    fn the_catalog_reaches_the_wire_whole() {
        for (rows, catalog) in catalogs() {
            let models = catalog.list().data;
            assert_eq!(models.len(), served(rows).count(), "catalog size");

            for row in served(rows) {
                let id = row.sha256;
                let model = models
                    .iter()
                    .find(|m| m.sha256 == id)
                    .unwrap_or_else(|| panic!("{id} missing from the published list"));

                assert_eq!(model.id, row.list_id(), "{id} id");
                assert!(!model.id.contains('@'), "{id}: a listed id never holds @");
                assert_eq!(model.tags, row.tags, "{id} tags");
                assert_eq!(model.architecture, row.architecture.name(), "{id} arch");
                assert_eq!(model.weights.repo, row.repo, "{id} repo");
                assert_eq!(model.weights.revision, row.revision, "{id} revision");
                assert_eq!(model.weights.files.len(), row.files.len(), "{id} files");
                for (wire, file) in model.weights.files.iter().zip(row.files) {
                    assert_eq!(
                        (wire.path.as_str(), wire.size, wire.sha256.as_str()),
                        (file.path, file.size, file.sha256)
                    );
                }
                match row.lifecycle {
                    Lifecycle::Current => {
                        assert_eq!(model.lifecycle.status, LifecycleStatus::Current);
                        assert_eq!(model.lifecycle.retire_after, None);
                    }
                    Lifecycle::Deprecated { retire_after } => {
                        assert_eq!(model.lifecycle.status, LifecycleStatus::Deprecated);
                        assert_eq!(model.lifecycle.retire_after.as_deref(), Some(retire_after));
                    }
                    Lifecycle::Retired { .. } => unreachable!(),
                }
                assert_eq!(model.name, row.name, "{id} name");
                assert_eq!(model.description, row.description, "{id} description");
                assert_eq!(model.context_length, row.context_length, "{id} context");
                assert_eq!(
                    model.max_output_tokens, row.max_output_tokens,
                    "{id} ceiling"
                );
                assert_eq!(
                    model.output_budget_class, row.output_budget_class,
                    "{id} budget class"
                );
                assert_eq!(model.capabilities.tool_calling.supported, row.tool_calling);
                assert_eq!(model.capabilities.reasoning.supported, row.reasoning);
                assert_eq!(model.capabilities.input_modalities, row.input_modalities);
                assert_eq!(model.capabilities.output_modalities, row.output_modalities);

                match row.price {
                    RowPrice::PerRequest { usd } => assert_eq!(
                        model.pricing.per_request.as_ref().map(|p| p.value),
                        Some(usd_per_req_to_scaled_credits(usd, DEFAULT_PRICING_MARKUP)),
                        "{id} request price"
                    ),
                    RowPrice::PerToken {
                        input_per_m,
                        output_per_m,
                    } => {
                        assert!(model.pricing.per_request.is_none(), "{id} request price");
                        assert_eq!(
                            model.pricing.per_prompt_token.value,
                            usd_per_m_to_scaled_credits(input_per_m, DEFAULT_PRICING_MARKUP),
                            "{id} prompt price"
                        );
                        assert_eq!(
                            model.pricing.per_completion_token.value,
                            usd_per_m_to_scaled_credits(output_per_m, DEFAULT_PRICING_MARKUP),
                            "{id} completion price"
                        );
                    }
                }
            }
            for retired in rows.iter().filter(|r| r.lifecycle.is_retired()) {
                assert!(
                    !models.iter().any(|m| m.sha256 == retired.sha256),
                    "{} is retired but listed",
                    retired.sha256
                );
            }
        }
    }

    /// This server registers exactly one inference route, so every row it
    /// sells has to be something a conversation can actually be routed to.
    #[test]
    fn every_catalog_row_is_a_chat_model() {
        for (rows, _) in catalogs() {
            for row in rows {
                let id = row.sha256;
                assert!(row.context_length > 0, "{id} declares no context window");
                assert!(
                    row.input_modalities.contains(&Modality::Text)
                        && row.output_modalities.contains(&Modality::Text),
                    "{id} does not take text in and give text back"
                );
                assert!(
                    matches!(row.price, RowPrice::PerToken { .. }),
                    "{id} is priced per request"
                );
            }
        }
    }

    /// A declared output ceiling is a promise about one response, and a
    /// response shares the window with its prompt.
    #[test]
    fn every_declared_output_ceiling_leaves_room_for_a_prompt() {
        for (rows, _) in catalogs() {
            for row in rows {
                if let Some(ceiling) = row.max_output_tokens {
                    assert!(ceiling < row.context_length, "{}", row.sha256);
                }
            }
        }
    }

    #[test]
    fn pricing_overrides_are_keyed_by_weights_hash() {
        let overrides = format!(r#"{{"{ENGINE_SHA}": {{"input": 2.0, "output": 6.0}}}}"#);
        let models = fixture(1.0, Some(&overrides)).list().data;
        let overridden = models.iter().find(|m| m.sha256 == ENGINE_SHA).unwrap();
        assert_eq!(overridden.pricing.per_prompt_token.value, 2_000_000);
        assert_eq!(overridden.pricing.per_completion_token.value, 6_000_000);
        assert!(overridden.pricing.per_request.is_none());

        // A row with no override keeps its catalog price.
        let other = models.iter().find(|m| m.sha256 == STANDARD_SHA).unwrap();
        assert_eq!(
            other.pricing.per_prompt_token.value,
            usd_per_m_to_scaled_credits(0.20, 1.0)
        );
    }

    /// An override the catalog cannot apply refuses the boot rather than
    /// being ignored: a tag, an unknown or retired hash, a price of the wrong
    /// shape, a negative price, an unknown member, or JSON that does not parse.
    #[test]
    fn an_override_that_cannot_apply_fails_loudly() {
        let refused = |json: &str| {
            Catalog::from_parts(FIXTURE_CATALOG, FIXTURE_PINS, 1.0, Some(json))
                .err()
                .unwrap_or_else(|| panic!("{json} was accepted"))
        };
        assert!(refused(r#"{"engine-fixture": {"input": 1.0}}"#).contains("weights hash"));
        assert!(
            refused(&format!(r#"{{"{}": {{"input": 1.0}}}}"#, "9".repeat(64)))
                .contains("weights hash")
        );
        assert!(refused(&format!(r#"{{"{RETIRED_SHA}": {{"input": 1.0}}}}"#)).contains("served"));
        assert!(
            refused(&format!(r#"{{"{ENGINE_SHA}": {{"request": 1.0}}}}"#)).contains("per token")
        );
        assert!(refused(&format!(r#"{{"{ENGINE_SHA}": {{}}}}"#)).contains("per token"));
        assert!(
            refused(&format!(r#"{{"{ENGINE_SHA}": {{"input": -1.0}}}}"#)).contains("non-negative")
        );
        assert!(
            refused(&format!(r#"{{"{ENGINE_SHA}": {{"inptu": 1.0}}}}"#))
                .contains("PRICING_OVERRIDES")
        );
        assert!(refused("not json").contains("PRICING_OVERRIDES"));
        assert!(Catalog::from_parts(FIXTURE_CATALOG, FIXTURE_PINS, 1.0, Some("{}")).is_ok());
    }

    /// The per-request pricing shape, exercised against a **synthetic** row.
    ///
    /// No chat model is priced per request. The shape stays because it is
    /// contract, not because a particular row needs it.
    const PER_REQUEST_FIXTURE: &[CatalogRow] = &[CatalogRow {
        tool_calling: false,
        reasoning: false,
        max_output_tokens: None,
        output_budget_class: OutputBudgetClass::Standard,
        price: RowPrice::PerRequest { usd: 0.01 },
        ..row(PER_REQUEST_SHA, &["per-request-fixture"])
    }];
    const PER_REQUEST_PINS: &[PinnedModel] = &[pin(PER_REQUEST_SHA, false)];

    #[test]
    fn test_per_request_model_pricing() {
        let catalog =
            Catalog::from_parts(PER_REQUEST_FIXTURE, PER_REQUEST_PINS, 1.0, None).unwrap();
        let model = &catalog.list().data[0];
        assert_eq!(model.pricing.per_prompt_token.value, 0);
        assert_eq!(model.pricing.per_completion_token.value, 0);
        // $0.01 * 1e6 * 1.0 * 1e6 = 10_000_000_000
        assert_eq!(
            model.pricing.per_request.as_ref().unwrap().value,
            10_000_000_000
        );

        let overrides = format!(r#"{{"{PER_REQUEST_SHA}": {{"request": 0.02}}}}"#);
        let catalog =
            Catalog::from_parts(PER_REQUEST_FIXTURE, PER_REQUEST_PINS, 1.0, Some(&overrides))
                .unwrap();
        assert_eq!(
            catalog.list().data[0]
                .pricing
                .per_request
                .as_ref()
                .unwrap()
                .value,
            20_000_000_000
        );
        // A zero base price keeps its shape: still per request, and still
        // overridable per request.
        const FREE: &[CatalogRow] = &[CatalogRow {
            price: RowPrice::PerRequest { usd: 0.0 },
            ..PER_REQUEST_FIXTURE[0]
        }];
        let free = Catalog::from_parts(FREE, PER_REQUEST_PINS, 1.0, Some(&overrides)).unwrap();
        let free = &free.list().data[0];
        assert_eq!(
            free.pricing.per_request.as_ref().unwrap().value,
            20_000_000_000
        );
        assert_eq!(free.pricing.per_prompt_token.value, 0);
        let wrong_shape = format!(r#"{{"{PER_REQUEST_SHA}": {{"input": 0.02}}}}"#);
        assert!(
            Catalog::from_parts(
                PER_REQUEST_FIXTURE,
                PER_REQUEST_PINS,
                1.0,
                Some(&wrong_shape)
            )
            .is_err()
        );
    }

    /// References resolve to the hash at the edge, and the echo names the
    /// tag the client used with the hash it resolved to.
    #[test]
    fn a_reference_resolves_to_its_hash() {
        let catalog = fixture(1.5, None);
        let resolved = catalog.resolve("engine-latest").unwrap();
        assert_eq!(resolved.model.sha256, ENGINE_SHA);
        assert_eq!(resolved.echo, format!("engine-latest@sha256:{ENGINE_SHA}"));
        let pinned = catalog
            .resolve(&format!("engine-fixture@sha256:{ENGINE_SHA}"))
            .unwrap();
        assert_eq!(pinned.model.sha256, ENGINE_SHA);
        let bare = catalog.resolve(&format!("sha256:{STANDARD_SHA}")).unwrap();
        assert_eq!(bare.echo, format!("sha256:{STANDARD_SHA}"));
        assert_eq!(
            catalog.resolve("nonexistent").unwrap_err(),
            ModelRefError::Unknown
        );
        assert_eq!(
            catalog.resolve("retired-fixture").unwrap_err(),
            ModelRefError::Retired
        );
        assert_eq!(
            catalog
                .resolve(&format!("engine-fixture@sha256:{STANDARD_SHA}"))
                .unwrap_err(),
            ModelRefError::Mismatch
        );
    }

    /// The shipped catalog publishes exactly its served rows, each pinned,
    /// and with no pinned deployment that is nothing: `/v1/models` is empty
    /// and every chat request is refused.
    #[test]
    fn the_shipped_catalog_lists_only_its_pinned_rows() {
        let catalog = Catalog::new(Some(DEFAULT_PRICING_MARKUP), None).unwrap();
        let listed = catalog.list().data;
        assert_eq!(listed.len(), served(MODEL_CATALOG).count());
        for model in &listed {
            assert!(
                engine_trust::pinned_model(&model.sha256).is_some(),
                "{} is listed without a pin",
                model.sha256
            );
        }
        assert!(catalog.resolve("nonexistent").is_err());
    }

    /// A row publishes its pin's weights and prompt-cache retention.
    #[test]
    fn a_row_publishes_what_its_pin_fixes() {
        let engine = &fixture(1.5, None).list().data[0];
        assert_eq!(
            engine.capabilities.prompt_cache,
            PromptCacheCapability {
                supported: true,
                idle_ttl_secs: Some(900),
                max_age_secs: Some(7200),
            }
        );
        assert_eq!(
            engine.capabilities.pinned_weights,
            PinnedWeightsCapability {
                supported: true,
                sha256: Some(ENGINE_SHA.to_string()),
                repo: Some("example/engine-fixture".to_string()),
                revision: Some("4444444444444444444444444444444444444444".to_string()),
            }
        );
        let wire = serde_json::to_value(engine).unwrap();
        assert_eq!(wire["capabilities"]["prompt_cache"]["idle_ttl_secs"], 900);
        assert_eq!(wire["capabilities"]["pinned_weights"]["supported"], true);
        assert_eq!(wire["sha256"], ENGINE_SHA);
        assert_eq!(wire["lifecycle"], serde_json::json!({"status": "current"}));
    }

    /// A disabled cache publishes no retention bounds.
    #[test]
    fn a_disabled_prompt_cache_is_published_as_unsupported() {
        assert_eq!(
            engine_capabilities(&pin(ENGINE_SHA, false)).0,
            PromptCacheCapability::unsupported()
        );
        let models = fixture(1.5, None).list().data;
        let wire = serde_json::to_value(&models[1]).unwrap();
        assert_eq!(
            wire["capabilities"]["prompt_cache"],
            serde_json::json!({"supported": false})
        );
        assert_eq!(
            wire["lifecycle"],
            serde_json::json!({"status": "deprecated", "retire_after": "2027-01-01"})
        );
    }
}
