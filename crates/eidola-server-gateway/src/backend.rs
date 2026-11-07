//! The model catalog, and the shapes an upstream's answer takes.
//!
//! Every model this gateway sells is served by Eidola's own inference engine,
//! in the deployments `releases/trust/engine-enclaves.json` pins for it, and
//! reached through `engine_router`. The catalog is hardcoded and bound to
//! those pins at compile time; only pricing can be overridden at runtime, via
//! `PRICING_OVERRIDES`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::engine_trust::{self, PinnedModel};
use crate::types::{
    Capability, ChatCompletionChunk, ChatCompletionResponse, Modality, Model, ModelCapabilities,
    ModelPricing, ModelsResponse, OutputBudgetClass, PinnedWeightsCapability,
    PromptCacheCapability, ScaledPrice,
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

/// A static model catalog entry: what we sell, at what price, and what it can
/// do.
///
/// The capability fields describe the model the pinned deployments serve. The
/// two output-limit fields are declared by us (see `max_output_tokens`), and
/// the weights and prompt-cache capabilities are not on the row at all: they
/// come from the model's pin.
struct CatalogEntry {
    /// The model id, which is also its key in
    /// `releases/trust/engine-enclaves.json`: the row is served by the engine
    /// deployments pinned for it, and the binding is checked at compile time
    /// (`catalog_binding_problem`).
    id: &'static str,
    name: &'static str,
    description: &'static str,
    context_length: u64,
    /// Whether the model accepts a `tools` request field.
    tool_calling: bool,
    /// Whether the model produces reasoning content before its answer.
    reasoning: bool,
    /// Content kinds the model accepts / produces.
    input_modalities: &'static [Modality],
    output_modalities: &'static [Modality],
    /// The largest completion we are prepared to ask this model for.
    ///
    /// **Declared by us, not transcribed.** It is a ceiling we commit to: a
    /// length the model has been seen to reach comfortably, kept well inside
    /// its context window so a long prompt still leaves room to answer.
    /// Budget ladders live *under* this number; none may cross it.
    max_output_tokens: Option<u64>,
    /// Which public output-budget ladder the model draws from. Declared
    /// rather than derived from `reasoning`, so a model can be moved between
    /// ladders without the catalog having to lie about what it does.
    output_budget_class: OutputBudgetClass,
    /// USD per million input tokens (0.0 for per-request models).
    input_per_m: f64,
    /// USD per million output tokens (0.0 for per-request models).
    output_per_m: f64,
    /// USD per request (0.0 for token-based models).
    per_request_usd: f64,
}

/// Hardcoded model catalog. Model identifiers, descriptions and capabilities
/// are bound to the image contents; only pricing can be overridden at runtime
/// via `PRICING_OVERRIDES`.
///
/// **This is a build input, never a boot-time read of anything.** The catalog
/// is what an enclave *publishes* about the models it sells, and clients shape
/// requests from it — whether an agent is offered tools, most of all. Loading
/// it at boot would make that an unmeasured input, so what a client is told a
/// model can do could change without any measurement moving. Same argument
/// the GUI makes about translations being build inputs rather than runtime
/// loads.
///
/// **Every row is a pinned engine model.** A row is sold only when
/// `releases/trust/engine-enclaves.json` pins deployments for its id, and a
/// pinned id must have a row (`catalog_binding_problem`). No engine
/// deployment is pinned yet, so the catalog is empty: `/v1/models` lists
/// nothing and every chat request is refused as an unknown model. The first
/// row lands with the first measured deployment's pin and its price.
///
/// Prices are USD per million tokens (per request for a per-request model),
/// before `PRICING_MARKUP`.
///
/// **Cached input is not modeled.** The pricing contract (`eidola-common`,
/// and the `ModelPricing` shape clients read from `/v1/models`) has a single
/// prompt-token rate with no cache-hit term. Charging every prompt token at
/// the full input price therefore over-covers a turn whose prefix the engine
/// reused rather than under-covering it, which is the safe direction for both
/// the client's hold and the server's charge; modeling a discount would be a
/// wire-visible contract change on both sides.
const MODEL_CATALOG: &[CatalogEntry] = &[];

/// Why the catalog and the compiled-in engine pins disagree, if they do.
///
/// Every row must have a pin and every pin a row: a row without a pin would
/// publish a model no accepted deployment serves, and a pin without a row
/// would accept deployments for a model nothing sells. A `const` so the
/// production pair is checked by the compiler (the assertion below), and
/// callable at runtime so tests can show it catching fixtures.
const fn catalog_binding_problem(
    catalog: &[CatalogEntry],
    pins: &[PinnedModel],
) -> Option<&'static str> {
    let mut i = 0;
    while i < catalog.len() {
        if !pinned_in(pins, catalog[i].id) {
            return Some("a catalog row has no pin in releases/trust/engine-enclaves.json");
        }
        i += 1;
    }
    let mut j = 0;
    while j < pins.len() {
        let mut sold = false;
        let mut i = 0;
        while i < catalog.len() {
            if str_eq(catalog[i].id, pins[j].id()) {
                sold = true;
            }
            i += 1;
        }
        if !sold {
            return Some(
                "a model pinned in releases/trust/engine-enclaves.json has no catalog row",
            );
        }
        j += 1;
    }
    None
}

const fn pinned_in(pins: &[PinnedModel], id: &str) -> bool {
    let mut j = 0;
    while j < pins.len() {
        if str_eq(pins[j].id(), id) {
            return true;
        }
        j += 1;
    }
    false
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// The compile-time binding: this build's catalog and its engine pins agree,
/// or the gateway does not build.
const _: () =
    if let Some(problem) = catalog_binding_problem(MODEL_CATALOG, engine_trust::PINNED_MODELS) {
        panic!("{}", problem);
    };

/// Why a catalog row's declared worst-case hold is past
/// [`eidola_common::MAX_DECLARED_HOLD`] at `markup`, if one is.
///
/// A row's declared worst case is the hold of a request that fills its
/// context window with prompt and asks for its whole output ceiling, or its
/// flat price when it is priced per request — computed from the scaled
/// integer prices the row publishes, so the number is the one a client
/// holds. Every hold a client makes for a request whose contract-counted
/// prompt fits the window is at most this, and the bound keeps it to one
/// `HOLDS_PER_CREDENTIAL`-th of a credential (see `eidola_common`).
///
/// A `const` so the production catalog at the default markup is checked by
/// the compiler (the assertion below); [`Catalog::new`] runs the same rule
/// over the published prices, so a runtime markup or `PRICING_OVERRIDES`
/// entry that breaks it refuses the boot. A row the bound refuses lowers its
/// ceiling: `max_output ≤ (D / k − context_length × prompt_rate) /
/// completion_rate`.
const fn hold_bound_problem(catalog: &[CatalogEntry], markup: f64) -> Option<&'static str> {
    let mut i = 0;
    while i < catalog.len() {
        let entry = &catalog[i];
        let per_request = if entry.per_request_usd > 0.0 {
            Some(usd_per_req_to_scaled_credits(entry.per_request_usd, markup))
        } else {
            None
        };
        let declared = declared_hold(
            entry.context_length,
            entry.max_output_tokens,
            usd_per_m_to_scaled_credits(entry.input_per_m, markup),
            usd_per_m_to_scaled_credits(entry.output_per_m, markup),
            per_request,
        );
        match declared {
            None => {
                return Some(
                    "a token-priced catalog row declares no max_output_tokens, so its hold is unbounded",
                );
            }
            Some(hold) if hold > eidola_common::MAX_DECLARED_HOLD as u128 => {
                return Some(
                    "a catalog row's worst-case hold exceeds eidola_common::MAX_DECLARED_HOLD; lower its \
                     max_output_tokens to (D/k - context_length * prompt_rate) / completion_rate",
                );
            }
            Some(_) => {}
        }
        i += 1;
    }
    None
}

/// A row's declared worst-case hold in credits, from its scaled integer
/// prices (scale [`PRICING_SCALE_FACTOR`]): the flat price when the row is
/// priced per request, otherwise a full window of prompt plus the full output
/// ceiling. `None` for a token-priced row with no output ceiling, whose hold
/// nothing bounds.
const fn declared_hold(
    context_length: u64,
    max_output_tokens: Option<u64>,
    prompt_value: u64,
    completion_value: u64,
    per_request_value: Option<u64>,
) -> Option<u128> {
    let sf = PRICING_SCALE_FACTOR as u128;
    if let Some(value) = per_request_value {
        return Some((value as u128).div_ceil(sf));
    }
    match max_output_tokens {
        Some(max_output) => Some(eidola_common::token_hold_credits(
            context_length as u128,
            max_output as u128,
            prompt_value as u128,
            completion_value as u128,
            sf,
        )),
        None => None,
    }
}

/// The compile-time hold bound: every row this build sells declares, at the
/// default markup, a worst case within `eidola_common::MAX_DECLARED_HOLD`.
const _: () = if let Some(problem) = hold_bound_problem(MODEL_CATALOG, DEFAULT_PRICING_MARKUP) {
    panic!("{}", problem);
};

/// The same bound over a published list — prices after the runtime markup
/// and every `PRICING_OVERRIDES` entry — naming the model and the numbers.
fn published_hold_bound_problem(models: &[Model]) -> Option<String> {
    let bound = eidola_common::MAX_DECLARED_HOLD as u128;
    models.iter().find_map(|model| {
        let pricing = &model.pricing;
        let declared = declared_hold(
            model.context_length,
            model.max_output_tokens,
            pricing.per_prompt_token.value,
            pricing.per_completion_token.value,
            pricing.per_request.as_ref().map(|p| p.value),
        );
        match declared {
            None => Some(format!(
                "{} is priced per token and declares no max_output_tokens, so its hold is unbounded",
                model.id
            )),
            Some(hold) if hold > bound => Some(format!(
                "{}'s worst-case hold is {hold} credits, past the {bound}-credit bound (one \
                 credential of {} credits over {} holds); lower its price or its max_output_tokens",
                model.id,
                eidola_common::CREDENTIAL_DENOMINATION,
                eidola_common::HOLDS_PER_CREDENTIAL,
            )),
            Some(_) => None,
        }
    })
}

/// Convert USD per million tokens to scaled integer credits, applying markup.
///
/// The 1e6 (USD→µ$) and /1e6 (per-M→per-token) factors cancel, leaving:
/// `scaled = usd_per_million * markup * PRICING_SCALE_FACTOR`
const fn usd_per_m_to_scaled_credits(usd_per_million: f64, markup: f64) -> u64 {
    (usd_per_million * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

/// Convert USD per request to scaled integer credits, applying markup.
const fn usd_per_req_to_scaled_credits(usd_per_request: f64, markup: f64) -> u64 {
    // credits/request = USD * 1e6 (USD→µ$) * markup
    // scaled = credits/request * PRICING_SCALE_FACTOR
    (usd_per_request * 1e6 * markup * PRICING_SCALE_FACTOR as f64).ceil() as u64
}

/// Runtime pricing override for a single model, parsed from `PRICING_OVERRIDES`.
///
/// Example JSON: `{"some-model": {"input": 4.0, "output": 20.0}}`
#[derive(Debug, Deserialize)]
struct PricingOverride {
    /// Override: USD per million input tokens.
    input: Option<f64>,
    /// Override: USD per million output tokens.
    output: Option<f64>,
    /// Override: USD per request.
    request: Option<f64>,
}

// ---------------------------------------------------------------------------
// The published catalog
// ---------------------------------------------------------------------------

/// The models this gateway sells, as published on `/v1/models`: the
/// `MODEL_CATALOG` rows with their pins' capabilities and with prices after
/// markup and any `PRICING_OVERRIDES`.
pub struct Catalog {
    models: Vec<Model>,
}

impl Catalog {
    /// The published catalog at `pricing_markup` (the default when `None`)
    /// with `PRICING_OVERRIDES` applied — refused when any row's worst-case
    /// hold at those prices is past `eidola_common::MAX_DECLARED_HOLD`, so a
    /// runtime price the compile-time check cannot see refuses the boot.
    pub fn new(pricing_markup: Option<f64>) -> Result<Self, String> {
        let markup = pricing_markup.unwrap_or(DEFAULT_PRICING_MARKUP);
        let overrides = Self::parse_pricing_overrides();
        Self::checked(Self::build_model_list(markup, &overrides))
    }

    /// The bound every published list is held to.
    fn checked(models: Vec<Model>) -> Result<Self, String> {
        match published_hold_bound_problem(&models) {
            Some(problem) => Err(problem),
            None => Ok(Self { models }),
        }
    }

    fn parse_pricing_overrides() -> HashMap<String, PricingOverride> {
        match std::env::var("PRICING_OVERRIDES") {
            Ok(json) => match serde_json::from_str(&json) {
                Ok(overrides) => {
                    tracing::info!("Loaded pricing overrides from PRICING_OVERRIDES");
                    overrides
                }
                Err(e) => {
                    tracing::warn!("Failed to parse PRICING_OVERRIDES: {}", e);
                    HashMap::new()
                }
            },
            Err(_) => HashMap::new(),
        }
    }

    fn build_model_list(markup: f64, overrides: &HashMap<String, PricingOverride>) -> Vec<Model> {
        Self::build_list_from(
            MODEL_CATALOG,
            engine_trust::PINNED_MODELS,
            markup,
            overrides,
        )
    }

    /// The catalog → wire projection, over an explicit catalog slice so the
    /// projection can be exercised against fixture rows and pins while the
    /// shipped catalog is empty, and so the per-request pricing shape can be
    /// exercised against a synthetic row. The shape outlives any one model
    /// that uses it (transcription is priced per request, and nothing says
    /// the next thing we sell will not be), so it is kept and tested.
    ///
    /// `pins` supplies each row's weights and prompt-cache capability; the
    /// pair is bound at compile time for the production catalog, and a row
    /// without a pin here is a fixture bug.
    fn build_list_from(
        catalog: &[CatalogEntry],
        pins: &[PinnedModel],
        markup: f64,
        overrides: &HashMap<String, PricingOverride>,
    ) -> Vec<Model> {
        catalog
            .iter()
            .map(|entry| {
                let ovr = overrides.get(entry.id);

                let pricing = if entry.per_request_usd > 0.0 {
                    let per_req = ovr.and_then(|o| o.request).unwrap_or(entry.per_request_usd);
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
                } else {
                    let input = ovr.and_then(|o| o.input).unwrap_or(entry.input_per_m);
                    let output = ovr.and_then(|o| o.output).unwrap_or(entry.output_per_m);
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
                };

                let pin = pins
                    .iter()
                    .find(|p| p.id() == entry.id)
                    .unwrap_or_else(|| panic!("{} has no engine pin", entry.id));
                let (prompt_cache, pinned_weights) = engine_capabilities(pin);

                Model {
                    id: entry.id.to_string(),
                    name: entry.name.to_string(),
                    description: entry.description.to_string(),
                    context_length: entry.context_length,
                    max_output_tokens: entry.max_output_tokens,
                    output_budget_class: entry.output_budget_class,
                    capabilities: ModelCapabilities {
                        tool_calling: Capability::new(entry.tool_calling),
                        reasoning: Capability::new(entry.reasoning),
                        input_modalities: entry.input_modalities.to_vec(),
                        output_modalities: entry.output_modalities.to_vec(),
                        prompt_cache,
                        pinned_weights,
                    },
                    pricing,
                }
            })
            .collect()
    }

    /// A catalog over fixture rows and pins.
    #[cfg(test)]
    fn from_parts(
        catalog: &[CatalogEntry],
        pins: &[PinnedModel],
        markup: f64,
        overrides: &HashMap<String, PricingOverride>,
    ) -> Self {
        Self {
            models: Self::build_list_from(catalog, pins, markup, overrides),
        }
    }

    /// Look up a model by ID.
    pub fn lookup_model(&self, model_id: &str) -> Option<Model> {
        self.models.iter().find(|m| m.id == model_id).cloned()
    }

    /// The published model list.
    pub fn list(&self) -> ModelsResponse {
        ModelsResponse {
            data: self.models.clone(),
        }
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
mod tests {
    use super::*;
    use crate::engine_trust::{PinnedWeights, PromptCachePolicy};

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

    const fn pin(id: &'static str, cache: bool) -> PinnedModel {
        PinnedModel::fixture(
            id,
            PinnedWeights {
                sha256: "3333333333333333333333333333333333333333333333333333333333333333",
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

    const ENGINE_PIN: PinnedModel = pin("engine-fixture", true);

    const fn row(id: &'static str) -> CatalogEntry {
        CatalogEntry {
            id,
            name: "Fixture",
            description: "A synthetic row",
            context_length: 8_192,
            tool_calling: true,
            reasoning: true,
            input_modalities: &[Modality::Text],
            output_modalities: &[Modality::Text],
            max_output_tokens: Some(4_096),
            output_budget_class: OutputBudgetClass::Reasoning,
            input_per_m: 1.0,
            output_per_m: 2.0,
            per_request_usd: 0.0,
        }
    }

    /// A fixture catalog with the variety a real one has: a reasoning model
    /// with a prompt cache, and a standard one without tools or a cache.
    const FIXTURE_CATALOG: &[CatalogEntry] = &[
        CatalogEntry {
            id: "engine-fixture",
            name: "Engine Fixture",
            description: "A synthetic reasoning model with tool calling and image input",
            context_length: 262_144,
            tool_calling: true,
            reasoning: true,
            input_modalities: &[Modality::Text, Modality::Image],
            output_modalities: &[Modality::Text],
            max_output_tokens: Some(32_768),
            output_budget_class: OutputBudgetClass::Reasoning,
            input_per_m: 0.10,
            output_per_m: 0.30,
            per_request_usd: 0.0,
        },
        CatalogEntry {
            id: "standard-fixture",
            name: "Standard Fixture",
            description: "A synthetic model that answers directly and takes no tools",
            context_length: 32_768,
            tool_calling: false,
            reasoning: false,
            input_modalities: &[Modality::Text],
            output_modalities: &[Modality::Text],
            max_output_tokens: Some(4_096),
            output_budget_class: OutputBudgetClass::Standard,
            input_per_m: 0.20,
            output_per_m: 0.60,
            per_request_usd: 0.0,
        },
    ];
    const FIXTURE_PINS: &[PinnedModel] = &[ENGINE_PIN, pin("standard-fixture", false)];

    /// The production catalog and the fixture one, each with its pins: every
    /// check below runs over both.
    fn catalogs() -> [(&'static [CatalogEntry], &'static [PinnedModel]); 2] {
        [
            (MODEL_CATALOG, engine_trust::PINNED_MODELS),
            (FIXTURE_CATALOG, FIXTURE_PINS),
        ]
    }

    #[test]
    fn test_model_catalog_completeness() {
        for (catalog, pins) in catalogs() {
            let models = Catalog::build_list_from(catalog, pins, 1.5, &HashMap::new());
            assert_eq!(models.len(), catalog.len());

            // Verify all models have non-empty fields
            for model in &models {
                assert!(!model.id.is_empty());
                assert!(!model.name.is_empty());
                assert!(!model.description.is_empty());
                assert!(
                    !model.capabilities.input_modalities.is_empty(),
                    "{} declares no input modality",
                    model.id
                );
                assert!(
                    !model.capabilities.output_modalities.is_empty(),
                    "{} declares no output modality",
                    model.id
                );
            }
        }
    }

    /// Every catalog row reaches the wire **whole**: same window, same
    /// capabilities, same declared output ceiling, prices multiplied by the
    /// markup and by nothing else.
    ///
    /// The table this walks is the catalog itself rather than a second
    /// hand-written copy of the same numbers beside it. Two static copies in
    /// one file agree by construction: they catch a stray edit to one of
    /// them and nothing else. Left here is the half a unit test can actually
    /// prove, and the numbers exist once in the tree instead of twice.
    #[test]
    fn the_catalog_reaches_the_wire_whole() {
        for (catalog, pins) in catalogs() {
            let models =
                Catalog::build_list_from(catalog, pins, DEFAULT_PRICING_MARKUP, &HashMap::new());
            assert_eq!(models.len(), catalog.len(), "catalog size");

            for entry in catalog {
                let id = entry.id;
                let model = models
                    .iter()
                    .find(|m| m.id == id)
                    .unwrap_or_else(|| panic!("{id} missing from the published list"));

                assert_eq!(model.name, entry.name, "{id} name");
                assert_eq!(model.description, entry.description, "{id} description");
                assert_eq!(model.context_length, entry.context_length, "{id} context");
                assert_eq!(
                    model.max_output_tokens, entry.max_output_tokens,
                    "{id} output ceiling"
                );
                assert_eq!(
                    model.output_budget_class, entry.output_budget_class,
                    "{id} budget class"
                );
                assert_eq!(
                    model.capabilities.tool_calling.supported, entry.tool_calling,
                    "{id} tool calling"
                );
                assert_eq!(
                    model.capabilities.reasoning.supported, entry.reasoning,
                    "{id} reasoning"
                );
                assert_eq!(
                    model.capabilities.input_modalities, entry.input_modalities,
                    "{id} input modalities"
                );
                assert_eq!(
                    model.capabilities.output_modalities, entry.output_modalities,
                    "{id} output modalities"
                );

                if entry.per_request_usd > 0.0 {
                    assert_eq!(
                        model.pricing.per_request.as_ref().map(|p| p.value),
                        Some(usd_per_req_to_scaled_credits(
                            entry.per_request_usd,
                            DEFAULT_PRICING_MARKUP
                        )),
                        "{id} request price"
                    );
                } else {
                    assert!(model.pricing.per_request.is_none(), "{id} request price");
                    assert_eq!(
                        model.pricing.per_prompt_token.value,
                        usd_per_m_to_scaled_credits(entry.input_per_m, DEFAULT_PRICING_MARKUP),
                        "{id} prompt price"
                    );
                    assert_eq!(
                        model.pricing.per_completion_token.value,
                        usd_per_m_to_scaled_credits(entry.output_per_m, DEFAULT_PRICING_MARKUP),
                        "{id} completion price"
                    );
                }
            }
        }
    }

    /// This server registers exactly one inference route, so every row it
    /// sells has to be something a conversation can actually be routed to.
    ///
    /// The strongest form of "never offer a choice we know to be impossible"
    /// is not to list it: a model with no endpoint here never reaches a picker
    /// to be filtered out of, and no downstream filter is as reliable as the
    /// absence of the row.
    #[test]
    fn every_catalog_row_is_a_chat_model() {
        for (catalog, _) in catalogs() {
            for entry in catalog {
                let id = entry.id;
                assert!(
                    entry.context_length > 0,
                    "{id} declares no context window, so it cannot be a chat model"
                );
                assert!(
                    entry.input_modalities.contains(&Modality::Text)
                        && entry.output_modalities.contains(&Modality::Text),
                    "{id} does not take text in and give text back"
                );
                assert_eq!(
                    entry.per_request_usd, 0.0,
                    "{id} is priced per request, which no chat model here is"
                );
            }
        }
    }

    #[test]
    fn no_model_is_listed_twice() {
        for (catalog, _) in catalogs() {
            for (i, entry) in catalog.iter().enumerate() {
                assert!(
                    !catalog[i + 1..].iter().any(|o| o.id == entry.id),
                    "{} is listed twice",
                    entry.id
                );
            }
        }
    }

    /// A declared output ceiling is a promise about one response, and a
    /// response shares the window with its prompt — so a ceiling at or above
    /// the window promises a turn with no room to ask anything.
    #[test]
    fn every_declared_output_ceiling_leaves_room_for_a_prompt() {
        for (catalog, _) in catalogs() {
            for entry in catalog {
                let Some(ceiling) = entry.max_output_tokens else {
                    continue;
                };
                assert!(
                    ceiling < entry.context_length,
                    "{}'s output ceiling ({ceiling}) does not fit inside its \
                     context window ({})",
                    entry.id,
                    entry.context_length
                );
            }
        }
    }

    #[test]
    fn test_pricing_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "engine-fixture".to_string(),
            PricingOverride {
                input: Some(2.0),
                output: Some(6.0),
                request: None,
            },
        );

        let models = Catalog::build_list_from(FIXTURE_CATALOG, FIXTURE_PINS, 1.0, &overrides);
        let overridden = models.iter().find(|m| m.id == "engine-fixture").unwrap();

        // With 1.0x markup and $2.0/M input override
        assert_eq!(overridden.pricing.per_prompt_token.value, 2_000_000);
        assert_eq!(overridden.pricing.per_completion_token.value, 6_000_000);
        assert!(overridden.pricing.per_request.is_none());

        // A row with no override keeps its catalog price.
        let other = models.iter().find(|m| m.id == "standard-fixture").unwrap();
        assert_eq!(
            other.pricing.per_prompt_token.value,
            usd_per_m_to_scaled_credits(0.20, 1.0)
        );
    }

    /// The per-request pricing shape, exercised against a **synthetic** row.
    ///
    /// No chat model is priced per request. The shape stays because it is
    /// contract, not because a particular row needs it, and a fixture is the
    /// honest way to keep testing a shape that no shipped row wears.
    const PER_REQUEST_FIXTURE: &[CatalogEntry] = &[CatalogEntry {
        id: "per-request-fixture",
        name: "Per-Request Fixture",
        description: "A synthetic row priced per request rather than per token",
        context_length: 8_192,
        tool_calling: false,
        reasoning: false,
        input_modalities: &[Modality::Text],
        output_modalities: &[Modality::Text],
        max_output_tokens: None,
        output_budget_class: OutputBudgetClass::Standard,
        input_per_m: 0.0,
        output_per_m: 0.0,
        per_request_usd: 0.01,
    }];
    const PER_REQUEST_PINS: &[PinnedModel] = &[pin("per-request-fixture", false)];

    #[test]
    fn test_per_request_model_pricing() {
        let overrides = HashMap::new();
        let models =
            Catalog::build_list_from(PER_REQUEST_FIXTURE, PER_REQUEST_PINS, 1.0, &overrides);

        let model = models
            .iter()
            .find(|m| m.id == "per-request-fixture")
            .unwrap();
        assert!(model.pricing.per_request.is_some());
        assert_eq!(model.pricing.per_prompt_token.value, 0);
        assert_eq!(model.pricing.per_completion_token.value, 0);

        // $0.01 * 1e6 * 1.0 * 1e6 = 10_000_000_000
        assert_eq!(
            model.pricing.per_request.as_ref().unwrap().value,
            10_000_000_000
        );
    }

    #[test]
    fn test_per_request_override() {
        let mut overrides = HashMap::new();
        overrides.insert(
            "per-request-fixture".to_string(),
            PricingOverride {
                input: None,
                output: None,
                request: Some(0.02),
            },
        );

        let models =
            Catalog::build_list_from(PER_REQUEST_FIXTURE, PER_REQUEST_PINS, 1.0, &overrides);
        let model = models
            .iter()
            .find(|m| m.id == "per-request-fixture")
            .unwrap();

        // $0.02 * 1e6 * 1.0 * 1e6 = 20_000_000_000
        assert_eq!(
            model.pricing.per_request.as_ref().unwrap().value,
            20_000_000_000
        );
    }

    #[test]
    fn test_lookup_model() {
        let catalog = Catalog::from_parts(FIXTURE_CATALOG, FIXTURE_PINS, 1.5, &HashMap::new());
        assert!(catalog.lookup_model("engine-fixture").is_some());
        assert!(catalog.lookup_model("standard-fixture").is_some());
        assert!(catalog.lookup_model("nonexistent").is_none());
        assert_eq!(catalog.list().data.len(), FIXTURE_CATALOG.len());
    }

    /// The shipped catalog publishes exactly its rows, and a model it does
    /// not list is unknown: with no pinned deployment that is every model,
    /// so `/v1/models` is empty and every chat request is refused.
    #[test]
    fn the_shipped_catalog_lists_only_its_pinned_rows() {
        let catalog = Catalog::new(Some(DEFAULT_PRICING_MARKUP)).expect("the shipped catalog");
        let listed = catalog.list().data;
        assert_eq!(listed.len(), MODEL_CATALOG.len());
        for model in &listed {
            assert!(
                engine_trust::PINNED_MODELS
                    .iter()
                    .any(|p| p.id() == model.id),
                "{} is listed without a pin",
                model.id
            );
        }
        assert!(catalog.lookup_model("nonexistent").is_none());
    }

    /// The production pair is checked by the compiler; this is the same
    /// function shown catching each way the pair can disagree.
    #[test]
    fn the_catalog_and_the_engine_pins_bind_both_ways() {
        assert_eq!(
            catalog_binding_problem(MODEL_CATALOG, engine_trust::PINNED_MODELS),
            None
        );
        assert_eq!(catalog_binding_problem(&[], &[]), None);
        assert_eq!(catalog_binding_problem(FIXTURE_CATALOG, FIXTURE_PINS), None);

        let bound = [row("engine-fixture")];
        assert_eq!(catalog_binding_problem(&bound, &[ENGINE_PIN]), None);

        let unpinned = [row("engine-fixture")];
        assert!(
            catalog_binding_problem(&unpinned, &[])
                .is_some_and(|p| p.contains("catalog row has no pin"))
        );

        let other = [row("other-fixture")];
        assert!(
            catalog_binding_problem(&other, &[ENGINE_PIN])
                .is_some_and(|p| p.contains("catalog row has no pin"))
        );

        assert!(
            catalog_binding_problem(&[], &[ENGINE_PIN])
                .is_some_and(|p| p.contains("has no catalog row"))
        );
    }

    /// The production catalog is checked by the compiler; this is the same
    /// function holding the fixtures and refusing each way a row can break
    /// the hold bound.
    #[test]
    fn every_row_declares_a_hold_within_a_sixteenth_of_a_credential() {
        assert_eq!(
            hold_bound_problem(MODEL_CATALOG, DEFAULT_PRICING_MARKUP),
            None
        );
        for catalog in [FIXTURE_CATALOG, PER_REQUEST_FIXTURE] {
            assert_eq!(hold_bound_problem(catalog, DEFAULT_PRICING_MARKUP), None);
        }

        // A frontier-priced row with a long window: refused.
        let expensive = [CatalogEntry {
            input_per_m: 3.0,
            output_per_m: 15.0,
            context_length: 200_000,
            max_output_tokens: Some(32_768),
            ..row("engine-fixture")
        }];
        assert!(
            hold_bound_problem(&expensive, DEFAULT_PRICING_MARKUP)
                .is_some_and(|p| p.contains("MAX_DECLARED_HOLD"))
        );

        // A token-priced row with no ceiling has no bounded hold at all.
        let unbounded = [CatalogEntry {
            max_output_tokens: None,
            ..row("engine-fixture")
        }];
        assert!(
            hold_bound_problem(&unbounded, DEFAULT_PRICING_MARKUP)
                .is_some_and(|p| p.contains("no max_output_tokens"))
        );

        // A flat price past the bound: refused, whatever its window.
        let flat = [CatalogEntry {
            per_request_usd: 0.05,
            ..PER_REQUEST_FIXTURE[0]
        }];
        assert!(hold_bound_problem(&flat, DEFAULT_PRICING_MARKUP).is_some());
    }

    /// The derived cap is exactly where the bound sits: a row at
    /// `(D/k − context × prompt_rate) / completion_rate` passes, and one
    /// token more is refused.
    #[test]
    fn the_derived_output_cap_is_the_bound() {
        let base = row("engine-fixture");
        let sf = PRICING_SCALE_FACTOR as u128;
        let prompt = usd_per_m_to_scaled_credits(base.input_per_m, DEFAULT_PRICING_MARKUP) as u128;
        let completion =
            usd_per_m_to_scaled_credits(base.output_per_m, DEFAULT_PRICING_MARKUP) as u128;
        let prompt_side = (base.context_length as u128 * prompt).div_ceil(sf);
        let room = eidola_common::MAX_DECLARED_HOLD as u128 - prompt_side;
        let cap = (room * sf / completion) as u64;
        let at_cap = [CatalogEntry {
            max_output_tokens: Some(cap),
            ..row("engine-fixture")
        }];
        assert_eq!(hold_bound_problem(&at_cap, DEFAULT_PRICING_MARKUP), None);
        let past_cap = [CatalogEntry {
            max_output_tokens: Some(cap + 1),
            ..row("engine-fixture")
        }];
        assert!(hold_bound_problem(&past_cap, DEFAULT_PRICING_MARKUP).is_some());
    }

    /// The runtime half: a markup or an override the compile-time check
    /// could not see is refused by the catalog the boot builds, naming the
    /// model.
    #[test]
    fn a_runtime_price_past_the_bound_refuses_the_catalog() {
        let fixtures =
            || Catalog::build_list_from(FIXTURE_CATALOG, FIXTURE_PINS, 1.5, &HashMap::new());
        assert!(Catalog::checked(fixtures()).is_ok());

        let mut overrides = HashMap::new();
        overrides.insert(
            "engine-fixture".to_string(),
            PricingOverride {
                input: Some(1.80),
                output: Some(5.75),
                request: None,
            },
        );
        let overridden = Catalog::build_list_from(FIXTURE_CATALOG, FIXTURE_PINS, 1.5, &overrides);
        let refused = Catalog::checked(overridden)
            .err()
            .expect("an override past the bound");
        assert!(refused.contains("engine-fixture"), "{refused}");

        let marked_up =
            Catalog::build_list_from(FIXTURE_CATALOG, FIXTURE_PINS, 10.0, &HashMap::new());
        assert!(Catalog::checked(marked_up).is_err());
    }

    /// A row publishes its pin's weights and prompt-cache retention.
    #[test]
    fn a_row_publishes_what_its_pin_fixes() {
        let catalog = [row("engine-fixture")];
        let models = Catalog::build_list_from(&catalog, &[ENGINE_PIN], 1.5, &HashMap::new());

        let engine = &models[0];
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
                sha256: Some(ENGINE_PIN.weights().sha256.to_string()),
                repo: Some("example/engine-fixture".to_string()),
                revision: Some("4444444444444444444444444444444444444444".to_string()),
            }
        );
        let wire = serde_json::to_value(engine).unwrap();
        assert_eq!(wire["capabilities"]["prompt_cache"]["idle_ttl_secs"], 900);
        assert_eq!(wire["capabilities"]["pinned_weights"]["supported"], true);
    }

    /// A disabled cache publishes no retention bounds.
    #[test]
    fn a_disabled_prompt_cache_is_published_as_unsupported() {
        assert_eq!(
            engine_capabilities(&pin("engine-fixture", false)).0,
            PromptCacheCapability::unsupported()
        );
        let models = Catalog::build_list_from(FIXTURE_CATALOG, FIXTURE_PINS, 1.5, &HashMap::new());
        let wire = serde_json::to_value(&models[1]).unwrap();
        assert_eq!(
            wire["capabilities"]["prompt_cache"],
            serde_json::json!({"supported": false})
        );
    }
}
