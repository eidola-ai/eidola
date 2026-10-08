//! The Eidola-hosted inference engines this gateway build accepts.
//!
//! `releases/trust/engine-enclaves.json` is a build input: per model id, the
//! engine deployments (`eidola-server-engine` in a confidential VM) a gateway
//! built from this tree will talk to. `build.rs` checks it against the
//! committed deployment configs it names (`manifest.rs`) and embeds it, so
//! what this module exposes is a function of the gateway's own measured
//! source:
//!
//! - [`PINNED_MODELS`] — each pinned model's weights (the engine's weights
//!   hash and their `repo@revision` provenance) and prompt-cache retention, as
//!   the deployments' measured configs set them. The catalog
//!   (`backend::MODEL_CATALOG`) is bound to this list at compile time: an
//!   Eidola-hosted row without a pin, or a pin without such a row, does not
//!   build.
//! - [`allowed_measurements`] — the attestation pins for one model's
//!   deployments, in `tinfoil-verifier`'s shape.
//! - [`protocol`] — the headers and body the gateway sends a node.
//!
//! **Same-sha pinning.** A gateway build accepts exactly the engine
//! deployments committed in its own tree: there is no list of previous
//! measurements and nothing that widens the set at runtime. Rolling an engine
//! forward is therefore ordered: new engines deploy beside the old, a gateway
//! pinning the new ones ships, the client pinning that gateway ships, traffic
//! shifts, and only then do the old engines retire.
//!
//! **Weights are compiled in.** The weights hash the gateway sends a node, and
//! the one it publishes to clients, come from [`PINNED_MODELS`] and nowhere
//! else: placement data (which node serves which model) can choose a node,
//! never what it must be serving.

use tinfoil_verifier::measurement::AllowedMeasurement;

pub(crate) mod file;
#[cfg(test)]
pub(crate) mod manifest;
pub mod protocol;

include!(concat!(env!("OUT_DIR"), "/engine_enclaves.gen.rs"));

/// A model this gateway build pins, with what its deployments' configs fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedModel {
    id: &'static str,
    weights: PinnedWeights,
    prompt_cache: PromptCachePolicy,
    deployments: &'static [&'static str],
}

/// A pinned model's weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinnedWeights {
    /// The engine's weights hash (`eidola-server-engine` → The weights hash),
    /// lowercase hex.
    pub sha256: &'static str,
    /// The repository the weights were taken from (`<owner>/<name>`).
    pub repo: &'static str,
    /// The repository revision (a 40-hex-digit commit).
    pub revision: &'static str,
}

/// A pinned model's prompt-cache retention, as its measured configs set it
/// (`EIDOLA_ENGINE_PREFIX_CACHE`, `EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS`,
/// `EIDOLA_ENGINE_CACHE_MAX_AGE_SECS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PromptCachePolicy {
    pub enabled: bool,
    pub idle_ttl_secs: u64,
    pub max_age_secs: u64,
}

impl PinnedModel {
    /// The model id, as the catalog and the node both name it.
    pub const fn id(&self) -> &'static str {
        self.id
    }

    /// The weights every deployment of this model serves.
    pub const fn weights(&self) -> &PinnedWeights {
        &self.weights
    }

    /// The prompt-cache retention every deployment of this model applies.
    pub const fn prompt_cache(&self) -> &PromptCachePolicy {
        &self.prompt_cache
    }

    /// The committed config of every accepted deployment.
    pub const fn deployments(&self) -> &'static [&'static str] {
        self.deployments
    }

    /// A pin that is not in the compiled-in list, for tests of what consumes
    /// one.
    #[cfg(test)]
    pub(crate) const fn fixture(
        id: &'static str,
        weights: PinnedWeights,
        prompt_cache: PromptCachePolicy,
    ) -> Self {
        Self {
            id,
            weights,
            prompt_cache,
            deployments: &[],
        }
    }
}

/// The compiled-in pin for `model_id`, if this build pins it.
pub fn pinned_model(model_id: &str) -> Option<&'static PinnedModel> {
    PINNED_MODELS.iter().find(|m| m.id == model_id)
}

/// The attestation pins for every accepted deployment of `model_id`, in
/// `tinfoil-verifier`'s shape. Empty when this build pins no such model.
///
/// The file was checked when the gateway was built, so a failure here means
/// a pin `tinfoil-verifier` cannot read even though its shape passed: the
/// returned error names the model.
pub fn allowed_measurements(model_id: &str) -> Result<Vec<AllowedMeasurement>, String> {
    allowed_measurements_in(ENGINE_ENCLAVES_JSON, model_id)
}

fn allowed_measurements_in(json: &str, model_id: &str) -> Result<Vec<AllowedMeasurement>, String> {
    // The parse the build check also runs first, so a file the build accepted
    // reads here.
    let file = file::parse(json.as_bytes())?;
    Ok(file
        .models
        .into_iter()
        .filter(|(id, _)| id == model_id)
        .flat_map(|(_, deployments)| deployments.into_iter().map(|d| d.pin))
        .collect())
}

#[cfg(test)]
mod tests;
