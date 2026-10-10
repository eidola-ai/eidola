//! `releases/trust/engine-enclaves.json` as typed data: the one parse both
//! readers of the file use.
//!
//! The gateway's build (`manifest.rs`, through `build.rs`) parses the file
//! with this before checking anything else, and the running gateway reads its
//! pins with it (`allowed_measurements`), so a file the build accepts is a
//! file the runtime can read. Every level is strict: unknown members and
//! repeated members are refused, the latter including a model id repeated in
//! `models` (which a plain map would silently collapse to its last value).
//!
//! Compiled into `build.rs` too (`#[path]`), so it uses nothing from the
//! crate and only `serde` and `tinfoil-verifier`, both build-dependencies.
// Each reader uses only some fields; the shape is still the whole file.
#![allow(dead_code)]

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer};
use tinfoil_verifier::AllowedMeasurement;

/// The whole file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EngineEnclaves {
    pub schema_version: u64,
    #[serde(deserialize_with = "unique_keys")]
    pub models: BTreeMap<String, Vec<Deployment>>,
}

/// One accepted deployment of a model.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Deployment {
    pub config: String,
    pub config_sha256: String,
    pub cvm_version: String,
    /// In the attesting client's own shape.
    pub pin: AllowedMeasurement,
    pub weights: Weights,
    pub prompt_cache: PromptCache,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Weights {
    pub sha256: String,
    pub repo: String,
    pub revision: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PromptCache {
    pub enabled: bool,
    pub idle_ttl_secs: u64,
    pub max_age_secs: u64,
}

/// Parse the file strictly.
pub fn parse(json: &[u8]) -> Result<EngineEnclaves, String> {
    serde_json::from_slice(json).map_err(|e| format!("engine-enclaves.json: {e}"))
}

/// A map whose keys must all differ.
fn unique_keys<'de, D, V>(d: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Visitor<V>(std::marker::PhantomData<V>);
    impl<'de, V: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<V> {
        type Value = BTreeMap<String, V>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an object with distinct keys")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, V>()? {
                if out.contains_key(&key) {
                    return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
                }
                out.insert(key, value);
            }
            Ok(out)
        }
    }
    d.deserialize_map(Visitor(std::marker::PhantomData))
}
