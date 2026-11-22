//! How the gateway's logs name an engine upstream: by its pinned deployment
//! and an opaque id derived from its address, never the address itself.
//! Each probe's lifecycle line carries both, and the node's own
//! `instance_id` (its telemetry's `service.instance.id`), so an operator can
//! go from a placement row to the series that node exports.

use eidola_common::engine_deployment::is_safe_component;
use sha2::{Digest, Sha256};

/// Where a pinned deployment's config lives, relative to the workspace root:
/// `deploy/engine/<name>/<variant>/tinfoil-config.yml`.
const CONFIG_PREFIX: &str = "deploy/engine/";
const CONFIG_SUFFIX: &str = "/tinfoil-config.yml";

/// How many hex digits of a config hash name a deployment whose config path
/// does not give it a name.
const DEPLOYMENT_HASH_DIGITS: usize = 16;

/// How many hex digits of the SHA-256 of an upstream's base URL make its
/// `node` label.
const NODE_ID_DIGITS: usize = 12;

/// The `deployment` label of a pinned deployment: `<name>/<variant>` from
/// its config's path, which the gateway's build holds to
/// `deploy/engine/<name>/<variant>/tinfoil-config.yml` with both components
/// in `[a-z0-9][a-z0-9._-]*`; otherwise the first sixteen hex digits of its
/// config hash.
pub(crate) fn deployment_label(config: Option<&str>, config_sha256: &str) -> String {
    config
        .and_then(|path| path.strip_prefix(CONFIG_PREFIX))
        .and_then(|rest| rest.strip_suffix(CONFIG_SUFFIX))
        .and_then(|rest| rest.split_once('/'))
        .filter(|(name, variant)| is_safe_component(name) && is_safe_component(variant))
        .map_or_else(
            || {
                config_sha256
                    .get(..DEPLOYMENT_HASH_DIGITS)
                    .unwrap_or(config_sha256)
                    .to_string()
            },
            |(name, variant)| format!("{name}/{variant}"),
        )
}

/// The `node` label of the upstream at `base_url` (normalized, as placement
/// gives it): the first twelve hex digits of its SHA-256. Stable across
/// gateway restarts and gateways, and carrying no hostname; the operator
/// maps it back to a placement row by hashing the row's normalized URL.
pub(crate) fn node_id(base_url: &str) -> String {
    let mut id = hex::encode(Sha256::digest(base_url.as_bytes()));
    id.truncate(NODE_ID_DIGITS);
    id
}

/// What one upstream's log lines are labelled by, besides its model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NodeLabels {
    /// [`deployment_label`] of its placement row's deployment.
    pub deployment: String,
    /// [`node_id`] of its base URL.
    pub node: String,
}

impl NodeLabels {
    pub(crate) fn new(deployment: String, base_url: &str) -> Self {
        Self {
            deployment,
            node: node_id(base_url),
        }
    }
}

/// A node's `instance_id` from its info answer, when it is one: 32
/// lowercase hex digits, random per node process. Anything else reads as
/// none, and decides nothing.
pub(crate) fn instance_id(info: &serde_json::Value) -> Option<String> {
    info.get("instance_id")
        .and_then(serde_json::Value::as_str)
        .filter(|id| id.len() == 32 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    const URL_A: &str = "https://engine-a.example";
    const URL_B: &str = "https://engine-b.example";

    #[test]
    fn a_deployment_is_labelled_by_its_config_paths_name_or_its_hash() {
        let hash = "fde1d9f96fa9f20e8cf375e9315719b9c21018e164aa010faaa10178a55570dd";
        assert_eq!(
            deployment_label(
                Some("deploy/engine/mimo-v2.6-flash/b300x1/tinfoil-config.yml"),
                hash
            ),
            "mimo-v2.6-flash/b300x1"
        );
        for config in [
            None,
            Some("deploy/engine/mimo/tinfoil-config.yml"),
            Some("deploy/engine/mimo/b300x1/extra/tinfoil-config.yml"),
            Some("deploy/engine/Mimo/b300x1/tinfoil-config.yml"),
            Some("elsewhere/mimo/b300x1/tinfoil-config.yml"),
        ] {
            assert_eq!(deployment_label(config, hash), "fde1d9f96fa9f20e");
        }
    }

    /// Every compiled-in deployment has a name from its config path.
    #[test]
    fn every_pinned_deployment_has_a_name() {
        for model in crate::engine_trust::PINNED_MODELS {
            for deployment in crate::engine_trust::accepted_deployments(model.id()).unwrap() {
                let label = deployment_label(Some(&deployment.config), &deployment.config_sha256);
                assert_eq!(
                    format!("{CONFIG_PREFIX}{label}{CONFIG_SUFFIX}"),
                    deployment.config
                );
            }
        }
    }

    #[test]
    fn a_node_is_labelled_by_a_prefix_of_its_urls_hash() {
        let id = node_id(URL_A);
        assert_eq!(id.len(), NODE_ID_DIGITS);
        assert!(id.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(hex::encode(Sha256::digest(URL_A.as_bytes())).starts_with(&id));
        assert_eq!(id, node_id(URL_A));
        assert_ne!(id, node_id(URL_B));
    }

    #[test]
    fn an_instance_id_is_read_only_in_its_shape() {
        let id = "0123456789abcdef0123456789abcdef";
        assert_eq!(
            instance_id(&serde_json::json!({"instance_id": id})),
            Some(id.to_string())
        );
        for bad in [
            serde_json::json!({}),
            serde_json::json!({"instance_id": 7}),
            serde_json::json!({"instance_id": "0123456789ABCDEF0123456789ABCDEF"}),
            serde_json::json!({"instance_id": "0123456789abcdef"}),
            serde_json::json!({"instance_id": format!("{id}00")}),
        ] {
            assert_eq!(instance_id(&bad), None, "{bad}");
        }
    }
}
