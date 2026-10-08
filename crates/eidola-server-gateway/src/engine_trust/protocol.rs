//! What the gateway sends an inference node (`eidola-server-engine`).
//!
//! Every chat request to a node carries:
//!
//! - `Authorization: Bearer <engine token>` — the token the node verifies
//!   against the Argon2id hash in its measured config. It keeps anyone but the
//!   gateway (which bills) from spending the node's compute; no privacy
//!   property depends on it.
//! - [`WEIGHTS_HEADER`] — the weights hash the gateway expects that node to
//!   serve, taken from the compiled-in [`PinnedModel`] and never from
//!   placement data, so a misregistered node refuses the request (412) before
//!   reading its body.
//!
//! and a body in the node's strict request subset: the gateway's
//! [`ChatCompletionRequest`] as it serializes for any upstream, plus the
//! client's `cache_key` when there is one ([`engine_request_body`]). The
//! Tinfoil upstream never receives the key: the request type does not
//! serialize it, and only this module writes it back.
//!
//! Which node serves a model, and the attested client that reaches it, are
//! the router's concern, not this module's.

use base64::Engine as _;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};
use zeroize::{Zeroize, Zeroizing};

pub use eidola_common::engine_protocol::WEIGHTS_HEADER;

use super::PinnedModel;
use crate::types::ChatCompletionRequest;

/// The bearer token the gateway presents to every node. Secret: held only in
/// a scrubbed-on-drop buffer, printed redacted.
pub struct EngineToken(Zeroizing<String>);

impl EngineToken {
    /// Wrap the token. Refuses an empty token, or one that cannot travel in a
    /// header (anything but visible ASCII), without echoing it.
    pub fn new(mut token: String) -> Result<Self, &'static str> {
        if token.is_empty() || !token.bytes().all(|b| b.is_ascii_graphic()) {
            token.zeroize();
            return Err("the engine token must be non-empty visible ASCII");
        }
        Ok(Self(Zeroizing::new(token)))
    }
}

impl std::fmt::Debug for EngineToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("EngineToken(<redacted>)")
    }
}

/// The headers of a chat request to a node serving `model`.
///
/// The authorization value is marked sensitive, so `http`'s `Debug` prints it
/// redacted. The `HeaderValue` itself is an ordinary buffer the HTTP stack
/// copies and frees without scrubbing, as it does the request's body; the
/// formatted intermediate is scrubbed here.
pub fn engine_request_headers(token: &EngineToken, model: &PinnedModel) -> HeaderMap {
    let mut bearer = Zeroizing::new(String::with_capacity(7 + token.0.len()));
    bearer.push_str("Bearer ");
    bearer.push_str(&token.0);
    let mut authorization =
        HeaderValue::from_str(&bearer).expect("EngineToken::new admits only visible ASCII");
    authorization.set_sensitive(true);

    let mut headers = HeaderMap::with_capacity(2);
    headers.insert(AUTHORIZATION, authorization);
    headers.insert(
        HeaderName::from_static(WEIGHTS_HEADER),
        HeaderValue::from_static(model.weights().sha256),
    );
    headers
}

/// The body of a chat request to a node: `request` as it serializes for any
/// upstream, with the client's `cache_key` (if any) written back in its wire
/// form. The returned buffer is scrubbed when dropped, and so is every
/// intermediate copy of the key made here.
pub fn engine_request_body(request: &ChatCompletionRequest) -> Zeroizing<Vec<u8>> {
    let mut body = serde_json::to_value(request).expect("a parsed request re-serializes");
    if let Some(key) = &request.cache_key {
        let text =
            Zeroizing::new(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key.as_bytes()));
        body["cache_key"] = serde_json::Value::String(String::clone(&text));
    }
    let bytes = Zeroizing::new(serde_json::to_vec(&body).expect("a JSON value serializes"));
    if let Some(serde_json::Value::String(text)) = body.get_mut("cache_key") {
        text.zeroize();
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_trust::{PinnedWeights, PromptCachePolicy};

    const WEIGHTS: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn model() -> PinnedModel {
        PinnedModel::fixture(
            "fixture-model",
            PinnedWeights {
                sha256: WEIGHTS,
                repo: "example/fixture",
                revision: "0123456789abcdef0123456789abcdef01234567",
            },
            PromptCachePolicy {
                enabled: true,
                idle_ttl_secs: 900,
                max_age_secs: 7200,
            },
        )
    }

    /// The node reads its weights header by exactly this name, and the shared
    /// constant is what both sides spell it with.
    #[test]
    fn the_weights_header_carries_the_pinned_hash() {
        let token = EngineToken::new("dev-gateway-token".into()).unwrap();
        let headers = engine_request_headers(&token, &model());
        assert_eq!(WEIGHTS_HEADER, "x-eidola-weights-sha256");
        assert_eq!(headers[WEIGHTS_HEADER], WEIGHTS);
        assert_eq!(headers[AUTHORIZATION], "Bearer dev-gateway-token");
        assert!(headers[AUTHORIZATION].is_sensitive());
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn the_token_never_prints() {
        let token = EngineToken::new("dev-gateway-token".into()).unwrap();
        assert!(!format!("{token:?}").contains("dev-gateway-token"));
        let headers = engine_request_headers(&token, &model());
        assert!(!format!("{headers:?}").contains("dev-gateway-token"));
    }

    #[test]
    fn an_unusable_token_is_refused() {
        assert!(EngineToken::new(String::new()).is_err());
        assert!(EngineToken::new("two words".into()).is_err());
        assert!(EngineToken::new("line\nbreak".into()).is_err());
    }

    /// A request carrying a key reaches the node with it, in the shape the
    /// node decodes; one without reaches it without.
    #[test]
    fn the_body_carries_the_cache_key_only_when_the_client_sent_one() {
        let key = "_-0123456789abcdefghijklmnopqrstuvwxyzABCDE";
        let with: ChatCompletionRequest = serde_json::from_str(&format!(
            r#"{{"model":"fixture-model","messages":[{{"role":"user","content":"hi"}}],"cache_key":"{key}"}}"#
        ))
        .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&engine_request_body(&with)).unwrap();
        assert_eq!(body["cache_key"], key);
        assert_eq!(body["model"], "fixture-model");

        let without: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"fixture-model","messages":[{"role":"user","content":"hi"}]}"#,
        )
        .unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&engine_request_body(&without)).unwrap();
        assert!(body.get("cache_key").is_none());
    }
}
