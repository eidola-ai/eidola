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
//! and a body in the node's strict request subset: the client's body as sent
//! (kept with its strict parse in a [`ValidatedRequest`], because the node
//! renders the prompt from the original bytes), with the gateway's
//! `stream_options` and the client's `cache_key` written over it
//! ([`engine_request_body`]). The Tinfoil upstream never receives the key:
//! the request type does not serialize it, and only this module writes it.
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

/// A chat request as the client sent it: the strict parse, and the exact
/// bytes it was parsed from.
///
/// The node renders its prompt from the body as sent, through an
/// order-preserving parser: object key order and number spellings inside
/// `messages` and `tools` reach the chat template (a tool schema's property
/// order is part of the prompt). Re-serializing the parse would sort keys and
/// respell numbers, so a request bound for a node keeps its bytes, and the
/// only constructor parses exactly those bytes, so the two cannot disagree.
pub struct ValidatedRequest {
    raw: bytes::Bytes,
    parsed: ChatCompletionRequest,
}

impl ValidatedRequest {
    /// Parse `raw` with the gateway's strict request type, keeping `raw`.
    pub fn from_bytes(raw: bytes::Bytes) -> Result<Self, serde_json::Error> {
        let parsed = serde_json::from_slice(&raw)?;
        Ok(Self { raw, parsed })
    }

    /// The strict parse.
    pub fn request(&self) -> &ChatCompletionRequest {
        &self.parsed
    }
}

/// The body of a chat request to a node: every top-level member of the
/// client's body exactly as sent (bytes, nested key order, number spellings),
/// except the three the gateway decides —
///
/// - `stream` / `stream_options`: for a streaming request, `"stream":true`
///   and `"stream_options":{"include_usage":true}` (usage is what the refund
///   is computed from, as on the Tinfoil path); otherwise neither;
/// - `cache_key`: re-encoded from the validated key, when there is one.
///
/// The returned buffer is scrubbed when dropped, and so are the copies of the
/// key made here: the re-encoded text and the client's member as parsed out
/// of `raw`. `raw` itself is an HTTP-stack buffer, out of reach like the
/// node's own read buffers.
pub fn engine_request_body(request: &ValidatedRequest) -> Zeroizing<Vec<u8>> {
    let members: TopLevel =
        serde_json::from_slice(&request.raw).expect("the bytes the request was parsed from");
    let mut body = Zeroizing::new(Vec::with_capacity(request.raw.len() + 96));
    body.push(b'{');
    let mut first = true;
    let mut member = |body: &mut Vec<u8>, key: &str, value: &[u8]| {
        if !first {
            body.push(b',');
        }
        first = false;
        serde_json::to_writer(&mut *body, key).expect("a string serializes");
        body.push(b':');
        body.extend_from_slice(value);
    };
    for (key, value) in members.0 {
        match key.as_str() {
            "stream" | "stream_options" => {}
            "cache_key" => {
                let mut text: Box<str> = value.into();
                text.zeroize();
            }
            _ => member(&mut body, &key, value.get().as_bytes()),
        }
    }
    let parsed = &request.parsed;
    if parsed.stream {
        member(&mut body, "stream", b"true");
        member(&mut body, "stream_options", br#"{"include_usage":true}"#);
    }
    if let Some(key) = &parsed.cache_key {
        let text =
            Zeroizing::new(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key.as_bytes()));
        let quoted = Zeroizing::new(format!("\"{}\"", text.as_str()));
        member(&mut body, "cache_key", quoted.as_bytes());
    }
    body.push(b'}');
    body
}

/// A JSON object's top-level members in document order, each value kept as
/// the exact text it was written as.
struct TopLevel(Vec<(String, Box<serde_json::value::RawValue>)>);

impl<'de> serde::Deserialize<'de> for TopLevel {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Members;
        impl<'de> serde::de::Visitor<'de> for Members {
            type Value = TopLevel;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<TopLevel, A::Error> {
                let mut members = Vec::new();
                while let Some(member) = map.next_entry()? {
                    members.push(member);
                }
                Ok(TopLevel(members))
            }
        }
        d.deserialize_map(Members)
    }
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
        let with = ValidatedRequest::from_bytes(bytes::Bytes::from(format!(
            r#"{{"model":"fixture-model","messages":[{{"role":"user","content":"hi"}}],"cache_key":"{key}"}}"#
        )))
        .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&engine_request_body(&with)).unwrap();
        assert_eq!(body["cache_key"], key);
        assert_eq!(body["model"], "fixture-model");

        let without = ValidatedRequest::from_bytes(bytes::Bytes::from_static(
            br#"{"model":"fixture-model","messages":[{"role":"user","content":"hi"}]}"#,
        ))
        .unwrap();
        let body = engine_request_body(&without);
        assert_eq!(
            body.as_slice(),
            br#"{"model":"fixture-model","messages":[{"role":"user","content":"hi"}]}"#
        );
    }

    /// Everything but the members the gateway decides reaches the node byte
    /// for byte: nested key order (a tool schema's property order is part of
    /// the prompt) and number spellings (`1.0`, `1e2`) included.
    #[test]
    fn the_body_keeps_the_clients_bytes() {
        let tools = r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"zeta":{"type":"number","default":1.0},"alpha":{"type":"integer","maximum":1e2}}}}}]"#;
        let messages = r#"[{"role":"user","content":"hi","name":"z"}]"#;
        let raw = format!(
            r#"{{"stream_options":{{"include_usage":false}},"tools":{tools},"model":"fixture-model","messages":{messages},"temperature":0.50,"stream":true,"cache_key":"_-0123456789abcdefghijklmnopqrstuvwxyzABCDE"}}"#
        );
        let request = ValidatedRequest::from_bytes(bytes::Bytes::from(raw)).unwrap();
        let body = engine_request_body(&request);
        let text = std::str::from_utf8(&body).unwrap();
        assert_eq!(
            text,
            format!(
                r#"{{"tools":{tools},"model":"fixture-model","messages":{messages},"temperature":0.50,"stream":true,"stream_options":{{"include_usage":true}},"cache_key":"_-0123456789abcdefghijklmnopqrstuvwxyzABCDE"}}"#
            )
        );
        // And it is still a request the strict type accepts.
        serde_json::from_slice::<ChatCompletionRequest>(&body).unwrap();
    }
}
