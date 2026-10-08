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

/// A chat request as the client sent it: the strict parse, and the bytes it
/// was parsed from.
///
/// The node renders its prompt from the body as sent, through an
/// order-preserving parser: object key order inside `messages` and `tools`
/// reaches the chat template (a tool schema's property order is part of the
/// prompt), and the gateway's parse (`serde_json`'s sorted maps) does not keep
/// it. So a request bound for a node keeps its bytes, and the only
/// constructor parses exactly those bytes, so the two cannot disagree.
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

/// The body of a chat request to a node.
///
/// **What is forwarded is what was priced.** The pricing contract
/// (`eidola_common::prompt_charge`, computed identically by the client and by
/// the gateway over the parsed request) measures tool schemas and tool calls
/// as their compact `serde_json` serialization. Forwarding the client's raw
/// spellings would let `1.0000000000000000000001` or an escaped string reach
/// the node at more bytes than were priced. So every scalar is re-spelled
/// exactly as `serde_json` serializes the parsed value — the same text the
/// contract measures, and the text Eidola's own client sends — while object
/// key order, the one thing the parse loses that the prompt depends on, is
/// kept from the client's bytes (a key repeated within an object keeps its
/// first position and its last value, `serde_json`'s reading of it). The
/// output is compact; the Tinfoil path forwards the same scalars and the same
/// measures, with keys sorted.
///
/// Three top-level members are the gateway's to decide, and are written last:
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
    let Canonical::Object(members) =
        serde_json::from_slice(&request.raw).expect("the bytes the request was parsed from")
    else {
        unreachable!("a parsed request is a JSON object");
    };
    let mut body = Zeroizing::new(Vec::with_capacity(request.raw.len() + 96));
    body.push(b'{');
    let mut first = true;
    let mut member = |body: &mut Vec<u8>, key: &str, write: &dyn Fn(&mut Vec<u8>)| {
        if !first {
            body.push(b',');
        }
        first = false;
        serde_json::to_writer(&mut *body, key).expect("a string serializes");
        body.push(b':');
        write(body);
    };
    for (key, mut value) in members {
        match key.as_str() {
            "stream" | "stream_options" => {}
            "cache_key" => value.scrub(),
            _ => member(&mut body, &key, &|out| value.write(out)),
        }
    }
    let parsed = &request.parsed;
    if parsed.stream {
        member(&mut body, "stream", &|out| out.extend_from_slice(b"true"));
        member(&mut body, "stream_options", &|out| {
            out.extend_from_slice(br#"{"include_usage":true}"#)
        });
    }
    if let Some(key) = &parsed.cache_key {
        let text =
            Zeroizing::new(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(key.as_bytes()));
        member(&mut body, "cache_key", &|out| {
            out.push(b'"');
            out.extend_from_slice(text.as_bytes());
            out.push(b'"');
        });
    }
    body.push(b'}');
    body
}

/// A JSON value with object members in document order and scalars as
/// `serde_json` reads them.
enum Canonical {
    Scalar(serde_json::Value),
    Array(Vec<Canonical>),
    Object(Vec<(String, Canonical)>),
}

impl Canonical {
    /// The compact serialization: scalars exactly as `serde_json` writes them,
    /// objects in member order.
    fn write(&self, out: &mut Vec<u8>) {
        match self {
            Canonical::Scalar(value) => {
                serde_json::to_writer(&mut *out, value).expect("a scalar serializes")
            }
            Canonical::Array(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    item.write(out);
                }
                out.push(b']');
            }
            Canonical::Object(members) => {
                out.push(b'{');
                for (i, (key, value)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    serde_json::to_writer(&mut *out, key).expect("a string serializes");
                    out.push(b':');
                    value.write(out);
                }
                out.push(b'}');
            }
        }
    }

    /// Zero every string this value holds.
    fn scrub(&mut self) {
        match self {
            Canonical::Scalar(serde_json::Value::String(s)) => s.zeroize(),
            Canonical::Scalar(_) => {}
            Canonical::Array(items) => items.iter_mut().for_each(Canonical::scrub),
            Canonical::Object(members) => members.iter_mut().for_each(|(k, v)| {
                k.zeroize();
                v.scrub();
            }),
        }
    }
}

impl<'de> serde::Deserialize<'de> for Canonical {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Canonical;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a JSON value")
            }
            // Each scalar becomes the `serde_json::Value` `serde_json`'s own
            // visitor builds from the same call, so it serializes identically.
            fn visit_bool<E>(self, v: bool) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(v.into()))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(v.into()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(
                    serde_json::Number::from_f64(v).map_or(serde_json::Value::Null, Into::into),
                ))
            }
            fn visit_str<E>(self, v: &str) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(v.into()))
            }
            fn visit_string<E>(self, v: String) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(v.into()))
            }
            fn visit_unit<E>(self) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(serde_json::Value::Null))
            }
            fn visit_none<E>(self) -> Result<Canonical, E> {
                Ok(Canonical::Scalar(serde_json::Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> Result<Canonical, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Canonical::Array(items))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Canonical, A::Error> {
                let mut members: Vec<(String, Canonical)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, Canonical>()? {
                    match members.iter_mut().find(|(k, _)| *k == key) {
                        Some((_, previous)) => {
                            previous.scrub();
                            *previous = value;
                        }
                        None => members.push((key, value)),
                    }
                }
                Ok(Canonical::Object(members))
            }
        }
        d.deserialize_any(Visitor)
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

    /// The node receives the client's key order (a tool schema's property
    /// order is part of the prompt) with every scalar spelled as the pricing
    /// contract measured it: a long number spelling, an exponent, a trailing
    /// zero, an escaped string and whitespace all arrive in `serde_json`'s
    /// compact form.
    #[test]
    fn the_body_keeps_key_order_and_canonical_scalars() {
        let raw = r#"{ "stream_options":{"include_usage":false},
            "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object",
              "properties":{"zeta":{"type":"number","default":1.0000000000000000000001},
                            "alpha":{"type":"integer","maximum":1e2,"description":"\u0041"}}}}}],
            "model":"fixture-model","messages":[{"role":"user","content":"hi","name":"z"}],
            "temperature":0.50,"stream":true,
            "cache_key":"_-0123456789abcdefghijklmnopqrstuvwxyzABCDE"}"#;
        let request =
            ValidatedRequest::from_bytes(bytes::Bytes::from_static(raw.as_bytes())).unwrap();
        let body = engine_request_body(&request);
        assert_eq!(
            std::str::from_utf8(&body).unwrap(),
            r#"{"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"zeta":{"type":"number","default":1.0},"alpha":{"type":"integer","maximum":100.0,"description":"A"}}}}}],"model":"fixture-model","messages":[{"role":"user","content":"hi","name":"z"}],"temperature":0.5,"stream":true,"stream_options":{"include_usage":true},"cache_key":"_-0123456789abcdefghijklmnopqrstuvwxyzABCDE"}"#
        );
        serde_json::from_slice::<ChatCompletionRequest>(&body).unwrap();
    }

    /// The bytes forwarded are the bytes priced: each forwarded tool schema
    /// and tool call is exactly as long as the pricing contract measured it,
    /// and the forwarded body prices to the same prompt tokens as the request
    /// the gateway charged for.
    #[test]
    fn the_engine_receives_exactly_what_was_priced() {
        let raw = r#"{"model":"m","messages":[
            {"role":"user","content":"\u0068i"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function",
              "function":{"name":"f","arguments":"{}"},"x":1.0000000000000000000001,"x":2E+0}]},
            {"role":"tool","tool_call_id":"c","content":"4"}],
          "tools":[{"type":"function","function":{"name":"f","parameters":{
            "properties":{"b":{"maximum":1.0000000000000000000001e0},"a":{"minimum":-0.0}}}}}]}"#;
        let request =
            ValidatedRequest::from_bytes(bytes::Bytes::from_static(raw.as_bytes())).unwrap();
        let body = engine_request_body(&request);

        // What the gateway priced, through its own pricing function.
        let priced = crate::handlers::chargeable_prompt_tokens_for(request.request());
        // What the node receives, priced by the same contract.
        let forwarded: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let received = eidola_common::prompt_charge(
            forwarded["messages"].as_array().unwrap(),
            forwarded["tools"].as_array().map(Vec::as_slice),
        );
        assert_eq!(received.chargeable_prompt_tokens(), priced);

        // And byte for byte per measured entry: the text the node receives for
        // each tool and tool call is as long as its measure.
        #[derive(serde::Deserialize)]
        struct Raw<'a> {
            #[serde(borrow)]
            tools: Vec<&'a serde_json::value::RawValue>,
            #[serde(borrow)]
            messages: Vec<Message<'a>>,
        }
        #[derive(serde::Deserialize)]
        struct Message<'a> {
            #[serde(borrow, default)]
            tool_calls: Option<Vec<&'a serde_json::value::RawValue>>,
        }
        let sent: Raw<'_> = serde_json::from_slice(&body).unwrap();
        let measured = serde_json::to_value(request.request()).unwrap();
        for (text, value) in sent.tools.iter().zip(measured["tools"].as_array().unwrap()) {
            assert_eq!(
                text.get().len() as u64,
                eidola_common::json_text_bytes(value)
            );
        }
        let calls = sent.messages[1].tool_calls.as_ref().unwrap();
        let measured_calls = measured["messages"][1]["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), measured_calls.len());
        for (text, value) in calls.iter().zip(measured_calls) {
            assert_eq!(
                text.get().len() as u64,
                eidola_common::json_text_bytes(value)
            );
        }
    }
}
