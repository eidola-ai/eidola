//! The node's accepted subset, checked before a request is routed to a node.
//!
//! The gateway's public request type is wider than what an inference node
//! (`eidola-server-engine`) accepts: `tool_choice: "required"` and named
//! functions, a zero completion limit, long or many stop sequences, image
//! parts, unknown keys on a text part, and sampling values the engine cannot
//! define all parse here and are refused there. A request bound for a node is
//! held to the node's rules first, so a publicly valid request the node would
//! refuse is refused here, with the node's reason, and never forwarded.
//!
//! The rules that are a function of a parsed request are
//! `eidola_common::engine_protocol`'s, the functions the node itself calls, so
//! the two sides cannot drift. Two rules are mirrored, each with its source:
//! the content-part shape (the node's `api::ContentPart` denies unknown keys
//! on every variant, while the gateway's type ignores them, so the parts are
//! read again from the request's bytes), and the sampling ranges
//! (`eidola_engine::sampling::SamplingParams::new`).
//!
//! Every refusal is a fixed message: nothing from the request is quoted, so
//! the error is log-safe as well as client-facing.

use eidola_common::engine_protocol::{
    SubsetError, check_max_completion_tokens, check_stop, check_tool_choice,
};
use serde::Deserialize;

use crate::engine_trust::protocol::ValidatedRequest;
use crate::error::ServerError;
use crate::types::StopSequence;

/// Refuse `request` unless an inference node accepts it.
pub fn check(request: &ValidatedRequest) -> Result<(), ServerError> {
    let parsed = request.request();
    let refuse = |e: SubsetError| ServerError::BadRequest {
        message: e.to_string(),
    };
    if parsed.messages.is_empty() {
        return Err(refuse(SubsetError::EmptyMessages));
    }
    check_content_parts(request.raw())?;
    check_max_completion_tokens(parsed.max_completion_tokens).map_err(refuse)?;
    match &parsed.stop {
        None => {}
        Some(StopSequence::Single(s)) => check_stop(std::slice::from_ref(s)).map_err(refuse)?,
        Some(StopSequence::Multiple(v)) => check_stop(v).map_err(refuse)?,
    }
    check_tool_choice(parsed.tool_choice.as_ref()).map_err(refuse)?;
    check_sampling(parsed.temperature, parsed.top_p)
}

/// The engine's sampling ranges (`SamplingParams::new`): a temperature finite
/// and at least 0, a `top_p` in (0, 1]. A JSON number too large for an `f32`
/// parses to infinity, which the node refuses.
fn check_sampling(temperature: Option<f32>, top_p: Option<f32>) -> Result<(), ServerError> {
    if let Some(t) = temperature
        && !(t.is_finite() && t >= 0.0)
    {
        return Err(ServerError::BadRequest {
            message: "temperature must be finite and at least 0".to_string(),
        });
    }
    if let Some(p) = top_p
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(ServerError::BadRequest {
            message: "top_p must be greater than 0 and at most 1".to_string(),
        });
    }
    Ok(())
}

/// The messages as the node reads their content parts. Every other member is
/// ignored here: the gateway's strict type already holds them to the node's
/// shape.
#[derive(Deserialize)]
struct Body<'a> {
    #[serde(borrow)]
    messages: Vec<MessageContent<'a>>,
}

#[derive(Deserialize)]
struct MessageContent<'a> {
    #[serde(borrow, default)]
    content: Option<&'a serde_json::value::RawValue>,
}

/// The node's content part, variant for variant, unknown keys denied on each.
#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum NodeContentPart {
    Text {
        #[allow(dead_code)]
        text: serde::de::IgnoredAny,
    },
    ImageUrl {
        #[allow(dead_code)]
        image_url: serde::de::IgnoredAny,
    },
}

/// Text parts only, each exactly `{type, text}`. The node's template renders
/// parts from the body as sent and reads an `image_url`, `image`, `audio` or
/// `video` key on any part as multimodal content, so it refuses every key it
/// does not name; the gateway's own type ignores such keys, so they are read
/// here from the bytes that will be forwarded.
fn check_content_parts(raw: &[u8]) -> Result<(), ServerError> {
    let unreadable = || ServerError::BadRequest {
        message: "a content part may carry only \"type\" and \"text\"".to_string(),
    };
    let body: Body<'_> = serde_json::from_slice(raw).map_err(|_| unreadable())?;
    for message in &body.messages {
        let Some(content) = message.content else {
            continue;
        };
        if !content.get().starts_with('[') {
            continue;
        }
        let parts: Vec<NodeContentPart> =
            serde_json::from_str(content.get()).map_err(|_| unreadable())?;
        if parts
            .iter()
            .any(|p| matches!(p, NodeContentPart::ImageUrl { .. }))
        {
            return Err(ServerError::BadRequest {
                message: SubsetError::ImageContent.to_string(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(extra: &str, content: &str) -> ValidatedRequest {
        let text =
            format!(r#"{{"model":"m","messages":[{{"role":"user","content":{content}}}]{extra}}}"#);
        ValidatedRequest::from_bytes(bytes::Bytes::from(text)).expect("the gateway accepts it")
    }

    fn refusal(request: &ValidatedRequest) -> String {
        match check(request) {
            Err(ServerError::BadRequest { message }) => message,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// Every request the gateway's type accepts but the node refuses is
    /// refused here, each with the node's reason.
    #[test]
    fn what_the_node_refuses_is_refused_before_routing() {
        let cases: &[(&str, &str, &str)] = &[
            (r#","tool_choice":"required""#, r#""hi""#, "tool_choice"),
            (
                r#","tool_choice":{"type":"function","function":{"name":"f"}}"#,
                r#""hi""#,
                "tool_choice",
            ),
            (r#","max_completion_tokens":0"#, r#""hi""#, "at least 1"),
            (r#","stop":["a","b","c","d","e"]"#, r#""hi""#, "at most 4"),
            (r#","stop":"""#, r#""hi""#, "must not be empty"),
            (r#","temperature":-0.5"#, r#""hi""#, "temperature"),
            (r#","temperature":1e39"#, r#""hi""#, "temperature"),
            (r#","top_p":0"#, r#""hi""#, "top_p"),
            (r#","top_p":1.5"#, r#""hi""#, "top_p"),
            (
                "",
                r#"[{"type":"image_url","image_url":{"url":"https://x"}}]"#,
                "text only",
            ),
            (
                "",
                r#"[{"type":"text","text":"a","image":"x"}]"#,
                "only \"type\" and \"text\"",
            ),
        ];
        for (extra, content, reason) in cases {
            let message = refusal(&request(extra, content));
            assert!(message.contains(reason), "{extra} {content}: {message}");
        }
        let long_stop = format!(r#","stop":["{}"]"#, "x".repeat(257));
        assert!(refusal(&request(&long_stop, r#""hi""#)).contains("256 bytes"));
        let empty = ValidatedRequest::from_bytes(bytes::Bytes::from_static(
            br#"{"model":"m","messages":[]}"#,
        ))
        .unwrap();
        assert!(refusal(&empty).contains("must not be empty"));
    }

    /// What the node accepts passes: its whole accepted shape, explicit nulls
    /// included.
    #[test]
    fn what_the_node_accepts_passes() {
        let accepted: &[(&str, &str)] = &[
            ("", r#""hi""#),
            ("", "null"),
            (
                "",
                r#"[{"type":"text","text":"a"},{"type":"text","text":"b"}]"#,
            ),
            (r#","tool_choice":"auto","tools":[]"#, r#""hi""#),
            (r#","tool_choice":"none""#, r#""hi""#),
            (r#","tool_choice":null"#, r#""hi""#),
            (r#","max_completion_tokens":1"#, r#""hi""#),
            (r#","max_completion_tokens":null,"stop":null"#, r#""hi""#),
            (r#","stop":"\n\n""#, r#""hi""#),
            (r#","stop":["a","b","c","d"]"#, r#""hi""#),
            (r#","temperature":0,"top_p":1"#, r#""hi""#),
            (r#","temperature":2.5,"top_p":0.01"#, r#""hi""#),
            (
                r#","stream":true,"stream_options":{"include_usage":false}"#,
                r#""hi""#,
            ),
        ];
        for (extra, content) in accepted {
            let request = request(extra, content);
            assert!(check(&request).is_ok(), "{extra} {content}");
        }
        let at_cap = format!(r#","stop":["{}"]"#, "x".repeat(256));
        assert!(check(&request(&at_cap, r#""hi""#)).is_ok());
    }

    /// A refusal never quotes the request.
    #[test]
    fn a_refusal_quotes_nothing_from_the_request() {
        let secret = "do-not-echo-this";
        let request = request(
            "",
            &format!(r#"[{{"type":"text","text":"{secret}","{secret}":1}}]"#),
        );
        assert!(!refusal(&request).contains(secret));
        assert!(!check(&request).unwrap_err().to_string().contains(secret));
    }
}
