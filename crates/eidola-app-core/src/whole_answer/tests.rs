//! The assembled answer, held to the record of what a blocking answer from
//! the gateway was, and the endings that are not one.

use super::*;

/// The inference node's blocking answers to a varied set of requests, each
/// beside the stream the node sent for the same request: scripted engine
/// steps through the node's own response code (content, reasoning, one and
/// several tool calls, `length` endings continuable and not, a stop
/// sequence, an empty answer, escapes and multi-byte text), and requests to
/// the node's dev model on its CPU executor. The gateway answered a blocking
/// request with the node's blocking answer, so this is the record of what a
/// blocking answer looked like.
const PARITY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/blocking-parity.json"
));

struct ParityCase {
    name: String,
    blocking: Value,
    /// The node's stream, `[DONE]` last.
    stream: Vec<String>,
}

fn parity_cases() -> Vec<ParityCase> {
    let cases: Vec<Value> = serde_json::from_str(PARITY).unwrap();
    cases
        .into_iter()
        .map(|c| ParityCase {
            name: c["name"].as_str().unwrap().to_string(),
            blocking: c["blocking"].clone(),
            stream: c["stream"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e.as_str().unwrap().to_string())
                .collect(),
        })
        .collect()
}

/// What the gateway's terminal metadata event carries, and so what a
/// blocking answer carried beside its completion.
fn metadata_members() -> Value {
    serde_json::json!({
        "privacy": {"authorization": {"method": "anonymous_credential", "linkable": false}},
        "verification": {},
        "refunds": [{"refund": "cmVmdW5k", "issuer_key_id": "00"}],
    })
}

/// The node's stream as the gateway forwards it to a client: each chunk
/// through the gateway's own chunk type, the engine's running count kept
/// back (a chunk that is only a count is not sent, and a chunk with choices
/// loses its `usage`), then the metadata event and `[DONE]`.
fn as_the_gateway_forwards(stream: &[String]) -> Vec<String> {
    use eidola_server_gateway::types::ChatCompletionChunk;
    let mut out = Vec::new();
    for event in &stream[..stream.len() - 1] {
        let mut chunk: ChatCompletionChunk = serde_json::from_str(event).unwrap();
        if chunk.usage.is_some() && !chunk.choices.is_empty() {
            let says_nothing = chunk.choices.iter().all(|c| {
                c.finish_reason.is_none()
                    && c.continuable.is_none()
                    && c.delta.role.is_none()
                    && c.delta.content.is_none()
                    && c.delta.reasoning_content.is_none()
                    && c.delta.reasoning.is_none()
                    && c.delta.tool_calls.is_none()
            });
            if says_nothing {
                continue;
            }
            chunk.usage = None;
        }
        out.push(serde_json::to_string(&chunk).unwrap());
    }
    let mut metadata = metadata_members();
    metadata["object"] = METADATA_OBJECT.into();
    metadata["id"] = "".into();
    out.push(metadata.to_string());
    out.push("[DONE]".to_string());
    out
}

/// The node's blocking answer as the gateway answered a blocking request
/// with it: its usage through the gateway's usage type, the Eidola object,
/// and the metadata members.
fn as_the_gateway_answered(blocking: &Value) -> Value {
    let mut v = blocking.clone();
    if let Some(usage) = v.get_mut("usage").and_then(Value::as_object_mut) {
        usage.retain(|k, _| ["prompt_tokens", "completion_tokens", "total_tokens"].contains(&&**k));
    }
    v["object"] = "eidola.chat.completion".into();
    for (k, member) in metadata_members().as_object().unwrap() {
        v[k] = member.clone();
    }
    v
}

/// A completion with what differs between two requests set to fixed values:
/// its `id` and `created`, and each tool call's `id` (a per-response random
/// prefix), by position.
fn normalised(mut v: Value) -> Value {
    v["id"] = "chatcmpl-normalised".into();
    v["created"] = 0.into();
    for choice in v["choices"].as_array_mut().unwrap() {
        if let Some(calls) = choice["message"]["tool_calls"].as_array_mut() {
            for (i, call) in calls.iter_mut().enumerate() {
                call["id"] = format!("call_{i}").into();
            }
        }
    }
    v
}

/// The events, framed as the gateway frames a stream: fixed-size frames, the
/// events waiting plus padding.
fn framed(events: &[String]) -> Vec<u8> {
    use eidola_server_gateway::padding::{Cadence, Framer, StreamEvent};
    let mut framer = Framer::new(Cadence::CLIENT_FACING.frames);
    let mut out = framer.frame();
    for event in events {
        framer.push(&if event == "[DONE]" {
            StreamEvent::Done
        } else {
            StreamEvent::Json(event.clone())
        });
        out.extend(framer.frame());
    }
    while framer.backlog() > 0 {
        out.extend(framer.frame());
    }
    out
}

/// A one-shot loopback server answering `status` with `body` as
/// `content_type`; with `cut`, the response promises more than it sends and
/// the connection closes.
async fn serving_once(
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
    cut: bool,
) -> reqwest::Response {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let Ok((mut socket, _)) = listener.accept().await else {
            return;
        };
        let mut request = Vec::new();
        let mut byte = [0u8; 1];
        while !request.ends_with(b"\r\n\r\n") {
            match socket.read(&mut byte).await {
                Ok(1) => request.push(byte[0]),
                _ => break,
            }
        }
        let length = body.len() + if cut { 4096 } else { 0 };
        let head = format!(
            "HTTP/1.1 {status} OK\r\nContent-Type: {content_type}\r\nContent-Length: {length}\r\n\
             Connection: close\r\n\r\n"
        );
        let _ = socket.write_all(head.as_bytes()).await;
        let _ = socket.write_all(&body).await;
        let _ = socket.shutdown().await;
    });
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    reqwest::Client::builder()
        .build()
        .unwrap()
        .get(format!("http://{addr}/"))
        .send()
        .await
        .unwrap()
}

async fn assembled(events: &[String]) -> Result<Answer, Box<AnswerFailure>> {
    let response = serving_once(200, "text/event-stream", framed(events), false).await;
    WholeAnswer::Assembled
        .read(response, peer_read::MAX_RESPONSE_BYTES, "eidola")
        .await
}

/// **The assembled answer is the blocking answer**, for every case of the
/// record: content, reasoning, tool calls with their fields, the finish
/// reason, `continuable`, usage, and the metadata members a blocking body
/// carried (its refunds among them).
#[tokio::test]
async fn the_assembled_answer_is_the_blocking_answer() {
    let cases = parity_cases();
    assert!(cases.len() >= 10);
    for case in cases {
        let answer = match assembled(&as_the_gateway_forwards(&case.stream)).await {
            Ok(answer) => answer,
            Err(failure) => panic!("{}: {}", case.name, failure.error),
        };
        assert!(!answer.body.over_ceiling);
        let body: Value = serde_json::from_slice(&answer.body.bytes).unwrap();
        assert_eq!(
            normalised(body),
            normalised(as_the_gateway_answered(&case.blocking)),
            "{}",
            case.name
        );
    }
}

/// The record covers what it says: every message field, each finish reason
/// and both `continuable` verdicts.
#[test]
fn the_parity_cases_cover_every_shape() {
    let blocking: Vec<Value> = parity_cases().into_iter().map(|c| c.blocking).collect();
    let choice = |b: &Value| b["choices"][0].clone();
    let any = |f: &dyn Fn(&Value) -> bool| blocking.iter().any(|b| f(&choice(b)));
    assert!(any(&|c| c["message"]["content"].is_string()));
    assert!(any(&|c| c["message"]["content"].is_null()));
    assert!(any(
        &|c| c["message"]["reasoning_content"].is_string() && c["message"]["content"].is_string()
    ));
    assert!(any(&|c| c["message"]["tool_calls"]
        .as_array()
        .map(Vec::len)
        == Some(1)));
    assert!(any(&|c| c["message"]["tool_calls"]
        .as_array()
        .map(Vec::len)
        > Some(1)));
    for reason in ["stop", "length", "tool_calls"] {
        assert!(any(&|c| c["finish_reason"] == reason), "{reason}");
    }
    assert!(any(&|c| c["continuable"] == true));
    assert!(any(&|c| c["continuable"] == false));
}

/// A tool call that arrives in fragments is joined by its index, and its
/// provider fields are kept, by the streaming turn's rule.
#[test]
fn a_tool_call_in_fragments_is_joined_by_its_index() {
    let chunk = |delta: Value| serde_json::json!({"choices": [{"index": 0, "delta": delta}]});
    let mut a = Assembler::default();
    for delta in [
        serde_json::json!({"tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "get_", "arguments": "{\"ci"}}]}),
        serde_json::json!({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "x": 1, "function": {"name": "f", "arguments": "{}"}}]}),
        serde_json::json!({"tool_calls": [{"index": 0, "function": {"name": "weather", "arguments": "ty\": 1}"}}]}),
    ] {
        a.push(&chunk(delta));
    }
    assert_eq!(
        a.finish(None)["choices"][0]["message"]["tool_calls"],
        serde_json::json!([
            {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": 1}"}},
            {"id": "call_b", "type": "function", "x": 1, "function": {"name": "f", "arguments": "{}"}},
        ])
    );
}

/// The content of the stream the gateway cut short.
fn cut_stream() -> Vec<String> {
    let mut metadata = metadata_members();
    metadata["object"] = METADATA_OBJECT.into();
    vec![
        serde_json::json!({"choices": [{"index": 0, "delta": {"content": "Half an an"}}]})
            .to_string(),
        metadata.to_string(),
    ]
}

/// **A stream the gateway cut short is no whole answer**, and its refunds,
/// delivered in band, are handed back to settle the spend; the Record keeps
/// what arrived.
#[tokio::test]
async fn a_stream_cut_short_is_a_failure_carrying_its_refunds() {
    let Err(failure) = assembled(&cut_stream()).await else {
        panic!("a stream with no ending is not an answer");
    };
    assert!(
        failure
            .error
            .to_string()
            .contains("ended without finishing"),
        "{}",
        failure.error
    );
    assert_eq!(failure.refund, Some(metadata_members()["refunds"].clone()));
    let record = String::from_utf8_lossy(&failure.record);
    assert!(record.contains("Half an an"), "{record}");
}

/// A stream that said it finished in a `finish_reason`, without `[DONE]`,
/// is a whole answer, as the streaming twin reads it.
#[tokio::test]
async fn a_finish_reason_without_done_is_a_whole_answer() {
    let events = vec![
        serde_json::json!({"choices": [{"index": 0, "delta": {"content": "Done."}, "finish_reason": "stop"}]})
            .to_string(),
    ];
    let answer = assembled(&events).await.ok().expect("a whole answer");
    let body: Value = serde_json::from_slice(&answer.body.bytes).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Done.");
    assert!(body.get("refunds").is_none());
}

/// A stream whose read fails partway hands back the refunds it delivered
/// before the failure (none here) and what arrived.
#[tokio::test]
async fn a_stream_that_breaks_is_a_failure_keeping_what_arrived() {
    let body = framed(&cut_stream()[..1]);
    let response = serving_once(200, "text/event-stream", body, true).await;
    let Err(failure) = WholeAnswer::Assembled
        .read(response, peer_read::MAX_RESPONSE_BYTES, "eidola")
        .await
    else {
        panic!("a broken stream is not an answer");
    };
    assert!(failure.error.to_string().contains("stream read failed"));
    assert!(String::from_utf8_lossy(&failure.record).contains("Half an an"));
}

/// **The read cap bounds the assembled answer**: a stream whose data passes
/// the caller's ceiling is refused, padding uncounted.
#[tokio::test]
async fn the_ceiling_bounds_the_assembled_answer() {
    let piece = "y".repeat(4096);
    let events: Vec<String> = (0..8)
        .map(|_| {
            serde_json::json!({"choices": [{"index": 0, "delta": {"content": piece}}]}).to_string()
        })
        .chain(["[DONE]".to_string()])
        .collect();
    let response = serving_once(200, "text/event-stream", framed(&events), false).await;
    let Err(failure) = WholeAnswer::Assembled
        .read(response, 16 * 1024, "eidola")
        .await
    else {
        panic!("an answer past the ceiling is refused");
    };
    assert!(
        failure.error.to_string().contains("ceiling"),
        "{}",
        failure.error
    );
    // Under it, the same stream is whole, however many padding frames it took.
    let response = serving_once(200, "text/event-stream", framed(&events), false).await;
    assert!(
        WholeAnswer::Assembled
            .read(response, 64 * 1024, "eidola")
            .await
            .is_ok()
    );
}

/// **The ceiling is on the answer, not on the stream it arrives in.** Every
/// delta comes wrapped in an event many times its size, so a long answer in
/// small deltas is a stream far past the ceiling and an answer well under it;
/// the same stream with the deltas past the ceiling is refused, with the
/// refunds its end carried.
#[tokio::test]
async fn the_ceiling_counts_the_answer_not_its_envelope() {
    let ceiling = 4096;
    let stream = |deltas: usize| -> Vec<String> {
        let mut metadata = metadata_members();
        metadata["object"] = METADATA_OBJECT.into();
        (0..deltas)
            .map(|_| {
                serde_json::json!({
                    "id": "chatcmpl-0123456789abcdef0123456789abcdef",
                    "object": "chat.completion.chunk",
                    "created": 1_700_000_000,
                    "model": "a-model-name-of-some-length@sha256:0123456789abcdef",
                    "choices": [{"index": 0, "delta": {"content": "y"}, "finish_reason": null}],
                })
                .to_string()
            })
            .chain([metadata.to_string(), "[DONE]".to_string()])
            .collect()
    };
    let read = |events: Vec<String>| async move {
        let response = serving_once(200, "text/event-stream", framed(&events), false).await;
        WholeAnswer::Assembled
            .read(response, ceiling, "eidola")
            .await
    };

    let events = stream(128);
    assert!(events.iter().map(String::len).sum::<usize>() > 4 * ceiling);
    let answer = read(events)
        .await
        .ok()
        .expect("an answer under the ceiling");
    let body: Value = serde_json::from_slice(&answer.body.bytes).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "y".repeat(128));

    let Err(failure) = read(stream(1024)).await else {
        panic!("an answer past the ceiling is refused");
    };
    assert!(
        failure.error.to_string().contains("ceiling"),
        "{}",
        failure.error
    );
    assert_eq!(failure.refund, Some(metadata_members()["refunds"].clone()));
}

/// A `2xx` that is not a stream is refused, after its refunds are read.
#[tokio::test]
async fn a_blocking_body_to_a_stream_ask_is_refused_with_its_refunds() {
    let body = serde_json::json!({
        "choices": [{"message": {"role": "assistant", "content": "x"}}],
        "refunds": metadata_members()["refunds"],
    });
    let response = serving_once(
        200,
        "application/json",
        body.to_string().into_bytes(),
        false,
    )
    .await;
    let Err(failure) = WholeAnswer::Assembled
        .read(response, peer_read::MAX_RESPONSE_BYTES, "eidola")
        .await
    else {
        panic!("not the shape asked for");
    };
    assert!(failure.error.to_string().contains("not server-sent events"));
    assert_eq!(failure.refund, Some(metadata_members()["refunds"].clone()));
}

/// A non-2xx is an error document on either transport, read whole with its
/// refunds in it, and a body from any backend but Eidola's is read as it is.
#[tokio::test]
async fn error_documents_and_other_backends_are_read_as_bodies() {
    let body = br#"{"error":{"message":"no"},"refunds":[{"refund":"r"}]}"#.to_vec();
    let response = serving_once(503, "application/json", body.clone(), false).await;
    let answer = WholeAnswer::Assembled
        .read(response, peer_read::MAX_RESPONSE_BYTES, "eidola")
        .await
        .ok()
        .expect("read whole");
    assert_eq!(answer.body.bytes, body);
    for kind in [
        BackendKind::Local,
        BackendKind::OpenAi,
        BackendKind::LlamaCpp,
    ] {
        let whole = WholeAnswer::for_backend(kind);
        assert!(!whole.streams());
        let response = serving_once(200, "application/json", body.clone(), false).await;
        let answer = whole
            .read(response, peer_read::MAX_RESPONSE_BYTES, "other")
            .await
            .ok()
            .expect("read whole");
        assert_eq!(answer.body.bytes, body);
    }
    assert!(WholeAnswer::for_backend(BackendKind::Eidola).streams());
}
