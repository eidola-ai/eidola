use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Json;
use futures_util::StreamExt;
use proptest::prelude::*;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{Notify, mpsc};
use tokio::time::Instant;

use super::*;
use crate::types::{ChatCompletionChunk, Usage};

// ---------------------------------------------------------------------------
// Parsers a client might use
// ---------------------------------------------------------------------------

/// The data of every event a spec-compliant parser dispatches (the WHATWG
/// event-stream interpretation: lines end at CRLF, LF or CR; a line starting
/// with `:` is a comment; `data` fields are joined with LF; a blank line
/// dispatches the event, unless its data buffer is empty; an unterminated
/// event at the end is discarded).
fn spec_events(bytes: &[u8]) -> Vec<String> {
    let text = std::str::from_utf8(bytes).expect("the stream is UTF-8");
    let mut lines = Vec::new();
    let (mut start, mut i) = (0, 0);
    let b = text.as_bytes();
    while i < b.len() {
        match b[i] {
            b'\r' => {
                lines.push(&text[start..i]);
                i += if b.get(i + 1) == Some(&b'\n') { 2 } else { 1 };
                start = i;
            }
            b'\n' => {
                lines.push(&text[start..i]);
                i += 1;
                start = i;
            }
            _ => i += 1,
        }
    }
    let mut out = Vec::new();
    let mut data = String::new();
    for line in lines {
        if line.is_empty() {
            if !data.is_empty() {
                data.pop();
                out.push(std::mem::take(&mut data));
            }
            continue;
        }
        if line.starts_with(':') {
            continue;
        }
        let (field, value) = match line.find(':') {
            Some(i) => {
                let v = &line[i + 1..];
                (&line[..i], v.strip_prefix(' ').unwrap_or(v))
            }
            None => (line, ""),
        };
        if field == "data" {
            data.push_str(value);
            data.push('\n');
        }
    }
    out
}

/// What a naive "OpenAI-compatible" reader takes: split on blank lines, keep
/// the blocks that start `data: `, stop at `[DONE]`.
fn naive_events(bytes: &[u8]) -> Vec<String> {
    let text = std::str::from_utf8(bytes).expect("the stream is UTF-8");
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        if let Some(data) = block.strip_prefix("data: ") {
            out.push(data.to_string());
            if data == "[DONE]" {
                break;
            }
        }
    }
    out
}

/// An event's data as a client reads it: `[DONE]`, or a JSON value.
#[derive(Debug, PartialEq)]
enum Read {
    Done,
    Json(serde_json::Value),
}

fn read(data: &str) -> Read {
    if data == "[DONE]" {
        Read::Done
    } else {
        Read::Json(serde_json::from_str(data).expect("every dispatched payload is JSON"))
    }
}

fn expected(events: &[StreamEvent]) -> Vec<Read> {
    events
        .iter()
        .map(|e| match e {
            StreamEvent::Json(s) => Read::Json(serde_json::from_str(s).unwrap()),
            StreamEvent::Done => Read::Done,
        })
        .collect()
}

/// Every frame a framer cuts from `events` pushed at once.
fn frames_of(events: &[StreamEvent], frame_bytes: usize) -> Vec<Bytes> {
    let mut framer = Framer::new(frame_bytes);
    for e in events {
        framer.push(e);
    }
    let mut frames = Vec::new();
    while framer.backlog() > 0 {
        frames.push(framer.frame());
    }
    frames
}

fn concat(frames: &[Bytes]) -> Vec<u8> {
    frames.iter().flat_map(|f| f.iter().copied()).collect()
}

// ---------------------------------------------------------------------------
// A stream's events, as the handler produces them
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Delta {
    Content(String),
    Reasoning(String),
    ToolArguments(String),
}

fn chunk_json(delta: &Delta) -> String {
    let delta = match delta {
        Delta::Content(s) => serde_json::json!({ "content": s }),
        Delta::Reasoning(s) => serde_json::json!({ "reasoning_content": s }),
        Delta::ToolArguments(s) => serde_json::json!({
            "tool_calls": [{ "index": 0, "function": { "arguments": s } }]
        }),
    };
    serde_json::json!({
        "id": "chatcmpl-0123456789abcdef0123456789abcdef",
        "object": "chat.completion.chunk",
        "created": 1_728_400_000u64,
        "model": "test-model",
        "choices": [{ "index": 0, "delta": delta, "finish_reason": null }]
    })
    .to_string()
}

fn usage_json(usage: &Usage) -> String {
    serde_json::json!({
        "id": "chatcmpl-0123456789abcdef0123456789abcdef",
        "object": "chat.completion.chunk",
        "created": 1_728_400_000u64,
        "model": "test-model",
        "choices": [],
        "usage": usage,
    })
    .to_string()
}

fn metadata_json(refund_bytes: usize) -> String {
    serde_json::json!({
        "object": "eidola.chat.completion.metadata",
        "id": "chatcmpl-0123456789abcdef0123456789abcdef",
        "refund": { "refund": "r".repeat(refund_bytes), "issuer_key_id": "00ff" },
    })
    .to_string()
}

/// A whole stream: the deltas, a finish chunk, usage, metadata, `[DONE]`.
fn stream_events(deltas: &[Delta], usage: &Usage) -> Vec<StreamEvent> {
    let mut events: Vec<StreamEvent> = deltas
        .iter()
        .map(|d| StreamEvent::Json(chunk_json(d)))
        .collect();
    events.push(StreamEvent::Json(
        serde_json::json!({
            "id": "chatcmpl-0123456789abcdef0123456789abcdef",
            "object": "chat.completion.chunk",
            "created": 1_728_400_000u64,
            "model": "test-model",
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
        })
        .to_string(),
    ));
    events.push(StreamEvent::Json(usage_json(usage)));
    events.push(StreamEvent::Json(metadata_json(3000)));
    events.push(StreamEvent::Done);
    events
}

fn delta_strategy() -> impl Strategy<Value = Delta> {
    // Any text, including the line breaks, quotes and multi-byte characters
    // that change an event's encoded length.
    let text = proptest::collection::vec(any::<char>(), 0..40)
        .prop_map(|chars| chars.into_iter().collect::<String>());
    prop_oneof![
        text.clone().prop_map(Delta::Content),
        text.clone().prop_map(Delta::Reasoning),
        text.prop_map(Delta::ToolArguments),
    ]
}

fn usage_strategy() -> impl Strategy<Value = Usage> {
    (0u32..100_000, 0u32..100_000).prop_map(|(p, c)| Usage {
        prompt_tokens: p,
        completion_tokens: c,
        total_tokens: p + c,
    })
}

proptest! {
    /// Every frame is the frame size, and how many there are is a function of
    /// the stream's total encoded length alone: however the text is split into
    /// tokens, and however long each token is, the frames are the same.
    #[test]
    fn frame_sizes_do_not_depend_on_token_lengths(
        deltas in proptest::collection::vec(delta_strategy(), 0..60),
        usage in usage_strategy(),
        frame_bytes in prop_oneof![Just(FRAME_BYTES), 16usize..600],
    ) {
        let events = stream_events(&deltas, &usage);
        let total: usize = {
            let mut framer = Framer::new(frame_bytes);
            for e in &events {
                framer.push(e);
            }
            framer.backlog()
        };
        let frames = frames_of(&events, frame_bytes);
        prop_assert!(frames.iter().all(|f| f.len() == frame_bytes));
        prop_assert_eq!(frames.len(), total.div_ceil(frame_bytes));
    }

    /// A spec-compliant parser reads exactly the original events, in order,
    /// with nothing extra; so does a naive one that splits on blank lines and
    /// keeps `data: ` blocks.
    #[test]
    fn a_client_reads_exactly_the_original_events(
        deltas in proptest::collection::vec(delta_strategy(), 0..60),
        usage in usage_strategy(),
        frame_bytes in prop_oneof![Just(FRAME_BYTES), 16usize..600],
    ) {
        let events = stream_events(&deltas, &usage);
        let wire = concat(&frames_of(&events, frame_bytes));
        let want = expected(&events);
        let spec: Vec<Read> = spec_events(&wire).iter().map(|d| read(d)).collect();
        prop_assert_eq!(&spec, &want);
        let naive: Vec<Read> = naive_events(&wire).iter().map(|d| read(d)).collect();
        prop_assert_eq!(&naive, &want);
    }

    /// Billing reads usage before framing, and the client reads it after: the
    /// usage a client parses out of the padded stream is the usage the
    /// handler settled on, so the charge is the same either way.
    #[test]
    fn billing_is_identical_with_and_without_padding(
        deltas in proptest::collection::vec(delta_strategy(), 0..20),
        usage in usage_strategy(),
    ) {
        let events = stream_events(&deltas, &usage);
        let wire = concat(&frames_of(&events, FRAME_BYTES));
        let read_back: Vec<Usage> = spec_events(&wire)
            .iter()
            .filter(|d| *d != "[DONE]")
            .filter_map(|d| serde_json::from_str::<ChatCompletionChunk>(d).ok())
            .filter_map(|c| c.usage)
            .collect();
        prop_assert_eq!(read_back.len(), 1);
        let model = crate::handlers::tests::test_model();
        let cost = |u: &Usage| crate::handlers::settled_cost(Some(u), &model, 48, 4096, u128::MAX);
        prop_assert_eq!(cost(&read_back[0]), cost(&usage));
        prop_assert_eq!(read_back[0].prompt_tokens, usage.prompt_tokens);
        prop_assert_eq!(read_back[0].completion_tokens, usage.completion_tokens);
    }
}

/// The frames whose remainder is too small for a comment: one or two bytes
/// left after a JSON event go inside it as trailing whitespace, and after
/// `[DONE]` as blank lines. Both parsers still read exactly the events.
#[test]
fn a_one_or_two_byte_remainder_is_still_padding_every_client_ignores() {
    let frame = 64;
    for room in [1usize, 2] {
        // `data: ` + json + `\n\n` = frame - room.
        let json_len = frame - room - 8;
        let json = format!("{{\"c\":\"{}\"}}", "x".repeat(json_len - 8));
        assert_eq!(json.len(), json_len);
        let events = [
            StreamEvent::Json(json.clone()),
            StreamEvent::Json("{\"n\":1}".into()),
            StreamEvent::Done,
        ];
        // The first event is all that is waiting when its frame is cut.
        let mut framer = Framer::new(frame);
        framer.push(&events[0]);
        let mut frames = vec![framer.frame()];
        framer.push(&events[1]);
        framer.push(&events[2]);
        while framer.backlog() > 0 {
            frames.push(framer.frame());
        }
        assert!(frames.iter().all(|f| f.len() == frame));
        let first = std::str::from_utf8(&frames[0]).unwrap();
        assert_eq!(
            first,
            format!("data: {json}{}\n\n", " ".repeat(room)),
            "trailing whitespace inside the event"
        );
        let wire = concat(&frames);
        assert_eq!(
            spec_events(&wire)
                .iter()
                .map(|d| read(d))
                .collect::<Vec<_>>(),
            expected(&events)
        );
        assert_eq!(
            naive_events(&wire)
                .iter()
                .map(|d| read(d))
                .collect::<Vec<_>>(),
            expected(&events)
        );

        // `[DONE]` last, with one or two bytes to spare.
        let json = format!("{{\"c\":\"{}\"}}", "x".repeat(frame - room - 14 - 8 - 8));
        let events = [StreamEvent::Json(json.clone()), StreamEvent::Done];
        let frames = frames_of(&events, frame);
        assert_eq!(frames.len(), 1);
        let only = std::str::from_utf8(&frames[0]).unwrap();
        assert!(
            only.ends_with(&format!("data: [DONE]\n\n{}", "\n".repeat(room))),
            "{only:?}"
        );
        let wire = concat(&frames);
        assert_eq!(
            spec_events(&wire)
                .iter()
                .map(|d| read(d))
                .collect::<Vec<_>>(),
            expected(&events)
        );
        assert_eq!(
            naive_events(&wire)
                .iter()
                .map(|d| read(d))
                .collect::<Vec<_>>(),
            expected(&events)
        );
    }
}

/// With nothing waiting, a frame is one comment event; an event larger than a
/// frame is carried across frames whole.
#[test]
fn an_empty_tick_is_a_comment_and_a_large_event_spans_frames() {
    let mut framer = Framer::new(64);
    let empty = framer.frame();
    assert_eq!(empty.len(), 64);
    assert!(empty.starts_with(b":") && empty.ends_with(b"\n\n"));
    assert!(spec_events(&empty).is_empty());

    let big = StreamEvent::Json(metadata_json(500));
    framer.push(&big);
    framer.push(&StreamEvent::Done);
    let mut frames = vec![empty];
    while framer.backlog() > 0 {
        frames.push(framer.frame());
    }
    assert!(frames.len() > 8, "the event spans frames");
    assert!(frames.iter().all(|f| f.len() == 64));
    assert_eq!(
        spec_events(&concat(&frames))
            .iter()
            .map(|d| read(d))
            .collect::<Vec<_>>(),
        expected(&[big, StreamEvent::Done])
    );
}

// ---------------------------------------------------------------------------
// Cadence
// ---------------------------------------------------------------------------

const TEST_CADENCE: Cadence = Cadence {
    frame_bytes: 64,
    tick: Duration::from_millis(50),
    max_backlog_bytes: 1 << 20,
};

/// Frames go out on the tick, not when an event arrives: an event sent 7 ms in
/// waits for the 50 ms frame, the tick between events is padding of the same
/// size, and the stream ends with the frame that carries `[DONE]`.
#[tokio::test(start_paused = true)]
async fn frames_go_out_on_ticks_not_on_arrival() {
    let (tx, rx) = mpsc::channel(32);
    let mut stream = PaddedStream::new(rx, TEST_CADENCE);
    let start = Instant::now();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(7)).await;
        tx.send(StreamEvent::Json("{\"a\":1}".into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        tx.send(StreamEvent::Json("{\"b\":2}".into()))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        tx.send(StreamEvent::Done).await.unwrap();
        // Still open: the stream ends on `[DONE]`, not on the channel closing.
        tokio::time::sleep(Duration::from_secs(60)).await;
        drop(tx);
    });

    let mut seen = Vec::new();
    while let Some(Ok(frame)) = stream.next().await {
        seen.push((start.elapsed(), frame));
    }
    let times: Vec<u128> = seen.iter().map(|(t, _)| t.as_millis()).collect();
    assert_eq!(times, [0, 50, 100, 150]);
    assert!(seen.iter().all(|(_, f)| f.len() == 64));
    let per_frame: Vec<Vec<String>> = seen.iter().map(|(_, f)| spec_events(f)).collect();
    assert_eq!(
        per_frame,
        [
            vec![],
            vec!["{\"a\":1}".to_string()],
            vec![],
            vec!["{\"b\":2}".to_string(), "[DONE]".to_string()],
        ]
    );
}

/// A producer that outruns the frames is held at the backlog bound: the stream
/// stops taking events, the channel fills, and the producer waits for frames.
#[tokio::test(start_paused = true)]
async fn a_producer_faster_than_the_frames_waits_at_the_backlog_bound() {
    let cadence = Cadence {
        max_backlog_bytes: 128,
        ..TEST_CADENCE
    };
    let (tx, rx) = mpsc::channel(1);
    let mut stream = PaddedStream::new(rx, cadence);
    let start = Instant::now();
    let producer = tokio::spawn(async move {
        for i in 0..20 {
            // 58 bytes encoded: twenty of them are many frames' worth.
            let json = format!("{{\"i\":{i:02},\"pad\":\"{}\"}}", "p".repeat(33));
            tx.send(StreamEvent::Json(json)).await.unwrap();
        }
        tx.send(StreamEvent::Done).await.unwrap();
        start.elapsed()
    });
    let mut frames = 0;
    while let Some(Ok(frame)) = stream.next().await {
        assert_eq!(frame.len(), 64);
        frames += 1;
    }
    let producer_done = producer.await.unwrap();
    assert!(frames >= (20 * 58 + 14usize).div_ceil(64), "{frames}");
    assert!(
        producer_done >= Duration::from_millis(500),
        "the producer waited for frames: {producer_done:?}"
    );
}

/// A client that goes away drops the stream, and with it the receiver: the
/// producer's next send fails, which is how the handler sees the disconnect.
#[tokio::test]
async fn a_dropped_stream_fails_the_producers_send() {
    let (tx, rx) = mpsc::channel(4);
    let stream = PaddedStream::new(rx, TEST_CADENCE);
    drop(stream);
    assert!(tx.send(StreamEvent::Done).await.is_err());
}

/// Over a real HTTP/1.1 connection, every frame is one chunk of the frame
/// size, written when its tick comes: the client reads padding frames before
/// the producer has sent anything, so a frame is flushed on its own rather
/// than held for the data.
#[tokio::test]
async fn each_tick_is_one_http_chunk_of_the_frame_size() {
    let cadence = Cadence {
        frame_bytes: 256,
        tick: Duration::from_millis(10),
        max_backlog_bytes: 1 << 20,
    };
    let (tx, rx) = mpsc::channel(32);
    let slot = Arc::new(Mutex::new(Some(rx)));
    let app = axum::Router::new().route(
        "/",
        axum::routing::get(move || {
            let rx = slot.lock().unwrap().take().expect("one request");
            async move { padded_sse_response(rx, cadence) }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let events = stream_events(
        &[
            Delta::Content("Hello".into()),
            Delta::Reasoning("thinking about it".into()),
            Delta::ToolArguments("{\"x\":1}".into()),
        ],
        &Usage {
            prompt_tokens: 3,
            completion_tokens: 4,
            total_tokens: 7,
        },
    );
    let release = Arc::new(Notify::new());
    {
        let release = release.clone();
        let events = events.clone();
        tokio::spawn(async move {
            release.notified().await;
            for e in events {
                tx.send(e).await.unwrap();
            }
        });
    }

    let socket = tokio::net::TcpStream::connect(addr).await.unwrap();
    let (read_half, mut write_half) = socket.into_split();
    write_half
        .write_all(b"GET / HTTP/1.1\r\nHost: test\r\n\r\n")
        .await
        .unwrap();
    let mut reader = BufReader::new(read_half);
    let mut head = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        if line == "\r\n" {
            break;
        }
        head.push_str(&line.to_ascii_lowercase());
    }
    assert!(head.contains("content-type: text/event-stream"), "{head}");
    assert!(
        head.contains("cache-control: no-cache, no-transform"),
        "{head}"
    );
    assert!(head.contains("transfer-encoding: chunked"), "{head}");

    let mut sizes = Vec::new();
    let mut body = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let size = usize::from_str_radix(line.trim_end(), 16).unwrap();
        let mut chunk = vec![0; size + 2];
        reader.read_exact(&mut chunk).await.unwrap();
        assert_eq!(&chunk[size..], b"\r\n");
        if size == 0 {
            break;
        }
        sizes.push(size);
        body.extend_from_slice(&chunk[..size]);
        if sizes.len() == 3 {
            assert!(
                spec_events(&body).is_empty(),
                "padding came first, frame by frame"
            );
            release.notify_one();
        }
    }
    assert!(sizes.len() > 3);
    assert!(sizes.iter().all(|&s| s == 256), "{sizes:?}");
    assert_eq!(
        spec_events(&body)
            .iter()
            .map(|d| read(d))
            .collect::<Vec<_>>(),
        expected(&events)
    );
}

// ---------------------------------------------------------------------------
// Blocking bodies
// ---------------------------------------------------------------------------

#[test]
fn body_buckets_are_powers_of_two_from_the_floor() {
    assert_eq!(body_bucket(0), MIN_BODY_BUCKET);
    assert_eq!(body_bucket(MIN_BODY_BUCKET), MIN_BODY_BUCKET);
    assert_eq!(body_bucket(MIN_BODY_BUCKET + 1), 2 * MIN_BODY_BUCKET);
    assert_eq!(body_bucket(100_000), 131_072);
}

/// A JSON body (a completion, or an error carrying a refund) is padded to its
/// bucket and parses to the same value; an event stream is left alone.
#[tokio::test]
async fn a_json_body_is_padded_to_its_bucket_and_parses_the_same() {
    for len in [0usize, 10, 4000, 5000, 70_000] {
        let value =
            serde_json::json!({ "choices": [{ "message": { "content": "x".repeat(len) } }] });
        let response = pad_json_response(Json(value.clone()).into_response()).await;
        let declared: usize = response.headers()[header::CONTENT_LENGTH]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(bytes.len(), declared);
        assert_eq!(bytes.len(), body_bucket(value.to_string().len()));
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
            value
        );
    }

    let error = ServerError::BadRequest {
        message: "unknown model".into(),
    }
    .into_response();
    let status = error.status();
    let padded = pad_json_response(error).await;
    assert_eq!(padded.status(), status);
    let bytes = axum::body::to_bytes(padded.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(bytes.len(), MIN_BODY_BUCKET);

    let (_tx, rx) = mpsc::channel(1);
    let sse = padded_sse_response(rx, TEST_CADENCE);
    let passed = pad_json_response(sse).await;
    assert!(passed.headers().get(header::CONTENT_LENGTH).is_none());
    assert_eq!(
        passed.headers()[header::CONTENT_TYPE],
        HeaderValue::from_static("text/event-stream")
    );
}

/// The chat handler writes a stream only through the framer, and pads every
/// other answer: no axum `Sse` (whose events are relayed as they come), no
/// keep-alive (a comment of its own size, on its own schedule), no event
/// written past the framer. A source scan, because the handler needs a
/// database to run and what is asserted is the absence of the other paths.
#[test]
fn the_chat_handler_answers_only_through_the_padding() {
    let source = include_str!("../handlers.rs");
    let source = &source[..source.find("#[cfg(test)]").unwrap()];
    for absent in ["Sse::new", "KeepAlive", "Event::default", "sse::"] {
        assert!(!source.contains(absent), "{absent} in handlers.rs");
    }
    assert!(source.contains("padding::padded_sse_response(rx, Cadence::CLIENT_FACING)"));
    assert!(source.contains("padding::pad_json_response(response).await"));
}

/// A reader that falls behind (hyper polls the body only when the socket can
/// take more) gets the next frame when it asks and the one after a tick later:
/// missed ticks are not made up as frames written back to back.
#[tokio::test(start_paused = true)]
async fn a_late_reader_never_gets_frames_back_to_back() {
    let (_tx, rx) = mpsc::channel(4);
    let mut stream = PaddedStream::new(rx, TEST_CADENCE);
    let start = Instant::now();
    stream.next().await.unwrap().unwrap();
    tokio::time::sleep(Duration::from_millis(175)).await;
    let mut times = Vec::new();
    for _ in 0..3 {
        stream.next().await.unwrap().unwrap();
        times.push(start.elapsed().as_millis());
    }
    assert_eq!(times, [175, 225, 275]);
}
