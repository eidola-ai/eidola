//! A batch client of the node's chat API (development only; nothing in it is part of the
//! node or its image): it sends chat requests from a JSONL file to a running node as the
//! gateway does, reads each framed stream back with the shared reader, and writes one JSONL
//! line per request.
//!
//! ```sh
//! GATEWAY_TOKEN=<token> cargo run --release -p eidola-server-engine --example chat_batch -- \
//!     <node URL> <weights sha256> <requests.jsonl> <out.jsonl> [--concurrency N]
//! ```
//!
//! **Input.** Each line is a JSON object: a request body in the node's subset without
//! `model`, `stream` and `stream_options`, which this sets (the model `/v1/engine/info`
//! reports, and the stream with usage every request asks for), plus an optional string `id`,
//! echoed in the output (the line's 0-based number otherwise). Every other member is sent in
//! the order and spelling the line has it (the chat crate's order-preserving JSON), since the
//! template renders tools and arguments from the bytes sent. Each body is first checked by the
//! node's own parser (`api::parse_request`): a line the node would refuse is reported, not
//! sent.
//!
//! **The node.** The bearer token is `GATEWAY_TOKEN` from the environment; every request names
//! the weights hash given (the `X-Eidola-Weights-Sha256` header), which must also be the hash
//! `/v1/engine/info` reports, or nothing is sent. A request refused `overloaded` (before
//! admission) is sent again after a pause.
//!
//! **Output.** One line per request, written as it finishes (not in input order):
//! `{"id", "status", "seconds", "response", "error"}`. `response` is the stream joined into
//! one `chat.completion` (`content`, `reasoning_content`, `tool_calls`, `finish_reason`,
//! `continuable`, `usage`); `error` is the node's error body (a refusal, or an `event: error`
//! ending a stream) or this client's own failure; `status` is the HTTP status, null when
//! nothing was sent. A rerun onto the same output file skips every id it already holds a
//! response for and appends the rest, so an interrupted batch resumes; a reader takes the last
//! line for an id. The totals go to standard error at the end.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

use eidola_common::engine_protocol::{WEIGHTS_HEADER, error_type};
use eidola_common::stream_shape::{StreamEvent, StreamReader};
use eidola_engine_chat::json::{self as chat_json, DumpOptions, Json};
use futures_util::StreamExt;
use serde_json::{Value, json};

/// Members this client sets on every body; an input line naming one is refused.
const SET_HERE: [&str; 3] = ["model", "stream", "stream_options"];
/// How often a request refused `overloaded` is sent again, and the pause before each.
const OVERLOADED_RETRIES: u32 = 20;
const OVERLOADED_PAUSE: Duration = Duration::from_secs(2);

fn fail(message: impl std::fmt::Display) -> ! {
    eprintln!("chat_batch: {message}");
    std::process::exit(1)
}

/// One request of the batch: its id and the body as sent.
struct Job {
    id: String,
    body: String,
}

/// What became of one request.
struct Outcome {
    status: Option<u16>,
    response: Option<Value>,
    error: Option<Value>,
}

impl Outcome {
    fn client_error(status: Option<u16>, message: impl std::fmt::Display) -> Self {
        Outcome {
            status,
            response: None,
            error: Some(json!({"error": {"message": message.to_string(), "type": "client"}})),
        }
    }
}

/// The body sent for an input line: `model` first, then the line's members in its order
/// without `id`, then the stream settings, as compact JSON.
fn job(index: usize, line: &str, model: &str) -> Job {
    let at = |m: &str| -> Job { fail(format!("line {}: {m}", index + 1)) };
    let parsed = match chat_json::parse_strict(line) {
        Ok(p) => p,
        Err(e) => return at(&e.to_string()),
    };
    let Json::Object(members) = parsed else {
        return at("not a JSON object");
    };
    let mut id = index.to_string();
    let mut body = vec![("model".to_string(), Json::Str(model.to_string()))];
    for (key, value) in members {
        if key == "id" {
            let Json::Str(text) = value else {
                return at("`id` is not a string");
            };
            id = text;
        } else if SET_HERE.contains(&key.as_str()) {
            return at(&format!("`{key}` is set by this client"));
        } else {
            body.push((key, value));
        }
    }
    body.push(("stream".into(), Json::Bool(true)));
    body.push((
        "stream_options".into(),
        Json::Object(vec![("include_usage".into(), Json::Bool(true))]),
    ));
    let body = Json::Object(body);
    if !body.is_finite() {
        return at("a number is not finite");
    }
    let compact = DumpOptions {
        item_separator: ",".into(),
        key_separator: ":".into(),
        ..DumpOptions::default()
    };
    Job {
        id,
        body: chat_json::dumps(&body, &compact),
    }
}

/// A finished stream's chunks joined into one `chat.completion`: the deltas' text joined,
/// each tool call without its `index`, the finishing choice's `finish_reason` and
/// `continuable`, and the final usage.
fn completion(chunks: &[Value]) -> Value {
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    let mut choice = json!({"index": 0, "finish_reason": null});
    let mut usage = Value::Null;
    for chunk in chunks {
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            usage = u.clone();
        }
        for c in chunk["choices"].as_array().into_iter().flatten() {
            let delta = &c["delta"];
            content.push_str(delta["content"].as_str().unwrap_or(""));
            reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let mut call = call.clone();
                if let Some(o) = call.as_object_mut() {
                    o.remove("index");
                }
                calls.push(call);
            }
            if !c["finish_reason"].is_null() {
                choice["finish_reason"] = c["finish_reason"].clone();
                if let Some(v) = c.get("continuable") {
                    choice["continuable"] = v.clone();
                }
            }
        }
    }
    let mut message = json!({"role": "assistant", "content": null});
    if !content.is_empty() {
        message["content"] = content.into();
    }
    if !reasoning.is_empty() {
        message["reasoning_content"] = reasoning.into();
    }
    if !calls.is_empty() {
        message["tool_calls"] = calls.into();
    }
    choice["message"] = message;
    let first = chunks.first().cloned().unwrap_or(Value::Null);
    json!({
        "id": first["id"],
        "object": "chat.completion",
        "created": first["created"],
        "model": first["model"],
        "choices": [choice],
        "usage": usage,
    })
}

/// A padded JSON body (every response but a stream is padded with trailing spaces).
fn padded_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes).unwrap_or_else(
        |_| json!({"error": {"message": String::from_utf8_lossy(bytes).trim_end(), "type": "client"}}),
    )
}

/// One attempt: the request sent, and its stream read to `[DONE]` or its error.
async fn attempt(
    client: &reqwest::Client,
    url: &str,
    token: &str,
    weights: &str,
    body: &str,
) -> Outcome {
    let sent = client
        .post(format!("{url}/v1/chat/completions"))
        .bearer_auth(token)
        .header(WEIGHTS_HEADER, weights)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await;
    let response = match sent {
        Ok(r) => r,
        Err(e) => return Outcome::client_error(None, format!("send: {e}")),
    };
    let status = response.status().as_u16();
    if status != 200 {
        let bytes = response.bytes().await.unwrap_or_default();
        return Outcome {
            status: Some(status),
            response: None,
            error: Some(padded_json(&bytes)),
        };
    }
    let mut reader = StreamReader::new();
    let mut chunks = Vec::new();
    let mut wire = response.bytes_stream();
    while let Some(bytes) = wire.next().await {
        let bytes = match bytes {
            Ok(b) => b,
            Err(e) => return Outcome::client_error(Some(status), format!("read: {e}")),
        };
        reader.push(&bytes);
        loop {
            match reader.next_event() {
                Ok(None) => break,
                Ok(Some(StreamEvent::Json(data))) => match serde_json::from_str(&data) {
                    Ok(chunk) => chunks.push(chunk),
                    Err(e) => return Outcome::client_error(Some(status), format!("chunk: {e}")),
                },
                Ok(Some(StreamEvent::Error(data))) => {
                    return Outcome {
                        status: Some(status),
                        response: None,
                        error: Some(padded_json(data.as_bytes())),
                    };
                }
                Ok(Some(StreamEvent::Done)) => {
                    return Outcome {
                        status: Some(status),
                        response: Some(completion(&chunks)),
                        error: None,
                    };
                }
                Err(e) => return Outcome::client_error(Some(status), format!("event: {e:?}")),
            }
        }
    }
    Outcome::client_error(Some(status), "the stream ended before [DONE]")
}

/// A request sent until it is not refused `overloaded`, or the retries run out.
async fn run(
    client: &reqwest::Client,
    url: &str,
    token: &str,
    weights: &str,
    job: &Job,
) -> (Outcome, f64) {
    let mut tries = 0;
    loop {
        let start = Instant::now();
        let outcome = attempt(client, url, token, weights, &job.body).await;
        let overloaded = outcome.status == Some(503)
            && outcome
                .error
                .as_ref()
                .is_some_and(|e| e["error"]["type"].as_str() == Some(error_type::OVERLOADED));
        if !overloaded || tries == OVERLOADED_RETRIES {
            return (outcome, start.elapsed().as_secs_f64());
        }
        tries += 1;
        tokio::time::sleep(OVERLOADED_PAUSE).await;
    }
}

/// The ids an earlier run onto `out` already holds a response for.
fn answered(out: &std::path::Path) -> HashSet<String> {
    let Ok(file) = std::fs::File::open(out) else {
        return HashSet::new();
    };
    std::io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .filter(|v| !v["response"].is_null())
        .filter_map(|v| v["id"].as_str().map(str::to_string))
        .collect()
}

#[tokio::main]
async fn main() {
    let usage = "usage: GATEWAY_TOKEN=<token> chat_batch <node URL> <weights sha256> \
                 <requests.jsonl> <out.jsonl> [--concurrency N]";
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut concurrency = 1usize;
    if let Some(i) = args.iter().position(|a| a == "--concurrency") {
        let n = args.get(i + 1).unwrap_or_else(|| fail(usage));
        concurrency = n
            .parse()
            .ok()
            .filter(|&n| n > 0)
            .unwrap_or_else(|| fail("--concurrency: a positive count"));
        args.drain(i..i + 2);
    }
    let [url, weights, input, out] = args.as_slice() else {
        fail(usage)
    };
    let url = url.trim_end_matches('/');
    let token = std::env::var("GATEWAY_TOKEN").unwrap_or_else(|_| fail("GATEWAY_TOKEN is not set"));
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_else(|e| fail(e));

    // The node's model id, and its weights hash checked against the one given.
    let info = client
        .get(format!("{url}/v1/engine/info"))
        .bearer_auth(&token)
        .send()
        .await
        .unwrap_or_else(|e| fail(format!("{url}/v1/engine/info: {e}")));
    let status = info.status();
    let info = padded_json(&info.bytes().await.unwrap_or_default());
    if !status.is_success() {
        fail(format!("/v1/engine/info: {status} {info}"));
    }
    let model = info["model"]
        .as_str()
        .unwrap_or_else(|| fail("/v1/engine/info names no model"))
        .to_string();
    let served = info["weights_sha256"].as_str().unwrap_or("");
    if !served.eq_ignore_ascii_case(weights) {
        fail(format!(
            "the node serves weights {served}, not the {weights} given"
        ));
    }

    let text = std::fs::read_to_string(input).unwrap_or_else(|e| fail(format!("{input}: {e}")));
    let done = answered(std::path::Path::new(out));
    let mut ids = HashSet::new();
    let mut jobs = Vec::new();
    let mut refused = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let job = job(i, line, &model);
        if !ids.insert(job.id.clone()) {
            fail(format!("line {}: the id {} again", i + 1, job.id));
        }
        if done.contains(&job.id) {
            continue;
        }
        // The node's own parser, so a line it would refuse is never sent.
        match eidola_server_engine::api::parse_request(job.body.as_bytes(), &model) {
            Ok(_) => jobs.push(job),
            Err(e) => refused.push((job.id, e.to_body())),
        }
    }

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out)
        .unwrap_or_else(|e| fail(format!("{out}: {e}")));
    let mut write = |line: Value| {
        writeln!(file, "{line}")
            .and_then(|()| file.flush())
            .unwrap_or_else(|e| fail(format!("{out}: {e}")));
    };
    let mut failed = 0usize;
    for (id, error) in refused {
        failed += 1;
        write(json!({"id": id, "status": null, "seconds": 0.0, "response": null, "error": error}));
    }
    let start = Instant::now();
    let total = jobs.len();
    let mut finished = 0usize;
    let client = &client;
    let (url, token, weights) = (url, token.as_str(), weights.as_str());
    let mut results = futures_util::stream::iter(jobs)
        .map(|job| async move {
            let (outcome, seconds) = run(client, url, token, weights, &job).await;
            (job.id, outcome, seconds)
        })
        .buffer_unordered(concurrency);
    while let Some((id, outcome, seconds)) = results.next().await {
        finished += 1;
        if outcome.response.is_none() {
            failed += 1;
        }
        write(json!({
            "id": id,
            "status": outcome.status,
            "seconds": seconds,
            "response": outcome.response,
            "error": outcome.error,
        }));
        if finished.is_multiple_of(50) || finished == total {
            eprintln!(
                "chat_batch: {finished}/{total} in {:.0} s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    eprintln!(
        "{}",
        json!({"requests": total, "skipped": done.len(), "failed": failed,
               "seconds": start.elapsed().as_secs_f64(), "concurrency": concurrency})
    );
    if failed > 0 {
        std::process::exit(2);
    }
}
