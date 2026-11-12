//! Shared harness: the real node, in process, on a loopback port, over the synthetic dev
//! model and the CPU executor.

#![allow(dead_code)]

pub mod fixture;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use eidola_server_engine::config::{Config, ExecutorKind, WeightsStorage, env};
use eidola_server_engine::model::LoadedModel;
use eidola_server_engine::worker::{EngineHandle, Stats};
use serde_json::{Value, json};

pub const MODEL_ID: &str = "mimo-test";
pub const TOKEN: &str = "test-gateway-token";

/// The test client. It speaks plain HTTP, but reqwest's features are unified across the
/// workspace, where other crates select TLS without a built-in crypto provider, so the
/// same RustCrypto provider they use is installed before a client is built.
pub fn http_client() -> reqwest::Client {
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    reqwest::Client::new()
}

/// The dev model directory (built once, cached under the target directory).
pub fn model_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("engine-dev-model");
        fixture::ensure_dev_model(&dir).expect("build the dev model");
        dir
    })
}

/// The dev model's weights hash, computed independently of the crate.
pub fn weights_hash() -> &'static str {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| fixture::weights_hash_of(model_dir()))
}

/// Argon2id hash of [`TOKEN`].
pub fn token_hash() -> &'static str {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| {
        use argon2::PasswordHasher;
        argon2::Argon2::default()
            .hash_password(TOKEN.as_bytes())
            .unwrap()
            .to_string()
    })
}

/// The model, verified and loaded once per test binary (it is ~300 MB as f32).
pub fn loaded() -> Arc<LoadedModel> {
    static MODEL: OnceLock<Arc<LoadedModel>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            Arc::new(
                LoadedModel::load(
                    model_dir(),
                    weights_hash(),
                    WeightsStorage::DevWritable,
                    ExecutorKind::Cpu,
                )
                .unwrap(),
            )
        })
        .clone()
}

/// A complete, valid environment.
pub fn env_map() -> HashMap<&'static str, String> {
    HashMap::from([
        (env::MODEL_ID, MODEL_ID.to_string()),
        (env::WEIGHTS_DIR, model_dir().display().to_string()),
        (env::WEIGHTS_SHA256, weights_hash().to_string()),
        (env::WEIGHTS_STORAGE, "dev-writable".into()),
        (env::GATEWAY_TOKEN, TOKEN.to_string()),
        (env::GATEWAY_TOKEN_HASH, token_hash().to_string()),
        (env::EXECUTOR, "cpu".into()),
        (env::BIND_ADDR, "127.0.0.1:0".into()),
        (env::KV_BLOCK_SIZE, "16".into()),
        (env::KV_BLOCKS, "256".into()),
        (env::MAX_MODEL_LEN, "1024".into()),
        (env::MAX_SEQS, "4".into()),
        (env::MAX_BATCHED_TOKENS, "256".into()),
        (env::MAX_PREFILL_CHUNK, "256".into()),
        (env::DRAFT_TOKENS, "2".into()),
        (env::MAX_REQUESTS, "8".into()),
        (env::PREFIX_CACHE, "true".into()),
        (env::CACHE_IDLE_TTL_SECS, "900".into()),
        (env::CACHE_MAX_AGE_SECS, "7200".into()),
        (env::KV_HOST_BYTES, "0".into()),
    ])
}

/// A swap list naming no active area, for tests that boot a node the swap check applies
/// to and assert what happens after it.
pub fn no_swap_list() -> &'static std::path::Path {
    static PATH: OnceLock<std::path::PathBuf> = OnceLock::new();
    PATH.get_or_init(|| {
        let p = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("no-swap-{}", std::process::id()));
        std::fs::write(&p, "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n").unwrap();
        p
    })
}

/// Boots `config` past the swap check ([`no_swap_list`]).
pub fn boot_without_swap(
    config: Config,
) -> Result<eidola_server_engine::Node, eidola_server_engine::BootError> {
    eidola_server_engine::boot_with_swap_list(config, no_swap_list())
}

pub fn config_from(map: &HashMap<&'static str, String>) -> Result<Config, String> {
    Config::from_lookup(|k| map.get(k).cloned()).map_err(|e| e.to_string())
}

/// A running node.
pub struct TestNode {
    pub base: String,
    pub engine: EngineHandle,
    pub admission: Arc<eidola_server_engine::worker::Admission>,
    pub client: reqwest::Client,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for TestNode {
    fn drop(&mut self) {
        self.server.abort();
    }
}

impl TestNode {
    /// Starts a node over the shared model, with `overrides` applied to [`env_map`].
    pub async fn start(overrides: &[(&'static str, &str)]) -> TestNode {
        let mut map = env_map();
        for (k, v) in overrides {
            map.insert(k, v.to_string());
        }
        let config = config_from(&map).unwrap();
        let node = eidola_server_engine::start(config, loaded()).unwrap();
        Self::serve(node).await
    }

    pub async fn serve(node: eidola_server_engine::Node) -> TestNode {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let router = node.router;
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        TestNode {
            base,
            engine: node.engine,
            admission: node.admission,
            client: http_client(),
            server,
        }
    }

    pub fn stats(&self) -> Stats {
        self.engine.stats()
    }

    /// A chat request with valid gateway headers, asking for a stream with its usage as
    /// the gateway always does: `stream` and `stream_options` are added where `body`
    /// does not name them itself.
    pub fn chat(&self, body: &Value) -> reqwest::RequestBuilder {
        let mut body = body.clone();
        if let Some(fields) = body.as_object_mut() {
            fields.entry("stream").or_insert(true.into());
            fields
                .entry("stream_options")
                .or_insert(json!({"include_usage": true}));
        }
        self.client
            .post(format!("{}/v1/chat/completions", self.base))
            .bearer_auth(TOKEN)
            .header("x-eidola-weights-sha256", weights_hash())
            .json(&body)
    }

    /// The status and the answer as one completion: a refusal's body as it is, and a
    /// streamed answer joined the way a client's accumulator joins it ([`completion`]).
    pub async fn chat_json(&self, body: &Value) -> (u16, Value) {
        let r = self.chat(body).send().await.unwrap();
        let status = r.status().as_u16();
        if status != 200 {
            return (status, r.json().await.unwrap());
        }
        let wire = r.bytes().await.unwrap();
        (status, completion(&events_of(&wire)))
    }

    /// Streams a request and returns every SSE `data:` payload, read out of its frames
    /// as the gateway reads them (`eidola_common::stream_shape::StreamReader`), with
    /// the trailing whitespace a frame may pad a JSON payload with removed. Every
    /// frame is checked to be the cadence's size.
    pub async fn chat_stream(&self, body: &Value) -> Vec<String> {
        use futures_util::StreamExt;
        let r = self.chat(body).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200);
        assert_eq!(
            r.headers()["content-type"].to_str().unwrap(),
            "text/event-stream"
        );
        let mut wire = Vec::new();
        let mut body = r.bytes_stream();
        while let Some(bytes) = body.next().await {
            wire.extend_from_slice(&bytes.unwrap());
        }
        events_of(&wire)
    }

    /// Waits until `pred` holds for the stats (or panics after `timeout`).
    pub async fn wait_for(&self, timeout: Duration, pred: impl Fn(&Stats) -> bool) -> Stats {
        let start = Instant::now();
        loop {
            let s = self.stats();
            if pred(&s) {
                return s;
            }
            assert!(start.elapsed() < timeout, "timed out; stats {s:?}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

/// A greedy request for one user message.
pub fn request(text: &str, max_tokens: u32) -> Value {
    json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": text}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    })
}

/// A stream's events read out of its frames as the gateway reads them
/// (`eidola_common::stream_shape::StreamReader`), with the trailing whitespace a frame
/// may pad a JSON payload with removed: each JSON payload, `error: <payload>`, or
/// `[DONE]`. The stream is checked to be whole frames of the cadence's size.
pub fn events_of(wire: &[u8]) -> Vec<String> {
    use eidola_common::stream_shape::{StreamEvent, StreamReader};
    let frame = eidola_common::engine_protocol::STREAM_CADENCE.frame_bytes();
    assert_eq!(wire.len() % frame, 0, "the stream is whole frames");
    let mut reader = StreamReader::new();
    reader.push(wire);
    let mut out = Vec::new();
    while let Some(event) = reader.next_event().unwrap() {
        out.push(match event {
            StreamEvent::Json(data) => data.trim_end().to_string(),
            StreamEvent::Error(data) => format!("error: {}", data.trim_end()),
            StreamEvent::Done => "[DONE]".to_string(),
        });
    }
    out
}

/// A finished stream as one `chat.completion`: its deltas joined (`content` null when
/// empty, `reasoning_content` only when there was some, each tool call without its
/// `index`), its finish choice's `finish_reason` and `continuable`, and the final usage.
/// A stream that ended in an error is that error's body.
pub fn completion(events: &[String]) -> Value {
    if let Some(error) = events.iter().find_map(|e| e.strip_prefix("error: ")) {
        return serde_json::from_str(error).unwrap();
    }
    assert_eq!(
        events.last().map(String::as_str),
        Some("[DONE]"),
        "{events:?}"
    );
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    let (mut content, mut reasoning, mut calls) = (String::new(), String::new(), Vec::new());
    let mut choice = json!({"index": 0});
    for chunk in &chunks {
        for c in chunk["choices"].as_array().unwrap() {
            let delta = &c["delta"];
            content.push_str(delta["content"].as_str().unwrap_or(""));
            reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
            for call in delta["tool_calls"].as_array().into_iter().flatten() {
                let mut call = call.clone();
                call.as_object_mut().unwrap().remove("index");
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
    let last = chunks.last().unwrap();
    json!({
        "id": chunks[0]["id"],
        "object": "chat.completion",
        "created": chunks[0]["created"],
        "model": chunks[0]["model"],
        "choices": [choice],
        "usage": last["usage"],
    })
}
