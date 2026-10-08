//! Shared harness: the real node, in process, on a loopback port, over the synthetic dev
//! model and the CPU executor.

#![allow(dead_code)]

pub mod fixture;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use eidola_server_engine::config::{Config, env};
use eidola_server_engine::model::LoadedModel;
use eidola_server_engine::worker::{EngineHandle, Stats};
use serde_json::{Value, json};

pub const MODEL_ID: &str = "mimo-test";
pub const TOKEN: &str = "test-gateway-token";

/// The dev model directory (built once, cached under the target directory).
pub fn model_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("engine-dev-model-v1");
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
        .get_or_init(|| Arc::new(LoadedModel::load(model_dir(), weights_hash()).unwrap()))
        .clone()
}

/// A complete, valid environment.
pub fn env_map() -> HashMap<&'static str, String> {
    HashMap::from([
        (env::MODEL_ID, MODEL_ID.to_string()),
        (env::WEIGHTS_DIR, model_dir().display().to_string()),
        (env::WEIGHTS_SHA256, weights_hash().to_string()),
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
    ])
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
            client: reqwest::Client::new(),
            server,
        }
    }

    pub fn stats(&self) -> Stats {
        self.engine.stats()
    }

    /// A chat request with valid gateway headers.
    pub fn chat(&self, body: &Value) -> reqwest::RequestBuilder {
        self.client
            .post(format!("{}/v1/chat/completions", self.base))
            .bearer_auth(TOKEN)
            .header("x-eidola-weights-sha256", weights_hash())
            .json(body)
    }

    pub async fn chat_json(&self, body: &Value) -> (u16, Value) {
        let r = self.chat(body).send().await.unwrap();
        let status = r.status().as_u16();
        (status, r.json().await.unwrap())
    }

    /// Streams a request and returns every SSE `data:` payload.
    pub async fn chat_stream(&self, body: &Value) -> Vec<String> {
        let r = self.chat(body).send().await.unwrap();
        assert_eq!(r.status().as_u16(), 200);
        assert_eq!(
            r.headers()["content-type"].to_str().unwrap(),
            "text/event-stream"
        );
        let text = r.text().await.unwrap();
        text.split("\n\n")
            .filter(|e| !e.trim().is_empty())
            .map(|e| {
                e.lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .collect()
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
