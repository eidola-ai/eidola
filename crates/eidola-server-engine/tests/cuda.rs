//! The CUDA executor's boot (`cuda` feature).
//!
//! Without a device: the configuration (the executor's settings required with it and the
//! CPU's refused), the boot order (storage, weights hash and model files before anything
//! of the executor's), the executor's own refusal of a model it does not support, the
//! node's refusals before the device (drafting, a KV budget that cannot hold one
//! sequence), and a device-less machine failing closed, in process and as the real
//! binary, with nothing served and no CPU fallback.
//!
//! With a device (`EIDOLA_TEST_CUDA_WEIGHTS` and `EIDOLA_ENGINE_KERNELS_DIR`): the node
//! booted on real MiMo-V2.6-Flash weights answers a chat request end to end, and its
//! output equals, token for token, the engine core driven directly over a CUDA executor
//! built from the same configuration.

#![cfg(feature = "cuda")]

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use eidola_engine_cuda::{CudaGraphs, Gpu};
use eidola_server_engine::config::{ExecutorConfig, ExecutorKind, WeightsStorage, env};
use serde_json::{Value, json};

/// The message a machine without a CUDA driver or device refuses with.
const NO_DEVICE: &str = "no CUDA driver or device";

/// The CUDA executor's settings, replacing the CPU's, over `dir` and `hash`.
fn cuda_env(dir: &Path, hash: &str) -> HashMap<&'static str, String> {
    let mut map = env_map();
    map.remove(env::KV_BLOCKS);
    map.insert(env::EXECUTOR, "cuda".into());
    map.insert(env::WEIGHTS_DIR, dir.display().to_string());
    map.insert(env::WEIGHTS_SHA256, hash.to_string());
    map.insert(env::DRAFT_TOKENS, "0".into());
    map.insert(env::KV_DEVICE_BYTES, (8u64 << 30).to_string());
    map.insert(env::KERNELS_DIR, "/nonexistent/eidola-kernels".into());
    // The process's own setting when it has one (the GPU tests run either way), else
    // off.
    map.insert(
        env::CUDA_GRAPHS,
        std::env::var(env::CUDA_GRAPHS).unwrap_or_else(|_| "off".into()),
    );
    map
}

/// A weights directory the CUDA executor's host-side checks accept: Flash's
/// `config.json`, the dev model's chat files, and one shard holding no tensors (a device
/// would refuse it at the weights layout check, after the kernels).
fn flash_shaped_dir() -> &'static (PathBuf, String) {
    static DIR: std::sync::OnceLock<(PathBuf, String)> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("engine-flash-shaped-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../eidola-engine-model/tests/data/flash-mopd.config.json"
            ),
            dir.join("config.json"),
        )
        .unwrap();
        for name in eidola_server_engine::model::CHAT_FILES {
            std::fs::copy(model_dir().join(name), dir.join(name)).unwrap();
        }
        let mut shard = 2u64.to_le_bytes().to_vec();
        shard.extend_from_slice(b"{}");
        std::fs::write(dir.join("model.safetensors"), shard).unwrap();
        let hash = fixture::weights_hash_of(&dir);
        (dir, hash)
    })
}

fn boot_err(map: &HashMap<&'static str, String>) -> String {
    match eidola_server_engine::boot(config_from(map).unwrap()) {
        Ok(_) => panic!("a cuda boot that cannot have a usable executor succeeded"),
        Err(e) => e.to_string(),
    }
}

// ---------------------------------------------------------------------------------------
// Without a device.
// ---------------------------------------------------------------------------------------

#[test]
fn cuda_settings_are_required_with_it_and_the_cpus_refused() {
    let full = cuda_env(model_dir(), weights_hash());
    let config = config_from(&full).unwrap();
    assert_eq!(config.executor.kind(), ExecutorKind::Cuda);
    for key in [env::KV_DEVICE_BYTES, env::KERNELS_DIR, env::CUDA_GRAPHS] {
        let mut map = full.clone();
        map.remove(key);
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(key), "{key}: {err}");
        map.insert(key, String::new());
        assert!(config_from(&map).unwrap_err().contains(key), "{key} empty");
    }
    for bad in ["0", "-1", "8GiB", "18446744073709551616"] {
        let mut map = full.clone();
        map.insert(env::KV_DEVICE_BYTES, bad.into());
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(env::KV_DEVICE_BYTES), "{bad}: {err}");
    }
    // Exactly `on` or `off`.
    for bad in ["ON", "true", "1", "auto", "on "] {
        let mut map = full.clone();
        map.insert(env::CUDA_GRAPHS, bad.into());
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(env::CUDA_GRAPHS), "{bad}: {err}");
    }
    for (value, graphs) in [("on", CudaGraphs::On), ("off", CudaGraphs::Off)] {
        let mut map = full.clone();
        map.insert(env::CUDA_GRAPHS, value.into());
        match config_from(&map).unwrap().executor {
            ExecutorConfig::Cuda { graphs: g, .. } => assert_eq!(g, graphs),
            _ => unreachable!(),
        }
    }
    // The CPU executor's block count is refused, not ignored.
    let mut map = full.clone();
    map.insert(env::KV_BLOCKS, "256".into());
    let err = config_from(&map).unwrap_err();
    assert!(err.contains(env::KV_BLOCKS) && err.contains("cpu"), "{err}");
}

/// Storage, then the weights hash, then the model's files, all before the executor: a
/// wrong expected hash is the refusal, whatever the device.
#[test]
fn the_weights_hash_is_checked_before_the_executor() {
    let mut map = cuda_env(model_dir(), weights_hash());
    map.insert(env::WEIGHTS_SHA256, "ab".repeat(32));
    let err = boot_err(&map);
    assert!(err.contains("weights hash"), "{err}");
    assert!(!err.contains(NO_DEVICE), "{err}");
    assert!(!err.contains("unsupported"), "{err}");

    // `verified-readonly` refuses a writable directory before anything is hashed.
    let mut map = cuda_env(model_dir(), weights_hash());
    map.insert(env::WEIGHTS_STORAGE, "verified-readonly".into());
    let err = boot_err(&map);
    assert!(err.contains("read-only"), "{err}");
}

/// A model the executor does not support is refused with the executor's own error,
/// naming the field, before the device is opened (so the same on any machine).
#[test]
fn an_unsupported_model_is_refused_by_the_executor_before_the_device() {
    let err = boot_err(&cuda_env(model_dir(), weights_hash()));
    assert!(err.contains("unsupported model"), "{err}");
    assert!(err.contains("hidden_size"), "{err}");
}

/// The node's own refusals come before the device too.
#[test]
fn drafting_and_a_small_kv_budget_are_refused_before_the_device() {
    let (dir, hash) = flash_shaped_dir();
    let mut map = cuda_env(dir, hash);
    map.insert(env::DRAFT_TOKENS, "2".into());
    let err = boot_err(&map);
    assert!(err.contains(env::DRAFT_TOKENS), "{err}");

    let mut map = cuda_env(dir, hash);
    map.insert(env::KV_DEVICE_BYTES, (1u64 << 20).to_string());
    let err = boot_err(&map);
    assert!(err.contains(env::KV_DEVICE_BYTES), "{err}");

    // A KV geometry the executor cannot index is the executor's refusal.
    let mut map = cuda_env(dir, hash);
    map.insert(env::KV_BLOCK_SIZE, "16777217".into());
    let err = boot_err(&map);
    assert!(err.contains("KV geometry"), "{err}");
}

/// A configuration the executor accepts on the host fails closed at the device: on a
/// machine without one, with that refusal; never by falling back to the CPU executor.
#[test]
fn a_supported_configuration_without_a_device_fails_closed() {
    let (dir, hash) = flash_shaped_dir();
    let err = boot_err(&cuda_env(dir, hash));
    if !Gpu::available() {
        assert!(err.contains(NO_DEVICE), "{err}");
    }
}

/// The real binary, over the same configuration: exits non-zero without ever listening.
#[test]
fn the_binary_refuses_to_serve_without_a_usable_executor() {
    let (dir, hash) = flash_shaped_dir();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_eidola-server-engine"))
        .env_clear()
        .envs(cuda_env(dir, hash))
        .env("RUST_LOG", "info")
        .output()
        .unwrap();
    let log = String::from_utf8_lossy(&output.stdout);
    assert!(!output.status.success(), "{log}");
    assert!(log.contains("refusing to serve"), "{log}");
    assert!(!log.contains("listening"), "{log}");
    assert!(!log.contains("engine started"), "{log}");
    if !Gpu::available() {
        assert!(log.contains(NO_DEVICE), "{log}");
    }
}

// ---------------------------------------------------------------------------------------
// With a device: real Flash weights.
// ---------------------------------------------------------------------------------------

/// Boots the node on real MiMo-V2.6-Flash weights and checks a chat request end to end
/// against the engine core driven directly over the same executor configuration. Runs
/// where `EIDOLA_TEST_CUDA_WEIGHTS` names the checkpoint directory (shards, `config.json`,
/// and the chat files) and `EIDOLA_ENGINE_KERNELS_DIR` the kernel build output;
/// `EIDOLA_TEST_CUDA_WEIGHTS_SHA256` skips computing the weights hash here. Skipped
/// otherwise.
#[tokio::test(flavor = "multi_thread")]
async fn real_flash_serves_what_the_executor_computes() {
    let (Some(weights), Some(kernels)) = (
        std::env::var_os("EIDOLA_TEST_CUDA_WEIGHTS"),
        std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR"),
    ) else {
        eprintln!(
            "skipped: set EIDOLA_TEST_CUDA_WEIGHTS and EIDOLA_ENGINE_KERNELS_DIR to run on a GPU"
        );
        return;
    };
    let weights = PathBuf::from(weights);
    let hash = match std::env::var("EIDOLA_TEST_CUDA_WEIGHTS_SHA256") {
        Ok(h) => h.to_ascii_lowercase(),
        Err(_) => fixture::weights_hash_of(&weights),
    };
    let mut map = cuda_env(&weights, &hash);
    map.insert(
        env::KERNELS_DIR,
        PathBuf::from(kernels).display().to_string(),
    );
    map.insert(env::KV_DEVICE_BYTES, (32u64 << 30).to_string());
    map.insert(env::MAX_MODEL_LEN, "4096".into());
    let config = config_from(&map).unwrap();
    let (sizing, cache) = (config.sizing, config.cache);
    let (kernels_dir, kv_device_bytes, graphs) = match &config.executor {
        ExecutorConfig::Cuda {
            kernels_dir,
            kv_device_bytes,
            graphs,
        } => (kernels_dir.clone(), *kv_device_bytes, *graphs),
        _ => unreachable!(),
    };

    // Verified once; the node and the direct run share it.
    let model = {
        let (w, h) = (weights.clone(), hash.clone());
        tokio::task::spawn_blocking(move || {
            eidola_server_engine::model::LoadedModel::load(
                &w,
                &h,
                WeightsStorage::DevWritable,
                ExecutorKind::Cuda,
            )
            .unwrap()
        })
        .await
        .unwrap()
    };
    let model = Arc::new(model);

    let node = {
        let model = model.clone();
        tokio::task::spawn_blocking(move || eidola_server_engine::start(config, model))
            .await
            .unwrap()
            .unwrap()
    };
    let engine_stopped = node.engine_stopped;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let router = node.router;
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let client = http_client();

    let info: Value = client
        .get(format!("{base}/v1/engine/info"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    eprintln!("info: {info}");
    assert_eq!(info["executor"], "cuda");
    assert!(!info["device"]["name"].as_str().unwrap().is_empty());
    assert!(
        info["device"]["compute_capability"]
            .as_str()
            .unwrap()
            .starts_with("10.")
    );
    assert!(info["device"]["image"].is_string());

    let text = "What is the capital of France? Answer in one word.";
    let max_tokens = 48;
    let chat = |body: Value| {
        client
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(TOKEN)
            .header("x-eidola-weights-sha256", &hash)
            .json(&body)
    };
    let body = json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": text}],
        "max_completion_tokens": max_tokens,
        "temperature": 0.0,
    });
    let r = chat(body.clone()).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let v: Value = r.json().await.unwrap();
    eprintln!("node: {v}");
    let message = &v["choices"][0]["message"];
    let node_reasoning = message["reasoning_content"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let node_content = message["content"].as_str().unwrap_or("").to_string();
    let node_finish = v["choices"][0]["finish_reason"]
        .as_str()
        .unwrap()
        .to_string();
    let node_completion = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(
        !(node_reasoning.is_empty() && node_content.is_empty()),
        "{v}"
    );

    // Streaming the same request: the same text.
    let mut streamed = body.clone();
    streamed["stream"] = json!(true);
    let sse = chat(streamed).send().await.unwrap().text().await.unwrap();
    let (mut reasoning, mut content) = (String::new(), String::new());
    for line in sse.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            break;
        }
        let chunk: Value = serde_json::from_str(line).unwrap();
        if let Some(delta) = chunk["choices"].get(0).map(|c| &c["delta"]) {
            reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or(""));
            content.push_str(delta["content"].as_str().unwrap_or(""));
        }
    }
    assert_eq!(
        (reasoning, content),
        (node_reasoning.clone(), node_content.clone())
    );

    // Stop the node and wait for its engine thread to release the device: the server and
    // its connections (the client's, kept alive) hold the last engine handles.
    drop(client);
    server.abort();
    let _ = server.await;
    drop(node.engine);
    drop(node.admission);
    tokio::time::timeout(Duration::from_secs(120), engine_stopped)
        .await
        .expect("the engine thread stops once nothing can reach it")
        .ok();

    // The engine core driven directly, over an executor built from the same
    // configuration, on the same prompt tokens.
    let expected = tokio::task::spawn_blocking(move || {
        use eidola_engine::engine::{CacheScope, Engine, FinishReason, Request, SchedulerConfig};
        use eidola_engine::kv::CachePolicy;
        use eidola_engine::sampling::SamplingParams;
        use eidola_engine_chat::{
            ChatDelta, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, RenderOptions,
            StopCause,
        };

        let messages = json!([{"role": "user", "content": text}]).to_string();
        let input = ChatInput::from_json(&messages, None).unwrap();
        let prompt = model
            .template()
            .render(
                &input,
                RenderOptions {
                    add_generation_prompt: true,
                    enable_thinking: None,
                },
            )
            .unwrap();
        let prompt = model.tokenizer().encode(&prompt).unwrap();
        let exec = eidola_server_engine::cuda::prepare(
            &model,
            &sizing,
            &cache,
            &kernels_dir,
            kv_device_bytes,
            graphs,
        )
        .unwrap()
        .build()
        .unwrap();
        let mut engine = Engine::new(
            exec,
            SchedulerConfig {
                max_batched_tokens: sizing.max_batched_tokens,
                max_seqs: sizing.max_seqs,
                max_prefill_chunk: sizing.max_prefill_chunk,
                eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
                speculative: false,
                cache: CachePolicy::default(),
                sweep_interval_ms: 1000,
            },
        )
        .unwrap();
        engine
            .submit(Request {
                id: 1,
                prompt,
                sampling: SamplingParams::greedy(),
                max_tokens,
                stop_token_ids: Vec::new(),
                cache: CacheScope::Private,
            })
            .unwrap();
        let (mut tokens, mut finish) = (Vec::new(), None);
        'outer: for now in 0.. {
            for ev in engine.step(now).unwrap() {
                tokens.extend(ev.tokens);
                if let Some(f) = ev.finish {
                    finish = Some(f);
                    break 'outer;
                }
            }
        }
        let finish = finish.unwrap();
        let mut parser =
            OutputParser::new(OutputConfig::new(None, None), PrefixedCallIds("c".into()));
        let mut all = ChatDelta::default();
        let mut text_tokens = &tokens[..];
        if finish == FinishReason::Stop {
            text_tokens = &tokens[..tokens.len() - 1];
        }
        for &t in text_tokens {
            let d = parser.push_token(model.tokenizer(), t).unwrap();
            all.reasoning_content.push_str(&d.reasoning_content);
            all.content.push_str(&d.content);
        }
        let cause = match finish {
            FinishReason::Stop => StopCause::EndOfSequence,
            _ => StopCause::MaxTokens,
        };
        let (d, reason) = parser.finish(cause);
        all.reasoning_content.push_str(&d.reasoning_content);
        all.content.push_str(&d.content);
        (all, reason.as_str().to_string(), tokens.len() as u64)
    })
    .await
    .unwrap();
    let (delta, finish, completion) = expected;
    assert_eq!(node_reasoning, delta.reasoning_content);
    assert_eq!(node_content, delta.content);
    assert_eq!(node_finish, finish);
    assert_eq!(node_completion, completion);
}
