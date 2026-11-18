//! The CUDA executor's boot (`cuda` feature).
//!
//! Without a device: the configuration (the executor's settings required with it and the
//! CPU's refused), the boot order (storage, weights hash and model files before anything
//! of the executor's), the executor's own refusal of a model it does not support, the
//! refusals before the device (a draft width past the model's MTP layers and an
//! oversized block in the grammar, a KV budget that cannot hold one sequence in the
//! node), drafting within the MTP layers reaching the device, the GPU
//! confidential-computing check refusing before the device unless the driver reports it,
//! and a device-less machine failing closed, in process and as the real binary, with
//! nothing served and no CPU fallback.
//!
//! DeepSeek-V4.1 (its dev model), without a device: one KV budget per device, graphs
//! refused, the weights hash and the executor's own refusals before the devices, the KV
//! and host-tier budgets refused before the devices, the CC check (with the GPU count)
//! before the devices, and the node's executor configuration run over host devices
//! answering token for token as the CPU executor does.
//!
//! Qwen3.8 (its dev model), without a device: graphs refused, one KV budget, the state
//! pool and one sequence in it, the host tier and the weights hash before the device; and
//! the node's executor configuration run over the kernels' bodies compiled for the host
//! answering token for token as the CPU executor does, eagerly and on the rung path, and
//! in the fast profile its pools and step scratch within the shared rules' counts.
//!
//! With a device (`EIDOLA_TEST_CUDA_WEIGHTS` and `EIDOLA_ENGINE_KERNELS_DIR`): the node
//! booted on real MiMo-V2.6-Flash weights answers a chat request end to end, and its
//! output equals, token for token, the engine core driven directly over a CUDA executor
//! built from the same configuration; and a generation cut off at a small ceiling, in its
//! reasoning and in its answer, continues into exactly the uninterrupted output, served
//! from the prefix the cut-off request cached. On two GPUs (ignored; `EIDOLA_ENGINE_KERNELS_DIR`):
//! the node booted on the DeepSeek-V4.1 dev model answers greedy requests with exactly the
//! CPU executor's text.

#![cfg(feature = "cuda")]

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use common::*;
use eidola_engine_cuda::Gpu;
use eidola_engine_cuda::mimo::CudaGraphs;
use eidola_server_engine::config::{ExecutorConfig, ExecutorKind, GpuCc, WeightsStorage, env};
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
    // Likewise the GPU confidential-computing check: `required` on a CC-mode GPU, else
    // `not-required` (a development GPU, or none).
    map.insert(
        env::GPU_CC,
        std::env::var(env::GPU_CC).unwrap_or_else(|_| "not-required".into()),
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

/// Boots past the swap check (`no_swap_list`): what these tests check comes after it.
fn boot_err(map: &HashMap<&'static str, String>) -> String {
    match boot_without_swap(config_from(map).unwrap()) {
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
    for key in [
        env::KV_DEVICE_BYTES,
        env::KERNELS_DIR,
        env::CUDA_GRAPHS,
        env::GPU_CC,
    ] {
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
    // An absolute, clean path.
    for bad in [
        "kernels",
        "/",
        "/opt/../kernels",
        "/opt//kernels",
        "/opt/kernels/",
    ] {
        let mut map = full.clone();
        map.insert(env::KERNELS_DIR, bad.into());
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(env::KERNELS_DIR), "{bad}: {err}");
    }
    // Exactly `on` or `off`.
    for bad in ["ON", "true", "1", "auto", "on "] {
        let mut map = full.clone();
        map.insert(env::CUDA_GRAPHS, bad.into());
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(env::CUDA_GRAPHS), "{bad}: {err}");
    }
    // Exactly `required` or `not-required`.
    for bad in ["Required", "yes", "off", "not_required", "required "] {
        let mut map = full.clone();
        map.insert(env::GPU_CC, bad.into());
        let err = config_from(&map).unwrap_err();
        assert!(err.contains(env::GPU_CC), "{bad}: {err}");
    }
    for (value, gpu_cc) in [
        ("required", GpuCc::Required),
        ("not-required", GpuCc::NotRequired),
    ] {
        let mut map = full.clone();
        map.insert(env::GPU_CC, value.into());
        match config_from(&map).unwrap().executor {
            ExecutorConfig::Cuda { gpu_cc: g, .. } => assert_eq!(g, gpu_cc),
            _ => unreachable!(),
        }
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

/// A draft width past the model's MTP layers, and a block size past its cap, are refused
/// by the configuration grammar; a KV budget that cannot hold one sequence is the node's
/// refusal, before the device. Drafting within the model's MTP layers passes every
/// host-side check and reaches the device.
#[test]
fn wide_drafts_and_a_small_kv_budget_are_refused_before_the_device() {
    let (dir, hash) = flash_shaped_dir();
    let mut map = cuda_env(dir, hash);
    map.insert(env::DRAFT_TOKENS, "4".into());
    let err = config_from(&map).unwrap_err();
    assert!(
        err.contains(env::DRAFT_TOKENS) && err.contains("cuda"),
        "{err}"
    );
    for k in ["1", "3"] {
        let mut map = cuda_env(dir, hash);
        map.insert(env::DRAFT_TOKENS, k.into());
        let err = boot_err(&map);
        assert!(!err.contains("unsupported"), "{k}: {err}");
        if !Gpu::available() {
            assert!(err.contains(NO_DEVICE), "{k}: {err}");
        }
    }
    // Drafting also needs its drafter pool from the budget: the least budget without
    // drafting is refused with three draft depths.
    let mut map = cuda_env(dir, hash);
    let sizing = config_from(&map).unwrap().sizing;
    let without = eidola_common::engine_deployment::cuda_kv_min_bytes(
        &eidola_common::architecture::MIMO_V26_FLASH,
        &sizing,
        &config_from(&map).unwrap().cache,
    );
    map.insert(env::KV_DEVICE_BYTES, without.to_string());
    assert!(boot_err(&map).contains(NO_DEVICE) || Gpu::available());
    map.insert(env::DRAFT_TOKENS, "3".into());
    let err = boot_err(&map);
    assert!(err.contains(env::KV_DEVICE_BYTES), "{err}");

    let mut map = cuda_env(dir, hash);
    map.insert(env::KV_DEVICE_BYTES, (1u64 << 20).to_string());
    let err = boot_err(&map);
    assert!(err.contains(env::KV_DEVICE_BYTES), "{err}");

    // A host tier too small for two blocks in every group is refused before the device.
    let mut map = cuda_env(dir, hash);
    map.insert(env::KV_HOST_BYTES, (1u64 << 20).to_string());
    let err = boot_err(&map);
    assert!(err.contains(env::KV_HOST_BYTES), "{err}");

    // A block size the executor could not index never reaches it.
    let mut map = cuda_env(dir, hash);
    map.insert(env::KV_BLOCK_SIZE, "16777217".into());
    let err = config_from(&map).unwrap_err();
    assert!(err.contains(env::KV_BLOCK_SIZE), "{err}");
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

/// Whether the driver here reports the GPUs in production confidential-computing mode.
fn driver_reports_cc() -> bool {
    eidola_server_engine::gpu_cc::query()
        .is_ok_and(|state| eidola_server_engine::gpu_cc::check(&state).is_ok())
}

/// With `GPU_CC=required`, a configuration the executor accepts on the host is refused
/// before the device opens unless the driver reports confidential computing on, DevTools
/// off, production, and the GPUs accepting work: on a machine without NVML (or without a
/// CC-mode GPU) the refusal is the CC check's, never the device's. Where the driver does
/// report it, the boot goes on to the device. In process and as the real binary.
#[test]
fn gpu_cc_required_is_checked_before_the_device() {
    let (dir, hash) = flash_shaped_dir();
    let mut map = cuda_env(dir, hash);
    map.insert(env::GPU_CC, "required".into());
    let err = boot_err(&map);
    let refusal = "EIDOLA_ENGINE_GPU_CC is required, but the GPUs are not in production \
                   confidential-computing mode";
    if driver_reports_cc() {
        assert!(!err.contains(env::GPU_CC), "{err}");
    } else {
        assert!(err.contains(refusal), "{err}");
        assert!(!err.contains(NO_DEVICE), "{err}");

        let output = std::process::Command::new(env!("CARGO_BIN_EXE_eidola-server-engine"))
            .env_clear()
            .envs(&map)
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&output.stdout);
        assert!(!output.status.success(), "{log}");
        assert!(log.contains(refusal), "{log}");
        assert!(!log.contains("listening"), "{log}");
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
// DeepSeek-V4.1 without a device.
// ---------------------------------------------------------------------------------------

/// The CUDA executor's settings over the DeepSeek-V4.1 dev model: one 64 MiB KV budget
/// per device.
fn ds_cuda_env() -> HashMap<&'static str, String> {
    let mut map = cuda_env(deepseek_model_dir(), deepseek_weights_hash());
    map.insert(env::MODEL_ID, DEEPSEEK_MODEL_ID.into());
    map.insert(env::KV_DEVICE_BYTES, format!("{0},{0}", 64u64 << 20));
    map
}

/// Whether the driver here reports the GPUs in production confidential-computing mode,
/// and at least two of them.
fn driver_reports_two_cc_gpus() -> bool {
    eidola_server_engine::gpu_cc::query_devices()
        .is_ok_and(|(state, n)| eidola_server_engine::gpu_cc::check_devices(&state, n, 2).is_ok())
}

/// A DeepSeek-V4.1 configuration the executor accepts on the host goes on to the devices
/// and, without one, fails closed there; with drafting within the drafter's slots too.
/// Never a CPU fallback.
#[test]
fn deepseek_reaches_the_devices_and_fails_closed_without_them() {
    for k in ["0", "3"] {
        let mut map = ds_cuda_env();
        map.insert(env::DRAFT_TOKENS, k.into());
        let err = boot_err(&map);
        assert!(!err.contains("unsupported"), "{k}: {err}");
        assert!(!err.contains(env::KV_DEVICE_BYTES), "{k}: {err}");
        if !Gpu::available() {
            assert!(err.contains(NO_DEVICE), "{k}: {err}");
        }
    }
}

/// `KV_DEVICE_BYTES` names one budget per device: DeepSeek-V4.1 runs on two.
#[test]
fn deepseek_names_one_kv_budget_per_device() {
    for budgets in [
        format!("{}", 64u64 << 20),
        format!("{0},{0},{0}", 64u64 << 20),
    ] {
        let mut map = ds_cuda_env();
        map.insert(env::KV_DEVICE_BYTES, budgets.clone());
        let err = boot_err(&map);
        assert!(
            err.contains(env::KV_DEVICE_BYTES) && err.contains("device budgets"),
            "{budgets}: {err}"
        );
        assert!(!err.contains(NO_DEVICE), "{err}");
    }
}

/// DeepSeek-V4.1's executor captures no decode graphs: `CUDA_GRAPHS=on` is refused before
/// the devices, `off` accepted.
#[test]
fn deepseek_refuses_decode_graphs() {
    let mut map = ds_cuda_env();
    map.insert(env::CUDA_GRAPHS, "on".into());
    let err = boot_err(&map);
    assert!(
        err.contains(env::CUDA_GRAPHS) && err.contains("no decode graphs"),
        "{err}"
    );
    assert!(!err.contains(NO_DEVICE), "{err}");
    map.insert(env::CUDA_GRAPHS, "off".into());
    let err = boot_err(&map);
    assert!(!err.contains(env::CUDA_GRAPHS), "{err}");
}

/// Storage, the weights hash and the model's files come before anything of the
/// executor's: a wrong expected hash is the refusal, graphs on or not, whatever the
/// devices.
#[test]
fn deepseek_checks_the_weights_hash_before_the_executor() {
    for graphs in ["off", "on"] {
        let mut map = ds_cuda_env();
        map.insert(env::CUDA_GRAPHS, graphs.into());
        map.insert(env::WEIGHTS_SHA256, "ab".repeat(32));
        let err = boot_err(&map);
        assert!(err.contains("weights hash"), "{err}");
        assert!(!err.contains(NO_DEVICE), "{err}");
        assert!(!err.contains(env::CUDA_GRAPHS), "{err}");
    }
}

/// Before the devices: a device budget that cannot hold every seat's windows and one
/// `MAX_MODEL_LEN` sequence of banks (either device's), a host tier under two blocks, and what the executor's preflight
/// refuses (a block size the pooling ratio does not divide). A draft width past the
/// cuda cap never parses.
#[test]
fn deepseek_budgets_are_refused_before_the_devices() {
    for (budgets, device) in [
        (format!("{},{}", 4096, 64u64 << 20), "device 0"),
        (format!("{},{}", 64u64 << 20, 4096), "device 1"),
    ] {
        let mut map = ds_cuda_env();
        map.insert(env::KV_DEVICE_BYTES, budgets);
        let err = boot_err(&map);
        assert!(
            err.contains(env::KV_DEVICE_BYTES) && err.contains("window KV") && err.contains(device),
            "{err}"
        );
        assert!(!err.contains(NO_DEVICE), "{err}");
    }
    let mut map = ds_cuda_env();
    map.insert(env::KV_HOST_BYTES, "4096".into());
    let err = boot_err(&map);
    assert!(err.contains(env::KV_HOST_BYTES), "{err}");
    assert!(!err.contains(NO_DEVICE), "{err}");

    let mut map = ds_cuda_env();
    map.insert(env::KV_BLOCK_SIZE, "1".into());
    map.insert(env::PREFIX_CACHE, "false".into());
    let err = boot_err(&map);
    assert!(err.contains("must divide the block size"), "{err}");
    assert!(!err.contains(NO_DEVICE), "{err}");

    let mut map = ds_cuda_env();
    map.insert(env::DRAFT_TOKENS, "4".into());
    let err = config_from(&map).unwrap_err();
    assert!(
        err.contains(env::DRAFT_TOKENS) && err.contains("cuda"),
        "{err}"
    );
}

/// With `GPU_CC=required`, DeepSeek-V4.1 is refused before any device opens unless the
/// driver reports the production CC state and at least two GPUs; where it does, the
/// boot goes on to the devices.
#[test]
fn deepseek_gpu_cc_required_is_checked_before_the_devices() {
    let mut map = ds_cuda_env();
    map.insert(env::GPU_CC, "required".into());
    let err = boot_err(&map);
    let refusal = "EIDOLA_ENGINE_GPU_CC is required, but the driver does not report 2 GPUs in \
                   production confidential-computing mode";
    if driver_reports_two_cc_gpus() {
        assert!(!err.contains(env::GPU_CC), "{err}");
    } else {
        assert!(err.contains(refusal), "{err}");
        assert!(!err.contains(NO_DEVICE), "{err}");
    }
}

/// Greedy generation of `prompt` through the engine core over `exec`.
fn greedy<E: eidola_engine::executor::Executor>(
    exec: E,
    speculative: bool,
    prompt: Vec<u32>,
    max_tokens: u32,
) -> Vec<u32> {
    use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
    use eidola_engine::kv::CachePolicy;
    use eidola_engine::sampling::SamplingParams;
    let model = deepseek_loaded();
    let mut engine = Engine::new(
        exec,
        SchedulerConfig {
            max_batched_tokens: 256,
            max_seqs: 4,
            max_prefill_chunk: 256,
            eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
            speculative,
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
            required_cached_tokens: 0,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
        })
        .unwrap();
    let mut out = Vec::new();
    for now in 0.. {
        for ev in engine.step(now).unwrap() {
            out.extend(ev.tokens);
            if ev.finish.is_some() {
                return out;
            }
        }
    }
    unreachable!()
}

/// The executor configuration the node derives for DeepSeek-V4.1 (`cuda::deepseek_config`,
/// over the dev model's verified configuration, its Engram map and its sampleable
/// vocabulary), run over host devices in place of the GPUs, generates exactly the CPU
/// executor's greedy tokens, without drafts and drafting two.
#[test]
fn the_nodes_deepseek_configuration_computes_what_the_cpu_executor_does() {
    use eidola_engine_cpu::deepseek::{DsCpuExecutor, DsCpuExecutorConfig};
    use eidola_engine_cuda::deepseek::DsCudaExecutor;
    use eidola_engine_cuda::deepseek::host::HostDevice;
    use eidola_server_engine::model::Served;

    let model = deepseek_loaded();
    let Served::DeepSeek {
        config: ds,
        token_map,
        model: Some(reference),
    } = model.served()
    else {
        panic!("the DeepSeek dev model, loaded for the cpu executor");
    };
    assert!(token_map.is_some(), "the dev model has Engram layers");
    let input = eidola_engine_chat::ChatInput::from_json(
        &json!([{"role": "user", "content": "Name three primary colours."}]).to_string(),
        None,
    )
    .unwrap();
    let prompt = eidola_engine_chat::deepseek::encode(
        &input,
        eidola_server_engine::pipeline::deepseek_options(),
    )
    .unwrap();
    let prompt = model.tokenizer().encode_prompt(&prompt).unwrap();
    let max_tokens = 12;

    let cpu = DsCpuExecutor::new(
        reference.clone(),
        DsCpuExecutorConfig {
            block_size: 16,
            num_blocks: 256,
            window_blocks: 256,
            host_blocks: 0,
            num_state_slots: 4,
            max_model_len: 1024,
            buckets: vec![eidola_engine::spec::Bucket {
                max_seqs: 4,
                max_tokens: 256,
            }],
            draft_width: 0,
            sampleable_vocab_size: model.sampleable_vocab_size(),
            record: false,
        },
    );
    let expected = greedy(cpu, false, prompt.clone(), max_tokens);
    assert!(!expected.is_empty());

    for k in ["0", "2"] {
        let mut map = ds_cuda_env();
        map.insert(env::DRAFT_TOKENS, k.into());
        let config = config_from(&map).unwrap();
        let ExecutorConfig::Cuda {
            kv_device_bytes,
            further_kv_device_bytes,
            graphs,
            ..
        } = &config.executor
        else {
            unreachable!()
        };
        let exec_config = eidola_server_engine::cuda::deepseek_config(
            ds,
            &config.sizing,
            &config.cache,
            [*kv_device_bytes, further_kv_device_bytes[0]],
            config.kv_host_bytes,
            *graphs,
            model.sampleable_vocab_size(),
        )
        .unwrap();
        let exec = DsCudaExecutor::with_devices(
            Box::new(HostDevice),
            Box::new(HostDevice),
            model.store(),
            ds,
            token_map.clone(),
            &exec_config,
        )
        .unwrap();
        let got = greedy(
            exec,
            config.sizing.draft_tokens > 0,
            prompt.clone(),
            max_tokens,
        );
        assert_eq!(got, expected, "draft width {k}");
    }
}

// ---------------------------------------------------------------------------------------
// Qwen3.8 without a device.
// ---------------------------------------------------------------------------------------

/// The CUDA executor's settings over the Qwen3.8 dev model: a 64 MiB KV budget.
fn qwen_cuda_env() -> HashMap<&'static str, String> {
    let mut map = cuda_env(qwen_model_dir(), qwen_weights_hash());
    map.insert(env::MODEL_ID, QWEN_MODEL_ID.into());
    map.insert(env::KV_DEVICE_BYTES, (64u64 << 20).to_string());
    map
}

/// A Qwen3.8 configuration the node accepts on the host goes on to the device and,
/// without one, fails closed there, drafting or not, with graphs on or off. Never a CPU
/// fallback.
#[test]
fn qwen_reaches_the_device_and_fails_closed_without_it() {
    for (k, graphs) in [("0", "off"), ("3", "off"), ("3", "on")] {
        let mut map = qwen_cuda_env();
        map.insert(env::DRAFT_TOKENS, k.into());
        map.insert(env::CUDA_GRAPHS, graphs.into());
        let err = boot_err(&map);
        assert!(!err.contains(env::KV_DEVICE_BYTES), "{k}: {err}");
        if !Gpu::available() {
            assert!(err.contains(NO_DEVICE), "{k}: {err}");
        }
    }
}

/// Before the device: a budget that names two devices or cannot hold the state pool and one sequence, a host tier under two of
/// each, and a wrong weights hash before anything of the executor's.
#[test]
fn qwen_refusals_come_before_the_device() {
    let refused = |overrides: &[(&'static str, &str)], needles: &[&str]| {
        let mut map = qwen_cuda_env();
        for (k, v) in overrides {
            map.insert(k, v.to_string());
        }
        let err = boot_err(&map);
        for needle in needles {
            assert!(err.contains(needle), "{overrides:?}: {err}");
        }
        assert!(!err.contains(NO_DEVICE), "{overrides:?}: {err}");
    };
    let two = format!("{0},{0}", 64u64 << 20);
    refused(&[(env::KV_DEVICE_BYTES, &two)], &["device budgets"]);
    refused(
        &[(env::KV_DEVICE_BYTES, "4096")],
        &[env::KV_DEVICE_BYTES, "state pool"],
    );
    let states = eidola_common::engine_deployment::recurrent_state_buffers(
        &config_from(&qwen_cuda_env()).unwrap().sizing,
        &config_from(&qwen_cuda_env()).unwrap().cache,
    );
    let image = eidola_server_engine::cuda::qwen_dims(qwen_loaded().qwen_config().unwrap())
        .state_image_bytes();
    let only_states = (states * image + 1000).to_string();
    refused(&[(env::KV_DEVICE_BYTES, &only_states)], &["fewer than the"]);
    refused(&[(env::KV_HOST_BYTES, "4096")], &[env::KV_HOST_BYTES]);
    for graphs in ["off", "on"] {
        refused(
            &[
                (env::WEIGHTS_SHA256, &"ab".repeat(32)),
                (env::CUDA_GRAPHS, graphs),
            ],
            &["weights hash"],
        );
    }
}

/// An allocation the counting device made, kept with its size and the device's tally.
struct Counted {
    _inner: Box<dyn std::any::Any>,
    bytes: u64,
    tally: Arc<std::sync::Mutex<Tally>>,
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.tally.lock().unwrap().live -= self.bytes;
    }
}

/// The device allocations live, their peak, and every allocation in order: its size
/// and the bytes live after it.
#[derive(Default)]
struct Tally {
    live: u64,
    peak: u64,
    allocs: Vec<(u64, u64)>,
}

/// A Qwen3.8 device that counts its allocations and otherwise is the device it wraps.
struct Counting<D> {
    inner: D,
    tally: Arc<std::sync::Mutex<Tally>>,
}

mod counting {
    use super::*;
    use eidola_engine::kv_cipher::KvNonce;
    use eidola_engine::secret::KvKey;
    use eidola_engine_cuda::Result;
    use eidola_engine_cuda::qwen::abi::*;
    use eidola_engine_cuda::qwen::attention::Fa2Args;
    use eidola_engine_cuda::qwen::device::{QwenDevice, QwenProfile};
    use eidola_engine_cuda::qwen::kernels::{AcceptLaunch, QwenKernels, SampleArgs};

    impl<D: QwenKernels> QwenKernels for Counting<D> {
        unsafe fn embed(&self, p: &EmbedParams) -> Result<()> {
            unsafe { self.inner.embed(p) }
        }
        unsafe fn norm(&self, p: &NormParams) -> Result<()> {
            unsafe { self.inner.norm(p) }
        }
        unsafe fn quant(&self, p: &QuantParams) -> Result<()> {
            unsafe { self.inner.quant(p) }
        }
        unsafe fn lin8(&self, p: &Lin8Params) -> Result<()> {
            unsafe { self.inner.lin8(p) }
        }
        unsafe fn linear(&self, p: &LinearParams, fast: Option<u64>) -> Result<()> {
            unsafe { self.inner.linear(p, fast) }
        }
        unsafe fn gemm_bf16(&self, p: &LinearParams) -> Result<()> {
            unsafe { self.inner.gemm_bf16(p) }
        }
        unsafe fn copy_rows(&self, p: &CopyParams) -> Result<()> {
            unsafe { self.inner.copy_rows(p) }
        }
        unsafe fn sample(&self, a: &SampleArgs) -> Result<()> {
            unsafe { self.inner.sample(a) }
        }
        unsafe fn accept(&self, a: &AcceptLaunch) -> Result<()> {
            unsafe { self.inner.accept(a) }
        }
        unsafe fn fa2(&self, a: &Fa2Args) -> Result<()> {
            unsafe { self.inner.fa2(a) }
        }
        unsafe fn gate(&self, p: &GateParams) -> Result<()> {
            unsafe { self.inner.gate(p) }
        }
        unsafe fn kv_write(&self, p: &KvWriteParams) -> Result<()> {
            unsafe { self.inner.kv_write(p) }
        }
        unsafe fn attention(&self, p: &AttnParams) -> Result<()> {
            unsafe { self.inner.attention(p) }
        }
        unsafe fn dn_prep(&self, p: &DnPrepParams) -> Result<()> {
            unsafe { self.inner.dn_prep(p) }
        }
        unsafe fn dn_rec(&self, p: &DnRecParams) -> Result<()> {
            unsafe { self.inner.dn_rec(p) }
        }
        unsafe fn dn_rec_short(&self, p: &DnRecParams) -> Result<()> {
            unsafe { self.inner.dn_rec_short(p) }
        }
        unsafe fn dn_norm(&self, p: &DnNormParams) -> Result<()> {
            unsafe { self.inner.dn_norm(p) }
        }
        unsafe fn widen(&self, p: &WidenParams) -> Result<()> {
            unsafe { self.inner.widen(p) }
        }
        unsafe fn round(&self, p: &RoundParams) -> Result<()> {
            unsafe { self.inner.round(p) }
        }
    }

    impl<D: QwenDevice> QwenDevice for Counting<D> {
        fn profile(&self) -> QwenProfile {
            self.inner.profile()
        }
        fn alloc(&self, bytes: usize) -> Result<(u64, Box<dyn std::any::Any>)> {
            let (addr, keep) = self.inner.alloc(bytes)?;
            let bytes = bytes as u64;
            let mut t = self.tally.lock().unwrap();
            t.live += bytes;
            t.peak = t.peak.max(t.live);
            let live = t.live;
            t.allocs.push((bytes, live));
            drop(t);
            Ok((
                addr,
                Box::new(Counted {
                    _inner: keep,
                    bytes,
                    tally: self.tally.clone(),
                }),
            ))
        }
        unsafe fn zero(&self, addr: u64, bytes: usize) -> Result<()> {
            unsafe { self.inner.zero(addr, bytes) }
        }
        unsafe fn upload(&self, addr: u64, src: &[u8]) -> Result<()> {
            unsafe { self.inner.upload(addr, src) }
        }
        unsafe fn download(&self, addr: u64, dst: &mut [u8]) -> Result<()> {
            unsafe { self.inner.download(addr, dst) }
        }
        unsafe fn copy(&self, dst: u64, src: u64, bytes: usize) -> Result<()> {
            unsafe { self.inner.copy(dst, src, bytes) }
        }
        unsafe fn cipher(
            &self,
            addr: u64,
            words: u32,
            counter: u32,
            key: &KvKey,
            nonce: &KvNonce,
        ) -> Result<()> {
            unsafe { self.inner.cipher(addr, words, counter, key, nonce) }
        }
        fn alloc_host(&self, bytes: usize) -> Result<(*mut u8, Box<dyn std::any::Any>)> {
            self.inner.alloc_host(bytes)
        }
        unsafe fn copy_to_host(&self, dst: *mut u8, src: u64, bytes: usize) -> Result<()> {
            unsafe { self.inner.copy_to_host(dst, src, bytes) }
        }
        unsafe fn copy_from_host(&self, dst: u64, src: *const u8, bytes: usize) -> Result<()> {
            unsafe { self.inner.copy_from_host(dst, src, bytes) }
        }
        fn synchronize(&self) -> Result<()> {
            self.inner.synchronize()
        }
        fn capture(&self, run: &mut dyn FnMut() -> Result<()>) -> Result<Box<dyn std::any::Any>> {
            self.inner.capture(run)
        }
        fn replay(&self, graph: &dyn std::any::Any) -> Result<()> {
            self.inner.replay(graph)
        }
        fn free_memory(&self) -> Result<i64> {
            self.inner.free_memory()
        }
    }
}

/// The kernels' own bodies compiled for the host (the CUDA crate's
/// `oracle/qwen_launch_host.cc`), built once per test process, or `None` without a C++
/// compiler.
fn qwen_host_bodies() -> Option<Arc<libloading::Library>> {
    use std::process::Command;
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    if Command::new(&cxx).arg("--version").output().is_err() {
        eprintln!("SKIPPED: no C++ compiler ({cxx}) for the Qwen3.8 kernel bodies");
        return None;
    }
    static BUILT: std::sync::OnceLock<Arc<libloading::Library>> = std::sync::OnceLock::new();
    Some(
        BUILT
            .get_or_init(|| {
                let cuda = Path::new(env!("CARGO_MANIFEST_DIR")).join("../eidola-engine-cuda");
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos());
                let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
                    "libqwen_launch_host-node-{}-{nanos}.{}",
                    std::process::id(),
                    std::env::consts::DLL_EXTENSION
                ));
                let status = Command::new(&cxx)
                    .args(["-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-fPIC"])
                    .arg("-I")
                    .arg(cuda.join("../eidola-engine-kernels/csrc"))
                    .arg(cuda.join("oracle/qwen_launch_host.cc"))
                    .arg("-o")
                    .arg(&out)
                    .status()
                    .expect("run the C++ compiler");
                assert!(
                    status.success(),
                    "{cxx} failed on oracle/qwen_launch_host.cc"
                );
                // SAFETY: a library this test just built from the CUDA crate's own source;
                // it has no initialisers.
                Arc::new(unsafe { libloading::Library::new(&out) }.expect("load the bodies"))
            })
            .clone(),
    )
}

/// The compiled bodies as a host device's launches.
fn qwen_launch(lib: Arc<libloading::Library>) -> Box<eidola_engine_cuda::qwen::host::HostLaunch> {
    type HostFn = unsafe extern "C" fn(*const std::ffi::c_void, u32, u32);
    Box::new(move |family, params, gx, gy| {
        let name = format!("qwen_host_{family}\0");
        // SAFETY: every `qwen_host_*` symbol takes its family's parameter struct.
        unsafe {
            let f: libloading::Symbol<HostFn> =
                lib.get(name.as_bytes()).expect("a host body per family");
            f(params, gx, gy);
        }
    })
}

/// Greedy generation of `prompt` through the engine core over `exec` at the node's
/// scheduler limits for `sizing`.
fn qwen_greedy<E: eidola_engine::executor::Executor>(
    exec: E,
    sizing: &eidola_server_engine::config::Sizing,
    prompt: Vec<u32>,
    max_tokens: u32,
) -> Vec<u32> {
    use eidola_engine::engine::{CacheScope, Engine, Request, SchedulerConfig};
    use eidola_engine::kv::CachePolicy;
    use eidola_engine::sampling::SamplingParams;
    let model = qwen_loaded();
    let mut engine = Engine::new(
        exec,
        SchedulerConfig {
            max_batched_tokens: sizing.max_batched_tokens,
            max_seqs: sizing.max_seqs,
            max_prefill_chunk: sizing.max_prefill_chunk,
            eos_token_ids: model.tokenizer().eos_token_ids().to_vec(),
            speculative: sizing.draft_tokens > 0,
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
            required_cached_tokens: 0,
            stop_token_ids: Vec::new(),
            cache: CacheScope::Private,
        })
        .unwrap();
    let mut out = Vec::new();
    for now in 0.. {
        for ev in engine.step(now).unwrap() {
            out.extend(ev.tokens);
            if ev.finish.is_some() {
                return out;
            }
        }
    }
    unreachable!()
}

/// The executor configuration the node derives for Qwen3.8 (`cuda::qwen_config`, over
/// the dev model's verified configuration and sampleable vocabulary), run over the
/// kernels' bodies compiled for the host in the exact profile, generates exactly the CPU
/// executor's greedy tokens, without drafts and drafting two, eagerly and on the padded
/// rung path (run directly: the host device captures no graphs).
#[test]
fn the_nodes_qwen_configuration_computes_what_the_cpu_executor_does() {
    use eidola_engine_cpu::qwen::QwenCpuExecutor;
    use eidola_engine_cuda::qwen::{DecodePath, QwenCudaExecutor};
    use eidola_server_engine::model::Served;

    let model = qwen_loaded();
    let Served::Qwen {
        model: reference, ..
    } = model.served()
    else {
        panic!("the Qwen dev model");
    };
    let prompt = qwen_long_prompt();
    let max_tokens = 8;
    let base = config_from(&qwen_env_map()).unwrap();
    let sizing = eidola_server_engine::config::Sizing {
        draft_tokens: 0,
        ..base.sizing
    };
    let cpu_config = eidola_server_engine::qwen_cpu_config(
        reference.config(),
        &sizing,
        &base.cache,
        256,
        0,
        model.sampleable_vocab_size(),
    )
    .unwrap();
    let expected = qwen_greedy(
        QwenCpuExecutor::new(reference.clone(), cpu_config),
        &sizing,
        prompt.clone(),
        max_tokens,
    );
    assert!(!expected.is_empty());

    let Some(lib) = qwen_host_bodies() else {
        return;
    };
    for k in ["0", "2"] {
        for path in [DecodePath::Eager, DecodePath::Direct] {
            let (exec_config, sizing) = qwen_cuda_config(k, "off");
            let dev = eidola_engine_cuda::qwen::host::HostDevice::new(qwen_launch(lib.clone()));
            let mut exec = QwenCudaExecutor::new(dev, reference, exec_config).unwrap();
            exec.set_decode_path(path).unwrap();
            let got = qwen_greedy(exec, &sizing, prompt.clone(), max_tokens);
            assert_eq!(got, expected, "draft width {k}, {path:?}");
        }
    }
    // Graphs on: the host device captures none, so construction refuses.
    let (on, _) = qwen_cuda_config("2", "on");
    let dev = eidola_engine_cuda::qwen::host::HostDevice::new(qwen_launch(lib));
    assert!(QwenCudaExecutor::new(dev, reference, on).is_err());
}

/// A prompt that fills the dev node's 1,024-position window in whole steps.
fn qwen_long_prompt() -> Vec<u32> {
    let model = qwen_loaded();
    let eidola_server_engine::model::Served::Qwen { template, .. } = model.served() else {
        panic!("the Qwen dev model");
    };
    let text = "Name three primary colours, and say why each is one. ".repeat(60);
    let input = eidola_engine_chat::ChatInput::from_json(
        &json!([{"role": "user", "content": text}]).to_string(),
        None,
    )
    .unwrap();
    let prompt = template
        .render_prompt(&input, &eidola_server_engine::pipeline::qwen_options())
        .unwrap();
    let prompt = model.tokenizer().encode_prompt(&prompt).unwrap();
    assert!(
        prompt.len() > 700 && prompt.len() < 1000,
        "{}",
        prompt.len()
    );
    prompt
}

/// The node's Qwen3.8 executor configuration over the dev model at draft width `k`,
/// graphs `graphs`, and its sizing.
fn qwen_cuda_config(
    k: &str,
    graphs: &str,
) -> (
    eidola_engine_cuda::qwen::QwenCudaExecutorConfig,
    eidola_server_engine::config::Sizing,
) {
    let model = qwen_loaded();
    qwen_cuda_config_for(
        model.qwen_config().unwrap(),
        model.sampleable_vocab_size(),
        k,
        graphs,
    )
}

/// [`qwen_cuda_config`] for the model `qwen`, `sampleable` of its ids sampled.
fn qwen_cuda_config_for(
    qwen: &eidola_engine_model::qwen::QwenConfig,
    sampleable: u32,
    k: &str,
    graphs: &str,
) -> (
    eidola_engine_cuda::qwen::QwenCudaExecutorConfig,
    eidola_server_engine::config::Sizing,
) {
    let mut map = qwen_cuda_env();
    map.insert(env::DRAFT_TOKENS, k.into());
    map.insert(env::CUDA_GRAPHS, graphs.into());
    let config = config_from(&map).unwrap();
    let ExecutorConfig::Cuda {
        kv_device_bytes,
        graphs,
        ..
    } = &config.executor
    else {
        unreachable!()
    };
    let exec_config = eidola_server_engine::cuda::qwen_config(
        qwen,
        &config.sizing,
        &config.cache,
        *kv_device_bytes,
        config.kv_host_bytes,
        *graphs,
        sampleable,
    )
    .unwrap();
    (exec_config, config.sizing)
}

/// In the fast profile, which the node runs (planned as on the device, FA2 checked
/// rather than computed; over the model crate's 256-wide fixture, since the profile's
/// attention is compiled for heads of 256 and the dev model's are 128), the executor's
/// pools are the sizes the shared rules count
/// (`engine_deployment::recurrent_kv_block_bytes` with each pool's pad block, the state
/// image, `recurrent_seat_bytes` for every seat and the padding seat), and everything
/// else it allocates, at construction and as its eager table grows over a prompt that
/// fills the window in whole steps and then decodes, stays within
/// `recurrent_step_scratch_bytes`, without drafts and drafting two.
#[test]
fn the_nodes_qwen_pools_and_scratch_are_the_shared_counts() {
    use eidola_common::engine_deployment::{
        recurrent_kv_block_bytes, recurrent_seat_bytes, recurrent_step_scratch_bytes,
    };
    use eidola_engine_cuda::qwen::{QwenCudaExecutor, QwenProfile};
    let Some(lib) = qwen_host_bodies() else {
        return;
    };
    let reference = eidola_engine_model::qwen::load_reference(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../eidola-engine-model/tests/fixtures/qwen_tiny256"),
        Ok,
    )
    .unwrap();
    let vocab = reference.config().vocab_size as u32;
    let dims = eidola_server_engine::cuda::qwen_dims(reference.config());
    // A prompt that fills the window in whole steps.
    let prompt: Vec<u32> = (0..900u32).map(|i| (i * 37 + 11) % 500).collect();
    for k in ["0", "2"] {
        let (exec_config, s) = qwen_cuda_config_for(reference.config(), vocab, k, "off");
        let tally = Arc::new(std::sync::Mutex::new(Tally::default()));
        let dev = Counting {
            inner: eidola_engine_cuda::qwen::host::HostDevice::planning_fast(qwen_launch(
                lib.clone(),
            )),
            tally: tally.clone(),
        };
        assert_eq!(
            eidola_engine_cuda::qwen::QwenDevice::profile(&dev),
            QwenProfile::Fast
        );
        let exec = QwenCudaExecutor::new(dev, &reference, exec_config.clone()).unwrap();

        // The pools, in the executor's order after its weights: the target's KV, with
        // drafting the drafter's KV and its taps, the states, then each seat's live
        // image, tail (its convolution rows, then its decay and write rows) and kept
        // states, the padding seat among them.
        let blocks = u64::from(exec_config.num_blocks) + 1;
        let b = u64::from(s.kv_block_size);
        let mut want = vec![blocks * b * dims.kv_bytes_per_position()];
        if s.draft_tokens > 0 {
            want.push(blocks * b * dims.drafter_kv_bytes_per_position());
            want.push(blocks * dims.tap_bytes());
        }
        want.push(u64::from(exec_config.state_buffers) * dims.state_image_bytes());
        let (base, pools) = {
            let t = tally.lock().unwrap();
            let at = t
                .allocs
                .iter()
                .position(|&(bytes, _)| bytes == want[0])
                .expect("the target's pool");
            let n = want.len() + 4;
            (
                t.allocs[at + n - 1].1,
                t.allocs[at..at + n].iter().map(|a| a.0).collect::<Vec<_>>(),
            )
        };
        assert_eq!(pools[..want.len()], want[..], "k = {k}");
        assert_eq!(
            want[..want.len() - 1].iter().sum::<u64>() / blocks,
            recurrent_kv_block_bytes(&dims, &s),
            "k = {k}"
        );
        assert_eq!(
            pools[want.len()..].iter().sum::<u64>(),
            (u64::from(s.max_seqs) + 1) * recurrent_seat_bytes(&dims, &s),
            "k = {k}"
        );

        let _ = qwen_greedy(exec, &s, prompt.clone(), 24);
        let t = tally.lock().unwrap();
        let scratch = t.peak - base;
        let bound = recurrent_step_scratch_bytes(&dims, &s);
        eprintln!("k = {k}: step scratch {scratch} of the {bound} counted");
        assert!(scratch <= bound, "k = {k}: {scratch} > {bound}");
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
    // A host tier when the run asks for one (bytes), none otherwise.
    if let Ok(bytes) = std::env::var("EIDOLA_TEST_KV_HOST_BYTES") {
        map.insert(env::KV_HOST_BYTES, bytes);
    }
    let config = config_from(&map).unwrap();
    let kv_host_bytes = config.kv_host_bytes;
    let (sizing, cache) = (config.sizing, config.cache);
    let (kernels_dir, kv_device_bytes, graphs, gpu_cc) = match &config.executor {
        ExecutorConfig::Cuda {
            kernels_dir,
            kv_device_bytes,
            graphs,
            gpu_cc,
            ..
        } => (kernels_dir.clone(), *kv_device_bytes, *graphs, *gpu_cc),
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
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let r = chat(body.clone()).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    let v = completion(&events_of(&r.bytes().await.unwrap()));
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

    // The same request read as plain server-sent events: the same text.
    let sse = chat(body).send().await.unwrap().text().await.unwrap();
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
        use eidola_engine_chat::mimo::RenderOptions;
        use eidola_engine_chat::{
            ChatDelta, ChatInput, OutputConfig, OutputParser, PrefixedCallIds, StopCause,
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
                    continue_final_message: None,
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
            kv_host_bytes,
            graphs,
            gpu_cc,
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
                required_cached_tokens: 0,
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
            FinishReason::Length | FinishReason::KvExhausted => StopCause::MaxTokens,
            FinishReason::Cancelled | FinishReason::CachedPrefixMissing => {
                unreachable!("the direct run cancels nothing and requires no prefix")
            }
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

/// **A continuation on the real model resumes exactly where its generation stopped.**
/// Greedy request A runs to its end; B is A cut off at a small ceiling, once inside the
/// reasoning and once inside the answer; C continues B (`continue_final_message`: the
/// reasoning while B had no answer, the answer after, with `add_generation_prompt:
/// false`). Greedy decoding over a batch-invariant engine makes B's text joined to C's
/// equal A's byte for byte, and C under B's cache key is served from the prefix B left:
/// at least everything B fed the engine up to its last sealed block. Same environment as
/// [`real_flash_serves_what_the_executor_computes`]; skipped without it.
#[tokio::test(flavor = "multi_thread")]
async fn real_flash_continues_a_cut_off_generation_exactly() {
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
    let block = u64::from(config.sizing.kv_block_size);
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

    let question = "Why is the sky blue? Explain in two short sentences.";
    let n: u64 = 1024;
    let send = |mut body: Value| {
        body["stream"] = true.into();
        body["stream_options"] = json!({"include_usage": true});
        let request = client
            .post(format!("{base}/v1/chat/completions"))
            .bearer_auth(TOKEN)
            .header("x-eidola-weights-sha256", &hash)
            .json(&body);
        async move {
            let r = request.send().await.unwrap();
            let status = r.status().as_u16();
            let bytes = r.bytes().await.unwrap();
            assert_eq!(status, 200, "{}", String::from_utf8_lossy(&bytes));
            completion(&events_of(&bytes))
        }
    };
    let text_of = |v: &Value, key: &str| {
        v["choices"][0]["message"][key]
            .as_str()
            .unwrap_or("")
            .to_string()
    };
    let key = |byte: u8| {
        use base64::Engine as _;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([byte; 32])
    };

    // A: the whole answer, greedy, with no key so it shares nothing with B or C.
    let a = send(json!({
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": question}],
        "max_completion_tokens": n,
        "temperature": 0.0,
    }))
    .await;
    eprintln!("A: {a}");
    assert_eq!(
        a["choices"][0]["finish_reason"], "stop",
        "A must finish: {a}"
    );
    let (a_reasoning, a_content) = (text_of(&a, "reasoning_content"), text_of(&a, "content"));
    let a_prompt = a["usage"]["prompt_tokens"].as_u64().unwrap();
    let reasoning_tokens = model.tokenizer().encode(&a_reasoning).unwrap().len() as u64;
    let content_tokens = model.tokenizer().encode(&a_content).unwrap().len() as u64;
    assert!(
        reasoning_tokens >= 8 && content_tokens >= 8,
        "A needs reasoning and an answer to cut inside: {reasoning_tokens}/{content_tokens}"
    );

    // Generated tokens are `<think>`, the reasoning, `</think>`, the answer: cut in the
    // middle of the reasoning, then in the middle of the answer.
    let cuts = [
        ("reasoning", 1 + reasoning_tokens / 2, 1u8),
        ("answer", 2 + reasoning_tokens + content_tokens / 2, 2u8),
    ];
    for (place, k, key_byte) in cuts {
        let cache_key = key(key_byte);
        let b = send(json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": question}],
            "max_completion_tokens": k,
            "temperature": 0.0,
            "cache_key": cache_key,
        }))
        .await;
        eprintln!("B ({place}, k={k}): {b}");
        assert_eq!(b["choices"][0]["finish_reason"], "length", "{place}: {b}");
        assert_eq!(
            b["choices"][0]["continuable"], true,
            "{place}: a clean cut says it can be continued exactly: {b}"
        );
        let (b_reasoning, b_content) = (text_of(&b, "reasoning_content"), text_of(&b, "content"));
        let field = match place {
            "reasoning" => {
                assert!(b_content.is_empty(), "the cut is inside the reasoning: {b}");
                json!("reasoning_content")
            }
            _ => {
                assert!(!b_content.is_empty(), "the cut is inside the answer: {b}");
                json!(true)
            }
        };

        let c = send(json!({
            "model": MODEL_ID,
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "reasoning_content": b_reasoning, "content": b_content},
            ],
            "max_completion_tokens": n - k,
            "temperature": 0.0,
            "add_generation_prompt": false,
            "continue_final_message": field,
            "cache_key": cache_key,
        }))
        .await;
        eprintln!("C ({place}): {c}");
        assert_eq!(c["choices"][0]["finish_reason"], "stop", "{place}: {c}");
        assert_eq!(
            format!("{b_reasoning}{}", text_of(&c, "reasoning_content")),
            a_reasoning,
            "{place}: the reasoning resumes exactly"
        );
        assert_eq!(
            format!("{b_content}{}", text_of(&c, "content")),
            a_content,
            "{place}: the answer resumes exactly"
        );

        // B fed the engine its prompt and all but its last sampled token; the
        // continuation is served from B's last sealed block at least.
        let cached = c["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap();
        let c_prompt = c["usage"]["prompt_tokens"].as_u64().unwrap();
        let sealed = (a_prompt + k - 1) / block * block;
        eprintln!("{place}: cached {cached} of {c_prompt}; B sealed {sealed}");
        assert!(c_prompt >= a_prompt + k - 1, "{place}: {c_prompt}");
        assert!(
            cached >= sealed,
            "{place}: {cached} cached, expected at least {sealed}"
        );
    }

    drop(client);
    server.abort();
    let _ = server.await;
    drop(node.engine);
    drop(node.admission);
    tokio::time::timeout(Duration::from_secs(120), engine_stopped)
        .await
        .expect("the engine thread stops once nothing can reach it")
        .ok();
}

// ---------------------------------------------------------------------------------------
// With two devices: the DeepSeek-V4.1 dev model.
// ---------------------------------------------------------------------------------------

/// **The node serves DeepSeek-V4.1 on two GPUs as the CPU executor does.** Boots the node
/// on the DeepSeek-V4.1 dev model with `cuda` (device 0 on GPU 0, device 1 on GPU 1; no
/// decode graphs), at draft widths 0 and 2, and holds its greedy answers (reasoning,
/// content, finish reason, prompt and completion tokens) to a node on the same model with
/// the CPU executor: the dev model on the GPU computes the CPU executor's bits, so the
/// text is identical. Needs two GPUs and `EIDOLA_ENGINE_KERNELS_DIR` (the kernel build
/// output), and fails without them; `EIDOLA_ENGINE_GPU_CC` as for the other device tests
/// (`not-required` when unset).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs two GPUs and EIDOLA_ENGINE_KERNELS_DIR"]
async fn deepseek_dev_model_on_two_gpus_answers_as_the_cpu_executor() {
    let kernels = std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR")
        .expect("EIDOLA_ENGINE_KERNELS_DIR names the kernel build output");
    assert!(
        Gpu::open(0).is_ok() && Gpu::open(1).is_ok(),
        "DeepSeek-V4.1 runs on two GPUs, device d on GPU d"
    );
    let cuda_model = Arc::new(
        tokio::task::spawn_blocking(|| {
            eidola_server_engine::model::LoadedModel::load(
                deepseek_model_dir(),
                deepseek_weights_hash(),
                WeightsStorage::DevWritable,
                ExecutorKind::Cuda,
            )
            .unwrap()
        })
        .await
        .unwrap(),
    );
    let requests = [
        json!({
            "model": DEEPSEEK_MODEL_ID,
            "messages": [{"role": "user", "content": "Name three primary colours."}],
            "max_completion_tokens": 32,
            "temperature": 0.0,
        }),
        json!({
            "model": DEEPSEEK_MODEL_ID,
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "The quick brown fox jumps over the lazy dog. \
                    What does the fox jump over, and why might it?"},
            ],
            "max_completion_tokens": 48,
            "temperature": 0.0,
        }),
    ];
    for k in ["0", "2"] {
        let cpu = TestNode::start_deepseek(&[(env::DRAFT_TOKENS, k)]).await;
        let mut map = ds_cuda_env();
        map.insert(env::DRAFT_TOKENS, k.into());
        map.insert(
            env::KERNELS_DIR,
            PathBuf::from(&kernels).display().to_string(),
        );
        let config = config_from(&map).unwrap();
        let node = {
            let model = cuda_model.clone();
            tokio::task::spawn_blocking(move || eidola_server_engine::start(config, model))
                .await
                .unwrap()
                .unwrap()
        };
        let gpu = TestNode::serve(node).await;

        let info: Value = gpu
            .client
            .get(format!("{}/v1/engine/info", gpu.base))
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

        for body in &requests {
            let (status, want) = cpu.chat_json(body).await;
            assert_eq!(status, 200, "{want}");
            let (status, got) = gpu.chat_json(body).await;
            assert_eq!(status, 200, "{got}");
            eprintln!("k={k} cuda: {got}");
            for key in ["reasoning_content", "content"] {
                assert_eq!(
                    got["choices"][0]["message"][key], want["choices"][0]["message"][key],
                    "k={k} {key}"
                );
            }
            assert_eq!(
                got["choices"][0]["finish_reason"], want["choices"][0]["finish_reason"],
                "k={k}"
            );
            for count in ["prompt_tokens", "completion_tokens"] {
                assert_eq!(got["usage"][count], want["usage"][count], "k={k} {count}");
            }
        }
    }
}
