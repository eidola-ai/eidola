//! The released Qwen checkpoints' recorded bytes, for the real-weights tests:
//! `golden/qwen/upstream-sha256.json` (Qwen3.8-27B-FP8) and
//! `golden/qwen/upstream-qwen3.5-*-sha256.json` (Qwen3.5-2B and -0.8B) hold
//! the SHA-256 of every published file used here at the pinned revisions. A
//! test loads nothing it has not checked against its checkpoint's list: a
//! download of the right size and the wrong bytes gives a model that agrees
//! with a reference run on the same bytes and is not the model.

use std::collections::BTreeMap;
use std::path::Path;

use eidola_engine_model::WeightSet;
use eidola_engine_model::safetensors::SEMANTIC_FILES;
use sha2::{Digest, Sha256};

/// Every recorded list, by the repository it names.
const RECORDED: [(&str, &str); 3] = [
    (
        "Qwen/Qwen3.8-27B-FP8",
        include_str!("../../golden/qwen/upstream-sha256.json"),
    ),
    (
        "Qwen/Qwen3.5-2B",
        include_str!("../../golden/qwen/upstream-qwen3.5-2b-sha256.json"),
    ),
    (
        "Qwen/Qwen3.5-0.8B",
        include_str!("../../golden/qwen/upstream-qwen3.5-0.8b-sha256.json"),
    ),
];

/// The recorded list whose `config.json` is the directory's (which names the
/// checkpoint), and that repository's name.
pub fn recorded_for(dir: &Path) -> (&'static str, serde_json::Value) {
    let config = std::fs::read(dir.join("config.json")).expect("the directory has a config.json");
    let digest = hex::encode(Sha256::digest(&config));
    for (repo, text) in RECORDED {
        let recorded: serde_json::Value = serde_json::from_str(text).unwrap();
        assert_eq!(recorded["repository"], repo);
        if recorded["files"]["config.json"]["sha256"] == digest.as_str() {
            return (repo, recorded);
        }
    }
    panic!("config.json ({digest}) is not a recorded checkpoint's");
}

/// Refuse the directory's files unless each is the recorded upstream one.
/// Returns the checkpoint's repository.
pub fn verify(store: &WeightSet, dir: &Path) -> &'static str {
    let (repo, recorded) = recorded_for(dir);
    let mut manifest = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if !name.ends_with(".safetensors") && !SEMANTIC_FILES.contains(&name.as_str()) {
            continue;
        }
        let digest = recorded["files"][&name]["sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} is not a recorded file of {repo}"));
        manifest.insert(name, digest.to_owned());
    }
    store
        .verify_sha256(&manifest)
        .expect("every checkpoint file is the recorded upstream one");
    println!(
        "{} files of {repo} match their recorded hashes",
        manifest.len()
    );
    repo
}

/// Next-token accuracy a healthy model reaches over the first 1,024 tokens of
/// the checkpoint's own `LICENSE` (the Apache-2.0 text), scored after the
/// first token. The released model served by vLLM measured 0.998.
#[allow(dead_code)]
pub const LICENSE_FLOOR: f64 = 0.95;

/// Each row's most likely token (the lowest index among equals).
#[allow(dead_code)]
pub fn argmax_rows(logits: &[f32], vocab: usize) -> Vec<u32> {
    logits
        .chunks_exact(vocab)
        .map(|row| {
            let mut best = 0;
            for (i, v) in row.iter().enumerate() {
                if *v > row[best] {
                    best = i;
                }
            }
            best as u32
        })
        .collect()
}

/// The share of positions whose most likely next token is the one that
/// follows. `logits` is row-major `[tokens, vocab]`.
#[allow(dead_code)]
pub fn next_token_accuracy(logits: &[f32], tokens: &[u32], vocab: usize) -> f64 {
    let predicted = argmax_rows(logits, vocab);
    let hits = (0..tokens.len() - 1)
        .filter(|&i| predicted[i] == tokens[i + 1])
        .count();
    hits as f64 / (tokens.len() - 1) as f64
}
