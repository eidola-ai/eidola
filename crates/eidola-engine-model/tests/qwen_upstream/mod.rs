//! The released Qwen3.8-27B-FP8 checkpoint's recorded bytes, for the
//! real-weights tests: `golden/qwen/upstream-sha256.json` holds the SHA-256 of
//! every published file used here at the pinned revision. A test loads
//! nothing it has not checked against that list: a download of the right size
//! and the wrong bytes gives a model that agrees with a reference run on the
//! same bytes and is not the model.

use std::collections::BTreeMap;
use std::path::Path;

use eidola_engine_model::WeightSet;
use eidola_engine_model::safetensors::SEMANTIC_FILES;

const RECORDED: &str = include_str!("../../golden/qwen/upstream-sha256.json");

/// Next-token accuracy a healthy model reaches over the first 1,024 tokens of
/// the checkpoint's own `LICENSE` (the Apache-2.0 text), scored after the
/// first token. The released model served by vLLM measured 0.998.
#[allow(dead_code)]
pub const LICENSE_FLOOR: f64 = 0.95;

/// Refuse the directory's files unless each is the recorded upstream one.
pub fn verify(store: &WeightSet, dir: &Path) {
    let recorded: serde_json::Value = serde_json::from_str(RECORDED).unwrap();
    assert_eq!(recorded["repository"], "Qwen/Qwen3.8-27B-FP8");
    let mut manifest = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        if !name.ends_with(".safetensors") && !SEMANTIC_FILES.contains(&name.as_str()) {
            continue;
        }
        let digest = recorded["files"][&name]["sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("{name} is not a recorded file of the checkpoint"));
        manifest.insert(name, digest.to_owned());
    }
    store
        .verify_sha256(&manifest)
        .expect("every checkpoint file is the recorded upstream one");
    println!(
        "{} checkpoint files match their recorded hashes",
        manifest.len()
    );
}

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
