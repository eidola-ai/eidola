//! The released DeepSeek-V4.1-Flash-1017 checkpoint's recorded bytes, for the
//! real-weights tests: `golden/deepseek/upstream-sha256.json` holds the
//! SHA-256 of every published file at the pinned revision and of every
//! re-packed file of each truncation. A test loads nothing it has not checked
//! against that list: a download of the right size and the wrong bytes gives
//! a model that agrees with a reference run on the same bytes and is not the
//! model.

use std::collections::BTreeMap;
use std::path::Path;

use eidola_engine_model::WeightSet;
use eidola_engine_model::safetensors::SEMANTIC_FILES;

const RECORDED: &str = include_str!("../../golden/deepseek/upstream-sha256.json");

/// Next-token accuracy a healthy model reaches over the passage the
/// whole-model goldens are made on, scored after its first token. The
/// reference on the verified checkpoint measures 0.670; public GPT-2 small
/// and large measure 0.380 and 0.431 on the same text.
#[allow(dead_code)]
pub const PASSAGE_FLOOR: f64 = 0.55;
/// The same over the story of the health check as a chat reply (measured
/// 0.665).
#[allow(dead_code)]
pub const STORY_FLOOR: f64 = 0.55;

/// Refuse the directory's files unless each is the recorded one: the upstream
/// file, or with `truncation` that truncation's re-pack of the shard.
pub fn verify(store: &WeightSet, dir: &Path, truncation: Option<&str>) {
    let recorded: serde_json::Value = serde_json::from_str(RECORDED).unwrap();
    let mut manifest = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        let shard = name.ends_with(".safetensors");
        if !shard && !SEMANTIC_FILES.contains(&name.as_str()) {
            continue;
        }
        let digest = match truncation {
            Some(t) if shard => recorded["truncations"][t][&name].as_str(),
            _ => recorded["files"][&name]["sha256"].as_str(),
        };
        let digest = digest.unwrap_or_else(|| {
            panic!("{name} is not a recorded file of the checkpoint (truncation: {truncation:?})")
        });
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

/// The share of positions from `from` on whose most likely next token is the
/// one that follows. `logits` is row-major `[tokens, vocab]`.
#[allow(dead_code)]
pub fn next_token_accuracy(logits: &[f32], tokens: &[u32], vocab: usize, from: usize) -> f64 {
    let predicted = argmax_rows(logits, vocab);
    let scored = from..tokens.len() - 1;
    let hits = scored
        .clone()
        .filter(|&i| predicted[i] == tokens[i + 1])
        .count();
    hits as f64 / scored.len() as f64
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
