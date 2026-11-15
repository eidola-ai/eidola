//! The model shapes the DeepSeek-V4.1 executor runs, checked before any device memory is
//! touched: [`check_supported`] returns the [`DsShape`] the executor splits the model
//! by, or the field that rules it out.
//!
//! The split and the groups are the CPU reference executor's
//! (`eidola-engine-cpu/AGENTS.md` → DeepSeek-V4.1 → Shape); the remaining conditions are
//! what the kernel ABI fixes (32-wide quantisation blocks, ratio-2 pooling, 32-bit table
//! rows).

use eidola_engine_model::deepseek::DsConfig;

use crate::support::Unsupported;

/// The model as the executor splits it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DsShape {
    /// The first decoder layer: a per-position global KV source.
    pub split: usize,
    pub layers: usize,
    /// The encoder's pooled KV-source layers, ascending.
    pub sources: Vec<usize>,
    /// Their pooling ratio.
    pub ratio: u32,
    /// Token ids before a position its Engram lookups read.
    pub history: usize,
    /// Per Engram layer of the config: the logical device holding its table.
    pub engram_device: Vec<u32>,
    /// Per Engram layer of the config: the model layer its module runs before, when the
    /// (possibly truncated) model keeps it.
    pub engram_layer: Vec<Option<usize>>,
}

impl DsShape {
    /// Chain layers: the decoder's.
    pub fn depth(&self) -> usize {
        self.layers - self.split
    }

    /// The encoder bank's layer of source layer `l`.
    pub fn source_index(&self, l: usize) -> Option<usize> {
        self.sources.iter().position(|s| *s == l)
    }
}

fn unsupported(field: impl Into<String>, found: impl ToString, required: &str) -> Unsupported {
    Unsupported {
        field: field.into(),
        found: found.to_string(),
        required: required.into(),
    }
}

fn require(ok: bool, field: &str, found: impl ToString, required: &str) -> Result<(), Unsupported> {
    if ok {
        Ok(())
    } else {
        Err(unsupported(field, found, required))
    }
}

/// The shape of `c` when the executor can run it with `draft_width` drafts, else the
/// first field that rules it out.
pub fn check_supported(c: &DsConfig, draft_width: u32) -> Result<DsShape, Unsupported> {
    require(
        draft_width == 0,
        "draft_width",
        draft_width,
        "0 (this executor does not draft)",
    )?;
    let a = &c.attention;
    let split = c
        .layers
        .iter()
        .position(|l| l.is_kv_source && l.compress_ratio == 1)
        .ok_or_else(|| {
            unsupported(
                "layers",
                "no per-position KV source",
                "a layer with compress ratio 1 that is a KV source, to split before",
            )
        })?;
    require(
        split > 0,
        "layers",
        "the first layer is the per-position KV source",
        "an encoder layer before the split",
    )?;
    let stub = &c.layers[split];
    require(
        stub.engram.is_none() && stub.is_index_source,
        &format!("layers[{split}]"),
        "an Engram layer, or not an index source",
        "the split layer runs its own indexer and no Engram module",
    )?;
    let sources: Vec<usize> = (0..split).filter(|l| c.layers[*l].is_kv_source).collect();
    require(
        !sources.is_empty(),
        "layers",
        "no KV source before the split",
        "a pooled KV source in the encoder",
    )?;
    let ratio = c.layers[sources[0]].compress_ratio;
    for (i, l) in c.layers.iter().enumerate() {
        let field = format!("layers[{i}]");
        if i < split {
            require(
                !l.is_kv_source || l.compress_ratio == 2,
                &format!("{field}.compress_ratio"),
                l.compress_ratio,
                "2 for every encoder KV source (the pooling kernel pools pairs)",
            )?;
            require(
                l.kv_source.is_none_or(|s| sources.contains(&s)),
                &format!("{field}.kv_source"),
                format!("{:?}", l.kv_source),
                "an encoder KV source",
            )?;
        } else {
            require(
                l.engram.is_none(),
                &format!("{field}.engram"),
                "an Engram module",
                "none from the split on",
            )?;
            require(
                l.kv_source.is_none_or(|s| s == split) && (i == split || !l.is_kv_source),
                &format!("{field}.kv_source"),
                format!("{:?}", l.kv_source),
                "the split layer's bank, and no other KV source from the split on",
            )?;
        }
        if l.uses_candidates || l.is_candidate_source {
            require(
                c.candidates.is_some(),
                "candidates",
                "none",
                "a candidate pool for layers that publish or use one",
            )?;
        }
    }
    // A layer that reuses a selection, or selects inside candidate blocks, must follow
    // the layer that published them over the same bank, on its own half: rows enter a
    // half at its first layer, so what a layer reads was written for every row it has.
    for half in [0..split, split..c.layers.len()] {
        let (mut selected, mut candidates) = (None, None);
        for i in half {
            let l = &c.layers[i];
            let field = format!("layers[{i}]");
            let Some(src) = l.kv_source else { continue };
            if l.is_index_source {
                if l.uses_candidates {
                    require(
                        candidates == Some(src),
                        &format!("{field}.uses_candidates"),
                        "no candidate blocks of its bank published before it",
                        "a candidate source over the same bank earlier in the half",
                    )?;
                }
                if l.is_candidate_source {
                    candidates = Some(src);
                }
                selected = Some(src);
            } else {
                require(
                    selected == Some(src),
                    &format!("{field}.index_source"),
                    "no selection over its bank published before it",
                    "an index source over the same bank earlier in the half",
                )?;
            }
        }
    }
    for (field, width) in [
        ("dim", c.dim),
        ("attention.head_dim", a.head_dim),
        ("attention.q_lora_rank", a.q_lora_rank),
        (
            "attention.o_groups * o_lora_rank",
            a.o_groups * a.o_lora_rank,
        ),
        ("indexer.head_dim", c.indexer.head_dim),
        ("moe.inter_dim", c.moe.inter_dim),
    ] {
        require(
            width > 0 && width.is_multiple_of(32),
            field,
            width,
            "a multiple of 32 (the quantisers' block)",
        )?;
    }
    require(
        (a.n_heads * a.head_dim).is_multiple_of(a.o_groups),
        "attention.o_groups",
        a.o_groups,
        "a divisor of the heads' total width",
    )?;
    require(
        a.rope_dim.is_multiple_of(2)
            && a.rope_dim <= a.head_dim
            && a.rope_dim <= c.indexer.head_dim,
        "attention.rope_dim",
        a.rope_dim,
        "even, and at most the head and index head widths",
    )?;
    require(
        c.hc_sinkhorn_iters >= 1 && c.hc_mult >= 1,
        "hc_sinkhorn_iters",
        c.hc_sinkhorn_iters,
        "at least one step and one copy",
    )?;
    require(
        c.moe.n_active >= 1 && c.moe.n_active <= c.moe.n_routed,
        "moe.n_active",
        c.moe.n_active,
        "between 1 and the routed experts",
    )?;
    // What the kernels hold in a block's shared memory or registers (the wrappers refuse
    // the same; refusing here names the field before a device is touched).
    for (field, found, most) in [
        ("hc_mult", c.hc_mult, super::ops::MAX_MIX_COPIES as usize),
        (
            "attention.head_dim",
            a.head_dim,
            super::attention::MAX_HEAD_DIM as usize,
        ),
        (
            "indexer.n_heads",
            c.indexer.n_heads,
            super::index::MAX_INDEX_HEADS as usize,
        ),
        (
            "indexer.head_dim",
            c.indexer.head_dim,
            super::index::MAX_INDEX_DIM as usize,
        ),
        (
            "moe.n_routed",
            c.moe.n_routed,
            super::ops::MAX_EXPERTS as usize,
        ),
        (
            "moe.n_active",
            c.moe.n_active,
            super::ops::MAX_ACTIVE as usize,
        ),
    ] {
        require(found <= most, field, found, &format!("at most {most}"))?;
    }
    let engram = c.engram.as_ref();
    if let Some(e) = engram {
        require(
            e.head_dim.is_multiple_of(32),
            "engram.head_dim",
            e.head_dim,
            "a multiple of 32",
        )?;
        for (slot, l) in e.layers.iter().enumerate() {
            require(
                u32::try_from(l.num_rows).is_ok(),
                &format!("engram.layers[{slot}].num_rows"),
                l.num_rows,
                "table rows within 32 bits",
            )?;
        }
    }
    let slots = engram.map_or(0, |e| e.layers.len());
    Ok(DsShape {
        split,
        layers: c.layers.len(),
        sources,
        ratio: u32::try_from(ratio).expect("checked: 2"),
        history: engram.map_or(0, |e| e.max_ngram - 1),
        // The first table beside the encoder; the others beside the decoder, whose
        // device has the room. Only gathered rows cross.
        engram_device: (0..slots).map(|slot| u32::from(slot > 0)).collect(),
        engram_layer: (0..slots)
            .map(|slot| c.layers.iter().position(|l| l.engram == Some(slot)))
            .collect(),
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    pub(crate) fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../eidola-engine-model/tests/fixtures/deepseek_tiny")
    }

    pub(crate) fn tiny_config() -> DsConfig {
        DsConfig::from_file(&fixture_dir().join("config.json")).unwrap()
    }

    pub(crate) fn released_config() -> DsConfig {
        DsConfig::from_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../eidola-engine-chat/tests/fixtures/deepseek_v4_1/config.json"),
        )
        .unwrap()
    }

    #[test]
    fn the_fixture_and_the_released_model_split_as_the_reference_executor_does() {
        let s = check_supported(&tiny_config(), 0).unwrap();
        assert_eq!((s.split, s.layers, s.ratio, s.history), (4, 8, 2, 3));
        assert_eq!(s.sources, vec![2]);
        assert_eq!(s.engram_device, vec![0, 1]);
        assert_eq!(s.engram_layer, vec![Some(1), Some(2)]);

        let s = check_supported(&released_config(), 0).unwrap();
        assert_eq!((s.split, s.layers, s.ratio, s.history), (20, 40, 2, 3));
        assert_eq!(s.sources, vec![2, 8, 14]);
        assert_eq!(s.engram_layer, vec![Some(1), Some(14)]);
    }

    #[test]
    fn each_condition_alone_is_refused() {
        let field = |c: &DsConfig, w: u32| check_supported(c, w).unwrap_err().field;
        let base = tiny_config();
        assert_eq!(field(&base, 1), "draft_width");

        let mut c = base.clone();
        c.layers[2].compress_ratio = 4;
        assert_eq!(field(&c, 0), "layers[2].compress_ratio");

        let mut c = base.clone();
        c.layers[5].engram = Some(1);
        assert_eq!(field(&c, 0), "layers[5].engram");

        let mut c = base.clone();
        c.layers[5].kv_source = Some(2);
        assert_eq!(field(&c, 0), "layers[5].kv_source");

        let mut c = base.clone();
        c.layers[4].is_index_source = false;
        assert_eq!(field(&c, 0), "layers[4]");

        let mut c = base.clone();
        c.layers[3].is_index_source = true;
        c.layers[3].uses_candidates = true;
        assert_eq!(field(&c, 0), "layers[3].uses_candidates");

        let mut c = base.clone();
        c.layers[2].is_index_source = false;
        assert_eq!(field(&c, 0), "layers[2].index_source");

        let mut c = base.clone();
        c.attention.head_dim = 48;
        assert_eq!(field(&c, 0), "attention.head_dim");

        let mut c = base.clone();
        c.indexer.n_heads = 33;
        assert_eq!(field(&c, 0), "indexer.n_heads");

        let mut c = base.clone();
        c.hc_mult = 7;
        assert_eq!(field(&c, 0), "hc_mult");

        let mut c = base.clone();
        c.attention.rope_dim = 7;
        assert_eq!(field(&c, 0), "attention.rope_dim");

        let mut c = base.clone();
        c.engram.as_mut().unwrap().layers[1].num_rows = 1 << 32;
        assert_eq!(field(&c, 0), "engram.layers[1].num_rows");

        let mut c = base;
        for l in &mut c.layers {
            l.is_kv_source = false;
        }
        assert_eq!(field(&c, 0), "layers");
    }
}
