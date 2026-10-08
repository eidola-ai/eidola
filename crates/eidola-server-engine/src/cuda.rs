//! The CUDA executor's part of boot (`cuda` feature): every refusal that needs no device,
//! the KV block counts derived from the configured device memory, opening the device, and
//! the executor's construction on the engine thread.
//!
//! The executor supports exactly MiMo-V2.6-Flash's configuration on one GPU
//! (`eidola-engine-cuda`'s `check_supported` and `check_device`), and the node never
//! narrows or widens that: it hands the executor the configuration and returns the
//! executor's own refusal. Nothing here changes a number the model computes; it only
//! sizes and constructs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine::spec::{AttentionKind, Bucket, KvGroupSpec, KvRole, ModelSpec};
use eidola_engine_cuda::kv::{GroupGeometry, KvLayout};
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, Gpu, KernelDir, KvBlocks, MtpHidden,
};
use eidola_engine_model::safetensors::WeightSet;

use crate::BootError;
use crate::config::{CacheConfig, Sizing, env};
use crate::http::DeviceReport;
use crate::model::LoadedModel;

/// The device the node runs on. A node serves one model on one GPU; which physical GPU
/// that is is the deployment's choice (`CUDA_VISIBLE_DEVICES`, or the one device a
/// confidential VM is given).
pub const DEVICE_ORDINAL: usize = 0;

/// Retention points a keyed sequence may hold per sliding group when the prefix cache is
/// enabled (branch, regeneration, continuation: `eidola-engine`'s KV design).
const RETENTION_POINTS: u64 = 3;

/// Derives the executor's per-group KV block counts from `kv_device_bytes`.
///
/// `spec` is the executor's own [`ModelSpec`] (from [`CudaExecutor::preflight`]); only its
/// group shapes are read, and `hidden` (the model's hidden size, the width of a drafter
/// block's boundary tap). With `B` the block size and, for a sliding group of window `W`,
/// `R = ceil((W - 1) / B) + 1` (the core's bound on one window's blocks):
///
/// * each **sliding** group gets `1 + max_seqs × ((1 + P) × R + 1) + ceil(max_batched_tokens / B)`
///   blocks: the null block; per seated sequence its live window, the partly filled block
///   after it, and `P` retained windows (`P` = 3 with the prefix cache, else 0); and the
///   positions one step adds. So sliding blocks never run out while the global pool still
///   has room, and the global pool is the one that bounds concurrency and context. The
///   **drafter** group (MTP depths' KV, when drafting) is a sliding window too, and gets the
///   same count by the same rule;
/// * each **global** group gets what is left, `floor((kv_device_bytes − sliding bytes) /
///   global block bytes) − 1`, and must hold at least one `MAX_MODEL_LEN` sequence plus the
///   null block.
///
/// Every pool also holds one pad block past the counts given here (the executor's
/// `GroupGeometry::pad_block`, where padding rows of a replayed decode graph write), so
/// the bytes charged to the budget are one block more per group than the count. Block
/// bytes are the executor's own (`GroupGeometry::block_bytes`). The rule reads only
/// measured values, never the device, so the same configuration derives the same counts
/// on every node. Its least accepted budget is
/// `eidola_common::engine_deployment::cuda_kv_min_bytes`, which the gateway's build
/// holds every pinned deployment to; the two are kept equal by a test here.
pub fn derive_kv_blocks(
    spec: &ModelSpec,
    hidden: u32,
    kv_device_bytes: u64,
    sizing: &Sizing,
    cache: &CacheConfig,
) -> Result<KvBlocks, BootError> {
    let b = u64::from(sizing.kv_block_size);
    let overflow = || BootError(format!("{} overflows the KV sizing", env::KV_DEVICE_BYTES));
    let block_bytes = |g: &KvGroupSpec| -> Result<u64, BootError> {
        let geometry = GroupGeometry {
            num_layers: g.num_layers,
            num_kv_heads: g.num_kv_heads,
            head_dim_qk: g.head_dim_qk,
            head_dim_v: g.head_dim_v,
            block_size: sizing.kv_block_size,
            num_blocks: 2,
            // A drafter block holds a tap of every depth's level beside its KV.
            layout: match g.role {
                KvRole::Target => KvLayout::Blocked,
                KvRole::Drafter => KvLayout::Planar {
                    taps: g.num_layers,
                    hidden,
                },
            },
        };
        geometry
            .validate()
            .map_err(|e| BootError(format!("the engine refused its configuration: {e}")))?;
        u64::try_from(geometry.block_bytes()).map_err(|_| overflow())
    };
    let points = if cache.enabled { RETENTION_POINTS } else { 0 };

    let (mut global, mut sliding, mut drafter) = (None, None, None);
    let mut sliding_bytes = 0u64;
    for g in &spec.kv_groups {
        match g.attention {
            AttentionKind::Sliding { window } => {
                let window_blocks = u64::from(window.saturating_sub(1)).div_ceil(b) + 1;
                let per_seq = (1 + points)
                    .checked_mul(window_blocks)
                    .and_then(|x| x.checked_add(1))
                    .ok_or_else(overflow)?;
                let blocks = u64::from(sizing.max_seqs)
                    .checked_mul(per_seq)
                    .and_then(|x| {
                        x.checked_add(1 + u64::from(sizing.max_batched_tokens).div_ceil(b))
                    })
                    .ok_or_else(overflow)?;
                // The count, plus the pool's pad block.
                sliding_bytes = (blocks + 1)
                    .checked_mul(block_bytes(g)?)
                    .and_then(|x| x.checked_add(sliding_bytes))
                    .ok_or_else(overflow)?;
                let slot = match g.role {
                    KvRole::Target => &mut sliding,
                    KvRole::Drafter => &mut drafter,
                };
                if slot.replace(blocks).is_some() {
                    return Err(BootError(
                        "the executor reports two sliding KV groups of one role".into(),
                    ));
                }
            }
            AttentionKind::Full => {
                if global.replace(block_bytes(g)?).is_some() {
                    return Err(BootError(
                        "the executor reports two global KV groups".into(),
                    ));
                }
            }
        }
    }
    let narrow = |n: u64| u32::try_from(n).map_err(|_| overflow());
    let sliding = sliding.map(narrow).transpose()?.unwrap_or(0);
    let drafter = drafter.map(narrow).transpose()?.unwrap_or(0);
    let global = match global {
        None => 0,
        Some(bytes) => {
            let rest = kv_device_bytes.checked_sub(sliding_bytes).ok_or_else(|| {
                BootError(format!(
                    "{} ({kv_device_bytes}) cannot hold the sliding-window KV \
                     ({sliding_bytes} bytes for {} sequences)",
                    env::KV_DEVICE_BYTES,
                    sizing.max_seqs
                ))
            })?;
            // What fits, less the pool's pad block.
            let blocks = (rest / bytes).saturating_sub(1);
            let needed = u64::from(sizing.max_model_len).div_ceil(b) + 1;
            if blocks < needed {
                return Err(BootError(format!(
                    "{} ({kv_device_bytes}) leaves {blocks} global KV blocks, fewer than the \
                     {needed} one {} sequence needs",
                    env::KV_DEVICE_BYTES,
                    env::MAX_MODEL_LEN
                )));
            }
            // Past `i32::MAX` the executor's tables cannot index a block; more memory than
            // that is not usable, so the count stops there.
            narrow(blocks.min(i32::MAX as u64))?
        }
    };
    Ok(KvBlocks {
        global,
        sliding,
        drafter,
    })
}

/// Everything checked and the device open: what the engine thread builds the executor
/// from.
pub struct Prepared {
    gpu: Gpu,
    kernels_dir: PathBuf,
    store: Arc<WeightSet>,
    config: CudaExecutorConfig,
    report: DeviceReport,
}

impl Prepared {
    /// The device, for `/v1/engine/info`.
    pub fn report(&self) -> DeviceReport {
        self.report.clone()
    }

    /// The executor's configuration, as derived.
    pub fn config(&self) -> &CudaExecutorConfig {
        &self.config
    }

    /// Builds the executor: the device checks, the kernel images (each checked against the
    /// compiled-in manifest), then the weights and every pool. Runs on the engine thread.
    pub fn build(self) -> Result<CudaExecutor, String> {
        CudaExecutor::new(
            self.gpu,
            &KernelDir::new(self.kernels_dir),
            self.store,
            None,
            self.config,
        )
        .map_err(|e| e.to_string())
    }
}

/// The refusals that need no device, in order (the executor's own preflight of the model,
/// its MTP layers for the draft width, and the step shape; the KV derivation and the
/// derived geometry), then the device. Nothing is allocated on the device here.
pub fn prepare(
    model: &LoadedModel,
    sizing: &Sizing,
    cache: &CacheConfig,
    kernels_dir: &Path,
    kv_device_bytes: u64,
    graphs: CudaGraphs,
) -> Result<Prepared, BootError> {
    let refused = |e: eidola_engine_cuda::CudaError| {
        BootError(format!("the engine refused its configuration: {e}"))
    };
    let drafting = sizing.draft_tokens > 0;
    // One step shape, as on the CPU executor: the scheduler's limits.
    let mut config = CudaExecutorConfig {
        block_size: sizing.kv_block_size,
        num_blocks: KvBlocks {
            global: 2,
            sliding: 2,
            drafter: if drafting { 2 } else { 0 },
        },
        num_state_slots: sizing.max_seqs,
        max_model_len: sizing.max_model_len,
        buckets: vec![Bucket {
            max_seqs: sizing.max_seqs,
            max_tokens: sizing.max_batched_tokens,
        }],
        sampleable_vocab_size: u32::try_from(model.tokenizer().vocab_size())
            .map_err(|_| BootError("the tokenizer's vocabulary does not fit u32".into()))?,
        image: None,
        graphs,
        draft_tokens: sizing.draft_tokens,
        mtp_hidden: MtpHidden::Normed,
    };
    let spec = CudaExecutor::preflight(model.store(), None, &config).map_err(refused)?;
    let hidden = u32::try_from(model.config().hidden_size)
        .map_err(|_| BootError("the model's hidden size does not fit u32".into()))?;
    config.num_blocks = derive_kv_blocks(&spec, hidden, kv_device_bytes, sizing, cache)?;
    CudaExecutor::preflight(model.store(), None, &config).map_err(refused)?;

    let gpu = Gpu::open(DEVICE_ORDINAL).map_err(refused)?;
    let info = gpu.info();
    let total = info.total_mem_bytes as u64;
    if kv_device_bytes >= total {
        return Err(BootError(format!(
            "{} ({kv_device_bytes}) is not less than the device's memory ({total} bytes)",
            env::KV_DEVICE_BYTES
        )));
    }
    let (major, minor) = info.compute_capability;
    let report = DeviceReport {
        name: info.name.clone(),
        compute_capability: format!("{major}.{minor}"),
        image: gpu.image_arch().map(|a| a.as_str()),
        sm_count: info.sm_count,
        memory_bytes: total,
    };
    Ok(Prepared {
        gpu,
        kernels_dir: kernels_dir.to_path_buf(),
        store: model.store().clone(),
        config,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_engine::spec::KvRole;

    fn group(attention: AttentionKind, layers: u32, heads: u32) -> KvGroupSpec {
        KvGroupSpec {
            name: "g".into(),
            role: KvRole::Target,
            attention,
            num_layers: layers,
            num_kv_heads: heads,
            head_dim_qk: 192,
            head_dim_v: 128,
            num_blocks: 2,
        }
    }

    /// Flash's groups with `k` draft depths: a drafter group of `k` MTP layers × 8
    /// heads after the target groups.
    fn flash_drafting(k: u32) -> ModelSpec {
        let mut spec = flash();
        if k > 0 {
            spec.kv_groups.push(KvGroupSpec {
                role: KvRole::Drafter,
                ..group(AttentionKind::Sliding { window: 128 }, k, 8)
            });
            spec.max_draft_tokens = k;
        }
        spec
    }

    /// Flash's two groups: 9 global layers × 4 KV heads, 39 sliding layers × 8 heads.
    fn flash() -> ModelSpec {
        ModelSpec {
            vocab_size: 152_576,
            sampleable_vocab_size: 151_675,
            block_size: 16,
            max_model_len: 4096,
            kv_groups: vec![
                group(AttentionKind::Full, 9, 4),
                group(AttentionKind::Sliding { window: 128 }, 39, 8),
            ],
            max_draft_tokens: 0,
            draft_step_tokens: 0,
            num_state_slots: 8,
            buckets: vec![Bucket {
                max_seqs: 8,
                max_tokens: 512,
            }],
        }
    }

    fn sizing() -> Sizing {
        Sizing {
            kv_block_size: 16,
            max_model_len: 4096,
            max_seqs: 8,
            max_batched_tokens: 512,
            max_prefill_chunk: 512,
            draft_tokens: 0,
            max_requests: 16,
        }
    }

    fn cache(enabled: bool) -> CacheConfig {
        CacheConfig {
            enabled,
            idle_ttl_secs: 900,
            max_age_secs: 7200,
        }
    }

    #[test]
    fn kv_blocks_follow_the_documented_rule() {
        // Block bytes: 16 positions × layers × heads × (192 + 128) × 2 bytes.
        let global_bytes = 16 * 9 * 4 * 320 * 2u64;
        let sliding_bytes = 16 * 39 * 8 * 320 * 2u64;
        // R = ceil(127 / 16) + 1 = 9; per sequence (1 + 3) × 9 + 1 = 37 with the cache.
        let sliding = 1 + 8 * 37 + 512 / 16;
        let budget = 40 << 30;
        let blocks = derive_kv_blocks(&flash(), 4096, budget, &sizing(), &cache(true)).unwrap();
        assert_eq!(blocks.sliding, sliding as u32);
        // Each pool holds one pad block past its count.
        assert_eq!(
            u64::from(blocks.global),
            (budget - (sliding + 1) * sliding_bytes) / global_bytes - 1
        );
        // Every derived byte, pad blocks included, fits the budget.
        assert!(
            (u64::from(blocks.global) + 1) * global_bytes
                + (u64::from(blocks.sliding) + 1) * sliding_bytes
                <= budget
        );

        // No retention without the cache: (1 + 0) × 9 + 1 = 10 per sequence.
        let blocks = derive_kv_blocks(&flash(), 4096, budget, &sizing(), &cache(false)).unwrap();
        assert_eq!(blocks.sliding, 1 + 8 * 10 + 32);
        assert_eq!(blocks.drafter, 0);

        // Drafting three tokens: the drafter group gets the sliding count, each block 16
        // positions × 3 depths × 8 heads × (192 + 128) × 2 bytes and a tap of 3 × 4,096
        // f32; the global pool what is left.
        let drafter_bytes = 16 * 3 * 8 * 320 * 2 + 3 * 4096 * 4u64;
        let blocks =
            derive_kv_blocks(&flash_drafting(3), 4096, budget, &sizing(), &cache(true)).unwrap();
        assert_eq!(
            (blocks.sliding, blocks.drafter),
            (sliding as u32, sliding as u32)
        );
        assert_eq!(
            u64::from(blocks.global),
            (budget - (sliding + 1) * (sliding_bytes + drafter_bytes)) / global_bytes - 1
        );
    }

    #[test]
    fn a_budget_that_cannot_hold_one_sequence_is_refused() {
        // The sliding count plus its pad block.
        let sliding_bytes = (1 + 8 * 37 + 32 + 1) * 16 * 39 * 8 * 320 * 2u64;
        let global_bytes = 16 * 9 * 4 * 320 * 2u64;
        // One 4096-token sequence needs 256 blocks plus the null block, and the pool
        // its pad block.
        let enough = sliding_bytes + 258 * global_bytes;
        let blocks = derive_kv_blocks(&flash(), 4096, enough, &sizing(), &cache(true)).unwrap();
        assert_eq!(blocks.global, 257);
        let e = derive_kv_blocks(&flash(), 4096, enough - 1, &sizing(), &cache(true)).unwrap_err();
        assert!(e.0.contains(env::KV_DEVICE_BYTES), "{e}");
        assert!(e.0.contains(env::MAX_MODEL_LEN), "{e}");
        // Not even the sliding windows fit.
        let e = derive_kv_blocks(&flash(), 4096, sliding_bytes - 1, &sizing(), &cache(true))
            .unwrap_err();
        assert!(e.0.contains("sliding"), "{e}");
    }

    /// The gateway's build holds a pinned deployment's budget to
    /// `cuda_kv_min_bytes`; it is exactly the least this derivation accepts, across
    /// block sizes, seats, step sizes, the cache switch and the draft width.
    #[test]
    fn the_shared_minimum_is_the_derivations() {
        use eidola_common::engine_deployment::cuda_kv_min_bytes;
        for block in [1, 16, 64, 1024] {
            for (seats, step) in [(1, 4), (8, 512), (64, 8192)] {
                for (enabled, draft) in [(false, 0), (true, 0), (false, 1), (true, 3)] {
                    let sizing = Sizing {
                        kv_block_size: block,
                        max_seqs: seats,
                        max_batched_tokens: step,
                        max_prefill_chunk: step,
                        draft_tokens: draft,
                        ..sizing()
                    };
                    let mut spec = flash_drafting(draft);
                    spec.block_size = block;
                    let min = cuda_kv_min_bytes(&sizing, &cache(enabled));
                    let at = (block, seats, step, enabled, draft);
                    derive_kv_blocks(&spec, 4096, min, &sizing, &cache(enabled))
                        .unwrap_or_else(|e| panic!("{at:?}: {e}"));
                    assert!(
                        derive_kv_blocks(&spec, 4096, min - 1, &sizing, &cache(enabled)).is_err(),
                        "{at:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn global_blocks_stop_at_the_tables_index_range() {
        let blocks = derive_kv_blocks(&flash(), 4096, u64::MAX, &sizing(), &cache(true)).unwrap();
        assert_eq!(blocks.global, i32::MAX as u32);
    }
}
