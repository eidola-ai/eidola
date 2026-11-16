//! The CUDA executors' part of boot (`cuda` feature): every refusal that needs no device,
//! the KV block counts derived from the configured device memory, opening the devices,
//! and the executor's construction on the engine thread.
//!
//! [`prepare`] is MiMo-V2.6-Flash's executor on one GPU (`eidola-engine-cuda`'s
//! `check_supported` and `check_device`); [`prepare_deepseek`] is DeepSeek-V4.1's on two,
//! device `d` on GPU `d` (`eidola_engine_cuda::ds`). The node never narrows or widens
//! what an executor supports: it hands the executor the configuration and returns the
//! executor's own refusal. Nothing here changes a number the model computes; it only
//! sizes and constructs.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use eidola_engine::spec::{AttentionKind, Bucket, KvGroupSpec, KvRole, ModelSpec};
use eidola_engine_cuda::ds::weights::Half;
use eidola_engine_cuda::ds::{DsCudaExecutor, DsCudaExecutorConfig};
use eidola_engine_cuda::kv::{GroupGeometry, KvLayout};
use eidola_engine_cuda::{
    CudaExecutor, CudaExecutorConfig, CudaGraphs, Gpu, KernelDir, KvBlocks, MtpHidden,
};
use eidola_engine_model::deepseek::DsConfig;
use eidola_engine_model::safetensors::WeightSet;

use crate::BootError;
use crate::config::{CacheConfig, GpuCc, Sizing, env};
use crate::http::DeviceReport;
use crate::model::{LoadedModel, Served};

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
/// The executor's bytes per block of group `g` (`GroupGeometry::block_bytes`: BF16 K and
/// V of every layer, and a drafter block's boundary tap of `hidden` f32 per depth).
fn group_block_bytes(g: &KvGroupSpec, sizing: &Sizing, hidden: u32) -> Result<u64, BootError> {
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
    u64::try_from(geometry.block_bytes())
        .map_err(|_| BootError(format!("{} overflows the KV sizing", env::KV_DEVICE_BYTES)))
}

/// Derives the host tier's per-group block counts from `kv_host_bytes`: the device
/// pools' mix (`device`, as [`derive_kv_blocks`] gave it) scaled to the budget, so each
/// group `g` with `n_g` device blocks of `b_g` bytes gets `floor(n_g × H / Σ n_h b_h)`
/// host blocks (`H` = `kv_host_bytes`), the null block among them. The blocks' bytes then
/// sum to at most `H`, which the gateway's build charges to the VM's memory
/// (`engine_deployment::check_resources`). The host tier takes what the device pools hold
/// when they run short, in the same mix. 0 gives no host tier; a budget too small for two
/// blocks in every group is refused. The least accepted budget is
/// `eidola_common::engine_deployment::cuda_kv_host_min_bytes`, which the gateway's build
/// holds every pinned deployment to; the two are kept equal by a test here.
pub fn derive_host_blocks(
    spec: &ModelSpec,
    hidden: u32,
    kv_host_bytes: u64,
    sizing: &Sizing,
    device: &KvBlocks,
) -> Result<KvBlocks, BootError> {
    if kv_host_bytes == 0 {
        return Ok(KvBlocks::default());
    }
    let count = |g: &KvGroupSpec| -> u64 {
        u64::from(match (g.attention, g.role) {
            (AttentionKind::Full, _) => device.global,
            (AttentionKind::Sliding { .. }, KvRole::Target) => device.sliding,
            (AttentionKind::Sliding { .. }, KvRole::Drafter) => device.drafter,
        })
    };
    let mut device_bytes = 0u128;
    for g in &spec.kv_groups {
        device_bytes += u128::from(count(g)) * u128::from(group_block_bytes(g, sizing, hidden)?);
    }
    let mut host = KvBlocks::default();
    for g in &spec.kv_groups {
        let n = u128::from(count(g)) * u128::from(kv_host_bytes) / device_bytes.max(1);
        if n < 2 {
            return Err(BootError(format!(
                "{} ({kv_host_bytes}) gives the {} KV group fewer than two host blocks",
                env::KV_HOST_BYTES,
                g.name
            )));
        }
        // Past `i32::MAX` no block id fits the executor's tables; the count stops there.
        let n = u32::try_from(n.min(i32::MAX as u128)).expect("capped");
        match (g.attention, g.role) {
            (AttentionKind::Full, _) => host.global = n,
            (AttentionKind::Sliding { .. }, KvRole::Target) => host.sliding = n,
            (AttentionKind::Sliding { .. }, KvRole::Drafter) => host.drafter = n,
        }
    }
    Ok(host)
}

pub fn derive_kv_blocks(
    spec: &ModelSpec,
    hidden: u32,
    kv_device_bytes: u64,
    sizing: &Sizing,
    cache: &CacheConfig,
) -> Result<KvBlocks, BootError> {
    // The memory bounds the deployment was checked against count this many
    // groups' block tables, host and device (`engine_deployment::kv_groups`).
    let groups = eidola_common::engine_deployment::kv_groups(sizing.draft_tokens);
    if spec.kv_groups.len() as u64 != groups {
        return Err(BootError(format!(
            "the executor reports {} KV groups; {} {} accounts for {groups}",
            spec.kv_groups.len(),
            env::DRAFT_TOKENS,
            sizing.draft_tokens
        )));
    }
    let b = u64::from(sizing.kv_block_size);
    let overflow = || BootError(format!("{} overflows the KV sizing", env::KV_DEVICE_BYTES));
    let block_bytes = |g: &KvGroupSpec| group_block_bytes(g, sizing, hidden);
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
/// derived geometry), then, with `gpu_cc` required, the GPUs' confidential-computing state
/// (`crate::gpu_cc`), then the device. Nothing is allocated on the device here.
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    model: &LoadedModel,
    sizing: &Sizing,
    cache: &CacheConfig,
    kernels_dir: &Path,
    kv_device_bytes: u64,
    kv_host_bytes: u64,
    graphs: CudaGraphs,
    gpu_cc: GpuCc,
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
        host_blocks: KvBlocks::default(),
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
    config.host_blocks =
        derive_host_blocks(&spec, hidden, kv_host_bytes, sizing, &config.num_blocks)?;
    CudaExecutor::preflight(model.store(), None, &config).map_err(refused)?;

    // No device is opened, and nothing reaches one, until the driver reports the GPUs in
    // confidential-computing mode and accepting work.
    match gpu_cc {
        GpuCc::Required => crate::gpu_cc::query()
            .and_then(|state| crate::gpu_cc::check(&state))
            .map_err(|e| {
                BootError(format!(
                    "{} is required, but the GPUs are not in production confidential-computing \
                     mode: {e}",
                    env::GPU_CC
                ))
            })?,
        GpuCc::NotRequired => tracing::warn!(
            "{} is not-required: the GPUs' confidential-computing mode is not checked \
             (development only; no gateway pins this configuration)",
            env::GPU_CC
        ),
    }

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

/// The GPU of each of the DeepSeek-V4.1 executor's devices (encoder, decoder): device `d`
/// on GPU `d`. Which physical GPUs those are is the deployment's choice, as for
/// [`DEVICE_ORDINAL`].
pub const DEEPSEEK_DEVICES: [usize; 2] = [0, 1];

/// Bytes of one block of every KV group on `half`'s device together: the executor's own
/// block layouts (`eidola_engine_cuda::ds::executor::geometry`, each group's block one
/// image padded to 16 bytes).
fn deepseek_half_block_bytes(
    ds: &DsConfig,
    spec: &ModelSpec,
    half: Half,
) -> Result<u64, BootError> {
    let groups = eidola_engine_cuda::ds::executor::geometry(ds, spec, half)
        .map_err(|e| BootError(format!("the engine refused its configuration: {e}")))?;
    let mut bytes = 0u64;
    for g in groups {
        bytes = u64::try_from(g.block_bytes())
            .ok()
            .and_then(|b| bytes.checked_add(b))
            .ok_or_else(|| {
                BootError(format!("{} overflows the KV sizing", env::KV_DEVICE_BYTES))
            })?;
    }
    Ok(bytes)
}

/// The DeepSeek-V4.1 executor's block count from the two device budgets (`budgets`, device
/// order). The executor gives every KV group the same count (the null block among them),
/// as the CPU reference executor does with `KV_BLOCKS`, so the count is what the tighter
/// budget holds: `min_d floor(budgets[d] / S_d)`, `S_d` the bytes of one block of every
/// group on device `d` together. It must hold one `MAX_MODEL_LEN` sequence and the null
/// block (`ceil(MAX_MODEL_LEN / B) + 1`), and stops at the tables' `i32` range. The
/// executor captures no graph, so no pool holds a pad block. Measured values only, never
/// the device.
pub fn derive_deepseek_kv_blocks(
    ds: &DsConfig,
    spec: &ModelSpec,
    budgets: [u64; 2],
    sizing: &Sizing,
) -> Result<u32, BootError> {
    let needed = u64::from(sizing.max_model_len).div_ceil(u64::from(sizing.kv_block_size)) + 1;
    let mut blocks = i32::MAX as u64;
    for (d, (half, budget)) in [Half::Encoder, Half::Decoder]
        .into_iter()
        .zip(budgets)
        .enumerate()
    {
        let per_block = deepseek_half_block_bytes(ds, spec, half)?.max(1);
        let fits = budget / per_block;
        if fits < needed {
            return Err(BootError(format!(
                "{} ({budget}, device {d}) holds {fits} KV blocks of {per_block} bytes, fewer \
                 than the {needed} one {} sequence needs",
                env::KV_DEVICE_BYTES,
                env::MAX_MODEL_LEN
            )));
        }
        blocks = blocks.min(fits);
    }
    Ok(u32::try_from(blocks).expect("capped at i32::MAX"))
}

/// The DeepSeek-V4.1 executor's host-tier block count from `kv_host_bytes`: one count for
/// every group of both devices (each group's tier holds that many block images), so it is
/// `floor(kv_host_bytes / (S_0 + S_1))` (`S_d` as in [`derive_deepseek_kv_blocks`]) and the
/// tier's bytes never exceed the budget. 0 is no tier; a budget under two blocks (the null
/// block and one more) is refused.
pub fn derive_deepseek_host_blocks(
    ds: &DsConfig,
    spec: &ModelSpec,
    kv_host_bytes: u64,
) -> Result<u32, BootError> {
    if kv_host_bytes == 0 {
        return Ok(0);
    }
    let per_block = deepseek_half_block_bytes(ds, spec, Half::Encoder)?
        .saturating_add(deepseek_half_block_bytes(ds, spec, Half::Decoder)?)
        .max(1);
    let blocks = kv_host_bytes / per_block;
    if blocks < 2 {
        return Err(BootError(format!(
            "{} ({kv_host_bytes}) holds fewer than two host blocks of {per_block} bytes (every \
             KV group of both devices)",
            env::KV_HOST_BYTES
        )));
    }
    Ok(u32::try_from(blocks.min(i32::MAX as u64)).expect("capped"))
}

/// The DeepSeek-V4.1 CUDA executor's configuration from the node's measured one, with no
/// device: one step shape (`MAX_SEQS` × `MAX_BATCHED_TOKENS`), as on the CPU executor; the
/// block count from the two device budgets ([`derive_deepseek_kv_blocks`]) and the host
/// tier's from `kv_host_bytes` ([`derive_deepseek_host_blocks`]); `DRAFT_TOKENS` drafts;
/// devices [`DEEPSEEK_DEVICES`]. The executor's own preflight runs before the derivation
/// (the model, the draft width against the drafter's slots, the block size against the
/// pooling ratio, the step shape) and again over the derived counts. `graphs` must be
/// `off`: this executor captures no decode graphs.
pub fn deepseek_config(
    ds: &DsConfig,
    sizing: &Sizing,
    kv_device_bytes: [u64; 2],
    kv_host_bytes: u64,
    graphs: CudaGraphs,
    sampleable_vocab_size: u32,
) -> Result<DsCudaExecutorConfig, BootError> {
    if graphs == CudaGraphs::On {
        return Err(BootError(format!(
            "{} must be off: the DeepSeek-V4.1 cuda executor captures no decode graphs",
            env::CUDA_GRAPHS
        )));
    }
    let refused = |e: eidola_engine_cuda::CudaError| {
        BootError(format!("the engine refused its configuration: {e}"))
    };
    let mut config = DsCudaExecutorConfig {
        block_size: sizing.kv_block_size,
        num_blocks: 2,
        num_state_slots: sizing.max_seqs,
        max_model_len: sizing.max_model_len,
        buckets: vec![Bucket {
            max_seqs: sizing.max_seqs,
            max_tokens: sizing.max_batched_tokens,
        }],
        sampleable_vocab_size,
        devices: DEEPSEEK_DEVICES,
        draft_width: sizing.draft_tokens,
        host_blocks: 0,
        position_tags: false,
        record: false,
        gemm: eidola_engine_cuda::ds::gemm::GemmForm::Exact,
    };
    let (_, spec) = DsCudaExecutor::preflight(ds, &config).map_err(refused)?;
    config.num_blocks = derive_deepseek_kv_blocks(ds, &spec, kv_device_bytes, sizing)?;
    config.host_blocks = derive_deepseek_host_blocks(ds, &spec, kv_host_bytes)?;
    DsCudaExecutor::preflight(ds, &config).map_err(refused)?;
    Ok(config)
}

/// Everything checked and both devices open: what the engine thread builds the
/// DeepSeek-V4.1 executor from.
pub struct DsPrepared {
    /// The two GPUs, held open (their primary contexts retained) until the executor has
    /// opened its own streams on them.
    gpus: [Gpu; 2],
    kernels_dir: PathBuf,
    store: Arc<WeightSet>,
    ds: DsConfig,
    token_map: Option<Vec<u32>>,
    config: DsCudaExecutorConfig,
    report: DeviceReport,
}

impl DsPrepared {
    /// Device 0's, for `/v1/engine/info`.
    pub fn report(&self) -> DeviceReport {
        self.report.clone()
    }

    /// The executor's configuration, as derived.
    pub fn config(&self) -> &DsCudaExecutorConfig {
        &self.config
    }

    /// Builds the executor: on each device the kernel images (each checked against the
    /// compiled-in manifest), then the weights and every pool. Runs on the engine thread.
    pub fn build(self) -> Result<DsCudaExecutor, String> {
        let exec = DsCudaExecutor::new(
            &KernelDir::new(self.kernels_dir),
            &self.store,
            &self.ds,
            self.token_map,
            &self.config,
        )
        .map_err(|e| e.to_string());
        drop(self.gpus);
        exec
    }
}

/// DeepSeek-V4.1's refusals that need no device, in order ([`deepseek_config`]: graphs
/// off, the executor's preflight, the block counts from both budgets and the host tier's),
/// then, with `gpu_cc` required, the GPUs' confidential-computing state and their count
/// (`crate::gpu_cc::check_devices`: at least as many GPUs reported as devices), then each
/// device `d` on GPU `d`, whose budget must be less than its memory. Nothing is allocated
/// on a device here.
pub fn prepare_deepseek(
    model: &LoadedModel,
    sizing: &Sizing,
    kernels_dir: &Path,
    kv_device_bytes: [u64; 2],
    kv_host_bytes: u64,
    graphs: CudaGraphs,
    gpu_cc: GpuCc,
) -> Result<DsPrepared, BootError> {
    let Served::DeepSeek {
        config: ds,
        token_map,
        ..
    } = model.served()
    else {
        return Err(BootError("the loaded model is not DeepSeek-V4.1".into()));
    };
    let config = deepseek_config(
        ds,
        sizing,
        kv_device_bytes,
        kv_host_bytes,
        graphs,
        model.sampleable_vocab_size(),
    )?;

    // No device is opened, and nothing reaches one, until the driver reports the GPUs in
    // confidential-computing mode, accepting work, and at least one per device.
    let devices = DEEPSEEK_DEVICES.len() as u32;
    match gpu_cc {
        GpuCc::Required => crate::gpu_cc::query_devices()
            .and_then(|(state, reported)| crate::gpu_cc::check_devices(&state, reported, devices))
            .map_err(|e| {
                BootError(format!(
                    "{} is required, but the driver does not report {devices} GPUs in \
                     production confidential-computing mode: {e}",
                    env::GPU_CC
                ))
            })?,
        GpuCc::NotRequired => tracing::warn!(
            "{} is not-required: the GPUs' confidential-computing mode is not checked \
             (development only; no gateway pins this configuration)",
            env::GPU_CC
        ),
    }

    let refused = |e: eidola_engine_cuda::CudaError| {
        BootError(format!("the engine refused its configuration: {e}"))
    };
    let mut opened = Vec::with_capacity(DEEPSEEK_DEVICES.len());
    for (d, (ordinal, budget)) in DEEPSEEK_DEVICES
        .into_iter()
        .zip(kv_device_bytes)
        .enumerate()
    {
        let gpu = Gpu::open(ordinal).map_err(refused)?;
        let info = gpu.info();
        let total = info.total_mem_bytes as u64;
        if budget >= total {
            return Err(BootError(format!(
                "{} ({budget}, device {d}) is not less than its GPU's memory ({total} bytes)",
                env::KV_DEVICE_BYTES
            )));
        }
        let (major, minor) = info.compute_capability;
        tracing::info!(
            device = d,
            gpu = ordinal,
            name = %info.name,
            compute_capability = %format!("{major}.{minor}"),
            memory_bytes = total,
            "opened the GPU"
        );
        opened.push(gpu);
    }
    let gpus: [Gpu; 2] = match opened.try_into() {
        Ok(gpus) => gpus,
        Err(_) => unreachable!("one GPU per device"),
    };
    let info = gpus[0].info();
    let (major, minor) = info.compute_capability;
    let report = DeviceReport {
        name: info.name.clone(),
        compute_capability: format!("{major}.{minor}"),
        image: gpus[0].image_arch().map(|a| a.as_str()),
        sm_count: info.sm_count,
        memory_bytes: info.total_mem_bytes as u64,
    };
    Ok(DsPrepared {
        gpus,
        kernels_dir: kernels_dir.to_path_buf(),
        store: model.store().clone(),
        ds: ds.clone(),
        token_map: token_map.clone(),
        config,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use eidola_common::architecture::{Shape, UniformKvGroup};
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
            host_blocks: 0,
            ratio: 1,
            device: 0,
            coverage: eidola_engine::spec::Coverage::All,
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
            num_state_slots: 8,
            buckets: vec![Bucket {
                max_seqs: 8,
                max_tokens: 512,
            }],
            window_chain: None,
            token_history: 0,
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

    /// The host tier takes the device pools' mix scaled to its budget, its bytes never
    /// past the budget; 0 is none, and too small a budget is refused.
    #[test]
    fn host_blocks_scale_the_device_mix_to_the_budget() {
        let hidden = 4096;
        for k in [0u32, 3] {
            let spec = flash_drafting(k);
            let s = Sizing {
                draft_tokens: k,
                ..sizing()
            };
            let device = derive_kv_blocks(&spec, hidden, 8 << 30, &s, &cache(true)).unwrap();
            assert_eq!(
                derive_host_blocks(&spec, hidden, 0, &s, &device).unwrap(),
                KvBlocks::default()
            );
            let bytes = |b: &KvBlocks| -> u64 {
                spec.kv_groups
                    .iter()
                    .map(|g| {
                        let n = match (g.attention, g.role) {
                            (AttentionKind::Full, _) => b.global,
                            (_, KvRole::Target) => b.sliding,
                            (_, KvRole::Drafter) => b.drafter,
                        };
                        u64::from(n) * group_block_bytes(g, &s, hidden).unwrap()
                    })
                    .sum()
            };
            let on_device = bytes(&device);
            for budget in [on_device, 3 * on_device, 64 << 30, (64 << 30) + 12345] {
                let host = derive_host_blocks(&spec, hidden, budget, &s, &device).unwrap();
                assert!(bytes(&host) <= budget, "k {k} budget {budget}");
                // Each group's share follows the device's, within a block.
                let ratio = budget as f64 / on_device as f64;
                for (h, d) in [
                    (host.global, device.global),
                    (host.sliding, device.sliding),
                    (host.drafter, device.drafter),
                ] {
                    let want = f64::from(d) * ratio;
                    assert!((f64::from(h) - want).abs() <= 1.0, "{h} vs {want}");
                }
                assert_eq!(host.drafter == 0, k == 0);
            }
            assert_eq!(
                derive_host_blocks(&spec, hidden, on_device, &s, &device).unwrap(),
                device
            );
            let e = derive_host_blocks(&spec, hidden, 1 << 20, &s, &device).unwrap_err();
            assert!(e.0.contains(env::KV_HOST_BYTES), "{e}");
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
        let drafting = Sizing {
            draft_tokens: 3,
            ..sizing()
        };
        let blocks =
            derive_kv_blocks(&flash_drafting(3), 4096, budget, &drafting, &cache(true)).unwrap();
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

    /// The device reserve counts split attention's partial rows as the executor
    /// sizes them, from the node's one bucket.
    #[test]
    fn the_reserved_partial_rows_are_the_executors() {
        for (seats, step) in [(1u32, 4u32), (8, 512), (64, 8192), (1024, 2048)] {
            for draft in 0..=3u32 {
                for len in [1u32, 1024, 1025, 131_072, 1 << 20] {
                    let sizing = Sizing {
                        max_seqs: seats,
                        max_batched_tokens: step,
                        max_model_len: len,
                        draft_tokens: draft,
                        ..sizing()
                    };
                    let bucket = Bucket {
                        max_seqs: seats,
                        max_tokens: step,
                    };
                    assert_eq!(
                        u64::from(
                            eidola_engine_cuda::executor::split_partial_rows(bucket, draft, len)
                                .unwrap()
                        ),
                        eidola_common::engine_deployment::split_partial_rows(&sizing),
                        "{sizing:?}"
                    );
                }
            }
        }
    }

    /// The gateway's build holds a pinned deployment's budget to
    /// `cuda_kv_min_bytes`; it is exactly the least this derivation accepts, across
    /// block sizes, seats, step sizes, the cache switch and the draft width.
    #[test]
    fn the_shared_minimum_is_the_derivations() {
        use eidola_common::architecture::MIMO_V26_FLASH;
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
                    let min = cuda_kv_min_bytes(&MIMO_V26_FLASH, &sizing, &cache(enabled));
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

    /// The gateway's build holds a pinned deployment's non-zero host tier to
    /// `cuda_kv_host_min_bytes`; it is exactly the least the host derivation accepts
    /// for the device pools the same configuration derives, across block sizes, seats,
    /// step sizes, the draft width and device budgets.
    #[test]
    fn the_shared_host_minimum_is_the_derivations() {
        use eidola_common::architecture::MIMO_V26_FLASH;
        use eidola_common::engine_deployment::{cuda_kv_host_min_bytes, cuda_kv_min_bytes};
        for block in [1, 16, 64, 1024] {
            for (seats, step) in [(1, 4), (8, 512), (64, 8192)] {
                for draft in [0, 1, 3] {
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
                    let least = cuda_kv_min_bytes(&MIMO_V26_FLASH, &sizing, &cache(true));
                    for device_bytes in [least, least + 12_345, 3 * least, 200 << 30] {
                        let at = (block, seats, step, draft, device_bytes);
                        let device =
                            derive_kv_blocks(&spec, 4096, device_bytes, &sizing, &cache(true))
                                .unwrap_or_else(|e| panic!("{at:?}: {e}"));
                        let min = cuda_kv_host_min_bytes(
                            &MIMO_V26_FLASH,
                            &sizing,
                            &cache(true),
                            device_bytes,
                        );
                        let host = derive_host_blocks(&spec, 4096, min, &sizing, &device)
                            .unwrap_or_else(|e| panic!("{at:?}: {e}"));
                        assert!(
                            host.global.min(host.sliding) >= 2 && (draft == 0 || host.drafter >= 2),
                            "{at:?}: {host:?}"
                        );
                        assert!(
                            derive_host_blocks(&spec, 4096, min - 1, &sizing, &device).is_err(),
                            "{at:?}"
                        );
                    }
                }
            }
        }
    }

    /// The host and device bounds count `kv_groups(DRAFT_TOKENS)` block tables;
    /// an executor reporting another count is refused rather than undercounted.
    #[test]
    fn the_group_count_is_the_accounted_one() {
        let drafting = Sizing {
            draft_tokens: 3,
            ..sizing()
        };
        let budget = u64::MAX;
        derive_kv_blocks(&flash_drafting(3), 4096, budget, &drafting, &cache(true)).unwrap();
        let e = derive_kv_blocks(&flash_drafting(3), 4096, budget, &sizing(), &cache(true))
            .unwrap_err();
        assert!(e.0.contains("3 KV groups"), "{e}");
        let e = derive_kv_blocks(&flash(), 4096, budget, &drafting, &cache(true)).unwrap_err();
        assert!(e.0.contains("2 KV groups"), "{e}");
    }

    #[test]
    fn global_blocks_stop_at_the_tables_index_range() {
        let blocks = derive_kv_blocks(&flash(), 4096, u64::MAX, &sizing(), &cache(true)).unwrap();
        assert_eq!(blocks.global, i32::MAX as u32);
    }

    /// The model crate's tiny DeepSeek-V4.1 configuration (pooling ratio 2, a drafter).
    fn deepseek_tiny() -> DsConfig {
        DsConfig::from_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../eidola-engine-model/tests/fixtures/deepseek_tiny/config.json"),
        )
        .unwrap()
    }

    /// The released DeepSeek-V4.1-Flash configuration (the chat crate's copy of its
    /// `config.json`).
    fn deepseek_released() -> DsConfig {
        DsConfig::from_file(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../eidola-engine-chat/tests/fixtures/deepseek_v4_1/config.json"),
        )
        .unwrap()
    }

    fn ds_sizing(draft_tokens: u32) -> Sizing {
        Sizing {
            max_model_len: 64,
            max_seqs: 2,
            max_batched_tokens: 32,
            max_prefill_chunk: 32,
            draft_tokens,
            ..sizing()
        }
    }

    /// The executor's spec under `sizing`, and the bytes of one block of every group on
    /// each device, summed from the executor's own block layouts.
    fn ds_spec(ds: &DsConfig, sizing: &Sizing) -> (ModelSpec, [u64; 2]) {
        let config = deepseek_config(ds, sizing, [u64::MAX; 2], 0, CudaGraphs::Off, 90).unwrap();
        let (_, spec) = DsCudaExecutor::preflight(ds, &config).unwrap();
        let per = |half| {
            eidola_engine_cuda::ds::executor::geometry(ds, &spec, half)
                .unwrap()
                .iter()
                .map(|g| g.block_bytes() as u64)
                .sum::<u64>()
        };
        let per_block = [per(Half::Encoder), per(Half::Decoder)];
        (spec, per_block)
    }

    /// The dev model's KV groups in the shared contract's terms (the model crate's tiny
    /// configuration: a 32-wide entry, 35-byte bank entries and 33-byte window entries;
    /// one pooled source at ratio 2 and four window layers on device 0; one bank layer,
    /// four window layers and the drafter's three on device 1).
    const DEV_GROUPS: [&[UniformKvGroup]; 2] = [
        &[
            UniformKvGroup {
                layers: 1,
                ratio: 2,
                entry_bytes: 35,
                drafter: false,
            },
            UniformKvGroup {
                layers: 4,
                ratio: 1,
                entry_bytes: 33,
                drafter: false,
            },
        ],
        &[
            UniformKvGroup {
                layers: 1,
                ratio: 1,
                entry_bytes: 35,
                drafter: false,
            },
            UniformKvGroup {
                layers: 4,
                ratio: 1,
                entry_bytes: 33,
                drafter: false,
            },
            UniformKvGroup {
                layers: 3,
                ratio: 1,
                entry_bytes: 33,
                drafter: true,
            },
        ],
    ];

    /// The gateway's build holds a pinned DeepSeek-V4.1 deployment's per-device budgets
    /// to `cuda_kv_min_bytes` and its host tier to the devices' `cuda_kv_host_min_bytes`
    /// summed. Over the released configuration (the shared `DEEPSEEK_V41_FLASH` shapes)
    /// and the dev model's (`DEV_GROUPS`), across block sizes, context lengths and every
    /// draft width a deployment can set: the shared block bytes are the executor's, the
    /// node's derivation accepts exactly the shared minimum on each device and refuses
    /// one byte below on either, and likewise the host tier.
    #[test]
    fn the_shared_deepseek_minimums_are_the_derivations() {
        use eidola_common::architecture::DEEPSEEK_V41_FLASH;
        use eidola_common::engine_deployment::{
            cuda_kv_host_min_bytes, cuda_kv_min_bytes, uniform_kv_block_bytes,
        };
        let dev = DEV_GROUPS.map(|groups| Shape {
            uniform_kv_groups: groups,
            ..DEEPSEEK_V41_FLASH[0]
        });
        let cases = [
            (
                deepseek_released(),
                DEEPSEEK_V41_FLASH,
                vec![4096, 131_072, 1 << 20],
            ),
            (deepseek_tiny(), dev, vec![64, 1000, 1024]),
        ];
        for (ds, shapes, lengths) in cases {
            for block in [2u32, 16, 64, 256] {
                for &len in &lengths {
                    for draft in 0..=3u32 {
                        let sizing = Sizing {
                            kv_block_size: block,
                            max_model_len: len,
                            ..ds_sizing(draft)
                        };
                        let at = (block, len, draft);
                        let (spec, per_block) = ds_spec(&ds, &sizing);
                        let shared = shapes.map(|s| uniform_kv_block_bytes(&s, &sizing));
                        assert_eq!(shared, per_block, "{at:?}");
                        let min = shapes.map(|s| cuda_kv_min_bytes(&s, &sizing, &cache(true)));
                        let needed = len.div_ceil(block) + 1;
                        assert_eq!(
                            derive_deepseek_kv_blocks(&ds, &spec, min, &sizing).unwrap(),
                            needed,
                            "{at:?}"
                        );
                        for short in [[min[0] - 1, min[1]], [min[0], min[1] - 1]] {
                            assert!(
                                derive_deepseek_kv_blocks(&ds, &spec, short, &sizing).is_err(),
                                "{at:?}"
                            );
                        }
                        let host: u64 = shapes
                            .iter()
                            .zip(min)
                            .map(|(s, b)| cuda_kv_host_min_bytes(s, &sizing, &cache(true), b))
                            .sum();
                        assert_eq!(
                            derive_deepseek_host_blocks(&ds, &spec, host).unwrap(),
                            2,
                            "{at:?}"
                        );
                        assert!(
                            derive_deepseek_host_blocks(&ds, &spec, host - 1).is_err(),
                            "{at:?}"
                        );
                    }
                }
            }
        }
    }

    /// One count for every group: what the tighter device budget holds, each budget
    /// charged its own device's groups; the drafter's group counts on the decoder's device.
    #[test]
    fn deepseek_blocks_are_what_the_tighter_device_holds() {
        let ds = deepseek_tiny();
        for k in [0u32, 2] {
            let s = ds_sizing(k);
            let (spec, [s0, s1]) = ds_spec(&ds, &s);
            assert!(s0 > 0 && s1 > 0);
            for (b0, b1) in [
                (100 * s0, 300 * s1),
                (300 * s0 + 7, 100 * s1 + 7),
                (200 * s0, 200 * s1),
            ] {
                let n = derive_deepseek_kv_blocks(&ds, &spec, [b0, b1], &s).unwrap();
                assert_eq!(u64::from(n), (b0 / s0).min(b1 / s1), "k {k}");
            }
        }
        // Drafting adds the drafter's group to the decoder's device only.
        let (_, [e0, d0]) = ds_spec(&ds, &ds_sizing(0));
        let (spec, [e2, d2]) = ds_spec(&ds, &ds_sizing(2));
        assert_eq!(e0, e2);
        assert!(d2 > d0, "{d2} vs {d0}");
        assert_eq!(spec.kv_groups.len(), 5);
    }

    /// Each device must hold one `MAX_MODEL_LEN` sequence and the null block; the count
    /// stops at the tables' `i32` range.
    #[test]
    fn deepseek_blocks_hold_one_sequence_and_stop_at_the_index_range() {
        let ds = deepseek_tiny();
        let s = ds_sizing(0);
        let (spec, [s0, s1]) = ds_spec(&ds, &s);
        // 64 positions in blocks of 16, and the null block.
        let needed = 64 / 16 + 1;
        assert_eq!(
            derive_deepseek_kv_blocks(&ds, &spec, [needed * s0, needed * s1], &s).unwrap(),
            needed as u32
        );
        for (budgets, device) in [
            ([needed * s0 - 1, needed * s1], "device 0"),
            ([needed * s0, needed * s1 - 1], "device 1"),
        ] {
            let e = derive_deepseek_kv_blocks(&ds, &spec, budgets, &s).unwrap_err();
            assert!(e.0.contains(env::KV_DEVICE_BYTES), "{e}");
            assert!(e.0.contains(env::MAX_MODEL_LEN), "{e}");
            assert!(e.0.contains(device), "{e}");
        }
        assert_eq!(
            derive_deepseek_kv_blocks(&ds, &spec, [u64::MAX, u64::MAX], &s).unwrap(),
            i32::MAX as u32
        );
    }

    /// The host tier: one count for every group of both devices, its bytes within the
    /// budget; 0 is none; under two blocks refused.
    #[test]
    fn deepseek_host_blocks_fit_their_budget() {
        let ds = deepseek_tiny();
        let s = ds_sizing(2);
        let (spec, [s0, s1]) = ds_spec(&ds, &s);
        assert_eq!(derive_deepseek_host_blocks(&ds, &spec, 0).unwrap(), 0);
        for budget in [2 * (s0 + s1), 2 * (s0 + s1) + 1, 1000 * (s0 + s1) - 1] {
            let n = derive_deepseek_host_blocks(&ds, &spec, budget).unwrap();
            assert_eq!(u64::from(n), budget / (s0 + s1));
            assert!(u64::from(n) * (s0 + s1) <= budget);
        }
        let e = derive_deepseek_host_blocks(&ds, &spec, 2 * (s0 + s1) - 1).unwrap_err();
        assert!(e.0.contains(env::KV_HOST_BYTES), "{e}");
    }

    /// Graphs are refused (the executor captures none); the executor's own preflight
    /// refuses a draft width past the drafter's slots and a block size the pooling ratio
    /// does not divide; the derived configuration is the node's sizing.
    #[test]
    fn the_deepseek_configuration_is_the_nodes() {
        let ds = deepseek_tiny();
        let e =
            deepseek_config(&ds, &ds_sizing(0), [1 << 30; 2], 0, CudaGraphs::On, 90).unwrap_err();
        assert!(e.0.contains(env::CUDA_GRAPHS), "{e}");
        assert!(e.0.contains("no decode graphs"), "{e}");
        let slots = ds.drafter.as_ref().unwrap().slots as u32;
        let e = deepseek_config(
            &ds,
            &ds_sizing(slots + 1),
            [1 << 30; 2],
            0,
            CudaGraphs::Off,
            90,
        )
        .unwrap_err();
        assert!(e.0.contains("draft_width"), "{e}");
        let odd = Sizing {
            kv_block_size: 5,
            ..ds_sizing(0)
        };
        let e = deepseek_config(&ds, &odd, [1 << 30; 2], 0, CudaGraphs::Off, 90).unwrap_err();
        assert!(e.0.contains("ratio 2 must divide the block size"), "{e}");

        let s = ds_sizing(2);
        let c = deepseek_config(&ds, &s, [1 << 24, 1 << 23], 1 << 22, CudaGraphs::Off, 90).unwrap();
        let (spec, [s0, s1]) = ds_spec(&ds, &s);
        assert_eq!(
            c.num_blocks,
            derive_deepseek_kv_blocks(&ds, &spec, [1 << 24, 1 << 23], &s).unwrap()
        );
        assert_eq!(u64::from(c.host_blocks), (1u64 << 22) / (s0 + s1));
        assert_eq!(c.devices, DEEPSEEK_DEVICES);
        assert_eq!(
            (
                c.block_size,
                c.num_state_slots,
                c.max_model_len,
                c.draft_width
            ),
            (16, 2, 64, 2)
        );
        assert_eq!(
            c.buckets,
            vec![Bucket {
                max_seqs: 2,
                max_tokens: 32
            }]
        );
        assert_eq!(c.sampleable_vocab_size, 90);
        assert!(!c.position_tags && !c.record);
    }
}
