//! The grammar of an engine deployment's measured configuration, shared by
//! everything that reads one.
//!
//! An Eidola-hosted engine deployment is a `tinfoil-config.yml` whose
//! `eidola-server-engine` container boots from its environment. Three parties
//! read that config and must agree on what it means, so each rule here is the
//! one function they all call:
//!
//! - the node (`eidola-server-engine`), which boots from the values or
//!   refuses to;
//! - the gateway's build (`engine_trust/manifest.rs`), which refuses a pin
//!   for a config the node would refuse, or that it reads differently;
//! - `measure-enclave`, which turns the config into that pin.
//!
//! The Argon2id rule, and with it [`parse_measured`] (the whole measured
//! environment), need the `argon2` crate and sit behind this crate's
//! `argon2` feature: only the node and the gateway enable it, and both
//! already depend on `argon2` themselves (see the crate docs' dependency
//! rule).

pub use crate::architecture::{Architecture, HybridDims, Shape};

/// The node's environment variables. Every one is required (an empty value
/// counts as missing) except where noted.
pub mod env {
    /// The one model id this node serves.
    pub const MODEL_ID: &str = "EIDOLA_ENGINE_MODEL_ID";
    /// Directory holding the model's weights and chat artifacts.
    pub const WEIGHTS_DIR: &str = "EIDOLA_ENGINE_WEIGHTS_DIR";
    /// The weights hash the directory must have (64 hex digits).
    pub const WEIGHTS_SHA256: &str = "EIDOLA_ENGINE_WEIGHTS_SHA256";
    /// `verified-readonly` (production: a read-only, kernel-verified mount) or
    /// `dev-writable`.
    pub const WEIGHTS_STORAGE: &str = "EIDOLA_ENGINE_WEIGHTS_STORAGE";
    /// The gateway's bearer token. Secret, so not part of the measured
    /// configuration [`super::parse_measured`] reads.
    pub const GATEWAY_TOKEN: &str = "GATEWAY_TOKEN";
    /// Argon2id hash of the gateway token (measured).
    pub const GATEWAY_TOKEN_HASH: &str = "GATEWAY_TOKEN_HASH";
    /// `cpu` or `cuda`.
    pub const EXECUTOR: &str = "EIDOLA_ENGINE_EXECUTOR";
    /// Listen address, `host:port`.
    pub const BIND_ADDR: &str = "EIDOLA_ENGINE_BIND_ADDR";
    /// Positions per KV block.
    pub const KV_BLOCK_SIZE: &str = "EIDOLA_ENGINE_KV_BLOCK_SIZE";
    /// Physical KV blocks per group, including the reserved null block
    /// (`cpu` only).
    pub const KV_BLOCKS: &str = "EIDOLA_ENGINE_KV_BLOCKS";
    /// Device memory for the KV pools, in bytes, one value per device the
    /// model runs on, comma-separated in device order; the node derives the
    /// per-group block counts from it (`cuda` only).
    pub const KV_DEVICE_BYTES: &str = "EIDOLA_ENGINE_KV_DEVICE_BYTES";
    /// The kernel build output the CUDA executor loads its images from, each
    /// checked against the compiled-in kernel manifest (`cuda` only).
    pub const KERNELS_DIR: &str = "EIDOLA_ENGINE_KERNELS_DIR";
    /// Whether the CUDA executor replays captured graphs for decode steps:
    /// `on` or `off` (`cuda` only).
    pub const CUDA_GRAPHS: &str = "EIDOLA_ENGINE_CUDA_GRAPHS";
    /// Host memory for the prefix cache's host tier, in bytes: page-locked
    /// memory of the confidential VM where idle entries evicted from the KV
    /// pools under pressure are kept (as ciphertext) instead of dropped. 0
    /// disables the tier; a non-zero value needs the prefix cache. The node
    /// derives the per-group host block counts from it.
    pub const KV_HOST_BYTES: &str = "EIDOLA_ENGINE_KV_HOST_BYTES";
    /// Whether the node requires its GPUs in confidential-computing mode
    /// before it opens one: `required` (production; checked through NVML) or
    /// `not-required` (development on a GPU without it) (`cuda` only).
    pub const GPU_CC: &str = "EIDOLA_ENGINE_GPU_CC";
    /// Longest sequence (prompt plus completion), in tokens.
    pub const MAX_MODEL_LEN: &str = "EIDOLA_ENGINE_MAX_MODEL_LEN";
    /// Sequences per step (and per-sequence state slots).
    pub const MAX_SEQS: &str = "EIDOLA_ENGINE_MAX_SEQS";
    /// Query tokens per step.
    pub const MAX_BATCHED_TOKENS: &str = "EIDOLA_ENGINE_MAX_BATCHED_TOKENS";
    /// Largest prefill chunk for one sequence in one step.
    pub const MAX_PREFILL_CHUNK: &str = "EIDOLA_ENGINE_MAX_PREFILL_CHUNK";
    /// Speculative draft width `k` (MTP depths); 0 disables speculation.
    pub const DRAFT_TOKENS: &str = "EIDOLA_ENGINE_DRAFT_TOKENS";
    /// Requests admitted to the engine at once (running plus queued).
    pub const MAX_REQUESTS: &str = "EIDOLA_ENGINE_MAX_REQUESTS";
    /// Whether finished keyed requests' KV is kept for reuse (`true` / `false`).
    pub const PREFIX_CACHE: &str = "EIDOLA_ENGINE_PREFIX_CACHE";
    /// Prefix-cache idle lifetime, seconds.
    pub const CACHE_IDLE_TTL_SECS: &str = "EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS";
    /// Prefix-cache maximum age, seconds.
    pub const CACHE_MAX_AGE_SECS: &str = "EIDOLA_ENGINE_CACHE_MAX_AGE_SECS";

    /// Every variable of the measured environment: exactly the names
    /// [`super::parse_measured`] reads. A deployment's env sets these and
    /// nothing else, so no variable a dependency happens to consume
    /// (`TOKIO_WORKER_THREADS`, `RUST_MIN_STACK`, `MALLOC_*`, `LD_*`,
    /// `CUDA_*`, …) can reach the node through its measured config.
    pub const MEASURED: &[&str] = &[
        MODEL_ID,
        WEIGHTS_DIR,
        WEIGHTS_SHA256,
        WEIGHTS_STORAGE,
        GATEWAY_TOKEN_HASH,
        EXECUTOR,
        BIND_ADDR,
        KV_BLOCK_SIZE,
        KV_BLOCKS,
        KV_DEVICE_BYTES,
        KERNELS_DIR,
        CUDA_GRAPHS,
        KV_HOST_BYTES,
        GPU_CC,
        MAX_MODEL_LEN,
        MAX_SEQS,
        MAX_BATCHED_TOKENS,
        MAX_PREFILL_CHUNK,
        DRAFT_TOKENS,
        MAX_REQUESTS,
        PREFIX_CACHE,
        CACHE_IDLE_TTL_SECS,
        CACHE_MAX_AGE_SECS,
    ];
}

/// Whether a deployment's measured env sets only [`env::MEASURED`] names.
pub fn check_env_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    match names.into_iter().find(|name| !env::MEASURED.contains(name)) {
        None => Ok(()),
        Some(name) => Err(format!(
            "the engine container's env may not set {name:?}; it sets exactly the node's \
             measured variables"
        )),
    }
}

/// Caps on the node's allocation-driving sizes. Each follows from the
/// implemented architectures (MiMo-V2.6-Flash, whose CUDA executor
/// `eidola-engine-cuda`'s `support::check_supported` names, and
/// DeepSeek-V4.1-Flash, whose context window and vocabulary are within
/// MiMo's) and the CUDA executor's own limits, generous but finite.
pub mod caps {
    /// The model's context window (`support::MAX_POSITIONS`, 2^20).
    pub const MAX_MODEL_LEN: u32 = 1 << 20;
    /// Positions per KV block: a block is the attention kernel's page and the
    /// prefix cache's unit of reuse; beyond 1,024 positions paging stops
    /// paying, and the page stride (32-bit) stays far from overflow.
    pub const KV_BLOCK_SIZE: u32 = 1024;
    /// Device memory for the CUDA executor's KV pools on each device: less
    /// than the supported GPU's memory ([`super::SUPPORTED_GPU_MEMORY_BYTES`]), as the
    /// node itself refuses a value not less than its device's. What a pinned
    /// deployment may give the pools, beside its weights and the executor's
    /// own reserve, is held separately ([`super::check_resources`]).
    pub const KV_DEVICE_BYTES: u64 = super::SUPPORTED_GPU_MEMORY_BYTES - 1;
    /// Host memory for the prefix cache's host tier: at most the largest VM
    /// memory a deployment may have ([`super::MAX_MEMORY_MIB`], 2 TiB). What
    /// a pinned deployment's VM must hold besides is held separately
    /// ([`super::check_resources`]).
    pub const KV_HOST_BYTES: u64 = super::MAX_MEMORY_MIB << 20;
    /// Blocks per KV group (the `cpu` executor's): the device block tables are `i32`
    /// (`kv::GroupGeometry::validate`). The device bytes they cost are held
    /// to the attached GPUs separately ([`super::check_resources`]).
    pub const KV_BLOCKS: u32 = i32::MAX as u32;
    /// Sequences per step, which is also the state slots: the sampler's
    /// scratch is rows × the 152,576-entry vocabulary × 4 bytes (625 MB at
    /// 1,024), and each slot holds a block table.
    pub const MAX_SEQS: u32 = 1024;
    /// Query tokens per step: the forward's activation scratch scales with it
    /// (tokens × the 16,384-wide dense intermediate × 2 bytes, 2 GiB at
    /// 65,536).
    pub const MAX_BATCHED_TOKENS: u32 = 65_536;
    /// Largest prefill chunk: bounded by a step's tokens, so the same.
    pub const MAX_PREFILL_CHUNK: u32 = 65_536;
    /// Speculative draft width: MiMo's MTP heads are a handful; the node also
    /// refuses more than the loaded model has.
    pub const DRAFT_TOKENS: u32 = 8;
    /// Speculative draft width with the CUDA executor: the most MTP layers
    /// (`num_nextn_predict_layers`, one per draft depth) an implemented
    /// architecture ships (`crate::architecture::Shape::mtp_layers`;
    /// MiMo-V2.6-Flash's three, `eidola-engine-cuda` `support::MTP_LAYERS`).
    pub const CUDA_DRAFT_TOKENS: u32 = 3;
    /// Admission slots: each may hold a request body of up to
    /// `engine_protocol::MAX_REQUEST_BODY_BYTES` and its parse, held to the
    /// VM's memory separately ([`super::check_resources`]).
    pub const MAX_REQUESTS: u32 = 4096;
}

// The numbers a model's shape fixes (KV bytes per position, the sliding
// window, the drafter's layers, the executor's per-token scratch) are its
// architecture's: `crate::architecture::Shape`, read through
// `Architecture::shape`. What follows holds for every architecture.

/// Retention points a keyed sequence may hold per sliding group with the
/// prefix cache enabled (the node's `cuda::RETENTION_POINTS`).
pub const RETENTION_POINTS: u64 = 3;

/// The one GPU the CUDA executor runs on (`eidola-engine-cuda`: Blackwell,
/// `sm_103a`), by the name its driver reports. A node uses one of them.
pub const SUPPORTED_GPU: &str = "NVIDIA B300 SXM6 AC";

/// The supported GPU's device memory, in bytes, as its driver reports it
/// (`cuDeviceTotalMem`, the node's `/v1/engine/info` `device.memory_bytes`
/// on that part): about 267.7 GiB, not the 288 GB of HBM it is sold with.
pub const SUPPORTED_GPU_MEMORY_BYTES: u64 = 287_428_640_768;

/// Device memory the CUDA executor holds whatever its sizing: the CUDA
/// context, the loaded kernel modules, the decode graphs' captures (a few
/// MiB a rung; with drafting, a ladder each for `k` drafts and for none), and the slack of the weights' device layouts over their file
/// bytes (each fused QKV chunk padded to whole 128-row tiles, norm and bias
/// vectors widened to f32): 4 GiB.
pub const DEVICE_FIXED_RESERVE_BYTES: u64 = 4 << 30;

/// Sampler rows per seat (`MAX_SEQS`): one without drafting; with `k`
/// drafts, a step keeps `k` draft distributions, `k + 1` target
/// distributions and acceptance's residual per row (`draft::probs_rows`).
pub fn sampler_rows(draft_tokens: u32) -> u64 {
    if draft_tokens == 0 {
        1
    } else {
        2 * u64::from(draft_tokens) + 2
    }
}

/// Rows of the hidden size (4,096 f32, 16 KiB) a drafting executor keeps
/// per seat for `k` drafts: the slot's drafter state (every depth's level
/// at each of the `k + 1` positions a step can leave it at), and the step's
/// level rows (`draft::Levels`: `k` loaded, `2 (k + 1) k` write-only).
pub fn drafter_seat_rows(draft_tokens: u32) -> u64 {
    let k = u64::from(draft_tokens);
    if k == 0 {
        return 0;
    }
    (k + 1) * k + k + 2 * (k + 1) * k
}

/// Step-table bytes a drafting executor may hold per seat beside its global
/// page lists, in a graph's table and an eager step's together: every other
/// per-row array of a drafted step (copy lists, token ids, positions, KV
/// targets, work lists, the drafter's one-position page lists, sampler and
/// acceptance rows; about 2,400 words a table for three depths), bounded by
/// 32 KiB. An eager step's arrays per token beyond those (about 30 words for
/// three depths) are inside [`Shape::step_token_device_bytes`]' rounding.
pub const DRAFT_TABLE_SEAT_BYTES: u64 = 32 << 10;

/// Query slots a step of decode rows takes when every seat has one:
/// `MAX_SEQS × (1 + DRAFT_TOKENS)`, a row's token and its drafts. The
/// grammar holds `MAX_BATCHED_TOKENS` to at least this ([`parse_measured`]).
pub fn decode_step_tokens(sizing: &Sizing) -> u64 {
    u64::from(sizing.max_seqs) * (u64::from(sizing.draft_tokens) + 1)
}

/// Keys per chunk of the CUDA executor's split global attention
/// (`eidola-engine-cuda` `attention::SPLIT_KEYS`): a global row's keys are
/// reduced in chunks of this many positions, each a partial row, then merged.
pub const SPLIT_ATTENTION_KEYS: u64 = 1024;

/// Split attention's partial-row scratch (`eidola-engine-cuda`
/// `executor::split_partial_rows`): the most tokens a captured step holds,
/// `min(MAX_SEQS × (1 + DRAFT_TOKENS), MAX_BATCHED_TOKENS)`, times the
/// chunks of a `MAX_MODEL_LEN` sequence. Saturates.
pub fn split_partial_rows(sizing: &Sizing) -> u64 {
    u64::from(sizing.max_seqs)
        .saturating_mul(u64::from(sizing.draft_tokens) + 1)
        .min(u64::from(sizing.max_batched_tokens))
        .max(1)
        .saturating_mul(u64::from(sizing.max_model_len.max(1)).div_ceil(SPLIT_ATTENTION_KEYS))
}

/// Step-table bytes a drafting executor holds per seat for each chunk of
/// [`SPLIT_ATTENTION_KEYS`] of `MAX_MODEL_LEN`: a row's global work items
/// (its verify run cut in at most two pieces at a 64-key boundary, each a
/// work item per chunk, three words each), in a graph's table and an eager
/// step's.
pub const DRAFT_TABLE_SEAT_BYTES_PER_CHUNK: u64 = 2 * 2 * 3 * 4;

/// Device memory the CUDA executor takes beside its weights and KV pools,
/// bounded from above, for a model of shape `shape`:
/// [`DEVICE_FIXED_RESERVE_BYTES`], plus
/// [`Shape::step_token_device_bytes`] per `MAX_BATCHED_TOKENS`,
/// [`Shape::expert_floor_device_bytes`], per `MAX_SEQS` [`sampler_rows`] of
/// [`Shape::sampler_row_device_bytes`], the device block tables (`MAX_SEQS` ×
/// ⌈`MAX_MODEL_LEN` / `KV_BLOCK_SIZE`⌉ `i32` entries per KV group,
/// [`kv_groups`]) and the RoPE tables
/// ([`Shape::rope_device_bytes_per_position`] per `MAX_MODEL_LEN`), and split
/// attention's [`split_partial_rows`] of [`Shape::split_partial_row_bytes`]. With
/// `DRAFT_TOKENS` `k` > 0, per `MAX_SEQS` also [`drafter_seat_rows`] rows of
/// [`Shape::hidden_row_bytes`], [`DRAFT_TABLE_SEAT_BYTES`], the global page lists of a drafted
/// step's table four times over (a graph's and an eager step's, each a row's
/// pages twice when its verify run is cut at a 64-key boundary, one `i32` per
/// block of `MAX_MODEL_LEN`), and [`DRAFT_TABLE_SEAT_BYTES_PER_CHUNK`] per
/// chunk of `MAX_MODEL_LEN`. The MTP layers' own scratch (their
/// copy lists, 36 bytes a token) and the drafted step's copy of the target's
/// states (one hidden row a token) are inside [`Shape::step_token_device_bytes`]'
/// rounding, and their weights inside the pack. Saturates.
///
/// A shape with a [`Shape::step_scratch`] (DeepSeek-V4.1's) counts the step's
/// scratch by that instead of the per-token, expert-floor and sampler fields:
/// [`DEVICE_FIXED_RESERVE_BYTES`], [`step_scratch_bytes`], the device block
/// tables and the RoPE tables as above, the per-seat terms its
/// [`StepScratch`](crate::architecture::StepScratch) names in place of the
/// drafter's.
pub fn cuda_device_reserve_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    if let Some(dims) = &shape.recurrent {
        return recurrent_device_reserve_bytes(dims, sizing);
    }
    let blocks_per_seq =
        u64::from(sizing.max_model_len).div_ceil(u64::from(sizing.kv_block_size.max(1)));
    if shape.step_scratch.is_some() {
        return DEVICE_FIXED_RESERVE_BYTES
            .saturating_add(step_scratch_bytes(shape, sizing))
            .saturating_add(
                u64::from(sizing.max_seqs)
                    .saturating_mul(blocks_per_seq)
                    .saturating_mul(kv_groups(sizing.draft_tokens) * 4),
            )
            .saturating_add(
                u64::from(sizing.max_model_len)
                    .saturating_mul(shape.rope_device_bytes_per_position),
            );
    }
    let blocks_per_seq =
        u64::from(sizing.max_model_len).div_ceil(u64::from(sizing.kv_block_size.max(1)));
    let seats = u64::from(sizing.max_seqs);
    let drafting = sizing.draft_tokens > 0;
    let groups = kv_groups(sizing.draft_tokens);
    let per_seat = sampler_rows(sizing.draft_tokens)
        .saturating_mul(shape.sampler_row_device_bytes)
        .saturating_add(
            drafter_seat_rows(sizing.draft_tokens).saturating_mul(shape.hidden_row_bytes),
        )
        .saturating_add(if drafting {
            let chunks = u64::from(sizing.max_model_len).div_ceil(SPLIT_ATTENTION_KEYS);
            DRAFT_TABLE_SEAT_BYTES
                .saturating_add(blocks_per_seq.saturating_mul(4 * 4))
                .saturating_add(chunks.saturating_mul(DRAFT_TABLE_SEAT_BYTES_PER_CHUNK))
        } else {
            0
        });
    DEVICE_FIXED_RESERVE_BYTES
        .saturating_add(
            u64::from(sizing.max_batched_tokens).saturating_mul(shape.step_token_device_bytes),
        )
        .saturating_add(shape.expert_floor_device_bytes)
        .saturating_add(seats.saturating_mul(per_seat))
        .saturating_add(
            seats
                .saturating_mul(blocks_per_seq)
                .saturating_mul(groups * 4),
        )
        .saturating_add(split_partial_rows(sizing).saturating_mul(shape.split_partial_row_bytes))
        .saturating_add(
            u64::from(sizing.max_model_len).saturating_mul(shape.rope_device_bytes_per_position),
        )
}

/// State buffers of a recurrent model's pool (Qwen3.8's): the reserved zero
/// state and, per seat (`MAX_SEQS`), its base, the one fresh buffer a step
/// may store for it, and with the prefix cache [`RETENTION_POINTS`]
/// snapshots (its branch, regeneration and continuation points):
/// 1 + `MAX_SEQS` × (2 + `P`). The serving core's minimum progress needs
/// 1 + 2 × `MAX_SEQS` (`eidola-engine/AGENTS.md` → State pool); the rest
/// holds snapshots, as the sliding pools hold retained windows
/// ([`cuda_kv_min_bytes`]). Saturates.
pub fn recurrent_state_buffers(sizing: &Sizing, cache: &CacheConfig) -> u64 {
    let points = if cache.enabled { RETENTION_POINTS } else { 0 };
    u64::from(sizing.max_seqs)
        .saturating_mul(2 + points)
        .saturating_add(1)
}

/// Bytes of one KV block of a recurrent model on the device, every group
/// together: the target's attention layers and, drafting, the drafter's
/// layer and its block's boundary tap. Every group holds one count of
/// blocks. Saturates.
pub fn recurrent_kv_block_bytes(dims: &HybridDims, sizing: &Sizing) -> u64 {
    let b = u64::from(sizing.kv_block_size.max(1));
    let target = b.saturating_mul(dims.kv_bytes_per_position());
    if sizing.draft_tokens > 0 {
        target
            .saturating_add(b.saturating_mul(dims.drafter_kv_bytes_per_position()))
            .saturating_add(dims.tap_bytes())
    } else {
        target
    }
}

/// Device bytes a recurrent model's executor keeps per seat: its live
/// state image, its tail of `KV_BLOCK_SIZE − 1 + DRAFT_TOKENS` positions
/// (the raw inputs, and the fast profile's blocked DeltaNet's tail cache:
/// [`HybridDims::chunk_tail_bytes_per_position`]), and `DRAFT_TOKENS + 1`
/// kept normed states. Saturates.
pub fn recurrent_seat_bytes(dims: &HybridDims, sizing: &Sizing) -> u64 {
    let k = u64::from(sizing.draft_tokens);
    let tail = (u64::from(sizing.kv_block_size.max(1)) - 1 + k).max(1);
    dims.state_image_bytes()
        .saturating_add(
            tail.saturating_mul(
                dims.tail_bytes_per_position() + dims.chunk_tail_bytes_per_position(),
            ),
        )
        .saturating_add((k + 1).saturating_mul(dims.kept_row_bytes()))
}

/// Bounds the program's allocations of no row (the sink, the status and
/// zero rows, each allocation's eight-byte floor).
const RECURRENT_SCRATCH_FIXED_BYTES: u64 = 64 << 10;

/// Step-table words a row of a forward takes besides its attention's pages:
/// its token, positions, KV target, DeltaNet sequence and copy items, its
/// FA2 request and work items, sampling and acceptance words. Bounded from
/// above by the node's `cuda::tests`, which hold the executor's tables to it.
pub const RECURRENT_TABLE_ROW_WORDS: u64 = 1024;

/// Words of one step table at its largest, for a recurrent model's executor
/// (`eidola-engine-cuda`'s `qwen::plan`): per target row (the `N` host
/// tokens and drafts of passes A and B) the pages of a `MAX_MODEL_LEN + k +
/// 1`-position sequence, a word a block, and per drafter row (with drafting,
/// `N + 2Sk`: the host positions, the chain and the accepted drafts'
/// rewrite) a word a position (the drafter's pages are positions), each with
/// [`RECURRENT_TABLE_ROW_WORDS`]. Saturates.
pub fn recurrent_table_words(sizing: &Sizing) -> u64 {
    let n = u64::from(sizing.max_batched_tokens);
    let s = u64::from(sizing.max_seqs);
    let k = u64::from(sizing.draft_tokens);
    let positions = u64::from(sizing.max_model_len) + k + 1;
    let blocks = positions.div_ceil(u64::from(sizing.kv_block_size.max(1)));
    let target = n.saturating_mul(blocks.saturating_add(RECURRENT_TABLE_ROW_WORDS));
    let drafter = if k > 0 {
        n.saturating_add(s.saturating_mul(2 * k))
            .saturating_mul(positions.saturating_add(RECURRENT_TABLE_ROW_WORDS))
    } else {
        0
    };
    target.saturating_add(drafter)
}

/// Keys per chunk of FA2's split attention (`eidola-engine-cuda`'s
/// `qwen::plan::SPLIT_KEYS`): a row's keys are reduced in chunks of this
/// many positions, each a partial row.
pub const RECURRENT_SPLIT_KEYS: u64 = 1024;

/// A recurrent model's step scratch in the fast profile (the executor's
/// `capacities` and `program::alloc_scratch`, sized once at construction,
/// plus its step tables), bounded from above. With `N` =
/// `MAX_BATCHED_TOKENS`, `S` = `MAX_SEQS`, `k` = `DRAFT_TOKENS`, `B` =
/// `KV_BLOCK_SIZE`, the tail `T` = `B − 1 + k`, the forward's rows `R` =
/// max(`N`, `Sk`), padded to 4, and the head's rows `H` = max(`S`, `Sk`, 64):
///
/// - per row of `R`, [`HybridDims::program_row_bytes`]; pass A's and pass
///   B's normed rows;
/// - the DeltaNet's staging: max(`N + ST`, `S(k + 1 + T)`) staged positions
///   ([`HybridDims::staged_position_bytes`]) and three more convolution rows a
///   seat;
/// - FA2's partial rows: `R` rows times the chunks of [`RECURRENT_SPLIT_KEYS`]
///   of `MAX_MODEL_LEN + k + 1` positions ([`HybridDims::partial_row_bytes`]):
///   about 12 bytes per row and position of the context at Qwen3.8's shape,
///   where the exact profile's scores, which this profile does not allocate,
///   take 96 a layer;
/// - the head's input, its f32 output (or the drafter's and the stacked
///   `a`/`b` projections', the largest), the target's logit rows, max(`S`,
///   64) and `Sk` more, and the drafter's (`H`), f32 over the head's rows, and the
///   sampler's f64 distributions: `S` plain, `S(k + 1)` target, `Sk` draft and
///   `S` acceptance rows;
/// - the padding rows of the row copies (a staged position's convolution and
///   decay rows for every staged position, besides the rows' and the state
///   writes' own), the draft lanes and the read-back;
/// - the blocked DeltaNet's scratch: a block of
///   [`HybridDims::chunk_tile_bytes`] for every `C` staged positions and one
///   more a sequence and one besides, `C` the block size's largest
///   power-of-two divisor, at most 16;
/// - the step tables: the rungs' fixed table and the eager table, which grows
///   to a power of two, together at most three times
///   [`recurrent_table_words`] words;
/// - `RECURRENT_SCRATCH_FIXED_BYTES` (64 KiB).
///
/// Saturates.
pub fn recurrent_step_scratch_bytes(dims: &HybridDims, sizing: &Sizing) -> u64 {
    let pad4 = |x: u64| x.next_multiple_of(4).max(4);
    let n = u64::from(sizing.max_batched_tokens);
    let s = u64::from(sizing.max_seqs);
    let k = u64::from(sizing.draft_tokens);
    let b = u64::from(sizing.kv_block_size.max(1));
    let tail = b - 1 + k;
    let positions = u64::from(sizing.max_model_len).saturating_add(k + 1);
    let (h, v) = (dims.hidden, dims.vocab);
    let rows = n.max(s * k);
    let r = pad4(rows);
    let head = s.max(s * k).max(64);
    let staged = n
        .saturating_add(s.saturating_mul(tail))
        .max(s.saturating_mul(k + 1 + tail));
    let conv_rows = 3 * s + staged;
    let partial_rows = rows.saturating_mul(positions.div_ceil(RECURRENT_SPLIT_KEYS).max(1));
    let head_tmp = (pad4(head) * v).max(r * h).max(r * dims.ab_rows()) * 4;
    let logit_rows = s.max(64) + s * k + head;
    let distributions = s + s * (k + 1) + (s * k).max(1) + s;
    let pad_conv = 3 * s + staged + 3 * 4 * s + 3 * s + s * (k + 1);
    let pad_ab = staged + s * (k + 1);
    let pad_h = s * (k + 1) + s * ((k + 1) / b + 1) + s * k + s;
    let readback = 1 + s + s * (k + 1) + s;
    let chunk = (b & b.wrapping_neg()).clamp(1, 16);
    let chunk_tiles = staged / chunk + s + 1;
    r.saturating_mul(dims.program_row_bytes())
        .saturating_add((pad4(n) + pad4((s * k).max(1))) * h * 2)
        .saturating_add(staged.saturating_mul(dims.staged_position_bytes()))
        .saturating_add(conv_rows.saturating_mul(dims.conv_row_bytes()))
        .saturating_add(partial_rows.saturating_mul(dims.partial_row_bytes()))
        .saturating_add(pad4(head) * h * 2 + head_tmp)
        .saturating_add(logit_rows.saturating_mul(v * 4))
        .saturating_add(distributions.saturating_mul(v * 8))
        .saturating_add(
            pad_conv * dims.conv_row_bytes()
                + pad_ab * dims.ab_row_bytes()
                + s.max(s * k) * dims.value_row_bytes()
                + pad_h * h * 2
                + (s * k + s) * 4
                + (s * k).max(1) * 4
                + n.max(s * k) * 4
                + readback * 4,
        )
        .saturating_add(chunk_tiles.saturating_mul(dims.chunk_tile_bytes()))
        .saturating_add(recurrent_table_words(sizing).saturating_mul(3 * 4))
        .saturating_add(RECURRENT_SCRATCH_FIXED_BYTES)
}

/// [`cuda_device_reserve_bytes`] for a recurrent model: [`DEVICE_FIXED_RESERVE_BYTES`],
/// [`recurrent_seat_bytes`] for every seat and the padding rows' seat, the RoPE
/// tables over `MAX_MODEL_LEN + DRAFT_TOKENS + 1` positions, and
/// [`recurrent_step_scratch_bytes`]. Saturates.
fn recurrent_device_reserve_bytes(dims: &HybridDims, sizing: &Sizing) -> u64 {
    let rope_positions = u64::from(sizing.max_model_len) + u64::from(sizing.draft_tokens) + 1;
    DEVICE_FIXED_RESERVE_BYTES
        .saturating_add(
            (u64::from(sizing.max_seqs) + 1).saturating_mul(recurrent_seat_bytes(dims, sizing)),
        )
        .saturating_add(rope_positions.saturating_mul(dims.rope_bytes_per_position()))
        .saturating_add(recurrent_step_scratch_bytes(dims, sizing))
}

/// The rows a step's scratch is sized for on a device with a
/// [`Shape::step_scratch`]: `MAX_BATCHED_TOKENS`, or with drafting every slot
/// of `MAX_SEQS` drafting rows when that is more. 0 for a shape without one.
pub fn step_scratch_rows(shape: &Shape, sizing: &Sizing) -> u64 {
    let Some(s) = shape.step_scratch else {
        return 0;
    };
    let pass = if sizing.draft_tokens > 0 {
        u64::from(sizing.max_seqs).saturating_mul(s.draft_slots)
    } else {
        0
    };
    u64::from(sizing.max_batched_tokens).max(pass)
}

/// Bytes one step row's index scores and candidate mask take on a device
/// with a [`Shape::step_scratch`], for sequences of `MAX_MODEL_LEN`: a 16-bit
/// score for every entry the device's indexer can see (`MAX_MODEL_LEN /
/// index_ratio + 1`), and a 32-bit mask word for every 32 candidate blocks of
/// those entries. 0 for a shape without one. Saturates.
pub fn index_scratch_row_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    let Some(s) = shape.step_scratch else {
        return 0;
    };
    let entries = u64::from(sizing.max_model_len) / s.index_ratio.max(1) + 1;
    let mask_words = entries.div_ceil(s.candidate_entries.max(1)).div_ceil(32);
    entries
        .saturating_mul(2)
        .saturating_add(mask_words.saturating_mul(4))
}

/// A device's step scratch by its [`Shape::step_scratch`] (0 for a shape
/// without one): per row of [`step_scratch_rows`] its row bytes and
/// [`index_scratch_row_bytes`], per `MAX_SEQS` its seat bytes, and its fixed
/// bytes. Saturates.
pub fn step_scratch_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    let Some(s) = shape.step_scratch else {
        return 0;
    };
    let drafting = sizing.draft_tokens > 0;
    let row = s
        .row_bytes
        .saturating_add(if drafting { s.draft_row_bytes } else { 0 })
        .saturating_add(index_scratch_row_bytes(shape, sizing));
    let seat = s
        .seat_bytes
        .saturating_add(u64::from(sizing.draft_tokens).saturating_mul(s.per_draft_seat_bytes))
        .saturating_add(if drafting { s.drafting_seat_bytes } else { 0 });
    step_scratch_rows(shape, sizing)
        .saturating_mul(row)
        .saturating_add(u64::from(sizing.max_seqs).saturating_mul(seat))
        .saturating_add(s.fixed_bytes)
}

/// The weights a device holds as its executor loads them for `sizing`:
/// [`Shape::weights_bytes`], less [`Shape::drafter_weights_bytes`] when the
/// executor does not draft. `None` where the shape states no bound.
pub fn device_weights_bytes(shape: &Shape, sizing: &Sizing) -> Option<u64> {
    let w = shape.weights_bytes?;
    Some(if sizing.draft_tokens > 0 {
        w
    } else {
        w.saturating_sub(shape.drafter_weights_bytes)
    })
}

/// Bytes a request's parses take per JSON value it holds (object keys
/// included; `engine_protocol::check_request_json` counts them, and refuses
/// a body of more than `MAX_REQUEST_JSON_VALUES` before either parse runs),
/// both parses together: the strict `serde_json` tree and the chat crate's
/// order-preserving tree, live at once in `api::parse_request`.
///
/// Measured by `eidola-server-engine`'s `tests/parse_memory.rs` with a
/// counting allocator, over the densest bodies at the value cap: the worst
/// is 425.5 bytes a value (one-member objects nested 30 deep, `serde_json`'s
/// default B-tree maps), and 414.3 with its `preserve_order` feature, which a
/// whole-workspace build unifies in (single-element arrays nested 59 deep).
/// Rounded up to 512.
pub const PARSE_PER_VALUE_BYTES: u64 = 512;

/// Bytes a request's parses take per byte of its body, beside the body
/// itself: every string's text is copied once into each tree (an escape only
/// shortens it). Measured as above: 32 MiB of text peaks at three times the
/// body, the body and its two copies.
pub const PARSE_PER_TEXT_BYTE: u64 = 2;

/// Bytes the order-preserving tree an admitted request keeps takes per JSON
/// value: measured as above, at most 126.9 (single-element arrays nested 59
/// deep), rounded up to 160.
pub const ORDERED_PER_VALUE_BYTES: u64 = 160;

/// Bytes that tree takes per byte of the body: its strings' text, once.
pub const ORDERED_PER_TEXT_BYTE: u64 = 1;

/// The most a read slot holds while `api::parse_request` runs, for a body of
/// `body` bytes and `values` JSON values: the body, its text copied into both
/// trees ([`PARSE_PER_TEXT_BYTE`]) and both trees' nodes
/// ([`PARSE_PER_VALUE_BYTES`] a value).
pub fn read_slot_bytes(body: u64, values: u64) -> u64 {
    body.saturating_mul(1 + PARSE_PER_TEXT_BYTE)
        .saturating_add(values.saturating_mul(PARSE_PER_VALUE_BYTES))
}

/// The most an admitted request's order-preserving tree holds, for a body of
/// `body` bytes and `values` JSON values.
pub fn ordered_tree_bytes(body: u64, values: u64) -> u64 {
    body.saturating_mul(ORDERED_PER_TEXT_BYTE)
        .saturating_add(values.saturating_mul(ORDERED_PER_VALUE_BYTES))
}

/// Worst-case bytes a prepared request adds per byte of its body
/// (`pipeline::prepare`): the rendered prompt (at most twice the message
/// text, the template's per-message markup counted) and its tokens (a `u32`
/// per prompt byte at worst, so four bytes each), 2 + 8.
pub const PREPARED_PER_BODY_BYTE: u64 = 10;

/// KV groups an executor keeps block tables for, host mirror and device
/// copy alike: MiMo's global and sliding target groups, and with
/// `DRAFT_TOKENS` > 0 the drafter's. The node refuses to boot an executor
/// that reports another count (`cuda::derive_kv_blocks` in
/// `eidola-server-engine`), so the memory bounds below cannot undercount it.
pub fn kv_groups(draft_tokens: u32) -> u64 {
    if draft_tokens > 0 { 3 } else { 2 }
}

/// The node process's own fixed footprint beside its pools (runtime, engine
/// state, the model's host-side metadata): 4 GiB.
pub const PROCESS_HEADROOM_BYTES: u64 = 4 << 30;

/// The host memory a node's configuration commits it to, at worst, from its
/// own code paths (`http::chat`, `api::parse_request`, `pipeline::prepare`),
/// for bodies of up to `MAX_REQUEST_BODY_BYTES` and `MAX_REQUEST_JSON_VALUES`:
///
/// - the **read pool**: `MAX_REQUESTS` slots, each a body live at once with
///   its strict parse and its order-preserving parse (`parse_request` builds
///   both from the body before the body is dropped): [`read_slot_bytes`];
/// - the **admission pool**: `MAX_REQUESTS` admitted requests, each keeping
///   its order-preserving tree while the prompt is rendered and tokenized:
///   [`ordered_tree_bytes`] plus body × [`PREPARED_PER_BODY_BYTE`]. Both
///   pools can be full at once (a slot of each per request in flight);
/// - the block tables' host mirror: `MAX_SEQS` × ⌈`MAX_MODEL_LEN` /
///   `KV_BLOCK_SIZE`⌉ `i32` entries per KV group ([`kv_groups`]: two, three
///   with drafting);
/// - [`PROCESS_HEADROOM_BYTES`].
///
/// At the caps a request slot is 645,922,816 bytes (about 616 MiB): 234.9 MB
/// read, 411.0 MB admitted.
pub fn host_memory_bytes(sizing: &Sizing) -> u64 {
    let body = crate::engine_protocol::MAX_REQUEST_BODY_BYTES as u64;
    let values = crate::engine_protocol::MAX_REQUEST_JSON_VALUES as u64;
    let requests = u64::from(sizing.max_requests);
    let read_pool = requests.saturating_mul(read_slot_bytes(body, values));
    let admission_pool = requests.saturating_mul(
        ordered_tree_bytes(body, values).saturating_add(body * PREPARED_PER_BODY_BYTE),
    );
    let blocks_per_seq = u64::from(sizing.max_model_len).div_ceil(u64::from(sizing.kv_block_size));
    let tables = u64::from(sizing.max_seqs) * blocks_per_seq * 4 * kv_groups(sizing.draft_tokens);
    read_pool + admission_pool + tables + PROCESS_HEADROOM_BYTES
}

/// The fewest bytes of device memory the CUDA executor's KV pools need on a
/// device of shape `shape`: the node's own derivation (`cuda::derive_kv_blocks` in `eidola-server-engine`)
/// at its smallest, for a model of shape `shape`. With `B` = `KV_BLOCK_SIZE`,
/// `R` = ⌈([`Shape::sliding_window`] − 1) / `B`⌉ + 1, and `P` =
/// [`RETENTION_POINTS`] with the prefix cache (else 0):
///
/// - the **sliding** pool: 1 + `MAX_SEQS` × ((1 + `P`) × `R` + 1) +
///   ⌈`MAX_BATCHED_TOKENS` / `B`⌉ blocks;
/// - the **global** pool: one `MAX_MODEL_LEN` sequence and the null block,
///   ⌈`MAX_MODEL_LEN` / `B`⌉ + 1 blocks;
/// - with `DRAFT_TOKENS` `k` > 0, the **drafter** pool (the MTP depths' KV,
///   a sliding window of the same width): as many blocks as the sliding
///   pool;
///
/// each pool one pad block more (where a replayed decode graph's padding rows
/// write), each block `B` positions of [`Shape::sliding_kv_bytes_per_position`],
/// [`Shape::global_kv_bytes_per_position`] or `k` ×
/// [`Shape::drafter_kv_bytes_per_position_per_depth`] plus
/// [`Shape::drafter_kv_bytes_per_position`], a drafter block also `k` ×
/// [`Shape::drafter_tap_bytes_per_depth`] of boundary tap. A device that holds
/// no drafter has drafter bytes of 0. Saturates rather than wraps.
///
/// A shape with [`Shape::uniform_kv_groups`] (DeepSeek-V4.1's) follows its
/// executor's rule instead (`cuda::derive_deepseek_kv_blocks` in
/// `eidola-server-engine`): every window group of every device holds
/// [`uniform_kv_window_blocks`] blocks, and every bank group one count, which
/// must hold one `MAX_MODEL_LEN` sequence and the null block, so a device
/// needs ⌈`MAX_MODEL_LEN` / `B`⌉ + 1 blocks of [`uniform_kv_bank_block_bytes`]
/// beside [`uniform_kv_window_blocks`] of [`uniform_kv_window_block_bytes`].
/// No pool holds a pad block.
///
/// A shape with a [`Shape::recurrent`] state (Qwen3.8's) follows its node's
/// rule (`cuda::derive_qwen_pools` in `eidola-server-engine`): the state pool
/// holds [`recurrent_state_buffers`] images of
/// [`HybridDims::state_image_bytes`], and every KV group one count of blocks,
/// which must hold one `MAX_MODEL_LEN` sequence and the null block, so the
/// device needs ⌈`MAX_MODEL_LEN` / `B`⌉ + 1 blocks of
/// [`recurrent_kv_block_bytes`] and each pool's pad block (where a decode
/// rung's padding rows write) beside the state pool.
pub fn cuda_kv_min_bytes(shape: &Shape, sizing: &Sizing, cache: &CacheConfig) -> u64 {
    let b = u64::from(sizing.kv_block_size.max(1));
    if let Some(dims) = &shape.recurrent {
        return (u64::from(sizing.max_model_len).div_ceil(b) + 2)
            .saturating_mul(recurrent_kv_block_bytes(dims, sizing))
            .saturating_add(
                recurrent_state_buffers(sizing, cache).saturating_mul(dims.state_image_bytes()),
            );
    }
    if !shape.uniform_kv_groups.is_empty() {
        return (u64::from(sizing.max_model_len).div_ceil(b) + 1)
            .saturating_mul(uniform_kv_bank_block_bytes(shape, sizing))
            .saturating_add(
                uniform_kv_window_blocks(shape, sizing, cache)
                    .saturating_mul(uniform_kv_window_block_bytes(shape, sizing)),
            );
    }
    let global_blocks = u64::from(sizing.max_model_len).div_ceil(b) + 1;
    global_blocks
        .saturating_add(1)
        .saturating_mul(global_block_bytes(shape, sizing))
        .saturating_add(cuda_windowed_pools_bytes(shape, sizing, cache))
}

/// Bytes of one block of every group of `shape`'s [`Shape::uniform_kv_groups`]
/// together, the drafter's only when drafting: per group `layers` ×
/// (`KV_BLOCK_SIZE` / `ratio`) entries of `entry_bytes`, padded to 16 bytes.
/// 0 for a shape without uniform groups. Saturates rather than wraps.
pub fn uniform_kv_block_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    uniform_kv_bank_block_bytes(shape, sizing)
        .saturating_add(uniform_kv_window_block_bytes(shape, sizing))
}

/// [`uniform_kv_block_bytes`] of the bank groups alone.
pub fn uniform_kv_bank_block_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    uniform_group_block_bytes(shape, sizing, false)
}

/// [`uniform_kv_block_bytes`] of the window groups alone.
pub fn uniform_kv_window_block_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    uniform_group_block_bytes(shape, sizing, true)
}

/// Blocks of every window group of a shape with [`Shape::uniform_kv_groups`],
/// the null block among them: with `B` = `KV_BLOCK_SIZE`, `W` =
/// [`Shape::sliding_window`] (the most positions before a row's own that a
/// window group holds for it: the drafter's context), `R` = ⌈`W` / `B`⌉ + 1
/// and `P` = [`RETENTION_POINTS`] with the prefix cache (else 0),
/// 1 + `MAX_SEQS` × ((1 + `P`) × `R` + 1) + ⌈`MAX_BATCHED_TOKENS` / `B`⌉:
/// per seat its live window, the block after it and `P` retained windows, and
/// the positions one step adds. Saturates rather than wraps.
pub fn uniform_kv_window_blocks(shape: &Shape, sizing: &Sizing, cache: &CacheConfig) -> u64 {
    let b = u64::from(sizing.kv_block_size.max(1));
    let points = if cache.enabled { RETENTION_POINTS } else { 0 };
    let window_blocks = shape.sliding_window.div_ceil(b) + 1;
    u64::from(sizing.max_seqs)
        .saturating_mul((1 + points).saturating_mul(window_blocks).saturating_add(1))
        .saturating_add(1 + u64::from(sizing.max_batched_tokens).div_ceil(b))
}

fn uniform_group_block_bytes(shape: &Shape, sizing: &Sizing, window: bool) -> u64 {
    let b = u64::from(sizing.kv_block_size);
    shape
        .uniform_kv_groups
        .iter()
        .filter(|g| g.window == window && (sizing.draft_tokens > 0 || !g.drafter))
        .fold(0u64, |sum, g| {
            let image = g
                .layers
                .saturating_mul(b / g.ratio.max(1))
                .saturating_mul(g.entry_bytes);
            sum.saturating_add(image.saturating_add(15) / 16 * 16)
        })
}

/// Blocks of the CUDA executor's sliding pool, and of its drafter pool when
/// drafting (the pad block not among them): [`cuda_kv_min_bytes`]' count.
fn sliding_pool_blocks(shape: &Shape, sizing: &Sizing, cache: &CacheConfig) -> u64 {
    let b = u64::from(sizing.kv_block_size.max(1));
    let points = if cache.enabled { RETENTION_POINTS } else { 0 };
    let window_blocks = (shape.sliding_window - 1).div_ceil(b) + 1;
    u64::from(sizing.max_seqs)
        .saturating_mul((1 + points).saturating_mul(window_blocks).saturating_add(1))
        .saturating_add(1 + u64::from(sizing.max_batched_tokens).div_ceil(b))
}

/// Bytes of one block of the global pool.
fn global_block_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    u64::from(sizing.kv_block_size.max(1)).saturating_mul(shape.global_kv_bytes_per_position)
}

/// Bytes of one block of the sliding pool.
fn sliding_block_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    u64::from(sizing.kv_block_size.max(1)).saturating_mul(shape.sliding_kv_bytes_per_position)
}

/// Bytes of one block of the drafter pool (its boundary tap included); 0
/// without drafting, when there is no such pool.
fn drafter_block_bytes(shape: &Shape, sizing: &Sizing) -> u64 {
    let k = u64::from(sizing.draft_tokens);
    let fixed = if k > 0 {
        shape.drafter_kv_bytes_per_position
    } else {
        0
    };
    u64::from(sizing.kv_block_size.max(1))
        .saturating_mul((k * shape.drafter_kv_bytes_per_position_per_depth).saturating_add(fixed))
        .saturating_add(k * shape.drafter_tap_bytes_per_depth)
}

/// Device bytes of the sliding pool and, when drafting, the drafter pool,
/// each with its pad block: what the global pool does not get.
fn cuda_windowed_pools_bytes(shape: &Shape, sizing: &Sizing, cache: &CacheConfig) -> u64 {
    sliding_pool_blocks(shape, sizing, cache)
        .saturating_add(1)
        .saturating_mul(
            sliding_block_bytes(shape, sizing).saturating_add(drafter_block_bytes(shape, sizing)),
        )
}

/// The fewest bytes of host memory a non-zero host tier needs with the CUDA
/// executor: the node's own derivation (`cuda::derive_host_blocks` in
/// `eidola-server-engine`) at its smallest, for the device pools
/// `kv_device_bytes` gives (at least [`cuda_kv_min_bytes`]).
///
/// The node scales the device pools' mix to the tier's budget `H`: a pool of
/// `n` device blocks gets ⌊`n` × `H` / `D`⌋ host blocks, `D` the device
/// pools' bytes (pad blocks not among them), and every pool needs two (the
/// null block and one to hold an entry). The device pools are the sliding
/// pool and, when drafting, the drafter pool at [`cuda_kv_min_bytes`]'
/// count, and a global pool of what is left of `kv_device_bytes` after them,
/// less its pad block, at most `i32::MAX` blocks. So the least `H` is
/// ⌈2 × `D` / the smallest pool's blocks⌉. Saturates rather than wraps.
///
/// A shape with [`Shape::uniform_kv_groups`] follows its executor's rule
/// (`cuda::derive_deepseek_host_blocks`): the tier holds one count of blocks
/// for every group of every device, at least two, so this device's share of
/// the least tier is two blocks of [`uniform_kv_block_bytes`], whatever
/// `kv_device_bytes`; [`check_device_resources`] sums the devices' shares.
///
/// A shape with a [`Shape::recurrent`] state scales the same way over its two
/// pools (`cuda::derive_qwen_host_pools`): the KV blocks `kv_device_bytes`
/// leaves beside the state pool, less the pad block (at most `i32::MAX`), of
/// [`recurrent_kv_block_bytes`], and the [`recurrent_state_buffers`] state
/// images, each needing two on the host.
pub fn cuda_kv_host_min_bytes(
    shape: &Shape,
    sizing: &Sizing,
    cache: &CacheConfig,
    kv_device_bytes: u64,
) -> u64 {
    if !shape.uniform_kv_groups.is_empty() {
        return uniform_kv_block_bytes(shape, sizing).saturating_mul(2);
    }
    if let Some(dims) = &shape.recurrent {
        let block = recurrent_kv_block_bytes(dims, sizing).max(1);
        let states = recurrent_state_buffers(sizing, cache);
        let state_bytes = states.saturating_mul(dims.state_image_bytes());
        let blocks = (kv_device_bytes.saturating_sub(state_bytes) / block)
            .saturating_sub(1)
            .min(i32::MAX as u64);
        let device = u128::from(blocks) * u128::from(block) + u128::from(state_bytes);
        let smallest = u128::from(blocks.min(states).max(1));
        return u64::try_from((2 * device).div_ceil(smallest)).unwrap_or(u64::MAX);
    }
    let windowed = sliding_pool_blocks(shape, sizing, cache);
    let global_bytes = global_block_bytes(shape, sizing).max(1);
    let global = (kv_device_bytes.saturating_sub(cuda_windowed_pools_bytes(shape, sizing, cache))
        / global_bytes)
        .saturating_sub(1)
        .min(i32::MAX as u64);
    let device = u128::from(global) * u128::from(global_bytes)
        + u128::from(windowed)
            * (u128::from(sliding_block_bytes(shape, sizing))
                + u128::from(drafter_block_bytes(shape, sizing)));
    let smallest = u128::from(global.min(windowed).max(1));
    u64::try_from((2 * device).div_ceil(smallest)).unwrap_or(u64::MAX)
}

/// Whether the deployment's VM and GPUs hold what its configuration
/// allocates, as far as configuration alone fixes it. Host memory (`memory`,
/// MiB) holds [`host_memory_bytes`], the prefix cache's host tier
/// (`kv_host_bytes`, `KV_HOST_BYTES`: page-locked, so it can never be paged
/// out to make room), plus, for the `cpu` executor, its KV cache
/// (`KV_BLOCKS` × `KV_BLOCK_SIZE` × [`Shape::kv_bytes_per_position`]). For the
/// `cuda` executor, which runs on one [`SUPPORTED_GPU`], the VM has a GPU,
/// `KV_DEVICE_BYTES` holds at least [`cuda_kv_min_bytes`], a non-zero
/// `KV_HOST_BYTES` at least [`cuda_kv_host_min_bytes`], and
/// `KV_DEVICE_BYTES` + `weights_bytes` + [`cuda_device_reserve_bytes`] fit
/// [`SUPPORTED_GPU_MEMORY_BYTES`]. `weights_bytes` bounds the weights' device
/// footprint from above: the pack's data size ([`artifact_data_bytes`]),
/// every file of it, whether the node loads it or not.
#[allow(clippy::too_many_arguments)]
pub fn check_resources(
    shape: &Shape,
    sizing: &Sizing,
    cache: &CacheConfig,
    executor: &ExecutorSettings,
    kv_host_bytes: u64,
    weights_bytes: u64,
    memory_mib: u64,
    gpus: u64,
) -> Result<(), String> {
    let host_kv = match executor {
        ExecutorSettings::Cpu { kv_blocks } => u64::from(*kv_blocks)
            .saturating_mul(u64::from(sizing.kv_block_size))
            .saturating_mul(shape.kv_bytes_per_position())
            .saturating_add(shape.recurrent.as_ref().map_or(0, |dims| {
                recurrent_state_buffers(sizing, cache).saturating_mul(dims.state_image_bytes())
            })),
        ExecutorSettings::Cuda { .. } => 0,
    };
    let host = host_memory_bytes(sizing)
        .saturating_add(host_kv)
        .saturating_add(kv_host_bytes);
    let host_limit = memory_mib.saturating_mul(1 << 20);
    if host > host_limit {
        return Err(format!(
            "the node's request pools, block tables, host KV, host tier ({}) and headroom \
             need {host} bytes, more than the VM's {host_limit} (memory)",
            env::KV_HOST_BYTES
        ));
    }
    if let ExecutorSettings::Cuda {
        kv_device_bytes, ..
    } = executor
    {
        if gpus == 0 {
            return Err("the cuda executor needs a GPU, and the VM has none".into());
        }
        let needed = cuda_kv_min_bytes(shape, sizing, cache);
        if *kv_device_bytes < needed {
            return Err(format!(
                "{} ({kv_device_bytes}) is less than the {needed} bytes the KV pools need at \
                 their smallest (one {} sequence, every seat's sliding window, pad blocks)",
                env::KV_DEVICE_BYTES,
                env::MAX_MODEL_LEN
            ));
        }
        let host_needed = cuda_kv_host_min_bytes(shape, sizing, cache, *kv_device_bytes);
        if kv_host_bytes > 0 && kv_host_bytes < host_needed {
            return Err(format!(
                "{} ({kv_host_bytes}) is less than the {host_needed} bytes that give every KV \
                 pool two host blocks; 0 is no host tier",
                env::KV_HOST_BYTES
            ));
        }
        let reserve = cuda_device_reserve_bytes(shape, sizing);
        let device = kv_device_bytes
            .saturating_add(weights_bytes)
            .saturating_add(reserve);
        if device > SUPPORTED_GPU_MEMORY_BYTES {
            return Err(format!(
                "{} ({kv_device_bytes}), the weights ({weights_bytes}) and the executor's \
                 reserve ({reserve}) need {device} bytes of device memory, more than the \
                 {SUPPORTED_GPU}'s {SUPPORTED_GPU_MEMORY_BYTES}",
                env::KV_DEVICE_BYTES
            ));
        }
    }
    Ok(())
}

/// Whether a deployment of an architecture split across devices (`devices`,
/// one [`Shape`] per device, [`crate::architecture::Architecture::devices`])
/// fits its VM and GPUs, as far as configuration alone fixes it: the rule of
/// [`check_resources`] on every device, with each device's own weights.
///
/// - The `cuda` executor runs device `d` on GPU `d`, so the VM attaches at
///   least as many GPUs as there are devices, and `KV_DEVICE_BYTES` names one
///   budget per device, in device order.
/// - On device `d`: its budget holds at least [`cuda_kv_min_bytes`] of its
///   shape, and the budget, [`Shape::weights_bytes`] and
///   [`cuda_device_reserve_bytes`] of its shape fit
///   [`SUPPORTED_GPU_MEMORY_BYTES`].
/// - The pack's data size (`weights_bytes`, [`artifact_data_bytes`]) is at
///   most the devices' weights together, so the per-device bounds are bounds
///   on the weights the deployment pins.
/// - Host memory holds [`host_memory_bytes`] and the host tier, and a
///   non-zero tier holds at least the sum over devices of
///   [`cuda_kv_host_min_bytes`] (every pool of every device two host blocks).
///
/// The `cpu` executor runs every device in host memory: the VM holds its KV
/// pools (`KV_BLOCKS` × `KV_BLOCK_SIZE` × each device's
/// [`Shape::kv_bytes_per_position`] and drafter bytes), and the weights,
/// which it does not bound here.
///
/// There is no setting for running a device's experts or Engram tables out
/// of host memory: a deployment places every tensor on its device.
#[allow(clippy::too_many_arguments)]
pub fn check_device_resources(
    devices: &[Shape],
    sizing: &Sizing,
    cache: &CacheConfig,
    executor: &ExecutorSettings,
    kv_host_bytes: u64,
    weights_bytes: u64,
    memory_mib: u64,
    gpus: u64,
) -> Result<(), String> {
    let host_kv = match executor {
        ExecutorSettings::Cpu { kv_blocks } => devices.iter().fold(0u64, |sum, d| {
            let per_position =
                d.kv_bytes_per_position()
                    .saturating_add(if sizing.draft_tokens > 0 {
                        d.drafter_kv_bytes_per_position
                    } else {
                        0
                    });
            sum.saturating_add(
                u64::from(*kv_blocks)
                    .saturating_mul(u64::from(sizing.kv_block_size))
                    .saturating_mul(per_position),
            )
        }),
        ExecutorSettings::Cuda { .. } => 0,
    };
    let host = host_memory_bytes(sizing)
        .saturating_add(host_kv)
        .saturating_add(kv_host_bytes);
    let host_limit = memory_mib.saturating_mul(1 << 20);
    if host > host_limit {
        return Err(format!(
            "the node's request pools, block tables, host KV, host tier ({}) and headroom \
             need {host} bytes, more than the VM's {host_limit} (memory)",
            env::KV_HOST_BYTES
        ));
    }
    let ExecutorSettings::Cuda { .. } = executor else {
        return Ok(());
    };
    let n = devices.len();
    if gpus < n as u64 {
        return Err(format!(
            "the model runs on {n} GPUs, one per device, and the VM attaches {gpus}"
        ));
    }
    let budgets = executor.kv_device_budgets();
    if budgets.len() != n {
        return Err(format!(
            "{} names {} budgets; the model runs on {n} devices, one budget each, in device \
             order",
            env::KV_DEVICE_BYTES,
            budgets.len()
        ));
    }
    let mut device_weights = 0u64;
    let mut host_needed = 0u64;
    for (d, (shape, &budget)) in devices.iter().zip(&budgets).enumerate() {
        let weights = shape
            .weights_bytes
            .ok_or_else(|| format!("device {d} states no weights bound"))?;
        device_weights = device_weights.saturating_add(weights);
        // A block of a pooled group holds whole entries (the executor refuses
        // any other block size).
        if let Some(g) = shape
            .uniform_kv_groups
            .iter()
            .find(|g| !u64::from(sizing.kv_block_size).is_multiple_of(g.ratio.max(1)))
        {
            return Err(format!(
                "device {d}: {} ({}) is not a multiple of the pooling ratio {}",
                env::KV_BLOCK_SIZE,
                sizing.kv_block_size,
                g.ratio
            ));
        }
        let needed = cuda_kv_min_bytes(shape, sizing, cache);
        if budget < needed {
            return Err(format!(
                "device {d}: its {} budget ({budget}) is less than the {needed} bytes its KV \
                 pools need at their smallest (one {} sequence, every seat's sliding window, \
                 pad blocks)",
                env::KV_DEVICE_BYTES,
                env::MAX_MODEL_LEN
            ));
        }
        host_needed =
            host_needed.saturating_add(cuda_kv_host_min_bytes(shape, sizing, cache, budget));
        let reserve = cuda_device_reserve_bytes(shape, sizing);
        // What the executor loads: the drafter's weights only when it drafts.
        let loaded = device_weights_bytes(shape, sizing).unwrap_or(weights);
        let device = budget.saturating_add(loaded).saturating_add(reserve);
        if device > SUPPORTED_GPU_MEMORY_BYTES {
            return Err(format!(
                "device {d}: its {} budget ({budget}), its weights ({loaded}) and the \
                 executor's reserve ({reserve}) need {device} bytes of device memory, more \
                 than the {SUPPORTED_GPU}'s {SUPPORTED_GPU_MEMORY_BYTES}",
                env::KV_DEVICE_BYTES
            ));
        }
    }
    if weights_bytes > device_weights {
        return Err(format!(
            "the weights pack holds {weights_bytes} bytes, more than the {device_weights} the \
             devices' weights bounds allow together"
        ));
    }
    if kv_host_bytes > 0 && kv_host_bytes < host_needed {
        return Err(format!(
            "{} ({kv_host_bytes}) is less than the {host_needed} bytes that give every KV pool \
             of every device two host blocks; 0 is no host tier",
            env::KV_HOST_BYTES
        ));
    }
    Ok(())
}

/// Fewest vCPUs a deployment may give its VM.
pub const MIN_CPUS: u64 = 1;
/// Most vCPUs a deployment may give its VM.
pub const MAX_CPUS: u64 = 256;
/// Smallest VM memory, in MiB.
pub const MIN_MEMORY_MIB: u64 = 8192;
/// Largest VM memory, in MiB.
pub const MAX_MEMORY_MIB: u64 = 2 * 1024 * 1024;

/// Whether the VM's `cpus` and `memory` (MiB) are a shape worth pinning:
/// `cpus` from [`MIN_CPUS`] to [`MAX_CPUS`], and `memory` a whole number of
/// GiB from [`MIN_MEMORY_MIB`] (8 GiB) to [`MAX_MEMORY_MIB`] (2 TiB). The
/// platform's production GPU shapes include sizes that are not powers of two
/// (768 GiB), so the rule asks only for whole GiB. These bounds are ours: they
/// keep an obviously unlaunchable shape from being pinned, and are not a
/// security property, since `memory` is part of the measured configuration
/// either way. The same rule holds for every `cvm-version` an engine pins.
pub fn check_vm_resources(cpus: u64, memory_mib: u64) -> Result<(), String> {
    if !(MIN_CPUS..=MAX_CPUS).contains(&cpus) {
        return Err(format!(
            "cpus must be from {MIN_CPUS} to {MAX_CPUS} (got {cpus})"
        ));
    }
    if !memory_mib.is_multiple_of(1024) || !(MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&memory_mib)
    {
        return Err(format!(
            "memory must be a whole number of GiB from {MIN_MEMORY_MIB} to {MAX_MEMORY_MIB} MiB (got {memory_mib})"
        ));
    }
    Ok(())
}

/// Which executor the configuration selects. Whether a given node build can
/// run it is the node's concern (a `cuda` node build), not the grammar's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Executor {
    /// The f32 reference executor.
    Cpu,
    /// The CUDA executor.
    Cuda,
}

/// What backs the weights directory, as the measured configuration declares
/// it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightsStorage {
    /// Production: every weights file on read-only, kernel-verified storage
    /// (checked by the node before any weights file is opened).
    VerifiedReadonly,
    /// Development: any filesystem, reported as such by the node.
    DevWritable,
}

impl WeightsStorage {
    /// The configuration spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            WeightsStorage::VerifiedReadonly => "verified-readonly",
            WeightsStorage::DevWritable => "dev-writable",
        }
    }
}

/// KV memory and scheduler sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sizing {
    pub kv_block_size: u32,
    pub max_model_len: u32,
    pub max_seqs: u32,
    pub max_batched_tokens: u32,
    pub max_prefill_chunk: u32,
    pub draft_tokens: u32,
    pub max_requests: u32,
}

/// Prefix-cache lifetime policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheConfig {
    pub enabled: bool,
    pub idle_ttl_secs: u64,
    pub max_age_secs: u64,
}

/// Whether the CUDA executor replays captured graphs for decode steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaGraphs {
    /// Every step runs eagerly.
    Off,
    /// Pure-decode steps replay graphs captured at boot.
    On,
}

/// Whether the CUDA executor's node requires its GPUs in confidential-computing
/// mode before it opens one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuCc {
    /// Production: the node refuses to open a GPU unless the driver reports
    /// confidential computing on, DevTools mode off, the production
    /// environment, and the GPUs accepting work. Every pinned deployment sets
    /// this.
    Required,
    /// Development, on a GPU without confidential computing: the node skips the
    /// check. A gateway never pins a deployment that sets this.
    NotRequired,
}

impl GpuCc {
    /// The configuration spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            GpuCc::Required => "required",
            GpuCc::NotRequired => "not-required",
        }
    }
}

/// The executor and the settings only it takes. A setting of the other
/// executor is refused rather than ignored, so the measured configuration
/// never shows a value that does nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutorSettings {
    Cpu {
        /// Physical blocks per KV group, including the null block.
        kv_blocks: u32,
    },
    Cuda {
        /// The kernel build output: an absolute, clean path.
        kernels_dir: String,
        /// Device memory for the KV pools of device 0, in bytes.
        kv_device_bytes: u64,
        /// The KV pools' budgets of devices 1 onwards, in device order: empty
        /// for a model on one device.
        further_kv_device_bytes: Vec<u64>,
        graphs: CudaGraphs,
        /// Whether the GPUs must be in confidential-computing mode.
        gpu_cc: GpuCc,
    },
}

impl ExecutorSettings {
    /// Which executor these settings are for.
    pub fn kind(&self) -> Executor {
        match self {
            ExecutorSettings::Cpu { .. } => Executor::Cpu,
            ExecutorSettings::Cuda { .. } => Executor::Cuda,
        }
    }

    /// The KV budget of every device, in device order (`cuda`); empty for the
    /// `cpu` executor.
    pub fn kv_device_budgets(&self) -> Vec<u64> {
        match self {
            ExecutorSettings::Cpu { .. } => Vec::new(),
            ExecutorSettings::Cuda {
                kv_device_bytes,
                further_kv_device_bytes,
                ..
            } => core::iter::once(*kv_device_bytes)
                .chain(further_kv_device_bytes.iter().copied())
                .collect(),
        }
    }
}

/// Whether `path` is absolute and clean: no empty, `.` or `..` component,
/// and not `/` itself.
pub fn is_clean_absolute_path(path: &str) -> bool {
    path.strip_prefix('/').is_some_and(|rest| {
        !rest.is_empty()
            && rest
                .split('/')
                .all(|c| !c.is_empty() && c != "." && c != "..")
    })
}

/// A node's measured configuration: every variable of its environment except
/// the secret gateway token, parsed and validated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasuredConfig {
    pub model_id: String,
    pub weights_dir: String,
    /// Lowercase hex.
    pub weights_sha256: String,
    pub weights_storage: WeightsStorage,
    /// A validated Argon2id PHC string.
    pub gateway_token_hash: String,
    pub executor: ExecutorSettings,
    pub bind_addr: core::net::SocketAddr,
    pub sizing: Sizing,
    pub cache: CacheConfig,
    /// Host memory for the prefix cache's host tier, in bytes (0: none).
    pub kv_host_bytes: u64,
}

/// A refused configuration. Names the variable and the problem, never its
/// value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl core::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "configuration refused: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Parse a node's measured configuration through `lookup` (its environment,
/// or a deployment config's env list). **This is the node's boot grammar**:
/// the node calls it and then verifies its secret token against the hash, so
/// a configuration this accepts is one the node boots with, up to checks
/// against its weights (model length, MTP depth, executor availability) that
/// need the model itself. The gateway's build check runs every pinned
/// deployment's whole env through it.
#[cfg(feature = "argon2")]
pub fn parse_measured(
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<MeasuredConfig, ConfigError> {
    let get = |name: &str| -> Result<String, ConfigError> {
        lookup(name)
            .filter(|v| !v.is_empty())
            .ok_or_else(|| ConfigError(format!("{name} is not set")))
    };
    let positive = |name: &str| -> Result<u32, ConfigError> {
        match get(name)?.parse::<u32>() {
            Ok(n) if n > 0 => Ok(n),
            _ => Err(ConfigError(format!("{name} must be a positive integer"))),
        }
    };
    let non_negative = |name: &str| -> Result<u32, ConfigError> {
        get(name)?
            .parse::<u32>()
            .map_err(|_| ConfigError(format!("{name} must be a non-negative integer")))
    };
    let seconds = |name: &str| -> Result<u64, ConfigError> {
        parse_cache_seconds(&get(name)?)
            .ok_or_else(|| ConfigError(format!("{name} must be a positive number of seconds")))
    };

    let model_id = get(env::MODEL_ID)?;
    if model_id
        .chars()
        .any(|c| c.is_control() || c.is_whitespace())
    {
        return Err(ConfigError(format!(
            "{} must not contain whitespace or control characters",
            env::MODEL_ID
        )));
    }
    let weights_dir = get(env::WEIGHTS_DIR)?;
    let weights_sha256 = get(env::WEIGHTS_SHA256)?.to_ascii_lowercase();
    if !(weights_sha256.len() == 64
        && weights_sha256
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
    {
        return Err(ConfigError(format!(
            "{} must be 64 hexadecimal digits",
            env::WEIGHTS_SHA256
        )));
    }
    let weights_storage = match get(env::WEIGHTS_STORAGE)?.as_str() {
        "verified-readonly" => WeightsStorage::VerifiedReadonly,
        "dev-writable" => WeightsStorage::DevWritable,
        _ => {
            return Err(ConfigError(format!(
                "{} must be `verified-readonly` or `dev-writable`",
                env::WEIGHTS_STORAGE
            )));
        }
    };
    let gateway_token_hash = get(env::GATEWAY_TOKEN_HASH)?;
    parse_gateway_token_hash(&gateway_token_hash)
        .map_err(|e| ConfigError(format!("{}: {e}", env::GATEWAY_TOKEN_HASH)))?;
    let kind = match get(env::EXECUTOR)?.as_str() {
        "cpu" => Executor::Cpu,
        "cuda" => Executor::Cuda,
        _ => {
            return Err(ConfigError(format!(
                "{}: expected `cpu` or `cuda`",
                env::EXECUTOR
            )));
        }
    };
    // Each executor's own settings are required with it and refused without
    // it.
    let refuse_unless = |name: &str, executor: &str| -> Result<(), ConfigError> {
        match lookup(name) {
            Some(v) if !v.is_empty() => Err(ConfigError(format!(
                "{name} applies only to the {executor} executor"
            ))),
            _ => Ok(()),
        }
    };
    let executor = match kind {
        Executor::Cpu => {
            refuse_unless(env::KV_DEVICE_BYTES, "cuda")?;
            refuse_unless(env::KERNELS_DIR, "cuda")?;
            refuse_unless(env::CUDA_GRAPHS, "cuda")?;
            refuse_unless(env::GPU_CC, "cuda")?;
            let kv_blocks = positive(env::KV_BLOCKS)?;
            if kv_blocks < 2 {
                return Err(ConfigError(format!(
                    "{} must be at least 2 (block 0 is reserved)",
                    env::KV_BLOCKS
                )));
            }
            if kv_blocks > caps::KV_BLOCKS {
                return Err(ConfigError(format!(
                    "{} must be at most {}",
                    env::KV_BLOCKS,
                    caps::KV_BLOCKS
                )));
            }
            ExecutorSettings::Cpu { kv_blocks }
        }
        Executor::Cuda => {
            refuse_unless(env::KV_BLOCKS, "cpu")?;
            let mut budgets = parse_kv_device_bytes(&get(env::KV_DEVICE_BYTES)?)?.into_iter();
            let kv_device_bytes = budgets.next().expect("at least one budget");
            let further_kv_device_bytes: Vec<u64> = budgets.collect();
            let kernels_dir = get(env::KERNELS_DIR)?;
            if !is_clean_absolute_path(&kernels_dir) {
                return Err(ConfigError(format!(
                    "{} must be an absolute path with no empty, `.` or `..` component",
                    env::KERNELS_DIR
                )));
            }
            let graphs = match get(env::CUDA_GRAPHS)?.as_str() {
                "on" => CudaGraphs::On,
                "off" => CudaGraphs::Off,
                _ => {
                    return Err(ConfigError(format!(
                        "{}: expected `on` or `off`",
                        env::CUDA_GRAPHS
                    )));
                }
            };
            let gpu_cc = match get(env::GPU_CC)?.as_str() {
                "required" => GpuCc::Required,
                "not-required" => GpuCc::NotRequired,
                _ => {
                    return Err(ConfigError(format!(
                        "{}: expected `required` or `not-required`",
                        env::GPU_CC
                    )));
                }
            };
            ExecutorSettings::Cuda {
                kernels_dir,
                kv_device_bytes,
                further_kv_device_bytes,
                graphs,
                gpu_cc,
            }
        }
    };
    let bind_addr = get(env::BIND_ADDR)?
        .parse::<core::net::SocketAddr>()
        .map_err(|_| ConfigError(format!("{} must be host:port", env::BIND_ADDR)))?;

    let sizing = Sizing {
        kv_block_size: positive(env::KV_BLOCK_SIZE)?,
        max_model_len: positive(env::MAX_MODEL_LEN)?,
        max_seqs: positive(env::MAX_SEQS)?,
        max_batched_tokens: positive(env::MAX_BATCHED_TOKENS)?,
        max_prefill_chunk: positive(env::MAX_PREFILL_CHUNK)?,
        draft_tokens: non_negative(env::DRAFT_TOKENS)?,
        max_requests: positive(env::MAX_REQUESTS)?,
    };
    // The engine's boot refusals that depend only on these values (a zero
    // seat, prefill chunk or block count) are the positivity rules above; the
    // rest need the model (its position limit and MTP depth) or the node
    // build (its executor).
    // Every allocation-driving value is capped, so a configuration cannot ask
    // the node for an allocation it cannot make (a table of `MAX_SEQS` slots,
    // a scratch of `MAX_BATCHED_TOKENS` rows): refused here, before any
    // weight is loaded. The caps are in [`caps`], each with its reason.
    for (name, value, cap) in [
        (
            env::KV_BLOCK_SIZE,
            sizing.kv_block_size,
            caps::KV_BLOCK_SIZE,
        ),
        (
            env::MAX_MODEL_LEN,
            sizing.max_model_len,
            caps::MAX_MODEL_LEN,
        ),
        (env::MAX_SEQS, sizing.max_seqs, caps::MAX_SEQS),
        (
            env::MAX_BATCHED_TOKENS,
            sizing.max_batched_tokens,
            caps::MAX_BATCHED_TOKENS,
        ),
        (
            env::MAX_PREFILL_CHUNK,
            sizing.max_prefill_chunk,
            caps::MAX_PREFILL_CHUNK,
        ),
        (env::DRAFT_TOKENS, sizing.draft_tokens, caps::DRAFT_TOKENS),
        (env::MAX_REQUESTS, sizing.max_requests, caps::MAX_REQUESTS),
    ] {
        if value > cap {
            return Err(ConfigError(format!("{name} must be at most {cap}")));
        }
    }
    // A decode row costs `1 + draft_tokens` query slots, and the scheduler
    // leaves a row a step cannot afford out of that step: with fewer slots
    // than every seat's decode row needs, the newest seated sequences would
    // hold their seats and KV without advancing while older ones decode. It
    // also covers the scheduler's own boot refusal (`Engine::new`: a step
    // holds one decode row).
    if u64::from(sizing.max_batched_tokens) < decode_step_tokens(&sizing) {
        return Err(ConfigError(format!(
            "{} must be at least {} x (1 + {}): a step holds every seat's decode row, \
             1 + {} query slots each",
            env::MAX_BATCHED_TOKENS,
            env::MAX_SEQS,
            env::DRAFT_TOKENS,
            env::DRAFT_TOKENS
        )));
    }
    // The CUDA executor drafts with the model's MTP layers, one per depth:
    // no wider than any implemented architecture ships (a deployment is held
    // to its own architecture's layers by `deployment::check_deployment`,
    // and the node to its checkpoint's when it loads).
    if kind == Executor::Cuda && sizing.draft_tokens > caps::CUDA_DRAFT_TOKENS {
        return Err(ConfigError(format!(
            "{} must be at most {} with the cuda executor (the model's MTP layers)",
            env::DRAFT_TOKENS,
            caps::CUDA_DRAFT_TOKENS
        )));
    }

    let enabled = parse_prefix_cache(&get(env::PREFIX_CACHE)?)
        .ok_or_else(|| ConfigError(format!("{} must be `true` or `false`", env::PREFIX_CACHE)))?;
    let cache = CacheConfig {
        enabled,
        idle_ttl_secs: seconds(env::CACHE_IDLE_TTL_SECS)?,
        max_age_secs: seconds(env::CACHE_MAX_AGE_SECS)?,
    };
    if cache.idle_ttl_secs > cache.max_age_secs {
        return Err(ConfigError(format!(
            "{} must not exceed {}",
            env::CACHE_IDLE_TTL_SECS,
            env::CACHE_MAX_AGE_SECS
        )));
    }
    // A client declares a cached prefix in multiples of a quantum, which must
    // land on the block boundaries the prefix cache resumes at.
    if cache.enabled && !crate::engine_protocol::quantum_covers_block_size(sizing.kv_block_size) {
        return Err(ConfigError(format!(
            "{} must divide {} when {} is true",
            env::KV_BLOCK_SIZE,
            crate::engine_protocol::CACHED_PREFIX_QUANTUM,
            env::PREFIX_CACHE
        )));
    }
    let kv_host_bytes = get(env::KV_HOST_BYTES)?.parse::<u64>().map_err(|_| {
        ConfigError(format!(
            "{} must be a non-negative number of bytes",
            env::KV_HOST_BYTES
        ))
    })?;
    if kv_host_bytes > caps::KV_HOST_BYTES {
        return Err(ConfigError(format!(
            "{} must be at most {}",
            env::KV_HOST_BYTES,
            caps::KV_HOST_BYTES
        )));
    }
    // A host tier holds prefix-cache entries: without the cache it would
    // hold nothing, and the measured configuration never shows a value that
    // does nothing.
    if kv_host_bytes > 0 && !cache.enabled {
        return Err(ConfigError(format!(
            "{} must be 0 when {} is false",
            env::KV_HOST_BYTES,
            env::PREFIX_CACHE
        )));
    }

    Ok(MeasuredConfig {
        model_id,
        weights_dir,
        weights_sha256,
        weights_storage,
        gateway_token_hash,
        executor,
        bind_addr,
        sizing,
        cache,
        kv_host_bytes,
    })
}

/// `EIDOLA_ENGINE_KV_DEVICE_BYTES`: one budget per device, comma-separated
/// in device order (a model on one GPU names one), each a positive number of
/// bytes of at most [`caps::KV_DEVICE_BYTES`], at most as many as
/// [`CUDA_GPU_COUNTS`] allows GPUs. How many devices the model runs on is its
/// architecture's ([`check_device_resources`]); the node holds the count to
/// its weights.
pub fn parse_kv_device_bytes(text: &str) -> Result<Vec<u64>, ConfigError> {
    let mut budgets = Vec::new();
    for part in text.split(',') {
        let n = match part.parse::<u64>() {
            Ok(n) if n > 0 => n,
            _ => {
                return Err(ConfigError(format!(
                    "{} must be one positive number of bytes per device, comma-separated",
                    env::KV_DEVICE_BYTES
                )));
            }
        };
        if n > caps::KV_DEVICE_BYTES {
            return Err(ConfigError(format!(
                "{} must be at most {} per device",
                env::KV_DEVICE_BYTES,
                caps::KV_DEVICE_BYTES
            )));
        }
        budgets.push(n);
    }
    if budgets.len() as u64 > *CUDA_GPU_COUNTS.end() {
        return Err(ConfigError(format!(
            "{} names more devices than the {} GPUs a deployment may attach",
            env::KV_DEVICE_BYTES,
            CUDA_GPU_COUNTS.end()
        )));
    }
    Ok(budgets)
}

/// A deployment path component (a deployment's name, a variant):
/// `[a-z0-9][a-z0-9._-]*`. `deploy/engine/<name>/<variant>/` directories
/// are named this way; the gateway's build refuses a pin naming any other,
/// and `measure-enclave` refuses to measure one.
pub fn is_safe_component(s: &str) -> bool {
    let mut bytes = s.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && bytes.all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'.' | b'_' | b'-')
        })
}

/// Whether the engine container's GPU access matches its executor: the
/// `cuda` executor needs `runtime: nvidia` and every attested GPU
/// (`gpus: all`, so the GPUs the shim attests are the ones the node uses);
/// the `cpu` executor needs neither. `gpus` is the container's selection as
/// written (a number reads as its decimal text).
pub fn check_container_gpu_access(
    executor: Executor,
    runtime: Option<&str>,
    gpus: Option<&str>,
) -> Result<(), String> {
    match (executor, runtime, gpus) {
        (Executor::Cuda, Some("nvidia"), Some("all")) => Ok(()),
        (Executor::Cuda, _, _) => {
            Err("the cuda executor's container needs `runtime: nvidia` and `gpus: all`".into())
        }
        (Executor::Cpu, None, None) => Ok(()),
        (Executor::Cpu, _, _) => {
            Err("the cpu executor's container takes no `runtime` or `gpus`".into())
        }
    }
}

/// Shortest gateway token accepted, in bytes.
pub const GATEWAY_TOKEN_MIN_LEN: usize = 16;
/// Longest gateway token accepted, in bytes.
pub const GATEWAY_TOKEN_MAX_LEN: usize = 1024;

/// The shape of the gateway's bearer token (`GATEWAY_TOKEN`): visible ASCII
/// (no space, no control character), [`GATEWAY_TOKEN_MIN_LEN`] to
/// [`GATEWAY_TOKEN_MAX_LEN`] bytes, so it travels in an `Authorization`
/// header unchanged. The gateway refuses to hold any other token and the node
/// refuses to boot with one.
pub fn is_gateway_token(token: &str) -> bool {
    (GATEWAY_TOKEN_MIN_LEN..=GATEWAY_TOKEN_MAX_LEN).contains(&token.len())
        && token.bytes().all(|b| b.is_ascii_graphic())
}

/// The one secret an engine container takes: everything else it reads is in
/// the measured env. A secret is delivered outside the measurement, so any
/// other one (a log level, say) would change the node's behaviour unmeasured.
pub const SECRETS: &[&str] = &[env::GATEWAY_TOKEN];

/// Whether the engine container's `secrets` are exactly [`SECRETS`].
pub fn check_secrets<'a>(secrets: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    let secrets: Vec<&str> = secrets.into_iter().collect();
    if secrets == SECRETS {
        Ok(())
    } else {
        Err(format!(
            "the engine container's secrets must be exactly {SECRETS:?}; any other secret \
             reaches the node outside the measurement"
        ))
    }
}

/// Where Tinfoil mounts a model pack granted to a container: read-only, at
/// `/tinfoil/models/<name>`, in the granted containers only (cvmimage
/// `docs/runtime-policy.md`; `ContainerModelsDir` in its boot paths).
pub const MODEL_MOUNT_ROOT: &str = "/tinfoil/models";

/// Where Tinfoil mounts what the deployment attaches (model packs and the
/// like) inside the container.
pub const TINFOIL_MOUNT_ROOT: &str = "/tinfoil";

/// Whether `kernels_dir` (`EIDOLA_ENGINE_KERNELS_DIR`) names a directory of
/// the measured image: the kernels the CUDA executor loads ship in the image
/// the deployment pins, never in an attached mount.
pub fn check_kernels_dir(kernels_dir: &str) -> Result<(), String> {
    let under_mount = kernels_dir
        .strip_prefix(TINFOIL_MOUNT_ROOT)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'));
    if under_mount || !is_clean_absolute_path(kernels_dir) {
        return Err(format!(
            "{} must be a clean absolute path inside the image, not under {TINFOIL_MOUNT_ROOT}",
            env::KERNELS_DIR
        ));
    }
    Ok(())
}

/// Whether `mpk` is a modelwrap artifact reference, `rootHash_hashOffset_uuid`
/// (tinfoilsh/modelwrap `modelwrap.go` `ParseRef` at e61ae511): a 64-hex
/// root hash, a decimal hash offset that fits a `u64` (`HashOffsetBytes`), and
/// a lowercase UUID (8-4-4-4-12 hex), joined by exactly two underscores.
pub fn is_artifact_ref(mpk: &str) -> bool {
    artifact_data_bytes(mpk).is_some()
}

/// The data size of a modelwrap artifact reference: its `hashOffset`, where
/// the dm-verity hash tree starts, which is the byte length of the EROFS
/// image before it (modelwrap `SPEC.md`: "the MWP data area is exactly the
/// bytes preceding the hash area"). The image is built uncompressed
/// (`wrap.go` runs `mkfs.erofs` without `-z`), so every file in the pack is
/// stored whole inside it, and this bounds the pack's file bytes from above.
/// `None` unless `mpk` is a whole reference ([`is_artifact_ref`]).
pub fn artifact_data_bytes(mpk: &str) -> Option<u64> {
    let hex = |s: &str, n: usize| {
        s.len() == n && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    };
    let parts: Vec<&str> = mpk.split('_').collect();
    let [root, offset, uuid] = parts.as_slice() else {
        return None;
    };
    let uuid_ok = {
        let groups: Vec<&str> = uuid.split('-').collect();
        groups.len() == 5 && groups.iter().zip([8, 4, 4, 4, 12]).all(|(g, n)| hex(g, n))
    };
    let ok = hex(root, 64)
        && !offset.is_empty()
        && offset.bytes().all(|b| b.is_ascii_digit())
        && uuid_ok;
    offset.parse::<u64>().ok().filter(|_| ok)
}

/// A model pack as an engine config declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPack<'a> {
    pub name: &'a str,
    /// `<owner>/<name>@<revision>`.
    pub repo: Option<&'a str>,
    /// The pinned pack: `<root hash>_<size>_<id>`.
    pub mpk: Option<&'a str>,
}

/// Whether the weights the node reads are the verified model pack: exactly
/// one pack, from `<weights_repo>@<weights_revision>` and pinned by its root
/// hash (`mpk`), granted to the engine container (and only it), with the
/// node's weights directory inside that pack's mount.
pub fn check_weights_pack(
    packs: &[ModelPack<'_>],
    granted: &[&str],
    weights_dir: &str,
    weights_repo: &str,
    weights_revision: &str,
) -> Result<(), String> {
    let [pack] = packs else {
        return Err("an engine deployment declares exactly one model pack, its weights".into());
    };
    let name_ok = {
        let mut bytes = pack.name.bytes();
        bytes.next().is_some_and(|b| b.is_ascii_alphanumeric())
            && pack.name.len() <= 128
            && bytes.all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
    };
    if !name_ok {
        return Err(format!(
            "model pack name {:?} is not a valid pack name",
            pack.name
        ));
    }
    let source = format!("{weights_repo}@{weights_revision}");
    if pack.repo != Some(source.as_str()) {
        return Err(format!(
            "the model pack's repo must be {source:?}, the weights' recorded provenance"
        ));
    }
    if !pack.mpk.is_some_and(is_artifact_ref) {
        return Err("the model pack must be pinned by its artifact reference \
             (`mpk: <64 hex root hash>_<offset>_<uuid>`)"
            .into());
    }
    if granted != [pack.name] {
        return Err(format!(
            "the engine container must be granted exactly the weights pack ({:?})",
            pack.name
        ));
    }
    let mount = format!("{MODEL_MOUNT_ROOT}/{}", pack.name);
    let inside = weights_dir == mount
        || weights_dir
            .strip_prefix(&format!("{mount}/"))
            .is_some_and(|rest| {
                rest.split('/')
                    .all(|part| !part.is_empty() && part != "." && part != "..")
            });
    if !inside {
        return Err(format!(
            "EIDOLA_ENGINE_WEIGHTS_DIR must be inside the weights pack's mount, {mount}"
        ));
    }
    Ok(())
}

/// The node routes the shim must expose: the chat route the gateway sends
/// requests to, the info route it checks a node's weights with, and the
/// health route probes read.
pub const NODE_ROUTES: &[&str] = &["/v1/chat/completions", "/v1/engine/info", "/healthz"];

/// Whether a shim path pattern matches a request path, as the shim decides
/// it (cvmimage `tinfoil/cmd/shim/api.go`, `pathMatchesPattern`): an exact
/// path, or a trailing `*` matching on a segment boundary (`/v1/*` matches
/// `/v1/x`; `/v1*` matches `/v1` and `/v1/x`).
pub fn shim_path_matches(pattern: &str, path: &str) -> bool {
    match pattern.strip_suffix('*') {
        None => pattern == path,
        Some(prefix) if prefix.ends_with('/') => path.starts_with(prefix),
        Some(prefix) => {
            path == prefix
                || path
                    .strip_prefix(prefix)
                    .is_some_and(|rest| rest.starts_with('/'))
        }
    }
}

/// Whether the shim's `paths` expose exactly the node's routes: every route
/// in [`NODE_ROUTES`] matched, and every pattern matching at least one of
/// them (a wildcard such as `/*` does). An absent or empty list is refused
/// even though the shim would then forward every path: the exposure should
/// be stated, not defaulted.
pub fn check_shim_paths(paths: &[&str]) -> Result<(), String> {
    if paths.is_empty() {
        return Err("shim.paths must list the node's routes".into());
    }
    for pattern in paths {
        if !pattern.starts_with('/') || !NODE_ROUTES.iter().any(|r| shim_path_matches(pattern, r)) {
            return Err(format!(
                "shim.paths entry {pattern:?} exposes none of the node's routes {NODE_ROUTES:?}"
            ));
        }
    }
    for route in NODE_ROUTES {
        if !paths.iter().any(|p| shim_path_matches(p, route)) {
            return Err(format!("shim.paths must expose {route}"));
        }
    }
    Ok(())
}

/// The GPU counts a `cuda` deployment may attach (the config's top-level
/// `gpus`): one to eight, the range Tinfoil's validator accepts
/// (`tinfoil-config/validation.go`: `gpus must be between 0 and 8`) less the
/// zero that attaches none. The node computes on one GPU per device of its
/// architecture (device `d` on GPU `d`; [`check_device_resources`] holds the
/// count to at least that); any further GPU a host shape attaches is attested
/// like the others and left idle, so the count is the platform's shape to
/// choose, and what protects the deployment is that every attached GPU is
/// attested (below).
pub const CUDA_GPU_COUNTS: core::ops::RangeInclusive<u64> = 1..=8;

/// Whether a deployment's GPUs are attested on every handshake.
///
/// The attestation shim collects exactly the config's top-level `gpus` worth
/// of GPU evidence items (Tinfoil's `shimConfig.ExpectedGPUs = config.GPUs`),
/// and a pin's `expected_gpus` requires exactly that many. So:
///
/// - the `cuda` executor needs GPUs, a count in [`CUDA_GPU_COUNTS`], and a pin
///   requiring all of them, or its accelerators would go unattested (or, with
///   no `gpus`, every handshake would fail for want of evidence);
/// - the `cpu` executor attaches none and requires none. A pinned deployment
///   never takes this branch (the deployment check refuses `cpu` first); it
///   stays so the rule is total over the grammar's executors.
///
/// `gpus` and `expected_gpus` are `None` when absent; `0` reads as absent.
pub fn check_gpu_attestation(
    executor: Executor,
    gpus: Option<u64>,
    expected_gpus: Option<u64>,
) -> Result<(), String> {
    let gpus = gpus.filter(|n| *n > 0);
    let expected = expected_gpus.filter(|n| *n > 0);
    match executor {
        Executor::Cuda => match gpus {
            None => Err("the cuda executor needs the config to attach GPUs (`gpus`)".into()),
            Some(n) if !CUDA_GPU_COUNTS.contains(&n) => Err(format!(
                "`gpus: {n}` is out of range; it must be from {} to {}",
                CUDA_GPU_COUNTS.start(),
                CUDA_GPU_COUNTS.end()
            )),
            Some(n) if expected != Some(n) => Err(format!(
                "the config attaches {n} GPUs, so expected_gpus must be {n} (the shim collects \
                 exactly that many evidence items)"
            )),
            Some(_) => Ok(()),
        },
        Executor::Cpu => {
            if gpus.is_some() || expected.is_some() {
                Err(
                    "the cpu executor attaches no GPUs and requires no GPU evidence \
                     (`gpus` and expected_gpus absent or 0)"
                        .into(),
                )
            } else {
                Ok(())
            }
        }
    }
}

#[cfg(feature = "deployment")]
pub mod deployment;
#[cfg(feature = "deployment")]
mod yaml_events;

/// Largest prefix-cache retention bound, in seconds: the engine core keeps
/// retention in milliseconds as a `u64`.
pub const MAX_CACHE_SECONDS: u64 = u64::MAX / 1000;

/// A prefix-cache retention bound (`EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS`,
/// `EIDOLA_ENGINE_CACHE_MAX_AGE_SECS`): a positive whole number of seconds,
/// at most [`MAX_CACHE_SECONDS`].
pub fn parse_cache_seconds(text: &str) -> Option<u64> {
    text.parse::<u64>()
        .ok()
        .filter(|n| *n > 0 && *n <= MAX_CACHE_SECONDS)
}

/// The prefix-cache switch (`EIDOLA_ENGINE_PREFIX_CACHE`): exactly `true` or
/// `false`.
pub fn parse_prefix_cache(text: &str) -> Option<bool> {
    match text {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// The platform release a config's `cvm-version` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CvmVersion<'a> {
    /// The bare version (`0.15.0`; no leading `v`).
    pub version: &'a str,
    /// The release manifest's SHA-256, lowercase hex, when the config pins it
    /// inline (`0.15.0@sha256:<hex>`).
    pub manifest_sha256: Option<&'a str>,
}

/// Why a `cvm-version` value is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvmVersionError {
    /// The version is empty or starts with `v`.
    NotBare,
    /// Something follows `@` other than `sha256:` and 64 lowercase hex
    /// digits.
    MalformedPin,
}

impl core::fmt::Display for CvmVersionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotBare => "cvm-version must be a bare version such as 0.15.0",
            Self::MalformedPin => {
                "malformed cvm-version pin: expected VERSION@sha256:<64 lowercase hex>"
            }
        })
    }
}

/// Parse a config's `cvm-version`: a bare version, optionally followed by
/// exactly one `@sha256:<64 lowercase hex>` release-manifest pin.
pub fn parse_cvm_version(raw: &str) -> Result<CvmVersion<'_>, CvmVersionError> {
    let (version, manifest_sha256) = match raw.split_once('@') {
        None => (raw, None),
        Some((version, pin)) => {
            let hex = pin
                .strip_prefix("sha256:")
                .filter(|h| {
                    h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                })
                .ok_or(CvmVersionError::MalformedPin)?;
            (version, Some(hex))
        }
    };
    if version.is_empty() || version.starts_with('v') {
        return Err(CvmVersionError::NotBare);
    }
    Ok(CvmVersion {
        version,
        manifest_sha256,
    })
}

/// The build tag that names a platform release's GPU image
/// (`0.15.3-rc.1+gpu`).
pub const GPU_IMAGE_TAG: &str = "gpu";

/// Whether a `cvm-version` names the image its executor needs.
///
/// Each platform release publishes two TDX images with different MRTDs: the
/// CPU image (`X`) and the GPU image (`X+gpu`). Only the GPU image carries the
/// NVIDIA driver, and its measured boot opens the GPUs to work only after it
/// has checked their device IDs and NVIDIA's local verifier has passed for
/// exactly `gpus` of them (cvmimage `tinfoil/cmd/boot/gpu.go`,
/// `gpuattest.go`); the CPU image refuses a config that attaches any
/// (`tinfoil/cmd/boot/gpu_disabled.go`). So the `cuda` executor needs `X+gpu`,
/// and the `cpu` executor, which attaches no GPUs ([`check_gpu_attestation`]),
/// needs the plain `X`, with no build tag.
pub fn check_cvm_image(executor: Executor, release: &CvmVersion<'_>) -> Result<(), String> {
    let tag = release.version.split_once('+').map(|(_, tag)| tag);
    match (executor, tag) {
        (Executor::Cuda, Some(GPU_IMAGE_TAG)) | (Executor::Cpu, None) => Ok(()),
        (Executor::Cuda, _) => Err(format!(
            "cvm-version {:?} is not a GPU image; the cuda executor runs on the release's \
             `+{GPU_IMAGE_TAG}` image (VERSION+{GPU_IMAGE_TAG}@sha256:<its manifest>), whose \
             measured boot attests the GPUs before opening them",
            release.version
        )),
        (Executor::Cpu, Some(_)) => Err(format!(
            "cvm-version {:?} names a tagged image; the cpu executor runs on the release's \
             plain CPU image",
            release.version
        )),
    }
}

/// Why a gateway-token hash is refused.
#[cfg(feature = "argon2")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenHashError {
    /// Not a PHC string `argon2` can read.
    NotAPhcString,
    /// A PHC string for an algorithm other than Argon2id.
    NotArgon2id,
    /// No salt, or no hash output, so no token can ever verify against it.
    Incomplete,
    /// Parameters outside the operational profile ([`TOKEN_HASH_M_COST`] and
    /// its siblings), refused before anything is hashed.
    OutsideProfile,
    /// Parameters, version, salt or output length `argon2` refuses to hash
    /// with (a memory cost below its minimum, a salt shorter than eight
    /// bytes, …), so no token can ever verify against it.
    Unusable,
}

#[cfg(feature = "argon2")]
impl core::fmt::Display for TokenHashError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotAPhcString => "not a valid Argon2 hash string",
            Self::NotArgon2id => "must be an Argon2id hash",
            Self::Incomplete => "must carry a salt and a hash output",
            Self::Unusable => "has parameters, a salt or an output Argon2 cannot verify with",
            Self::OutsideProfile => {
                "must use the hash-secret profile: Argon2id v=19, m 19456 to 65536 KiB, t 2 to 4, \
                 p 1 to 4, a 32-byte output"
            }
        })
    }
}

/// The Argon2 version a gateway-token hash must use (0x13, `v=19`).
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_VERSION: u32 = 0x13;
/// Memory cost, KiB: from `hash-secret`'s own (`Argon2::default()`, 19,456,
/// OWASP's minimum for Argon2id) up to 64 MiB, a small band for a stronger
/// hash that still costs a boot nothing worth noticing.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_M_COST: std::ops::RangeInclusive<u32> = 19_456..=65_536;
/// Passes: `hash-secret`'s 2 up to 4.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_T_COST: std::ops::RangeInclusive<u32> = 2..=4;
/// Lanes: `hash-secret`'s 1 up to 4.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_P_COST: std::ops::RangeInclusive<u32> = 1..=4;
/// Output length: `hash-secret`'s 32 bytes, exactly.
#[cfg(feature = "argon2")]
pub const TOKEN_HASH_OUTPUT_LEN: usize = 32;

/// Parse the gateway token's measured hash (`GATEWAY_TOKEN_HASH`): an
/// Argon2id PHC string that a token can actually verify against.
///
/// Verification (`PasswordVerifier::verify_password`, which the node runs at
/// boot) needs a salt and an output, and hashes the candidate with the
/// hash's own algorithm, version and parameters; any of those `argon2`
/// refuses makes every token fail. So this runs that same hashing step, with
/// the same `Argon2::default()` the node verifies with, over an empty
/// candidate, and keeps only hashes for which it succeeds: the set this
/// accepts is the set some token verifies against. The comparison itself is
/// the node's, against its secret.
#[cfg(feature = "argon2")]
pub fn parse_gateway_token_hash(hash: &str) -> Result<argon2::PasswordHash, TokenHashError> {
    use argon2::CustomizedPasswordHasher as _;

    let parsed = argon2::PasswordHash::new(hash).map_err(|_| TokenHashError::NotAPhcString)?;
    if parsed.algorithm.as_str() != "argon2id" {
        return Err(TokenHashError::NotArgon2id);
    }
    let (Some(salt), Some(_)) = (&parsed.salt, &parsed.hash) else {
        return Err(TokenHashError::Incomplete);
    };
    let params = argon2::Params::try_from(&parsed).map_err(|_| TokenHashError::Unusable)?;
    // The operational profile, checked before anything is hashed: a hash's
    // parameters decide how much memory and time verifying it costs, so an
    // extreme one would turn the check (and the node's boot) into an
    // allocation of the hash's choosing.
    let in_profile = parsed.version == Some(TOKEN_HASH_VERSION)
        && TOKEN_HASH_M_COST.contains(&params.m_cost())
        && TOKEN_HASH_T_COST.contains(&params.t_cost())
        && TOKEN_HASH_P_COST.contains(&params.p_cost())
        && params.output_len() == Some(TOKEN_HASH_OUTPUT_LEN);
    if !in_profile {
        return Err(TokenHashError::OutsideProfile);
    }
    argon2::Argon2::default()
        .hash_password_customized(
            b"",
            salt,
            Some(parsed.algorithm.as_str()),
            parsed.version,
            params,
        )
        .map_err(|_| TokenHashError::Unusable)?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_seconds_are_positive_and_fit_in_milliseconds() {
        assert_eq!(parse_cache_seconds("900"), Some(900));
        assert_eq!(
            parse_cache_seconds(&MAX_CACHE_SECONDS.to_string()),
            Some(MAX_CACHE_SECONDS)
        );
        for bad in [
            "0",
            "-1",
            "",
            " 900",
            "9.5",
            &(MAX_CACHE_SECONDS + 1).to_string(),
        ] {
            assert_eq!(parse_cache_seconds(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_deployments_gpus_are_attested_exactly() {
        use Executor::{Cpu, Cuda};
        for n in 1..=8 {
            assert!(check_gpu_attestation(Cuda, Some(n), Some(n)).is_ok(), "{n}");
        }
        assert!(check_gpu_attestation(Cpu, None, None).is_ok());
        assert!(check_gpu_attestation(Cpu, Some(0), Some(0)).is_ok());
        for (executor, gpus, expected) in [
            (Cuda, None, Some(4)),
            (Cuda, None, None),
            (Cuda, Some(0), Some(8)),
            (Cuda, Some(9), Some(9)),
            (Cuda, Some(8), Some(4)),
            (Cuda, Some(8), None),
            (Cuda, Some(8), Some(0)),
            (Cpu, Some(8), Some(8)),
            (Cpu, None, Some(1)),
        ] {
            assert!(
                check_gpu_attestation(executor, gpus, expected).is_err(),
                "{executor:?} gpus={gpus:?} expected={expected:?}"
            );
        }
    }

    fn cuda(kv_device_bytes: u64) -> ExecutorSettings {
        ExecutorSettings::Cuda {
            kernels_dir: "/opt/eidola/kernels".into(),
            kv_device_bytes,
            further_kv_device_bytes: Vec::new(),
            graphs: CudaGraphs::On,
            gpu_cc: GpuCc::Required,
        }
    }

    fn cuda_on(budgets: &[u64]) -> ExecutorSettings {
        ExecutorSettings::Cuda {
            kernels_dir: "/opt/eidola/kernels".into(),
            kv_device_bytes: budgets[0],
            further_kv_device_bytes: budgets[1..].to_vec(),
            graphs: CudaGraphs::On,
            gpu_cc: GpuCc::Required,
        }
    }

    /// DeepSeek-V4.1-Flash's released sizing on two B300s: its whole context
    /// window, 64 seats, a step of 8,192 tokens, three drafts, the cache on.
    fn ds_sizing() -> Sizing {
        Sizing {
            kv_block_size: 256,
            max_model_len: 1 << 20,
            max_seqs: 64,
            max_batched_tokens: 8192,
            max_prefill_chunk: 4096,
            draft_tokens: 3,
            max_requests: 64,
        }
    }

    const DS_CACHE: CacheConfig = CacheConfig {
        enabled: true,
        idle_ttl_secs: 900,
        max_age_secs: 7200,
    };

    /// The published text-only pack, bounded: the shards and files of the
    /// released checkpoint, 510.29 GB, under the devices' 510.66 GB.
    const DS_PACK_BYTES: u64 = 510_300_000_000;

    /// The executor's bank groups hold one `MAX_MODEL_LEN` sequence at least,
    /// and its window groups every seat's windows: a bank block is 128
    /// entries of 356 bytes for each of the three pooled sources on device 0
    /// (136,704 bytes) and 256 entries of 356 on device 1 (91,136); a window
    /// block is 256 entries of 528 bytes for each of 20 layers (2,703,360
    /// bytes), and on device 1 when drafting the drafter's three more
    /// (3,108,864). A window group holds 1 + 64 × ((1 + 3) × 2 + 1) + 32 =
    /// 609 blocks with the cache, 225 without.
    #[test]
    fn deepseek_kv_minimum_is_a_sequence_of_banks_beside_every_seats_windows() {
        use crate::architecture::Architecture;
        let [d0, d1] = Architecture::DeepSeekV41Flash.devices() else {
            panic!()
        };
        let sizing = ds_sizing();
        let undrafted = Sizing {
            draft_tokens: 0,
            ..sizing
        };
        let bank = |d, s| uniform_kv_bank_block_bytes(d, s);
        let window = |d, s| uniform_kv_window_block_bytes(d, s);
        assert_eq!(
            (bank(d0, &sizing), window(d0, &sizing)),
            (136_704, 2_703_360)
        );
        assert_eq!(
            (bank(d1, &sizing), window(d1, &sizing)),
            (91_136, 3_108_864)
        );
        assert_eq!(window(d1, &undrafted), 2_703_360);
        // The host tier still counts a block of every group together.
        assert_eq!(uniform_kv_block_bytes(d0, &sizing), 2_840_064);
        assert_eq!(uniform_kv_block_bytes(d1, &sizing), 3_200_000);
        assert_eq!(uniform_kv_block_bytes(d1, &undrafted), 2_794_496);
        let no_cache = CacheConfig {
            enabled: false,
            ..DS_CACHE
        };
        assert_eq!(uniform_kv_window_blocks(d0, &sizing, &DS_CACHE), 609);
        assert_eq!(uniform_kv_window_blocks(d1, &sizing, &DS_CACHE), 609);
        assert_eq!(uniform_kv_window_blocks(d1, &sizing, &no_cache), 225);
        // 2^20 positions are 4,096 blocks and the null block; 2^17, 512.
        assert_eq!(
            cuda_kv_min_bytes(d0, &sizing, &DS_CACHE),
            4097 * 136_704 + 609 * 2_703_360
        );
        assert_eq!(
            cuda_kv_min_bytes(d1, &sizing, &DS_CACHE),
            4097 * 91_136 + 609 * 3_108_864
        );
        assert_eq!(
            cuda_kv_min_bytes(d1, &undrafted, &DS_CACHE),
            4097 * 91_136 + 609 * 2_703_360
        );
        let short = Sizing {
            max_model_len: 1 << 17,
            ..sizing
        };
        assert_eq!(
            cuda_kv_min_bytes(d0, &short, &DS_CACHE),
            513 * 136_704 + 609 * 2_703_360
        );
        assert_eq!(
            cuda_kv_min_bytes(d1, &short, &DS_CACHE),
            513 * 91_136 + 609 * 3_108_864
        );
        // The seats, the step and the cache size the windows; the context does not.
        let crowded = Sizing {
            max_seqs: 128,
            ..short
        };
        assert_eq!(
            uniform_kv_window_blocks(d1, &crowded, &DS_CACHE),
            1 + 128 * 9 + 32
        );
        assert_eq!(
            cuda_kv_min_bytes(d1, &short, &no_cache),
            513 * 91_136 + 225 * 3_108_864
        );
        // A block rounds up to 16 bytes per group: at two positions, device
        // 0's bank is 3 × 356 = 1,068 bytes, held in 1,072, and a window
        // group holds 1 + 64 × (4 × 65 + 1) + 4,096 blocks.
        let tiny = Sizing {
            kv_block_size: 2,
            ..short
        };
        assert_eq!(uniform_kv_bank_block_bytes(d0, &tiny), 1_072);
        assert_eq!(uniform_kv_window_block_bytes(d0, &tiny), 20 * 2 * 528);
        assert_eq!(
            uniform_kv_window_blocks(d0, &tiny, &DS_CACHE),
            1 + 64 * 261 + 4096
        );
        // The host tier's least is two blocks of every group of each device.
        assert_eq!(
            cuda_kv_host_min_bytes(d0, &sizing, &DS_CACHE, 1),
            2 * 2_840_064
        );
        assert_eq!(
            cuda_kv_host_min_bytes(d1, &sizing, &DS_CACHE, u64::MAX),
            2 * 3_200_000
        );
    }

    /// DeepSeek-V4.1 at its whole 2^20-position window on two B300s. The step's
    /// scratch grows with its rows times the context (the indexer scores every
    /// visible entry: 2.1 MB a row on device 1, 1.06 MB on device 0), so the
    /// step size decides: undrafted at 8,192-token steps device 1 keeps
    /// 10.2 GB of KV room, drafting three at 4,096-token steps 10.7 GB (the
    /// least KV is about 2.0 and 2.2 GB, and the room holds about 21 whole
    /// sequences of banks), and drafting three at 8,192-token steps device 1's
    /// weights and reserve alone pass its memory, so that is refused at any
    /// budget.
    #[test]
    fn deepseek_fits_two_b300s_at_its_whole_window_by_its_step() {
        use crate::architecture::Architecture;
        let devices = Architecture::DeepSeekV41Flash.devices();
        let [d0, d1] = devices else { panic!() };
        let room = |d: &Shape, s: &Sizing| {
            (SUPPORTED_GPU_MEMORY_BYTES as i128)
                - device_weights_bytes(d, s).unwrap() as i128
                - cuda_device_reserve_bytes(d, s) as i128
        };
        let check = |s: &Sizing, budgets: &[u64]| {
            check_device_resources(
                devices,
                s,
                &DS_CACHE,
                &cuda_on(budgets),
                0,
                DS_PACK_BYTES,
                196_608,
                2,
            )
        };
        for (draft_tokens, step, room1) in [(0, 8192, 10_233_054_144), (3, 4096, 10_713_172_224)] {
            let sizing = Sizing {
                draft_tokens,
                max_batched_tokens: step,
                ..ds_sizing()
            };
            let (room0, got1) = (room(d0, &sizing), room(d1, &sizing));
            assert_eq!(got1, room1, "drafts {draft_tokens}");
            let (room0, room1) = (room0 as u64, room1 as u64);
            let min0 = cuda_kv_min_bytes(d0, &sizing, &DS_CACHE);
            let min1 = cuda_kv_min_bytes(d1, &sizing, &DS_CACHE);
            assert!(
                min0 < room0 && min1 < room1,
                "{min0} {room0} {min1} {room1}"
            );
            // The banks the room holds beside the windows, on the tighter device.
            let windows = uniform_kv_window_blocks(d1, &sizing, &DS_CACHE);
            let banks = (room1 - windows * uniform_kv_window_block_bytes(d1, &sizing))
                / uniform_kv_bank_block_bytes(d1, &sizing);
            assert!(banks / 4096 >= 21, "{banks}");
            assert_eq!(check(&sizing, &[room0, room1]), Ok(()));
            assert_eq!(check(&sizing, &[min0, min1]), Ok(()));
            let err = check(&sizing, &[room0, room1 + 1]).unwrap_err();
            assert!(
                err.starts_with("device 1:") && err.contains("device memory"),
                "{err}"
            );
            let err = check(&sizing, &[room0 + 1, room1]).unwrap_err();
            assert!(
                err.starts_with("device 0:") && err.contains("device memory"),
                "{err}"
            );
            let err = check(&sizing, &[room0, min1 - 1]).unwrap_err();
            assert!(
                err.starts_with("device 1:") && err.contains("smallest"),
                "{err}"
            );
        }
        let crowded = Sizing {
            draft_tokens: 3,
            max_batched_tokens: 8192,
            ..ds_sizing()
        };
        assert_eq!(room(d1, &crowded), -384_161_536);
        let min = [d0, d1].map(|d| cuda_kv_min_bytes(d, &crowded, &DS_CACHE));
        let err = check(&crowded, &min).unwrap_err();
        assert!(
            err.starts_with("device 1:") && err.contains("device memory"),
            "{err}"
        );
        // A step's index scratch is its rows times a row's: 2 bytes a visible
        // entry and a mask word per 32 candidate blocks of 8.
        assert_eq!(
            index_scratch_row_bytes(d1, &crowded),
            2 * 1_048_577 + 4 * 4097
        );
        assert_eq!(
            index_scratch_row_bytes(d0, &crowded),
            2 * 524_289 + 4 * 2049
        );
        assert_eq!(step_scratch_rows(d1, &crowded), 8192);
        let few = Sizing {
            max_batched_tokens: 256,
            ..crowded
        };
        // A draft pass computes every slot of every drafting row.
        assert_eq!(step_scratch_rows(d0, &few), 64 * 5);
        // Undrafted, device 1 loads no drafter.
        let undrafted = Sizing {
            draft_tokens: 0,
            ..crowded
        };
        assert_eq!(
            device_weights_bytes(d1, &undrafted),
            Some(259_790_000_000 - 9_257_000_000)
        );
        assert_eq!(device_weights_bytes(d1, &crowded), Some(259_790_000_000));
    }

    /// The fit arithmetic at the released sizes with a 2^17-position context:
    /// device 0 holds 250.87 GB of weights and device 1 259.79 GB, each
    /// beside its KV budget and the executor's reserve on a 287.4 GB B300.
    #[test]
    fn deepseek_fits_two_b300s_at_a_131072_position_context() {
        use crate::architecture::Architecture;
        let devices = Architecture::DeepSeekV41Flash.devices();
        let sizing = Sizing {
            max_model_len: 1 << 17,
            ..ds_sizing()
        };
        let [d0, d1] = devices else { panic!() };
        let reserve0 = cuda_device_reserve_bytes(d0, &sizing);
        let reserve1 = cuda_device_reserve_bytes(d1, &sizing);
        // The step's scratch: device 0's indexers score a bank of half the
        // positions, device 1's holds the head's and the drafter's rows.
        assert_eq!((reserve0, reserve1), (10_383_024_128, 12_400_461_696));
        let room0 = SUPPORTED_GPU_MEMORY_BYTES - d0.weights_bytes.unwrap() - reserve0;
        let room1 = SUPPORTED_GPU_MEMORY_BYTES - d1.weights_bytes.unwrap() - reserve1;
        // About 26 GB of KV on device 0 and 15 GB on device 1.
        assert_eq!(room0 / 1_000_000_000, 26, "{room0}");
        assert_eq!(room1 / 1_000_000_000, 15, "{room1}");
        let min0 = cuda_kv_min_bytes(d0, &sizing, &DS_CACHE);
        let min1 = cuda_kv_min_bytes(d1, &sizing, &DS_CACHE);
        assert_eq!(
            (min0, min1),
            (
                513 * 136_704 + 609 * 2_703_360,
                513 * 91_136 + 609 * 3_108_864
            )
        );
        assert!(min0 < room0 && min1 < room1);

        let memory_mib = 196_608;
        let check = |budgets: &[u64], gpus: u64, pack: u64, host: u64| {
            check_device_resources(
                devices,
                &sizing,
                &DS_CACHE,
                &cuda_on(budgets),
                host,
                pack,
                memory_mib,
                gpus,
            )
        };
        assert_eq!(check(&[room0, room1], 2, DS_PACK_BYTES, 0), Ok(()));
        assert_eq!(check(&[min0, min1], 2, DS_PACK_BYTES, 0), Ok(()));
        // More GPUs than devices are attested and left idle.
        assert_eq!(check(&[room0, room1], 8, DS_PACK_BYTES, 0), Ok(()));
        // One byte over on either device, or under its least.
        let err = check(&[room0 + 1, room1], 2, DS_PACK_BYTES, 0).unwrap_err();
        assert!(
            err.starts_with("device 0:") && err.contains("device memory"),
            "{err}"
        );
        let err = check(&[room0, room1 + 1], 2, DS_PACK_BYTES, 0).unwrap_err();
        assert!(
            err.starts_with("device 1:") && err.contains("device memory"),
            "{err}"
        );
        let err = check(&[min0 - 1, room1], 2, DS_PACK_BYTES, 0).unwrap_err();
        assert!(
            err.starts_with("device 0:") && err.contains("smallest"),
            "{err}"
        );
        let err = check(&[room0, min1 - 1], 2, DS_PACK_BYTES, 0).unwrap_err();
        assert!(
            err.starts_with("device 1:") && err.contains("smallest"),
            "{err}"
        );
        // Device 0's room is not device 1's: the same budget on both does not
        // fit.
        assert!(check(&[room0, room0], 2, DS_PACK_BYTES, 0).is_err());
        // The wrong device count, either way, and too few GPUs.
        for budgets in [&[room0][..], &[room0, room1, room1][..]] {
            let err = check(budgets, 3, DS_PACK_BYTES, 0).unwrap_err();
            assert!(err.contains("runs on 2 devices"), "{err}");
        }
        let err = check(&[room0, room1], 1, DS_PACK_BYTES, 0).unwrap_err();
        assert!(err.contains("runs on 2 GPUs"), "{err}");
        // A pack heavier than the devices' bounds together is not this model.
        let total = d0.weights_bytes.unwrap() + d1.weights_bytes.unwrap();
        assert_eq!(check(&[room0, room1], 2, total, 0), Ok(()));
        let err = check(&[room0, room1], 2, total + 1, 0).unwrap_err();
        assert!(err.contains("weights pack"), "{err}");
        // A host tier gives every pool of both devices two blocks, or is none.
        let least: u64 = devices
            .iter()
            .zip([room0, room1])
            .map(|(d, b)| cuda_kv_host_min_bytes(d, &sizing, &DS_CACHE, b))
            .sum();
        assert_eq!(check(&[room0, room1], 2, DS_PACK_BYTES, least), Ok(()));
        let err = check(&[room0, room1], 2, DS_PACK_BYTES, least - 1).unwrap_err();
        assert!(err.contains("every KV pool of every device"), "{err}");
        // And the VM holds the tier.
        let err = check(&[room0, room1], 2, DS_PACK_BYTES, (memory_mib << 20) - 1).unwrap_err();
        assert!(err.contains("(memory)"), "{err}");
        // A block of the pooled sources holds whole entries.
        let odd = Sizing {
            kv_block_size: 15,
            ..sizing
        };
        let err = check_device_resources(
            devices,
            &odd,
            &DS_CACHE,
            &cuda_on(&[room0, room1]),
            0,
            DS_PACK_BYTES,
            memory_mib,
            2,
        )
        .unwrap_err();
        assert!(
            err.starts_with("device 0:") && err.contains("pooling ratio 2"),
            "{err}"
        );

        // The cpu executor runs both devices' pools in host memory.
        let cpu = ExecutorSettings::Cpu { kv_blocks: 64 };
        let per_block = 256 * (534 + 20 * 528 + 356 + 20 * 528 + 3 * 528);
        let host = |mib| check_device_resources(devices, &sizing, &DS_CACHE, &cpu, 0, 0, mib, 0);
        let needed = host_memory_bytes(&sizing) + 64 * per_block;
        assert_eq!(host(needed.div_ceil(1 << 20)), Ok(()));
        assert!(host(needed / (1 << 20) - 1).is_err());
    }

    /// A one-device model's budget is unchanged by the list grammar.
    #[test]
    fn kv_device_bytes_name_one_budget_per_device() {
        assert_eq!(
            parse_kv_device_bytes("85899345920"),
            Ok(vec![85_899_345_920])
        );
        assert_eq!(parse_kv_device_bytes("1,2"), Ok(vec![1, 2]));
        assert_eq!(parse_kv_device_bytes(&["1"; 8].join(",")).unwrap().len(), 8);
        let cap = caps::KV_DEVICE_BYTES;
        assert_eq!(
            parse_kv_device_bytes(&format!("{cap},{cap}")),
            Ok(vec![cap, cap])
        );
        for bad in [
            "".to_string(),
            "0".into(),
            "1,0".into(),
            "1,".into(),
            ",1".into(),
            "1,,2".into(),
            "1 ,2".into(),
            "1;2".into(),
            "-1".into(),
            format!("1,{}", cap + 1),
            ["1"; 9].join(","),
        ] {
            assert!(parse_kv_device_bytes(&bad).is_err(), "{bad:?}");
        }
    }

    /// The staged-expert path and host-memory Engram tables are development
    /// settings of the executors with no spelling in the measured
    /// configuration, so no pinned deployment can name either.
    #[test]
    fn dev_only_placements_have_no_spelling() {
        for name in [
            "EIDOLA_ENGINE_STAGED_EXPERTS",
            "EIDOLA_ENGINE_EXPERTS_ON_HOST",
            "EIDOLA_ENGINE_ENGRAM_ON_HOST",
            "EIDOLA_ENGINE_ENGRAM_HOST_BYTES",
        ] {
            assert!(check_env_names([name]).is_err(), "{name}");
        }
        for name in env::MEASURED {
            assert!(
                !name.contains("EXPERT") && !name.contains("ENGRAM") && !name.contains("STAGE"),
                "{name}"
            );
        }
    }

    #[test]
    fn each_executor_runs_on_its_own_image() {
        use Executor::{Cpu, Cuda};
        let pin = "ab".repeat(32);
        let check =
            |executor, raw: &str| check_cvm_image(executor, &parse_cvm_version(raw).unwrap());
        for raw in [
            "0.15.3-rc.1+gpu".to_string(),
            format!("0.15.3-rc.1+gpu@sha256:{pin}"),
            "0.16.0+gpu".to_string(),
        ] {
            assert!(check(Cuda, &raw).is_ok(), "{raw}");
            assert!(check(Cpu, &raw).unwrap_err().contains("cpu"), "{raw}");
        }
        for raw in [
            "0.15.3-rc.1".to_string(),
            format!("0.15.3-rc.1@sha256:{pin}"),
        ] {
            assert!(check(Cpu, &raw).is_ok(), "{raw}");
            assert!(check(Cuda, &raw).unwrap_err().contains("+gpu"), "{raw}");
        }
        // Only the exact tag names the GPU image.
        for raw in [
            "0.15.3-rc.1+GPU",
            "0.15.3-rc.1+gpu2",
            "0.15.3-rc.1+gpu+gpu",
            "0.15.3-rc.1+",
            "0.15.3-rc.1-gpu",
        ] {
            assert!(check(Cuda, raw).is_err(), "{raw}");
        }
        for raw in ["0.15.3-rc.1+GPU", "0.15.3-rc.1+", "0.15.3+cpu"] {
            assert!(check(Cpu, raw).is_err(), "{raw}");
        }
    }

    #[test]
    fn resources_fit_the_vm_and_gpus() {
        let sizing = Sizing {
            kv_block_size: 16,
            max_model_len: 131_072,
            max_seqs: 64,
            max_batched_tokens: 8192,
            max_prefill_chunk: 4096,
            draft_tokens: 0,
            max_requests: 8,
        };
        let cache = CacheConfig {
            enabled: true,
            idle_ttl_secs: 900,
            max_age_secs: 3600,
        };
        let kv = cuda(64 << 30);
        assert!(check_resources(FLASH, &sizing, &cache, &kv, 0, FIXTURE_PACK, 65_536, 8).is_ok());
        // A request slot is 645,922,816 bytes at the caps: a 32 MiB body read
        // with both parses (3 × 32 MiB of text, 512 bytes for each of 2^18
        // values) and admitted with its tree (32 MiB of text, 160 bytes a
        // value) and its prompt and tokens (10 × 32 MiB).
        let one = Sizing {
            max_requests: 1,
            ..sizing
        };
        let tables = 64 * 8192 * 8;
        assert_eq!(
            host_memory_bytes(&one) - tables - PROCESS_HEADROOM_BYTES,
            645_922_816
        );
        // Drafting adds the drafter group's block tables to the host mirror
        // (and to the device's): a third 64 × 8,192 × 4 bytes.
        for k in 1..=3 {
            let drafting = Sizing {
                draft_tokens: k,
                ..one
            };
            assert_eq!(kv_groups(k), 3);
            assert_eq!(
                host_memory_bytes(&drafting) - host_memory_bytes(&one),
                64 * 8192 * 4
            );
        }
        assert_eq!(kv_groups(0), 2);
        let near = |n| Sizing {
            max_requests: n,
            ..sizing
        };
        // Six slots, the headroom and the tables fit 8 GiB; eight do not.
        assert!(check_resources(FLASH, &near(6), &cache, &kv, 0, FIXTURE_PACK, 8192, 8).is_ok());
        assert!(check_resources(FLASH, &near(8), &cache, &kv, 0, FIXTURE_PACK, 8192, 8).is_err());
        // Near the limit: 99 slots fit 64 GiB; 100 do not.
        assert!(check_resources(FLASH, &near(99), &cache, &kv, 0, FIXTURE_PACK, 65_536, 8).is_ok());
        assert!(
            check_resources(FLASH, &near(100), &cache, &kv, 0, FIXTURE_PACK, 65_536, 8).is_err()
        );
        // The cuda executor needs a GPU.
        assert!(check_resources(FLASH, &sizing, &cache, &kv, 0, FIXTURE_PACK, 65_536, 0).is_err());
        // Block size 1 at the longest model length: 2^20 entries per slot.
        let tables = Sizing {
            kv_block_size: 1,
            max_model_len: caps::MAX_MODEL_LEN,
            max_seqs: caps::MAX_SEQS,
            ..sizing
        };
        assert!(check_resources(FLASH, &tables, &cache, &kv, 0, FIXTURE_PACK, 16_384, 8).is_err());
        // The cpu executor's KV is host memory: 2^20 blocks of 16 positions
        // are 3.4 TiB.
        let cpu = |kv_blocks| ExecutorSettings::Cpu { kv_blocks };
        assert!(
            check_resources(
                FLASH,
                &sizing,
                &cache,
                &cpu(1024),
                0,
                FIXTURE_PACK,
                65_536,
                0
            )
            .is_ok()
        );
        assert!(
            check_resources(
                FLASH,
                &sizing,
                &cache,
                &cpu(1 << 20),
                0,
                FIXTURE_PACK,
                65_536,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn kv_device_bytes_hold_the_pools_at_their_smallest() {
        let sizing = Sizing {
            kv_block_size: 16,
            max_model_len: 131_072,
            max_seqs: 64,
            max_batched_tokens: 8192,
            max_prefill_chunk: 4096,
            draft_tokens: 0,
            max_requests: 8,
        };
        let cache = |enabled| CacheConfig {
            enabled,
            idle_ttl_secs: 900,
            max_age_secs: 3600,
        };
        // Sliding: R = ⌈127/16⌉ + 1 = 9; 1 + 64 × (4 × 9 + 1) + 512 = 2,881
        // blocks, one pad more, × 16 × 199,680. Global: 8,192 + 1 blocks,
        // one pad more, × 16 × 23,040.
        assert_eq!(
            cuda_kv_min_bytes(FLASH, &sizing, &cache(true)),
            12_228_280_320
        );
        // Without the cache, no retained windows: 1 + 64 × 10 + 512 = 1,153.
        assert_eq!(
            cuda_kv_min_bytes(FLASH, &sizing, &cache(false)),
            6_707_527_680
        );
        // Drafting k tokens adds a drafter pool of the sliding pool's
        // blocks, each 16 positions × k × 5,120 bytes and a tap of k × 16 KiB.
        for k in 1..=3u64 {
            let drafting = Sizing {
                draft_tokens: u32::try_from(k).unwrap(),
                ..sizing
            };
            assert_eq!(
                cuda_kv_min_bytes(FLASH, &drafting, &cache(true)),
                12_228_280_320 + 2_882 * (16 * k * 5_120 + k * 16_384)
            );
        }
        let min = cuda_kv_min_bytes(FLASH, &sizing, &cache(true));
        let fits = |bytes| {
            check_resources(
                FLASH,
                &sizing,
                &cache(true),
                &cuda(bytes),
                0,
                FIXTURE_PACK,
                65_536,
                1,
            )
        };
        assert!(fits(min).is_ok());
        assert!(fits(min - 1).is_err());
    }

    /// The shape every check below is held to.
    const FLASH: &Shape = &crate::architecture::MIMO_V26_FLASH;

    /// The weights pack in the gateway's fixture: 17,419,419,648 bytes.
    const FIXTURE_PACK: u64 = 17_419_419_648;

    /// MiMo-V2.6-Flash's checkpoint, about 177.7 GB.
    const FLASH_PACK: u64 = 177_700_003_840;

    #[test]
    fn the_device_holds_kv_weights_and_the_executors_reserve() {
        let sizing = Sizing {
            kv_block_size: 16,
            max_model_len: 131_072,
            max_seqs: 64,
            max_batched_tokens: 8192,
            max_prefill_chunk: 4096,
            draft_tokens: 0,
            max_requests: 8,
        };
        let cache = CacheConfig {
            enabled: true,
            idle_ttl_secs: 900,
            max_age_secs: 3600,
        };
        // 4 GiB fixed; 8,192 step tokens × 1.25 MiB; the 1.5 GiB expert
        // floor; 64 seats × one sampler row of 1.25 MiB; 64 × 8,192 table
        // entries × 2 groups × 4 bytes; split attention's 64 tokens × 128
        // chunks of partial rows of 16,640 bytes; 131,072 positions × 768
        // bytes of RoPE tables.
        let reserve = cuda_device_reserve_bytes(FLASH, &sizing);
        assert_eq!(split_partial_rows(&sizing), 64 * 128);
        assert_eq!(FLASH.split_partial_row_bytes, 16_640);
        assert_eq!(
            reserve,
            (4 << 30)
                + 8192 * (5 << 18)
                + 65_536 * (24 << 10)
                + 64 * (5 << 18)
                + 64 * 8192 * 8
                + 64 * 128 * 16_640
                + 131_072 * 768
        );
        assert_eq!(reserve, 16_968_056_832);
        // A sampler row holds an f64 distribution over every id of the head.
        const { assert!(crate::architecture::MIMO_V26_FLASH.sampler_row_device_bytes >= 152_576 * 8) };
        // Drafting three tokens: per seat 8 sampler rows, 39 rows of 16 KiB
        // (12 of drafter state, 3 + 24 level rows), 32 KiB of step tables,
        // four global page lists of 8,192 entries and 48 bytes of work items
        // per chunk; a third group's block tables; and the partial rows of
        // 64 rows of four tokens.
        let drafting = Sizing {
            draft_tokens: 3,
            ..sizing
        };
        assert_eq!(sampler_rows(3), 8);
        assert_eq!(drafter_seat_rows(3), 39);
        assert_eq!(split_partial_rows(&drafting), 256 * 128);
        assert_eq!(
            cuda_device_reserve_bytes(FLASH, &drafting),
            (4 << 30)
                + 8192 * (5 << 18)
                + 65_536 * (24 << 10)
                + 64 * (8 * (5 << 18) + 39 * (16 << 10) + (32 << 10) + 8192 * 16 + 128 * 48)
                + 64 * 8192 * 12
                + 256 * 128 * 16_640
                + 131_072 * 768
        );
        for k in 1..=3 {
            assert!(
                cuda_device_reserve_bytes(
                    FLASH,
                    &Sizing {
                        draft_tokens: k,
                        ..sizing
                    }
                ) > reserve
            );
        }
        // Pinned deployments draft three tokens: with the drafter pool in the
        // KV budget and the larger reserve, 64 GiB of KV still fits beside
        // Flash's checkpoint, and the device is still the bound.
        let drafting_reserve = cuda_device_reserve_bytes(FLASH, &drafting);
        assert!(cuda_kv_min_bytes(FLASH, &drafting, &cache) < 64 << 30);
        let fits_drafting = |kv: u64| {
            check_resources(
                FLASH,
                &drafting,
                &cache,
                &cuda(kv),
                0,
                FLASH_PACK,
                65_536,
                8,
            )
        };
        assert!(fits_drafting(64 << 30).is_ok());
        let room = SUPPORTED_GPU_MEMORY_BYTES - FLASH_PACK - drafting_reserve;
        assert!(fits_drafting(room).is_ok());
        assert!(fits_drafting(room + 1).is_err());
        let fits = |kv: u64, weights: u64, gpus: u64| {
            check_resources(FLASH, &sizing, &cache, &cuda(kv), 0, weights, 65_536, gpus)
        };
        // A realistic Flash deployment: 64 GiB of KV beside the checkpoint.
        assert!(fits(64 << 30, FLASH_PACK, 1).is_ok());
        assert!(fits(64 << 30, FLASH_PACK, 8).is_ok());
        // Exactly the device, and one byte past it.
        let room = SUPPORTED_GPU_MEMORY_BYTES - FLASH_PACK - reserve;
        assert!(fits(room, FLASH_PACK, 1).is_ok());
        let err = fits(room + 1, FLASH_PACK, 1).unwrap_err();
        assert!(err.contains(SUPPORTED_GPU), "{err}");
        // The device is what the driver reports, not the HBM it is sold
        // with: a budget that would fill 288 GiB does not fit.
        assert!(fits((288 << 30) - FLASH_PACK - reserve, FLASH_PACK, 1).is_err());
        // A budget that fits only without the weights.
        let without_weights = SUPPORTED_GPU_MEMORY_BYTES - reserve;
        assert!(fits(without_weights, 0, 1).is_ok());
        assert!(fits(without_weights, FIXTURE_PACK, 1).is_err());
        // 288 GiB is more than the device holds, whatever else; and eight
        // GPUs do not widen one node's device.
        assert!(fits(288 << 30, 0, 8).is_err());
        // The node refuses a budget not less than its device's memory; this
        // rule refuses one less than that too, as the reserve is never zero.
        assert!(fits(SUPPORTED_GPU_MEMORY_BYTES, 0, 8).is_err());
        assert!(fits(SUPPORTED_GPU_MEMORY_BYTES - 1, 0, 8).is_err());
    }

    #[test]
    fn an_artifact_ref_gives_its_data_size() {
        let root = "d".repeat(64);
        let uuid = "3892cd2f-a06e-5aee-8276-93140b9f06ec";
        assert_eq!(
            artifact_data_bytes(&format!("{root}_17419419648_{uuid}")),
            Some(17_419_419_648)
        );
        assert_eq!(artifact_data_bytes(&format!("{root}_+5_{uuid}")), None);
        assert_eq!(artifact_data_bytes(&format!("{root}_5_{uuid}x")), None);
    }

    #[test]
    fn kernels_come_from_the_image() {
        assert!(check_kernels_dir("/opt/eidola/kernels").is_ok());
        assert!(check_kernels_dir("/tinfoilx/kernels").is_ok());
        for bad in [
            "/tinfoil",
            "/tinfoil/models/weights/kernels",
            "/tinfoil/kernels",
            "/",
            "",
            "kernels",
            "/opt/../tinfoil/kernels",
            "/opt//kernels",
            "/opt/./kernels",
            "/opt/kernels/",
        ] {
            assert!(check_kernels_dir(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn an_artifact_ref_is_modelwraps() {
        let root = "d".repeat(64);
        let uuid = "3892cd2f-a06e-5aee-8276-93140b9f06ec";
        assert!(is_artifact_ref(&format!("{root}_17419419648_{uuid}")));
        assert!(is_artifact_ref(&format!("{root}_0_{uuid}")));
        for bad in [
            root.clone(),
            format!("{root}_garbage"),
            format!("{root}_x_{uuid}"),
            format!("{root}__{uuid}"),
            format!("{root}_99999999999999999999_{uuid}"),
            format!("{root}_1_{}", uuid.to_uppercase()),
            format!("{root}_1_{uuid}_extra"),
            format!("{}_1_{uuid}", "D".repeat(64)),
            format!("{root}_1_3892cd2fa06e5aee827693140b9f06ec"),
        ] {
            assert!(!is_artifact_ref(&bad), "{bad}");
        }
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn an_extreme_token_hash_is_refused_before_hashing() {
        let good = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E";
        assert!(parse_gateway_token_hash(good).is_ok());
        let started = std::time::Instant::now();
        for params in [
            "m=4294967295,t=2,p=1",
            "m=19456,t=4294967295,p=1",
            "m=19456,t=2,p=255",
            "m=8192,t=2,p=1",
            "m=19456,t=1,p=1",
        ] {
            let extreme = good.replace("m=19456,t=2,p=1", params);
            assert_eq!(
                parse_gateway_token_hash(&extreme).unwrap_err(),
                TokenHashError::OutsideProfile,
                "{params}"
            );
        }
        // A 16-byte output and a pre-1.3 version are outside the profile too.
        let short = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$aGFzaGhhc2hoYXNoaGFzaA";
        assert_eq!(
            parse_gateway_token_hash(short).unwrap_err(),
            TokenHashError::OutsideProfile
        );
        assert_eq!(
            parse_gateway_token_hash(&good.replace("v=19", "v=16")).unwrap_err(),
            TokenHashError::OutsideProfile
        );
        // Refused without hashing: an m of 4 TiB would take far longer.
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
    }

    #[test]
    fn path_components_are_safe() {
        for good in ["mimo-v2", "tdx-8x.h200", "0a_b"] {
            assert!(is_safe_component(good), "{good}");
        }
        for bad in [
            "",
            "Upper",
            "has space",
            ".hidden",
            "-dash",
            "a/b",
            "..",
            "é",
        ] {
            assert!(!is_safe_component(bad), "{bad:?}");
        }
    }

    #[test]
    fn container_gpu_access_follows_the_executor() {
        use Executor::{Cpu, Cuda};
        assert!(check_container_gpu_access(Cuda, Some("nvidia"), Some("all")).is_ok());
        assert!(check_container_gpu_access(Cpu, None, None).is_ok());
        for (executor, runtime, gpus) in [
            (Cuda, None, None),
            (Cuda, Some("nvidia"), None),
            (Cuda, Some("nvidia"), Some("1")),
            (Cuda, Some("nvidia"), Some("0,1")),
            (Cuda, Some("runc"), Some("all")),
            (Cpu, Some("nvidia"), None),
            (Cpu, None, Some("all")),
        ] {
            assert!(check_container_gpu_access(executor, runtime, gpus).is_err());
        }
    }

    #[test]
    fn the_env_names_are_exactly_the_measured_ones() {
        assert!(check_env_names(env::MEASURED.iter().copied()).is_ok());
        for extra in [
            "TOKIO_WORKER_THREADS",
            "RUST_MIN_STACK",
            "CUDA_VISIBLE_DEVICES",
            "LD_PRELOAD",
            "GATEWAY_TOKEN",
        ] {
            assert!(check_env_names([env::MODEL_ID, extra]).is_err(), "{extra}");
        }
    }

    #[test]
    fn vm_resources_are_a_launchable_shape() {
        assert!(check_vm_resources(16, 65536).is_ok());
        assert!(check_vm_resources(1, 8192).is_ok());
        assert!(check_vm_resources(MAX_CPUS, MAX_MEMORY_MIB).is_ok());
        // The platform's 768 GiB GPU shape, and other whole-GiB sizes.
        assert!(check_vm_resources(32, 786_432).is_ok());
        assert!(check_vm_resources(16, 9216).is_ok());
        for (cpus, memory) in [
            (0, 65536),
            (MAX_CPUS + 1, 65536),
            (16, 1),
            (16, 4096),
            (16, 7168),
            (16, 65535),
            (16, 65000),
            (16, MAX_MEMORY_MIB + 1024),
            (16, MAX_MEMORY_MIB * 2),
        ] {
            assert!(check_vm_resources(cpus, memory).is_err(), "{cpus} {memory}");
        }
    }

    #[test]
    fn the_gateway_token_shape() {
        assert!(is_gateway_token("dev-gateway-token"));
        assert!(is_gateway_token(&"a".repeat(GATEWAY_TOKEN_MAX_LEN)));
        for bad in [
            "short".to_string(),
            "has a space in it".to_string(),
            "line\nbreak-token-xx".to_string(),
            "non-ascii-tøken-xx".to_string(),
            "a".repeat(GATEWAY_TOKEN_MAX_LEN + 1),
        ] {
            assert!(!is_gateway_token(&bad), "{bad:?}");
        }
    }

    #[test]
    fn the_only_secret_is_the_gateway_token() {
        assert!(check_secrets(["GATEWAY_TOKEN"]).is_ok());
        assert!(check_secrets([]).is_err());
        assert!(check_secrets(["GATEWAY_TOKEN", "RUST_LOG"]).is_err());
        assert!(check_secrets(["GATEWAY_TOKEN", "GATEWAY_TOKEN"]).is_err());
    }

    #[test]
    fn shim_paths_match_as_the_shim_matches() {
        assert!(shim_path_matches(
            "/v1/chat/completions",
            "/v1/chat/completions"
        ));
        assert!(!shim_path_matches(
            "/v1/chat/completions",
            "/v1/chat/completions/x"
        ));
        assert!(shim_path_matches("/*", "/healthz"));
        assert!(shim_path_matches("/v1/*", "/v1/engine/info"));
        assert!(!shim_path_matches("/v1/*", "/v1"));
        assert!(shim_path_matches("/v1*", "/v1"));
        assert!(shim_path_matches("/v1*", "/v1/x"));
        assert!(!shim_path_matches("/v1*", "/v10"));

        assert!(check_shim_paths(&["/*"]).is_ok());
        assert!(check_shim_paths(&["/v1/*", "/healthz"]).is_ok());
        assert!(check_shim_paths(NODE_ROUTES).is_ok());
        assert!(check_shim_paths(&[]).is_err());
        assert!(check_shim_paths(&["/v1/chat/completions"]).is_err());
        assert!(check_shim_paths(&["/*", "/admin"]).is_err());
        assert!(check_shim_paths(&["v1/*", "/healthz"]).is_err());
    }

    #[test]
    fn the_weights_are_the_granted_pinned_pack() {
        let mpk = format!(
            "{}_17419419648_3892cd2f-a06e-5aee-8276-93140b9f06ec",
            "d".repeat(64)
        );
        let pack = ModelPack {
            name: "weights",
            repo: Some("o/m@4444"),
            mpk: Some(&mpk),
        };
        let ok = |packs: &[ModelPack<'_>], granted: &[&str], dir: &str| {
            check_weights_pack(packs, granted, dir, "o/m", "4444")
        };
        assert!(ok(&[pack], &["weights"], "/tinfoil/models/weights").is_ok());
        assert!(ok(&[pack], &["weights"], "/tinfoil/models/weights/original").is_ok());
        assert!(ok(&[], &[], "/tinfoil/models/weights").is_err());
        assert!(ok(&[pack, pack], &["weights"], "/tinfoil/models/weights").is_err());
        assert!(ok(&[pack], &[], "/tinfoil/models/weights").is_err());
        assert!(ok(&[pack], &["other"], "/tinfoil/models/weights").is_err());
        for dir in [
            "/weights",
            "/tinfoil/models/weightsx",
            "/tinfoil/models/weights/../x",
            "/tinfoil/models/weights//x",
        ] {
            assert!(ok(&[pack], &["weights"], dir).is_err(), "{dir}");
        }
        let other_repo = ModelPack {
            repo: Some("o/m@5555"),
            ..pack
        };
        assert!(ok(&[other_repo], &["weights"], "/tinfoil/models/weights").is_err());
        let unpinned = ModelPack { mpk: None, ..pack };
        assert!(ok(&[unpinned], &["weights"], "/tinfoil/models/weights").is_err());
    }

    #[test]
    fn the_prefix_cache_switch_is_exact() {
        assert_eq!(parse_prefix_cache("true"), Some(true));
        assert_eq!(parse_prefix_cache("false"), Some(false));
        for bad in ["True", "1", "yes", ""] {
            assert_eq!(parse_prefix_cache(bad), None);
        }
    }

    #[test]
    fn cvm_version_is_bare_with_at_most_one_inline_pin() {
        let pin = "ab".repeat(32);
        assert_eq!(
            parse_cvm_version("0.15.0"),
            Ok(CvmVersion {
                version: "0.15.0",
                manifest_sha256: None
            })
        );
        assert_eq!(
            parse_cvm_version(&format!("0.15.0@sha256:{pin}"))
                .unwrap()
                .manifest_sha256,
            Some(pin.as_str())
        );
        for (bad, error) in [
            ("".to_string(), CvmVersionError::NotBare),
            ("v0.15.0".to_string(), CvmVersionError::NotBare),
            (format!("@sha256:{pin}"), CvmVersionError::NotBare),
            (
                "0.15.0@sha256:AB".to_string(),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{}", pin.to_uppercase()),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha512:{pin}"),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{pin}@sha256:{pin}"),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{pin}x"),
                CvmVersionError::MalformedPin,
            ),
        ] {
            assert_eq!(parse_cvm_version(&bad), Err(error), "{bad:?}");
        }
    }

    #[cfg(feature = "argon2")]
    fn cuda_env() -> std::collections::BTreeMap<&'static str, String> {
        [
            (env::MODEL_ID, "m"),
            (env::WEIGHTS_DIR, "/tinfoil/models/weights"),
            (env::WEIGHTS_SHA256, &"ab".repeat(32)),
            (env::WEIGHTS_STORAGE, "verified-readonly"),
            (
                env::GATEWAY_TOKEN_HASH,
                "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E",
            ),
            (env::EXECUTOR, "cuda"),
            (env::BIND_ADDR, "0.0.0.0:8080"),
            (env::KV_BLOCK_SIZE, "16"),
            (env::KV_DEVICE_BYTES, "68719476736"),
            (env::KERNELS_DIR, "/opt/eidola/kernels"),
            (env::CUDA_GRAPHS, "on"),
            (env::GPU_CC, "required"),
            (env::MAX_MODEL_LEN, "131072"),
            (env::MAX_SEQS, "64"),
            (env::MAX_BATCHED_TOKENS, "8192"),
            (env::MAX_PREFILL_CHUNK, "4096"),
            (env::DRAFT_TOKENS, "3"),
            (env::MAX_REQUESTS, "8"),
            (env::PREFIX_CACHE, "true"),
            (env::CACHE_IDLE_TTL_SECS, "900"),
            (env::CACHE_MAX_AGE_SECS, "3600"),
            (env::KV_HOST_BYTES, "34359738368"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect()
    }

    /// With the prefix cache on, the block size divides the declared-prefix
    /// quantum (every divisor of 256 is accepted, nothing else); with it off,
    /// any block size within its cap is.
    #[cfg(feature = "argon2")]
    #[test]
    fn a_cached_block_size_divides_the_declared_prefix_quantum() {
        let base = cuda_env();
        for block in 1..=caps::KV_BLOCK_SIZE {
            let mut map = base.clone();
            map.insert(env::KV_BLOCK_SIZE, block.to_string());
            let parsed = parse_map(&map);
            if 256 % block == 0 {
                assert!(parsed.is_ok(), "{block}: {parsed:?}");
            } else {
                assert!(
                    parsed.as_ref().unwrap_err().contains(env::KV_BLOCK_SIZE),
                    "{block}: {parsed:?}"
                );
            }
        }
        let mut map = base.clone();
        map.insert(env::PREFIX_CACHE, "false".into());
        map.insert(env::KV_HOST_BYTES, "0".into());
        map.insert(env::KV_BLOCK_SIZE, "24".into());
        assert!(parse_map(&map).is_ok());
    }

    /// The host tier's size is required, a number of bytes up to its cap,
    /// bound into the measured configuration as given, refused with the
    /// prefix cache off unless 0, and charged to the VM's memory.
    #[cfg(feature = "argon2")]
    #[test]
    fn the_host_tier_size_is_measured_and_charged_to_the_vm() {
        let base = cuda_env();
        assert_eq!(parse_map(&base).unwrap().kv_host_bytes, 32 << 30);
        let mut map = base.clone();
        map.remove(env::KV_HOST_BYTES);
        assert!(parse_map(&map).unwrap_err().contains(env::KV_HOST_BYTES));
        for bad in ["", "-1", "1.5", "32GiB", " 1", "0x10"] {
            let mut map = base.clone();
            map.insert(env::KV_HOST_BYTES, bad.into());
            assert!(
                parse_map(&map).unwrap_err().contains(env::KV_HOST_BYTES),
                "{bad:?}"
            );
        }
        let mut map = base.clone();
        map.insert(env::KV_HOST_BYTES, caps::KV_HOST_BYTES.to_string());
        assert_eq!(parse_map(&map).unwrap().kv_host_bytes, caps::KV_HOST_BYTES);
        map.insert(env::KV_HOST_BYTES, (caps::KV_HOST_BYTES + 1).to_string());
        assert!(parse_map(&map).unwrap_err().contains(env::KV_HOST_BYTES));
        // Without the prefix cache only 0 parses.
        let mut map = base.clone();
        map.insert(env::PREFIX_CACHE, "false".into());
        let err = parse_map(&map).unwrap_err();
        assert!(
            err.contains(env::KV_HOST_BYTES) && err.contains(env::PREFIX_CACHE),
            "{err}"
        );
        map.insert(env::KV_HOST_BYTES, "0".into());
        assert_eq!(parse_map(&map).unwrap().kv_host_bytes, 0);
        assert!(env::MEASURED.contains(&env::KV_HOST_BYTES));

        // The VM must hold the tier besides everything else: exactly.
        let m = parse_map(&base).unwrap();
        let rest = host_memory_bytes(&m.sizing);
        let mib = |bytes: u64| bytes.div_ceil(1 << 20);
        let fits = |host: u64, memory: u64| {
            check_resources(
                FLASH,
                &m.sizing,
                &m.cache,
                &m.executor,
                host,
                FIXTURE_PACK,
                memory,
                8,
            )
        };
        let room = 65_536u64 << 20;
        assert!(fits(room - rest, mib(room)).is_ok());
        let err = fits(room - rest + 1, mib(room)).unwrap_err();
        assert!(err.contains(env::KV_HOST_BYTES), "{err}");

        // A tier is none (0) or holds two blocks of every pool: the node
        // refuses to boot with less.
        let ExecutorSettings::Cuda {
            kv_device_bytes, ..
        } = m.executor
        else {
            panic!("the cuda executor");
        };
        let least = cuda_kv_host_min_bytes(FLASH, &m.sizing, &m.cache, kv_device_bytes);
        assert!(least > 1 && least < 32 << 30, "{least}");
        assert!(fits(0, mib(room)).is_ok());
        assert!(fits(least, mib(room)).is_ok());
        for short in [1, least - 1] {
            let err = fits(short, mib(room)).unwrap_err();
            assert!(
                err.contains(env::KV_HOST_BYTES) && err.contains("two host blocks"),
                "{err}"
            );
        }
    }
    /// The settings only the cuda executor takes.
    #[cfg(feature = "argon2")]
    const CUDA_SETTINGS: [&str; 4] = [
        env::KV_DEVICE_BYTES,
        env::KERNELS_DIR,
        env::CUDA_GRAPHS,
        env::GPU_CC,
    ];

    #[cfg(feature = "argon2")]
    fn parse_map(
        map: &std::collections::BTreeMap<&'static str, String>,
    ) -> Result<MeasuredConfig, String> {
        parse_measured(&|name| map.get(name).cloned()).map_err(|e| e.0)
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn each_executor_takes_its_own_settings_and_refuses_the_others() {
        let cuda = cuda_env();
        assert_eq!(
            parse_map(&cuda).unwrap().executor,
            ExecutorSettings::Cuda {
                kernels_dir: "/opt/eidola/kernels".into(),
                kv_device_bytes: 64 << 30,
                further_kv_device_bytes: Vec::new(),
                graphs: CudaGraphs::On,
                gpu_cc: GpuCc::Required,
            }
        );
        // Two devices' budgets.
        let mut two = cuda.clone();
        two.insert(
            env::KV_DEVICE_BYTES,
            format!("{},{}", 64u64 << 30, 32u64 << 30),
        );
        let two = parse_map(&two).unwrap().executor;
        assert_eq!(two.kv_device_budgets(), [64 << 30, 32 << 30]);
        // Every cuda setting is required with cuda.
        for name in CUDA_SETTINGS {
            let mut map = cuda.clone();
            map.remove(name);
            assert!(parse_map(&map).unwrap_err().contains(name), "{name}");
        }
        // The cpu executor's block count is refused with cuda, not ignored.
        let mut map = cuda.clone();
        map.insert(env::KV_BLOCKS, "256".into());
        let err = parse_map(&map).unwrap_err();
        assert!(err.contains(env::KV_BLOCKS) && err.contains("cpu"), "{err}");
        // The cuda executor drafts up to the model's three MTP layers.
        for k in 1..=caps::CUDA_DRAFT_TOKENS {
            let mut map = cuda.clone();
            map.insert(env::DRAFT_TOKENS, k.to_string());
            assert_eq!(parse_map(&map).unwrap().sizing.draft_tokens, k);
        }
        let mut map = cuda.clone();
        map.insert(env::DRAFT_TOKENS, (caps::CUDA_DRAFT_TOKENS + 1).to_string());
        let err = parse_map(&map).unwrap_err();
        assert!(
            err.contains(env::DRAFT_TOKENS) && err.contains("cuda") && err.contains("at most 3"),
            "{err}"
        );

        // The cpu executor: its block count required, every cuda setting refused.
        let mut cpu = cuda.clone();
        cpu.insert(env::EXECUTOR, "cpu".into());
        for name in CUDA_SETTINGS {
            cpu.remove(name);
        }
        cpu.insert(env::KV_BLOCKS, "256".into());
        cpu.insert(env::DRAFT_TOKENS, "2".into());
        assert_eq!(
            parse_map(&cpu).unwrap().executor,
            ExecutorSettings::Cpu { kv_blocks: 256 }
        );
        for name in CUDA_SETTINGS {
            let mut map = cpu.clone();
            map.insert(name, cuda[name].clone());
            let err = parse_map(&map).unwrap_err();
            assert!(err.contains(name) && err.contains("cuda"), "{err}");
        }
        let mut map = cpu.clone();
        map.remove(env::KV_BLOCKS);
        assert!(parse_map(&map).unwrap_err().contains(env::KV_BLOCKS));
        for bad in ["1", "0", &(u64::from(caps::KV_BLOCKS) + 1).to_string()] {
            let mut map = cpu.clone();
            map.insert(env::KV_BLOCKS, bad.into());
            assert!(
                parse_map(&map).unwrap_err().contains(env::KV_BLOCKS),
                "{bad}"
            );
        }
    }

    /// A step that cannot hold every seat's decode row is refused: the
    /// scheduler would leave the rows it cannot afford out of each step.
    #[cfg(feature = "argon2")]
    #[test]
    fn a_step_holds_every_seats_decode_row() {
        let mut cpu = cuda_env();
        cpu.insert(env::EXECUTOR, "cpu".into());
        for name in CUDA_SETTINGS {
            cpu.remove(name);
        }
        cpu.insert(env::KV_BLOCKS, "256".into());
        cpu.insert(env::MAX_PREFILL_CHUNK, "2".into());
        // (seats, drafts): the fewest slots are seats × (1 + drafts).
        for (seats, drafts, least) in [(1u32, 2u32, 3u32), (1, 0, 1), (64, 0, 64), (64, 2, 192)] {
            cpu.insert(env::MAX_SEQS, seats.to_string());
            cpu.insert(env::DRAFT_TOKENS, drafts.to_string());
            cpu.insert(env::MAX_BATCHED_TOKENS, least.to_string());
            let m = parse_map(&cpu).unwrap_or_else(|e| panic!("{seats} x (1 + {drafts}): {e}"));
            assert_eq!(decode_step_tokens(&m.sizing), u64::from(least));
            if least > 1 {
                cpu.insert(env::MAX_BATCHED_TOKENS, (least - 1).to_string());
                let err = parse_map(&cpu).unwrap_err();
                assert!(
                    err.contains(
                        "EIDOLA_ENGINE_MAX_BATCHED_TOKENS must be at least \
                         EIDOLA_ENGINE_MAX_SEQS x (1 + EIDOLA_ENGINE_DRAFT_TOKENS)"
                    ),
                    "{err}"
                );
            }
        }
        // The pinned shape, and the cuda executor's widest draft.
        let mut cuda = cuda_env();
        parse_map(&cuda).expect("64 seats x 4 slots fit 8192");
        cuda.insert(env::MAX_BATCHED_TOKENS, "255".into());
        assert!(
            parse_map(&cuda)
                .unwrap_err()
                .contains(env::MAX_BATCHED_TOKENS)
        );
        cuda.insert(env::MAX_BATCHED_TOKENS, "256".into());
        parse_map(&cuda).expect("64 seats x 4 slots are 256");
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn the_cuda_settings_are_exact() {
        let cuda = cuda_env();
        let refused = |name: &'static str, value: &str| {
            let mut map = cuda.clone();
            map.insert(name, value.into());
            parse_map(&map).unwrap_err().contains(name)
        };
        for bad in [
            "0",
            "-1",
            "8GiB",
            " 1024",
            "18446744073709551616",
            &(caps::KV_DEVICE_BYTES + 1).to_string(),
        ] {
            assert!(refused(env::KV_DEVICE_BYTES, bad), "{bad:?}");
        }
        for bad in [
            "kernels",
            "/",
            "/opt/../kernels",
            "/opt/./kernels",
            "/opt//kernels",
            "/opt/kernels/",
        ] {
            assert!(refused(env::KERNELS_DIR, bad), "{bad:?}");
        }
        for bad in ["ON", "true", "1", "auto", "on "] {
            assert!(refused(env::CUDA_GRAPHS, bad), "{bad:?}");
        }
        for bad in [
            "Required",
            "true",
            "off",
            "not_required",
            "notrequired",
            "required ",
            "optional",
        ] {
            assert!(refused(env::GPU_CC, bad), "{bad:?}");
        }
        for (value, gpu_cc) in [
            ("required", GpuCc::Required),
            ("not-required", GpuCc::NotRequired),
        ] {
            let mut map = cuda.clone();
            map.insert(env::GPU_CC, value.into());
            assert!(matches!(
                parse_map(&map).unwrap().executor,
                ExecutorSettings::Cuda { gpu_cc: g, .. } if g == gpu_cc
            ));
            assert_eq!(gpu_cc.as_str(), value);
        }
        let mut off = cuda.clone();
        off.insert(env::CUDA_GRAPHS, "off".into());
        assert!(matches!(
            parse_map(&off).unwrap().executor,
            ExecutorSettings::Cuda {
                graphs: CudaGraphs::Off,
                ..
            }
        ));
        let mut most = cuda.clone();
        most.insert(env::KV_DEVICE_BYTES, caps::KV_DEVICE_BYTES.to_string());
        assert!(parse_map(&most).is_ok());
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn the_token_hash_is_a_whole_argon2id_phc_string() {
        let good = "$argon2id$v=19$m=19456,t=2,p=1$Mz9P1/uk98yKEflNjzvn5g$unNYT/KTNSNW0JCH9+9OQ2zBApPLxGNZiw746903Q8E";
        assert!(parse_gateway_token_hash(good).is_ok());
        assert_eq!(
            parse_gateway_token_hash("$argon2id$garbage").unwrap_err(),
            TokenHashError::NotAPhcString
        );
        assert_eq!(
            parse_gateway_token_hash(&good.replace("argon2id", "argon2i")).unwrap_err(),
            TokenHashError::NotArgon2id
        );
        assert_eq!(
            parse_gateway_token_hash("").unwrap_err(),
            TokenHashError::NotAPhcString
        );
    }

    /// The hashes `parse_gateway_token_hash` refuses are exactly ones no token
    /// verifies against: `verify_password` itself fails for each, even with
    /// the token that would match a well-formed hash.
    #[cfg(feature = "argon2")]
    #[test]
    fn the_token_hash_is_one_a_token_can_verify_against() {
        use argon2::{PasswordHasher, PasswordVerifier};

        let token = b"dev-gateway-token";
        let good = argon2::Argon2::default()
            .hash_password(token)
            .unwrap()
            .to_string();
        let parsed = parse_gateway_token_hash(&good).unwrap();
        argon2::Argon2::default()
            .verify_password(token, &parsed)
            .unwrap();

        let (head, tail) = good.split_once("$m=").unwrap();
        let (_, rest) = tail.split_once('$').unwrap();
        let (salt, output) = rest.split_once('$').unwrap();
        for (bad, error) in [
            (
                format!("{head}$m=1,t=2,p=1${rest}"),
                TokenHashError::Unusable,
            ),
            (
                format!("{head}$m=19456,t=0,p=1${rest}"),
                TokenHashError::Unusable,
            ),
            (
                format!("{head}$m=19456,t=2,p=1$c2FsdA${output}"),
                TokenHashError::NotAPhcString,
            ),
            (
                format!("{head}$m=19456,t=2,p=1"),
                TokenHashError::Incomplete,
            ),
            (
                format!("{head}$m=19456,t=2,p=1${salt}"),
                TokenHashError::Incomplete,
            ),
        ] {
            assert_eq!(parse_gateway_token_hash(&bad).unwrap_err(), error, "{bad}");
            if let Ok(phc) = argon2::PasswordHash::new(&bad) {
                assert!(
                    argon2::Argon2::default()
                        .verify_password(token, &phc)
                        .is_err(),
                    "{bad} verifies, so refusing it is wrong"
                );
            }
        }
    }
}
