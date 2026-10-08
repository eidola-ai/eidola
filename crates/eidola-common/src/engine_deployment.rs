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
    /// Device memory for the KV pools, in bytes; the node derives the
    /// per-group block counts from it (`cuda` only).
    pub const KV_DEVICE_BYTES: &str = "EIDOLA_ENGINE_KV_DEVICE_BYTES";
    /// The kernel build output the CUDA executor loads its images from, each
    /// checked against the compiled-in kernel manifest (`cuda` only).
    pub const KERNELS_DIR: &str = "EIDOLA_ENGINE_KERNELS_DIR";
    /// Whether the CUDA executor replays captured graphs for decode steps:
    /// `on` or `off` (`cuda` only).
    pub const CUDA_GRAPHS: &str = "EIDOLA_ENGINE_CUDA_GRAPHS";
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

/// Caps on the node's allocation-driving sizes. The engine serves one model
/// family, MiMo-V2.6-Flash (`eidola-engine-cuda`'s `support::check_supported`
/// refuses every other), so each cap follows from that model and the CUDA
/// executor's own limits, generous but finite.
pub mod caps {
    /// The model's context window (`support::MAX_POSITIONS`, 2^20).
    pub const MAX_MODEL_LEN: u32 = 1 << 20;
    /// Positions per KV block: a block is the attention kernel's page and the
    /// prefix cache's unit of reuse; beyond 1,024 positions paging stops
    /// paying, and the page stride (32-bit) stays far from overflow.
    pub const KV_BLOCK_SIZE: u32 = 1024;
    /// Device memory for the CUDA executor's KV pools: less than the
    /// supported GPU's memory ([`super::SUPPORTED_GPU_MEMORY_BYTES`]), as the
    /// node itself refuses a value not less than its device's. What a pinned
    /// deployment may give the pools, beside its weights and the executor's
    /// own reserve, is held separately ([`super::check_resources`]).
    pub const KV_DEVICE_BYTES: u64 = super::SUPPORTED_GPU_MEMORY_BYTES - 1;
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
    /// Speculative draft width with the CUDA executor: MiMo-V2.6-Flash's MTP
    /// layers (`num_nextn_predict_layers`), each serving one draft depth
    /// (`eidola-engine-cuda` `support::MTP_LAYERS`).
    pub const CUDA_DRAFT_TOKENS: u32 = 3;
    /// Admission slots: each may hold a request body of up to
    /// `engine_protocol::MAX_REQUEST_BODY_BYTES` and its parse, held to the
    /// VM's memory separately ([`super::check_resources`]).
    pub const MAX_REQUESTS: u32 = 4096;
}

/// KV-cache bytes per position, every layer of MiMo-V2.6-Flash
/// (`eidola-engine-cuda` `support`): 9 global layers of 4 KV heads and 39
/// sliding layers of 8, each head 192 K + 128 V dims, in bf16.
pub const KV_BYTES_PER_POSITION: u64 = (9 * 4 + 39 * 8) * (192 + 128) * 2;

/// KV-cache bytes per position of MiMo-V2.6-Flash's global layers: 9 layers
/// of 4 KV heads, 192 K + 128 V dims, bf16.
pub const GLOBAL_KV_BYTES_PER_POSITION: u64 = 9 * 4 * (192 + 128) * 2;

/// KV-cache bytes per position of MiMo-V2.6-Flash's sliding layers: 39
/// layers of 8 KV heads, 192 K + 128 V dims, bf16.
pub const SLIDING_KV_BYTES_PER_POSITION: u64 = 39 * 8 * (192 + 128) * 2;

/// Positions a sliding layer sees (`eidola-engine-cuda` `support::WINDOW`).
pub const SLIDING_WINDOW: u64 = 128;

/// Drafter KV bytes per position for each draft depth: one MTP layer of
/// MiMo-V2.6-Flash, 8 KV heads of 192 K + 128 V dims, bf16 (`eidola-engine-cuda`
/// `kv::GroupGeometry`, planar layout).
pub const DRAFTER_KV_BYTES_PER_POSITION_PER_DEPTH: u64 = 8 * (192 + 128) * 2;

/// Drafter boundary-tap bytes per block for each draft depth: one chain
/// level of the hidden size (4,096) in f32.
pub const DRAFTER_TAP_BYTES_PER_DEPTH: u64 = 4096 * 4;

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
/// MiB a rung), and the slack of the weights' device layouts over their file
/// bytes (each fused QKV chunk padded to whole 128-row tiles, norm and bias
/// vectors widened to f32): 4 GiB.
pub const DEVICE_FIXED_RESERVE_BYTES: u64 = 4 << 30;

/// Device scratch per token a step may hold (`MAX_BATCHED_TOKENS`), at
/// Flash's shapes (`eidola-engine-cuda` `model::ScratchSizes`, whose logit
/// rows are a step's tokens): the f32 logits row (152,576 × 4 bytes), the
/// residual stream and its projections, the quantized activations, Q, K, V
/// and attention output, the dense FFN, and eight expert rows of 22,580
/// bytes each, about 1.03 MB in all, rounded up to 1.25 MiB.
pub const STEP_TOKEN_DEVICE_BYTES: u64 = 5 << 18;

/// The expert scratch held whatever the step: rows of the masked layout (256
/// experts × 128) or the contiguous layout's per-expert padding, at most
/// 65,536 rows of at most 24 KiB.
pub const EXPERT_FLOOR_DEVICE_BYTES: u64 = 65_536 * (24 << 10);

/// Device memory per sampler row (`eidola-engine-cuda`'s sampler scratch):
/// the f64 distribution over at most the head's 152,576 ids (1,220,608
/// bytes) and the row's buffers, rounded up to 1.25 MiB.
pub const SAMPLER_ROW_DEVICE_BYTES: u64 = 5 << 18;

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
/// level rows (`draft::Levels`: `k` loaded, `k (k + 1) / 2` from the draft
/// phase, `2 (k + 1)` write-only).
pub fn drafter_seat_rows(draft_tokens: u32) -> u64 {
    let k = u64::from(draft_tokens);
    if k == 0 {
        return 0;
    }
    (k + 1) * k + k + k * (k + 1) / 2 + 2 * (k + 1)
}

/// Step-table bytes a drafting executor may hold per seat beside its global
/// page lists, in a graph's table and an eager step's together: every other
/// per-row array of a drafted step (copy lists, token ids, positions, KV
/// targets, work lists, the drafter's one-position page lists, sampler and
/// acceptance rows; about 1,300 words a table for three depths), bounded by
/// 32 KiB. An eager step's arrays per token beyond those (about 30 words for
/// three depths) are inside [`STEP_TOKEN_DEVICE_BYTES`]'s rounding.
pub const DRAFT_TABLE_SEAT_BYTES: u64 = 32 << 10;

/// RoPE table bytes per position (`MAX_MODEL_LEN`): 64 f32 for each distinct
/// θ, at most three.
pub const ROPE_DEVICE_BYTES_PER_POSITION: u64 = 3 * 64 * 4;

/// Device memory the CUDA executor takes beside its weights and KV pools,
/// bounded from above: [`DEVICE_FIXED_RESERVE_BYTES`], plus
/// [`STEP_TOKEN_DEVICE_BYTES`] per `MAX_BATCHED_TOKENS`,
/// [`EXPERT_FLOOR_DEVICE_BYTES`], per `MAX_SEQS` [`sampler_rows`] of
/// [`SAMPLER_ROW_DEVICE_BYTES`], the device block tables (`MAX_SEQS` ×
/// ⌈`MAX_MODEL_LEN` / `KV_BLOCK_SIZE`⌉ `i32` entries per KV group, two
/// groups, three with drafting) and the RoPE tables
/// ([`ROPE_DEVICE_BYTES_PER_POSITION`] per `MAX_MODEL_LEN`). With
/// `DRAFT_TOKENS` `k` > 0, per `MAX_SEQS` also [`drafter_seat_rows`] rows of
/// 16 KiB, [`DRAFT_TABLE_SEAT_BYTES`], and the global page lists of a
/// drafted step's table twice over (a graph's and an eager step's, one
/// `i32` per block of `MAX_MODEL_LEN`). The MTP layers' own scratch (their
/// copy lists, 36 bytes a token) is inside [`STEP_TOKEN_DEVICE_BYTES`]'s
/// rounding, and their weights inside the pack. Saturates.
pub fn cuda_device_reserve_bytes(sizing: &Sizing) -> u64 {
    let blocks_per_seq =
        u64::from(sizing.max_model_len).div_ceil(u64::from(sizing.kv_block_size.max(1)));
    let seats = u64::from(sizing.max_seqs);
    let drafting = sizing.draft_tokens > 0;
    let groups = if drafting { 3 } else { 2 };
    let per_seat = sampler_rows(sizing.draft_tokens)
        .saturating_mul(SAMPLER_ROW_DEVICE_BYTES)
        .saturating_add(drafter_seat_rows(sizing.draft_tokens).saturating_mul(16 << 10))
        .saturating_add(if drafting {
            DRAFT_TABLE_SEAT_BYTES.saturating_add(blocks_per_seq.saturating_mul(2 * 4))
        } else {
            0
        });
    DEVICE_FIXED_RESERVE_BYTES
        .saturating_add(
            u64::from(sizing.max_batched_tokens).saturating_mul(STEP_TOKEN_DEVICE_BYTES),
        )
        .saturating_add(EXPERT_FLOOR_DEVICE_BYTES)
        .saturating_add(seats.saturating_mul(per_seat))
        .saturating_add(
            seats
                .saturating_mul(blocks_per_seq)
                .saturating_mul(groups * 4),
        )
        .saturating_add(
            u64::from(sizing.max_model_len).saturating_mul(ROPE_DEVICE_BYTES_PER_POSITION),
        )
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
/// - the block tables: `MAX_SEQS` × ⌈`MAX_MODEL_LEN` / `KV_BLOCK_SIZE`⌉
///   `i32` entries per KV group, two groups;
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
    let tables = u64::from(sizing.max_seqs) * blocks_per_seq * 4 * 2;
    read_pool + admission_pool + tables + PROCESS_HEADROOM_BYTES
}

/// The fewest bytes of device memory the CUDA executor's KV pools need: the
/// node's own derivation (`cuda::derive_kv_blocks` in `eidola-server-engine`)
/// at its smallest. With `B` = `KV_BLOCK_SIZE`,
/// `R` = ⌈([`SLIDING_WINDOW`] − 1) / `B`⌉ + 1, and `P` =
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
/// write), each block `B` positions of [`SLIDING_KV_BYTES_PER_POSITION`],
/// [`GLOBAL_KV_BYTES_PER_POSITION`] or `k` ×
/// [`DRAFTER_KV_BYTES_PER_POSITION_PER_DEPTH`], a drafter block also `k` ×
/// [`DRAFTER_TAP_BYTES_PER_DEPTH`] of boundary tap. Saturates rather than
/// wraps.
pub fn cuda_kv_min_bytes(sizing: &Sizing, cache: &CacheConfig) -> u64 {
    let b = u64::from(sizing.kv_block_size.max(1));
    let points = if cache.enabled { RETENTION_POINTS } else { 0 };
    let window_blocks = (SLIDING_WINDOW - 1).div_ceil(b) + 1;
    let sliding_blocks = u64::from(sizing.max_seqs)
        .saturating_mul((1 + points).saturating_mul(window_blocks).saturating_add(1))
        .saturating_add(1 + u64::from(sizing.max_batched_tokens).div_ceil(b));
    let global_blocks = u64::from(sizing.max_model_len).div_ceil(b) + 1;
    let pool = |blocks: u64, per_position: u64| {
        blocks
            .saturating_add(1)
            .saturating_mul(b)
            .saturating_mul(per_position)
    };
    let k = u64::from(sizing.draft_tokens);
    let drafter = if k == 0 {
        0
    } else {
        sliding_blocks.saturating_add(1).saturating_mul(
            b.saturating_mul(k * DRAFTER_KV_BYTES_PER_POSITION_PER_DEPTH)
                .saturating_add(k * DRAFTER_TAP_BYTES_PER_DEPTH),
        )
    };
    pool(sliding_blocks, SLIDING_KV_BYTES_PER_POSITION)
        .saturating_add(pool(global_blocks, GLOBAL_KV_BYTES_PER_POSITION))
        .saturating_add(drafter)
}

/// Whether the deployment's VM and GPUs hold what its configuration
/// allocates, as far as configuration alone fixes it. Host memory (`memory`,
/// MiB) holds [`host_memory_bytes`], plus, for the `cpu` executor, its KV
/// cache (`KV_BLOCKS` × `KV_BLOCK_SIZE` × [`KV_BYTES_PER_POSITION`]). For the
/// `cuda` executor, which runs on one [`SUPPORTED_GPU`], the VM has a GPU,
/// `KV_DEVICE_BYTES` holds at least [`cuda_kv_min_bytes`], and
/// `KV_DEVICE_BYTES` + `weights_bytes` + [`cuda_device_reserve_bytes`] fit
/// [`SUPPORTED_GPU_MEMORY_BYTES`]. `weights_bytes` bounds the weights' device
/// footprint from above: the pack's data size ([`artifact_data_bytes`]),
/// every file of it, whether the node loads it or not.
pub fn check_resources(
    sizing: &Sizing,
    cache: &CacheConfig,
    executor: &ExecutorSettings,
    weights_bytes: u64,
    memory_mib: u64,
    gpus: u64,
) -> Result<(), String> {
    let host_kv = match executor {
        ExecutorSettings::Cpu { kv_blocks } => u64::from(*kv_blocks)
            .saturating_mul(u64::from(sizing.kv_block_size))
            .saturating_mul(KV_BYTES_PER_POSITION),
        ExecutorSettings::Cuda { .. } => 0,
    };
    let host = host_memory_bytes(sizing).saturating_add(host_kv);
    let host_limit = memory_mib.saturating_mul(1 << 20);
    if host > host_limit {
        return Err(format!(
            "the node's request pools, block tables, host KV and headroom need {host} bytes, \
             more than the VM's {host_limit} (memory)"
        ));
    }
    if let ExecutorSettings::Cuda {
        kv_device_bytes, ..
    } = executor
    {
        if gpus == 0 {
            return Err("the cuda executor needs a GPU, and the VM has none".into());
        }
        let needed = cuda_kv_min_bytes(sizing, cache);
        if *kv_device_bytes < needed {
            return Err(format!(
                "{} ({kv_device_bytes}) is less than the {needed} bytes the KV pools need at \
                 their smallest (one {} sequence, every seat's sliding window, pad blocks)",
                env::KV_DEVICE_BYTES,
                env::MAX_MODEL_LEN
            ));
        }
        let reserve = cuda_device_reserve_bytes(sizing);
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

/// Fewest vCPUs a deployment may give its VM.
pub const MIN_CPUS: u64 = 1;
/// Most vCPUs a deployment may give its VM.
pub const MAX_CPUS: u64 = 256;
/// Smallest VM memory, in MiB.
pub const MIN_MEMORY_MIB: u64 = 8192;
/// Largest VM memory, in MiB.
pub const MAX_MEMORY_MIB: u64 = 2 * 1024 * 1024;

/// Whether the VM's `cpus` and `memory` (MiB) are a shape the platform
/// launches: `cpus` from [`MIN_CPUS`] to [`MAX_CPUS`], and `memory` a power of
/// two from [`MIN_MEMORY_MIB`] (8 GiB) to [`MAX_MEMORY_MIB`] (2 TiB), the sizes
/// the platform provider's deployment accepts. Tinfoil's published checks
/// (`measure-image-action`) require only positive values, so these bounds are
/// ours; the same rule holds for every `cvm-version` an engine pins.
pub fn check_vm_resources(cpus: u64, memory_mib: u64) -> Result<(), String> {
    if !(MIN_CPUS..=MAX_CPUS).contains(&cpus) {
        return Err(format!(
            "cpus must be from {MIN_CPUS} to {MAX_CPUS} (got {cpus})"
        ));
    }
    if !memory_mib.is_power_of_two() || !(MIN_MEMORY_MIB..=MAX_MEMORY_MIB).contains(&memory_mib) {
        return Err(format!(
            "memory must be a power of two from {MIN_MEMORY_MIB} to {MAX_MEMORY_MIB} MiB (got {memory_mib})"
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
        /// Device memory for the KV pools, in bytes.
        kv_device_bytes: u64,
        graphs: CudaGraphs,
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
            let kv_device_bytes = match get(env::KV_DEVICE_BYTES)?.parse::<u64>() {
                Ok(n) if n > 0 => n,
                _ => {
                    return Err(ConfigError(format!(
                        "{} must be a positive number of bytes",
                        env::KV_DEVICE_BYTES
                    )));
                }
            };
            if kv_device_bytes > caps::KV_DEVICE_BYTES {
                return Err(ConfigError(format!(
                    "{} must be at most {}",
                    env::KV_DEVICE_BYTES,
                    caps::KV_DEVICE_BYTES
                )));
            }
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
            ExecutorSettings::Cuda {
                kernels_dir,
                kv_device_bytes,
                graphs,
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
    // The scheduler's own boot refusal (`Engine::new`): a decode row costs
    // `1 + draft_tokens` query slots, so a step must hold more than
    // `draft_tokens` of them. The engine's other boot refusals that depend
    // only on these values (a zero seat, prefill chunk or block count) are the
    // positivity rules above; the rest need the model (its position limit and
    // MTP depth) or the node build (its executor).
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
    if sizing.max_batched_tokens <= sizing.draft_tokens {
        return Err(ConfigError(format!(
            "{} must exceed {}: a decode row needs 1 + {} query slots",
            env::MAX_BATCHED_TOKENS,
            env::DRAFT_TOKENS,
            env::DRAFT_TOKENS
        )));
    }
    // The CUDA executor drafts with MiMo-V2.6-Flash's MTP layers, one per
    // depth: no wider than the checkpoint it serves ships.
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
    })
}

/// A deployment path component (a model id, a variant):
/// `[a-z0-9][a-z0-9._-]*`. `deploy/engine/<model>/<variant>/` directories
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

/// The GPU counts a confidential NVIDIA deployment may attach (the
/// config's top-level `gpus`): the platform provider's NVIDIA-CC shapes.
pub const CUDA_GPU_COUNTS: &[u64] = &[1, 8];

/// Whether a deployment's GPUs are attested on every handshake.
///
/// The attestation shim collects exactly the config's top-level `gpus` worth
/// of GPU evidence items (Tinfoil's `shimConfig.ExpectedGPUs = config.GPUs`),
/// and a pin's `expected_gpus` requires exactly that many. So:
///
/// - the `cuda` executor needs GPUs, in one of [`CUDA_GPU_COUNTS`], and a pin
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
                "`gpus: {n}` is not an NVIDIA-CC shape; it must be one of {CUDA_GPU_COUNTS:?}"
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
        assert!(check_gpu_attestation(Cuda, Some(8), Some(8)).is_ok());
        assert!(check_gpu_attestation(Cuda, Some(1), Some(1)).is_ok());
        assert!(check_gpu_attestation(Cpu, None, None).is_ok());
        assert!(check_gpu_attestation(Cpu, Some(0), Some(0)).is_ok());
        for (executor, gpus, expected) in [
            (Cuda, None, Some(4)),
            (Cuda, None, None),
            (Cuda, Some(0), Some(8)),
            (Cuda, Some(2), Some(2)),
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
            graphs: CudaGraphs::On,
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
        assert!(check_resources(&sizing, &cache, &kv, FIXTURE_PACK, 65_536, 8).is_ok());
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
        let near = |n| Sizing {
            max_requests: n,
            ..sizing
        };
        // Six slots, the headroom and the tables fit 8 GiB; eight do not.
        assert!(check_resources(&near(6), &cache, &kv, FIXTURE_PACK, 8192, 8).is_ok());
        assert!(check_resources(&near(8), &cache, &kv, FIXTURE_PACK, 8192, 8).is_err());
        // Near the limit: 99 slots fit 64 GiB; 100 do not.
        assert!(check_resources(&near(99), &cache, &kv, FIXTURE_PACK, 65_536, 8).is_ok());
        assert!(check_resources(&near(100), &cache, &kv, FIXTURE_PACK, 65_536, 8).is_err());
        // The cuda executor needs a GPU.
        assert!(check_resources(&sizing, &cache, &kv, FIXTURE_PACK, 65_536, 0).is_err());
        // Block size 1 at the longest model length: 2^20 entries per slot.
        let tables = Sizing {
            kv_block_size: 1,
            max_model_len: caps::MAX_MODEL_LEN,
            max_seqs: caps::MAX_SEQS,
            ..sizing
        };
        assert!(check_resources(&tables, &cache, &kv, FIXTURE_PACK, 16_384, 8).is_err());
        // The cpu executor's KV is host memory: 2^20 blocks of 16 positions
        // are 3.4 TiB.
        let cpu = |kv_blocks| ExecutorSettings::Cpu { kv_blocks };
        assert!(check_resources(&sizing, &cache, &cpu(1024), FIXTURE_PACK, 65_536, 0).is_ok());
        assert!(check_resources(&sizing, &cache, &cpu(1 << 20), FIXTURE_PACK, 65_536, 0).is_err());
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
        assert_eq!(cuda_kv_min_bytes(&sizing, &cache(true)), 12_228_280_320);
        // Without the cache, no retained windows: 1 + 64 × 10 + 512 = 1,153.
        assert_eq!(cuda_kv_min_bytes(&sizing, &cache(false)), 6_707_527_680);
        // Drafting k tokens adds a drafter pool of the sliding pool's
        // blocks, each 16 positions × k × 5,120 bytes and a tap of k × 16 KiB.
        for k in 1..=3u64 {
            let drafting = Sizing {
                draft_tokens: u32::try_from(k).unwrap(),
                ..sizing
            };
            assert_eq!(
                cuda_kv_min_bytes(&drafting, &cache(true)),
                12_228_280_320 + 2_882 * (16 * k * 5_120 + k * 16_384)
            );
        }
        let min = cuda_kv_min_bytes(&sizing, &cache(true));
        let fits =
            |bytes| check_resources(&sizing, &cache(true), &cuda(bytes), FIXTURE_PACK, 65_536, 1);
        assert!(fits(min).is_ok());
        assert!(fits(min - 1).is_err());
    }

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
        // entries × 2 groups × 4 bytes; 131,072 positions × 768 bytes of RoPE
        // tables.
        let reserve = cuda_device_reserve_bytes(&sizing);
        assert_eq!(
            reserve,
            (4 << 30)
                + 8192 * (5 << 18)
                + 65_536 * (24 << 10)
                + 64 * (5 << 18)
                + 64 * 8192 * 8
                + 131_072 * 768
        );
        assert_eq!(reserve, 16_831_741_952);
        // A sampler row holds an f64 distribution over every id of the head.
        const { assert!(SAMPLER_ROW_DEVICE_BYTES >= 152_576 * 8) };
        // Drafting three tokens: per seat 8 sampler rows, 29 rows of 16 KiB
        // (12 of drafter state, 3 + 6 + 8 level rows), 32 KiB of step tables
        // and two global page lists of 8,192 entries; and a third group's
        // block tables.
        let drafting = Sizing {
            draft_tokens: 3,
            ..sizing
        };
        assert_eq!(sampler_rows(3), 8);
        assert_eq!(drafter_seat_rows(3), 29);
        assert_eq!(
            cuda_device_reserve_bytes(&drafting),
            (4 << 30)
                + 8192 * (5 << 18)
                + 65_536 * (24 << 10)
                + 64 * (8 * (5 << 18) + 29 * (16 << 10) + (32 << 10) + 8192 * 8)
                + 64 * 8192 * 12
                + 131_072 * 768
        );
        for k in 1..=3 {
            assert!(
                cuda_device_reserve_bytes(&Sizing {
                    draft_tokens: k,
                    ..sizing
                }) > reserve
            );
        }
        let fits = |kv: u64, weights: u64, gpus: u64| {
            check_resources(&sizing, &cache, &cuda(kv), weights, 65_536, gpus)
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
        for (cpus, memory) in [
            (0, 65536),
            (MAX_CPUS + 1, 65536),
            (16, 1),
            (16, 4096),
            (16, 65535),
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
            (env::MAX_MODEL_LEN, "131072"),
            (env::MAX_SEQS, "64"),
            (env::MAX_BATCHED_TOKENS, "8192"),
            (env::MAX_PREFILL_CHUNK, "4096"),
            (env::DRAFT_TOKENS, "0"),
            (env::MAX_REQUESTS, "8"),
            (env::PREFIX_CACHE, "true"),
            (env::CACHE_IDLE_TTL_SECS, "900"),
            (env::CACHE_MAX_AGE_SECS, "3600"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect()
    }

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
                graphs: CudaGraphs::On,
            }
        );
        // Every cuda setting is required with cuda.
        for name in [env::KV_DEVICE_BYTES, env::KERNELS_DIR, env::CUDA_GRAPHS] {
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
        for name in [env::KV_DEVICE_BYTES, env::KERNELS_DIR, env::CUDA_GRAPHS] {
            cpu.remove(name);
        }
        cpu.insert(env::KV_BLOCKS, "256".into());
        cpu.insert(env::DRAFT_TOKENS, "2".into());
        assert_eq!(
            parse_map(&cpu).unwrap().executor,
            ExecutorSettings::Cpu { kv_blocks: 256 }
        );
        for name in [env::KV_DEVICE_BYTES, env::KERNELS_DIR, env::CUDA_GRAPHS] {
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

    /// A step that cannot hold one decode row is refused, as the node's
    /// scheduler would refuse it at boot.
    #[cfg(feature = "argon2")]
    #[test]
    fn a_step_holds_a_decode_row() {
        let mut cpu = cuda_env();
        cpu.insert(env::EXECUTOR, "cpu".into());
        for name in [env::KV_DEVICE_BYTES, env::KERNELS_DIR, env::CUDA_GRAPHS] {
            cpu.remove(name);
        }
        cpu.insert(env::KV_BLOCKS, "256".into());
        cpu.insert(env::DRAFT_TOKENS, "2".into());
        cpu.insert(env::MAX_BATCHED_TOKENS, "2".into());
        cpu.insert(env::MAX_PREFILL_CHUNK, "2".into());
        let err = parse_map(&cpu).unwrap_err();
        assert!(
            err.contains("EIDOLA_ENGINE_MAX_BATCHED_TOKENS must exceed EIDOLA_ENGINE_DRAFT_TOKENS"),
            "{err}"
        );
        cpu.insert(env::MAX_BATCHED_TOKENS, "3".into());
        parse_map(&cpu).expect("three slots hold a decode row with two drafts");
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
