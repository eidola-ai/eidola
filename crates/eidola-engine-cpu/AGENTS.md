# eidola-engine-cpu

The CPU reference executors of the engine. `CpuExecutor` implements the serving core's `Executor` (`eidola-engine`) for MiMo-V2.6 with the model crate's f32 reference numerics (`eidola-engine-model`), over paged KV, with MTP drafting and sampling inside the step. `DsCpuExecutor` (`src/deepseek/`) does the same for DeepSeek-V4.1, with KV in the model's packed formats and the model split across two logical devices; it has its own section, [DeepSeek-V4.1](#deepseek-v41), and everything before that section is about MiMo's. AGPL-3.0-only. Workspace conventions live in the root `AGENTS.md`.

It is not a serving backend. It exists for two reasons:

- **It is the oracle for every other executor.** A GPU executor is diffed against it row for row: same step inputs, same tokens, logits equal within that executor's documented tolerance.
- **It proves the contract can be met exactly.** Paged, chunked, prefix-cached, preempted and speculative execution through the real scheduler reproduces the dense reference forward bit for bit.

| Module | Contents |
|---|---|
| `executor.rs` | `CpuExecutor`, `CpuExecutorConfig`, `MtpHidden`, `RowRecord`. |
| `pool.rs` | Physical KV pools, laid out by block, with per-row position tags. |
| `oracle.rs` | `DenseOracle`: the dense forward plus every MTP depth in the executor's row layout; `dense_generate`, `check_generation`. |

## What the executor does

It follows the seam contract in `eidola-engine/src/executor.rs`, which is normative. This section covers only what that contract leaves to the executor.

**Memory.** There is one pool per KV group:

- target groups, one per (attention kind, KV shape): global and sliding for MiMo;
- one drafter group (`KvRole::Drafter`), whose layers are the draft depths.

A block holds every layer's `[K | V]` rows for `block_size` positions, contiguously. `Maintenance::Zero` overwrites the whole block with zero bytes, including the drafter tap and the tags. `Copy` copies all of it. `Encrypt` and `Decrypt` XOR all of it with the block's ChaCha20 keystream (`eidola_engine::kv_cipher`, called directly, so the reference *is* the definition), as four ranges each from a fresh keystream block: the KV rows, the tap, the row tags and the tap tag (`pool.rs`). Tags are encrypted too, so an idle block holds no position or token in the clear and a read of a block left encrypted fails its tag check; the pool records which blocks are encrypted (instrumentation, like the tags) and panics on a double encryption or a decryption of plaintext. The executor keeps no key: it uses the step's `kv_keys` and drops nothing it did not borrow.

Every pool's buffers (data, taps, tags) are page-aligned allocations of whole pages that no other allocation shares, zeroed before they are freed (`pool.rs` tests the alignment and the scrub).

**The host tier** (`CpuExecutorConfig::host_blocks`, 0 for none) is a second pool per group of the same layout, `mlock`ed when it is built (the executor refuses to start if that fails) and unlocked, after the scrub, when dropped; since a pool's pages are its own, that unlock releases no other allocation's lock. `Demote` copies a block's whole image (data, tap, tags, and the encrypted flag) into it and panics unless the block is encrypted; `Promote` copies one back and panics unless the host block holds ciphertext and the device block is zero; `ZeroHost` zeroes a host block. `kv_block_bytes` is what each host block costs (the node sizes the tier from `EIDOLA_ENGINE_KV_HOST_BYTES` with it). `tests/host_tier.rs` drives it through the scheduler: the host holds the first entry's ciphertext word for word and never its plaintext, a hit promoted and decrypted is bit-exact against the dense oracle (with and without speculation), outputs equal a device hit and a recompute, and a randomized workload under pressure in both tiers stays bit-exact; `tests/executor.rs` checks the seam operations directly. The locked pools count against `RLIMIT_MEMLOCK`, which CI runners keep small. So these tests raise the soft limit to the hard one and keep each tier to a few MB. The `host_tier` tests also run one at a time. An executor that still cannot lock its tier refuses with a message naming the limit.

Per-slot model state (the drafter's hidden states at the slot's last position) lives in buffers allocated once at full size and only overwritten in place, as device-resident state would be. `ResetSlot` fills them with zeros; it never drops them for fresh allocations, which would leave the activations in freed memory. `slot_state_is_zero` makes that observable. Step-local scratch (a step's activations, logits and draft distributions) is ordinary heap memory: this executor is a reference, not a serving backend, and a GPU executor's equivalent is device scratch overwritten by the next step.

Block tables are per slot and change only through `TableUpdate` and `ResetSlot`. KV is read only through them. Each stored row carries a tag (position and token). A read that lands on an unmapped block, a zeroed row or another position's row panics, and so does a write to a block mapped more than once. Contract violations are host bugs, so they panic rather than return an error.

**A step** runs in this order:

1. Maintenance, then table updates.
2. The target over every host token.
3. The drafter over every host position, depth by depth.
4. The draft chain.
5. The target over the drafts.
6. Sampling and `chain_accept`.
7. Drafter rows for accepted drafts that the chain did not already compute.
8. Drafter state and boundary taps.

The target runs in two passes (steps 2 and 5). Row independence makes that identical to one pass. The CUDA executor runs one pass, ordering the drafter first for decode rows (their drafter inputs all come from earlier steps), and so drafts only decode rows, which are the only rows the serving core gives drafts (`eidola-engine-cuda/AGENTS.md` → Drafting).

`pad_batches` pads the target batch to the bucket's token count. A test shows the padding rows change nothing.

**Sampling** calls `eidola_engine::sampling` (`sample`, `processed_probs`, `sample_from`, `chain_accept`) directly, so it matches the core's semantics bit for bit. Every row goes through `Logits` with `CpuExecutorConfig::sampleable_vocab_size` (the tokenizer's vocabulary size, never taken from the weights), so target samples, drafts, acceptance and the residual cover only real token ids; `tests/engine.rs` rigs the head's padded rows to win and checks none is ever produced. Draft draws use `Stream::Draft` at the drafted token's position. A greedy draft is the drafter's argmax, with a one-hot `q`.

## MTP row layout

This is the part of the design a GPU executor must copy.

**Rows.** MTP depth `d`'s row at **slot** `s`:

- consumes the token at `s`, and the **main model's** hidden state at the *anchor* `a = s - 1 - d` (level `d` at `s - 1`, where level `l` at `x` is the main model's state at `x - l`);
- uses RoPE position `a`;
- writes its KV at `s`;
- predicts the token at `s + 1`.

Rows exist for `s >= d + 1`. The draft for position `p + 1 + i` is depth `i`'s prediction at slot `p + i`, where `p` is the row's last host position: every depth is anchored at `p - 1`. A depth's output feeds only its logits and KV, never the next depth.

**The vendor's convention.** In the vendor's own indexing (by anchor), MTP layer `k` at anchor `x` combines the main model's normed `h_x` with `t_{x+k+1}`, attends at RoPE position `x` over its own KV (one row per anchor), and predicts `t_{x+k+2}`. That is SGLang's multi-layer MTP worker (`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`): MiMo is not in its `chain_mtp_hidden_states` architecture list ("Non-chain: each step uses the target model's hidden states"), so at prefill every layer is given the target's hidden states, with the input ids shifted one more per layer, and during drafting every step feeds the target's state at the last verified position rather than the previous layer's output; vLLM's MiMo MTP model (`vllm/model_executor/models/mimo_v2_mtp.py`) serves only layer 0, which agrees at depth 0. The slot layout is that indexing with the row moved from the anchor to the token it consumes (`s = x + k + 1`). A DeepSeek-style chain (depth `d` fed depth `d - 1`'s output at RoPE `s - 1`) is a different function: Flash's depth-2 acceptance under it measured 0.23 against depth 1's 0.93, and depth 3's 0.003. `tests/vendor_mtp.rs` checks the executor's drafter logits and greedy drafts bit for bit against a direct anchor-indexed transcription that shares no code with the executor or the oracle, and fails under the chained convention at depth 1.

**Why this layout.** Indexing a row by the token it consumes makes drafter KV at `s` a function of tokens `0 ..= s` alone. That is the core's block invariant (every block depends only on the tokens it covers), so drafter blocks are sealed and shared exactly like target blocks. The core's executor contract makes this layout normative.

The alternative is to index the row by the hidden state it continues from (`p` for `(h_p, t_{p+1})`, as vLLM stores it). Under that indexing, a block's last drafter row depends on the next block's first token, which the block hash does not cover, so a prefix hit with a different continuation would reuse wrong drafter KV. vLLM avoids the problem by dropping the last matched block on a hit.

**The draft chain.** With this layout, every speculative drafter row sits at a slot the host reserved for drafts (`p + 1 ..= p + k`). Depth `i` needs `i` speculative rows there, at slots `p + 1 ..= p + i`, each consuming drafts `1 ..= i` and the main model's states at `p - 1 ..= p - 1 - i + j`, all known before the step's target pass. Each such row is exact once its drafts are accepted. Step 7 fills in only the accepted rows the chain did not cover.

**Boundary taps.** A row continuing at context `c` needs levels `0 .. k` at `c - 1`: the main model's states at `c - 1 ..= c - k`.

- For a running sequence they are per-slot state, left by the previous step at its last valid position.
- After a prefix hit or a resume, the slot is fresh. The host only resumes on block boundaries, so the executor stores, in every drafter block, the levels at the block's last position (the *tap*). A fresh slot loads its state from the tap of the block ending at `c - 1`.

Taps are a function of the prefix, are zeroed and copied with their block, and cost `k · hidden` floats per drafter block. Resuming anywhere else panics.

**Which hidden state.** The oracles disagree on which of the main model's hidden states feeds the MTP layers:

| Source | State fed forward |
|---|---|
| vLLM and SGLang (the vendor's serving path) | the normed state (after the main model's `norm`) |
| llama.cpp | the pre-norm state |

`MtpHidden::Normed` is the default and `MtpHidden::PreNorm` is selectable. Both are exact against their oracle. The two are different functions (a test asserts this). Which one the checkpoint was trained for is settled by measuring acceptance rates on a GPU against the vendor's serving stack. The random-weight fixture cannot tell them apart.

**Depths.** `mtp_depths` maps each depth to a loaded MTP layer, and its length is `k`. Flash ships 3 layers, so the default is `[0, 1, 2]` (SGLang's multi-layer MTP). The committed fixture ships 2, so the tests also run `[0, 1, 0]` to get `k = 3`.

A row at position 0 cannot draft, because no hidden state precedes it. The seam forbids the host from asking (it never reserves drafts there), and the executor panics if asked, rather than quietly sampling plainly.

Block drafters (DFlash) fit the same seam: `num_drafts = k`, context KV in the drafter group, and target taps in per-slot state. Nothing here assumes MTP outside the drafter code.

## Numerics: why it is bit-exact

The executor calls the model crate's public kernels in the reference's order:

- `linear`, `rms_norm_rows`, `rope_cos_sin`/`apply_rope`, `attend`;
- `ReferenceModel::moe` and `dense_ffn`;
- the shared `lm_head`.

Attention gathers keys and values through the block tables in ascending position, which is exactly the set and order the dense reference passes to `attend`.

Every kernel computes one output row at a time in a fixed order, so a row's bits do not depend on the batch, the padding, the slot, the physical block or the chunking. KV is stored and read as exact f32 copies.

No tolerance is needed anywhere. The tests compare `f32::to_bits`.

## Proofs and how to run them

```sh
cargo test -p eidola-engine-cpu          # ~6 min unoptimised; no network, no Python
```

The tests use the model crate's committed synthetic fixture (4 layers, window 8, 2 MTP layers, vocabulary 256), plus models derived from it in Rust:

- the vocabulary cut to its first few ids, for statistics;
- an *echo* drafter, whose MTP layers carry the main model's state through, so each depth repeats the target's prediction at its anchor and greedy chains on a repeating target are accepted whole.

The random fixture's MTP layers are unrelated to its target, so its greedy acceptance is near zero. Under sampling it accepts about 5–20 %.

**Checking.** With `record`, the executor logs every row's tokens (the prefix read back through the block tables), its target logits at every computed position, its drafter logits at every drafter row, its drafts and its output. `tests/common` checks each record bit for bit against `DenseOracle` on the same tokens. It also recomputes every draft draw and `chain_accept` from the dense logits, and checks every finished output against dense generation.

| File | What it proves |
|---|---|
| `tests/engine.rs` | Through the engine: single requests (greedy and seeded), chunk sizes 1–1000, greedy speculation equals plain at `k` = 1, 2, 3, seeded sampling with three depths, prefix hits and salt isolation (including drafter resume from taps), divergent turns after a shared prefix hitting at the branch with outputs equal to cold runs, preemption and resume, idle-TTL eviction and eviction mid-workload (exactly the evicted blocks zeroed in the pools), cancellation, `PreNorm` chaining, unfolded value scale, padded batches, and a randomized workload. The randomized workload covers multi-turn keyed conversations, private requests, cancellations, TTL jumps, small pools and `k` ∈ {1, 2, 3}, and checks the KV invariants, "every free block is zero or queued" and "every free slot's model state is zero or its reset queued" after every step. |
| `tests/executor.rs` | Driven directly through the seam: zero really zeroes (and only the named block), `ResetSlot` really scrubs the slot's model state (and only that slot's), a copied block plus its tap resumes exactly in a fresh slot, encryption changes essentially every stored word of a block (KV, tap, tags) in every group and nothing else, another key gives other ciphertext, decryption restores every bit and the sequence then continues exactly as the dense oracle, zeroing an encrypted block scrubs it, and rows are independent of batch, slot and chunking. Panics on unmapped reads, zeroed reads, reads of an encrypted block, decrypting plaintext, shared-block writes, resuming off a boundary, and too many drafts. |
| `tests/idle_encryption.rs` | Through the engine, with and without speculation: a keyed request's cached blocks, in every group, are ciphertext in the pools once it ends (against their plaintext read back while mapped), another lineage's copy of the same plaintext is other ciphertext, a hit on the first lineage decrypts and every record is bit-exact against the dense oracle, and the outputs equal a run with the prefix cache disabled; no key is held afterwards. A lineage's blocks, expired and recomputed beside another sequence and in prefill chunks that straddle block boundaries, seal to the same ciphertext word for word in every group (fails under a one-ulp KV difference that depends on the batch or the chunking). The harness also checks after every step that idle blocks are encrypted and mapped ones plaintext, so every workload above (divergent turns at the branch, preemption and resume, the randomized workload) runs through encryption and decryption. |
| `tests/vendor_mtp.rs` | Every drafter logit row and every greedy draft, bit for bit, against a direct transcription of the vendor's anchor-indexed MTP inference (layer `k` at anchor `x`: `h_x`, `t_{x+k+1}`, RoPE `x`), sharing no code with the executor or `DenseOracle`, at three depths. Fails at depth 1 under a chained convention. |
| `tests/speculative.rs` | Echo drafter: greedy speculation equals plain decoding, with about 97 % acceptance and whole chains accepted. Vocabulary 4: the joint distribution of three speculative samples (three chained depths) passes a chi-square fit against the exact dense distribution at p = 0.001, for plain temperature and for filtered sampling. |
| `tests/real_flash.rs` (ignored) | The truncated real Flash checkpoint (layers 0, 1, 2, 5 plus 3 MTP layers): chunked prefill, MTP-3 speculation and a prefix hit through the engine. Every logit row is bit-exact against the dense forward. |

**Every proof fails under a deliberate bug.** Each of these mutations was checked to fail the suite:

- drafter RoPE at `s` instead of `s - 1`;
- no post-acceptance drafter rows;
- no boundary taps;
- `Zero` leaving the data;
- the window off by one;
- ignoring per-slot state;
- the wrong random stream for drafts;
- a chain row skipped;
- a drafter reading below its first row;
- a boundary tap holding the previous position's levels (caught at a branch hit);
- no encryption queued when a block goes idle, and no decryption queued on a hit (in the core; the second caught by the tag check on read);
- the executor ignoring every encryption and decryption;
- a drafter block's tap left in the clear;
- a pool buffer aligned only to its element type, and a scrub that leaves its data.

Keep that true when changing the tests.

The ignored real-weights test needs the cached checkpoint that the model crate's `golden/fetch_truncated.py` produces:

```sh
cargo test --release -p eidola-engine-cpu --test real_flash -- --ignored --nocapture
```

Measured on a 16-core Apple-silicon machine (128 GB), release build:

- **Load:** 5 s.
- **Run:** request 1 (40-token prompt, chunk 16, 4 greedy tokens with MTP-3) takes 6 steps in 4.2 s. Request 2 (60-token prompt, 32-token hit, 6 tokens) takes 7 steps in 2.8 s. Every computed position's logits go through the 152k-row head (`record`).
- **Check:** 21 s against the dense forward. 100 target rows and 246 drafter rows, all bit-exact. 45 s wall in total.
- **Acceptance:** 0 of 24 drafts. Four of 48 target layers do not predict like the model the MTP heads were trained against, so this run says nothing about acceptance on the full model.

## The contract a faster executor must meet

1. Implement `Executor`, honouring the seam contract and the MTP row layout above.
2. Run this crate's workloads with the faster executor in place of `CpuExecutor`. Compare each step's outputs and returned logits with `CpuExecutor` on the same inputs, or with `DenseOracle` directly.
3. Where bit equality with this executor is impossible (lower precision, another quantization recipe), document the tolerance and why; greedy outputs must still match except at documented near-ties. Bit equality with *itself* is not optional: the seam's determinism contract (`eidola-engine/AGENTS.md`) holds a faster executor to batch invariance, exactly.
4. Keep the zero-on-free observable: after maintenance, a freed block must hold no data from its previous owner, and a reset slot no per-sequence model state.

## DeepSeek-V4.1

`DsCpuExecutor` runs the window-chain model of `eidola-engine/AGENTS.md` (→ Window chain) on the model crate's DeepSeek-V4.1 kernels (`eidola-engine-model/AGENTS.md` → DeepSeek-V4.1). It reproduces the dense forward, every layer at every position, bit for bit, while computing the decoder only where each row's `chain_start` says.

| Module | Contents |
|---|---|
| `deepseek/mod.rs` | `DsCpuExecutor`, `DsCpuExecutorConfig`, `DsRowRecord`: maintenance, the step's order, drafting, sampling. |
| `deepseek/device.rs` | The two logical devices (`Encoder`, `Decoder`), what crosses between them (`ToEncoder`, `ToDecoder`, `StepTransfer`), the model's `Shape`. |
| `deepseek/layer.rs` | One model layer over a set of positions with paged KV (`run_layer`), the packed cells, the pooled sources' open entries. |
| `deepseek/packed.rs` | The packed forms: FP8 window KV, FP4 global KV, FP4 index keys. |
| `deepseek/pool.rs`, `deepseek/memory.rs` | A group's blocks of packed cells with tags; one device's pools and block tables. |
| `deepseek/oracle.rs` | `DsDenseOracle`, `ds_dense_generate`, `ds_check_generation`. |

### Shape

The executor splits the model before its first per-position KV source (layer 20 of the released model; `DsModel::check_split` must allow it) and panics at construction on a shape it does not run: the split layer gated or with an Engram module, encoder KV sources of different ratios, a decoder layer with an Engram module or reading an encoder bank, drafter target layers out of order. The KV groups it reports, in order:

| Group | Attention | Ratio | Device | Coverage | Layers | A cell |
|---|---|---|---|---|---|---|
| `encoder-bank` | full | the pooling ratio (2) | 0 | all | one per pooled KV source (3) | a global entry |
| `encoder-window` | sliding, the window | 1 | 0 | all | the layers before the split (20) | window KV |
| `decoder-bank` | full | 1 | 1 | all | 1 | a global entry |
| `decoder-window` | sliding, the window | 1 | 1 | chain | the layers from the split (20) | window KV |
| `drafter` (with drafts) | sliding, the window + 1 | 1 | 1 | chain | the draft blocks (3) | window KV |

The window chain is the decoder's layers; the token history is the Engram n-gram size less one (3).

### Memory images

A block holds `layers · block_size / ratio` **cells**, layer-major (`cell = layer · entries + entry`), each padded with zero bytes to a whole number of 32-bit words (no padding at the released sizes):

- **Window KV** (`Fp8`): `head_dim` FP8 E4M3 codes, then one E8M0 scale per 32 elements: 528 bytes.
- **A global entry**: the KV (`Fp4E4m3`: E2M1 codes, two per byte with element `2i` in the low nibble, then one FP8 E4M3 scale per 16 elements: 288 bytes), then the index key (`Fp4Pow2`: E2M1 codes the same way, then one E8M0 scale per 32: 68 bytes): 356 bytes.

Packing quantises with the model crate's `act_quant`, `quant_fp4_e4m3` and `quant_fp4_pow2` and stores their codes and scales exactly; a read is `bf16(code · scale)`, which is what the model's in-place quantisers leave, signed zeros included (`packed_forms_round_trip_the_models_quantisers`). So a pool holds the bytes a device executor holds, and a value read from it is the dense forward's.

Each cell has a tag: the position that completed the entry (plus one) and that position's token. A read checks it, so an unmapped block, a zeroed or encrypted one, another sequence's, or a chain layer that never computed that position panics.

**A block's image** (what a zero clears, a copy or a host-tier transfer moves, and the idle cipher XORs) is two ranges, each from a fresh keystream block: the cells' bytes as little-endian words in cell order, then the tags (`pos + 1`, then the token, per cell). A device executor's image is the first range.

### The device boundary

Device 0 (`Encoder`) holds the embedding, the layers before the split with their two groups, the first Engram table, and the split layer's **bank stub**: its attention-input norm and hyper-connection collapse, its compressor and its index-key projection. Device 1 (`Decoder`) holds the layers from the split on with their three groups, the head, the drafter with the embedding rows it reads (the given token and the noise token), and every later Engram table (the released model's second table belongs to layer 14, an encoder layer, and sits on device 1 for memory fit). Each struct owns only its own pools and block tables and indexes only its own layers, so neither can read the other's memory. What one needs of the other is a typed value, built once per step:

- **`ToEncoder`** (device 1 → 0): the rows gathered from the Engram tables device 1 holds, for every position of the step. The table never crosses; the token ids that address it reach both devices from the host, and the drafted ids from device 1.
- **`ToDecoder`** (device 0 → 1): one **bank entry** for every position of the step, in its stored 356-byte form; and a **boundary row** (the residual copies and the collapse weights entering the split layer) for exactly the positions the chain's bottom layer computes, `x >= chain_start - (depth - 1)(window - 1)`. Where the drafter's context is written (`x >= chain_start`) and the model has drafter target layers before the split (the released model has none; the fixture has one), the row also carries the means entering those layers.

`ToDecoder::check` panics unless the value is exactly that set, and `the_cross_device_transfer_is_exactly_the_declared_set` compares every step's `StepTransfer` with the set computed from the rows.

### A step

1. Maintenance in the order given, then table updates.
2. **Drafting**, on device 1, before the forward: a row whose last host position is `p` reads each draft block's context KV at the positions before `p` (in the reference's ring order) and runs the draft pass with the token at `p` in slot 0 (`DsModel::drafter_slots`). Slot `i`'s logits are the pass's base logits plus the Markov term of the token before it in the chain (the token at `p`, then each draft), so the drafts are sampled left to right on the draft stream, and the distribution kept for acceptance is the conditional sampled from.
3. **The encoder**, on device 0, over every position of the step (host tokens, then drafts), layer by layer: Engram, the layer's window KV written, a pooled source's entry written when its last position is computed, then attention and the feed-forward. Then the stub's bank entries.
4. The crossing, checked.
5. **The decoder**, on device 1: the bank entries stored, then each chain layer at the positions the row rule gives it (KV only where the next layer does not compute), the drafter's context KV written from the taps wherever the top layer computes (drafts included), and the head at each sampling row's last host position and at its drafts.
6. Sampling and `chain_accept`, with `sampling.rs` semantics over the sampleable vocabulary.

Within a layer every KV write of the step lands before any read. The attention list is the model crate's canonical one everywhere (the window right-aligned in `window` slots, then the selected global entries, ascending), which is why nothing depends on chunking.

**Per-slot model state** is one thing: each pooled source's **open entry**, the f32 `(kv, gate score)` of the positions computed since an entry closed, kept for the last `ratio + draft width + 1` positions in fixed buffers so the entry open after any accepted length is still there; `ResetSlot` scrubs it. The drafter needs none: a draft pass reads only context KV and token ids, so there is no tap state and no boundary tap. Everything else (hidden states, selections, candidate blocks, logits, drafts) is step-local.

### Drafter row layout

A row at `p` is the reference's draft pass at main position `p - 1` given the token chosen for `p`: draft slot `i` stands at `p + i` and predicts `p + 1 + i`. The reference's own entry point refuses main position 0 (it only seeds the window there); the pass is the same function at that position, and a row at `p = 1` runs it (`the_oracle_drafts_as_the_reference_entry_point_does` holds the oracle's pass equal to the reference's entry point wherever that one drafts). The pass always computes every slot of the drafter (each slot attends to all of them); a narrower draft width uses the first `k`.

### What is verified

On the model crate's committed fixture (`eidola-engine-model/tests/fixtures/deepseek_tiny`: eight layers split four and four, window 48, both top-ks pruning, an attention list spanning two 64-slot blocks, two Engram layers with the second table on device 1, a three-block drafter of four slots with one target layer before the split; the head's last row is the image token, so 95 ids are sampleable). With drafts the chain's top layer starts 48 positions before a target and its bottom layer 189.

`tests/ds_engine.rs` and `tests/ds_memory.rs` drive the real `Engine` and check every row of every step bit for bit against `DsDenseOracle` over the row's own tokens (rejected drafts included): the target logits it sampled or verified from, every draft slot's logits, every draft draw, the produced tokens; and every finished output against dense generation. Covered: chunked against unchunked prefill for prompts below, around and past the reach at chunks that cut triangles everywhere (and the chain's work equal across chunkings); a long prompt costing exactly one triangle; hits at regeneration, continuation and branch points, a short turn after a hit, a required prefix admitted and refused; a sequence cancelled mid-triangle retaining nothing; preemption and resume, mid-triangle too; shedding under KV pressure down to `KvExhausted`; greedy speculation equal to plain decoding at widths 0 to 4 across acceptance patterns from none to all (by narrowing the sampleable vocabulary and removing the drafter's Markov term); seeded sampling exact per step; a request's rows equal alone and beside others in another slot; the cross-device transfer; idle ciphertext for every group and a bit-exact hit after it; a recomputed block sealing to the same ciphertext in the three `All` groups and to another in the two chain-covered ones; the host tier round trip; and a randomized workload (host tier on odd seeds) with the engine's invariants, zero-on-free on real bytes and idle-ciphertext checks after every step. Both suites take under a minute together in the dev profile.

**A window past one attention block** (`tests/ds_wide.rs`, `ds_common::wide`). The fixture's window of 48 fits one 64-slot block, where the place of an empty window slot and the order of a draft pass's context change no bit. The same weights under a window of 80 and a five-slot drafter (neither shapes a tensor; the top-k and candidate pool prune as before; the top layer now starts 80 positions before a target and the bottom one 317) make both observable: a query's list is 80 window slots and up to 24 global entries, a draft pass's is 80 context entries and its slots. Through the engine, every row against the oracle: chunked against unchunked prefill, a hit and a short turn after it, preemption mid-triangle, greedy drafting at widths 0 to 5 across acceptance patterns from none to all five, and batch invariance (10 s; a sweep of eleven chunk sizes from one token up is ignored, 20 s). The oracle here is the model crate's forward under the same configuration, which is verified against the reference only at the committed fixture's shape: this model checks the executor against the oracle, not the oracle against the reference.

Each class was checked to fail under a deliberate bug: a chain layer starting one row late; the boundary row missing at the bottom layer's start; every boundary row crossing (caught only by the transfer test: the results are the same); `chain_start` ignored (every layer everywhere: caught by the chain's work, the transfer set and the retention tests, not by any logit); a global KV unpacked with its first scale; the attention list compacted (empty window slots dropped); the window left-aligned in its slots (no bit changes at the fixture's window of 48; five tests fail at 80); the open entry kept for two positions only; a reset leaving the open entry; the Engram lookback dropped; the drafter's context read in position order (one test fails at the fixture's window, where only the order of sums inside a block changes; six with the window of 80); the Markov term taken from the host token at every slot; no drafter context written at verified drafts; a zero leaving the bytes; encryption skipping the tags; a host-tier transfer dropping the tags; logits depending on the number of rows in the step; drafts sampled over the padded vocabulary; the split layer's bank entry rotated at the wrong position; a pooled entry rotated at its last position instead of its first; selection ignoring the candidate blocks.

**The released checkpoint** (`released_encoder_layers_page_exactly`, ignored, `EIDOLA_DEEPSEEK_PREFIX_DIR`): whole layers 0-3 with every expert and the first Engram module. That truncation has no layer to split before, so the whole executor cannot run on it; device 0 can, and does: 555 positions in uneven chunks through pools at the released cell sizes (528 and 356 bytes), the state leaving layer 3 equal to the dense forward bit for bit (147 s in release, about 28 GB resident, mostly the mapped checkpoint).
