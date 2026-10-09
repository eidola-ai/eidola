# eidola-engine-cpu

The CPU reference executor for the MiMo-V2.6 engine: `CpuExecutor` implements the serving core's `Executor` (`eidola-engine`) with the model crate's f32 reference numerics (`eidola-engine-model`), over paged KV, with MTP drafting and sampling inside the step. AGPL-3.0-only. Workspace conventions live in the root `AGENTS.md`.

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

A block holds every layer's `[K | V]` rows for `block_size` positions, contiguously. `Maintenance::Zero` overwrites the whole block with zero bytes, including the drafter tap and the tags. `Copy` copies all of it.

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
| `tests/executor.rs` | Driven directly through the seam: zero really zeroes (and only the named block), `ResetSlot` really scrubs the slot's model state (and only that slot's), a copied block plus its tap resumes exactly in a fresh slot, and rows are independent of batch, slot and chunking. Panics on unmapped reads, zeroed reads, shared-block writes, resuming off a boundary, and too many drafts. |
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
- a boundary tap holding the previous position's levels (caught at a branch hit).

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
3. Where bit equality is impossible (non-batch-invariant GEMMs or attention, lower precision), document the tolerance and why. Greedy outputs must still match except at documented near-ties.
4. Keep the zero-on-free observable: after maintenance, a freed block must hold no data from its previous owner, and a reset slot no per-sequence model state.
