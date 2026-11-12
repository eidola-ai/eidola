//! The fused-QKV RoPE + paged KV-write kernel against its one-block-per-token
//! reference form (`engine_ops_reference.cu`) bit for bit: rotated Q, the K
//! and V it writes into the pool, and every byte it must leave alone, for
//! Flash's global (one KV head per chunk) and sliding (two) layers, on every
//! image this device runs, from one token to a full prefill step.

mod common;

use common::{Lcg, setup};
use eidola_engine_cuda::bf16;
use eidola_engine_cuda::engine_ops::{EngineOps, QKV_THREADS, QkvArgs};
use eidola_engine_cuda::launch::dptr;

const CHUNKS: usize = 4;
const Q_HEADS: usize = 16;
const D: usize = 192;
const DV: usize = 128;
const ROTARY: usize = 64;
const BLOCK: usize = 16;
const MAX_POS: usize = 4096;
const TOKENS: [usize; 8] = [1, 2, 7, 64, 128, 513, 2048, 8192];

fn u32_of(x: usize) -> u32 {
    u32::try_from(x).unwrap()
}

/// The launch contract is the block the host's grid counts threads in.
#[test]
fn qkv_launch_contract() {
    let Some(su) = setup() else { return };
    for &arch in &su.archs {
        let m = su.module("engine_ops", arch);
        let meta = *m.kernel("eidola_qkv_rope_kv").unwrap().meta();
        assert_eq!(meta.block, [QKV_THREADS, 1, 1], "{arch:?}");
        assert_eq!(meta.cluster, [1, 1, 1], "{arch:?}");
    }
}

#[test]
fn qkv_rope_kv_matches_reference_bit_for_bit() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let s = gpu.stream();
    for &arch in &su.archs {
        let ops = EngineOps::from_module(su.module("engine_ops", arch)).unwrap();
        let reference = su
            .module("engine_ops_reference", arch)
            .kernel("eidola_reference_qkv_rope_kv")
            .unwrap();
        // Global layers (stride padded to whole 128-row tiles) and sliding.
        for (kv_heads, stride) in [(1usize, 3456usize), (2, 3712)] {
            let per_chunk = (Q_HEADS + kv_heads) * D + kv_heads * DV;
            assert!(stride >= per_chunk);
            let nkv = CHUNKS * kv_heads;
            // Two layers per block; the kernel writes the second.
            let layer = BLOCK * nkv * (D + DV);
            let (k_off, v_off) = (layer, layer + BLOCK * nkv * D);
            let block_elems = 2 * layer;
            for &tokens in &TOKENS {
                let mut rng = Lcg(0x9e0 ^ (tokens as u64) << 4 ^ kv_heads as u64);
                // Rows over a wide range of magnitudes, with infinities, NaN
                // and the padding past each chunk's heads.
                let mut qkv = vec![0u16; tokens * CHUNKS * stride];
                for (i, v) in qkv.iter_mut().enumerate() {
                    *v = match i % 997 {
                        5 => bf16::from_f32(f32::INFINITY),
                        6 => bf16::from_f32(f32::NEG_INFINITY),
                        7 => bf16::from_f32(f32::NAN),
                        _ => {
                            let e = i32::try_from(rng.below(41)).unwrap() - 20;
                            bf16::from_f32(rng.f32() * 2f32.powi(e))
                        }
                    };
                }
                let rope: Vec<f32> = (0..MAX_POS * ROTARY).map(|_| rng.f32()).collect();
                let positions: Vec<u32> = (0..tokens)
                    .map(|_| u32::try_from(rng.below(MAX_POS as u64)).unwrap())
                    .collect();
                // Token t: block 1 + t / BLOCK (block 0 stays untouched), slot
                // a permutation of t % BLOCK.
                let blocks = tokens.div_ceil(BLOCK) + 2;
                let kv_block: Vec<u32> = (0..tokens).map(|t| u32_of(1 + t / BLOCK)).collect();
                let kv_slot: Vec<u32> =
                    (0..tokens).map(|t| u32_of(t % BLOCK * 5 % BLOCK)).collect();
                let dqkv = s.clone_htod(&qkv).unwrap();
                let drope = s.clone_htod(&rope).unwrap();
                let dpos = s.clone_htod(&positions).unwrap();
                let dblock = s.clone_htod(&kv_block).unwrap();
                let dslot = s.clone_htod(&kv_slot).unwrap();
                let mut outs = Vec::new();
                for new in [true, false] {
                    let pool = s
                        .clone_htod(&vec![0xa5a5u16; blocks * block_elems])
                        .unwrap();
                    let q = s
                        .clone_htod(&vec![0x5a5au16; tokens * CHUNKS * Q_HEADS * D])
                        .unwrap();
                    let args = QkvArgs {
                        qkv: dptr(&dqkv, s),
                        q_out: dptr(&q, s),
                        pool: dptr(&pool, s),
                        positions: dptr(&dpos, s),
                        kv_block: dptr(&dblock, s),
                        kv_slot: dptr(&dslot, s),
                        rope: dptr(&drope, s),
                        block_elems: block_elems as u64,
                        k_off: k_off as u64,
                        v_off: v_off as u64,
                        chunk_stride: u32_of(stride),
                        chunks: u32_of(CHUNKS),
                        q_heads_per_chunk: u32_of(Q_HEADS),
                        kv_heads_per_chunk: u32_of(kv_heads),
                    };
                    unsafe {
                        if new {
                            ops.qkv_rope_kv(gpu, args, u32_of(tokens)).unwrap();
                        } else {
                            eidola_engine_cuda::launch!(
                                gpu,
                                reference,
                                [u32_of(tokens), 1, 1],
                                args
                            )
                            .unwrap();
                        }
                    }
                    outs.push((s.clone_dtoh(&q).unwrap(), s.clone_dtoh(&pool).unwrap()));
                }
                let what = format!("{arch:?} {kv_heads} KV heads/chunk, {tokens} tokens");
                let per_q = CHUNKS * Q_HEADS * D;
                for t in 0..tokens {
                    let r = t * per_q..(t + 1) * per_q;
                    assert_eq!(outs[0].0[r.clone()], outs[1].0[r], "{what}: Q of token {t}");
                }
                for b in 0..blocks {
                    let r = b * block_elems..(b + 1) * block_elems;
                    assert_eq!(outs[0].1[r.clone()], outs[1].1[r], "{what}: KV block {b}");
                }
                // The reference wrote every token's K and V (the comparison
                // has something to compare), and nothing in block 0.
                assert!(
                    outs[1].1[..block_elems].iter().all(|&v| v == 0xa5a5),
                    "{what}"
                );
                let slot0 = usize::try_from(kv_slot[0]).unwrap();
                let k0 = block_elems + k_off + slot0 * nkv * D;
                assert_ne!(
                    outs[1].1[k0..k0 + nkv * D],
                    vec![0xa5a5; nkv * D][..],
                    "{what}"
                );
            }
        }
    }
}
