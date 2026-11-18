//! Qwen3.8 kernel timings on a GPU (dev-only), one JSON line each:
//!
//!     qwen_kernel_bench <kernel dir> [--quick]
//!
//! - `copy`: device-to-device bandwidth (the streaming roofline every estimate assumes);
//! - `lin8`: the FP8 linear at the executor's stacked shapes, fast (CUTLASS, the GEMM
//!   `fp8_kind` dispatches to, split over K for the shapes `fp8_splits` splits) at 1 to
//!   2,048 rows and exact at 1 and 4: µs, TB/s of weights, TFLOP/s; and `lin8_kind`,
//!   every fast GEMM forced (the swapped kernel at each activation tile, and the unswapped
//!   one; for a shape that splits, every split instance at its split too, the partials'
//!   sum included) at 1, 8, 16, 32, 64, 256 and 2,048 rows: a split shape's unsplit kinds
//!   beside its split ones give the split's gain at decode sizes and its cost at prefill
//!   sizes;
//! - `small_gemm`: the DeltaNet's stacked `a` and `b` projections (96 × 5,120) on the
//!   small-N kernel as dispatched (`small_n`), in each of its forms forced
//!   (`small_n_rows`, `small_n_tiled`), and on CUTLASS's BF16 GEMM, at 1, 8, 64, 256 and
//!   2,048 rows;
//! - `norm` (the hidden norm with the residual add, and `q_rope`: the query heads' norm
//!   with RoPE), `widen` (the DeltaNet's conv inputs, 10,240 wide), `quant` (over
//!   activation-like values), `gate` (fast f32 and exact f64), `dn_prep`, `dn_norm`: one
//!   launch at the released widths, at 1, 8, 64, 256 and 2,048 rows;
//! - `fa2`: one target layer's FA2 decode launch (and its merge) as a rung runs it, at
//!   the released heads (24 query, 4 KV, 256 wide; pages of 256 positions): 1, 8, 16, 32
//!   and 64 rows of one query (`decode`) and of four (`drafted`: a host token and three
//!   drafts), each row at 1,024, 4,096 and 12,288 keys, its keys split at 256, 512 (the
//!   executor's `plan::SPLIT_KEYS`) and 1,024 (the split before it) positions;
//! - `head`: the BF16 head (248,320 × 5,120), fast;
//! - `sample`: Qwen's sampler over the released vocabulary at 1, 8, 64 and 256 rows,
//!   greedy, at temperature 1 alone, and with top-k 20 and top-p 0.95 (the conversation
//!   workload's); the shared sampler at 1 and 64 rows of the last, for comparison;
//! - `accept`: Qwen's chain acceptance of 3 drafts a sequence at 1, 8, 64 and 256
//!   sequences, every draft accepted (the bonus draw) and the first rejected (the
//!   residual's draw);
//! - `copy_rows`: a DeltaNet layer's staging copies at decode sizes (four conv rows and
//!   four `a`/`b` rows a sequence) for 1, 8, 64 and 256 sequences: the two groups in one
//!   launch, with as many self-copies again (a rung's padding), one launch a group, and
//!   MiMo's row copy (one launch a group);
//! - `dn_rec`: one DeltaNet layer's recurrence at the released width in both forms
//!   (`general`, `short`), 1 to 256 rows of one position, 1 to 64 rows with a fold of 0
//!   and 9 positions before 4 outputs; one row prefilling 2,048 and 10,240 positions; 64
//!   rows each prefilling 1,760 positions of their own;
//! - `dn_chunk`: the fast profile's blocked DeltaNet at the same width, blocks of 16
//!   (both kernels, and the intra kernel alone): prefills of 2,048 positions as one, two
//!   and four runs, one of 10,240 and 64 of 1,760 (tokens a second a layer, and the
//!   48 layers' time); one and 64 decode rows of one new position each, with 0, 7 and 15
//!   of its block's positions before it read from the tail cache;
//! - `cipher`: the idle cipher over one 157 MB state image;
//! - `host`: a 157 MB image to and from page-locked host memory.
//!
//! Wall time of launches queued back to back, the stream drained before and after.

use std::time::Instant;

use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;
use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::mimo::gemm::{GemmKind, SplitTile};
use eidola_engine_cuda::qwen::QwenDevice;
use eidola_engine_cuda::qwen::QwenProfile;
use eidola_engine_cuda::qwen::abi::*;
use eidola_engine_cuda::qwen::cuda::{CudaDevice, FP8_KINDS};
use eidola_engine_cuda::qwen::kernels::{
    AcceptLaunch, MAX_SPLITS, QwenKernels, SampleArgs, fp8_splits,
};
use eidola_engine_cuda::qwen::sampler::{accept_work_bytes, sample_work_bytes};
use eidola_engine_cuda::sampler::SampleRow;

fn fail(what: impl std::fmt::Display) -> ! {
    eprintln!("FAILED: {what}");
    std::process::exit(1)
}

fn alloc(d: &CudaDevice, bytes: usize, keep: &mut Vec<Box<dyn std::any::Any>>) -> u64 {
    let (a, k) = d.alloc(bytes).unwrap_or_else(|e| fail(e));
    keep.push(k);
    a
}

fn fill(d: &CudaDevice, addr: u64, bytes: usize, seed: u64) {
    let mut s = seed;
    let v: Vec<u8> = (0..bytes)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // No E4M3 NaN code (0x7f / 0xff), small BF16 magnitudes.
            u8::try_from((s >> 56) & 0x3e).unwrap_or(0)
        })
        .collect();
    // SAFETY: a fresh allocation of `bytes`.
    unsafe { d.upload(addr, &v) }.unwrap_or_else(|e| fail(e));
}

/// `count` BF16 values at `addr` shaped like activations: about normal (a sum of four
/// uniforms), one in a thousand twenty times larger. The quantiser's branch-free path
/// covers such blocks; `fill`'s values, spread over a hundred binades, would send most of
/// its blocks down the dividing path.
fn fill_activations(d: &CudaDevice, addr: u64, count: usize, seed: u64) {
    let mut s = seed;
    let mut uniform = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let v: Vec<u8> = (0..count)
        .flat_map(|i| {
            let x = uniform() + uniform() + uniform() + uniform() - 2.0;
            let x = if i % 1000 == 999 { 20.0 * x } else { x };
            eidola_engine_cuda::bf16::from_f32(x).to_le_bytes()
        })
        .collect();
    // SAFETY: an allocation of at least `2 · count` bytes.
    unsafe { d.upload(addr, &v) }.unwrap_or_else(|e| fail(e));
}

/// Mean seconds of `iters` runs of `f`, the stream drained around them.
fn time(d: &CudaDevice, iters: u32, mut f: impl FnMut()) -> f64 {
    f();
    d.synchronize().unwrap_or_else(|e| fail(e));
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    d.synchronize().unwrap_or_else(|e| fail(e));
    t.elapsed().as_secs_f64() / f64::from(iters)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .unwrap_or_else(|| fail("usage: qwen_kernel_bench <kernel dir> [--quick]"));
    let quick = args.iter().any(|a| a == "--quick");
    let dir = KernelDir::new(dir);
    let fast = CudaDevice::open(0, &dir, QwenProfile::Fast).unwrap_or_else(|e| fail(e));
    let exact = CudaDevice::open(0, &dir, QwenProfile::Exact).unwrap_or_else(|e| fail(e));
    let mut keep = Vec::new();
    let it = if quick { 3 } else { 20 };

    let gb = 1usize << 30;
    let (a, b) = (alloc(&fast, gb, &mut keep), alloc(&fast, gb, &mut keep));
    // SAFETY: two 1 GiB allocations.
    let s = time(&fast, it, || {
        unsafe { fast.copy(b, a, gb) }.unwrap_or_else(|e| fail(e))
    });
    println!(
        "{{\"bench\":\"copy\",\"gb_per_s\":{:.1}}}",
        2.0 * gb as f64 / s / 1e9
    );

    let shapes = [
        (16384u32, 5120u32),
        (5120, 6144),
        (14336, 5120),
        (34816, 5120),
        (5120, 17408),
    ];
    let max_rows = 2048u32;
    let kmax = 17408usize;
    let codes = alloc(&fast, max_rows as usize * kmax, &mut keep);
    let scales = alloc(&fast, (kmax / 128) * max_rows as usize * 4, &mut keep);
    fill(&fast, codes, max_rows as usize * kmax, 1);
    let out = alloc(&fast, max_rows as usize * 34816 * 2, &mut keep);
    // The split shapes' partial sums: at most `MAX_SPLITS` arrays of the widest split N.
    let parts = alloc(
        &fast,
        MAX_SPLITS as usize * max_rows as usize * 5120 * 4,
        &mut keep,
    );
    for (n, k) in shapes {
        let w = alloc(&fast, n as usize * k as usize, &mut keep);
        fill(&fast, w, n as usize * k as usize, u64::from(n + k));
        let ws = alloc(
            &fast,
            (n as usize / 128) * (k as usize / 128) * 4,
            &mut keep,
        );
        let sv: Vec<u8> = (0..(n / 128) * (k / 128))
            .flat_map(|_| 1e-3f32.to_le_bytes())
            .collect();
        // SAFETY: the scale array just made.
        unsafe { fast.upload(ws, &sv) }.unwrap_or_else(|e| fail(e));
        let sa: Vec<u8> = (0..(k / 128) * max_rows)
            .flat_map(|_| 1e-2f32.to_le_bytes())
            .collect();
        // SAFETY: as above.
        unsafe { fast.upload(scales, &sa) }.unwrap_or_else(|e| fail(e));
        for rows in [1u32, 4, 8, 16, 32, 64, 256, 1024, 2048] {
            let m_pad = rows.next_multiple_of(4);
            let p = Lin8Params {
                a: codes,
                a_scales: scales,
                w,
                w_scales: ws,
                out,
                rows,
                n,
                k,
                m_pad,
                out_stride: n,
                reserved: 0,
            };
            for (name, dev) in [("fast", &fast), ("exact", &exact)] {
                if name == "exact" && rows > 4 {
                    continue;
                }
                // SAFETY: every buffer covers `m_pad` rows of this shape.
                let s = time(dev, if name == "exact" { 1 } else { it }, || {
                    unsafe { dev.lin8(&p, Some(parts)) }.unwrap_or_else(|e| fail(e))
                });
                let bytes = f64::from(n) * f64::from(k);
                println!(
                    "{{\"bench\":\"lin8\",\"form\":\"{name}\",\"n\":{n},\"k\":{k},\"rows\":{rows},\"us\":{:.1},\"weight_tb_per_s\":{:.2},\"tflops\":{:.1}}}",
                    s * 1e6,
                    bytes / s / 1e12,
                    2.0 * bytes * f64::from(rows) / s / 1e12
                );
            }
            if [1, 8, 16, 32, 64, 256, 2048].contains(&rows) {
                let splits = fp8_splits(n, k);
                let split_kinds = SplitTile::ALL
                    .into_iter()
                    .filter(|_| splits > 1)
                    .map(|t| GemmKind::Fp8Split(t, splits));
                for kind in FP8_KINDS.into_iter().chain(split_kinds) {
                    // SAFETY: as above; the partials' scratch holds every split.
                    let s = time(&fast, it, || {
                        unsafe { fast.lin8_on(&p, kind, Some(parts)) }.unwrap_or_else(|e| fail(e))
                    });
                    let bytes = f64::from(n) * f64::from(k);
                    println!(
                        "{{\"bench\":\"lin8_kind\",\"kind\":\"{kind:?}\",\"n\":{n},\"k\":{k},\"rows\":{rows},\"us\":{:.1},\"weight_tb_per_s\":{:.2},\"tflops\":{:.1}}}",
                        s * 1e6,
                        bytes / s / 1e12,
                        2.0 * bytes * f64::from(rows) / s / 1e12
                    );
                }
            }
        }
    }

    small_kernels(&fast, it, &mut keep);
    fa2_decode(&fast, it, &mut keep);
    sampler(&fast, &dir, it, &mut keep);
    copies(&fast, &dir, it, &mut keep);

    let (vocab, hidden) = (248_320u32, 5120u32);
    let head = alloc(&fast, vocab as usize * hidden as usize * 2, &mut keep);
    fill(&fast, head, vocab as usize * hidden as usize * 2, 3);
    let x = alloc(&fast, 256 * hidden as usize * 2, &mut keep);
    fill(&fast, x, 256 * hidden as usize * 2, 4);
    let logits = alloc(&fast, 256 * vocab as usize * 4, &mut keep);
    let head_tmp = alloc(&fast, 256 * vocab as usize * 4, &mut keep);
    for rows in [1u32, 4, 64, 256] {
        let p = LinearParams {
            x,
            w: head,
            out: logits,
            rows,
            n: vocab,
            k: hidden,
            x_stride: hidden,
            out_stride: vocab,
            out_f32: 1,
        };
        // SAFETY: the buffers cover `rows`.
        let s = time(&fast, it, || {
            unsafe { fast.linear(&p, Some(head_tmp)) }.unwrap_or_else(|e| fail(e))
        });
        println!(
            "{{\"bench\":\"head\",\"rows\":{rows},\"us\":{:.1},\"weight_tb_per_s\":{:.2}}}",
            s * 1e6,
            f64::from(vocab) * f64::from(hidden) * 2.0 / s / 1e12
        );
    }

    // One DeltaNet layer at the released width: 16 key heads, 48 value heads of 128.
    let (kh, vh, kd, vd) = (16u32, 48u32, 128u32, 128u32);
    let state = (vh * kd * vd) as usize * 4;
    let max_pos = 64 * 1760usize;
    let q = alloc(&fast, max_pos * (kh * kd) as usize * 4, &mut keep);
    let kk = alloc(&fast, max_pos * (kh * kd) as usize * 4, &mut keep);
    let v = alloc(&fast, max_pos * (vh * vd) as usize * 4, &mut keep);
    let beta = alloc(&fast, max_pos * vh as usize * 4, &mut keep);
    let alpha = alloc(&fast, max_pos * vh as usize * 4, &mut keep);
    let o = alloc(&fast, max_pos * (vh * vd) as usize * 4, &mut keep);
    let al: Vec<u8> = (0..max_pos * vh as usize)
        .flat_map(|_| 0.99f32.to_le_bytes())
        .collect();
    // SAFETY: the decay array.
    unsafe { fast.upload(alpha, &al) }.unwrap_or_else(|e| fail(e));
    let states = alloc(&fast, 256 * state, &mut keep);
    // (rows, fold, outputs, whether each row reads positions of its own).
    for (rows, fold, outs, own) in [
        (1u32, 0u32, 1u32, false),
        (8, 0, 1, false),
        (64, 0, 1, false),
        (256, 0, 1, false),
        (64, 3, 4, false),
        (1u32, 0u32, 4u32, false),
        (1, 9, 4, false),
        (8, 0, 4, false),
        (8, 9, 4, false),
        (64, 2, 4, false),
        (64, 9, 4, false),
        (1, 0, 2048, false),
        (1, 0, 10240, false),
        (64, 0, 1760, true),
    ] {
        let seqs: Vec<DnSeq> = (0..rows)
            .map(|r| DnSeq {
                src: states + u64::from(r) * state as u64,
                live: states + u64::from(r) * state as u64,
                first: if own { r * (fold + outs) } else { 0 },
                count: fold + outs,
                out_from: fold,
                live_after: fold + 1,
                ..DnSeq::default()
            })
            .collect();
        let sb = alloc(&fast, seqs.len() * std::mem::size_of::<DnSeq>(), &mut keep);
        let bytes: Vec<u8> = seqs
            .iter()
            .flat_map(|s| {
                // SAFETY: a `repr(C)` struct's bytes.
                unsafe {
                    std::slice::from_raw_parts(
                        (s as *const DnSeq).cast::<u8>(),
                        std::mem::size_of::<DnSeq>(),
                    )
                }
                .to_vec()
            })
            .collect();
        // SAFETY: the descriptor array.
        unsafe { fast.upload(sb, &bytes) }.unwrap_or_else(|e| fail(e));
        let p = DnRecParams {
            seqs: sb,
            q,
            k: kk,
            v,
            beta,
            alpha,
            out: o,
            state_offset: 0,
            seq_count: rows,
            k_heads: kh,
            v_heads: vh,
            k_dim: kd,
            v_dim: vd,
            reserved: 0,
        };
        for short in [false, true] {
            // SAFETY: every position and state the descriptors name is allocated.
            let s = time(&fast, if outs > 100 { 2 } else { it }, || {
                if short {
                    unsafe { fast.dn_rec_short(&p) }
                } else {
                    unsafe { fast.dn_rec(&p) }
                }
                .unwrap_or_else(|e| fail(e))
            });
            let form = if short { "short" } else { "general" };
            println!(
                "{{\"bench\":\"dn_rec\",\"form\":\"{form}\",\"rows\":{rows},\"fold\":{fold},\"outputs\":{outs},\"us_per_layer\":{:.1},\"us_per_token_step\":{:.3},\"us_per_token\":{:.4},\"state_tb_per_s\":{:.2}}}",
                s * 1e6,
                s * 1e6 / f64::from(fold + outs),
                s * 1e6 / f64::from((fold + outs) * rows),
                2.0 * f64::from(rows) * state as f64 / s / 1e12
            );
        }
    }

    // The fast profile's blocked DeltaNet (`dn_chunk`, blocks of 16) on the same inputs,
    // the log decay of 0.99 a position: prefills of 2,048 positions as one, two and four
    // runs, one of 10,240, 64 runs of 1,760; decode rows of one new position, reading from
    // the tail cache 0, 7 and 15 positions of its block before it.
    let lg: Vec<u8> = (0..max_pos * vh as usize)
        .flat_map(|_| 0.99f32.ln().to_le_bytes())
        .collect();
    let g = alloc(&fast, lg.len(), &mut keep);
    // SAFETY: the log-decay array.
    unsafe { fast.upload(g, &lg) }.unwrap_or_else(|e| fail(e));
    let chunk = CHUNK_TILE;
    let max_tiles = max_pos / chunk as usize + 64;
    let (fw, fu, fp, ff) = chunk_scratch_floats(vd);
    let per = max_tiles * vh as usize * 4;
    let (sw, su, sp, sf) = (
        alloc(&fast, per * fw, &mut keep),
        alloc(&fast, per * fu, &mut keep),
        alloc(&fast, per * fp, &mut keep),
        alloc(&fast, per * ff, &mut keep),
    );
    let perk = max_tiles * kh as usize * 4 * CHUNK_QK_FLOATS;
    let (sq, sk) = (alloc(&fast, perk, &mut keep), alloc(&fast, perk, &mut keep));
    let skraw = alloc(&fast, perk, &mut keep);
    let srec = alloc(
        &fast,
        max_tiles * vh as usize * CHUNK_REC_FLOATS * 4,
        &mut keep,
    );
    // A tail cache of 32 rows a sequence for 64 sequences: a decode row's earlier
    // positions are read from it, its new one kept there, as in serving.
    let ring = 32u32;
    let entry = chunk_entry_floats(kh, vh, kd, vd);
    let tail = alloc(&fast, 64 * ring as usize * entry * 4, &mut keep);
    // (rows, fold, outputs).
    for (rows, fold, outs) in [
        (1u32, 0u32, 2048u32),
        (2, 0, 1024),
        (4, 0, 512),
        (1, 0, 10240),
        (64, 0, 1760),
        (1, 0, 1),
        (64, 0, 1),
        (64, 7, 1),
        (64, 15, 1),
    ] {
        let mut tile = 0;
        let seqs: Vec<DnSeq> = (0..rows)
            .map(|r| {
                let count = fold + outs;
                let d = DnSeq {
                    src: states + u64::from(r) * state as u64,
                    first: r * outs,
                    count,
                    out_from: fold,
                    live_after: count + 1,
                    tile,
                    pos0: 0,
                    tail_row: (r % 64) * ring,
                    // The positions from the last boundary on are kept.
                    keep_from: count / chunk * chunk,
                    ..DnSeq::default()
                };
                tile += (fold + outs).div_ceil(chunk);
                d
            })
            .collect();
        let sb = alloc(&fast, seqs.len() * std::mem::size_of::<DnSeq>(), &mut keep);
        let bytes: Vec<u8> = seqs
            .iter()
            .flat_map(|s| {
                // SAFETY: a `repr(C)` struct's bytes.
                unsafe {
                    std::slice::from_raw_parts(
                        (s as *const DnSeq).cast::<u8>(),
                        std::mem::size_of::<DnSeq>(),
                    )
                }
                .to_vec()
            })
            .collect();
        // SAFETY: the descriptor array.
        unsafe { fast.upload(sb, &bytes) }.unwrap_or_else(|e| fail(e));
        let p = DnChunkParams {
            seqs: sb,
            q,
            k: kk,
            v,
            beta,
            g,
            out: o,
            w: sw,
            u: su,
            pm: sp,
            fac: sf,
            qr: sq,
            kr: sk,
            kraw: skraw,
            rec: srec,
            tail,
            state_offset: 0,
            seq_count: rows,
            tiles: tile,
            k_heads: kh,
            v_heads: vh,
            k_dim: kd,
            v_dim: vd,
            chunk,
            ring,
        };
        let n = if outs > 100 { 3 } else { it };
        // SAFETY: every position, state and scratch block the descriptors name is
        // allocated.
        let s = time(&fast, n, || {
            unsafe { fast.dn_chunk(&p) }.unwrap_or_else(|e| fail(e))
        });
        let intra = time(&fast, n, || {
            unsafe { fast.dn_intra(&p) }.unwrap_or_else(|e| fail(e))
        });
        let state = time(&fast, n, || {
            unsafe { fast.dn_state(&p) }.unwrap_or_else(|e| fail(e))
        });
        let tokens = f64::from(outs * rows);
        println!(
            "{{\"bench\":\"dn_chunk\",\"rows\":{rows},\"fold\":{fold},\"outputs\":{outs},\"us_per_layer\":{:.1},\"intra_us\":{:.1},\"state_us\":{:.1},\"keep_us\":{:.1},\"tokens_per_s_per_layer\":{:.0},\"ms_48_layers\":{:.3}}}",
            s * 1e6,
            intra * 1e6,
            state * 1e6,
            (s - intra - state) * 1e6,
            tokens / s,
            s * 48.0 * 1e3
        );
    }

    let image = 156_893_184usize;
    let img = alloc(&fast, image, &mut keep);
    let key = KvKey::from_bytes([7; 32]);
    let nonce = KvNonce::from_words([1, 2, 3]);
    let words = u32::try_from(image / 4).unwrap_or_else(|_| fail("words"));
    // SAFETY: the image's words.
    let s = time(&fast, it, || {
        unsafe { fast.cipher(img, words, 0, &key, &nonce) }.unwrap_or_else(|e| fail(e))
    });
    println!(
        "{{\"bench\":\"cipher\",\"image_bytes\":{image},\"ms\":{:.3}}}",
        s * 1e3
    );
    let (host, _h) = fast.alloc_host(image).unwrap_or_else(|e| fail(e));
    // SAFETY: the page-locked host image and the device image, the stream drained after.
    let d2h = time(&fast, it, || {
        unsafe { fast.copy_to_host(host, img, image) }.unwrap_or_else(|e| fail(e))
    });
    let h2d = time(&fast, it, || {
        unsafe { fast.copy_from_host(img, host, image) }.unwrap_or_else(|e| fail(e))
    });
    println!(
        "{{\"bench\":\"host\",\"image_bytes\":{image},\"d2h_gb_per_s\":{:.1},\"h2d_gb_per_s\":{:.1}}}",
        image as f64 / d2h / 1e9,
        image as f64 / h2d / 1e9
    );
}

/// One JSON line of a small kernel's mean launch time.
/// One FA2 decode launch of a target layer as a rung makes it (`PassCaps` for its rows,
/// a fixed grid walking the work list, the merge), over KV pages of zeros: the kernels'
/// time does not depend on the values.
fn fa2_decode(fast: &CudaDevice, it: u32, keep: &mut Vec<Box<dyn std::any::Any>>) {
    use eidola_engine_cuda::mimo::attention::{
        AttnLayer, AttnRequest, HostPlan, Partials, PassCaps, Reduction,
    };
    use eidola_engine_cuda::qwen::attention::{
        Fa2Args, HEAD_DIM, MAX_TILE, PARTIAL_BYTES_PER_HEAD, pass_words,
    };
    let (hq, hk, bs) = (24u32, 4u32, 256u32);
    let w = u64::from(hk * HEAD_DIM);
    let block_bytes = u64::from(bs) * w * 2 * 2;
    let max_rows = 64u32;
    let max_queries = 4u32;
    let max_ctx = 12_288u32;
    let blocks = max_rows * max_ctx.div_ceil(bs);
    let us = |x: u64| usize::try_from(x).unwrap_or_else(|e| fail(e));
    let pool = alloc(fast, us(u64::from(blocks) * block_bytes), keep);
    let qrows = (max_rows * max_queries) as usize;
    let q = alloc(fast, qrows * hq as usize * HEAD_DIM as usize * 2, keep);
    fill(fast, q, qrows * hq as usize * HEAD_DIM as usize * 2, 7);
    let o = alloc(fast, qrows * hq as usize * HEAD_DIM as usize * 2, keep);
    let sink = alloc(fast, hq as usize * 4, keep);
    let inf: Vec<u8> = (0..hq)
        .flat_map(|_| f32::NEG_INFINITY.to_le_bytes())
        .collect();
    // SAFETY: `hq` f32 of the allocation just made.
    unsafe { fast.upload(sink, &inf) }.unwrap_or_else(|e| fail(e));
    let most_partials = max_rows * max_queries * max_ctx.div_ceil(256);
    let part_v = alloc(
        fast,
        most_partials as usize * hq as usize * HEAD_DIM as usize * 2,
        keep,
    );
    let part_s = alloc(fast, most_partials as usize * hq as usize * 4, keep);
    debug_assert_eq!(PARTIAL_BYTES_PER_HEAD, HEAD_DIM as usize * 2 + 4);
    let words = alloc(fast, 4 << 20, keep);
    let layer = AttnLayer {
        k_base: pool,
        v_base: pool + u64::from(bs) * w * 2,
        k_page_stride: w32(us(u64::from(bs) * w * 2)),
        v_page_stride: w32(us(u64::from(bs) * w * 2)),
        num_kv_heads: hk,
        page_size: bs,
        window_left: -1,
        sink,
    };
    for (form, queries) in [("decode", 1u32), ("drafted", max_queries)] {
        for rows in [1u32, 8, 16, 32, 64] {
            for ctx in [1024u32, 4096, 12_288] {
                for keys in [256u32, 512, 1024] {
                    let pages = ctx.div_ceil(bs);
                    let requests: Vec<AttnRequest> = (0..rows)
                        .map(|r| AttnRequest {
                            q_start: r * queries,
                            qo_len: queries,
                            pages: (r * pages..(r + 1) * pages).collect(),
                            kv_start: 0,
                            kv_len: ctx,
                        })
                        .collect();
                    let caps = PassCaps::with_split_keys(
                        Reduction::Split,
                        rows,
                        queries,
                        hq / hk,
                        bs,
                        pages,
                        ctx,
                        MAX_TILE,
                        keys,
                    )
                    .unwrap_or_else(|e| fail(e));
                    let plan = HostPlan::with_split_keys(
                        &requests,
                        hq / hk,
                        bs,
                        Reduction::Split,
                        most_partials,
                        MAX_TILE,
                        keys,
                    )
                    .unwrap_or_else(|e| fail(e));
                    let (table, view) =
                        pass_words(&plan, 0, &caps, words).unwrap_or_else(|e| fail(e));
                    // SAFETY: the table fits the 4 MiB just made.
                    unsafe { fast.upload(words, bytes_of(&table).as_slice()) }
                        .unwrap_or_else(|e| fail(e));
                    let a = Fa2Args {
                        view,
                        layer,
                        num_qo_heads: hq,
                        q,
                        o,
                        partials: Partials {
                            v: part_v,
                            s: part_s,
                            rows: most_partials,
                            heads: hq,
                        },
                        scale: 1.0 / 16.0,
                    };
                    // SAFETY: the plan's pages lie in the pool; the buffers hold its rows.
                    let t = time(fast, it, || {
                        unsafe { fast.fa2(&a) }.unwrap_or_else(|e| fail(e))
                    });
                    println!(
                        "{{\"bench\":\"fa2\",\"form\":\"{form}\",\"rows\":{rows},\"keys\":{ctx},\"split_keys\":{keys},\"us\":{:.1}}}",
                        t * 1e6
                    );
                }
            }
        }
    }
}

fn line(bench: &str, form: &str, rows: u32, s: f64) {
    println!(
        "{{\"bench\":\"{bench}\",\"form\":\"{form}\",\"rows\":{rows},\"us\":{:.2}}}",
        s * 1e6
    );
}

/// The kernels around the linears, at the released widths: the norm (hidden 5,120 with
/// the residual add), the quantiser (5,120 and 17,408), the gates (the MLP's 17,408,
/// fast and exact), the DeltaNet's stacked `a`/`b` projection (small-N and CUTLASS), its
/// preparation and gated norm.
fn small_kernels(fast: &CudaDevice, it: u32, keep: &mut Vec<Box<dyn std::any::Any>>) {
    let rows_max = 2048usize;
    let (hidden, inter) = (5120u32, 17408u32);
    let x = alloc(fast, rows_max * inter as usize * 2, keep);
    fill(fast, x, rows_max * inter as usize * 2, 11);
    let y = alloc(fast, rows_max * inter as usize * 2, keep);
    fill(fast, y, rows_max * inter as usize * 2, 12);
    let out = alloc(fast, rows_max * inter as usize * 4, keep);
    let w = alloc(fast, inter as usize * 4, keep);
    let ones: Vec<u8> = (0..inter).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    // SAFETY: the weight vector just made.
    unsafe { fast.upload(w, &ones) }.unwrap_or_else(|e| fail(e));
    // The DeltaNet's convolution weights: four taps of each of its 10,240 channels
    // (16 key heads twice and 48 value heads, all of 128).
    let conv_taps = 10_240 * 4;
    let conv_w = alloc(fast, conv_taps * 4, keep);
    let taps: Vec<u8> = (0..conv_taps).flat_map(|_| 0.25f32.to_le_bytes()).collect();
    // SAFETY: the convolution weights just made.
    unsafe { fast.upload(conv_w, &taps) }.unwrap_or_else(|e| fail(e));
    let codes = alloc(fast, rows_max * inter as usize, keep);
    let scales = alloc(fast, (inter as usize / 128) * rows_max * 4, keep);
    let acts = alloc(fast, rows_max * inter as usize * 2, keep);
    fill_activations(fast, acts, rows_max * inter as usize, 15);
    let ab_w = alloc(fast, 96 * hidden as usize * 2, keep);
    fill(fast, ab_w, 96 * hidden as usize * 2, 13);
    let rope = alloc(fast, 4096 * 64 * 4, keep);
    fill(fast, rope, 4096 * 64 * 4, 14);
    for rows in [1u32, 8, 64, 256, 2048] {
        let norm = NormParams {
            x,
            add: y,
            weight: w,
            out,
            rows,
            units: 1,
            n: hidden,
            x_stride: hidden,
            add_stride: hidden,
            out_stride: hidden,
            eps: 1e-6,
            ..NormParams::default()
        };
        // SAFETY: every buffer covers 2,048 rows of the widest width.
        let s = time(fast, it, || {
            unsafe { fast.norm(&norm) }.unwrap_or_else(|e| fail(e))
        });
        line("norm", "exact", rows, s);
        // The query heads' norm with RoPE: 24 heads of 256 a row, beside their gates
        // (unit stride 512), a warp a head.
        let positions: Vec<u8> = (0..rows).flat_map(|r| (r % 4096).to_le_bytes()).collect();
        let pos = alloc(fast, positions.len().max(8), keep);
        // SAFETY: the positions array just made.
        unsafe { fast.upload(pos, &positions) }.unwrap_or_else(|e| fail(e));
        let qk = NormParams {
            x,
            add: 0,
            weight: w,
            out,
            rope_cos: rope,
            rope_sin: rope,
            positions: pos,
            rows,
            units: 24,
            n: 256,
            x_stride: 24 * 512,
            x_unit_stride: 512,
            out_stride: 24 * 256,
            out_unit_stride: 256,
            rotary: 64,
            eps: 1e-6,
            ..NormParams::default()
        };
        // SAFETY: 2,048 rows of 12,288 BF16 values at `x`, of 6,144 at `out`; 4,096
        // positions of 64 rotation values.
        let s = time(fast, it, || {
            unsafe { fast.norm(&qk) }.unwrap_or_else(|e| fail(e))
        });
        line("norm", "q_rope", rows, s);
        let wid = WidenParams {
            x,
            out,
            rows,
            width: 10_240,
            x_stride: 10_240,
            out_stride: 10_240,
        };
        // SAFETY: 2,048 rows of 10,240 values at each.
        let s = time(fast, it, || {
            unsafe { fast.widen(&wid) }.unwrap_or_else(|e| fail(e))
        });
        line("widen", "conv_in", rows, s);
        for k in [hidden, inter] {
            let q = QuantParams {
                x: acts,
                codes,
                scales,
                rows,
                k,
                x_stride: k,
                m_pad: rows.next_multiple_of(4),
            };
            // SAFETY: as above.
            let s = time(fast, it, || {
                unsafe { fast.quant(&q) }.unwrap_or_else(|e| fail(e))
            });
            line("quant", &format!("k{k}"), rows, s);
        }
        let gate = GateParams {
            a: x,
            g: y,
            out,
            rows,
            width: inter,
            a_stride: inter,
            g_stride: inter,
            out_stride: inter,
            unit: inter,
            g_unit_stride: inter,
            kind: GATE_SWIGLU,
        };
        // SAFETY: as above.
        let s = time(fast, it, || {
            unsafe { fast.gate(&gate) }.unwrap_or_else(|e| fail(e))
        });
        line("gate", "fast", rows, s);
        // SAFETY: as above.
        let s = time(fast, it, || {
            unsafe { fast.gate_exact(&gate) }.unwrap_or_else(|e| fail(e))
        });
        line("gate", "exact", rows, s);
        let ab = LinearParams {
            x,
            w: ab_w,
            out,
            rows,
            n: 96,
            k: hidden,
            x_stride: hidden,
            out_stride: 96,
            out_f32: 1,
        };
        // SAFETY: as above.
        let s = time(fast, it, || {
            unsafe { fast.gemm_bf16(&ab) }.unwrap_or_else(|e| fail(e))
        });
        line("small_gemm", "small_n", rows, s);
        for (form, tiled) in [("small_n_rows", false), ("small_n_tiled", true)] {
            // SAFETY: as above.
            let s = time(fast, it, || {
                unsafe { fast.gemm_bf16_form(&ab, tiled) }.unwrap_or_else(|e| fail(e))
            });
            line("small_gemm", form, rows, s);
        }
        // SAFETY: as above.
        let s = time(fast, it, || {
            unsafe { fast.gemm_bf16_cutlass(&ab) }.unwrap_or_else(|e| fail(e))
        });
        line("small_gemm", "cutlass", rows, s);
        // The DeltaNet at the released width: 16 key heads, 48 value heads of 128; the
        // conv rows and gates read from the filled buffers, every window its own row.
        let (kh, vh, kd, vd) = (16u32, 48u32, 128u32, 128u32);
        let window: Vec<u8> = (0..rows).flat_map(|r| r.to_le_bytes()).collect();
        let wb = alloc(fast, window.len().max(8), keep);
        // SAFETY: the window array just made.
        unsafe { fast.upload(wb, &window) }.unwrap_or_else(|e| fail(e));
        let prep = DnPrepParams {
            conv_in: out,
            window: wb,
            ab: out,
            conv_w,
            exp_a_log: w,
            dt_bias: w,
            q: y,
            k: y,
            v: y,
            beta: y,
            alpha: y,
            positions: rows,
            k_heads: kh,
            v_heads: vh,
            k_dim: kd,
            v_dim: vd,
            log_decay: 0,
        };
        // SAFETY: the prep reads at most 2,048 + 3 rows of 10,240 f32 values from `out`
        // (2,048 × 17,408 f32) and 10,240 × 4 from `conv_w`, and writes 2,048 rows of
        // 6,144 f32 values into `y`.
        let s = time(fast, it, || {
            unsafe { fast.dn_prep(&prep) }.unwrap_or_else(|e| fail(e))
        });
        line("dn_prep", "exact", rows, s);
        let dn_norm = DnNormParams {
            o: out,
            z: x,
            weight: w,
            out: y,
            rows,
            v_heads: vh,
            v_dim: vd,
            z_stride: vh * vd,
            eps: 1e-6,
            reserved: 0,
        };
        // SAFETY: as above.
        let s = time(fast, it, || {
            unsafe { fast.dn_norm(&dn_norm) }.unwrap_or_else(|e| fail(e))
        });
        line("dn_norm", "exact", rows, s);
    }
}

/// A count as the kernels' 32-bit word.
fn w32(x: usize) -> u32 {
    u32::try_from(x).unwrap_or_else(|_| fail("a count past 32 bits"))
}

/// Bytes of a host slice of plain values.
fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    // SAFETY: plain `repr(C)` values' bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
        .to_vec()
}

/// Qwen's sampler and chain acceptance at the released vocabulary, and the shared sampler
/// for comparison.
fn sampler(fast: &CudaDevice, dir: &KernelDir, it: u32, keep: &mut Vec<Box<dyn std::any::Any>>) {
    use eidola_engine::sampling::{SamplingParams, Stream};
    let (vocab, n) = (248_320usize, 248_077u32);
    let rows_max = 256usize;
    // Logits in [0, 8) with a spike a row, every row its own.
    let mut s = 5u64;
    let host: Vec<f32> = (0..rows_max * vocab)
        .map(|i| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (s >> 40) as f32 / (1u64 << 24) as f32 * 8.0;
            if i % vocab == (i / vocab) * 997 % vocab {
                x + 6.0
            } else {
                x
            }
        })
        .collect();
    let logits = alloc(fast, host.len() * 4, keep);
    // SAFETY: the logits just allocated.
    unsafe { fast.upload(logits, &bytes_of(&host)) }.unwrap_or_else(|e| fail(e));
    let probs = alloc(fast, rows_max * n as usize * 8, keep);
    let tokens = alloc(fast, rows_max * 4, keep);
    let status = alloc(fast, 4, keep);
    let work = alloc(fast, sample_work_bytes(rows_max), keep);
    let rows_buf = alloc(fast, rows_max * 32, keep);
    let conversation = SamplingParams::new(1.0, 20, 0.95, 0.0, 7).unwrap_or_else(|e| fail(e));
    let forms = [
        ("greedy", SamplingParams::greedy()),
        (
            "temperature",
            SamplingParams::random(1.0, 3).unwrap_or_else(|e| fail(e)),
        ),
        ("top_k20_top_p0.95", conversation),
    ];
    let upload_rows = |p: &SamplingParams, rows: usize| {
        let r: Vec<SampleRow> = (0..rows)
            .map(|i| SampleRow::new(p, 1000 + w32(i), w32(i)))
            .collect();
        // SAFETY: the rows buffer holds 256.
        unsafe { fast.upload(rows_buf, &bytes_of(&r)) }.unwrap_or_else(|e| fail(e));
    };
    for rows in [1u32, 8, 64, 256] {
        for (form, p) in &forms {
            upload_rows(p, rows as usize);
            let a = SampleArgs {
                logits,
                logits_stride: vocab as u64,
                n,
                rows: rows_buf,
                num_rows: rows,
                draw: Some(Stream::Sample),
                probs,
                tokens,
                status,
                work,
            };
            // SAFETY: every buffer holds 256 rows.
            let s = time(fast, it, || {
                unsafe { fast.sample(&a) }.unwrap_or_else(|e| fail(e))
            });
            line("sample", form, rows, s);
        }
    }
    // The shared sampler, on its own buffers.
    {
        let gpu = fast.gpu();
        let shared =
            eidola_engine_cuda::sampler::Sampler::load(gpu, dir).unwrap_or_else(|e| fail(e));
        let st = gpu.stream();
        let dl = st
            .clone_htod(&host[..64 * vocab])
            .unwrap_or_else(|e| fail(e));
        for rows in [1usize, 64] {
            let hr: Vec<SampleRow> = (0..rows)
                .map(|i| SampleRow::new(&conversation, 1000 + w32(i), w32(i)))
                .collect();
            let mut dr = st.clone_htod(&hr).unwrap_or_else(|e| fail(e));
            let mut dp = st
                .alloc_zeros::<f64>(rows * n as usize)
                .unwrap_or_else(|e| fail(e));
            let mut dt = st.alloc_zeros::<u32>(rows).unwrap_or_else(|e| fail(e));
            let mut ds = st.alloc_zeros::<u32>(1).unwrap_or_else(|e| fail(e));
            let s = time(fast, 2, || {
                shared
                    .sample(
                        gpu,
                        &dl,
                        vocab,
                        n,
                        &hr,
                        &mut dr,
                        Some(Stream::Sample),
                        &mut dp,
                        &mut dt,
                        &mut ds,
                    )
                    .unwrap_or_else(|e| fail(e))
            });
            line("sample", "shared_top_k20_top_p0.95", w32(rows), s);
        }
    }
    // Chain acceptance over the conversation workload's distributions: 64 rows sampled
    // above, read as target rows (k + 1 a sequence) and draft rows, modulo 64.
    upload_rows(&conversation, 64);
    let a = SampleArgs {
        logits,
        logits_stride: vocab as u64,
        n,
        rows: rows_buf,
        num_rows: 64,
        draw: Some(Stream::Draft),
        probs,
        tokens,
        status,
        work,
    };
    // SAFETY: as above.
    unsafe { fast.sample(&a) }.unwrap_or_else(|e| fail(e));
    let mut drawn = vec![0u8; 64 * 4];
    // SAFETY: the tokens of 64 rows.
    unsafe { fast.download(tokens, &mut drawn) }.unwrap_or_else(|e| fail(e));
    let drawn: Vec<u32> = drawn
        .chunks(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let width = 3usize;
    // Draft ids by draft row (`drafts[draft_row + i]`): each row's own draw.
    let by_row: Vec<u32> = (0..64 + width).map(|j| drawn[j % 64]).collect();
    let ids = alloc(fast, by_row.len() * 4, keep);
    let acc_rows = alloc(fast, 256 * 32, keep);
    let (trow, drow, nd) = (
        alloc(fast, 256 * 4, keep),
        alloc(fast, 256 * 4, keep),
        alloc(fast, 256 * 4, keep),
    );
    let out = alloc(fast, 256 * (width + 1) * 4, keep);
    let counts = alloc(fast, 256 * 4, keep);
    let awork = alloc(fast, accept_work_bytes(256), keep);
    // SAFETY: the arrays just allocated.
    unsafe { fast.upload(ids, &bytes_of(&by_row)) }.unwrap_or_else(|e| fail(e));
    for (form, rejecting) in [("bonus", false), ("residual", true)] {
        for seqs in [1u32, 8, 64, 256] {
            let sr = seqs as usize;
            // Target rows t ..= t + 3; the drafts drawn from those rows (every one
            // accepted) or from the next ones (rejected unless among the target's top 20).
            let t: Vec<u32> = (0..sr).map(|r| w32((r * 4) % 60)).collect();
            let d: Vec<u32> = t.iter().map(|&x| x + u32::from(rejecting)).collect();
            let k: Vec<u32> = vec![w32(width); sr];
            let rr: Vec<SampleRow> = (0..sr)
                .map(|r| SampleRow::new(&conversation, 1000 + w32(r), 0))
                .collect();
            // SAFETY: each buffer holds 256 sequences.
            unsafe {
                fast.upload(acc_rows, &bytes_of(&rr))
                    .unwrap_or_else(|e| fail(e));
                fast.upload(trow, &bytes_of(&t)).unwrap_or_else(|e| fail(e));
                fast.upload(drow, &bytes_of(&d)).unwrap_or_else(|e| fail(e));
                fast.upload(nd, &bytes_of(&k)).unwrap_or_else(|e| fail(e));
            }
            let a = AcceptLaunch {
                target: probs,
                draft: probs,
                n,
                rows: acc_rows,
                target_row: trow,
                draft_row: drow,
                draft_step: 1,
                num_drafts: nd,
                drafts: ids,
                stride: w32(width + 1),
                scratch: awork,
                out,
                counts,
                status,
                num_rows: seqs,
            };
            // SAFETY: every row named lies inside the 64 sampled distributions.
            let s = time(fast, it, || {
                unsafe { fast.accept(&a) }.unwrap_or_else(|e| fail(e))
            });
            line("accept", form, seqs, s);
        }
    }
}

/// A DeltaNet layer's staging copies at decode sizes: per sequence four conv rows (10,240
/// f32) and four `a`/`b` rows (96 f32), from a state-like buffer into the staging buffers.
fn copies(fast: &CudaDevice, dir: &KernelDir, it: u32, keep: &mut Vec<Box<dyn std::any::Any>>) {
    let (cd, ab) = (10_240usize, 96usize);
    let max = 256usize * 4;
    let src_c = alloc(fast, max * cd * 4, keep);
    let dst_c = alloc(fast, max * cd * 4, keep);
    let src_a = alloc(fast, max * ab * 4, keep);
    let dst_a = alloc(fast, max * ab * 4, keep);
    let mimo = eidola_engine_cuda::mimo::engine_ops::EngineOps::load(fast.gpu(), dir)
        .unwrap_or_else(|e| fail(e));
    let status = alloc(fast, 4, keep);
    for seqs in [1u32, 8, 64, 256] {
        let items = seqs as usize * 4;
        // Real items (buffer 0 row i to buffer 1 row i), then as many self-copies.
        let mut sb = vec![0u32; items];
        let mut db = vec![1u32; items];
        let mut rows: Vec<u32> = (0..w32(items)).collect();
        sb.extend(vec![1u32; items]);
        db.extend(vec![1u32; items]);
        rows.extend(0..w32(items));
        let arr = |v: &[u32], keep: &mut Vec<Box<dyn std::any::Any>>| {
            let a = alloc(fast, v.len() * 4, keep);
            // SAFETY: the array just allocated.
            unsafe { fast.upload(a, &bytes_of(v)) }.unwrap_or_else(|e| fail(e));
            a
        };
        let (sba, dba, ra) = (arr(&sb, keep), arr(&db, keep), arr(&rows, keep));
        let group = |src: u64, dst: u64, width: usize, n: usize| {
            let mut g = CopyGroup {
                src_buf: sba,
                src_row: ra,
                dst_buf: dba,
                dst_row: ra,
                width: w32(width),
                items: w32(n),
                ..CopyGroup::default()
            };
            g.buf[0] = src;
            g.buf[1] = dst;
            g.rows[0] = max as u64;
            g.rows[1] = max as u64;
            g
        };
        let params = |groups: &[CopyGroup]| {
            let mut p = CopyParams {
                status,
                groups: w32(groups.len()),
                ..CopyParams::default()
            };
            p.group[..groups.len()].copy_from_slice(groups);
            p
        };
        let both = params(&[
            group(src_c, dst_c, cd, items),
            group(src_a, dst_a, ab, items),
        ]);
        let padded = params(&[
            group(src_c, dst_c, cd, 2 * items),
            group(src_a, dst_a, ab, 2 * items),
        ]);
        let (one_c, one_a) = (
            params(&[group(src_c, dst_c, cd, items)]),
            params(&[group(src_a, dst_a, ab, items)]),
        );
        // SAFETY: every buffer holds 1,024 rows of its width; the items name rows below.
        unsafe {
            let s = time(fast, it, || {
                fast.copy_rows(&both).unwrap_or_else(|e| fail(e))
            });
            line("copy_rows", "grouped", seqs, s);
            let s = time(fast, it, || {
                fast.copy_rows(&padded).unwrap_or_else(|e| fail(e))
            });
            line("copy_rows", "grouped_padded", seqs, s);
            let s = time(fast, it, || {
                fast.copy_rows(&one_c).unwrap_or_else(|e| fail(e));
                fast.copy_rows(&one_a).unwrap_or_else(|e| fail(e));
            });
            line("copy_rows", "separate", seqs, s);
            let mimo_args = |g: &CopyGroup| {
                let mut a = eidola_engine_cuda::mimo::engine_ops::CopyArgs {
                    src_buf: g.src_buf,
                    src_row: g.src_row,
                    dst_buf: g.dst_buf,
                    dst_row: g.dst_row,
                    status,
                    width: g.width,
                    items: g.items,
                    ..Default::default()
                };
                a.buf = g.buf;
                a.rows = g.rows;
                a
            };
            let (mc, ma) = (mimo_args(&padded.group[0]), mimo_args(&padded.group[1]));
            let s = time(fast, it, || {
                mimo.copy_rows(fast.gpu(), mc).unwrap_or_else(|e| fail(e));
                mimo.copy_rows(fast.gpu(), ma).unwrap_or_else(|e| fail(e));
            });
            line("copy_rows", "mimo_padded", seqs, s);
        }
    }
}
