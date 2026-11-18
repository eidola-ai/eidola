//! Qwen3.8 kernel timings on a GPU (dev-only), one JSON line each:
//!
//!     qwen_kernel_bench <kernel dir> [--quick]
//!
//! - `copy`: device-to-device bandwidth (the streaming roofline every estimate assumes);
//! - `lin8`: the FP8 linear at the released fused shapes, fast (CUTLASS, the GEMM
//!   `fp8_kind` dispatches to) at 1 to 2,048 rows and exact at 1 and 4: µs, TB/s of
//!   weights, TFLOP/s; and `lin8_kind`, every fast GEMM forced (the swapped kernel at
//!   each activation tile, and the unswapped one) at 1, 8, 64 and 2,048 rows;
//! - `small_gemm`: the DeltaNet's stacked `a` and `b` projections (96 × 5,120) on the
//!   small-N kernel and on CUTLASS's BF16 GEMM, at 1, 8, 64 and 2,048 rows;
//! - `norm`, `quant`, `gate` (fast f32 and exact f64), `dn_prep`, `dn_norm`: one launch
//!   at the released widths, at 1, 8, 64 and 2,048 rows;
//! - `head`: the BF16 head (248,320 × 5,120), fast;
//! - `dn_rec`: one DeltaNet layer's recurrence at the released width, 1 to 64 rows, with
//!   a fold of 0 and 9 positions before 4 outputs; one row prefilling 2,048 and 10,240
//!   positions; 64 rows each prefilling 1,760 positions of their own;
//! - `cipher`: the idle cipher over one 157 MB state image;
//! - `host`: a 157 MB image to and from page-locked host memory.
//!
//! Wall time of launches queued back to back, the stream drained before and after.

use std::time::Instant;

use eidola_engine::kv_cipher::KvNonce;
use eidola_engine::secret::KvKey;
use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::QwenDevice;
use eidola_engine_cuda::qwen::QwenProfile;
use eidola_engine_cuda::qwen::abi::*;
use eidola_engine_cuda::qwen::cuda::{CudaDevice, FP8_KINDS};
use eidola_engine_cuda::qwen::kernels::QwenKernels;

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
                    unsafe { dev.lin8(&p) }.unwrap_or_else(|e| fail(e))
                });
                let bytes = f64::from(n) * f64::from(k);
                println!(
                    "{{\"bench\":\"lin8\",\"form\":\"{name}\",\"n\":{n},\"k\":{k},\"rows\":{rows},\"us\":{:.1},\"weight_tb_per_s\":{:.2},\"tflops\":{:.1}}}",
                    s * 1e6,
                    bytes / s / 1e12,
                    2.0 * bytes * f64::from(rows) / s / 1e12
                );
            }
            if [1, 8, 64, 2048].contains(&rows) {
                for kind in FP8_KINDS {
                    // SAFETY: as above.
                    let s = time(&fast, it, || {
                        unsafe { fast.lin8_on(&p, kind) }.unwrap_or_else(|e| fail(e))
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
    let states = alloc(&fast, 64 * state, &mut keep);
    // (rows, fold, outputs, whether each row reads positions of its own).
    for (rows, fold, outs, own) in [
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
        // SAFETY: every position and state the descriptors name is allocated.
        let s = time(&fast, if outs > 100 { 2 } else { it }, || {
            unsafe { fast.dn_rec(&p) }.unwrap_or_else(|e| fail(e))
        });
        println!(
            "{{\"bench\":\"dn_rec\",\"rows\":{rows},\"fold\":{fold},\"outputs\":{outs},\"us_per_layer\":{:.1},\"us_per_token_step\":{:.3},\"us_per_token\":{:.4}}}",
            s * 1e6,
            s * 1e6 / f64::from(fold + outs),
            s * 1e6 / f64::from((fold + outs) * rows)
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
    let ab_w = alloc(fast, 96 * hidden as usize * 2, keep);
    fill(fast, ab_w, 96 * hidden as usize * 2, 13);
    for rows in [1u32, 8, 64, 2048] {
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
        for k in [hidden, inter] {
            let q = QuantParams {
                x,
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
            reserved: 0,
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
