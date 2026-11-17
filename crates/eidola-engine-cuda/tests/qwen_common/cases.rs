//! Per-kernel cases: each kernel family through [`QwenKernels`] (its grid function and
//! its body) against the model crate's function, bit for bit. Generic over the device, so
//! the same cases run on the host-compiled bodies (`tests/qwen_kernels_host.rs`) and on a
//! GPU (`tests/qwen_ops.rs`).
#![allow(dead_code)]

use std::sync::Arc;

use eidola_engine_cuda::qwen::QwenDevice;
use eidola_engine_cuda::qwen::abi::*;
use eidola_engine_model::Matrix;
use eidola_engine_model::qwen::attention::{apply_rope, attend, inv_freq, rope_cos_sin};
use eidola_engine_model::qwen::config::{AttentionSpec, DeltaNetSpec};
use eidola_engine_model::qwen::deltanet::{CoreWeights, DeltaNetState, token_core};
use eidola_engine_model::qwen::forward::rms_norm_zc;
use eidola_engine_model::qwen::linear::{
    Bf16Matrix, Fp8Linear, Stored, act_quant, lin8_rows, linear_bf16_rows,
};
use eidola_engine_model::qwen::numeric::{bf16_round, sigmoid32, silu32};

use super::{Lcg, bits, buf, read, widen, zeros};

/// A BF16's place in the total order of its values: units in the last place apart.
pub fn ord16(b: u16) -> i32 {
    let m = i32::from(b & 0x7fff);
    if b & 0x8000 != 0 { -m } else { m }
}

fn u(x: usize) -> u32 {
    u32::try_from(x).expect("a test width fits u32")
}

/// The first differing element of two bit patterns, as `Err(description)`.
pub fn same<T: PartialEq + std::fmt::Debug>(
    what: &str,
    got: &[T],
    want: &[T],
) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!(
            "{what}: {} elements, want {}",
            got.len(),
            want.len()
        ));
    }
    match got.iter().zip(want).position(|(a, b)| a != b) {
        None => Ok(()),
        Some(i) => Err(format!(
            "{what}: element {i} of {}: {:?}, want {:?}",
            got.len(),
            got[i],
            want[i]
        )),
    }
}

fn f32_bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The residual add and the zero-centred norm over one unit per row, and over several
/// rotated units (the query and key norms with RoPE).
pub fn norm<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    for &(rows, n) in &[(5usize, 200usize), (3, 5120), (1, 16), (4, 17)] {
        let x: Vec<f32> = (0..rows * n).map(|_| g.bf16(4.0)).collect();
        let add: Vec<f32> = (0..rows * n).map(|_| g.bf16(2.0)).collect();
        let w: Vec<f32> = (0..n).map(|_| 1.0 + g.bf16(0.5)).collect();
        let xb = buf(dev, &bits(&x));
        let ab = buf(dev, &bits(&add));
        let wb = buf(dev, &w);
        let ob = zeros(dev, rows * n * 2);
        let p = NormParams {
            x: xb.addr,
            add: ab.addr,
            weight: wb.addr,
            out: ob.addr,
            rows: u(rows),
            units: 1,
            n: u(n),
            x_stride: u(n),
            add_stride: u(n),
            out_stride: u(n),
            eps: 1e-6,
            ..NormParams::default()
        };
        // SAFETY: every buffer holds `rows · n` elements.
        unsafe { dev.norm(&p) }.map_err(|e| e.to_string())?;
        let mut h = Vec::new();
        let mut want = Vec::new();
        for r in 0..rows {
            let hr: Vec<f32> = (0..n)
                .map(|i| bf16_round(x[r * n + i] + add[r * n + i]))
                .collect();
            let mut o = vec![0.0; n];
            rms_norm_zc(&hr, &w, 1e-6, &mut o);
            h.extend(hr);
            want.extend(o);
        }
        same("residual", &read::<D, u16>(dev, &xb, rows * n), &bits(&h))?;
        same("norm", &read::<D, u16>(dev, &ob, rows * n), &bits(&want))?;
    }
    // Rotated units: 3 query heads of 256 interleaved with their gates (stride 512).
    let spec = AttentionSpec {
        num_q_heads: 3,
        num_kv_heads: 1,
        head_dim: 256,
        rotary_dim: 64,
        rope_theta: 1e7,
        softmax_scale: 1.0 / 16.0,
    };
    let freq = inv_freq(&spec);
    let positions: Vec<u32> = vec![0, 7, 1000, 262_143 % 4096];
    let max_pos = 4096usize;
    let mut cos = vec![0.0f32; max_pos * 64];
    let mut sin = vec![0.0f32; max_pos * 64];
    for pos in 0..max_pos {
        let (c, s) = rope_cos_sin(&freq, pos);
        cos[pos * 64..(pos + 1) * 64].copy_from_slice(&c);
        sin[pos * 64..(pos + 1) * 64].copy_from_slice(&s);
    }
    let rows = positions.len();
    let (units, n, xs) = (3usize, 256usize, 1536usize);
    let x: Vec<f32> = (0..rows * xs).map(|_| g.bf16(8.0)).collect();
    let w: Vec<f32> = (0..n).map(|_| 1.0 + g.bf16(0.3)).collect();
    let xb = buf(dev, &bits(&x));
    let wb = buf(dev, &w);
    let ob = zeros(dev, rows * units * n * 2);
    let cb = buf(dev, &cos);
    let sb = buf(dev, &sin);
    let pb = buf(dev, &positions);
    let p = NormParams {
        x: xb.addr,
        weight: wb.addr,
        out: ob.addr,
        rope_cos: cb.addr,
        rope_sin: sb.addr,
        positions: pb.addr,
        rows: u(rows),
        units: u(units),
        n: u(n),
        x_stride: u(xs),
        x_unit_stride: 512,
        out_stride: u(units * n),
        out_unit_stride: u(n),
        rotary: 64,
        eps: 1e-6,
        ..NormParams::default()
    };
    // SAFETY: the buffers hold what the struct names.
    unsafe { dev.norm(&p) }.map_err(|e| e.to_string())?;
    let mut want = Vec::new();
    for (r, &pos) in positions.iter().enumerate() {
        let (c, s) = rope_cos_sin(&freq, pos as usize);
        for u in 0..units {
            let src = &x[r * xs + u * 512..r * xs + u * 512 + n];
            let mut o = vec![0.0; n];
            rms_norm_zc(src, &w, 1e-6, &mut o);
            apply_rope(&mut o, &c, &s);
            want.extend(o);
        }
    }
    same(
        "rotated norm",
        &read::<D, u16>(dev, &ob, rows * units * n),
        &bits(&want),
    )
}

/// A random FP8 weight: codes with no NaN, BF16-valued scales `[n/128][k/128]`.
pub fn fp8_weight(g: &mut Lcg, n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let codes: Vec<u8> = (0..n * k)
        .map(|_| {
            let c = u8::try_from(g.below(256)).expect("a byte");
            if c & 0x7f == 0x7f { c & 0xfe } else { c }
        })
        .collect();
    let scales: Vec<f32> = (0..(n / 128) * (k / 128))
        .map(|_| bf16_round(1e-3 * (1.0 + g.f32().abs())))
        .collect();
    (codes, scales)
}

/// `[n/128][k/128]` (the checkpoint's) to `[k/128][n/128]` (the kernels').
pub fn transpose_scales(s: &[f32], n: usize, k: usize) -> Vec<f32> {
    let (nb, kb) = (n / 128, k / 128);
    let mut t = vec![0.0; s.len()];
    for i in 0..nb {
        for j in 0..kb {
            t[j * nb + i] = s[i * kb + j];
        }
    }
    t
}

/// The quantiser and the exact FP8 linear against `lin8_rows`.
pub fn lin8<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    for &(rows, n, k) in &[(9usize, 256usize, 384usize), (1, 128, 128), (17, 384, 256)] {
        let m_pad = rows.next_multiple_of(4);
        let mut x: Vec<f32> = (0..rows * k).map(|_| g.bf16(3.0)).collect();
        // A zero block, a spike and a tie-rich block.
        x[..128].fill(0.0);
        if rows > 1 {
            x[k + 5] = bf16_round(448.0 * 1000.0);
        }
        let (codes, scales) = fp8_weight(&mut g, n, k);
        let xb = buf(dev, &bits(&x));
        let cb = zeros(dev, m_pad * k);
        let sb = zeros(dev, (k / 128) * m_pad * 4);
        let q = QuantParams {
            x: xb.addr,
            codes: cb.addr,
            scales: sb.addr,
            rows: u(rows),
            k: u(k),
            x_stride: u(k),
            m_pad: u(m_pad),
        };
        let wb = buf(dev, &codes);
        let wsb = buf(dev, &transpose_scales(&scales, n, k));
        let ob = zeros(dev, m_pad * n * 2);
        let l = Lin8Params {
            a: cb.addr,
            a_scales: sb.addr,
            w: wb.addr,
            w_scales: wsb.addr,
            out: ob.addr,
            rows: u(rows),
            n: u(n),
            k: u(k),
            m_pad: u(m_pad),
            out_stride: u(n),
            reserved: 0,
        };
        // SAFETY: the buffers hold what the structs name.
        unsafe {
            dev.quant(&q).map_err(|e| e.to_string())?;
            dev.lin8(&l).map_err(|e| e.to_string())?;
        }
        let want = lin8_rows(
            &Matrix::from_vec(rows, k, x.clone()),
            &Fp8Linear::from_parts(n, k, codes, scales),
        );
        same(
            &format!("lin8 {rows}×{n}×{k}"),
            &read::<D, u16>(dev, &ob, rows * n),
            &bits(&want.data),
        )?;
    }
    Ok(())
}

/// The BF16 linear against `linear_bf16_rows`, to BF16 bits and to f32.
pub fn linear<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    for &(rows, n, k) in &[(9usize, 100usize, 200usize), (1, 300, 5120), (12, 48, 16)] {
        let x: Vec<f32> = (0..rows * k).map(|_| g.bf16(2.0)).collect();
        let w: Vec<f32> = (0..n * k).map(|_| g.bf16(0.1)).collect();
        let wbits = bits(&w);
        let bytes: Vec<u8> = wbits.iter().flat_map(|b| b.to_le_bytes()).collect();
        let m = Bf16Matrix {
            rows: n,
            cols: k,
            data: Stored::Owned(Arc::new(bytes)),
        };
        let want = linear_bf16_rows(&Matrix::from_vec(rows, k, x.clone()), &m);
        let xb = buf(dev, &bits(&x));
        let wb = buf(dev, &wbits);
        for out_f32 in [0u32, 1] {
            let ob = zeros(dev, rows * n * 4);
            let p = LinearParams {
                x: xb.addr,
                w: wb.addr,
                out: ob.addr,
                rows: u(rows),
                n: u(n),
                k: u(k),
                x_stride: u(k),
                out_stride: u(n),
                out_f32,
            };
            // SAFETY: the buffers hold what the struct names.
            unsafe { dev.linear(&p, true) }.map_err(|e| e.to_string())?;
            if out_f32 == 1 {
                same(
                    "linear f32",
                    &f32_bits(&read::<D, f32>(dev, &ob, rows * n)),
                    &f32_bits(&want.data),
                )?;
            } else {
                same(
                    "linear",
                    &read::<D, u16>(dev, &ob, rows * n),
                    &bits(&want.data),
                )?;
            }
        }
    }
    Ok(())
}

/// Both element-wise gates against their reference expressions.
pub fn gate<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    let (rows, heads, d) = (5usize, 3usize, 256usize);
    let width = heads * d;
    let a: Vec<f32> = (0..rows * width).map(|_| g.bf16(6.0)).collect();
    let gs: Vec<f32> = (0..rows * 2 * width).map(|_| g.bf16(12.0)).collect();
    let ab = buf(dev, &bits(&a));
    let gb = buf(dev, &bits(&gs));
    for kind in [GATE_SWIGLU, GATE_SIGMOID] {
        let ob = zeros(dev, rows * width * 2);
        let p = GateParams {
            a: ab.addr,
            g: gb.addr,
            out: ob.addr,
            rows: u(rows),
            width: u(width),
            a_stride: u(width),
            g_stride: u(2 * width),
            out_stride: u(width),
            unit: u(d),
            g_unit_stride: u(2 * d),
            kind,
        };
        // SAFETY: the buffers hold what the struct names.
        unsafe { dev.gate(&p) }.map_err(|e| e.to_string())?;
        let mut want = Vec::new();
        for r in 0..rows {
            for i in 0..width {
                let gv = gs[r * 2 * width + (i / d) * 2 * d + i % d];
                let av = a[r * width + i];
                want.push(if kind == GATE_SWIGLU {
                    bf16_round(bf16_round(silu32(gv)) * av)
                } else {
                    bf16_round(av * bf16_round(sigmoid32(gv)))
                });
            }
        }
        same(
            &format!("gate {kind}"),
            &read::<D, u16>(dev, &ob, rows * width),
            &bits(&want),
        )?;
    }
    Ok(())
}

/// The paged KV write and the exact attention core against `attend` over the same keys,
/// for target rows (keys from position 0) and drafter rows (keys from position 1).
pub fn attention<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    for &(head_dim, q_heads, kv_heads, key_first) in &[
        (128usize, 6usize, 2usize, 0u32),
        (256, 6, 1, 1),
        (64, 4, 4, 0),
    ] {
        let (layers, layer, bs) = (2usize, 1usize, 4usize);
        let width = kv_heads * head_dim;
        // Two sequences: 13 and 6 positions of KV, with tables over shuffled blocks.
        let lens = [13usize, 6];
        let blocks_per = [4usize, 2];
        let num_blocks = 1 + blocks_per.iter().sum::<usize>();
        let mut pool = vec![0u16; num_blocks * layers * 2 * bs * width];
        let mut tables = [-1i32; 2 * 4];
        let order = [3i32, 1, 6, 5, 2, 4];
        let mut next = 0;
        let mut keys: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut vals: Vec<Vec<Vec<f32>>> = Vec::new();
        for (s, &len) in lens.iter().enumerate() {
            for b in 0..blocks_per[s] {
                tables[s * 4 + b] = order[next];
                next += 1;
            }
            let mut ks = Vec::new();
            let mut vs = Vec::new();
            for pos in 0..len {
                let k: Vec<f32> = (0..width).map(|_| g.bf16(2.0)).collect();
                let v: Vec<f32> = (0..width).map(|_| g.bf16(2.0)).collect();
                let block = usize::try_from(tables[s * 4 + pos / bs]).expect("a mapped block");
                let base = (block * layers + layer) * 2 * bs * width + (pos % bs) * width;
                pool[base..base + width].copy_from_slice(&bits(&k));
                pool[base + bs * width..base + bs * width + width].copy_from_slice(&bits(&v));
                ks.push(k);
                vs.push(v);
            }
            keys.push(ks);
            vals.push(vs);
        }
        // Rows: queries at several positions of both sequences.
        let rows_of: Vec<(usize, u32)> = vec![(0, 12), (1, 5), (0, 3), (0, key_first), (1, 1)];
        let rows = rows_of.len();
        let q: Vec<f32> = (0..rows * q_heads * head_dim)
            .map(|_| g.bf16(2.0))
            .collect();
        let mut row_tables = vec![-1i32; rows * 4];
        let mut positions = Vec::new();
        for (r, &(s, pos)) in rows_of.iter().enumerate() {
            row_tables[r * 4..r * 4 + 4].copy_from_slice(&tables[s * 4..s * 4 + 4]);
            positions.push(pos);
        }
        // Rewrite the pool through the KV write for row 0's position, checking it lands.
        let poolb = buf(dev, &pool);
        let qb = buf(dev, &bits(&q));
        let tb = buf(dev, &row_tables);
        let pb = buf(dev, &positions);
        let ob = zeros(dev, rows * q_heads * head_dim * 2);
        let scores = zeros(dev, rows * q_heads * 16 * 4);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let p = AttnParams {
            q: qb.addr,
            pool: poolb.addr,
            tables: tb.addr,
            positions: pb.addr,
            out: ob.addr,
            scores: scores.addr,
            rows: u(rows),
            q_heads: u(q_heads),
            kv_heads: u(kv_heads),
            head_dim: u(head_dim),
            q_stride: u(q_heads * head_dim),
            q_head_stride: u(head_dim),
            out_stride: u(q_heads * head_dim),
            table_stride: 4,
            layer: u(layer),
            layers: u(layers),
            block_size: u(bs),
            score_stride: 16,
            key_first,
            scale,
        };
        // SAFETY: the buffers hold what the struct names; every visible position is mapped.
        unsafe { dev.attention(&p) }.map_err(|e| e.to_string())?;
        let mut want = Vec::new();
        let group = q_heads / kv_heads;
        for (r, &(s, pos)) in rows_of.iter().enumerate() {
            for h in 0..q_heads {
                let kvh = h / group;
                let ks: Vec<&[f32]> = (key_first as usize..=pos as usize)
                    .map(|x| &keys[s][x][kvh * head_dim..(kvh + 1) * head_dim])
                    .collect();
                let vs: Vec<&[f32]> = (key_first as usize..=pos as usize)
                    .map(|x| &vals[s][x][kvh * head_dim..(kvh + 1) * head_dim])
                    .collect();
                let qh = &q[(r * q_heads + h) * head_dim..(r * q_heads + h + 1) * head_dim];
                let mut o = vec![0.0; head_dim];
                attend(qh, &ks, &vs, scale, &mut o);
                want.extend(o);
            }
        }
        same(
            &format!("attention d{head_dim}"),
            &read::<D, u16>(dev, &ob, rows * q_heads * head_dim),
            &bits(&want),
        )?;
        // The KV write: two rows into fresh entries of block 2 and one skipped.
        let kw: Vec<f32> = (0..3 * width).map(|_| g.bf16(1.0)).collect();
        let vw: Vec<f32> = (0..3 * width).map(|_| g.bf16(1.0)).collect();
        let targets = [2 * u(bs) + 1, NO_TARGET, 2 * u(bs) + 3];
        let kb = buf(dev, &bits(&kw));
        let vb = buf(dev, &bits(&vw));
        let tgb = buf(dev, &targets);
        let wp = KvWriteParams {
            k: kb.addr,
            v: vb.addr,
            pool: poolb.addr,
            targets: tgb.addr,
            rows: 3,
            width: u(width),
            k_stride: u(width),
            v_stride: u(width),
            layer: u(layer),
            layers: u(layers),
            block_size: u(bs),
            reserved: 0,
        };
        // SAFETY: the targets are entries of the pool.
        unsafe { dev.kv_write(&wp) }.map_err(|e| e.to_string())?;
        let mut want_pool = pool.clone();
        for (r, &t) in targets.iter().enumerate() {
            if t == NO_TARGET {
                continue;
            }
            let (block, off) = (t as usize / bs, t as usize % bs);
            let base = (block * layers + layer) * 2 * bs * width + off * width;
            want_pool[base..base + width].copy_from_slice(&bits(&kw[r * width..(r + 1) * width]));
            want_pool[base + bs * width..base + bs * width + width]
                .copy_from_slice(&bits(&vw[r * width..(r + 1) * width]));
        }
        same(
            "kv write",
            &read::<D, u16>(dev, &poolb, pool.len()),
            &want_pool,
        )?;
    }
    Ok(())
}

/// One DeltaNet layer's per-token path (`token_core`) over a sequence, run as the
/// executor runs it: the whole sequence from the zero state with state writes at three
/// positions and a live store, then a second run resuming from a stored state with a fold
/// of the positions after it before the outputs.
pub fn deltanet<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    for &(k_heads, v_heads, v_dim) in &[(2usize, 6usize, 64usize), (1, 2, 128)] {
        let spec = DeltaNetSpec {
            num_k_heads: k_heads,
            num_v_heads: v_heads,
            k_head_dim: 128,
            v_head_dim: v_dim,
            conv_kernel: 4,
        };
        let cd = spec.conv_dim();
        let vd = spec.value_dim();
        let t = 11usize;
        let mixed: Vec<f32> = (0..t * cd).map(|_| g.bf16(3.0)).collect();
        let z: Vec<f32> = (0..t * vd).map(|_| g.bf16(3.0)).collect();
        let a: Vec<f32> = (0..t * v_heads).map(|_| g.bf16(4.0)).collect();
        let b: Vec<f32> = (0..t * v_heads).map(|_| g.bf16(4.0)).collect();
        let conv: Vec<f32> = (0..cd * 4).map(|_| g.bf16(0.5)).collect();
        let a_log: Vec<f32> = (0..v_heads)
            .map(|_| bf16_round(-3.0 + g.f32() * 2.0))
            .collect();
        let mut dt_bias: Vec<f32> = (0..v_heads).map(|_| g.bf16(6.0)).collect();
        dt_bias[0] = 19.25;
        let norm: Vec<f32> = (0..v_dim).map(|_| 0.85 + g.bf16(0.05)).collect();
        let w = CoreWeights {
            conv: &conv,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm: &norm,
        };
        // The reference: token by token, the states after each token kept.
        let mut st = DeltaNetState::new(&spec);
        let mut outs = Vec::new();
        let mut states = vec![st.recurrent.clone()];
        for p in 0..t {
            let o = token_core(
                &spec,
                &mut st,
                &mixed[p * cd..(p + 1) * cd],
                &z[p * vd..(p + 1) * vd],
                &a[p * v_heads..(p + 1) * v_heads],
                &b[p * v_heads..(p + 1) * v_heads],
                &w,
                1e-6,
            );
            outs.extend(o);
            states.push(st.recurrent.clone());
        }
        // Staging as the executor builds it: three zero rows, then the rows.
        let mut conv_in = vec![0.0f32; 3 * cd];
        conv_in.extend(&mixed);
        let window: Vec<u32> = (0..u(t)).collect();
        let mut ab: Vec<f32> = Vec::new();
        for p in 0..t {
            ab.extend(&a[p * v_heads..(p + 1) * v_heads]);
            ab.extend(&b[p * v_heads..(p + 1) * v_heads]);
        }
        let exp_a: Vec<f32> = a_log
            .iter()
            .map(|&x| eidola_engine_model::qwen::numeric::exp32(x))
            .collect();
        let cin = buf(dev, &conv_in);
        let wb = buf(dev, &window);
        let abb = buf(dev, &ab);
        let cwb = buf(dev, &conv);
        let eab = buf(dev, &exp_a);
        let dtb = buf(dev, &dt_bias);
        let key_dim = spec.key_dim();
        let qb = zeros(dev, t * key_dim * 4);
        let kb = zeros(dev, t * key_dim * 4);
        let vb = zeros(dev, t * vd * 4);
        let betab = zeros(dev, t * v_heads * 4);
        let alphab = zeros(dev, t * v_heads * 4);
        let prep = DnPrepParams {
            conv_in: cin.addr,
            window: wb.addr,
            ab: abb.addr,
            conv_w: cwb.addr,
            exp_a_log: eab.addr,
            dt_bias: dtb.addr,
            q: qb.addr,
            k: kb.addr,
            v: vb.addr,
            beta: betab.addr,
            alpha: alphab.addr,
            positions: u(t),
            k_heads: u(k_heads),
            v_heads: u(v_heads),
            k_dim: 128,
            v_dim: u(v_dim),
            reserved: 0,
        };
        // SAFETY: the buffers hold what the struct names.
        unsafe { dev.dn_prep(&prep) }.map_err(|e| e.to_string())?;
        let state_bytes = spec.state_len() * 4;
        let live = zeros(dev, state_bytes);
        let writes: Vec<super::Buf> = (0..3).map(|_| zeros(dev, state_bytes)).collect();
        let write_after = [3u32, 4, 8, 0];
        let ob = zeros(dev, t * vd * 4);
        let mut seq = DnSeq {
            src: 0,
            live: live.addr,
            first: 0,
            count: u(t),
            out_from: 0,
            live_after: 9,
            writes: 3,
            write_after,
            ..DnSeq::default()
        };
        for (i, w) in writes.iter().enumerate() {
            seq.write_dst[i] = w.addr;
        }
        let seqs = buf(dev, &[seq]);
        let rec = DnRecParams {
            seqs: seqs.addr,
            q: qb.addr,
            k: kb.addr,
            v: vb.addr,
            beta: betab.addr,
            alpha: alphab.addr,
            out: ob.addr,
            state_offset: 0,
            seq_count: 1,
            k_heads: u(k_heads),
            v_heads: u(v_heads),
            k_dim: 128,
            v_dim: u(v_dim),
            reserved: 0,
        };
        let zb = buf(dev, &bits(&z));
        let nwb = buf(dev, &norm);
        let nb = zeros(dev, t * vd * 2);
        let dn = DnNormParams {
            o: ob.addr,
            z: zb.addr,
            weight: nwb.addr,
            out: nb.addr,
            rows: u(t),
            v_heads: u(v_heads),
            v_dim: u(v_dim),
            z_stride: u(vd),
            eps: 1e-6,
            reserved: 0,
        };
        // SAFETY: the buffers hold what the structs name.
        unsafe {
            dev.dn_rec(&rec).map_err(|e| e.to_string())?;
            dev.dn_norm(&dn).map_err(|e| e.to_string())?;
        }
        same(
            "deltanet output",
            &read::<D, u16>(dev, &nb, t * vd),
            &bits(&outs),
        )?;
        for (i, w) in writes.iter().enumerate() {
            same(
                &format!("state after {}", write_after[i]),
                &f32_bits(&read::<D, f32>(dev, w, spec.state_len())),
                &f32_bits(&states[write_after[i] as usize]),
            )?;
        }
        same(
            "live state",
            &f32_bits(&read::<D, f32>(dev, &live, spec.state_len())),
            &f32_bits(&states[9]),
        )?;
        // Resume from the state after 4 positions: fold 4..8 without output, then output
        // 8..11, storing at 6 and the end.
        let src = &writes[1];
        let w2 = zeros(dev, state_bytes);
        let live2 = zeros(dev, state_bytes);
        let ob2 = zeros(dev, t * vd * 4);
        let mut seq2 = DnSeq {
            src: src.addr,
            live: live2.addr,
            first: 4,
            count: 7,
            out_from: 4,
            live_after: 7,
            writes: 1,
            ..DnSeq::default()
        };
        seq2.write_after[0] = 2;
        seq2.write_dst[0] = w2.addr;
        let seqs2 = buf(dev, &[seq2]);
        let rec2 = DnRecParams {
            seqs: seqs2.addr,
            out: ob2.addr,
            ..rec
        };
        // SAFETY: as above.
        unsafe { dev.dn_rec(&rec2) }.map_err(|e| e.to_string())?;
        let got: Vec<u32> = f32_bits(&read::<D, f32>(dev, &ob2, t * vd)[8 * vd..]);
        let want: Vec<u32> = f32_bits(&read::<D, f32>(dev, &ob, t * vd)[8 * vd..]);
        same("resumed output", &got, &want)?;
        same(
            "resumed write",
            &f32_bits(&read::<D, f32>(dev, &w2, spec.state_len())),
            &f32_bits(&states[6]),
        )?;
        same(
            "resumed live",
            &f32_bits(&read::<D, f32>(dev, &live2, spec.state_len())),
            &f32_bits(&states[11]),
        )?;
        let _ = widen;
    }
    Ok(())
}

/// The quantiser against `act_quant`: every code and every scale's bits, over 4,096 rows
/// of 128-blocks of every magnitude (zeros, ties, spikes).
pub fn quant<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    let (rows, k) = (4096usize, 256usize);
    let mut x = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let scale = f32::from_bits(
            0x3f80_0000u32.wrapping_add(((u32::try_from(r % 200).expect("small")) << 23) / 8),
        ) * 1e-3;
        for _ in 0..k {
            x.push(g.bf16(scale));
        }
    }
    x[..128].fill(0.0);
    let m_pad = rows;
    let xb = buf(dev, &bits(&x));
    let cb = zeros(dev, rows * k);
    let sb = zeros(dev, (k / 128) * m_pad * 4);
    let q = QuantParams {
        x: xb.addr,
        codes: cb.addr,
        scales: sb.addr,
        rows: u(rows),
        k: u(k),
        x_stride: u(k),
        m_pad: u(m_pad),
    };
    // SAFETY: the buffers hold what the struct names.
    unsafe { dev.quant(&q) }.map_err(|e| e.to_string())?;
    let codes = read::<D, u8>(dev, &cb, rows * k);
    let scales = read::<D, f32>(dev, &sb, (k / 128) * m_pad);
    for r in 0..rows {
        let mut c = vec![0.0f32; k];
        let mut s = vec![0.0f32; k / 128];
        act_quant(&x[r * k..(r + 1) * k], &mut c, &mut s);
        for b in 0..k / 128 {
            if scales[b * m_pad + r].to_bits() != s[b].to_bits() {
                return Err(format!(
                    "quant: row {r} block {b} scale {:e}, want {:e}",
                    scales[b * m_pad + r],
                    s[b]
                ));
            }
        }
        for j in 0..k {
            let want = c[j];
            let dec = eidola_engine_model::numeric::fp8_e4m3_to_f32(codes[r * k + j]);
            if dec.to_bits() != want.to_bits() && !(dec == 0.0 && want == 0.0) {
                return Err(format!(
                    "quant: row {r} element {j}: code {:#x} = {dec}, want {want}",
                    codes[r * k + j]
                ));
            }
        }
    }
    Ok(())
}

/// The f32 → BF16 rounding (both outputs) and the BF16 → f32 widening against
/// `bf16_round`.
pub fn conversions<D: QwenDevice>(dev: &D, seed: u64) -> Result<(), String> {
    let mut g = Lcg(seed);
    let (rows, width) = (7usize, 300usize);
    let x: Vec<f32> = (0..rows * width).map(|_| g.f32() * 1e3).collect();
    let xb = buf(dev, &x);
    let want: Vec<f32> = x.iter().map(|&v| bf16_round(v)).collect();
    for out_f32 in [0u32, 1] {
        let ob = zeros(dev, rows * width * 4);
        let p = RoundParams {
            x: xb.addr,
            out: ob.addr,
            rows: u(rows),
            width: u(width),
            x_stride: u(width),
            out_stride: u(width),
            out_f32,
            reserved: 0,
        };
        // SAFETY: the buffers hold what the struct names.
        unsafe { dev.round(&p) }.map_err(|e| e.to_string())?;
        if out_f32 == 1 {
            same(
                "round f32",
                &f32_bits(&read::<D, f32>(dev, &ob, rows * width)),
                &f32_bits(&want),
            )?;
        } else {
            let got = read::<D, u16>(dev, &ob, rows * width);
            same("round", &got, &bits(&want))?;
            let wb = zeros(dev, rows * width * 4);
            let w = WidenParams {
                x: ob.addr,
                out: wb.addr,
                rows: u(rows),
                width: u(width),
                x_stride: u(width),
                out_stride: u(width),
            };
            // SAFETY: as above.
            unsafe { dev.widen(&w) }.map_err(|e| e.to_string())?;
            same(
                "widen",
                &f32_bits(&read::<D, f32>(dev, &wb, rows * width)),
                &f32_bits(&want),
            )?;
        }
    }
    Ok(())
}

/// Every case, named.
pub fn all<D: QwenDevice>(dev: &D) -> Vec<(&'static str, Result<(), String>)> {
    vec![
        ("norm", norm(dev, 1)),
        ("quant", quant(dev, 7)),
        ("conversions", conversions(dev, 8)),
        ("lin8", lin8(dev, 2)),
        ("linear", linear(dev, 3)),
        ("gate", gate(dev, 4)),
        ("attention", attention(dev, 5)),
        ("deltanet", deltanet(dev, 6)),
    ]
}

/// What the fast profile's linears do against the exact ones: per shape and row count,
/// how many BF16 outputs differ and by how many units in the last place (recorded, not
/// asserted: the bound is fixed from these measurements), and whether every row's outputs
/// are the same bits computed alone and inside every larger batch (asserted: the fast
/// profile's reproducibility between calls rests on it).
pub fn fast_linears<E: QwenDevice, F: QwenDevice>(
    exact: &E,
    fast: &F,
    seed: u64,
) -> Result<String, String> {
    let mut g = Lcg(seed);
    let mut report = Vec::new();
    for &(n, k) in &[(1024usize, 640usize), (512, 1536), (256, 128)] {
        let (codes, scales) = fp8_weight(&mut g, n, k);
        let ts = transpose_scales(&scales, n, k);
        let rows_max = 64usize;
        let x: Vec<f32> = (0..rows_max * k).map(|_| g.bf16(3.0)).collect();
        let run = |dev: &dyn Fn(&QuantParams, &Lin8Params) -> Result<(), String>,
                   rows: usize,
                   first: usize,
                   xb: u64,
                   cb: u64,
                   sb: u64,
                   ob: u64,
                   wb: u64,
                   wsb: u64| {
            let m_pad = rows.next_multiple_of(4);
            let q = QuantParams {
                x: xb + (first * k * 2) as u64,
                codes: cb,
                scales: sb,
                rows: u(rows),
                k: u(k),
                x_stride: u(k),
                m_pad: u(m_pad),
            };
            let l = Lin8Params {
                a: cb,
                a_scales: sb,
                w: wb,
                w_scales: wsb,
                out: ob,
                rows: u(rows),
                n: u(n),
                k: u(k),
                m_pad: u(m_pad),
                out_stride: u(n),
                reserved: 0,
            };
            dev(&q, &l)
        };
        let launch_e = |q: &QuantParams, l: &Lin8Params| -> Result<(), String> {
            // SAFETY: the buffers below hold what the structs name.
            unsafe {
                exact.quant(q).map_err(|e| e.to_string())?;
                exact.lin8(l).map_err(|e| e.to_string())
            }
        };
        let launch_f = |q: &QuantParams, l: &Lin8Params| -> Result<(), String> {
            // SAFETY: as above, on the fast device.
            unsafe {
                fast.quant(q).map_err(|e| e.to_string())?;
                fast.lin8(l).map_err(|e| e.to_string())
            }
        };
        let (xe, we, wse) = (buf(exact, &bits(&x)), buf(exact, &codes), buf(exact, &ts));
        let (xf, wf, wsf) = (buf(fast, &bits(&x)), buf(fast, &codes), buf(fast, &ts));
        let ce = zeros(exact, rows_max * k);
        let se = zeros(exact, (k / 128) * rows_max * 4);
        let oe = zeros(exact, rows_max * n * 2);
        let cf = zeros(fast, rows_max * k);
        let sf = zeros(fast, (k / 128) * rows_max * 4);
        let of = zeros(fast, rows_max * n * 2);
        run(
            &launch_e, rows_max, 0, xe.addr, ce.addr, se.addr, oe.addr, we.addr, wse.addr,
        )?;
        run(
            &launch_f, rows_max, 0, xf.addr, cf.addr, sf.addr, of.addr, wf.addr, wsf.addr,
        )?;
        let want = read::<E, u16>(exact, &oe, rows_max * n);
        let batch = read::<F, u16>(fast, &of, rows_max * n);
        let mut hist = [0usize; 5];
        for (a, b) in want.iter().zip(&batch) {
            let d = usize::try_from((ord16(*a) - ord16(*b)).unsigned_abs()).expect("small");
            hist[d.min(4)] += 1;
        }
        report.push(format!("{n}x{k}: ulp 0/1/2/3/4+ = {hist:?}"));
        // Every row alone and in batches of 5, 17 and 64, at several offsets: one bits.
        for &(rows, first) in &[(1usize, 0usize), (1, 13), (1, 63), (5, 3), (17, 40)] {
            let o = zeros(fast, rows.next_multiple_of(4) * n * 2);
            run(
                &launch_f, rows, first, xf.addr, cf.addr, sf.addr, o.addr, wf.addr, wsf.addr,
            )?;
            let got = read::<F, u16>(fast, &o, rows * n);
            same(
                &format!(
                    "fast {n}x{k}: rows {first}..{} alone against the batch",
                    first + rows
                ),
                &got,
                &batch[first * n..(first + rows) * n],
            )?;
        }
    }
    Ok(report.join("; "))
}
