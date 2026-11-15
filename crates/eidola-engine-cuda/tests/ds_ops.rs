//! The DeepSeek-V4.1 scalar kernels (`ds_ops`, `ds_ops_div`) against the
//! model crate's reference functions, bit for bit, on every image this device
//! runs.
//!
//! Without a device: the vector file the kernels crate's host check runs
//! against is current, and every case the device suite builds itself is
//! built. With one (`device_suite`): the seeded cases of every kernel at
//! small shapes and at the released widths; outputs written over their
//! inputs; a row launched alone against the same row in its batch; the
//! quantisers over every finite BF16 value and at the scale floors, zeros and
//! exact scale boundaries; launches with no rows; the casts over every f32
//! through the numerics probe; and each kernel at the released model's
//! largest shapes, checked and then timed. The suite writes `ds_ops.json`.

mod common;
mod ds_cases;

use common::setup;
use ds_cases::device::{How, IN_PLACE_KERNELS, Kernels, Prepared, ROW_KERNELS, check};
use ds_cases::suite::Summary;
use ds_cases::{Buf, Case, OPS_KERNELS, Regime, Rng, fake_quant_packed, quant_fp8, sweep, vals};
use eidola_engine_cuda::ds::abi;
use eidola_engine_cuda::launch::dptr;

#[test]
fn vectors_are_current() {
    ds_cases::check_or_update("ds_ops.vectors", &OPS_KERNELS);
}

/// Every finite BF16 value as the first element of a 32-wide block, beside 31
/// elements within a few binades of it: 255 blocks a row.
fn every_bf16() -> Vec<u16> {
    let mut g = Rng(0xbf16);
    let mut x = Vec::new();
    for bits in 0..=u16::MAX {
        let exp = u32::from(bits >> 7) & 0xff;
        if exp == 255 {
            continue;
        }
        x.push(bits);
        x.extend(g.bf16_range(31, exp.saturating_sub(8).clamp(1, 243), 12));
    }
    x
}

/// Blocks at the quantisers' edges: zeros of both signs, the smallest values,
/// values either side of each scale floor, exact powers of two times the
/// largest code (where the scale steps), equal elements, one large element.
fn edge_blocks() -> Vec<u16> {
    let b = |x: f32| ds_cases::bits(x);
    let mut blocks: Vec<[u16; 32]> = vec![[0; 32], [0x8000; 32], [0x0001; 32], [0x8001; 32]];
    let mut with = |head: &[u16]| {
        let mut block = [0u16; 32];
        block[..head.len()].copy_from_slice(head);
        blocks.push(block);
    };
    with(&[0x0000, 0x8000, 0x0001, 0x8001, 0x007f, 0x0080]);
    // 1e-4 (the FP8 floor), 6·2^-9 and 6·2^-126 (the FP4 floors), as BF16
    // neighbours.
    for floor in [0x38d1u16, 0x3c40, 0x01c0] {
        for near in [floor - 1, floor, floor + 1] {
            with(&[near, near | 0x8000, near - 3]);
        }
    }
    for k in [-100i32, -9, -1, 0, 1, 20, 100] {
        let s = 2f32.powi(k);
        for top in [448.0f32, 6.0, 7.0, 1.0] {
            let t = b(top * s);
            with(&[t - 1, t, t + 1, b(-top * s), b(0.5 * top * s), b(0.25 * s)]);
        }
        // Midpoints between codes: ties.
        with(&[
            b(6.0 * s),
            b(5.0 * s),
            b(3.5 * s),
            b(2.5 * s),
            b(1.75 * s),
            b(0.25 * s),
        ]);
        with(&[
            b(448.0 * s),
            b(432.0 * s),
            b(13.5 * s),
            b(2.125 * s),
            b(-2.375 * s),
        ]);
    }
    blocks.push([b(3.0); 32]);
    blocks.push([0x7f7f; 32]);
    blocks.concat()
}

/// A quantiser case over `x` (rows of `width`): the FP8 activation quantiser
/// (`format` `None`), or a fake quantisation without rotation.
fn quantiser_case(x: &[u16], width: usize, format: Option<u32>) -> Case {
    let rows = x.len() / width;
    let mut n = std::collections::BTreeMap::new();
    let mut set = |name: &'static str, v: usize| {
        n.insert(name, u32::try_from(v).unwrap());
    };
    set("rows", rows);
    let (kernel, names, packed): (_, _, Vec<(Vec<u8>, Vec<u8>)>) = match format {
        None => {
            set("width", width);
            set("x_format", 0);
            (
                "quant_fp8",
                ["codes", "scales"],
                x.chunks_exact(width).map(|r| quant_fp8(&vals(r))).collect(),
            )
        }
        Some(format) => {
            set("heads", 1);
            set("head_dim", width);
            set("rope_dim", 0);
            set("rotation", 0);
            set("format", format as usize);
            set("cis_positions", 0);
            (
                if format == abi::KV_FP4_E4M3 {
                    "rot_quant_e4m3"
                } else {
                    "rot_quant"
                },
                ["dst_codes", "dst_scales"],
                x.chunks_exact(width)
                    .map(|r| fake_quant_packed(&vals(r), format))
                    .collect(),
            )
        }
    };
    let codes: Vec<u8> = packed.iter().flat_map(|p| p.0.clone()).collect();
    let scales: Vec<u8> = packed.iter().flat_map(|p| p.1.clone()).collect();
    let output = |name, after: Vec<u8>| ds_cases::Output {
        name,
        before: Buf::U8(vec![ds_cases::SENTINEL_U8; after.len()]),
        after: Buf::U8(after),
    };
    Case {
        kernel,
        n,
        f: Default::default(),
        inputs: [("x", Buf::U16(x.to_vec()))].into(),
        outputs: vec![output(names[0], codes), output(names[1], scales)],
    }
}

const FORMATS: [Option<u32>; 4] = [
    None,
    Some(abi::KV_FP8),
    Some(abi::KV_FP4_E8M0),
    Some(abi::KV_FP4_E4M3),
];

/// What the device suite builds for itself is built, and is what it says.
#[test]
fn device_cases_are_built_without_a_device() {
    let edges = edge_blocks();
    assert_eq!(edges.len() % 32, 0);
    for format in FORMATS {
        let case = quantiser_case(&edges, 32, format);
        assert_eq!(case.u("rows") as usize, edges.len() / 32);
    }
    // The first rows of the exhaustive input: built whole only for a device.
    let all = every_bf16();
    assert_eq!(all.len(), 65_280 * 32);
    for (block, bits) in all
        .chunks_exact(32)
        .zip((0..=u16::MAX).filter(|b| b >> 7 & 0xff != 255))
    {
        assert_eq!(block[0], bits);
    }
    for format in FORMATS {
        quantiser_case(&all[..2 * 8160], 8160, format);
    }
    // Every kernel's largest shapes can be drawn where that is cheap.
    for kernel in ["embed", "quant_fp8", "collapse", "place", "engram_gather"] {
        let case = ds_cases::build(kernel, Regime::Released, &mut Rng(1));
        assert!(!case.outputs.is_empty(), "{kernel}");
    }
}

/// The exhaustive digest of a one-argument function: the wrapping sum over
/// every f32 `b` of `fin((b << 32) | out(b))`, a NaN result canonical.
fn exhaustive_digest(
    gpu: &eidola_engine_cuda::Gpu,
    probe: &eidola_engine_cuda::Kernel,
    op: u32,
) -> u64 {
    fn fin(mut z: u64) -> u64 {
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    const CHUNK: u64 = 1 << 24;
    let s = gpu.stream();
    let mut input = s
        .alloc_zeros::<u64>(usize::try_from(CHUNK).unwrap())
        .unwrap();
    let out = s
        .alloc_zeros::<u64>(usize::try_from(CHUNK).unwrap())
        .unwrap();
    let mut sum = 0u64;
    for chunk in 0..(1u64 << 32) / CHUNK {
        let words: Vec<u64> = (chunk * CHUNK..(chunk + 1) * CHUNK).collect();
        s.memcpy_htod(&words, &mut input).unwrap();
        let p = abi::ProbeParams {
            input: dptr(&input, s),
            out: dptr(&out, s),
            items: u32::try_from(CHUNK).unwrap(),
            op,
            in_words: 1,
            out_words: 1,
        };
        // SAFETY: both arrays hold `items` words.
        unsafe {
            eidola_engine_cuda::launch!(gpu, probe, [u32::try_from(CHUNK / 256).unwrap(), 1, 1], p)
                .unwrap();
        }
        for (b, o) in words.iter().zip(s.clone_dtoh(&out).unwrap()) {
            let bits = u32::try_from(o).expect("an f32 result");
            let word = if f32::from_bits(bits).is_nan() {
                0x7fc0_0000
            } else {
                u64::from(bits)
            };
            sum = sum.wrapping_add(fin(b << 32 | word));
        }
    }
    sum
}

/// The committed digest of `name` (`tests/data/ds_numeric.exhaustive`).
fn committed_digest(name: &str) -> u64 {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../eidola-engine-kernels/tests/data/ds_numeric.exhaustive");
    let text = std::fs::read_to_string(path).unwrap();
    let line = text
        .lines()
        .find(|l| l.starts_with(&format!("x {name} = ")))
        .unwrap_or_else(|| panic!("no digest of {name}"));
    u64::from_str_radix(line.rsplit(' ').next().unwrap(), 16).unwrap()
}

#[test]
fn committed_digests_are_read() {
    for name in ["bf16_round", "round_e4m3", "round_e2m1"] {
        assert_ne!(committed_digest(name), 0);
    }
}

#[test]
fn device_suite() {
    let Some(su) = setup() else { return };
    let gpu = &su.gpu;
    let mut summary = Summary::new("ds_ops", gpu);
    for (i, &arch) in su.archs.iter().enumerate() {
        let k = Kernels::load(gpu, &su.dir, arch);
        for kernel in OPS_KERNELS {
            summary.check(format!("{arch:?}/cases/{kernel}"), || {
                for wide in [false, true] {
                    for (j, case) in sweep(kernel, wide).iter().enumerate() {
                        check(
                            gpu,
                            &k,
                            case,
                            How::default(),
                            &format!("case {j} wide {wide}"),
                        );
                    }
                }
            });
            if IN_PLACE_KERNELS.contains(&kernel) {
                summary.check(format!("{arch:?}/in_place/{kernel}"), || {
                    let how = How {
                        in_place: true,
                        ..How::default()
                    };
                    for wide in [false, true] {
                        for (j, case) in sweep(kernel, wide).iter().enumerate() {
                            check(
                                gpu,
                                &k,
                                case,
                                how,
                                &format!("in place, case {j} wide {wide}"),
                            );
                        }
                    }
                });
            }
            if ROW_KERNELS.contains(&kernel) {
                summary.check(format!("{arch:?}/row_alone/{kernel}"), || {
                    for wide in [false, true] {
                        for (j, case) in sweep(kernel, wide).iter().take(32).enumerate() {
                            for row in 0..case.u("rows") as usize {
                                let how = How {
                                    row: Some(row),
                                    ..How::default()
                                };
                                check(gpu, &k, case, how, &format!("row {row} alone, case {j}"));
                            }
                        }
                    }
                });
            }
            summary.check(format!("{arch:?}/no_rows/{kernel}"), || {
                // A launch over no rows writes nothing.
                let mut case = sweep(kernel, false).swap_remove(0);
                let rows = if kernel == "pool" { "entries" } else { "rows" };
                case.n.insert(rows, 0);
                if kernel == "place" {
                    return;
                }
                let prepared = Prepared::new(gpu, &case);
                prepared.launch(gpu, &k, How::default()).unwrap();
                for (o, got) in case
                    .outputs
                    .iter()
                    .zip(prepared.outputs(gpu, How::default()))
                {
                    assert!(got == o.before, "{} was written", o.name);
                }
            });
        }
        summary.check(format!("{arch:?}/quantisers/edges"), || {
            let edges = edge_blocks();
            for format in FORMATS {
                for width in [32, 96] {
                    let x = &edges[..edges.len() / width * width];
                    check(
                        gpu,
                        &k,
                        &quantiser_case(x, width, format),
                        How::default(),
                        "edges",
                    );
                }
            }
        });
        summary.check(format!("{arch:?}/quantisers/every_bf16"), || {
            let all = every_bf16();
            for format in FORMATS {
                check(
                    gpu,
                    &k,
                    &quantiser_case(&all, 8160, format),
                    How::default(),
                    "every BF16",
                );
            }
        });
        // The largest shapes, the timings and the 2^32 sweeps once, on the
        // device's own image.
        if i > 0 {
            continue;
        }
        for kernel in OPS_KERNELS {
            summary.released(gpu, &k, kernel, 50);
        }
        let probe = su
            .module("ds_probe", arch)
            .kernel("eidola_ds_probe")
            .unwrap();
        for (name, op) in [
            ("bf16_round", abi::PROBE_BF16_ROUND),
            ("round_e4m3", abi::PROBE_ROUND_E4M3),
            ("round_e2m1", abi::PROBE_ROUND_E2M1),
        ] {
            summary.check(format!("{arch:?}/every_f32/{name}"), || {
                assert_eq!(
                    exhaustive_digest(gpu, &probe, op),
                    committed_digest(name),
                    "{name} over every f32"
                );
            });
        }
    }
    summary.finish();
}

/// Every kernel's case at the released model's largest shapes, with its
/// reference outputs: what the device suite checks and times.
#[test]
#[ignore = "the largest shapes: run with --release"]
fn released_cases_are_built() {
    for kernel in OPS_KERNELS {
        let case = ds_cases::build(kernel, Regime::Released, &mut Rng(0xbe9c));
        assert!(!case.outputs.is_empty(), "{kernel}");
    }
}
