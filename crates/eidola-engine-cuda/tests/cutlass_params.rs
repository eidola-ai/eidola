//! The Rust-built CUTLASS `Params` against CUTLASS's own host path.
//!
//! `tests/data/cutlass_params.jsonl` is the output of the dev-only oracle
//! (`oracle/cutlass_params.cu`, run on a GPU host against the pinned CUTLASS
//! tree): for each problem, the Params bytes, which of them CUTLASS defines
//! (the rest is padding), the grid, and every descriptor-encoding call.
//!
//! Without a GPU, the descriptors come from the recording and everything else
//! is checked; with one, the descriptors are encoded by the driver too and the
//! whole defined image must match.

use eidola_engine_cuda::gemm::{GemmArgs, GemmKind, PARAMS_BYTES};
use eidola_engine_cuda::tma::{TmaSpec, TmaSwizzle, TmaType};
use serde_json::Value;

const A: u64 = 0x7f00_0000_0000;
const B: u64 = 0x7f10_0000_0000;
const D: u64 = 0x7f20_0000_0000;
const SFA: u64 = 0x7f30_0000_0000;
const SFB: u64 = 0x7f40_0000_0000;

struct Case {
    kind: GemmKind,
    args: GemmArgs,
    can_implement: bool,
    grid: [u32; 3],
    params: Vec<u8>,
    defined: Vec<bool>,
    encodes: Vec<Value>,
}

fn cases() -> Vec<Case> {
    include_str!("data/cutlass_params.jsonl")
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            let kind = match v["kernel"].as_str().unwrap() {
                "fp8_blockwise" => GemmKind::Fp8Blockwise,
                "bf16" => GemmKind::Bf16,
                k => panic!("{k}"),
            };
            let u = |k: &str| u32::try_from(v[k].as_u64().unwrap()).unwrap();
            let hex = v["params"].as_str().unwrap();
            Case {
                kind,
                args: GemmArgs {
                    m: u("m"),
                    n: u("n"),
                    k: u("k"),
                    a: A,
                    b: B,
                    d: D,
                    sfa: SFA,
                    sfb: SFB,
                    alpha: f32_exact(v["alpha"].as_f64().unwrap()),
                },
                can_implement: v["can_implement"].as_bool().unwrap(),
                grid: [0, 1, 2].map(|i| u32::try_from(v["grid"][i].as_u64().unwrap()).unwrap()),
                params: (0..hex.len() / 2)
                    .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
                    .collect(),
                defined: v["defined"]
                    .as_str()
                    .unwrap()
                    .bytes()
                    .map(|b| b == b'1')
                    .collect(),
                encodes: v["encodes"].as_array().unwrap().clone(),
            }
        })
        .collect()
}

fn spec_matches(spec: &TmaSpec, rec: &Value) {
    let dtype = match spec.dtype {
        TmaType::U8 => 0,
        TmaType::I32 => 3,
        TmaType::F32 => 7,
        TmaType::Bf16 => 9,
        TmaType::Fp4Unpacked => 14,
    };
    let swizzle = match spec.swizzle {
        TmaSwizzle::None => 0,
        TmaSwizzle::B32 => 1,
        TmaSwizzle::B64 => 2,
        TmaSwizzle::B128 => 3,
    };
    let list = |k: &str| -> Vec<u64> {
        rec[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect()
    };
    assert_eq!(rec["dtype"].as_u64().unwrap(), dtype, "{spec:?} vs {rec}");
    assert_eq!(
        rec["addr"].as_u64().unwrap(),
        spec.addr,
        "{spec:?} vs {rec}"
    );
    assert_eq!(list("dims"), spec.dims, "{spec:?} vs {rec}");
    assert_eq!(list("strides"), spec.strides, "{spec:?} vs {rec}");
    let box_dims: Vec<u64> = spec.box_dims.iter().map(|&b| b as u64).collect();
    assert_eq!(list("box"), box_dims, "{spec:?} vs {rec}");
    assert_eq!(list("elem_strides"), vec![1; spec.dims.len()]);
    assert_eq!(
        rec["swizzle"].as_u64().unwrap(),
        swizzle,
        "{spec:?} vs {rec}"
    );
    assert_eq!(rec["interleave"].as_u64().unwrap(), 0);
    assert_eq!(rec["l2"].as_u64().unwrap(), 2);
    assert_eq!(rec["oob"].as_u64().unwrap(), 0);
}

fn compare(case: &Case, ours: &[u8; PARAMS_BYTES]) {
    for (i, (&o, &theirs)) in ours.iter().zip(&case.params).enumerate() {
        if case.defined[i] {
            assert_eq!(
                o, theirs,
                "{:?} {:?}: byte {i} differs",
                case.kind, case.args
            );
        }
    }
}

/// Everything but the descriptor bytes themselves, with no GPU: the encode
/// calls (operands, extents, strides, boxes, swizzles), every scalar field, the
/// scheduler and the grid.
#[test]
fn params_match_cutlass_with_recorded_descriptors() {
    let mut checked = 0;
    for case in cases() {
        if !case.can_implement {
            assert!(case.kind.check(&case.args).is_err(), "{:?}", case.args);
            continue;
        }
        // CUTLASS encodes A, B, A (fallback), B (fallback), D.
        let order = [0usize, 1, 4];
        let offsets = [128usize, 384, 1664];
        let mut call = 0;
        let (ours, grid) = case
            .kind
            .params_with(&case.args, 148, |spec| {
                spec_matches(spec, &case.encodes[order[call]]);
                let off = offsets[call];
                call += 1;
                Ok(case.params[off..off + 128].try_into().unwrap())
            })
            .unwrap();
        assert_eq!(call, 3);
        assert_eq!(grid, case.grid, "{:?}", case.args);
        compare(&case, &ours);
        checked += 1;
    }
    assert!(checked >= 10);
}

/// The full defined image, descriptors encoded by this machine's driver.
#[test]
fn params_match_cutlass_with_driver_descriptors() {
    if !eidola_engine_cuda::Gpu::available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let _gpu = eidola_engine_cuda::Gpu::open(0).unwrap();
    for case in cases().into_iter().filter(|c| c.can_implement) {
        let (ours, _) = case
            .kind
            .params_with(&case.args, 148, |s| s.encode())
            .unwrap();
        compare(&case, &ours);
    }
}

/// An f64 the JSON holds for an f32 field, which must round-trip exactly.
fn f32_exact(x: f64) -> f32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "checked to round-trip below"
    )]
    let y = x as f32;
    assert_eq!(f64::from(y), x, "{x} is not an f32");
    y
}
