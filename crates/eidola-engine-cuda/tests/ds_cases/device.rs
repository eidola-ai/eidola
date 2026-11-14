//! A [`Case`] on a device: its arrays uploaded, its kernel launched through
//! the wrappers, its outputs read back.

use std::collections::BTreeMap;

use cudarc::driver::CudaSlice;
use eidola_engine_cuda::ds::abi::*;
use eidola_engine_cuda::ds::gemm_ref::DsGemmRef;
use eidola_engine_cuda::ds::head::DsHead;
use eidola_engine_cuda::ds::ops::DsOps;
use eidola_engine_cuda::launch::dptr;
use eidola_engine_cuda::{Gpu, ImageArch, ImageSource, KernelDir, KernelModule, Result};

use super::{Buf, Case, EPS, SENTINEL_U32};

/// Every DeepSeek scalar kernel of one architecture's images.
pub struct Kernels {
    pub ops: DsOps,
    pub gemm: DsGemmRef,
    pub head: DsHead,
}

impl Kernels {
    pub fn load(gpu: &Gpu, dir: &KernelDir, arch: ImageArch) -> Kernels {
        let module = |name: &str| {
            KernelModule::load_from(gpu, dir, name, ImageSource::Cubin(arch))
                .unwrap_or_else(|e| panic!("{name} {arch:?}: {e}"))
        };
        Kernels {
            ops: DsOps::from_modules(module(DsOps::IMAGE), module(DsOps::IMAGE_DIV)).unwrap(),
            gemm: DsGemmRef::from_module(module(DsGemmRef::IMAGE)).unwrap(),
            head: DsHead::from_module(module(DsHead::IMAGE)).unwrap(),
        }
    }
}

enum Dev {
    U8(CudaSlice<u8>),
    U16(CudaSlice<u16>),
    U32(CudaSlice<u32>),
    F32(CudaSlice<f32>),
}

impl Dev {
    fn upload(gpu: &Gpu, b: &Buf) -> Dev {
        let s = gpu.stream();
        match b {
            Buf::U8(v) => Dev::U8(s.clone_htod(v).unwrap()),
            Buf::U16(v) => Dev::U16(s.clone_htod(v).unwrap()),
            Buf::U32(v) => Dev::U32(s.clone_htod(v).unwrap()),
            Buf::F32(v) => Dev::F32(s.clone_htod(v).unwrap()),
        }
    }

    fn download(&self, gpu: &Gpu) -> Buf {
        let s = gpu.stream();
        match self {
            Dev::U8(d) => Buf::U8(s.clone_dtoh(d).unwrap()),
            Dev::U16(d) => Buf::U16(s.clone_dtoh(d).unwrap()),
            Dev::U32(d) => Buf::U32(s.clone_dtoh(d).unwrap()),
            Dev::F32(d) => Buf::F32(s.clone_dtoh(d).unwrap()),
        }
    }

    /// The address of element `at`.
    fn addr(&self, gpu: &Gpu, at: usize) -> u64 {
        let s = gpu.stream();
        let (base, size) = match self {
            Dev::U8(d) => (dptr(d, s), 1),
            Dev::U16(d) => (dptr(d, s), 2),
            Dev::U32(d) => (dptr(d, s), 4),
            Dev::F32(d) => (dptr(d, s), 4),
        };
        base + (at * size) as u64
    }
}

/// How a prepared case is launched.
#[derive(Clone, Copy, Default)]
pub struct How {
    /// Launch one row alone: `rows` 1, every per-row array advanced to it.
    /// For the kernels [`ROW_KERNELS`] names.
    pub row: Option<usize>,
    /// Write the output over the input (`out` is `x`), where the kernel
    /// allows it: [`IN_PLACE_KERNELS`].
    pub in_place: bool,
}

/// Kernels whose rows are whole rows of every array, so one can be launched
/// alone by advancing the addresses.
pub const ROW_KERNELS: [&str; 10] = [
    "quant_fp8",
    "rms_norm",
    "mhc_mix",
    "collapse",
    "expand",
    "linear",
    "head",
    "swiglu",
    "lin8",
    "grouped_dot",
];
pub const IN_PLACE_KERNELS: [&str; 3] = ["rms_norm", "expand", "engram_inject"];

/// A case's arrays on the device.
pub struct Prepared<'a> {
    case: &'a Case,
    inputs: BTreeMap<&'static str, Dev>,
    outputs: Vec<(&'static str, Dev)>,
    scratch: Option<Dev>,
}

impl<'a> Prepared<'a> {
    pub fn new(gpu: &Gpu, case: &'a Case) -> Prepared<'a> {
        Prepared {
            case,
            inputs: case
                .inputs
                .iter()
                .map(|(name, b)| (*name, Dev::upload(gpu, b)))
                .collect(),
            outputs: case
                .outputs
                .iter()
                .map(|o| (o.name, Dev::upload(gpu, &o.before)))
                .collect(),
            // The placement's counts start as garbage.
            scratch: case
                .n
                .get("groups")
                .map(|groups| Dev::upload(gpu, &Buf::U32(vec![SENTINEL_U32; *groups as usize]))),
        }
    }

    fn find(&self, name: &str) -> Option<&Dev> {
        self.inputs
            .get(name)
            .or_else(|| self.outputs.iter().find(|o| o.0 == name).map(|o| &o.1))
    }

    /// Every output array as the device holds it; under `in_place` the
    /// array named `out` is read from `x`.
    pub fn outputs(&self, gpu: &Gpu, how: How) -> Vec<Buf> {
        gpu.synchronize().unwrap();
        self.outputs
            .iter()
            .map(|(name, dev)| {
                if how.in_place && *name == "out" {
                    self.inputs["x"].download(gpu)
                } else {
                    dev.download(gpu)
                }
            })
            .collect()
    }

    /// Launch the case's kernel (for `lin8`, the quantiser and then the
    /// product).
    pub fn launch(&self, gpu: &Gpu, k: &Kernels, how: How) -> Result<()> {
        let c = self.case;
        let u = |name: &str| c.u(name);
        let row = how.row.unwrap_or(0);
        assert!(
            how.row.is_none() || ROW_KERNELS.contains(&c.kernel),
            "{} has no single-row launch",
            c.kernel
        );
        assert!(!how.in_place || IN_PLACE_KERNELS.contains(&c.kernel));
        let rows = if how.row.is_some() { 1 } else { u("rows") };
        // The address of array `name`, advanced `per_row` elements a row; 0
        // when the case has no such array.
        let at = |name: &str, per_row: u32| {
            let name = if how.in_place && name == "out" {
                "x"
            } else {
                name
            };
            self.find(name)
                .map_or(0, |d| d.addr(gpu, row * per_row as usize))
        };
        let a = |name: &str| at(name, 0);
        // SAFETY: every address is an array of this case, sized by its
        // builder for the counts passed with it.
        unsafe {
            match c.kernel {
                "embed" => k.ops.embed(
                    gpu,
                    &EmbedParams {
                        table: a("table"),
                        tokens: a("tokens"),
                        out: a("out"),
                        status: a("status"),
                        rows,
                        dim: u("dim"),
                        copies: u("copies"),
                        vocab: u("vocab"),
                    },
                ),
                "quant_fp8" => {
                    let width = u("width");
                    k.ops.quant_fp8(
                        gpu,
                        &QuantFp8Params {
                            x: at("x", width),
                            x_scales: at("x_scales", width / 32),
                            codes: at("codes", width),
                            scales: at("scales", width / 32),
                            rows,
                            width,
                            x_format: u("x_format"),
                            reserved: 0,
                        },
                    )
                }
                "rot_quant" | "rot_quant_e4m3" | "rotate" => {
                    let dst = if c.has("targets") {
                        KvPlane {
                            base: a("pool"),
                            block_bytes: u("block_bytes").into(),
                            codes_offset: u("codes_offset").into(),
                            scales_offset: u("scales_offset").into(),
                            entries_per_block: u("entries_per_block"),
                            format: u("format"),
                            width: u("head_dim"),
                            reserved: 0,
                        }
                    } else {
                        KvPlane::default()
                    };
                    k.ops.rot_quant(
                        gpu,
                        &RotQuantParams {
                            x: a("x"),
                            cis: a("cis"),
                            positions: a("positions"),
                            targets: a("targets"),
                            dst_codes: a("dst_codes"),
                            dst_scales: a("dst_scales"),
                            dst,
                            rows,
                            heads: u("heads"),
                            head_dim: u("head_dim"),
                            rope_dim: u("rope_dim"),
                            rotation: u("rotation"),
                            format: u("format"),
                            cis_positions: u("cis_positions"),
                            reserved: 0,
                        },
                    )
                }
                "rms_norm" => {
                    let width = u("width");
                    k.ops.rms_norm(
                        gpu,
                        &RmsNormParams {
                            x: at("x", width),
                            weight: a("weight"),
                            out: at("out", width),
                            rows,
                            width,
                            eps: EPS,
                            reserved: 0,
                        },
                    )
                }
                "mhc_mix" => {
                    let (copies, dim) = (u("copies"), u("dim"));
                    let Buf::F32(scale) = &c.inputs["scale"] else {
                        panic!("scale")
                    };
                    k.ops.mhc_mix(
                        gpu,
                        &MhcMixParams {
                            x: at("x", copies * dim),
                            projection: a("projection"),
                            base: a("base"),
                            out: at("out", mix_words(copies)),
                            rows,
                            dim,
                            copies,
                            iters: u("iters"),
                            scale: [scale[0], scale[1], scale[2]],
                            norm_eps: EPS,
                            eps: EPS,
                            reserved: 0,
                        },
                    )
                }
                "collapse" => {
                    let (copies, dim) = (u("copies"), u("dim"));
                    k.ops.collapse(
                        gpu,
                        &CollapseParams {
                            x: at("x", copies * dim),
                            pre: at("pre", u("pre_stride")),
                            out: at("out", dim),
                            rows,
                            dim,
                            copies,
                            pre_stride: u("pre_stride"),
                        },
                    )
                }
                "expand" => {
                    let (copies, dim) = (u("copies"), u("dim"));
                    k.ops.expand(
                        gpu,
                        &ExpandParams {
                            x: at("x", copies * dim),
                            y: at("y", dim),
                            mixes: at("mixes", u("mix_stride")),
                            out: at("out", copies * dim),
                            rows,
                            dim,
                            copies,
                            mix_stride: u("mix_stride"),
                        },
                    )
                }
                "stream_mean" => k.ops.stream_mean(
                    gpu,
                    &StreamMeanParams {
                        x: a("x"),
                        out: a("out"),
                        rows,
                        dim: u("dim"),
                        copies: u("copies"),
                        out_stride: u("out_stride"),
                    },
                ),
                "linear" | "head" | "markov" => {
                    let (n, kk) = (u("n"), u("k"));
                    let p = LinearParams {
                        // An indexed input is read through `x_rows`, which
                        // is the per-row array.
                        x: at("x", if c.has("x_rows") { 0 } else { kk }),
                        x_rows: at("x_rows", 1),
                        w: a("w"),
                        addend: at("addend", n),
                        out: at("out", n),
                        rows,
                        n,
                        k: kk,
                        x_dtype: u("x_dtype"),
                        w_dtype: u("w_dtype"),
                        out_dtype: u("out_dtype"),
                    };
                    if c.kernel != "linear" {
                        k.head.head(gpu, &p)
                    } else {
                        k.ops.linear(gpu, &p)
                    }
                }
                "pool" => k.ops.pool(
                    gpu,
                    &PoolParams {
                        kv: a("kv"),
                        score: a("score"),
                        first: a("first"),
                        second: a("second"),
                        out: a("out"),
                        entries: u("entries"),
                        width: u("width"),
                    },
                ),
                "route" => k.ops.route(
                    gpu,
                    &RouteParams {
                        logits: a("logits"),
                        bias: a("bias"),
                        ids: a("ids"),
                        weights: a("weights"),
                        rows,
                        experts: u("experts"),
                        active: u("active"),
                        normalize: u("normalize"),
                        route_scale: c.f["route_scale"],
                        reserved: 0,
                    },
                ),
                "place" => k.ops.place(
                    gpu,
                    &PlaceParams {
                        ids: a("ids"),
                        weights: a("weights"),
                        row_of: a("row_of"),
                        row_weight: a("row_weight"),
                        group_end: a("group_end"),
                        scratch: self.scratch.as_ref().unwrap().addr(gpu, 0),
                        rows,
                        active: u("active"),
                        groups: u("groups"),
                        row_align: u("row_align"),
                        layout_rows: u("layout_rows"),
                        reserved: 0,
                    },
                ),
                "swiglu" => {
                    let (width, stride) = (u("width"), u("in_stride"));
                    k.ops.swiglu(
                        gpu,
                        &SwigluParams {
                            gate: at("gate", stride),
                            up: at("up", stride),
                            weight: at("weight", 1),
                            codes: at("codes", width),
                            scales: at("scales", width / 32),
                            rows,
                            width,
                            in_stride: stride,
                            limit: c.f["limit"],
                        },
                    )
                }
                "place_rows" => k.ops.place_rows(
                    gpu,
                    &PlaceRowsParams {
                        src: a("src"),
                        dst: a("dst"),
                        row_of: a("row_of"),
                        status: a("status"),
                        rows,
                        active: u("active"),
                        row_bytes: u("row_bytes"),
                        layout_rows: u("layout_rows"),
                    },
                ),
                "combine" => k.ops.combine(
                    gpu,
                    &CombineParams {
                        routed: a("routed"),
                        shared: a("shared"),
                        row_of: a("row_of"),
                        out: a("out"),
                        rows,
                        active: u("active"),
                        dim: u("dim"),
                        reserved: 0,
                    },
                ),
                "engram_gather" => k.ops.engram_gather(
                    gpu,
                    &EngramGatherParams {
                        codes: a("codes"),
                        scales: a("scales"),
                        ids: a("ids"),
                        out_codes: a("out_codes"),
                        out_scales: a("out_scales"),
                        status: a("status"),
                        table_rows: u("table_rows").into(),
                        rows,
                        ids_per_row: u("ids_per_row"),
                        width: u("width"),
                        reserved: 0,
                    },
                ),
                "engram_inject" => k.ops.engram_inject(
                    gpu,
                    &EngramInjectParams {
                        x: a("x"),
                        kv: a("kv"),
                        gate_weight: a("gate_weight"),
                        out: a("out"),
                        rows,
                        dim: u("dim"),
                        copies: u("copies"),
                        norm_eps: EPS,
                    },
                ),
                "lin8" => {
                    let (n, kk) = (u("n"), u("k"));
                    k.ops.quant_fp8(
                        gpu,
                        &QuantFp8Params {
                            x: at("x", kk),
                            x_scales: 0,
                            codes: at("codes", kk),
                            scales: at("act_scales", kk / 32),
                            rows,
                            width: kk,
                            x_format: X_BF16,
                            reserved: 0,
                        },
                    )?;
                    k.gemm.lin8(
                        gpu,
                        &Lin8Params {
                            codes: at("codes", kk),
                            act_scales: at("act_scales", kk / 32),
                            w_codes: a("w_codes"),
                            w_scales: a("w_scales"),
                            out: at("out", n),
                            rows,
                            n,
                            k: kk,
                            reserved: 0,
                        },
                    )
                }
                "lin4" => k.gemm.lin4(
                    gpu,
                    &Lin4Params {
                        codes: a("codes"),
                        act_scales: a("act_scales"),
                        w_packed: a("w_packed"),
                        w_scales: a("w_scales"),
                        group_end: a("group_end"),
                        out: a("out"),
                        layout_rows: u("layout_rows"),
                        n: u("n"),
                        k: u("k"),
                        groups: u("groups"),
                        row_align: u("row_align"),
                        reserved: 0,
                    },
                ),
                "grouped_dot" => {
                    let (groups, rank, kk) = (u("groups"), u("rank"), u("k"));
                    k.gemm.grouped_dot(
                        gpu,
                        &GroupedDotParams {
                            codes: at("codes", groups * kk),
                            act_scales: at("act_scales", groups * kk / 32),
                            w_codes: a("w_codes"),
                            w_scales: a("w_scales"),
                            out: at("out", groups * rank),
                            rows,
                            groups,
                            rank,
                            k: kk,
                        },
                    )
                }
                other => panic!("no kernel {other}"),
            }
        }
    }
}

/// The first element at which two arrays differ.
fn first_difference(got: &Buf, want: &Buf) -> Option<usize> {
    fn diff<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
        a.iter().zip(b).position(|(x, y)| x != y)
    }
    match (got, want) {
        (Buf::U8(a), Buf::U8(b)) => diff(a, b),
        (Buf::U16(a), Buf::U16(b)) => diff(a, b),
        (Buf::U32(a), Buf::U32(b)) => diff(a, b),
        (Buf::F32(a), Buf::F32(b)) => a
            .iter()
            .zip(b)
            .position(|(x, y)| x.to_bits() != y.to_bits()),
        _ => Some(0),
    }
}

/// Launch `case` and require every output bit to be the reference's.
pub fn check(gpu: &Gpu, k: &Kernels, case: &Case, how: How, what: &str) {
    let prepared = Prepared::new(gpu, case);
    prepared
        .launch(gpu, k, how)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let got = prepared.outputs(gpu, how);
    for (o, got) in case.outputs.iter().zip(&got) {
        // A single-row launch writes its row only: the rows before and after
        // must be as they were.
        let want = match how.row {
            None => o.after.clone(),
            Some(row) => one_row(&o.before, &o.after, row, case.u("rows") as usize),
        };
        if let Some(at) = first_difference(got, &want) {
            panic!(
                "{what}: {} output {} differs from the reference at element {at} of {} ({:?})",
                case.kernel,
                o.name,
                want.len(),
                case.n
            );
        }
    }
}

/// `before` with row `row` of `rows` taken from `after`.
fn one_row(before: &Buf, after: &Buf, row: usize, rows: usize) -> Buf {
    fn splice<T: Clone>(before: &[T], after: &[T], row: usize, rows: usize) -> Vec<T> {
        let per = before.len() / rows;
        let mut out = before.to_vec();
        out[row * per..(row + 1) * per].clone_from_slice(&after[row * per..(row + 1) * per]);
        out
    }
    match (before, after) {
        (Buf::U8(b), Buf::U8(a)) => Buf::U8(splice(b, a, row, rows)),
        (Buf::U16(b), Buf::U16(a)) => Buf::U16(splice(b, a, row, rows)),
        (Buf::U32(b), Buf::U32(a)) => Buf::U32(splice(b, a, row, rows)),
        (Buf::F32(b), Buf::F32(a)) => Buf::F32(splice(b, a, row, rows)),
        _ => panic!("an output changes type"),
    }
}
