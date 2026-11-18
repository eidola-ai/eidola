//! The Qwen3.8 kernels on a GPU: every family against the model crate bit for bit on the
//! device's own image and on the `sm_100f` family image (the exact profile), the
//! quantiser's kernel against its reference form compiled for the host (codes and scales
//! bit for bit, edge blocks included), then the fast profile's FP8 linears and BF16 GEMM
//! (the DeltaNet's stacked `a` and `b`) against the exact ones (deviation recorded, and
//! bounded for the BF16 GEMM; a row's bits alone and in every batch asserted equal); every
//! FP8 GEMM the fast profile routes rows to giving every row the same bits
//! (`fast_fp8_configs_agree`); and the small-N BF16 GEMM against the order it documents.
//! Skipped, saying why, without a device or `EIDOLA_ENGINE_KERNELS_DIR`.

mod qwen_common;

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::QwenProfile;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use qwen_common::cases;

#[test]
fn qwen_kernels_on_the_device() {
    let Some(dir) = qwen_common::device_kernels("qwen_kernels_on_the_device") else {
        return;
    };
    let dir = KernelDir::new(dir);
    let exact = CudaDevice::open(0, &dir, QwenProfile::Exact).expect("the device");
    let mut failed = Vec::new();
    for (name, outcome) in cases::all(&exact) {
        match outcome {
            Ok(()) => eprintln!("ok      {name}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    match qwen_common::library() {
        Some(lib) => {
            let reference = qwen_common::quant_thread_reference(lib);
            match cases::quant_forms(&exact, &reference, 12) {
                Ok(()) => eprintln!("ok      quant forms"),
                Err(e) => {
                    eprintln!("FAILED  quant forms: {e}");
                    failed.push("quant forms");
                }
            }
        }
        None => {
            eprintln!("FAILED  quant forms: no C++ compiler for the reference form");
            failed.push("quant forms");
        }
    }
    let fast = CudaDevice::open(0, &dir, QwenProfile::Fast).expect("the device");
    match cases::fast_bf16_gemm(&exact, &fast, 13) {
        Ok(report) => {
            eprintln!("ok      fast_bf16_gemm: {report}");
            qwen_common::record("qwen_ops_fast_bf16_gemm", &report);
        }
        Err(e) => {
            eprintln!("FAILED  fast_bf16_gemm: {e}");
            failed.push("fast_bf16_gemm");
        }
    }
    match cases::fast_linears(&exact, &fast, 11) {
        Ok(report) => {
            eprintln!("ok      fast_linears: {report}");
            qwen_common::record("qwen_ops_fast_linears", &report);
        }
        Err(e) => {
            eprintln!("FAILED  fast_linears: {e}");
            failed.push("fast_linears");
        }
    }
    match fast_fp8_configs_agree(&fast, 15) {
        Ok(report) => {
            eprintln!("ok      fast_fp8_configs_agree: {report}");
            qwen_common::record("qwen_ops_fast_fp8_configs", &report);
        }
        Err(e) => {
            eprintln!("FAILED  fast_fp8_configs_agree: {e}");
            failed.push("fast_fp8_configs_agree");
        }
    }
    match fast_gates(&fast, 16) {
        Ok(report) => {
            eprintln!("ok      fast_gates: {report}");
            qwen_common::record("qwen_ops_fast_gates", &report);
        }
        Err(e) => {
            eprintln!("FAILED  fast_gates: {e}");
            failed.push("fast_gates");
        }
    }
    let small = cases::small_gemm_on(&fast);
    match cases::small_gemm(&small, 14) {
        Ok(report) => eprintln!("ok      fast small-N gemm: {report}"),
        Err(e) => {
            eprintln!("FAILED  fast small-N gemm: {e}");
            failed.push("fast small-N gemm");
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// Every FP8 GEMM the fast profile can route a row to (`fp8_kind`: the swapped kernel
/// at activation tiles 16, 32 and 64, the unswapped one above 64 rows) gives every row
/// the same bits: each kind over a 300-row batch, and over rows alone and in batches at
/// several offsets, against the unswapped kernel's 300-row batch; then the dispatch
/// itself at row counts either side of each boundary. This is what lets the executor
/// choose by row count.
fn fast_fp8_configs_agree(fast: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine_cuda::mimo::gemm::GemmKind;
    use eidola_engine_cuda::qwen::abi::{Lin8Params, QuantParams};
    use eidola_engine_cuda::qwen::cuda::FP8_KINDS;
    use eidola_engine_cuda::qwen::kernels::QwenKernels;
    use qwen_common::{Lcg, bits, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    for &(n, k) in &[(1024usize, 640usize), (384, 5120), (256, 128)] {
        let (codes, scales) = cases::fp8_weight(&mut g, n, k);
        let ts = cases::transpose_scales(&scales, n, k);
        let rows_max = 300usize;
        let x: Vec<f32> = (0..rows_max * k).map(|_| g.bf16(3.0)).collect();
        let (xb, wb, wsb) = (buf(fast, &bits(&x)), buf(fast, &codes), buf(fast, &ts));
        let cb = zeros(fast, rows_max * k);
        let sb = zeros(fast, (k / 128) * rows_max * 4);
        // Rows `first .. first + rows` through the quantiser and the GEMM of `kind` (the
        // dispatch's own choice with `None`).
        let run = |rows: usize, first: usize, kind: Option<GemmKind>| -> Result<Vec<u16>, String> {
            let m_pad = rows.next_multiple_of(4);
            let out = zeros(fast, m_pad * n * 2);
            let q = QuantParams {
                x: xb.addr + (first * k * 2) as u64,
                codes: cb.addr,
                scales: sb.addr,
                rows: u(rows),
                k: u(k),
                x_stride: u(k),
                m_pad: u(m_pad),
            };
            let l = Lin8Params {
                a: cb.addr,
                a_scales: sb.addr,
                w: wb.addr,
                w_scales: wsb.addr,
                out: out.addr,
                rows: u(rows),
                n: u(n),
                k: u(k),
                m_pad: u(m_pad),
                out_stride: u(n),
                reserved: 0,
            };
            // SAFETY: every buffer covers `m_pad` rows of this shape.
            unsafe {
                fast.quant(&q).map_err(|e| e.to_string())?;
                match kind {
                    Some(kind) => fast.lin8_on(&l, kind),
                    None => fast.lin8(&l),
                }
                .map_err(|e| e.to_string())?;
            }
            Ok(read::<CudaDevice, u16>(fast, &out, rows * n))
        };
        let want = run(rows_max, 0, Some(GemmKind::Fp8Blockwise))?;
        for kind in FP8_KINDS {
            let got = run(rows_max, 0, Some(kind))?;
            cases::same(&format!("{kind:?} {n}x{k}: 300 rows"), &got, &want)?;
            for &(rows, first) in &[
                (1usize, 0usize),
                (1, 77),
                (1, 299),
                (3, 5),
                (16, 100),
                (17, 31),
                (40, 200),
                (64, 1),
                (65, 150),
            ] {
                let got = run(rows, first, Some(kind))?;
                cases::same(
                    &format!("{kind:?} {n}x{k}: rows {first}..{}", first + rows),
                    &got,
                    &want[first * n..(first + rows) * n],
                )?;
                checked += 1;
            }
        }
        for rows in [1usize, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 257] {
            let first = (rows * 7) % (rows_max - rows + 1);
            let got = run(rows, first, None)?;
            cases::same(
                &format!("dispatch {n}x{k}: rows {first}..{}", first + rows),
                &got,
                &want[first * n..(first + rows) * n],
            )?;
            checked += 1;
        }
    }
    Ok(format!(
        "{checked} batches over {} kinds and the dispatch, every row's bits equal",
        FP8_KINDS.len()
    ))
}

/// The fast profile's gates (f32, the hardware exponential) against the exact ones on the
/// same rows: every output within two BF16 units (asserted; the histogram recorded), on
/// the 16-byte path (the MLP's product; the attention gate's interleaved units) and the
/// element path (a width that is not a multiple of 8); and each row's bits alone and in
/// batches equal (asserted).
fn fast_gates(fast: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine_cuda::qwen::abi::{GATE_SIGMOID, GATE_SWIGLU, GateParams};
    use eidola_engine_cuda::qwen::kernels::QwenKernels;
    use qwen_common::{Lcg, bits, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let mut report = Vec::new();
    // (rows, width, unit, g's unit stride, kind)
    for &(rows, width, unit, g_unit, kind) in &[
        (70usize, 2048usize, 2048usize, 2048usize, GATE_SWIGLU),
        (9, 768, 256, 512, GATE_SIGMOID),
        (7, 1003, 1003, 1003, GATE_SWIGLU),
        (5, 1003, 1003, 1003, GATE_SIGMOID),
    ] {
        let g_stride = (width / unit) * g_unit;
        let a: Vec<f32> = (0..rows * width).map(|_| g.bf16(6.0)).collect();
        let gs: Vec<f32> = (0..rows * g_stride).map(|_| g.bf16(12.0)).collect();
        let (ab, gb) = (buf(fast, &bits(&a)), buf(fast, &bits(&gs)));
        let params = |out: u64, rows: usize, first: usize| GateParams {
            a: ab.addr + (first * width * 2) as u64,
            g: gb.addr + (first * g_stride * 2) as u64,
            out,
            rows: u(rows),
            width: u(width),
            a_stride: u(width),
            g_stride: u(g_stride),
            out_stride: u(width),
            unit: u(unit),
            g_unit_stride: u(g_unit),
            kind,
        };
        let (oe, of) = (zeros(fast, rows * width * 2), zeros(fast, rows * width * 2));
        // SAFETY: the buffers hold what the structs name.
        unsafe {
            fast.gate_exact(&params(oe.addr, rows, 0))
                .map_err(|e| e.to_string())?;
            fast.gate(&params(of.addr, rows, 0))
                .map_err(|e| e.to_string())?;
        }
        let want = read::<CudaDevice, u16>(fast, &oe, rows * width);
        let batch = read::<CudaDevice, u16>(fast, &of, rows * width);
        let mut hist = [0usize; 4];
        for (i, (&e, &f)) in want.iter().zip(&batch).enumerate() {
            let d = (cases::ord16(e) - cases::ord16(f)).unsigned_abs() as usize;
            if d > 2 {
                return Err(format!(
                    "fast gate {width}/{unit} kind {kind}: element {i}: {f:#06x} against {e:#06x}"
                ));
            }
            hist[d.min(3)] += 1;
        }
        report.push(format!(
            "{width}/{unit} kind {kind}: ulp 0/1/2 = {:?}",
            &hist[..3]
        ));
        for &(n, first) in &[(1usize, 0usize), (1, rows - 1), (3, 2)] {
            let o = zeros(fast, n * width * 2);
            // SAFETY: as above.
            unsafe { fast.gate(&params(o.addr, n, first)) }.map_err(|e| e.to_string())?;
            cases::same(
                &format!("fast gate {width}/{unit}: rows {first}..{}", first + n),
                &read::<CudaDevice, u16>(fast, &o, n * width),
                &batch[first * width..(first + n) * width],
            )?;
        }
    }
    Ok(report.join("; "))
}
