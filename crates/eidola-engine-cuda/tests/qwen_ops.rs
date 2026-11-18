//! The Qwen3.8 kernels on a GPU: every family against the model crate bit for bit on the
//! device's own image and on the `sm_100f` family image (the exact profile), the
//! quantiser's kernel against its reference form compiled for the host (codes and scales
//! bit for bit, edge blocks included), then the fast profile's FP8 linears and BF16 GEMM
//! (the DeltaNet's stacked `a` and `b`) against the exact ones (deviation recorded, and
//! bounded for the BF16 GEMM; a row's bits alone and in every batch asserted equal); every
//! FP8 GEMM the fast profile routes rows to giving every row the same bits
//! (`fast_fp8_configs_agree`); the small-N BF16 GEMM, both its forms, against the order it
//! documents and against each other, and the dispatch past the tiled form's rows; the
//! kernels arranged for speed (the norm, the gated norm, the widening) against their reference
//! forms compiled for the host, and the quantiser against `quant_thread` over every BF16
//! value under 96 block maxima (`quant_sweep`); the blocked DeltaNet (`blocked_deltanet`);
//! and
//! Qwen's sampler and chain acceptance (`qwen_sampling`) against the serving core's
//! sampler, every probability's bits and every token (`qwen_sampler`, `qwen_accept`).
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
            use eidola_engine_cuda::qwen::abi::{DnNormParams, NormParams, WidenParams};
            let reference = qwen_common::quant_thread_reference(lib.clone());
            let norm_ref =
                qwen_common::host_body::<NormParams>(lib.clone(), "qwen_host_norm_reference");
            let dn_norm_ref =
                qwen_common::host_body::<DnNormParams>(lib.clone(), "qwen_host_dn_norm_reference");
            let widen_ref = qwen_common::host_body::<WidenParams>(lib, "qwen_host_widen_reference");
            for (name, outcome) in [
                (
                    "quant forms",
                    cases::quant_forms(&exact, &reference, 12).map(|()| String::new()),
                ),
                ("quant sweep", cases::quant_sweep(&exact, &reference, 19)),
                ("norm forms", cases::norm_forms(&exact, &norm_ref, 20)),
                (
                    "dn norm forms",
                    cases::dn_norm_forms(&exact, &dn_norm_ref, 21),
                ),
                ("widen forms", cases::widen_forms(&exact, &widen_ref, 22)),
            ] {
                match outcome {
                    Ok(report) => eprintln!("ok      {name} {report}"),
                    Err(e) => {
                        eprintln!("FAILED  {name}: {e}");
                        failed.push(name);
                    }
                }
            }
        }
        None => {
            eprintln!("FAILED  reference forms: no C++ compiler for the reference forms");
            failed.push("reference forms");
        }
    }
    match qwen_sampler(&exact, 17) {
        Ok(report) => {
            eprintln!("ok      qwen_sampler: {report}");
            qwen_common::record("qwen_ops_sampler", &report);
        }
        Err(e) => {
            eprintln!("FAILED  qwen_sampler: {e}");
            failed.push("qwen_sampler");
        }
    }
    match qwen_accept(&exact, 18) {
        Ok(report) => eprintln!("ok      qwen_accept: {report}"),
        Err(e) => {
            eprintln!("FAILED  qwen_accept: {e}");
            failed.push("qwen_accept");
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
    for (name, outcome) in [
        ("fast_fp8_split_forms", fast_fp8_split_forms(&fast, 25)),
        ("fast_fp8_split_order", fast_fp8_split_order(&fast, 26)),
        ("fast_fp8_stacks", fast_fp8_stacks(&fast, 27)),
    ] {
        match outcome {
            Ok(report) => {
                eprintln!("ok      {name}: {report}");
                qwen_common::record(&format!("qwen_ops_{name}"), &report);
            }
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
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
    let small_form = small_gemm_form_on(&fast, false);
    let tiled_form = small_gemm_form_on(&fast, true);
    for (name, outcome) in [
        ("fast small-N gemm", cases::small_gemm(&small, 14)),
        (
            "fast small-N gemm tiled",
            cases::small_gemm(&tiled_form, 14),
        ),
        (
            "fast small-N gemm forms",
            cases::small_gemm_forms(&small_form, &tiled_form, 23),
        ),
        (
            "fast small-N gemm dispatch",
            cases::small_gemm_forms(&small_form, &small, 24),
        ),
    ] {
        match outcome {
            Ok(report) => eprintln!("ok      {name}: {report}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    for (name, outcome) in blocked_deltanet(&fast) {
        match outcome {
            Ok(report) => {
                eprintln!("ok      {name}: {report}");
                qwen_common::record(&format!("qwen_ops_{}", name.replace(' ', "_")), &report);
            }
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}

/// The fast profile's blocked DeltaNet on the device (`qwen_common::chunk`): the intra
/// kernel's scratch bit for bit its body's compiled for the host; the whole form within
/// a bound of its definition written out (the tensor cores sum a K-step their own way;
/// the largest deviation recorded); its distance from the token recurrence (in BF16 units
/// from the exact profile's, and from f64), recorded; and reproducibility bit for bit: a
/// block's row the same with 1 to 16 of its positions present (served as the executor
/// serves it, the earlier ones from the tail cache and the later ones zeros in every
/// operand, against prefilled with them real), the run in chunks of 1 to 65 positions,
/// drafts accepted and rejected, and beside other runs; and the fused form the unfused
/// one's bits (1, 8 and 64 sequences a launch decoding with drafts, every fold: outputs,
/// states and the tail cache).
fn blocked_deltanet(fast: &CudaDevice) -> Vec<(&'static str, Result<String, String>)> {
    use qwen_common::chunk;
    let mut out = Vec::new();
    let intra = match qwen_common::library() {
        None => Err("no C++ compiler for the host body".to_string()),
        Some(lib) => {
            let host = qwen_common::host_device(lib.clone());
            let on_host = qwen_common::dn_intra_host(lib);
            let on_dev = |p: &eidola_engine_cuda::qwen::abi::DnChunkParams| {
                // SAFETY: the case's buffers hold what the struct names.
                unsafe { fast.dn_intra(p) }.map_err(|e| e.to_string())
            };
            chunk::intra_scratch(&host, &on_host, 31).and_then(|want| {
                let got = chunk::intra_scratch(fast, &on_dev, 31)?;
                for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                    cases::same(&format!("intra scratch array {i}"), g, w)?;
                }
                Ok(format!("{} arrays bit for bit", got.len()))
            })
        }
    };
    out.push(("blocked deltanet intra", intra));
    out.push((
        "blocked deltanet against its model",
        chunk::against_model(fast, 21, false, 1e-4),
    ));
    out.push((
        "blocked deltanet against the recurrence",
        chunk::against_recurrence(fast, 22, 2048, 1e-3),
    ));
    out.push((
        "blocked deltanet reproducible",
        chunk::invariant(fast, 23, 70),
    ));
    out.push((
        "blocked deltanet fused form",
        chunk::fused_equals_unfused(fast, 24, &[1, 8, 64]),
    ));
    out
}

/// [`cases::small_gemm`]'s runner on the small-N GEMM in the form named, whatever the
/// row count (`CudaDevice::gemm_bf16_form`).
fn small_gemm_form_on(fast: &CudaDevice, tiled: bool) -> impl cases::SmallGemmRun + '_ {
    use eidola_engine_cuda::qwen::abi::LinearParams;
    use qwen_common::{buf, read, zeros};
    let u = |x: usize| u32::try_from(x).expect("small");
    move |x, w, rows, first, n, k| {
        let (xb, wb) = (buf(fast, x), buf(fast, w));
        let out = zeros(fast, rows * n * 4);
        let p = LinearParams {
            x: xb.addr + (first * k * 2) as u64,
            w: wb.addr,
            out: out.addr,
            rows: u(rows),
            n: u(n),
            k: u(k),
            x_stride: u(k),
            out_stride: u(n),
            out_f32: 1,
        };
        // SAFETY: the buffers hold what the struct names.
        unsafe { fast.gemm_bf16_form(&p, tiled) }.map_err(|e| e.to_string())?;
        Ok(read::<CudaDevice, f32>(fast, &out, rows * n))
    }
}

/// Every FP8 GEMM the fast profile can route a row to (`fp8_kind`: the swapped kernel
/// at activation tiles 16, 32 and 64, the unswapped one above 64 rows) gives every row
/// the same bits: each kind over a 300-row batch, and over rows alone and in batches at
/// several offsets, against the unswapped kernel's 300-row batch; then the dispatch
/// itself at row counts either side of each boundary. This is what lets the executor
/// choose by row count. Its shapes are ones `fp8_splits` leaves whole (asserted), the
/// unsplit kinds' domain; the split shapes' instances are `fast_fp8_split_forms`' and
/// `fast_fp8_split_order`'s.
fn fast_fp8_configs_agree(fast: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine_cuda::mimo::gemm::GemmKind;
    use eidola_engine_cuda::qwen::abi::{Lin8Params, QuantParams};
    use eidola_engine_cuda::qwen::cuda::FP8_KINDS;
    use eidola_engine_cuda::qwen::kernels::QwenKernels;
    use qwen_common::{Lcg, bits, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    // K of 5,120 at 10,240 outputs: 80 tiles, too many to split.
    for &(n, k) in &[(1024usize, 640usize), (10240, 5120), (256, 128)] {
        if eidola_engine_cuda::qwen::kernels::fp8_splits(u(n), u(k)) != 1 {
            return Err(format!("{n}x{k} splits: not an unsplit kind's shape"));
        }
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
                    Some(kind) => fast.lin8_on(&l, kind, None),
                    None => fast.lin8(&l, None),
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

/// One FP8 weight and activation rows on the device, quantised by the device's own
/// quantiser into codes and scales for any run of the rows.
struct Fp8Case {
    n: usize,
    k: usize,
    x: qwen_common::Buf,
    w: qwen_common::Buf,
    ws: qwen_common::Buf,
    codes: qwen_common::Buf,
    scales: qwen_common::Buf,
}

impl Fp8Case {
    fn new(
        fast: &CudaDevice,
        g: &mut qwen_common::Lcg,
        n: usize,
        k: usize,
        rows_max: usize,
    ) -> Self {
        use qwen_common::{bits, buf, zeros};
        let (codes, scales) = cases::fp8_weight(g, n, k);
        let ts = cases::transpose_scales(&scales, n, k);
        let x: Vec<f32> = (0..rows_max * k).map(|_| g.bf16(3.0)).collect();
        Fp8Case {
            n,
            k,
            x: buf(fast, &bits(&x)),
            w: buf(fast, &codes),
            ws: buf(fast, &ts),
            codes: zeros(fast, rows_max.next_multiple_of(4) * k),
            scales: zeros(fast, (k / 128) * rows_max.next_multiple_of(4) * 4),
        }
    }

    /// Rows `first .. first + rows` quantised, and the linear's parameters over them into
    /// `out`.
    fn quant(
        &self,
        fast: &CudaDevice,
        rows: usize,
        first: usize,
        out: u64,
    ) -> Result<eidola_engine_cuda::qwen::abi::Lin8Params, String> {
        use eidola_engine_cuda::qwen::abi::{Lin8Params, QuantParams};
        use eidola_engine_cuda::qwen::kernels::QwenKernels;
        let u = |x: usize| u32::try_from(x).expect("small");
        let m_pad = rows.next_multiple_of(4);
        let q = QuantParams {
            x: self.x.addr + (first * self.k * 2) as u64,
            codes: self.codes.addr,
            scales: self.scales.addr,
            rows: u(rows),
            k: u(self.k),
            x_stride: u(self.k),
            m_pad: u(m_pad),
        };
        // SAFETY: the buffers hold `rows_max` rows of this shape.
        unsafe { fast.quant(&q) }.map_err(|e| e.to_string())?;
        Ok(Lin8Params {
            a: self.codes.addr,
            a_scales: self.scales.addr,
            w: self.w.addr,
            w_scales: self.ws.addr,
            out,
            rows: u(rows),
            n: u(self.n),
            k: u(self.k),
            m_pad: u(m_pad),
            out_stride: u(self.n),
            reserved: 0,
        })
    }
}

/// The split FP8 GEMM (`cutlass_fp8_blockwise_split`, its partials added by
/// `eidola_qwen_fast_split_sum`) gives every row the same bits in every instance a row
/// can be routed to: each instance over a 300-row batch, and over rows alone and in
/// batches at several offsets, against the unswapped instance's batch; then the dispatch
/// (`fp8_kind`) at row counts either side of each boundary, which takes the split form
/// at every row count for these shapes. The released model's `o_proj` / `out_proj` and
/// `down_proj` shapes are among them.
fn fast_fp8_split_forms(fast: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine_cuda::mimo::gemm::{GemmKind, SplitTile};
    use eidola_engine_cuda::qwen::kernels::{QwenKernels, fp8_splits};
    use qwen_common::{Lcg, read, zeros};

    let mut g = Lcg(seed);
    let mut checked = 0usize;
    for &(n, k) in &[
        (5120usize, 6144usize),
        (5120, 17408),
        (1024, 3072),
        (640, 2048),
    ] {
        let u = |x: usize| u32::try_from(x).expect("small");
        let splits = fp8_splits(u(n), u(k));
        if splits < 2 {
            return Err(format!("{n}x{k} does not split"));
        }
        let c = Fp8Case::new(fast, &mut g, n, k, 300);
        let parts = zeros(fast, splits as usize * 300usize.next_multiple_of(4) * n * 4);
        let run = |rows: usize, first: usize, kind: Option<GemmKind>| -> Result<Vec<u16>, String> {
            let out = zeros(fast, rows.next_multiple_of(4) * n * 2);
            let l = c.quant(fast, rows, first, out.addr)?;
            // SAFETY: every buffer covers `m_pad` rows of this shape.
            unsafe {
                match kind {
                    Some(kind) => fast.lin8_on(&l, kind, Some(parts.addr)),
                    None => fast.lin8(&l, Some(parts.addr)),
                }
            }
            .map_err(|e| e.to_string())?;
            Ok(read::<CudaDevice, u16>(fast, &out, rows * n))
        };
        let want = run(300, 0, Some(GemmKind::Fp8Split(SplitTile::Wide, splits)))?;
        for tile in SplitTile::ALL {
            let kind = GemmKind::Fp8Split(tile, splits);
            let got = run(300, 0, Some(kind))?;
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
            let first = (rows * 7) % (300 - rows + 1);
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
        "{checked} batches over {} split instances and the dispatch, every row's bits equal",
        SplitTile::ALL.len()
    ))
}

/// The split's arithmetic as documented: range `s` of a split launch is that range's own
/// GEMM from zero (the same instance over the range's columns and scale blocks copied
/// into a dense problem, unsplit), bit for bit; and the sum is the ranges' values added
/// in order from the first, rounded once to BF16 (computed here from the partials the
/// device left), bit for bit. A sum in any other order, or a range's sums carried into
/// the next, fails it.
fn fast_fp8_split_order(fast: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine_cuda::mimo::gemm::{GemmKind, SplitTile};
    use eidola_engine_cuda::qwen::abi::{Lin8Params, SplitSumParams};
    use eidola_engine_cuda::qwen::kernels::split_sum_grid;
    use qwen_common::{Lcg, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    for &(n, k, splits, rows) in &[
        (1024usize, 3072usize, 3usize, 37usize),
        (5120, 6144, 3, 1),
        (5120, 17408, 2, 64),
        (256, 2048, 4, 130),
    ] {
        let c = Fp8Case::new(fast, &mut g, n, k, rows);
        let m_pad = rows.next_multiple_of(4);
        let out = zeros(fast, m_pad * n * 2);
        let parts = zeros(fast, splits * m_pad * n * 4);
        let l = c.quant(fast, rows, 0, out.addr)?;
        let codes = read::<CudaDevice, u8>(fast, &c.codes, m_pad * k);
        let scales = read::<CudaDevice, f32>(fast, &c.scales, (k / 128) * m_pad);
        let w = read::<CudaDevice, u8>(fast, &c.w, n * k);
        let ws = read::<CudaDevice, f32>(fast, &c.ws, (k / 128) * (n / 128));
        let (ks, kbs, nb) = (k / splits, k / 128 / splits, n / 128);
        for tile in SplitTile::ALL {
            if tile == SplitTile::Wide && rows < 65 {
                continue;
            }
            // SAFETY: the buffers hold `m_pad` rows of this shape and `splits` arrays.
            unsafe { fast.fp8_split_parts(&l, tile, u(splits), parts.addr) }
                .map_err(|e| e.to_string())?;
            let p = read::<CudaDevice, f32>(fast, &parts, splits * m_pad * n);
            for s in 0..splits {
                // Range `s` as a dense problem of its own.
                let a: Vec<u8> = (0..m_pad)
                    .flat_map(|r| codes[r * k + s * ks..r * k + (s + 1) * ks].to_vec())
                    .collect();
                let sa = scales[s * kbs * m_pad..(s + 1) * kbs * m_pad].to_vec();
                let b: Vec<u8> = (0..n)
                    .flat_map(|o| w[o * k + s * ks..o * k + (s + 1) * ks].to_vec())
                    .collect();
                let sb = ws[s * kbs * nb..(s + 1) * kbs * nb].to_vec();
                let (ab, sab, bb, sbb) =
                    (buf(fast, &a), buf(fast, &sa), buf(fast, &b), buf(fast, &sb));
                let one = zeros(fast, m_pad * n * 4);
                // The partials alone: `out` is checked, not written.
                let lr = Lin8Params {
                    a: ab.addr,
                    a_scales: sab.addr,
                    w: bb.addr,
                    w_scales: sbb.addr,
                    out: one.addr,
                    rows: u(rows),
                    n: u(n),
                    k: u(ks),
                    m_pad: u(m_pad),
                    out_stride: u(n),
                    reserved: 0,
                };
                // SAFETY: dense buffers of the range's problem.
                unsafe { fast.fp8_split_parts(&lr, tile, 1, one.addr) }
                    .map_err(|e| e.to_string())?;
                let alone = read::<CudaDevice, f32>(fast, &one, rows * n);
                let got: Vec<u32> = p[s * m_pad * n..s * m_pad * n + rows * n]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                let want: Vec<u32> = alone.iter().map(|v| v.to_bits()).collect();
                cases::same(
                    &format!("{tile:?} {n}x{k}/{splits}: range {s} against its own GEMM"),
                    &got,
                    &want,
                )?;
            }
            let sp = SplitSumParams {
                parts: parts.addr,
                out: out.addr,
                rows: u(rows),
                n: u(n),
                splits: u(splits),
                part_rows: u(m_pad),
                out_stride: u(n),
                reserved: 0,
            };
            let grid = split_sum_grid(&sp).map_err(|e| e.to_string())?;
            // SAFETY: the partials and the output just sized.
            unsafe { fast.split_sum(&sp, grid) }.map_err(|e| e.to_string())?;
            let got = read::<CudaDevice, u16>(fast, &out, rows * n);
            let want: Vec<u16> = (0..rows * n)
                .map(|i| {
                    let mut acc = p[i];
                    for s in 1..splits {
                        acc += p[s * m_pad * n + i];
                    }
                    bf16_nearest(acc)
                })
                .collect();
            cases::same(&format!("{tile:?} {n}x{k}/{splits}: the sum"), &got, &want)?;
            // The dispatch's form is the same bits.
            let o2 = zeros(fast, m_pad * n * 2);
            let l2 = Lin8Params { out: o2.addr, ..l };
            // SAFETY: as above.
            unsafe { fast.lin8_on(&l2, GemmKind::Fp8Split(tile, u(splits)), Some(parts.addr)) }
                .map_err(|e| e.to_string())?;
            cases::same(
                &format!("{tile:?} {n}x{k}/{splits}: linear against parts and sum"),
                &read::<CudaDevice, u16>(fast, &o2, rows * n),
                &want,
            )?;
            checked += 1;
        }
    }
    Ok(format!(
        "{checked} launches: each range its own GEMM, the sum in range order, bit for bit"
    ))
}

/// The nearest BF16 of `x`, ties to even, a NaN as the one quiet NaN.
fn bf16_nearest(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7fc0;
    }
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// A stack of FP8 weights sharing an input (the executor's `qkvz`, `qkv`, `gate_up`) gives
/// each part's outputs the bits the part gives alone on the stack's own instance, at decode
/// and prefill row counts (the swapped kernel at every tile, the unswapped one): an
/// output's bits depend on its weight row, not on where the stack puts it. (A part's shape
/// alone may split, as `1024 × 5120` does; the executor runs only the stack.)
fn fast_fp8_stacks(fast: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine_cuda::qwen::abi::Lin8Params;
    use eidola_engine_cuda::qwen::kernels::{MAX_SPLITS, QwenKernels, fp8_splits};
    use qwen_common::{Lcg, bits, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let mut checked = 0usize;
    let k = 5120usize;
    // The executor's stacks are all wider than a split allows; this one too (80 tiles),
    // while its 1,024-row parts alone would split (asserted).
    let parts_n = [8192usize, 1024, 1024];
    let n: usize = parts_n.iter().sum();
    if fp8_splits(u(n), u(k)) != 1 || fp8_splits(1024, u(k)) == 1 {
        return Err(format!(
            "{n}x{k} must stay whole and its 1024-row parts split"
        ));
    }
    let weights: Vec<(Vec<u8>, Vec<f32>)> = parts_n
        .iter()
        .map(|&p| cases::fp8_weight(&mut g, p, k))
        .collect();
    let mut codes = Vec::new();
    for (c, _) in &weights {
        codes.extend_from_slice(c);
    }
    // The stack's scales `[k / 128][n / 128]`, each part's blocks in turn.
    let (nb, kb) = (n / 128, k / 128);
    let mut ts = vec![0.0f32; nb * kb];
    let mut first_block = 0;
    for (&pn, (_, sc)) in parts_n.iter().zip(&weights) {
        for i in 0..pn / 128 {
            for j in 0..kb {
                ts[j * nb + first_block + i] = sc[i * kb + j];
            }
        }
        first_block += pn / 128;
    }
    let rows_max = 300usize;
    let x: Vec<f32> = (0..rows_max * k).map(|_| g.bf16(3.0)).collect();
    let xb = buf(fast, &bits(&x));
    let cb = zeros(fast, rows_max.next_multiple_of(4) * k);
    let sb = zeros(fast, kb * rows_max.next_multiple_of(4) * 4);
    let pb = zeros(
        fast,
        MAX_SPLITS as usize * rows_max.next_multiple_of(4) * n * 4,
    );
    let (wb, wsb) = (buf(fast, &codes), buf(fast, &ts));
    let part_bufs: Vec<_> = parts_n
        .iter()
        .zip(&weights)
        .map(|(&pn, (c, sc))| (buf(fast, c), buf(fast, &cases::transpose_scales(sc, pn, k))))
        .collect();
    for rows in [1usize, 16, 33, 64, 65, 300] {
        let m_pad = rows.next_multiple_of(4);
        let q = eidola_engine_cuda::qwen::abi::QuantParams {
            x: xb.addr,
            codes: cb.addr,
            scales: sb.addr,
            rows: u(rows),
            k: u(k),
            x_stride: u(k),
            m_pad: u(m_pad),
        };
        let lin = |w: u64, ws: u64, n: usize, out: u64| Lin8Params {
            a: cb.addr,
            a_scales: sb.addr,
            w,
            w_scales: ws,
            out,
            rows: u(rows),
            n: u(n),
            k: u(k),
            m_pad: u(m_pad),
            out_stride: u(n),
            reserved: 0,
        };
        let out = zeros(fast, m_pad * n * 2);
        // SAFETY: every buffer covers `m_pad` rows of these shapes.
        unsafe {
            fast.quant(&q).map_err(|e| e.to_string())?;
            fast.lin8(&lin(wb.addr, wsb.addr, n, out.addr), Some(pb.addr))
                .map_err(|e| e.to_string())?;
        }
        let stacked = read::<CudaDevice, u16>(fast, &out, rows * n);
        let mut col = 0;
        for (&pn, (w, ws)) in parts_n.iter().zip(&part_bufs) {
            let o = zeros(fast, m_pad * pn * 2);
            // SAFETY: as above.
            unsafe {
                fast.lin8_on(
                    &lin(w.addr, ws.addr, pn, o.addr),
                    eidola_engine_cuda::qwen::cuda::fp8_kind(u(m_pad), u(n), u(k)),
                    Some(pb.addr),
                )
            }
            .map_err(|e| e.to_string())?;
            let alone = read::<CudaDevice, u16>(fast, &o, rows * pn);
            let want: Vec<u16> = (0..rows)
                .flat_map(|r| stacked[r * n + col..r * n + col + pn].to_vec())
                .collect();
            cases::same(
                &format!("stack {n}x{k}: part of {pn} at {col}, {rows} rows"),
                &alone,
                &want,
            )?;
            col += pn;
            checked += 1;
        }
    }
    Ok(format!(
        "{checked} parts against their stack, every output's bits equal"
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

/// The released head's rows and its tokenizer's sampleable ids.
const QWEN_VOCAB: usize = 248_320;
const QWEN_SAMPLEABLE: usize = 248_077;

/// Logit rows of eight kinds over `n` ids (the rest of `stride` padding that would win
/// were it sampleable): smooth, peaked, on a coarse grid, masked, wide, a third of them
/// tied at the top, decaying like a language model's, and a few live among underflowing.
fn sampler_logits(g: &mut qwen_common::Lcg, n: usize, stride: usize) -> Vec<f32> {
    let mut out = vec![0f32; 8 * stride];
    for kind in 0..8 {
        let row = &mut out[kind * stride..(kind + 1) * stride];
        for (i, v) in row[..n].iter_mut().enumerate() {
            let x = g.f32().abs();
            *v = match kind {
                0 => x * 4.0,
                1 => x * 2.0 + if i % 997 == 17 { 12.0 } else { 0.0 },
                2 => (x * 6.0).round() * 0.5,
                3 => {
                    if g.below(7) == 0 {
                        f32::NEG_INFINITY
                    } else {
                        x * 8.0
                    }
                }
                4 => x * 20.0,
                5 => {
                    if i % 3 == 0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                6 => -(i as f32) * 0.37 + x * 0.01,
                _ => {
                    if g.below(50) == 0 {
                        x * 30.0
                    } else {
                        -1000.0
                    }
                }
            };
        }
        row[n..].fill(100.0);
    }
    out
}

/// Every selection path's parameters: greedy; temperature alone; top-p alone (its
/// candidates from the histogram's bound, and past the capacity); top-k alone (1, 20,
/// 50, at and past the capacity of 2,048, past the vocabulary); top-k with top-p; min-p
/// alone and with both; temperatures from 0.01 to 2.
fn sampler_params() -> Vec<eidola_engine::sampling::SamplingParams> {
    use eidola_engine::sampling::SamplingParams;
    let p = |t, k, tp, mp, seed| SamplingParams::new(t, k, tp, mp, seed).expect("valid");
    vec![
        SamplingParams::greedy(),
        p(1.0, 0, 1.0, 0.0, 1),
        p(1.0, 0, 0.95, 0.0, 2),
        p(0.7, 0, 0.9, 0.0, 3),
        p(1.0, 50, 1.0, 0.0, 4),
        p(0.8, 40, 0.9, 0.05, 5),
        p(1.0, 0, 1.0, 0.1, 6),
        p(1.3, 0, 0.5, 0.0, 7),
        p(1.0, 1, 1.0, 0.0, 8),
        p(2.0, 200_000, 0.99, 0.0, 9),
        p(0.05, 0, 1.0, 0.0, 10),
        p(1.0, 20, 0.95, 0.0, 11),
        p(0.6, 20, 0.95, 0.0, 12),
        p(1.0, 20, 1.0, 0.0, 13),
        p(1.0, 2047, 1.0, 0.0, 14),
        p(1.0, 2049, 0.8, 0.0, 15),
        p(1.0, 5, 0.3, 0.5, 16),
        p(1.7, 0, 0.99, 0.0, 17),
        p(1.0, 4000, 1.0, 0.0, 18),
        p(1.0, 400_000, 1.0, 0.0, 19),
        p(1.0, 0, 1.0, 0.9, 20),
        p(0.01, 20, 0.95, 0.0, 21),
    ]
}

/// Qwen's sampler against `eidola_engine::sampling`: every row's processed distribution
/// bit for bit and its token, for every parameter set of [`sampler_params`] over every
/// logit kind of [`sampler_logits`], all in one launch, at the released vocabulary and at
/// 9,000 ids (where thousands tie), with the plain stream, the draft stream and no draw;
/// then 512 draws of one row at consecutive positions.
fn qwen_sampler(dev: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine::sampling::{self, Logits, Stream};
    use eidola_engine_cuda::qwen::kernels::{QwenKernels, SampleArgs};
    use eidola_engine_cuda::qwen::sampler::sample_work_bytes;
    use eidola_engine_cuda::sampler::SampleRow;
    use qwen_common::{Lcg, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let params = sampler_params();
    let mut checked = 0usize;
    let mut report = Vec::new();
    for &(n, stride) in &[(QWEN_SAMPLEABLE, QWEN_VOCAB), (9000usize, 9216usize)] {
        let logits = sampler_logits(&mut g, n, stride);
        let lb = buf(dev, &logits);
        let mut rows = Vec::new();
        for (pi, p) in params.iter().enumerate() {
            for kind in 0..8 {
                rows.push((SampleRow::new(p, u(pi * 31 + kind * 7 + 1), u(kind)), *p));
            }
        }
        let hrows: Vec<SampleRow> = rows.iter().map(|r| r.0).collect();
        let rb = buf(dev, &hrows);
        let want: Vec<Vec<f64>> = rows
            .iter()
            .map(|(r, p)| {
                let lr = r.logit_row as usize;
                sampling::processed_probs(
                    Logits::new(&logits[lr * stride..(lr + 1) * stride], u(n)),
                    p,
                )
            })
            .collect();
        for draw in [Some(Stream::Sample), Some(Stream::Draft), None] {
            let probs = zeros(dev, rows.len() * n * 8);
            let tokens = zeros(dev, rows.len() * 4);
            let status = zeros(dev, 4);
            let work = zeros(dev, sample_work_bytes(rows.len()));
            let a = SampleArgs {
                logits: lb.addr,
                logits_stride: stride as u64,
                n: u(n),
                rows: rb.addr,
                num_rows: u(rows.len()),
                draw,
                probs: probs.addr,
                tokens: if draw.is_some() { tokens.addr } else { 0 },
                status: status.addr,
                work: work.addr,
            };
            let t0 = std::time::Instant::now();
            // SAFETY: the buffers hold what the arguments name.
            unsafe { dev.sample(&a) }.map_err(|e| e.to_string())?;
            let got = read::<CudaDevice, f64>(dev, &probs, rows.len() * n);
            let elapsed = t0.elapsed();
            let got_tokens = read::<CudaDevice, u32>(dev, &tokens, rows.len());
            if read::<CudaDevice, u32>(dev, &status, 1)[0] != 0 {
                return Err(format!("{n} ids, {draw:?}: status raised"));
            }
            for (r, (row, p)) in rows.iter().enumerate() {
                let w = &want[r];
                let gr = &got[r * n..(r + 1) * n];
                if let Some(i) = w
                    .iter()
                    .zip(gr)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(format!(
                        "{n} ids, row {r} ({p:?}, logits {}): p[{i}] {:e}, want {:e}",
                        row.logit_row, gr[i], w[i]
                    ));
                }
                if let Some(stream) = draw {
                    let lr = row.logit_row as usize;
                    let l = Logits::new(&logits[lr * stride..(lr + 1) * stride], u(n));
                    let want_token = if p.is_greedy() {
                        sampling::argmax(l)
                    } else {
                        sampling::sample_from(
                            w,
                            sampling::uniform(p.seed(), u64::from(row.position), stream),
                        )
                    };
                    if got_tokens[r] != want_token {
                        return Err(format!(
                            "{n} ids, row {r} ({p:?}, logits {lr}, {stream:?}): token {}, want {want_token}",
                            got_tokens[r]
                        ));
                    }
                }
                checked += 1;
            }
            report.push(format!(
                "{n} ids {draw:?}: {} rows in {elapsed:?}",
                rows.len()
            ));
        }
    }
    // Many draws of one row (an LLM-like decay, top-p 0.9 then top-k 20 with top-p 0.95).
    let n = QWEN_SAMPLEABLE;
    let logits = sampler_logits(&mut g, n, QWEN_VOCAB);
    let lb = buf(dev, &logits);
    for p in [
        eidola_engine::sampling::SamplingParams::new(1.0, 0, 0.9, 0.0, 1234).expect("valid"),
        eidola_engine::sampling::SamplingParams::new(1.0, 20, 0.95, 0.0, 99).expect("valid"),
    ] {
        let hrows: Vec<SampleRow> = (0..512).map(|i| SampleRow::new(&p, i, 6)).collect();
        let rb = buf(dev, &hrows);
        let probs = zeros(dev, hrows.len() * n * 8);
        let tokens = zeros(dev, hrows.len() * 4);
        let status = zeros(dev, 4);
        let work = zeros(dev, sample_work_bytes(hrows.len()));
        let a = SampleArgs {
            logits: lb.addr,
            logits_stride: QWEN_VOCAB as u64,
            n: u(n),
            rows: rb.addr,
            num_rows: u(hrows.len()),
            draw: Some(Stream::Sample),
            probs: probs.addr,
            tokens: tokens.addr,
            status: status.addr,
            work: work.addr,
        };
        // SAFETY: as above.
        unsafe { dev.sample(&a) }.map_err(|e| e.to_string())?;
        let got = read::<CudaDevice, u32>(dev, &tokens, hrows.len());
        let w = sampling::processed_probs(
            Logits::new(&logits[6 * QWEN_VOCAB..7 * QWEN_VOCAB], u(n)),
            &p,
        );
        for (i, row) in hrows.iter().enumerate() {
            let want = sampling::sample_from(
                &w,
                sampling::uniform(p.seed(), u64::from(row.position), Stream::Sample),
            );
            if got[i] != want {
                return Err(format!(
                    "{p:?} at position {}: {} want {want}",
                    row.position, got[i]
                ));
            }
            checked += 1;
        }
    }
    Ok(format!("{checked} rows bit for bit; {}", report.join("; ")))
}

/// Qwen's chain acceptance against `sampling::chain_accept` at the released vocabulary:
/// 60 sequences of 1 to 3 drafts (greedy, plain, top-p, top-k with min-p and with top-p),
/// drafts drawn from the drafter's distributions, laid out as the program lays them (one
/// depth's drafts of every sequence together), with acceptance neither certain nor rare;
/// every sequence's tokens equal; and a draft id past the vocabulary ending its sequence
/// with no tokens and the status bit.
fn qwen_accept(dev: &CudaDevice, seed: u64) -> Result<String, String> {
    use eidola_engine::sampling::{self, Logits, SamplingParams, Stream};
    use eidola_engine_cuda::qwen::kernels::{AcceptLaunch, QwenKernels};
    use eidola_engine_cuda::qwen::sampler::accept_work_bytes;
    use eidola_engine_cuda::sampler::{STATUS_BAD_TOKEN, SampleRow};
    use qwen_common::{Lcg, buf, read, zeros};

    let u = |x: usize| u32::try_from(x).expect("small");
    let mut g = Lcg(seed);
    let n = QWEN_SAMPLEABLE;
    let width = 3usize;
    let params = [
        SamplingParams::greedy(),
        SamplingParams::random(1.0, 5).expect("valid"),
        SamplingParams::new(0.8, 0, 0.9, 0.0, 6).expect("valid"),
        SamplingParams::new(1.0, 20, 1.0, 0.02, 7).expect("valid"),
        SamplingParams::new(1.0, 20, 0.95, 0.0, 8).expect("valid"),
    ];
    let seqs = 60usize;
    // Target rows: k + 1 a sequence, consecutive; draft rows: depth-major, `seqs` a depth.
    let mut target = Vec::new();
    let mut draft = vec![0f64; width * seqs * n];
    let mut drafts = vec![0u32; width * seqs];
    let (mut rows, mut trow, mut drow, mut nd, mut want) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..seqs {
        let k = 1 + i % width;
        let p = params[i % params.len()];
        let pos = u64::from(u(i * 13 + 1));
        let mut ps = Vec::new();
        let mut tl = Vec::new();
        trow.push(u(target.len() / n));
        for _ in 0..=k {
            let l: Vec<f32> = (0..n).map(|_| g.f32().abs() * 3.0).collect();
            let pr = sampling::processed_probs(Logits::new(&l, u(n)), &p);
            target.extend_from_slice(&pr);
            ps.push(pr);
            tl.push(l);
        }
        let mut qs = Vec::new();
        let mut ids = Vec::new();
        for (j, t) in tl.iter().take(k).enumerate() {
            let noisy: Vec<f32> = t
                .iter()
                .map(|&v| {
                    if i % 4 == 3 {
                        v
                    } else {
                        v + g.f32().abs() * 0.7
                    }
                })
                .collect();
            let l = Logits::new(&noisy, u(n));
            let q = sampling::processed_probs(l, &p);
            let d = if p.is_greedy() {
                sampling::argmax(l)
            } else {
                sampling::sample_from(
                    &q,
                    sampling::uniform(p.seed(), pos + j as u64, Stream::Draft),
                )
            };
            let at = j * seqs + i;
            draft[at * n..(at + 1) * n].copy_from_slice(&q);
            drafts[at] = d;
            ids.push(d);
            qs.push(q);
        }
        want.push(sampling::chain_accept(&ps, &qs, &ids, &p, pos));
        rows.push(SampleRow::new(
            &p,
            u(usize::try_from(pos).expect("small")),
            0,
        ));
        drow.push(u(i));
        nd.push(u(k));
    }
    // The last sequence's first draft past the vocabulary.
    drafts[seqs - 1] = u(n);
    let (tb, db) = (buf(dev, &target), buf(dev, &draft));
    let (rb, trb, drb, ndb, idb) = (
        buf(dev, &rows),
        buf(dev, &trow),
        buf(dev, &drow),
        buf(dev, &nd),
        buf(dev, &drafts),
    );
    let stride = width + 1;
    let out = zeros(dev, seqs * stride * 4);
    let counts = zeros(dev, seqs * 4);
    let status = zeros(dev, 4);
    let work = zeros(dev, accept_work_bytes(seqs));
    let a = AcceptLaunch {
        target: tb.addr,
        draft: db.addr,
        n: u(n),
        rows: rb.addr,
        target_row: trb.addr,
        draft_row: drb.addr,
        draft_step: u(seqs),
        num_drafts: ndb.addr,
        drafts: idb.addr,
        stride: u(stride),
        scratch: work.addr,
        out: out.addr,
        counts: counts.addr,
        status: status.addr,
        num_rows: u(seqs),
    };
    // SAFETY: the buffers hold what the launch names.
    unsafe { dev.accept(&a) }.map_err(|e| e.to_string())?;
    let got = read::<CudaDevice, u32>(dev, &out, seqs * stride);
    let got_counts = read::<CudaDevice, u32>(dev, &counts, seqs);
    let mut rejected = 0;
    for i in 0..seqs - 1 {
        let c = got_counts[i] as usize;
        let tokens = &got[i * stride..i * stride + c.min(stride)];
        if tokens != want[i].as_slice() {
            return Err(format!(
                "sequence {i} ({:?}): {tokens:?}, want {:?}",
                params[i % params.len()],
                want[i]
            ));
        }
        rejected += usize::from(want[i].len() <= nd[i] as usize);
    }
    if got_counts[seqs - 1] != 0 || read::<CudaDevice, u32>(dev, &status, 1)[0] != STATUS_BAD_TOKEN
    {
        return Err("a draft past the vocabulary was not refused".into());
    }
    Ok(format!(
        "{} sequences equal ({rejected} with a rejection), a bad draft refused",
        seqs - 1
    ))
}
