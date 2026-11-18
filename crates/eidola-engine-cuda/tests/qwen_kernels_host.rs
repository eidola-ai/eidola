//! The Qwen3.8 kernels' bodies, compiled for the host, against the model crate's
//! functions bit for bit (`qwen_common::cases`): every family through its grid function
//! and the body a GPU compiles, threads in descending order and shared memory starting as
//! garbage; the bodies arranged for speed against their reference forms over random and edge
//! inputs (the norm, the gated norm and the widening against their plain forms, the
//! quantiser against `quant_thread` over edge blocks and over every BF16 value under 96
//! block maxima, and its encoder against `ds::e4m3_encode` over every f32); and the fast
//! profile's small-N BF16 GEMM, both forms, against the order it documents
//! (`cases::small_gemm_model`) and against each other, batches and rows alone alike.
//! Skipped, saying why, without a C++ compiler.

mod qwen_common;

use eidola_engine_cuda::qwen::abi::{DnNormParams, NormParams, WidenParams};
use qwen_common::cases;

#[test]
fn qwen_kernel_bodies_equal_the_model_crate() {
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let dev = qwen_common::host_device(lib.clone());
    let reference = qwen_common::quant_thread_reference(lib.clone());
    let norm_ref = qwen_common::host_body::<NormParams>(lib.clone(), "qwen_host_norm_reference");
    let dn_norm_ref =
        qwen_common::host_body::<DnNormParams>(lib.clone(), "qwen_host_dn_norm_reference");
    let widen_ref = qwen_common::host_body::<WidenParams>(lib.clone(), "qwen_host_widen_reference");
    let mut failed = Vec::new();
    let mut outcomes: Vec<(&str, Result<String, String>)> = cases::all(&dev)
        .into_iter()
        .map(|(name, r)| (name, r.map(|()| String::new())))
        .collect();
    outcomes.push((
        "quant forms",
        cases::quant_forms(&dev, &reference, 12).map(|()| String::new()),
    ));
    outcomes.push(("quant sweep", cases::quant_sweep(&dev, &reference, 19)));
    let mismatches = qwen_common::e4m3_code_mismatches(&lib);
    outcomes.push((
        "e4m3 code",
        if mismatches.is_empty() {
            Ok("every f32".into())
        } else {
            Err(format!("(count, first pattern) per range: {mismatches:x?}"))
        },
    ));
    outcomes.push(("norm forms", cases::norm_forms(&dev, &norm_ref, 20)));
    outcomes.push((
        "dn norm forms",
        cases::dn_norm_forms(&dev, &dn_norm_ref, 21),
    ));
    outcomes.push(("widen forms", cases::widen_forms(&dev, &widen_ref, 22)));
    let small = qwen_common::small_gemm_host(lib.clone());
    let tiled = qwen_common::small_gemm_tiled_host(lib);
    outcomes.push(("fast small-N gemm", cases::small_gemm(&small, 14)));
    outcomes.push(("fast small-N gemm tiled", cases::small_gemm(&tiled, 14)));
    outcomes.push((
        "fast small-N gemm forms",
        cases::small_gemm_forms(&small, &tiled, 23),
    ));
    for (name, outcome) in outcomes {
        match outcome {
            Ok(report) => eprintln!("ok      {name} {report}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
