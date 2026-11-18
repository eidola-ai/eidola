//! The Qwen3.8 kernels' bodies, compiled for the host, against the model crate's
//! functions bit for bit (`qwen_common::cases`): every family through its grid function
//! and the body a GPU compiles, threads in descending order and shared memory starting as
//! garbage; the quantiser's warp form against its reference form over edge blocks; and
//! the fast profile's small-N BF16 GEMM against the order it documents
//! (`cases::small_gemm_model`), batches and rows alone alike.
//! Skipped, saying why, without a C++ compiler.

mod qwen_common;

use qwen_common::cases;

#[test]
fn qwen_kernel_bodies_equal_the_model_crate() {
    let Some(lib) = qwen_common::library() else {
        return;
    };
    let dev = qwen_common::host_device(lib.clone());
    let reference = qwen_common::quant_thread_reference(lib.clone());
    let mut failed = Vec::new();
    let mut outcomes = cases::all(&dev);
    outcomes.push(("quant forms", cases::quant_forms(&dev, &reference, 12)));
    let small = qwen_common::small_gemm_host(lib);
    outcomes.push((
        "fast small-N gemm",
        cases::small_gemm(&small, 14).map(|_| ()),
    ));
    for (name, outcome) in outcomes {
        match outcome {
            Ok(()) => eprintln!("ok      {name}"),
            Err(e) => {
                eprintln!("FAILED  {name}: {e}");
                failed.push(name);
            }
        }
    }
    assert!(failed.is_empty(), "{failed:?}");
}
