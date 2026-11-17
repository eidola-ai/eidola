//! The fast profile's reproducibility between calls on a GPU, asserted (the serving
//! contract): greedy tokens and every logit's bits the same alone and beside traffic,
//! under every chunking, decoded and prefilled, from a cache hit and computed fresh, and
//! at every draft width (`qwen_common::invariance`). The exact profile is run the same
//! way beside it. Then the fast profile's deviation from the exact one on the same
//! requests, recorded. Skipped, saying why, without a device or
//! `EIDOLA_ENGINE_KERNELS_DIR`.

mod qwen_common;

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::QwenProfile;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use qwen_common::invariance;

#[test]
fn qwen_fast_profile_is_invariant() {
    let Some(dir) = qwen_common::device_kernels("qwen_fast_profile_is_invariant") else {
        return;
    };
    let mut failed = Vec::new();
    let mut books = Vec::new();
    for profile in [QwenProfile::Fast, QwenProfile::Exact] {
        let make = || CudaDevice::open(0, &KernelDir::new(&dir), profile).expect("the device");
        for (name, r) in invariance::all(&make) {
            match r {
                Ok(what) => eprintln!("ok      {} {name}: {what}", profile.name()),
                Err(e) => {
                    eprintln!("FAILED  {} {name}: {e}", profile.name());
                    failed.push(format!("{} {name}", profile.name()));
                }
            }
        }
        books.push(invariance::reference_book(&make));
    }
    let (fast, exact) = (&books[0], &books[1]);
    let report = invariance::deviation(fast, exact);
    eprintln!("deviation of the fast profile from the exact one: {report}");
    qwen_common::record("qwen_fast_deviation", &report);
    assert!(failed.is_empty(), "{failed:?}");
}
