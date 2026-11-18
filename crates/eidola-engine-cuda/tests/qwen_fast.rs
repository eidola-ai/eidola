//! The fast profile's reproducibility between calls on a GPU, asserted (the serving
//! contract): greedy tokens and every logit's bits the same alone and beside traffic,
//! under every chunking, decoded and prefilled, from a cache hit and computed fresh, at
//! every draft width (`qwen_common::invariance`), eagerly and with decode steps replaying
//! their captured graphs; and the same workloads giving the same tokens and logits bits on
//! every path (eager, a rung's launches run directly, its graph replayed). The fast
//! profile's attention is FA2 at head dimension 256. The exact profile is run the same way
//! beside it. Then the fast profile's deviation from the exact one on the same requests,
//! recorded. All on the fixture with 256-wide attention heads (`qwen_tiny256`), the
//! released geometry the FA2 instances are compiled for. Skipped, saying why, without a
//! device or `EIDOLA_ENGINE_KERNELS_DIR`.

mod qwen_common;

use eidola_engine_cuda::KernelDir;
use eidola_engine_cuda::qwen::cuda::CudaDevice;
use eidola_engine_cuda::qwen::{CudaGraphs, DecodePath, QwenProfile};
use qwen_common::invariance;

fn report(name: String, r: Result<String, String>, failed: &mut Vec<String>) {
    match r {
        Ok(what) => eprintln!("ok      {name}: {what}"),
        Err(e) => {
            eprintln!("FAILED  {name}: {e}");
            failed.push(name);
        }
    }
}

#[test]
fn qwen_fast_profile_is_invariant() {
    let Some(dir) = qwen_common::device_kernels("qwen_fast_profile_is_invariant") else {
        return;
    };
    qwen_common::with_fixture_256(|| run(&dir));
}

/// Both profiles on the fixture with 256-wide heads (the fast profile's attention kernels
/// are compiled for them).
fn run(dir: &str) {
    let mut failed = Vec::new();
    let mut books = Vec::new();
    for profile in [QwenProfile::Fast, QwenProfile::Exact] {
        let make = || CudaDevice::open(0, &KernelDir::new(dir), profile).expect("the device");
        for (graphs, path) in [
            (CudaGraphs::Off, DecodePath::Eager),
            (CudaGraphs::On, DecodePath::Replay),
        ] {
            qwen_common::with_path(graphs, path, || {
                for (name, r) in invariance::all(&make) {
                    report(
                        format!("{} {path:?} {name}", profile.name()),
                        r,
                        &mut failed,
                    );
                }
            });
        }
        let paths = [
            (CudaGraphs::Off, DecodePath::Eager),
            (CudaGraphs::Off, DecodePath::Direct),
            (CudaGraphs::On, DecodePath::Replay),
        ];
        for (name, r) in invariance::across_paths(&make, &paths) {
            report(format!("{} {name}", profile.name()), r, &mut failed);
        }
        books.push(invariance::reference_book(&make));
    }
    let (fast, exact) = (&books[0], &books[1]);
    let deviation = invariance::deviation(fast, exact);
    eprintln!("deviation of the fast profile from the exact one: {deviation}");
    qwen_common::record("qwen_fast_deviation", &deviation);
    assert!(failed.is_empty(), "{failed:?}");
}
