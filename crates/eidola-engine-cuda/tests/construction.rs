//! The executor refuses what it cannot run before touching device memory:
//! the configuration and the KV geometry (also checkable without a device,
//! `CudaExecutor::preflight`), then the device, then the kernel images, and
//! only then the weights. The checkpoint here is a configuration with no tensors,
//! so reaching the weights at all shows up as a missing-tensor error.

mod common;

use std::sync::Arc;

use eidola_engine::spec::Bucket;
use eidola_engine_cuda::{
    CudaError, CudaExecutor, CudaExecutorConfig, CudaGraphs, Gpu, ImageArch, KernelDir, KvBlocks,
};
use eidola_engine_model::safetensors::WeightSet;

/// A fresh directory per call, so tests running in parallel never share one.
fn tensorless_checkpoint() -> Arc<WeightSet> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("eidola-config-only-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../eidola-engine-model/tests/data/flash-mopd.config.json"
    );
    std::fs::copy(src, dir.join("config.json")).unwrap();
    Arc::new(WeightSet::open_dir(&dir).unwrap())
}

fn config() -> CudaExecutorConfig {
    CudaExecutorConfig {
        block_size: 16,
        num_blocks: KvBlocks {
            global: 8,
            sliding: 8,
        },
        num_state_slots: 2,
        max_model_len: 256,
        buckets: vec![Bucket {
            max_seqs: 2,
            max_tokens: 64,
        }],
        sampleable_vocab_size: 151_675,
        image: None,
        graphs: CudaGraphs::Off,
    }
}

/// The host-side refusals need no device: `preflight` makes them from the
/// checkpoint's configuration alone, and returns the spec `new` would report.
#[test]
fn preflight_refuses_without_a_device() {
    let store = tensorless_checkpoint();
    let spec = CudaExecutor::preflight(&store, None, &config()).unwrap();
    assert_eq!(spec.kv_groups.len(), 2);
    assert_eq!(spec.max_draft_tokens, 0);

    let mut cfg = config();
    cfg.block_size = 16_777_217;
    let e = CudaExecutor::preflight(&store, None, &cfg).unwrap_err();
    assert!(e.to_string().contains("KV geometry"), "{e}");

    let mut cfg = config();
    cfg.sampleable_vocab_size = 200_000;
    let e = CudaExecutor::preflight(&store, None, &cfg).unwrap_err();
    assert!(matches!(e, CudaError::Unsupported(_)), "{e}");

    let mut cfg = config();
    cfg.max_model_len = (1 << 20) + 1;
    let e = CudaExecutor::preflight(&store, None, &cfg).unwrap_err();
    let CudaError::Unsupported(u) = e else {
        panic!("a configuration refusal, not {e}")
    };
    assert_eq!(u.field, "max_model_len");

    let e = CudaExecutor::preflight(&store, Some(&[0]), &config()).unwrap_err();
    let CudaError::Unsupported(u) = e else {
        panic!("a configuration refusal, not {e}")
    };
    assert_eq!(u.field, "num_blocks.sliding");
}

#[test]
fn refusals_come_before_device_memory() {
    let Some(su) = common::setup() else { return };
    drop(su);
    let store = tensorless_checkpoint();
    let kernels = KernelDir::new(std::env::var_os("EIDOLA_ENGINE_KERNELS_DIR").unwrap());
    let open = || Gpu::open(0).unwrap();

    // A KV geometry whose sizes would wrap.
    let mut cfg = config();
    cfg.block_size = 16_777_217;
    let e = CudaExecutor::new(open(), &kernels, store.clone(), None, cfg).unwrap_err();
    assert!(e.to_string().contains("KV geometry"), "{e}");

    // A sampleable vocabulary the sampler cannot serve.
    let mut cfg = config();
    cfg.sampleable_vocab_size = 200_000;
    let e = CudaExecutor::new(open(), &kernels, store.clone(), None, cfg).unwrap_err();
    assert!(matches!(e, CudaError::Unsupported(_)), "{e}");

    // A context past the checkpoint's window.
    let mut cfg = config();
    cfg.max_model_len = (1 << 20) + 1;
    let e = CudaExecutor::new(open(), &kernels, store.clone(), None, cfg).unwrap_err();
    let CudaError::Unsupported(u) = e else {
        panic!("a configuration refusal, not {e}")
    };
    assert_eq!(u.field, "max_model_len");

    // A layer selection out of the checkpoint's order.
    let e =
        CudaExecutor::new(open(), &kernels, store.clone(), Some(&[1, 0]), config()).unwrap_err();
    let CudaError::Unsupported(u) = e else {
        panic!("a configuration refusal, not {e}")
    };
    assert_eq!(u.field, "keep_layers");

    // Blocks for a KV group no retained layer uses (layer 0 is global).
    let e = CudaExecutor::new(open(), &kernels, store.clone(), Some(&[0]), config()).unwrap_err();
    let CudaError::Unsupported(u) = e else {
        panic!("a configuration refusal, not {e}")
    };
    assert_eq!(u.field, "num_blocks.sliding");

    // A kernel directory with no images: refused before any weight is read.
    let empty = std::env::temp_dir().join(format!("eidola-no-kernels-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let e = CudaExecutor::new(
        open(),
        &KernelDir::new(&empty),
        store.clone(),
        None,
        config(),
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("reading"),
        "a kernel image error, not a weight one: {e}"
    );

    // A device the forced image cannot run: refused as a device requirement,
    // before the (empty) kernel directory is even read.
    let mut cfg = config();
    cfg.image = Some(match open().image_arch().unwrap() {
        ImageArch::Sm103a => ImageArch::Sm100a,
        _ => ImageArch::Sm103a,
    });
    let e =
        CudaExecutor::new(open(), &KernelDir::new(&empty), store.clone(), None, cfg).unwrap_err();
    let CudaError::Unsupported(u) = e else {
        panic!("a device refusal, not {e}")
    };
    assert_eq!(u.field, "device.compute_capability");

    // With the kernels present, the next thing it reaches is the weights.
    let e = CudaExecutor::new(open(), &kernels, store, None, config()).unwrap_err();
    assert!(!e.to_string().contains("reading"), "{e}");
}
