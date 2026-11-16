//! What the committed manifest must show of the DeepSeek attention, index and
//! selection images.

use eidola_engine_kernels::Manifest;

#[test]
fn images_hold_their_entries_and_fuse_nothing() {
    let manifest = Manifest::embedded();
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        let attention = manifest.cubin("ds_attention", arch).expect("ds_attention");
        let index = manifest.cubin("ds_index", arch).expect("ds_index");
        for (cubin, entries) in [
            (attention, ["eidola_ds_attn", "eidola_ds_window_slots"]),
            (index, ["eidola_ds_index_score", "eidola_ds_select"]),
        ] {
            assert_eq!(cubin.profile, "exact", "{arch}");
            for entry in entries {
                assert!(cubin.entry(entry).is_some(), "{arch}: {entry}");
            }
            assert_eq!(cubin.entries.len(), 2, "{arch}");
            assert!(cubin.mma_sass.is_empty(), "{arch}");
            let ops = cubin.float_ops.as_ref().expect("float_ops");
            assert_eq!((ops.ptx_fma, ops.ptx_unrounded), (0, 0), "{arch}");
            assert!(ops.ptx_rounded > 100, "{arch}: {}", ops.ptx_rounded);
        }
        // The index image has no division, so its SASS may hold no fused multiply-add at all.
        // The attention image divides once per output element (`values / normaliser`): its
        // fused multiply-adds are that division's expansion, five per inlined site and the
        // slow path's.
        assert_eq!(index.float_ops.as_ref().unwrap().sass_fma, 0, "{arch}");
        assert!(attention.float_ops.as_ref().unwrap().sass_fma > 0, "{arch}");
    }
}

/// `ds_abi.cuh`'s `constexpr` helpers are host functions. A device call of one compiles without
/// a diagnostic under the `exact` profile and becomes a trap, which removes the code after it
/// (the stores behind a plane's address were dropped that way: on a device, zero scores and an
/// illegal address; on a host, nothing). Device sources use their own (`ds::kv_code_bytes`).
#[test]
fn device_sources_call_no_host_constexpr_helper() {
    let csrc = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc");
    for entry in std::fs::read_dir(&csrc).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name == "ds_abi.cuh" || !(name.ends_with(".cu") || name.ends_with(".cuh")) {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        for helper in ["ds_kv_code_bytes(", "ds_kv_scale_bytes(", "ds_mix_words("] {
            let called = text
                .lines()
                .any(|l| l.contains(helper) && !l.trim_start().starts_with("//"));
            assert!(!called, "{name} calls the host function {helper}…)");
        }
    }
}
