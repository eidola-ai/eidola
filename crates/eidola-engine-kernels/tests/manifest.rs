//! The committed manifest against the committed build inputs. These run in a
//! plain `cargo test` with no CUDA toolkit: they cannot rebuild the kernels,
//! but they catch a manifest that no longer describes the sources next to it,
//! and they hold the SASS-level facts the kernel set was chosen for.

use std::path::Path;

use eidola_engine_kernels::{Manifest, sha256_hex};
use serde_json::Value;

fn crate_file(path: &str) -> String {
    let full = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{}: {e}", full.display()))
}

fn spec() -> Value {
    serde_json::from_str(&crate_file("csrc/kernels.json")).expect("kernels.json")
}

#[test]
fn embedded_manifest_is_valid() {
    let manifest = Manifest::embedded();
    assert!(!manifest.kernels.is_empty());
    assert!(!manifest.fatbins.is_empty());
}

/// Every committed input the build hashed still has that hash; a source or
/// build-script edit without a rebuild fails here.
#[test]
fn inputs_match_committed_files() {
    let manifest = Manifest::embedded();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut checked = 0;
    for input in &manifest.inputs {
        if input.path.starts_with("generated/") {
            continue;
        }
        let bytes =
            std::fs::read(dir.join(&input.path)).unwrap_or_else(|e| panic!("{}: {e}", input.path));
        assert_eq!(
            sha256_hex(&bytes),
            input.sha256,
            "{} changed since the manifest was built; rebuild the kernels and commit the new \
             manifest",
            input.path
        );
        checked += 1;
    }
    // And nothing new appeared in csrc/ or nix/ (the sources and the whole build
    // recipe, `default.nix` and `sources.nix` included) that the build has not seen.
    for sub in ["csrc", "nix"] {
        for entry in std::fs::read_dir(dir.join(sub)).expect(sub) {
            let name = entry.expect("entry").file_name();
            let path = format!("{sub}/{}", name.to_string_lossy());
            assert!(
                manifest.inputs.iter().any(|i| i.path == path),
                "{path} is not in the manifest; rebuild the kernels"
            );
        }
    }
    assert!(checked >= 3);
}

/// The manifest covers exactly the kernel x arch matrix in kernels.json, with
/// the flags it declares.
#[test]
fn manifest_matches_kernel_spec() {
    let manifest = Manifest::embedded();
    let spec = spec();
    let archs: Vec<&str> = spec["archs"]
        .as_array()
        .expect("archs")
        .iter()
        .map(|a| a.as_str().expect("arch"))
        .collect();
    let kernels = spec["kernels"].as_array().expect("kernels");
    assert_eq!(manifest.kernels.len(), kernels.len() * archs.len());
    for kernel in kernels {
        let name = kernel["name"].as_str().expect("name");
        for arch in &archs {
            let cubin = manifest
                .cubin(name, arch)
                .unwrap_or_else(|| panic!("no cubin {name} {arch}"));
            assert_eq!(cubin.profile, kernel["profile"].as_str().expect("profile"));
            assert_eq!(
                cubin.source,
                format!("csrc/{}", kernel["source"].as_str().expect("source"))
            );
        }
        let fatbin = manifest.fatbin(name).expect("fatbin");
        assert_eq!(
            serde_json::to_value(&fatbin.archs).expect("archs"),
            spec["fatbin_archs"]
        );
    }
    for (profile, flags) in &manifest.compile.profiles {
        assert_eq!(
            serde_json::to_value(&flags.flags).expect("flags"),
            spec["profiles"][profile]["flags"],
            "{profile}"
        );
    }
    assert_eq!(
        serde_json::to_value(&manifest.compile.common_flags).expect("flags"),
        spec["common_flags"]
    );
}

/// The upstream pins in the manifest are the ones nix/sources.nix fetches.
#[test]
fn source_pins_match_nix() {
    let manifest = Manifest::embedded();
    let sources_nix = crate_file("nix/sources.nix");
    assert_eq!(manifest.sources.len(), 3);
    for (name, pin) in &manifest.sources {
        assert!(
            sources_nix.contains(&format!("rev = \"{}\";", pin.rev)),
            "{name} rev {} not in sources.nix",
            pin.rev
        );
        assert!(
            sources_nix.contains(&format!("hash = \"{}\";", pin.nar_hash)),
            "{name} hash {} not in sources.nix",
            pin.nar_hash
        );
    }
}

fn mma(manifest: &Manifest, name: &str, arch: &str, family: &str) -> u64 {
    manifest
        .cubin(name, arch)
        .expect("cubin")
        .mma_sass
        .iter()
        .filter(|(op, _)| op.starts_with(family))
        .map(|(_, n)| n)
        .sum()
}

/// The GEMMs run on Blackwell's tcgen05 tensor cores (UTC* SASS: UTCQMMA is
/// the f8f6f4 MMA, UTCHMMA the 16-bit one), on every target; FlashInfer's FA2
/// attention is the mma.sync path (HMMA), with no tcgen05 at all; our own
/// kernels use no tensor cores.
#[test]
fn sass_uses_the_intended_tensor_core_path() {
    let manifest = Manifest::embedded();
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        for gemm in [
            "cutlass_fp8_blockwise_gemm",
            "cutlass_fp8_blockwise_swap",
            "deepgemm_fp8_fp4_grouped",
        ] {
            assert!(mma(manifest, gemm, arch, "UTCQMMA") > 0, "{gemm} {arch}");
            assert_eq!(mma(manifest, gemm, arch, "HMMA"), 0, "{gemm} {arch}");
        }
        let fa2 = "flashinfer_fa2_sink_paged";
        assert!(mma(manifest, fa2, arch, "HMMA") > 0, "{arch}");
        assert_eq!(mma(manifest, fa2, arch, "UTC"), 0, "{arch}");
        assert!(
            mma(manifest, "cutlass_bf16_gemm", arch, "UTCHMMA") > 0,
            "{arch}"
        );
        assert_eq!(
            mma(manifest, "cutlass_bf16_gemm", arch, "HMMA"),
            0,
            "{arch}"
        );
        for own in [
            "rmsnorm",
            "sampling",
            "engine_ops",
            "engine_ops_reference",
            "kv_cipher",
            "qwen_fast",
            "qwen_sampling",
        ] {
            let cubin = manifest.cubin(own, arch).expect("cubin");
            assert!(cubin.mma_sass.is_empty(), "{own} {arch}");
        }
    }
}

/// A profile that turns contraction off is held to it: every cubin built under
/// one records its floating-point operations, its PTX holds no fused
/// multiply-add and no multiply, add or subtract left for ptxas to fuse, and
/// no other cubin carries the record. The numerics probe without division is
/// what shows ptxas fuses none itself: hundreds of rounded operations (the
/// exponential's Horner steps, the pinned dots, the rotations) and not one
/// fused multiply-add in its SASS; the full probe's are its divisions'.
#[test]
fn exact_profiles_fuse_no_multiply_add() {
    let manifest = Manifest::embedded();
    let mut exact = 0;
    for cubin in &manifest.kernels {
        let flags = &manifest.compile.profiles[&cubin.profile].flags;
        let unfused = flags.iter().any(|f| f == "--fmad=false");
        assert!(
            !flags.iter().any(|f| f.contains("fast")) || !unfused,
            "{}: a profile without contraction takes no fast-math flag",
            cubin.profile
        );
        let Some(ops) = &cubin.float_ops else {
            assert!(!unfused, "{} {}: no float_ops", cubin.name, cubin.arch);
            continue;
        };
        assert!(unfused, "{} {}", cubin.name, cubin.arch);
        assert_eq!(ops.ptx_fma, 0, "{} {}", cubin.name, cubin.arch);
        assert_eq!(ops.ptx_unrounded, 0, "{} {}", cubin.name, cubin.arch);
        exact += 1;
    }
    assert!(exact > 0);
    assert_eq!(
        manifest.compile.profiles["exact"].flags,
        ["-std=c++17", "-O3", "--fmad=false"]
    );
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        let nodiv = manifest.cubin("ds_probe_nodiv", arch).expect("probe");
        let ops = nodiv.float_ops.as_ref().expect("float_ops");
        assert!(ops.ptx_rounded > 500, "{arch}: {}", ops.ptx_rounded);
        assert_eq!(ops.sass_fma, 0, "{arch}");
        assert!(nodiv.mma_sass.is_empty(), "{arch}");
        let full = manifest.cubin("ds_probe", arch).expect("probe");
        assert!(
            full.float_ops.as_ref().expect("float_ops").sass_fma > 0,
            "{arch}"
        );
        assert_eq!(full.entries[0].symbol, "eidola_ds_probe");
        assert_eq!(nodiv.entries[0].symbol, "eidola_ds_probe_nodiv");
    }
}

/// Every kernel entry is bound to exactly one launch-contract record and every
/// record to exactly one entry; entries we wrote ourselves keep their unmangled
/// names and `<entry>_meta` records; mangled entries are bound by the aliases
/// in kernels.json, and those aliases are exactly the bindings that differ
/// from the naming convention.
#[test]
fn entries_and_meta_records() {
    let manifest = Manifest::embedded();
    let spec = spec();
    for cubin in &manifest.kernels {
        let mut records: Vec<&str> = cubin.meta.iter().map(String::as_str).collect();
        let mut bound: Vec<&str> = cubin.entries.iter().map(|e| e.meta.as_str()).collect();
        records.sort_unstable();
        bound.sort_unstable();
        assert_eq!(records, bound, "{} {}", cubin.name, cubin.arch);
        let kernel = spec["kernels"]
            .as_array()
            .expect("kernels")
            .iter()
            .find(|k| k["name"] == cubin.name.as_str())
            .expect("kernel in spec");
        let aliases = kernel["meta_aliases"].as_object();
        for entry in &cubin.entries {
            assert!(entry.meta.ends_with("_meta"), "{}", entry.meta);
            let conventional = format!("{}_meta", entry.symbol);
            if entry.meta == conventional {
                assert!(!entry.symbol.starts_with("_Z"), "{}", entry.symbol);
            } else {
                let alias = aliases
                    .and_then(|a| a.get(&entry.meta))
                    .and_then(|v| v.as_str());
                assert_eq!(alias, Some(entry.symbol.as_str()), "{}", entry.meta);
            }
            assert_eq!(
                cubin.entry_for_meta(&entry.meta).map(|e| &e.symbol),
                Some(&entry.symbol)
            );
        }
        if let Some(aliases) = aliases {
            for (meta, symbol) in aliases {
                let entry = cubin
                    .entry(symbol.as_str().expect("symbol"))
                    .unwrap_or_else(|| panic!("{meta}: aliased entry not in {}", cubin.name));
                assert_eq!(&entry.meta, meta);
            }
        }
    }
    let rmsnorm = manifest.cubin("rmsnorm", "sm_100a").expect("rmsnorm");
    assert_eq!(rmsnorm.entries[0].symbol, "eidola_rmsnorm_bf16");
    assert_eq!(rmsnorm.entries[0].meta, "eidola_rmsnorm_bf16_meta");
    let fa2 = manifest
        .cubin("flashinfer_fa2_sink_paged", "sm_103a")
        .expect("fa2");
    // One reduction shape (one KV warp, 64-key tiles) at three query tiles,
    // for global layers (split and merged) and sliding ones (anchored), and
    // the merge: nothing else.
    assert_eq!(fa2.entries.len(), 7);
    for kind in ["", "anchored_"] {
        for tile in [16, 64, 128] {
            let symbol = format!("eidola_fa2_sink_paged_{kind}bf16_q{tile}");
            assert!(fa2.entry(&symbol).is_some(), "{symbol}");
        }
    }
    let merge = fa2
        .entry_for_meta("eidola_fa2_merge_states_bf16_d128_meta")
        .expect("merge");
    assert!(
        merge
            .demangled
            .starts_with("void flashinfer::PersistentVariableLengthMergeStatesKernel<"),
        "{}",
        merge.demangled
    );
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        let deepgemm = manifest
            .cubin("deepgemm_fp8_fp4_grouped", arch)
            .expect("deepgemm");
        assert_eq!(deepgemm.entries.len(), 2);
        // Each record describes the instance its name says: the psum layout
        // (`MGroupedContiguousWithPsumLayout`, GemmType 5); gate/up has
        // K = 4096, down K = 2048.
        for (proj, shape_k) in [("gate_up", 4096), ("down", 2048)] {
            let meta = format!("eidola_deepgemm_fp8_fp4_psum_{proj}_meta");
            let entry = deepgemm.entry_for_meta(&meta).expect(&meta);
            assert!(
                entry
                    .demangled
                    .starts_with("void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<"),
                "{}",
                entry.demangled
            );
            assert!(
                entry.demangled.contains("(deep_gemm::GemmType)5"),
                "{meta}: {}",
                entry.demangled
            );
            assert!(
                entry.demangled.contains(&format!(
                    "(unsigned int)0, (unsigned int)4096, (unsigned int){shape_k},"
                )),
                "{meta}: {}",
                entry.demangled
            );
        }
    }
}

/// Sampling comes only from our own kernels (`sampling.cu`, and Qwen's
/// `qwen_sampling.cu`), which reproduce the serving core's sampling semantics bit
/// for bit (see AGENTS.md): no upstream sampling entry is in the set.
#[test]
fn no_third_party_sampling_kernels() {
    let manifest = Manifest::embedded();
    for cubin in &manifest.kernels {
        for entry in &cubin.entries {
            assert!(
                !entry.demangled.to_lowercase().contains("sampling"),
                "{} {}: {}",
                cubin.name,
                cubin.arch,
                entry.demangled
            );
        }
        if cubin.name == "sampling" {
            let mut names: Vec<&str> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            names.sort_unstable();
            assert_eq!(
                names,
                ["eidola_chain_accept", "eidola_sample"],
                "{}",
                cubin.arch
            );
        }
    }
}

/// The DeepSeek scalar kernels: the images that neither divide nor take a
/// square root hold rounded operations and not one fused multiply-add in
/// their SASS (`exact_profiles_fuse_no_multiply_add` holds every image's PTX);
/// the image of the dividing kernels holds only its divisions' expansions.
/// Each image has exactly its entries.
#[test]
fn ds_scalar_kernels_fuse_nothing() {
    let manifest = Manifest::embedded();
    let images: [(&str, bool, &[&str]); 4] = [
        (
            "ds_ops",
            false,
            &[
                "eidola_ds_collapse",
                "eidola_ds_combine",
                "eidola_ds_embed",
                "eidola_ds_engram_gather",
                "eidola_ds_expand",
                "eidola_ds_linear",
                "eidola_ds_place",
                "eidola_ds_place_count",
                "eidola_ds_place_rows",
                "eidola_ds_quant_fp8",
                "eidola_ds_rot_quant",
            ],
        ),
        (
            "ds_ops_div",
            true,
            &[
                "eidola_ds_engram_inject",
                "eidola_ds_mhc_mix",
                "eidola_ds_pool",
                "eidola_ds_rms_norm",
                "eidola_ds_rot_quant_e4m3",
                "eidola_ds_route",
                "eidola_ds_stream_mean",
                "eidola_ds_swiglu",
            ],
        ),
        (
            "ds_reference",
            false,
            &[
                "eidola_ds_grouped_dot",
                "eidola_ds_reference_lin4",
                "eidola_ds_reference_lin8",
            ],
        ),
        ("ds_head", false, &["eidola_ds_head"]),
    ];
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        for (name, divides, entries) in images {
            let cubin = manifest
                .cubin(name, arch)
                .unwrap_or_else(|| panic!("no cubin {name} {arch}"));
            assert_eq!(cubin.profile, "exact", "{name}");
            let ops = cubin.float_ops.as_ref().expect("float_ops");
            assert!(ops.ptx_rounded > 0, "{name} {arch}");
            assert_eq!(ops.sass_fma > 0, divides, "{name} {arch}: {}", ops.sass_fma);
            assert!(cubin.mma_sass.is_empty(), "{name} {arch}");
            let mut symbols: Vec<&str> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            symbols.sort_unstable();
            assert_eq!(symbols, entries, "{name} {arch}");
        }
    }
}

/// DeepSeek's tensor-core GEMMs (`ds_gemm_tc`): warp-level MMAs (`HMMA`, F16
/// operands: these parts have no warp-level FP8 MMA, and the kernels convert
/// their FP8 and FP4 codes exactly) and no tcgen05; built without contraction
/// and dividing nothing, so their SASS holds no fused multiply-add at all (the
/// block accumulation `acc + (dot · scale) · scale` is three rounded
/// operations, as the reference's). The image has exactly its entries.
#[test]
fn ds_gemm_tc_runs_on_tensor_cores_and_fuses_nothing() {
    let manifest = Manifest::embedded();
    let mut want = vec!["eidola_ds_gemm_probe".to_owned()];
    for product in ["lin4", "lin8"] {
        for form in ["banded", "tolerance"] {
            for tile in ["s8", "t32", "t8"] {
                want.push(format!("eidola_ds_{product}_{form}_{tile}"));
            }
        }
    }
    want.sort_unstable();
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        let cubin = manifest.cubin("ds_gemm_tc", arch).expect("ds_gemm_tc");
        assert_eq!(cubin.profile, "exact");
        assert!(mma(manifest, "ds_gemm_tc", arch, "HMMA") > 0, "{arch}");
        assert_eq!(mma(manifest, "ds_gemm_tc", arch, "UTC"), 0, "{arch}");
        let ops = cubin.float_ops.as_ref().expect("float_ops");
        assert!(ops.ptx_rounded > 0, "{arch}");
        assert_eq!(ops.sass_fma, 0, "{arch}");
        let mut symbols: Vec<String> = cubin.entries.iter().map(|e| e.symbol.clone()).collect();
        symbols.sort_unstable();
        assert_eq!(symbols, want, "{arch}");
    }
}

/// Qwen3.8's own kernels: every image is built without contraction; the images that
/// neither divide nor take a square root (the linears, the KV write, the conversions and the
/// DeltaNet recurrence) hold rounded operations and not one fused multiply-add in their SASS; the
/// dividing image holds only its divisions' expansions. None uses tensor cores. Each
/// image has exactly its entries.
#[test]
fn qwen_kernels_fuse_nothing() {
    let manifest = Manifest::embedded();
    let images: [(&str, bool, &[&str]); 3] = [
        (
            "qwen_ops",
            false,
            &[
                "eidola_qwen_copy_rows",
                "eidola_qwen_embed",
                "eidola_qwen_kv_write",
                "eidola_qwen_lin8",
                "eidola_qwen_linear",
                "eidola_qwen_round",
                "eidola_qwen_widen",
            ],
        ),
        (
            "qwen_ops_div",
            true,
            &[
                "eidola_qwen_attn",
                "eidola_qwen_dn_norm",
                "eidola_qwen_dn_prep",
                "eidola_qwen_gate",
                "eidola_qwen_norm",
                "eidola_qwen_quant",
            ],
        ),
        (
            "qwen_deltanet",
            false,
            &["eidola_qwen_dn_rec", "eidola_qwen_dn_rec_short"],
        ),
    ];
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        for (name, divides, entries) in images {
            let cubin = manifest
                .cubin(name, arch)
                .unwrap_or_else(|| panic!("no cubin {name} {arch}"));
            assert_eq!(cubin.profile, "exact", "{name}");
            let ops = cubin.float_ops.as_ref().expect("float_ops");
            assert!(ops.ptx_rounded > 0, "{name} {arch}");
            assert_eq!(ops.sass_fma > 0, divides, "{name} {arch}: {}", ops.sass_fma);
            assert!(cubin.mma_sass.is_empty(), "{name} {arch}");
            let mut symbols: Vec<&str> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            symbols.sort_unstable();
            assert_eq!(symbols, entries, "{name} {arch}");
        }
    }
}

/// The fast profile's attention at head dimension 256: FlashInfer's FA2 (`HMMA`, no
/// tcgen05), its split instances at query tiles 16 and 64 and the d256 merge.
#[test]
fn qwen_fa2_d256_is_the_mma_sync_path() {
    let manifest = Manifest::embedded();
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        let name = "flashinfer_fa2_sink_paged_d256";
        assert!(mma(manifest, name, arch, "HMMA") > 0, "{arch}");
        assert_eq!(mma(manifest, name, arch, "UTC"), 0, "{arch}");
        let cubin = manifest.cubin(name, arch).expect("cubin");
        let mut metas: Vec<&str> = cubin.meta.iter().map(String::as_str).collect();
        metas.sort_unstable();
        assert_eq!(
            metas,
            [
                "eidola_fa2_merge_states_bf16_d256_meta",
                "eidola_fa2_sink_paged_d256_bf16_q16_meta",
                "eidola_fa2_sink_paged_d256_bf16_q64_meta",
            ],
            "{arch}"
        );
    }
}

/// The fast profile's own kernels: the `own` profile (contraction allowed: their bodies
/// spell out the fused multiply-adds they mean), no tensor cores, exactly their two
/// entries; the swapped FP8 GEMM's image exactly its three tile instances; and Qwen's
/// sampler, in both profiles, the `own` profile (every f64 operation it means rounded is
/// an explicit round-to-nearest intrinsic, as the shared sampler's), its four sample and
/// three acceptance kernels.
#[test]
fn qwen_fast_images_hold_their_entries() {
    let manifest = Manifest::embedded();
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        for (name, profile, entries) in [
            (
                "qwen_fast",
                "own",
                &["eidola_qwen_fast_gate", "eidola_qwen_fast_gemm_bf16"][..],
            ),
            (
                "cutlass_fp8_blockwise_swap",
                "cutlass",
                &[
                    "eidola_cutlass_fp8_blockwise_swap16_bf16",
                    "eidola_cutlass_fp8_blockwise_swap32_bf16",
                    "eidola_cutlass_fp8_blockwise_swap64_bf16",
                ][..],
            ),
            (
                "qwen_sampling",
                "own",
                &[
                    "eidola_qwen_accept_decide",
                    "eidola_qwen_accept_finish",
                    "eidola_qwen_accept_sums",
                    "eidola_qwen_sample_exp",
                    "eidola_qwen_sample_norm",
                    "eidola_qwen_sample_pick",
                    "eidola_qwen_sample_top",
                ][..],
            ),
        ] {
            let cubin = manifest
                .cubin(name, arch)
                .unwrap_or_else(|| panic!("no cubin {name} {arch}"));
            assert_eq!(cubin.profile, profile, "{name}");
            let mut symbols: Vec<&str> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            symbols.sort_unstable();
            assert_eq!(symbols, entries, "{name} {arch}");
        }
    }
}
