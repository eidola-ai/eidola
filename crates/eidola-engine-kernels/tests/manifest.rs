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
            "deepgemm_fp8_fp4_grouped",
            "deepgemm_fp8_fp4_grouped_variants",
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
        for own in ["rmsnorm", "sampling", "engine_ops", "engine_ops_reference"] {
            let cubin = manifest.cubin(own, arch).expect("cubin");
            assert!(cubin.mma_sass.is_empty(), "{own} {arch}");
        }
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
    for tile in [16, 64, 128] {
        let symbol = format!("eidola_fa2_sink_paged_bf16_q{tile}");
        assert!(fa2.entry(&symbol).is_some(), "{symbol}");
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
        // The bench-only variants: the same template over the psum layout,
        // each record bound to the instance its name says (block M, stages,
        // cluster).
        let variants = manifest
            .cubin("deepgemm_fp8_fp4_grouped_variants", arch)
            .expect("variants");
        assert_eq!(variants.entries.len(), 8);
        for (name, block_m, stages, cluster) in [
            ("m64s8", 64, 8, 2),
            ("m64s10", 64, 10, 2),
            ("m32s11", 32, 11, 2),
            ("m64s8c1", 64, 8, 1),
        ] {
            for (proj, shape_k) in [("gate_up", 4096), ("down", 2048)] {
                let meta = format!("eidola_deepgemm_variant_{name}_{proj}_meta");
                let entry = variants.entry_for_meta(&meta).expect(&meta);
                let want = format!(
                    "(unsigned int)0, (unsigned int)4096, (unsigned int){shape_k}, \
                     (unsigned int){block_m}, (unsigned int)128, (unsigned int)128, \
                     (unsigned int)256, (unsigned int)128, (unsigned int)128, \
                     (unsigned int)128, (unsigned int){stages}, (unsigned int)2, \
                     (unsigned int)128, (unsigned int)128, (unsigned int){cluster}, \
                     (bool)1, (unsigned int)148,"
                );
                assert!(
                    entry.demangled.contains(&want)
                        && entry.demangled.contains("(deep_gemm::GemmType)5"),
                    "{meta}: {}",
                    entry.demangled
                );
            }
        }
    }
}

/// Sampling comes only from our own kernel (`sampling.cu`), which reproduces
/// the serving core's sampling semantics bit for bit (see AGENTS.md): no
/// upstream sampling entry is in the set.
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
