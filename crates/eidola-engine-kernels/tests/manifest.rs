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
    // And nothing new appeared in csrc/ that the build has not seen.
    for entry in std::fs::read_dir(dir.join("csrc")).expect("csrc") {
        let name = entry.expect("csrc entry").file_name();
        let path = format!("csrc/{}", name.to_string_lossy());
        assert!(
            manifest.inputs.iter().any(|i| i.path == path),
            "{path} is not in the manifest; rebuild the kernels"
        );
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
/// the f8f6f4 MMA), on every target; FlashInfer's FA2 attention is the
/// mma.sync path (HMMA), with no tcgen05 at all.
#[test]
fn sass_uses_the_intended_tensor_core_path() {
    let manifest = Manifest::embedded();
    for arch in ["sm_100a", "sm_103a", "sm_100f"] {
        for gemm in ["cutlass_fp8_blockwise_gemm", "deepgemm_fp8_fp4_grouped"] {
            assert!(mma(manifest, gemm, arch, "UTCQMMA") > 0, "{gemm} {arch}");
            assert_eq!(mma(manifest, gemm, arch, "HMMA"), 0, "{gemm} {arch}");
        }
        let fa2 = "flashinfer_fa2_sink_paged";
        assert!(mma(manifest, fa2, arch, "HMMA") > 0, "{arch}");
        assert_eq!(mma(manifest, fa2, arch, "UTC"), 0, "{arch}");
        for plain in ["flashinfer_sampling", "rmsnorm"] {
            let cubin = manifest.cubin(plain, arch).expect("cubin");
            assert!(cubin.mma_sass.is_empty(), "{plain} {arch}");
        }
    }
}

/// Every kernel entry has a launch-contract record, and entries we wrote
/// ourselves keep their unmangled names.
#[test]
fn entries_and_meta_records() {
    let manifest = Manifest::embedded();
    for cubin in &manifest.kernels {
        assert!(!cubin.meta.is_empty(), "{} {}", cubin.name, cubin.arch);
        for meta in &cubin.meta {
            assert!(meta.ends_with("_meta"));
        }
    }
    let rmsnorm = manifest.cubin("rmsnorm", "sm_100a").expect("rmsnorm");
    assert_eq!(rmsnorm.entries[0].symbol, "eidola_rmsnorm_bf16");
    let fa2 = manifest
        .cubin("flashinfer_fa2_sink_paged", "sm_103a")
        .expect("fa2");
    for tile in [16, 64, 128] {
        let symbol = format!("eidola_fa2_sink_paged_bf16_q{tile}");
        assert!(fa2.entries.iter().any(|e| e.symbol == symbol), "{symbol}");
    }
    let deepgemm = manifest
        .cubin("deepgemm_fp8_fp4_grouped", "sm_100a")
        .expect("deepgemm");
    assert_eq!(deepgemm.entries.len(), 4);
    for entry in &deepgemm.entries {
        assert!(
            entry
                .demangled
                .starts_with("void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<"),
            "{}",
            entry.demangled
        );
    }
}
