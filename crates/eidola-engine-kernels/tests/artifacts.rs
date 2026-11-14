//! Loading images through the manifest: bytes come back only when their size
//! and digest match.

use std::path::PathBuf;

use eidola_engine_kernels::{ArtifactDir, KernelMeta, LoadError, Manifest, sha256_hex};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!(
            "eidola-engine-kernels-{tag}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("cubin")).expect("mkdir");
        std::fs::create_dir_all(dir.join("fatbin")).expect("mkdir");
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

const CUBIN: &[u8] = b"\x7fELF not really a cubin";
const FATBIN: &[u8] = b"not really a fatbin";

fn manifest() -> Manifest {
    let json = serde_json::json!({
        "schema_version": 2,
        "toolchain": {
            "nixpkgs": "rev", "cuda": "13.2.51", "host_compiler": "gcc 15.3.0",
            "components": {}
        },
        "sources": {},
        "compile": { "per_source_flags": [], "common_flags": [], "profiles": {
            "own": { "flags": [], "includes": [] }
        }},
        "inputs": [],
        "kernels": [{
            "name": "k", "arch": "sm_100a", "source": "csrc/k.cu", "profile": "own",
            "file": "cubin/k.sm_100a.cubin", "sha256": sha256_hex(CUBIN),
            "size": CUBIN.len(),
            "entries": [{ "symbol": "k", "demangled": "k", "meta": "k_meta" }],
            "meta": ["k_meta"], "mma_sass": {}
        }],
        "fatbins": [{
            "name": "k", "file": "fatbin/k.fatbin", "sha256": sha256_hex(FATBIN),
            "size": FATBIN.len(), "archs": ["sm_100a"]
        }]
    });
    Manifest::parse(&json.to_string()).expect("manifest")
}

#[test]
fn verified_bytes_round_trip() {
    let dir = TempDir::new("ok");
    std::fs::write(dir.0.join("cubin/k.sm_100a.cubin"), CUBIN).expect("write");
    std::fs::write(dir.0.join("fatbin/k.fatbin"), FATBIN).expect("write");
    let manifest = manifest();
    let artifacts = ArtifactDir::new(&dir.0, &manifest);
    assert_eq!(artifacts.cubin("k", "sm_100a").expect("cubin"), CUBIN);
    assert_eq!(artifacts.fatbin("k").expect("fatbin"), FATBIN);
    artifacts.verify_all().expect("verify");
}

#[test]
fn tampered_image_is_refused() {
    let dir = TempDir::new("tampered");
    let mut tampered = CUBIN.to_vec();
    tampered[5] ^= 1;
    std::fs::write(dir.0.join("cubin/k.sm_100a.cubin"), &tampered).expect("write");
    let manifest = manifest();
    let artifacts = ArtifactDir::new(&dir.0, &manifest);
    assert!(matches!(
        artifacts.cubin("k", "sm_100a"),
        Err(LoadError::DigestMismatch { .. })
    ));
}

#[test]
fn truncated_image_is_refused() {
    let dir = TempDir::new("short");
    std::fs::write(dir.0.join("cubin/k.sm_100a.cubin"), &CUBIN[1..]).expect("write");
    let manifest = manifest();
    let artifacts = ArtifactDir::new(&dir.0, &manifest);
    assert!(matches!(
        artifacts.cubin("k", "sm_100a"),
        Err(LoadError::SizeMismatch { .. })
    ));
}

#[test]
fn missing_and_unknown_images() {
    let dir = TempDir::new("missing");
    let manifest = manifest();
    let artifacts = ArtifactDir::new(&dir.0, &manifest);
    assert!(matches!(
        artifacts.cubin("k", "sm_100a"),
        Err(LoadError::Io { .. })
    ));
    assert!(matches!(
        artifacts.cubin("k", "sm_103a"),
        Err(LoadError::Unknown(_))
    ));
    assert!(matches!(
        artifacts.fatbin("nope"),
        Err(LoadError::Unknown(_))
    ));
}

#[test]
fn manifest_rejects_escaping_paths_and_bad_digests() {
    let good = serde_json::json!({
        "schema_version": 2,
        "toolchain": { "nixpkgs": "", "cuda": "", "host_compiler": "", "components": {} },
        "sources": {},
        "compile": { "per_source_flags": [], "common_flags": [], "profiles": {
            "own": { "flags": [], "includes": [] } } },
        "inputs": [],
        "kernels": [{
            "name": "k", "arch": "sm_100a", "source": "s", "profile": "own",
            "file": "cubin/k", "sha256": sha256_hex(b""), "size": 0,
            "entries": [{ "symbol": "k", "demangled": "k", "meta": "k_meta" }],
            "meta": ["k_meta"], "mma_sass": {}
        }],
        "fatbins": []
    });
    assert!(Manifest::parse(&good.to_string()).is_ok());

    let mut escaping = good.clone();
    escaping["kernels"][0]["file"] = "../../etc/passwd".into();
    assert!(Manifest::parse(&escaping.to_string()).is_err());

    let mut absolute = good.clone();
    absolute["kernels"][0]["file"] = "/etc/passwd".into();
    assert!(Manifest::parse(&absolute.to_string()).is_err());

    let mut digest = good.clone();
    digest["kernels"][0]["sha256"] = "ABC".into();
    assert!(Manifest::parse(&digest.to_string()).is_err());

    let mut unknown_field = good;
    unknown_field["build_host"] = "builder-7".into();
    assert!(Manifest::parse(&unknown_field.to_string()).is_err());
}

/// A manifest whose entries and launch-contract records do not correspond one
/// to one is refused: a record no entry claims, an entry without a record, two
/// entries claiming one record, or a duplicated symbol.
#[test]
fn manifest_requires_one_record_per_entry() {
    let good = serde_json::json!({
        "schema_version": 2,
        "toolchain": { "nixpkgs": "", "cuda": "", "host_compiler": "", "components": {} },
        "sources": {},
        "compile": { "per_source_flags": [], "common_flags": [], "profiles": {
            "own": { "flags": [], "includes": [] } } },
        "inputs": [],
        "kernels": [{
            "name": "k", "arch": "sm_100a", "source": "s", "profile": "own",
            "file": "cubin/k", "sha256": sha256_hex(b""), "size": 0,
            "entries": [
                { "symbol": "_Z1kv", "demangled": "k()", "meta": "k_alias_meta" },
                { "symbol": "plain", "demangled": "plain", "meta": "plain_meta" }
            ],
            "meta": ["k_alias_meta", "plain_meta"], "mma_sass": {}
        }],
        "fatbins": []
    });
    let manifest = Manifest::parse(&good.to_string()).expect("valid");
    let cubin = manifest.cubin("k", "sm_100a").expect("cubin");
    assert_eq!(cubin.entry("_Z1kv").expect("entry").meta, "k_alias_meta");
    assert_eq!(
        cubin.entry_for_meta("k_alias_meta").expect("entry").symbol,
        "_Z1kv"
    );
    assert!(cubin.entry("nope").is_none());

    let mut orphan = good.clone();
    orphan["kernels"][0]["meta"] = serde_json::json!(["k_alias_meta", "plain_meta", "x_meta"]);
    assert!(
        Manifest::parse(&orphan.to_string()).is_err(),
        "orphan record"
    );

    let mut unbound = good.clone();
    unbound["kernels"][0]["meta"] = serde_json::json!(["plain_meta"]);
    assert!(
        Manifest::parse(&unbound.to_string()).is_err(),
        "unbacked record"
    );

    let mut shared = good.clone();
    shared["kernels"][0]["entries"][0]["meta"] = "plain_meta".into();
    assert!(
        Manifest::parse(&shared.to_string()).is_err(),
        "shared record"
    );

    let mut dup_record = good.clone();
    dup_record["kernels"][0]["meta"] =
        serde_json::json!(["k_alias_meta", "plain_meta", "plain_meta"]);
    assert!(
        Manifest::parse(&dup_record.to_string()).is_err(),
        "duplicate record"
    );

    let mut dup_symbol = good;
    dup_symbol["kernels"][0]["entries"][1]["symbol"] = "_Z1kv".into();
    assert!(
        Manifest::parse(&dup_symbol.to_string()).is_err(),
        "duplicate symbol"
    );
}

#[test]
fn kernel_meta_decodes_little_endian_words() {
    let words: [u32; 8] = [256, 1, 1, 213_820, 2, 1, 1, 0];
    let mut bytes = [0u8; KernelMeta::SIZE];
    for (i, w) in words.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    assert_eq!(
        KernelMeta::from_bytes(&bytes),
        KernelMeta {
            block: [256, 1, 1],
            dynamic_smem_bytes: 213_820,
            cluster: [2, 1, 1],
            params_bytes: 0,
        }
    );
}
