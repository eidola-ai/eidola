//! Generates `$OUT_DIR/trust_root.gen.rs` with the pinned trust-root constants
//! the client embeds at build time.
//!
//! Inputs (read from the workspace root):
//!   * `releases/trust/server-enclave.json` — the paired-server enclave
//!     measurement, isolated from `artifact-manifest.json` so the per-artifact
//!     digests (including this crate's own) can't cache-bust the build.
//!   * `releases/trust/trust-constants.json` — non-derivable trust values
//!     (attestant fingerprints, CI identity, schema version pins, update
//!     discovery URLs, server URL template).
//!   * `releases/trust/sigstore-trusted-root.json` — pinned Sigstore tlog /
//!     Fulcio / CT log keys (snapshot of the upstream TrustedRoot).
//!   * `releases/schema/attestation-templates.json` — pinned claim
//!     templates the verifier re-renders during equality checks.
//!
//!   * `models/<family>/<label>.toml` — the model catalog the gateway
//!     serves, read by the gateway's own reader
//!     (`eidola_common::model_catalog`) and compiled in as
//!     `$OUT_DIR/model_catalog.gen.rs`, so the client can say what a model
//!     choice resolves to, when a tag has moved, and when a model retires,
//!     without asking anyone.
//!
//! Each input file is rerun-watched; the generator validates that everything
//! parses as JSON before emitting. The server URL is derived as
//! `template.replace("{hash}", sha256(snp_measurement_bytes ||
//! host_data_bytes)[..hash_length_hex_chars])`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root above crates/eidola-app-core")
        .to_path_buf();

    let server_enclave_path = workspace_root.join("releases/trust/server-enclave.json");
    let trust_constants_path = workspace_root.join("releases/trust/trust-constants.json");
    let sigstore_trusted_root_path =
        workspace_root.join("releases/trust/sigstore-trusted-root.json");
    let attestation_templates_path =
        workspace_root.join("releases/schema/attestation-templates.json");

    for p in [
        &server_enclave_path,
        &trust_constants_path,
        &sigstore_trusted_root_path,
        &attestation_templates_path,
    ] {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    let server_enclave = read_json(&server_enclave_path);
    let trust_constants = read_json(&trust_constants_path);

    // Validate sibling JSON files parse, but include the raw bytes verbatim
    // so the on-disk file is the canonical form (no re-serialization drift).
    let _ = read_json(&sigstore_trusted_root_path);

    // The templates file must load through the same code the verifier
    // uses at runtime — schema_version pin, deny_unknown_fields, all of
    // it — so a file/loader mismatch fails this build instead of every
    // release verification.
    let templates_json = fs::read_to_string(&attestation_templates_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", attestation_templates_path.display()));
    eidola_attestation::load_from_str(&templates_json).unwrap_or_else(|e| {
        panic!(
            "{} is not loadable by the pinned verifier: {e:#}",
            attestation_templates_path.display()
        )
    });

    let enclave_schema_version = get_u32(&server_enclave, "schema_version");
    if enclave_schema_version != 2 {
        panic!(
            "server-enclave.json schema_version `{enclave_schema_version}` is not supported by this build.rs (expected `2`)"
        );
    }
    exact_keys(
        &server_enclave,
        &["cvm_version", "igvm_sha256", "schema_version", "sev_snp"],
        "server-enclave.json",
    );
    let cvm_version = get_str(&server_enclave, "cvm_version");
    if cvm_version
        .split_once("@sha256:")
        .is_none_or(|(version, pin)| {
            version.is_empty() || version.contains('+') || !is_lower_hex(pin, 64)
        })
    {
        panic!(
            "server-enclave.json `cvm_version` must be a CPU image release with its manifest \
             pinned (VERSION@sha256:<64 lowercase hex>), got {cvm_version:?}"
        );
    }
    let igvm_sha256 = get_hex_str(&server_enclave, "igvm_sha256", 64);
    let sev_snp = server_enclave
        .get("sev_snp")
        .expect("server-enclave.json missing `sev_snp`");
    exact_keys(
        sev_snp,
        &["guest_policy", "host_data", "measurement"],
        "server-enclave.json sev_snp",
    );
    let snp = get_hex_str(sev_snp, "measurement", 96);
    let host_data = get_hex_str(sev_snp, "host_data", 64);
    // Its spelling and safety bits are checked by the verifier's pin
    // compiler when the client builds its pins; here, only that it is one.
    let guest_policy = get_str(sev_snp, "guest_policy");
    if !guest_policy.starts_with("0x") {
        panic!("server-enclave.json sev_snp.guest_policy must be 0x-prefixed hex");
    }

    let url_template = get_str(&trust_constants, "server_url_template");
    let url_hash_length = trust_constants
        .get("server_url_hash_length")
        .and_then(|v| v.as_u64())
        .expect("server_url_hash_length missing or not a number")
        as usize;
    if !url_template.contains("{hash}") {
        panic!("server_url_template must contain `{{hash}}` placeholder");
    }
    let server_url_hash = derive_url_hash(&snp, &host_data, url_hash_length);
    let server_url = url_template.replace("{hash}", &server_url_hash);

    let expected_ci_identity_pattern = get_str(&trust_constants, "expected_ci_identity_pattern");
    let expected_ci_issuer = get_str(&trust_constants, "expected_ci_issuer");
    let update_discovery_url = get_str(&trust_constants, "update_discovery_url");

    let trusted_attestant_fingerprints =
        get_str_array(&trust_constants, "trusted_attestant_fingerprints");
    for fp in &trusted_attestant_fingerprints {
        if fp.len() != 64 || !fp.chars().all(|c| c.is_ascii_hexdigit()) {
            panic!("trusted_attestant_fingerprint must be 64 lowercase hex chars: {fp:?}");
        }
    }
    let min_human_attestations = get_u32(&trust_constants, "min_human_attestations");
    if min_human_attestations == 0 {
        panic!("min_human_attestations must be ≥1 (got 0)");
    }
    let supported_release_schema_versions =
        get_u32_array(&trust_constants, "supported_release_schema_versions");
    let supported_attestation_schema_versions =
        get_u32_array(&trust_constants, "supported_attestation_schema_versions");

    let mut out = String::new();
    out.push_str("// @generated by build.rs — do not edit.\n\n");

    push_str_const(&mut out, "SERVER_URL", &server_url);
    push_str_const(&mut out, "SERVER_URL_HASH", &server_url_hash);
    push_str_const(&mut out, "SERVER_CVM_VERSION", cvm_version);
    push_str_const(&mut out, "SERVER_IGVM_SHA256", &igvm_sha256);
    push_str_const(&mut out, "SERVER_SNP_MEASUREMENT", &snp);
    push_str_const(&mut out, "SERVER_SNP_HOST_DATA", &host_data);
    push_str_const(&mut out, "SERVER_SNP_GUEST_POLICY", guest_policy);
    push_str_const(
        &mut out,
        "EXPECTED_CI_IDENTITY_PATTERN",
        expected_ci_identity_pattern,
    );
    push_str_const(&mut out, "EXPECTED_CI_ISSUER", expected_ci_issuer);
    push_str_const(&mut out, "UPDATE_DISCOVERY_URL", update_discovery_url);

    push_str_slice_const(
        &mut out,
        "TRUSTED_ATTESTANT_FINGERPRINTS",
        &trusted_attestant_fingerprints,
    );
    out.push_str(&format!(
        "pub const MIN_HUMAN_ATTESTATIONS: u32 = {min_human_attestations};\n"
    ));
    push_u32_slice_const(
        &mut out,
        "SUPPORTED_RELEASE_SCHEMA_VERSIONS",
        &supported_release_schema_versions,
    );
    push_u32_slice_const(
        &mut out,
        "SUPPORTED_ATTESTATION_SCHEMA_VERSIONS",
        &supported_attestation_schema_versions,
    );

    out.push_str(&format!(
        "pub const ATTESTATION_TEMPLATES_JSON: &str = include_str!({:?});\n",
        attestation_templates_path
    ));
    out.push_str(&format!(
        "pub const SIGSTORE_TRUSTED_ROOT_JSON: &str = include_str!({:?});\n",
        sigstore_trusted_root_path
    ));

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_dir.join("trust_root.gen.rs"), out).expect("write trust_root.gen.rs");
    fs::write(
        out_dir.join("model_catalog.gen.rs"),
        model_catalog(&workspace_root),
    )
    .expect("write model_catalog.gen.rs");
}

fn read_json(path: &Path) -> serde_json::Value {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn get_str<'a>(value: &'a serde_json::Value, key: &str) -> &'a str {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("missing string field `{key}`"))
}

fn get_str_array(value: &serde_json::Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("missing array field `{key}`"))
        .iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("`{key}` entries must be strings"))
                .to_string()
        })
        .collect()
}

fn get_u32(value: &serde_json::Value, key: &str) -> u32 {
    value
        .get(key)
        .and_then(|v| v.as_u64())
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or_else(|| panic!("missing or out-of-range u32 field `{key}`"))
}

fn get_u32_array(value: &serde_json::Value, key: &str) -> Vec<u32> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("missing array field `{key}`"))
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .unwrap_or_else(|| panic!("`{key}` entries must be u32 integers"))
        })
        .collect()
}

fn get_hex_str(value: &serde_json::Value, key: &str, expected_len: usize) -> String {
    let s = get_str(value, key).to_ascii_lowercase();
    if s.len() != expected_len || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        panic!("`{key}` must be {expected_len} lowercase hex chars, got {s:?}");
    }
    s
}

fn is_lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Refuse a member this build does not read: an unread field of the trust
/// root is a claim nobody enforces.
fn exact_keys(value: &serde_json::Value, keys: &[&str], at: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{at} must be an object"));
    let mut found: Vec<&str> = object.keys().map(String::as_str).collect();
    found.sort_unstable();
    if found != keys {
        panic!("{at} must have exactly the members {keys:?}, found {found:?}");
    }
}

/// The URL hash: the launch digest says which image, HOST_DATA which
/// deployment of it (the digest alone is the same for every deployment of a
/// release).
fn derive_url_hash(snp: &str, host_data: &str, hash_len: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(decode_hex(snp));
    hasher.update(decode_hex(host_data));
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        write!(hex, "{b:02x}").unwrap();
    }
    hex.truncate(hash_len);
    hex
}

fn decode_hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

fn push_str_const(out: &mut String, name: &str, value: &str) {
    out.push_str(&format!("pub const {name}: &str = {value:?};\n"));
}

fn push_str_slice_const(out: &mut String, name: &str, values: &[String]) {
    out.push_str(&format!("pub const {name}: &[&str] = &["));
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{v:?}"));
    }
    out.push_str("];\n");
}

fn push_u32_slice_const(out: &mut String, name: &str, values: &[u32]) {
    out.push_str(&format!("pub const {name}: &[u32] = &["));
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{v}"));
    }
    out.push_str("];\n");
}

/// The model catalog under `models/`, read and checked by the gateway's own
/// reader, rendered as `CATALOG: &[CatalogRow]` (the types live in
/// `src/model_catalog.rs`). Only what the client shows and reasons about is
/// compiled in: identity, tags, display name, provenance and manifest, and
/// lifecycle. The gateway's build additionally binds the rows to its engine
/// pins; the client has no pins, so that check is the gateway's alone.
fn model_catalog(workspace_root: &Path) -> String {
    use eidola_common::model_catalog::{self as catalog, Lifecycle};
    use std::fmt::Write as _;

    let root = workspace_root.join(catalog::MODELS_ROOT);
    println!("cargo:rerun-if-changed={}", root.display());
    let sorted = |dir: &Path| -> Vec<fs::DirEntry> {
        let Ok(read) = fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut list: Vec<fs::DirEntry> = read
            .map(|e| e.unwrap_or_else(|e| panic!("list {}: {e}", dir.display())))
            .collect();
        list.sort_by_key(|e| e.file_name());
        list
    };
    let name = |e: &fs::DirEntry| {
        e.file_name()
            .into_string()
            .unwrap_or_else(|n| panic!("{n:?} under models/ is not UTF-8"))
    };
    let mut rows = Vec::new();
    for family in sorted(&root) {
        let family_name = name(&family);
        if !family.path().is_dir() {
            continue;
        }
        println!("cargo:rerun-if-changed={}", family.path().display());
        for file in sorted(&family.path()) {
            let relative = format!("{}/{family_name}/{}", catalog::MODELS_ROOT, name(&file));
            println!("cargo:rerun-if-changed={}", file.path().display());
            let bytes = fs::read(file.path()).unwrap_or_else(|e| panic!("read {relative}: {e}"));
            rows.push(
                catalog::parse(&relative, &bytes)
                    .unwrap_or_else(|e| panic!("the model catalog is not valid: {e}")),
            );
        }
    }
    catalog::check_catalog(&rows).unwrap_or_else(|e| panic!("the model catalog is not valid: {e}"));

    let mut out = String::from("// @generated by build.rs from models/ — do not edit.\n");
    out.push_str("pub const CATALOG: &[CatalogRow] = &[\n");
    for row in &rows {
        let lifecycle = match &row.lifecycle {
            Lifecycle::Current => "Lifecycle::Current".to_string(),
            Lifecycle::Deprecated { retire_after } => {
                format!("Lifecycle::Deprecated {{ retire_after: {retire_after:?} }}")
            }
            Lifecycle::Retired { retire_after } => {
                format!("Lifecycle::Retired {{ retire_after: {retire_after:?} }}")
            }
        };
        let files: Vec<String> = row
            .files
            .iter()
            .map(|f| {
                format!(
                    "WeightsFile {{ path: {:?}, size: {}, sha256: {:?} }}",
                    f.path, f.size, f.sha256
                )
            })
            .collect();
        writeln!(
            out,
            "    CatalogRow {{ sha256: {:?}, tags: &{:?}, name: {:?}, repo: {:?}, \
             revision: {:?}, files: &[{}], lifecycle: {} }},",
            row.sha256,
            row.tags,
            row.name,
            row.repo,
            row.revision,
            files.join(", "),
            lifecycle,
        )
        .unwrap();
    }
    out.push_str("];\n");
    out
}
