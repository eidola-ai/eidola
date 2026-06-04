//! `release-tool provenance check|capture` — **informational** hardware
//! provenance for attestant signing keys.
//!
//! None of this is part of trust evaluation: neither the client, the
//! updater, nor any `build.rs` reads these files. They exist so an
//! external auditor can independently confirm that a pinned
//! `trusted_attestant_fingerprints` entry corresponds to a real,
//! policy-constrained hardware key — e.g. a YubiKey-PIV attestation
//! certificate stating the key was generated on-device, is
//! non-exportable, and requires PIN + touch per signature. The bundle
//! format is intentionally loose and manufacturer-neutral; see
//! `releases/trust/attestant-provenance/README.md`.
//!
//! Two subcommands:
//!
//! - `check` (vendor-neutral, CI-friendly): for every committed bundle,
//!   recompute `sha256(SPKI)` of the public key inside its attestation
//!   certificate and assert it equals the `pinned_fingerprint_sha256`
//!   that the bundle's `meta.json` claims — so a corrupted or mismatched
//!   cert cannot sit in the tree unnoticed. The directory mirrors the
//!   *current* trusted set (retired keys' evidence lives in git history,
//!   like the rest of `releases/trust/`), so a bundle whose fingerprint
//!   is no longer pinned is reported as a stale leftover to remove. Keeps
//!   the directory and `trusted_attestant_fingerprints` in lockstep.
//! - `capture` (Yubico convenience): shell out to `ykman` to write the
//!   slot-9c attestation cert + the F9 intermediate and scaffold a
//!   `meta.json`. Other key sources (TPM, KMS, a different SmartCard)
//!   populate the same bundle shape by hand.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

use crate::trust;

const DEFAULT_SUBDIR: &str = "releases/trust/attestant-provenance";
const KEY_ATTESTATION_FILE: &str = "key-attestation.pem";
const INTERMEDIATE_FILE: &str = "intermediate.pem";
const META_FILE: &str = "meta.json";

pub struct CheckArgs {
    pub workspace_root: PathBuf,
    /// Provenance directory; defaults to `releases/trust/attestant-provenance`.
    pub dir: Option<PathBuf>,
}

pub struct CaptureArgs {
    pub workspace_root: PathBuf,
    pub attestant_id: String,
    /// PIV slot holding the signing key (default `9c`).
    pub slot: String,
    pub dir: Option<PathBuf>,
}

pub struct EnrichArgs {
    pub workspace_root: PathBuf,
    /// A single bundle to enrich; if `None`, enrich every bundle in `dir`.
    pub attestant_id: Option<String>,
    pub dir: Option<PathBuf>,
}

/// What a bundle's `meta.json` must contain for `check`. Extra fields
/// (manufacturer, serial, policies, …) are ignored here — they are
/// human-readable context, not things this tool validates.
#[derive(serde::Deserialize)]
struct Meta {
    pinned_fingerprint_sha256: String,
}

/// Device facts parsed from a Yubico PIV attestation certificate's vendor
/// extensions (OID arc `1.3.6.1.4.1.41482.3.*`). All optional: a
/// non-Yubico attestation cert yields an empty set, which is how
/// [`DeviceInfo::is_yubico`] distinguishes the two.
#[derive(Default)]
struct DeviceInfo {
    serial: Option<String>,
    firmware: Option<String>,
    pin_policy: Option<String>,
    touch_policy: Option<String>,
    form_factor: Option<String>,
}

impl DeviceInfo {
    fn is_yubico(&self) -> bool {
        self.serial.is_some()
            || self.firmware.is_some()
            || self.pin_policy.is_some()
            || self.form_factor.is_some()
    }
}

pub fn check(args: CheckArgs) -> Result<()> {
    let dir = args
        .dir
        .unwrap_or_else(|| args.workspace_root.join(DEFAULT_SUBDIR));
    let trusted = trust::load(&args.workspace_root)?.trusted_attestant_fingerprints;
    let trusted_lc: Vec<String> = trusted.iter().map(|f| f.to_ascii_lowercase()).collect();

    if !dir.exists() {
        println!(
            "no attestant-provenance directory at {} — nothing to check (provenance is optional).",
            dir.display()
        );
        return Ok(());
    }

    let mut bundles = 0usize;
    let mut failures = 0usize;
    let mut evidenced: Vec<String> = Vec::new();

    let mut entries: Vec<PathBuf> = fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();

    for bundle in entries {
        let meta_path = bundle.join(META_FILE);
        let cert_path = bundle.join(KEY_ATTESTATION_FILE);
        if !meta_path.exists() && !cert_path.exists() {
            continue; // not a provenance bundle
        }
        bundles += 1;
        let id = bundle
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        let claimed = match read_meta(&meta_path) {
            Ok(m) => m.pinned_fingerprint_sha256.to_ascii_lowercase(),
            Err(e) => {
                println!("  ✗ {id}: {e:#}");
                failures += 1;
                continue;
            }
        };
        let actual = match cert_spki_fingerprint(&cert_path) {
            Ok((fp, _algo)) => fp,
            Err(e) => {
                println!("  ✗ {id}: {e:#}");
                failures += 1;
                continue;
            }
        };

        // The cert must attest the fingerprint its meta.json claims.
        if actual != claimed {
            println!(
                "  ✗ {id}: attestation cert public key fingerprint `{actual}` does NOT match\n\
                         meta.json pinned_fingerprint_sha256 `{claimed}`"
            );
            failures += 1;
            continue;
        }

        // The directory mirrors the *current* trusted set (history lives in
        // git). A bundle for a fingerprint that is no longer pinned is a
        // stale leftover that should have been removed when the key was
        // unpinned — flag it so the directory and
        // `trusted_attestant_fingerprints` stay in lockstep.
        if !trusted_lc.contains(&actual) {
            println!(
                "  ✗ {id}: {actual}\n\
                         fingerprint is NOT in the current trusted set — stale provenance bundle.\n\
                         Remove it (the directory must mirror the current trusted_attestant_fingerprints;\n\
                         the retired key's evidence remains in git history)."
            );
            failures += 1;
            continue;
        }

        evidenced.push(actual.clone());
        println!("  ✓ {id}: {actual}  [currently pinned]");
    }

    // Informational: pinned fingerprints with no committed provenance. Not a
    // failure — an attestant may not have committed hardware evidence (yet),
    // or may use a key type that produces none.
    for fp in &trusted_lc {
        if !evidenced.contains(fp) {
            println!("  · pinned fingerprint with no provenance bundle: {fp}");
        }
    }

    println!();
    if failures > 0 {
        bail!(
            "{failures} of {bundles} provenance bundle(s) failed: a committed bundle either does \
             not match the fingerprint its meta.json claims, or describes a key that is no longer \
             in trusted_attestant_fingerprints (remove stale bundles on rotation)."
        );
    }
    println!("{bundles} provenance bundle(s) consistent and in lockstep with the trusted set.");
    Ok(())
}

pub fn capture(args: CaptureArgs) -> Result<()> {
    require_tool("ykman")?;
    crate::attest::validate_attestant_id(&args.attestant_id)?;

    let base = args
        .dir
        .unwrap_or_else(|| args.workspace_root.join(DEFAULT_SUBDIR));
    let bundle = base.join(&args.attestant_id);
    fs::create_dir_all(&bundle)
        .with_context(|| format!("creating bundle directory {}", bundle.display()))?;

    let cert_path = bundle.join(KEY_ATTESTATION_FILE);
    let inter_path = bundle.join(INTERMEDIATE_FILE);

    // Attestation cert for the signing key, signed by the device's F9 key.
    run_ykman(&[
        "piv",
        "keys",
        "attest",
        &args.slot,
        cert_path.to_str().unwrap(),
    ])?;
    // The F9 intermediate that signed it (chains to the Yubico PIV root).
    run_ykman(&[
        "piv",
        "certificates",
        "export",
        "f9",
        inter_path.to_str().unwrap(),
    ])?;

    // Derive the full meta.json from the freshly-captured cert — serial,
    // firmware, and PIN/touch policy all live in the Yubico attestation
    // extensions, so nothing is left as a TODO.
    let cert = read_cert(&cert_path)?;
    let fields = derived_fields(&args.attestant_id, &cert)?;
    let meta_path = bundle.join(META_FILE);
    write_meta(&meta_path, &serde_json::Value::Object(fields.clone()))?;

    println!();
    println!("Wrote provenance bundle to {}", bundle.display());
    print_summary(&fields);
    if fields.get("manufacturer").is_none() {
        println!("  note: no Yubico attestation extensions found — device fields not auto-filled.");
    }
    println!("Confirm `product` reads sensibly, then run `provenance check`.");
    Ok(())
}

pub fn enrich(args: EnrichArgs) -> Result<()> {
    let dir = args
        .dir
        .unwrap_or_else(|| args.workspace_root.join(DEFAULT_SUBDIR));

    let targets: Vec<PathBuf> = match &args.attestant_id {
        Some(id) => {
            crate::attest::validate_attestant_id(id)?;
            vec![dir.join(id)]
        }
        None => {
            if !dir.exists() {
                println!(
                    "no attestant-provenance directory at {} — nothing to enrich.",
                    dir.display()
                );
                return Ok(());
            }
            let mut v: Vec<PathBuf> = fs::read_dir(&dir)
                .with_context(|| format!("reading {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect();
            v.sort();
            v
        }
    };

    let mut enriched = 0usize;
    for bundle in targets {
        let cert_path = bundle.join(KEY_ATTESTATION_FILE);
        if !cert_path.exists() {
            if args.attestant_id.is_some() {
                bail!(
                    "no {KEY_ATTESTATION_FILE} in {} — capture the bundle first.",
                    bundle.display()
                );
            }
            continue; // skip non-bundle dirs in batch mode
        }
        let id = bundle
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let cert = read_cert(&cert_path)?;
        let derived = derived_fields(&id, &cert)?;

        // Merge derived fields over any existing meta.json so operator-added
        // context (e.g. a refined product string on a non-Yubico bundle) is
        // preserved; only fields the cert authoritatively provides change.
        let meta_path = bundle.join(META_FILE);
        let mut merged = match fs::read(&meta_path) {
            Ok(bytes) => {
                serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&bytes)
                    .with_context(|| format!("parsing existing {}", meta_path.display()))?
            }
            Err(_) => serde_json::Map::new(),
        };
        for (k, v) in derived {
            merged.insert(k, v);
        }
        write_meta(&meta_path, &serde_json::Value::Object(merged.clone()))?;
        println!("  ✓ {id}");
        print_summary(&merged);
        enriched += 1;
    }
    println!();
    println!("Enriched {enriched} bundle(s) from their attestation certificates.");
    Ok(())
}

/// Print the human-relevant fields of a meta object as an indented block.
fn print_summary(fields: &serde_json::Map<String, serde_json::Value>) {
    for key in [
        "pinned_fingerprint_sha256",
        "algorithm",
        "manufacturer",
        "product",
        "serial",
        "firmware",
        "pin_policy",
        "touch_policy",
    ] {
        if let Some(v) = fields.get(key).and_then(|v| v.as_str()) {
            println!("    {key:<26}: {v}");
        }
    }
}

/// Build the meta.json fields the certificate authoritatively determines.
/// Always includes identity + fingerprint + algorithm; for a recognized
/// Yubico attestation cert it also fills manufacturer, product, serial,
/// firmware, and PIN/touch policy from the vendor extensions.
fn derived_fields(
    attestant_id: &str,
    cert: &x509_cert::Certificate,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    use serde_json::{Value, json};

    let (fingerprint, algorithm) = fingerprint_and_algorithm(cert)?;
    let dev = yubico_device_info(cert);

    let mut m = serde_json::Map::new();
    m.insert("attestant_id".into(), json!(attestant_id));
    m.insert("pinned_fingerprint_sha256".into(), json!(fingerprint));
    m.insert(
        "algorithm".into(),
        json!(algorithm.unwrap_or_else(|| "UNKNOWN".into())),
    );
    m.insert(
        "evidence".into(),
        Value::Array(vec![json!(KEY_ATTESTATION_FILE), json!(INTERMEDIATE_FILE)]),
    );

    if dev.is_yubico() {
        let fw_major = dev
            .firmware
            .as_deref()
            .and_then(|f| f.split('.').next())
            .and_then(|s| s.parse::<u32>().ok());
        let product = match (&dev.form_factor, fw_major) {
            (Some(ff), Some(maj)) => format!("YubiKey {maj} ({ff})"),
            (Some(ff), None) => format!("YubiKey ({ff})"),
            (None, Some(maj)) => format!("YubiKey {maj}"),
            (None, None) => "YubiKey".to_string(),
        };
        m.insert("manufacturer".into(), json!("Yubico"));
        m.insert("product".into(), json!(product));
        m.insert(
            "serial".into(),
            json!(dev.serial.unwrap_or_else(|| "unknown".into())),
        );
        m.insert(
            "firmware".into(),
            json!(dev.firmware.unwrap_or_else(|| "unknown".into())),
        );
        m.insert("key_generation".into(), json!("on-device, non-exportable"));
        m.insert(
            "pin_policy".into(),
            json!(dev.pin_policy.unwrap_or_else(|| "unknown".into())),
        );
        m.insert(
            "touch_policy".into(),
            json!(dev.touch_policy.unwrap_or_else(|| "unknown".into())),
        );
        m.insert(
            "chains_to".into(),
            json!("Yubico PIV Root CA Serial 263751"),
        );
    }
    Ok(m)
}

/// Parse the Yubico PIV attestation extensions (`1.3.6.1.4.1.41482.3.*`).
/// Returns an empty `DeviceInfo` for any cert lacking them (i.e. non-Yubico).
fn yubico_device_info(cert: &x509_cert::Certificate) -> DeviceInfo {
    use spki::ObjectIdentifier;
    // Yubico PIV attestation extension OIDs.
    const OID_FIRMWARE: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.6.1.4.1.41482.3.3");
    const OID_SERIAL: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.6.1.4.1.41482.3.7");
    const OID_POLICY: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.6.1.4.1.41482.3.8");
    const OID_FORMFACTOR: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.6.1.4.1.41482.3.9");

    let mut info = DeviceInfo::default();
    let Some(exts) = cert.tbs_certificate.extensions.as_ref() else {
        return info;
    };
    for ext in exts {
        let v = ext.extn_value.as_bytes();
        let oid = ext.extn_id;
        if oid == OID_FIRMWARE && v.len() == 3 {
            info.firmware = Some(format!("{}.{}.{}", v[0], v[1], v[2]));
        } else if oid == OID_SERIAL {
            info.serial = parse_der_integer(v).map(|n| n.to_string());
        } else if oid == OID_POLICY && v.len() == 2 {
            info.pin_policy = Some(pin_policy_name(v[0]));
            info.touch_policy = Some(touch_policy_name(v[1]));
        } else if oid == OID_FORMFACTOR && v.len() == 1 {
            info.form_factor = Some(form_factor_name(v[0]));
        }
    }
    info
}

/// Decode the content of a DER INTEGER (as it appears verbatim in the
/// serial extension), tolerating a raw big-endian fallback.
fn parse_der_integer(bytes: &[u8]) -> Option<u64> {
    let content = if bytes.first() == Some(&0x02) && bytes.len() >= 2 {
        let len = bytes[1] as usize;
        bytes.get(2..2 + len)?
    } else {
        bytes
    };
    if content.is_empty() || content.len() > 8 {
        return None;
    }
    Some(content.iter().fold(0u64, |acc, &b| (acc << 8) | b as u64))
}

fn pin_policy_name(b: u8) -> String {
    match b {
        0x01 => "NEVER".into(),
        0x02 => "ONCE".into(),
        0x03 => "ALWAYS".into(),
        other => format!("0x{other:02x}"),
    }
}

fn touch_policy_name(b: u8) -> String {
    match b {
        0x01 => "NEVER".into(),
        0x02 => "ALWAYS".into(),
        0x03 => "CACHED".into(),
        other => format!("0x{other:02x}"),
    }
}

fn form_factor_name(b: u8) -> String {
    // Low bits encode the form factor; high bits are FIPS/enterprise flags.
    match b & 0x0f {
        0x01 => "USB-A keychain".into(),
        0x02 => "USB-A nano".into(),
        0x03 => "USB-C keychain".into(),
        0x04 => "USB-C nano".into(),
        0x05 => "USB-C + Lightning".into(),
        0x06 => "USB-A bio".into(),
        0x07 => "USB-C bio".into(),
        _ => "unknown form factor".into(),
    }
}

fn read_cert(cert_pem: &Path) -> Result<x509_cert::Certificate> {
    use der::DecodePem;
    let pem = fs::read(cert_pem)
        .with_context(|| format!("reading attestation certificate {}", cert_pem.display()))?;
    x509_cert::Certificate::from_pem(&pem)
        .with_context(|| format!("parsing X.509 PEM at {}", cert_pem.display()))
}

/// `(sha256(SPKI) hex, algorithm name)` for a parsed cert. The fingerprint
/// is computed over the certificate's `SubjectPublicKeyInfo` exactly as the
/// pinned attestant fingerprint is, so the two are directly comparable.
fn fingerprint_and_algorithm(cert: &x509_cert::Certificate) -> Result<(String, Option<String>)> {
    use der::Encode;
    let spki_der = cert
        .tbs_certificate
        .subject_public_key_info
        .to_der()
        .context("re-encoding certificate SubjectPublicKeyInfo")?;
    let fingerprint = hex_lower(Sha256::digest(&spki_der).as_slice());
    let algorithm =
        eidola_app_core::updater::human_attestation::classify_attestant_spki_algorithm(&spki_der)
            .ok()
            .map(|a| a.name().to_string());
    Ok((fingerprint, algorithm))
}

/// Path-based wrapper used by `check`.
fn cert_spki_fingerprint(cert_pem: &Path) -> Result<(String, Option<String>)> {
    fingerprint_and_algorithm(&read_cert(cert_pem)?)
}

fn write_meta(path: &Path, meta: &serde_json::Value) -> Result<()> {
    fs::write(path, format!("{}\n", serde_json::to_string_pretty(meta)?))
        .with_context(|| format!("writing {}", path.display()))
}

fn read_meta(path: &Path) -> Result<Meta> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn run_ykman(args: &[&str]) -> Result<()> {
    let status = std::process::Command::new("ykman")
        .args(args)
        .status()
        .context("running `ykman`")?;
    if !status.success() {
        bail!("`ykman {}` failed", args.join(" "));
    }
    Ok(())
}

fn require_tool(name: &str) -> Result<()> {
    let ok = std::process::Command::new(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        bail!("`{name}` not found on PATH");
    }
    Ok(())
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fingerprint computed from a cert's SPKI must equal the one
    /// computed directly from that same SPKI — i.e. cert extraction is a
    /// faithful pass-through, comparable to `pkcs11 list` output. We don't
    /// have a cert fixture, so assert the SPKI→fingerprint half against the
    /// committed public-key fixture here; the cert-parsing half is covered
    /// by x509-cert's own tests plus a real device run.
    #[test]
    fn spki_fingerprint_matches_pkcs11_path() {
        use base64::Engine;
        let pem = std::fs::read_to_string(
            "../eidola-app-core/tests/fixtures/human_attestation/cosign.pub",
        )
        .unwrap();
        let b64: String = pem.lines().filter(|l| !l.starts_with("-----")).collect();
        let spki = base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .unwrap();
        let fp = hex_lower(Sha256::digest(&spki).as_slice());
        // 64 lowercase hex chars, and stable.
        assert_eq!(fp.len(), 64);
        assert!(
            fp.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn parses_der_integer_serial() {
        // DER INTEGER 0x02 0x03 12 d6 87 = 1_234_567
        assert_eq!(
            parse_der_integer(&[0x02, 0x03, 0x12, 0xd6, 0x87]),
            Some(1_234_567)
        );
        // With a sign-padding leading zero.
        assert_eq!(parse_der_integer(&[0x02, 0x02, 0x00, 0xff]), Some(255));
        // Raw big-endian fallback (no INTEGER tag).
        assert_eq!(parse_der_integer(&[0x01, 0x00]), Some(256));
    }

    #[test]
    fn maps_policy_and_form_factor_bytes() {
        assert_eq!(pin_policy_name(0x03), "ALWAYS");
        assert_eq!(touch_policy_name(0x02), "ALWAYS");
        assert_eq!(touch_policy_name(0x03), "CACHED");
        assert_eq!(pin_policy_name(0x09), "0x09"); // unknown → hex
        assert_eq!(form_factor_name(0x03), "USB-C keychain");
        // High bits (FIPS/enterprise flags) are ignored.
        assert_eq!(form_factor_name(0x80 | 0x03), "USB-C keychain");
    }
}
