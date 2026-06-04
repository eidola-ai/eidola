//! `release-tool pkcs11 list` — enumerate signing keys on a PKCS#11
//! device (YubiKey-PIV / SmartCard / HSM) and emit ready-to-use cosign
//! `--key` URIs, **without ever handling a PIN**.
//!
//! Why this exists, rather than `cosign pkcs11-tool list-keys-uris`:
//!
//! 1. cosign bakes the PIN into every URI it prints (`…&pin-value=<PIN>`),
//!    in plaintext. That string then tends to end up in shell history,
//!    `.envrc`, or a paste. This command reads only *public* objects —
//!    certificates and public keys carry `CKA_ID` and `CKA_LABEL` and do
//!    not require `C_Login` — so no PIN is ever requested or emitted. The
//!    PIN is supplied separately at sign time via `COSIGN_PKCS11_PIN`.
//! 2. cosign identifies the token by a volatile `slot-id`, and its PIN
//!    prompt calls `GetTokenInfo(slot-id)` which fails with
//!    `CKR_SLOT_ID_INVALID` for a `slot-id=0` URI. The URIs emitted here
//!    identify the token by its (stable) label instead and omit `slot-id`
//!    entirely, sidestepping that bug.
//!
//! It also prints the **sha256 SPKI fingerprint** to pin in
//! `releases/trust/trust-constants.json` — reconstructed from the public
//! key's `CKA_EC_PARAMS` + `CKA_EC_POINT` (or read directly from
//! `CKA_PUBLIC_KEY_INFO` when the module exposes it) and run through the
//! same `classify_attestant_spki_algorithm` the updater and `attest` use.
//! Because the SubjectPublicKeyInfo encoding for a named-curve key is
//! canonical, this fingerprint is identical to the one `attest` derives
//! from `cosign public-key` — so the operator never has to run a
//! PIN-bearing `cosign public-key` just to learn what to pin.
//!
//! `attest` stays generic over any cosign `--key` reference (PEM / KMS /
//! PKCS#11); this command only removes the error-prone manual URI
//! assembly. The authoritative algorithm check still happens in `attest`
//! (against the updater's allowlist); what is shown here is advisory.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use cryptoki::context::{CInitializeArgs, CInitializeFlags, Pkcs11};
use cryptoki::object::{Attribute, AttributeType, KeyType, ObjectClass};
use sha2::{Digest, Sha256};

pub struct Args {
    /// Path to the PKCS#11 module shared library. If `None`, probe the
    /// well-known libykcs11 install locations.
    pub module_path: Option<PathBuf>,
}

pub fn run(args: Args) -> Result<()> {
    let module_path = resolve_module_path(args.module_path)?;

    let pkcs11 = Pkcs11::new(&module_path)
        .with_context(|| format!("loading PKCS#11 module at {}", module_path.display()))?;
    pkcs11
        .initialize(CInitializeArgs::new(CInitializeFlags::OS_LOCKING_OK))
        .context("initializing the PKCS#11 module")?;

    let slots = pkcs11
        .get_slots_with_token()
        .context("listing PKCS#11 slots that have a token")?;
    if slots.is_empty() {
        bail!(
            "no PKCS#11 token found via {}. Is the YubiKey inserted?",
            module_path.display()
        );
    }

    let mut found_any = false;
    for slot in slots {
        let token = match pkcs11.get_token_info(slot) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  (skipping a slot whose token info could not be read: {e})");
                continue;
            }
        };
        let token_label = token.label().trim().to_string();
        let serial = token.serial_number().trim().to_string();

        // Read-only session, no login: public objects are visible without
        // a PIN, which is the whole point.
        let session = pkcs11
            .open_ro_session(slot)
            .with_context(|| format!("opening a read-only session on token `{token_label}`"))?;

        let handles = session
            .find_objects(&[Attribute::Class(ObjectClass::PUBLIC_KEY)])
            .context("enumerating public-key objects")?;

        for handle in handles {
            let attrs = session
                .get_attributes(
                    handle,
                    &[
                        AttributeType::Id,
                        AttributeType::Label,
                        AttributeType::KeyType,
                        AttributeType::EcParams,
                        AttributeType::EcPoint,
                        AttributeType::PublicKeyInfo,
                    ],
                )
                .unwrap_or_default();

            let mut id: Vec<u8> = Vec::new();
            let mut label = String::new();
            let mut key_type: Option<KeyType> = None;
            let mut ec_params: Vec<u8> = Vec::new();
            let mut ec_point: Vec<u8> = Vec::new();
            let mut public_key_info: Vec<u8> = Vec::new();
            for a in attrs {
                match a {
                    Attribute::Id(v) => id = v,
                    Attribute::Label(v) => label = String::from_utf8_lossy(&v).into_owned(),
                    Attribute::KeyType(kt) => key_type = Some(kt),
                    Attribute::EcParams(v) => ec_params = v,
                    Attribute::EcPoint(v) => ec_point = v,
                    Attribute::PublicKeyInfo(v) => public_key_info = v,
                    _ => {}
                }
            }
            // cosign can select by `id` alone; without one we can't build a
            // usable URI, so skip.
            if id.is_empty() {
                continue;
            }

            let analysis = analyze(key_type, &ec_params, &ec_point, &public_key_info);
            let uri = build_uri(&token_label, &id, &module_path);
            found_any = true;

            println!();
            print!("  {token_label}");
            if !serial.is_empty() {
                print!("  (serial {serial})");
            }
            println!();
            print!("    algorithm   : {}", analysis.algorithm);
            if !analysis.supported {
                print!("  — NOT a supported attestant algorithm (attest will reject)");
            }
            println!();
            if !label.is_empty() {
                println!("    label       : {label}");
            }
            println!("    id          : {}", hex_lower(&id));
            if let Some(fp) = &analysis.fingerprint {
                println!("    fingerprint : {fp}");
            }
            println!("    uri         : {uri}");
        }
    }

    if !found_any {
        bail!(
            "token present but no public-key objects were listable. Generate a key and a \
             self-signed certificate in the slot first — see \
             docs/contributing/release-attestant-yubikey.md."
        );
    }

    println!();
    println!("Next:");
    println!("  1. pin the `fingerprint` of your signing key in");
    println!("     releases/trust/trust-constants.json (trusted_attestant_fingerprints)");
    println!("  2. export EIDOLA_ATTESTANT_COSIGN_KEY='<uri above>' and run `just release-attest`");
    println!("     (it prompts for the PIN; the PIN never goes in the URI)");
    Ok(())
}

struct Analysis {
    /// Human-readable algorithm name. The updater's canonical name
    /// (`ECDSA-P256` / `ECDSA-P384` / `Ed25519`) when the key is
    /// supported, else a descriptive label.
    algorithm: String,
    /// `sha256(SPKI DER)` as lowercase hex — the value to pin in
    /// `trusted_attestant_fingerprints`. `None` for keys whose algorithm
    /// the updater rejects (pinning them would be useless).
    fingerprint: Option<String>,
    /// Whether the updater's verifier accepts this algorithm. Advisory;
    /// `attest` is authoritative.
    supported: bool,
}

/// Reconstruct the public key's SubjectPublicKeyInfo, classify it with the
/// same routine the updater and `attest` use, and (for supported keys)
/// compute the sha256 SPKI fingerprint to pin.
fn analyze(
    key_type: Option<KeyType>,
    ec_params: &[u8],
    ec_point: &[u8],
    public_key_info: &[u8],
) -> Analysis {
    let spki = match spki_der(key_type, ec_params, ec_point, public_key_info) {
        Some(der) => der,
        None => {
            return Analysis {
                algorithm: descriptive_label(key_type),
                fingerprint: None,
                supported: false,
            };
        }
    };
    match eidola_app_core::updater::human_attestation::classify_attestant_spki_algorithm(&spki) {
        Ok(algo) => Analysis {
            algorithm: algo.name().to_string(),
            fingerprint: Some(hex_lower(Sha256::digest(&spki).as_slice())),
            supported: true,
        },
        Err(_) => Analysis {
            algorithm: descriptive_label(key_type),
            fingerprint: None,
            supported: false,
        },
    }
}

/// Best-effort label for keys we cannot offer as attestant keys.
fn descriptive_label(key_type: Option<KeyType>) -> String {
    match key_type {
        Some(KeyType::EC) => "ECDSA (unsupported curve)".into(),
        Some(KeyType::EC_EDWARDS) => "Ed25519 (unreadable)".into(),
        Some(KeyType::RSA) => "RSA (unsupported)".into(),
        _ => "unknown / unsupported".into(),
    }
}

/// Produce the SubjectPublicKeyInfo DER for a public key object. Prefers a
/// module-provided `CKA_PUBLIC_KEY_INFO` (already a full SPKI); otherwise
/// reconstructs it from `CKA_EC_PARAMS` + `CKA_EC_POINT`. Returns `None`
/// if neither path applies (e.g. RSA, or attributes absent).
fn spki_der(
    key_type: Option<KeyType>,
    ec_params: &[u8],
    ec_point: &[u8],
    public_key_info: &[u8],
) -> Option<Vec<u8>> {
    if !public_key_info.is_empty() {
        return Some(public_key_info.to_vec());
    }
    match key_type {
        Some(KeyType::EC) if !ec_params.is_empty() && !ec_point.is_empty() => {
            ec_spki_der(ec_params, &unwrap_ec_point(ec_point)).ok()
        }
        Some(KeyType::EC_EDWARDS) if !ec_point.is_empty() => {
            ed25519_spki_der(&unwrap_ec_point(ec_point)).ok()
        }
        _ => None,
    }
}

/// `CKA_EC_POINT` is a DER `OCTET STRING` wrapping the ANSI X9.62 point
/// (`0x04 || X || Y`). Unwrap one `OCTET STRING` layer; tolerate modules
/// that return the bare point.
fn unwrap_ec_point(raw: &[u8]) -> Vec<u8> {
    use der::Decode;
    match der::asn1::OctetStringRef::from_der(raw) {
        Ok(os) => os.as_bytes().to_vec(),
        Err(_) => raw.to_vec(),
    }
}

/// Assemble the SPKI for a named-curve EC key: `SEQUENCE { SEQUENCE {
/// id-ecPublicKey, <curve OID> }, BIT STRING <point> }`. `curve_params`
/// is `CKA_EC_PARAMS` verbatim (the DER named-curve OID); `point` is the
/// uncompressed `0x04 || X || Y`.
fn ec_spki_der(curve_params: &[u8], point: &[u8]) -> Result<Vec<u8>> {
    use der::{Decode, Encode, asn1::BitString};
    use spki::{AlgorithmIdentifier, ObjectIdentifier, SubjectPublicKeyInfo};

    // id-ecPublicKey — 1.2.840.10045.2.1
    const ID_EC_PUBLIC_KEY: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
    let params = der::Any::from_der(curve_params).context("parsing CKA_EC_PARAMS as DER")?;
    let spki = SubjectPublicKeyInfo {
        algorithm: AlgorithmIdentifier {
            oid: ID_EC_PUBLIC_KEY,
            parameters: Some(params),
        },
        subject_public_key: BitString::from_bytes(point).context("EC point → BIT STRING")?,
    };
    spki.to_der().context("encoding EC SubjectPublicKeyInfo")
}

/// Assemble the SPKI for an Ed25519 key: `SEQUENCE { SEQUENCE {
/// id-Ed25519 }, BIT STRING <raw 32-byte key> }` (no AlgorithmIdentifier
/// parameters).
fn ed25519_spki_der(raw_key: &[u8]) -> Result<Vec<u8>> {
    use der::{Any, Encode, asn1::BitString};
    use spki::{AlgorithmIdentifier, ObjectIdentifier, SubjectPublicKeyInfo};

    // id-Ed25519 — 1.3.101.112
    const ID_ED25519: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.101.112");
    let spki = SubjectPublicKeyInfo {
        algorithm: AlgorithmIdentifier::<Any> {
            oid: ID_ED25519,
            parameters: None,
        },
        subject_public_key: BitString::from_bytes(raw_key).context("Ed25519 key → BIT STRING")?,
    };
    spki.to_der()
        .context("encoding Ed25519 SubjectPublicKeyInfo")
}

/// Build the cosign `--key` URI: token identified by label (stable, no
/// `slot-id` footgun), object selected by `id`, `type=private`, with the
/// module path as a query attribute. Deliberately carries no `pin-value`.
fn build_uri(token_label: &str, id: &[u8], module_path: &Path) -> String {
    let mut id_enc = String::new();
    for b in id {
        id_enc.push_str(&format!("%{b:02x}"));
    }
    format!(
        "pkcs11:token={};id={};type=private?module-path={}",
        pk11_path_encode(token_label),
        id_enc,
        module_path.display(),
    )
}

/// Percent-encode a PKCS#11-URI path component per RFC 7512 by encoding
/// everything outside the RFC 3986 unreserved set. Conservative but
/// always correct: e.g. `YubiKey PIV #37842605` →
/// `YubiKey%20PIV%20%2337842605`.
fn pk11_path_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn hex_lower(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Resolve the PKCS#11 module: an explicit path if given, else the first
/// existing well-known libykcs11 location.
fn resolve_module_path(explicit: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            bail!("--module-path {} does not exist", p.display());
        }
        return Ok(p);
    }
    const CANDIDATES: &[&str] = &[
        "/opt/homebrew/lib/libykcs11.dylib", // macOS Apple Silicon (Homebrew yubico-piv-tool)
        "/usr/local/lib/libykcs11.dylib",    // macOS Intel (Homebrew)
        "/usr/local/lib/libykcs11.so",       // Linux (source install)
        "/usr/lib/x86_64-linux-gnu/libykcs11.so", // Debian/Ubuntu package
        "/usr/lib/libykcs11.so",             // other Linux
    ];
    for c in CANDIDATES {
        let p = PathBuf::from(c);
        if p.exists() {
            return Ok(p);
        }
    }
    bail!(
        "could not locate libykcs11 (the YubiKey PKCS#11 module). Install it \
         (`brew install yubico-piv-tool`) or pass \
         --module-path /path/to/libykcs11.{{dylib,so}}."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn encodes_token_label_like_cosign() {
        assert_eq!(
            pk11_path_encode("YubiKey PIV #37842605"),
            "YubiKey%20PIV%20%2337842605"
        );
    }

    #[test]
    fn builds_pin_free_slot_id_free_uri() {
        let uri = build_uri(
            "YubiKey PIV #37842605",
            &[0x02],
            &PathBuf::from("/opt/homebrew/lib/libykcs11.dylib"),
        );
        assert_eq!(
            uri,
            "pkcs11:token=YubiKey%20PIV%20%2337842605;id=%02;type=private\
             ?module-path=/opt/homebrew/lib/libykcs11.dylib"
        );
        assert!(!uri.contains("pin-value"));
        assert!(!uri.contains("slot-id"));
    }

    #[test]
    fn multibyte_id_encodes_each_byte() {
        let uri = build_uri("t", &[0x0a, 0xff], &PathBuf::from("/m.so"));
        assert!(uri.contains(";id=%0a%ff;"));
    }

    /// The P-256 SPKI fixture committed for the human-attestation tests,
    /// originally emitted by cosign/OpenSSL. Decoding it gives the
    /// canonical DER our reconstruction must reproduce byte-for-byte.
    fn fixture_spki_der() -> Vec<u8> {
        use base64::Engine;
        let pem = std::fs::read_to_string(
            "../eidola-app-core/tests/fixtures/human_attestation/cosign.pub",
        )
        .expect("reading cosign.pub fixture");
        let b64: String = pem
            .lines()
            .filter(|l| !l.starts_with("-----"))
            .collect::<String>();
        base64::engine::general_purpose::STANDARD
            .decode(b64.trim())
            .expect("base64-decoding fixture SPKI")
    }

    #[test]
    fn ec_spki_der_reproduces_canonical_openssl_encoding() {
        use der::{Any, Decode, Encode, asn1::BitString};
        use spki::SubjectPublicKeyInfo;

        let reference = fixture_spki_der();
        let parsed =
            SubjectPublicKeyInfo::<Any, BitString>::from_der(&reference).expect("parse fixture");
        let point = parsed.subject_public_key.raw_bytes();
        let params_der = parsed
            .algorithm
            .parameters
            .as_ref()
            .expect("ec params")
            .to_der()
            .unwrap();

        // Rebuild from the same (curve OID, point) the device would expose
        // and require it to match cosign's encoding exactly — otherwise the
        // pinned fingerprint would not match what `attest` derives.
        let rebuilt = ec_spki_der(&params_der, point).expect("rebuild SPKI");
        assert_eq!(rebuilt, reference);
    }

    #[test]
    fn analyze_classifies_and_fingerprints_p256() {
        let reference = fixture_spki_der();
        // Feed the SPKI via the CKA_PUBLIC_KEY_INFO path.
        let a = analyze(Some(KeyType::EC), &[], &[], &reference);
        assert!(a.supported);
        assert_eq!(a.algorithm, "ECDSA-P256");
        let expected_fp = hex_lower(Sha256::digest(&reference).as_slice());
        assert_eq!(a.fingerprint.as_deref(), Some(expected_fp.as_str()));
    }

    #[test]
    fn analyze_rejects_rsa() {
        let a = analyze(Some(KeyType::RSA), &[], &[], &[]);
        assert!(!a.supported);
        assert!(a.fingerprint.is_none());
    }

    #[test]
    fn unwrap_ec_point_strips_octet_string_wrapper() {
        use der::{Encode, asn1::OctetStringRef};
        let point = [0x04u8, 0xaa, 0xbb, 0xcc];
        let wrapped = OctetStringRef::new(&point).unwrap().to_der().unwrap();
        assert_eq!(unwrap_ec_point(&wrapped), point);
        // A bare (unwrapped) point is returned as-is.
        assert_eq!(unwrap_ec_point(&point), point);
    }
}
