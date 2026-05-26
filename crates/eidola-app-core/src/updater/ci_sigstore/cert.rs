//! Fulcio leaf certificate parsing + chain walk + identity extraction.
//!
//! For a cosign sign-blob bundle, the leaf cert is what *signs the
//! blob* — its pubkey is what we verify the message signature against.
//! Its SAN URI + Fulcio's custom OIDC-issuer extension are what bind the
//! signature to a specific workflow identity.
//!
//! We don't trust the leaf until we've walked it back to one of the
//! Fulcio CAs in the embedded trusted root. The walk verifies each
//! link's signature with the parent's pubkey, and finally checks that
//! the topmost cert's subject is self-signed (a Fulcio root).

use const_oid::ObjectIdentifier;
use der::{Decode, Encode, asn1::Utf8StringRef};
use sha2::Digest;
use signature::hazmat::PrehashVerifier;
use spki::DecodePublicKey;
use x509_cert::Certificate;
use x509_cert::ext::pkix::SubjectAltName;
use x509_cert::ext::pkix::name::GeneralName;

use crate::error::AppError;
use crate::updater::trust::FulcioCa;

/// OIDs we recognize.
mod oid {
    use const_oid::ObjectIdentifier;
    /// `ecPublicKey` — generic EC SPKI algorithm. Curve is in
    /// AlgorithmIdentifier.parameters.
    pub const EC_PUBLIC_KEY: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
    /// NIST P-256 named curve.
    pub const NIST_P256: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.3.1.7");
    /// NIST P-384 named curve.
    pub const NIST_P384: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.132.0.34");

    /// `ecdsa-with-SHA256` cert signature algorithm.
    pub const ECDSA_WITH_SHA256: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.2");
    /// `ecdsa-with-SHA384` cert signature algorithm.
    pub const ECDSA_WITH_SHA384: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.3");

    /// Fulcio's OIDC issuer extension (v1) — value is a raw UTF-8 string.
    pub const FULCIO_OIDC_ISSUER_V1: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.1");
    /// Fulcio's OIDC issuer extension (v2) — value is a DER Utf8String.
    pub const FULCIO_OIDC_ISSUER_V2: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.8");
}

/// Information the rest of the pipeline needs from the leaf cert after a
/// successful chain walk.
#[derive(Debug, Clone)]
pub struct LeafCertInfo {
    /// SubjectPublicKeyInfo DER bytes — used to verify the signature over
    /// the manifest hash.
    pub spki_der: Vec<u8>,
    /// Pubkey algorithm + curve, used to dispatch the right ECDSA
    /// implementation when verifying the blob signature.
    pub leaf_key_alg: LeafKeyAlg,
    /// SAN URI — matched (glob) against `EXPECTED_CI_IDENTITY_PATTERN`.
    pub san_uri: String,
    /// OIDC issuer — exact-matched against `EXPECTED_CI_ISSUER`.
    pub oidc_issuer: String,
    /// Cert validity window — passed downstream so Rekor's `integratedTime`
    /// can be checked against it.
    pub not_before: u64,
    pub not_after: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeafKeyAlg {
    EcdsaP256,
    EcdsaP384,
}

/// Parse `leaf_der`, find a Fulcio CA whose validity covers the leaf,
/// verify every signature in the chain to a self-signed root, and
/// extract the identity material.
pub fn verify_chain_and_extract(
    leaf_der: &[u8],
    fulcio_cas: &[FulcioCa],
) -> Result<LeafCertInfo, AppError> {
    let leaf = Certificate::from_der(leaf_der).map_err(|e| AppError::Update {
        message: format!("parsing Fulcio leaf cert: {e}"),
    })?;

    let not_before = time_to_unix(leaf.tbs_certificate.validity.not_before);
    let not_after = time_to_unix(leaf.tbs_certificate.validity.not_after);

    // Find a CA whose validity covers the leaf's notBefore. `valid_from` /
    // `valid_until` come from the trusted root JSON.
    let ca = fulcio_cas
        .iter()
        .find(|ca| {
            let after_start = (not_before as i64) >= ca.valid_from;
            let before_end = ca.valid_until.is_none_or(|end| (not_before as i64) <= end);
            after_start && before_end
        })
        .ok_or_else(|| AppError::Update {
            message: format!(
                "no Fulcio CA in the trusted root covers leaf cert notBefore={not_before}; \
                 either the trusted root is stale or the cert is forged"
            ),
        })?;

    if ca.cert_chain_der.is_empty() {
        return Err(AppError::Update {
            message: "Fulcio CA has empty cert_chain — cannot verify the leaf".into(),
        });
    }

    // Walk the chain: leaf is signed by chain[0]; chain[i] is signed by
    // chain[i+1]; the final chain entry is self-signed (root). This
    // matches Sigstore's TrustedRoot format where each CA entry's
    // certificates list orders leaf-most-intermediate first, root last.
    let mut current = leaf.clone();
    for parent_der in &ca.cert_chain_der {
        let parent = Certificate::from_der(parent_der).map_err(|e| AppError::Update {
            message: format!("parsing Fulcio chain cert: {e}"),
        })?;
        verify_cert_signature(&current, &parent)?;
        current = parent;
    }
    // The topmost cert must be self-signed.
    verify_cert_signature(&current, &current)?;

    // Extract the leaf's signing material + identity.
    let spki_der = leaf
        .tbs_certificate
        .subject_public_key_info
        .to_der()
        .map_err(|e| AppError::Update {
            message: format!("re-encoding leaf SPKI: {e}"),
        })?;
    let leaf_key_alg = parse_leaf_key_alg(&leaf)?;
    let san_uri = extract_san_uri(&leaf)?;
    let oidc_issuer = extract_oidc_issuer(&leaf)?;

    Ok(LeafCertInfo {
        spki_der,
        leaf_key_alg,
        san_uri,
        oidc_issuer,
        not_before,
        not_after,
    })
}

fn parse_leaf_key_alg(cert: &Certificate) -> Result<LeafKeyAlg, AppError> {
    let spki = &cert.tbs_certificate.subject_public_key_info;
    if spki.algorithm.oid != oid::EC_PUBLIC_KEY {
        return Err(AppError::Update {
            message: format!(
                "leaf cert pubkey algorithm `{}` is not EC; Fulcio leaves are EC P-256",
                spki.algorithm.oid
            ),
        });
    }
    let curve_oid: ObjectIdentifier = spki
        .algorithm
        .parameters
        .as_ref()
        .and_then(|p| p.decode_as().ok())
        .ok_or_else(|| AppError::Update {
            message: "leaf cert SPKI missing EC named-curve parameter".into(),
        })?;
    if curve_oid == oid::NIST_P256 {
        Ok(LeafKeyAlg::EcdsaP256)
    } else if curve_oid == oid::NIST_P384 {
        Ok(LeafKeyAlg::EcdsaP384)
    } else {
        Err(AppError::Update {
            message: format!("leaf cert uses unsupported EC curve `{curve_oid}`"),
        })
    }
}

fn verify_cert_signature(cert: &Certificate, signer: &Certificate) -> Result<(), AppError> {
    let tbs_der = cert
        .tbs_certificate
        .to_der()
        .map_err(|e| AppError::Update {
            message: format!("re-encoding TBSCertificate: {e}"),
        })?;
    let sig_bytes = cert.signature.as_bytes().ok_or_else(|| AppError::Update {
        message: "cert signature BitString is not byte-aligned".into(),
    })?;
    let signer_spki = signer
        .tbs_certificate
        .subject_public_key_info
        .to_der()
        .map_err(|e| AppError::Update {
            message: format!("re-encoding signer SPKI: {e}"),
        })?;

    let alg_oid = &cert.signature_algorithm.oid;
    if *alg_oid == oid::ECDSA_WITH_SHA256 {
        verify_p256_ecdsa(&signer_spki, &tbs_der, sig_bytes, HashAlg::Sha256)
    } else if *alg_oid == oid::ECDSA_WITH_SHA384 {
        verify_p384_ecdsa(&signer_spki, &tbs_der, sig_bytes, HashAlg::Sha384)
    } else {
        Err(AppError::Update {
            message: format!(
                "cert is signed with unsupported algorithm `{alg_oid}` (we only support \
                 ecdsa-with-SHA256 and ecdsa-with-SHA384)"
            ),
        })
    }
}

#[derive(Copy, Clone)]
enum HashAlg {
    Sha256,
    Sha384,
}

fn verify_p256_ecdsa(
    signer_spki: &[u8],
    message: &[u8],
    sig_der: &[u8],
    hash: HashAlg,
) -> Result<(), AppError> {
    let key = p256::ecdsa::VerifyingKey::from_public_key_der(signer_spki).map_err(|e| {
        AppError::Update {
            message: format!("parsing signer P-256 pubkey: {e}"),
        }
    })?;
    let sig = p256::ecdsa::Signature::from_der(sig_der).map_err(|e| AppError::Update {
        message: format!("parsing P-256 signature DER: {e}"),
    })?;
    let prehash = match hash {
        HashAlg::Sha256 => sha2::Sha256::digest(message).to_vec(),
        HashAlg::Sha384 => {
            return Err(AppError::Update {
                message: "P-256 key signing under SHA-384 is not supported".into(),
            });
        }
    };
    key.verify_prehash(&prehash, &sig)
        .map_err(|e| AppError::Update {
            message: format!("ECDSA P-256 signature verification failed: {e}"),
        })
}

fn verify_p384_ecdsa(
    signer_spki: &[u8],
    message: &[u8],
    sig_der: &[u8],
    hash: HashAlg,
) -> Result<(), AppError> {
    let key = p384::ecdsa::VerifyingKey::from_public_key_der(signer_spki).map_err(|e| {
        AppError::Update {
            message: format!("parsing signer P-384 pubkey: {e}"),
        }
    })?;
    let sig = p384::ecdsa::Signature::from_der(sig_der).map_err(|e| AppError::Update {
        message: format!("parsing P-384 signature DER: {e}"),
    })?;
    let prehash = match hash {
        HashAlg::Sha256 => sha2::Sha256::digest(message).to_vec(),
        HashAlg::Sha384 => sha2::Sha384::digest(message).to_vec(),
    };
    key.verify_prehash(&prehash, &sig)
        .map_err(|e| AppError::Update {
            message: format!("ECDSA P-384 signature verification failed: {e}"),
        })
}

fn extract_san_uri(cert: &Certificate) -> Result<String, AppError> {
    let extensions = cert
        .tbs_certificate
        .extensions
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "leaf cert has no extensions (missing SAN)".into(),
        })?;

    for ext in extensions {
        if ext.extn_id == const_oid::db::rfc5280::ID_CE_SUBJECT_ALT_NAME {
            let san = SubjectAltName::from_der(ext.extn_value.as_bytes()).map_err(|e| {
                AppError::Update {
                    message: format!("parsing SAN extension: {e}"),
                }
            })?;
            for entry in san.0 {
                if let GeneralName::UniformResourceIdentifier(uri) = entry {
                    return Ok(uri.to_string());
                }
            }
            return Err(AppError::Update {
                message: "leaf cert SAN has no URI entry; cosign keyless signing always emits a workflow URI".into(),
            });
        }
    }
    Err(AppError::Update {
        message: "leaf cert has no SubjectAltName extension".into(),
    })
}

fn extract_oidc_issuer(cert: &Certificate) -> Result<String, AppError> {
    let extensions = cert
        .tbs_certificate
        .extensions
        .as_ref()
        .ok_or_else(|| AppError::Update {
            message: "leaf cert has no extensions (missing OIDC issuer)".into(),
        })?;

    let mut v1_value: Option<String> = None;
    let mut v2_value: Option<String> = None;

    for ext in extensions {
        if ext.extn_id == oid::FULCIO_OIDC_ISSUER_V2 {
            // V2: the extension value is a DER UTF8String.
            let utf8 = Utf8StringRef::from_der(ext.extn_value.as_bytes()).map_err(|e| {
                AppError::Update {
                    message: format!("parsing Fulcio OIDC issuer v2 extension: {e}"),
                }
            })?;
            v2_value = Some(utf8.as_str().to_string());
        } else if ext.extn_id == oid::FULCIO_OIDC_ISSUER_V1 {
            // V1: the extension value is a raw UTF-8 string (no DER tag).
            let s =
                std::str::from_utf8(ext.extn_value.as_bytes()).map_err(|e| AppError::Update {
                    message: format!("Fulcio OIDC issuer v1 extension is not UTF-8: {e}"),
                })?;
            v1_value = Some(s.to_string());
        }
    }

    // V2 wins over V1 per Fulcio's spec.
    v2_value.or(v1_value).ok_or_else(|| AppError::Update {
        message: "leaf cert has no Fulcio OIDC issuer extension (1.3.6.1.4.1.57264.1.1 or .1.8)"
            .into(),
    })
}

fn time_to_unix(t: x509_cert::time::Time) -> u64 {
    // `Time` is an enum of UtcTime (≤ 2049) and GeneralizedTime; both
    // expose `to_unix_duration()`.
    t.to_unix_duration().as_secs()
}

/// Verify that `signature_der` is a valid ECDSA signature over
/// `manifest_sha256` (the prehashed message) using the given leaf
/// pubkey + algorithm.
pub fn verify_blob_signature(
    leaf_spki_der: &[u8],
    leaf_key_alg: LeafKeyAlg,
    manifest_sha256: &[u8; 32],
    signature_der: &[u8],
) -> Result<(), AppError> {
    match leaf_key_alg {
        LeafKeyAlg::EcdsaP256 => {
            let key =
                p256::ecdsa::VerifyingKey::from_public_key_der(leaf_spki_der).map_err(|e| {
                    AppError::Update {
                        message: format!("parsing leaf P-256 pubkey for blob verify: {e}"),
                    }
                })?;
            let sig =
                p256::ecdsa::Signature::from_der(signature_der).map_err(|e| AppError::Update {
                    message: format!("parsing blob signature DER (P-256): {e}"),
                })?;
            key.verify_prehash(manifest_sha256, &sig)
                .map_err(|e| AppError::Update {
                    message: format!("blob ECDSA P-256 signature verification failed: {e}"),
                })
        }
        LeafKeyAlg::EcdsaP384 => {
            // For P-384 keys cosign typically signs sha256(blob), so we
            // use sha256 here too. (If we ever encounter a P-384 leaf that
            // signs sha384, we add an explicit toggle.)
            let key =
                p384::ecdsa::VerifyingKey::from_public_key_der(leaf_spki_der).map_err(|e| {
                    AppError::Update {
                        message: format!("parsing leaf P-384 pubkey for blob verify: {e}"),
                    }
                })?;
            let sig =
                p384::ecdsa::Signature::from_der(signature_der).map_err(|e| AppError::Update {
                    message: format!("parsing blob signature DER (P-384): {e}"),
                })?;
            key.verify_prehash(manifest_sha256, &sig)
                .map_err(|e| AppError::Update {
                    message: format!("blob ECDSA P-384 signature verification failed: {e}"),
                })
        }
    }
}

/// Glob-match `pattern` (containing literal `*` wildcards) against `s`,
/// anchored on both ends. Used for the cert identity check, where the
/// pinned `EXPECTED_CI_IDENTITY_PATTERN` ends with `…@refs/tags/v*`.
pub fn glob_matches(pattern: &str, s: &str) -> bool {
    let mut regex_str = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '*' => regex_str.push_str(".*"),
            // Regex metacharacters that need escaping in a literal match.
            '.' | '+' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' | '^' | '$' | '?' => {
                regex_str.push('\\');
                regex_str.push(ch);
            }
            other => regex_str.push(other),
        }
    }
    regex_str.push('$');
    match regex::Regex::new(&regex_str) {
        Ok(re) => re.is_match(s),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_literal() {
        assert!(glob_matches("hello", "hello"));
        assert!(!glob_matches("hello", "world"));
    }

    #[test]
    fn glob_matches_with_star() {
        let pattern =
            "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v*";
        assert!(glob_matches(
            pattern,
            "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v0.5.0"
        ));
        assert!(glob_matches(
            pattern,
            "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v1.2.3-pre"
        ));
        // Wrong repo
        assert!(!glob_matches(
            pattern,
            "https://github.com/attacker/eidola/.github/workflows/tinfoil-build.yml@refs/tags/v0.5.0"
        ));
        // Wrong workflow
        assert!(!glob_matches(
            pattern,
            "https://github.com/eidola-ai/eidola/.github/workflows/other.yml@refs/tags/v0.5.0"
        ));
        // Wrong ref kind
        assert!(!glob_matches(
            pattern,
            "https://github.com/eidola-ai/eidola/.github/workflows/tinfoil-build.yml@refs/heads/main"
        ));
    }

    #[test]
    fn glob_escapes_regex_metachars() {
        // `?` in the literal should match literally, not be a regex
        // quantifier.
        assert!(glob_matches("a?b", "a?b"));
        assert!(!glob_matches("a?b", "ab"));
        // `.` is literal too.
        assert!(glob_matches("a.b", "a.b"));
        assert!(!glob_matches("a.b", "aXb"));
    }
}
