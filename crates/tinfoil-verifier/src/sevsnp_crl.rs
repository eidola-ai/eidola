//! Offline verification of the AMD CRL carried by a v3 attestation document.
//!
//! The document is untrusted transport. Authenticity comes from the pinned
//! AMD ARK signature, not from Tinfoil; the signed `thisUpdate`/`nextUpdate`
//! interval supplies bounded freshness. The verifier accepts only a complete,
//! direct v2 CRL (never a delta, indirect, or scope-restricted CRL), requires
//! its CRL number and ARK key identifier, rejects unsupported critical
//! extensions, and checks the ASK/VCEK serials.
//!
//! This deliberately performs no AMD KDS request. A malicious relay cannot
//! alter the CRL or serve it outside its signed validity interval, but it can
//! replay an older AMD-signed CRL while that interval remains valid. That is
//! the explicit availability/privacy tradeoff of the self-contained flow.

use std::time::{SystemTime, UNIX_EPOCH};

use der::oid::AssociatedOid;
use der::{Decode, Encode};
use rsa::RsaPublicKey;
use rsa::pkcs1::RsaPssParams;
use rsa::pkcs1v15::VerifyingKey as Pkcs1v15VerifyingKey;
use rsa::pss::VerifyingKey as PssVerifyingKey;
use rsa::signature::Verifier;
use sha2::Sha384;
use x509_cert::crl::CertificateList;
use x509_cert::der::referenced::OwnedToRef;
use x509_cert::ext::pkix::{AuthorityKeyIdentifier, CrlNumber, SubjectKeyIdentifier};
use x509_cert::spki::ObjectIdentifier;
use x509_cert::time::Time;
use x509_cert::{Certificate, Version};

use crate::Error;

const RSA_PSS_OID: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.113549.1.1.10");
const RSA_PKCS1_SHA384_OID: ObjectIdentifier =
    ObjectIdentifier::new_unwrap("1.2.840.113549.1.1.12");
const DELTA_CRL_INDICATOR_OID: ObjectIdentifier = ObjectIdentifier::new_unwrap("2.5.29.27");
const ISSUING_DISTRIBUTION_POINT_OID: ObjectIdentifier = ObjectIdentifier::new_unwrap("2.5.29.28");
const CERTIFICATE_ISSUER_OID: ObjectIdentifier = ObjectIdentifier::new_unwrap("2.5.29.29");

/// Verify a document-carried, AMD-authenticated CRL and reject any listed
/// chain serial.
pub(crate) fn check_revocation(
    crl_der: &[u8],
    ark: &Certificate,
    serials: &[&[u8]],
) -> Result<(), Error> {
    check_revocation_at(crl_der, ark, serials, unix_now()?)
}

fn check_revocation_at(
    crl_der: &[u8],
    ark: &Certificate,
    serials: &[&[u8]],
    now: u64,
) -> Result<(), Error> {
    let crl = CertificateList::from_der(crl_der)
        .map_err(|e| Error::CertChain(format!("failed to parse AMD CRL DER: {e}")))?;

    if crl.signature_algorithm != crl.tbs_cert_list.signature {
        return Err(Error::CertChain(
            "AMD CRL outer and TBS signature algorithms differ".to_string(),
        ));
    }
    if crl.tbs_cert_list.issuer != ark.tbs_certificate.subject {
        return Err(Error::CertChain(
            "AMD CRL issuer does not match the trusted ARK subject".to_string(),
        ));
    }
    verify_crl_signature(ark, &crl)?;
    verify_crl_profile(ark, &crl)?;
    verify_validity_window(&crl, now)?;

    let revoked = crl
        .tbs_cert_list
        .revoked_certificates
        .as_deref()
        .unwrap_or(&[]);
    for serial in serials {
        if revoked
            .iter()
            .any(|entry| entry.serial_number.as_bytes() == *serial)
        {
            return Err(Error::CertChain(format!(
                "AMD CRL lists certificate serial {} as revoked",
                hex::encode(serial),
            )));
        }
    }
    Ok(())
}

fn unix_now() -> Result<u64, Error> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| Error::CertChain(format!("system clock is before UNIX_EPOCH: {e}")))?
        .as_secs())
}

fn verify_validity_window(crl: &CertificateList, now: u64) -> Result<(), Error> {
    let this_update = time_secs(crl.tbs_cert_list.this_update);
    let next_update = crl
        .tbs_cert_list
        .next_update
        .ok_or_else(|| Error::CertChain("AMD CRL has no nextUpdate validity bound".to_string()))?;
    let next_update = time_secs(next_update);
    if next_update <= this_update {
        return Err(Error::CertChain(format!(
            "AMD CRL validity interval is empty or reversed (thisUpdate={this_update}, nextUpdate={next_update})"
        )));
    }
    if now < this_update || now >= next_update {
        return Err(Error::CertChain(format!(
            "AMD CRL is outside its validity window (thisUpdate={this_update}, nextUpdate={next_update}, now={now})"
        )));
    }
    Ok(())
}

fn time_secs(time: Time) -> u64 {
    match time {
        Time::UtcTime(value) => value.to_unix_duration().as_secs(),
        Time::GeneralTime(value) => value.to_unix_duration().as_secs(),
    }
}

fn verify_crl_signature(ark: &Certificate, crl: &CertificateList) -> Result<(), Error> {
    let tbs_der = crl
        .tbs_cert_list
        .to_der()
        .map_err(|e| Error::CertChain(format!("failed to encode CRL tbs_cert_list: {e}")))?;
    let ark_spki = ark.tbs_certificate.subject_public_key_info.owned_to_ref();
    let ark_key = RsaPublicKey::try_from(ark_spki)
        .map_err(|e| Error::CertChain(format!("ARK does not contain an RSA public key: {e}")))?;
    let signature = crl.signature.raw_bytes();

    match crl.signature_algorithm.oid {
        RSA_PSS_OID => {
            require_pss_sha384_parameters(&crl.signature_algorithm)?;
            let key = PssVerifyingKey::<Sha384>::new(ark_key);
            let signature = rsa::pss::Signature::try_from(signature).map_err(|e| {
                Error::CertChain(format!("invalid CRL RSA-PSS signature bytes: {e}"))
            })?;
            key.verify(&tbs_der, &signature).map_err(|e| {
                Error::CertChain(format!(
                    "AMD CRL RSA-PSS signature verification failed: {e}"
                ))
            })
        }
        RSA_PKCS1_SHA384_OID => {
            if !matches!(
                crl.signature_algorithm.parameters.as_ref(),
                Some(parameters) if parameters.owned_to_ref() == der::asn1::AnyRef::NULL
            ) {
                return Err(Error::CertChain(
                    "AMD CRL PKCS#1 SHA-384 parameters must be ASN.1 NULL".to_string(),
                ));
            }
            let key = Pkcs1v15VerifyingKey::<Sha384>::new(ark_key);
            let signature = rsa::pkcs1v15::Signature::try_from(signature).map_err(|e| {
                Error::CertChain(format!("invalid CRL PKCS#1 signature bytes: {e}"))
            })?;
            key.verify(&tbs_der, &signature).map_err(|e| {
                Error::CertChain(format!("AMD CRL PKCS#1 signature verification failed: {e}"))
            })
        }
        oid => Err(Error::CertChain(format!(
            "AMD CRL uses unsupported signature algorithm OID: {oid}"
        ))),
    }
}

fn require_pss_sha384_parameters(
    algorithm: &x509_cert::spki::AlgorithmIdentifierOwned,
) -> Result<(), Error> {
    let parameters = algorithm
        .parameters
        .as_ref()
        .ok_or_else(|| Error::CertChain("AMD CRL RSA-PSS parameters are missing".to_string()))?;
    let parameters = parameters
        .decode_as::<RsaPssParams<'_>>()
        .map_err(|e| Error::CertChain(format!("invalid AMD CRL RSA-PSS parameters: {e}")))?;
    let expected = RsaPssParams::new::<Sha384>(48);
    if parameters != expected {
        return Err(Error::CertChain(
            "AMD CRL RSA-PSS parameters must use SHA-384, MGF1-SHA-384, a 48-byte salt, and trailerField 1"
                .to_string(),
        ));
    }
    Ok(())
}

/// Require the profile of a complete AMD generation CRL. In particular, a
/// valid signature over a delta or indirect CRL does not make that object a
/// complete revocation set.
fn verify_crl_profile(ark: &Certificate, crl: &CertificateList) -> Result<(), Error> {
    if crl.tbs_cert_list.version != Version::V2 {
        return Err(Error::CertChain("AMD CRL must be X.509 v2".to_string()));
    }

    let extensions = crl.tbs_cert_list.crl_extensions.as_deref().unwrap_or(&[]);
    let mut authority_key_identifier = None;
    let mut crl_number_seen = false;

    for extension in extensions {
        if extension.extn_id == AuthorityKeyIdentifier::OID {
            if authority_key_identifier.is_some() {
                return Err(Error::CertChain(
                    "AMD CRL carries duplicate authorityKeyIdentifier extensions".to_string(),
                ));
            }
            if extension.critical {
                return Err(Error::CertChain(
                    "AMD CRL authorityKeyIdentifier must be non-critical".to_string(),
                ));
            }
            authority_key_identifier = Some(
                AuthorityKeyIdentifier::from_der(extension.extn_value.as_bytes()).map_err(|e| {
                    Error::CertChain(format!(
                        "failed to parse AMD CRL authorityKeyIdentifier: {e}"
                    ))
                })?,
            );
        } else if extension.extn_id == CrlNumber::OID {
            if crl_number_seen {
                return Err(Error::CertChain(
                    "AMD CRL carries duplicate cRLNumber extensions".to_string(),
                ));
            }
            if extension.critical {
                return Err(Error::CertChain(
                    "AMD CRL cRLNumber must be non-critical".to_string(),
                ));
            }
            CrlNumber::from_der(extension.extn_value.as_bytes())
                .map_err(|e| Error::CertChain(format!("failed to parse AMD CRL cRLNumber: {e}")))?;
            crl_number_seen = true;
        } else if extension.extn_id == DELTA_CRL_INDICATOR_OID {
            return Err(Error::CertChain(
                "AMD CRL is a delta CRL; a complete generation CRL is required".to_string(),
            ));
        } else if extension.extn_id == ISSUING_DISTRIBUTION_POINT_OID {
            return Err(Error::CertChain(
                "AMD CRL is scope-restricted or indirect; a complete direct generation CRL is required"
                    .to_string(),
            ));
        } else if extension.critical {
            return Err(Error::CertChain(format!(
                "AMD CRL carries unsupported critical extension {}",
                extension.extn_id
            )));
        }
    }

    let authority_key_identifier = authority_key_identifier.ok_or_else(|| {
        Error::CertChain("AMD CRL has no authorityKeyIdentifier extension".to_string())
    })?;
    let crl_key_identifier = authority_key_identifier.key_identifier.ok_or_else(|| {
        Error::CertChain("AMD CRL authorityKeyIdentifier has no keyIdentifier".to_string())
    })?;
    let (_, ark_key_identifier) = ark
        .tbs_certificate
        .get::<SubjectKeyIdentifier>()
        .map_err(|e| Error::CertChain(format!("failed to parse ARK subjectKeyIdentifier: {e}")))?
        .ok_or_else(|| Error::CertChain("trusted ARK has no subjectKeyIdentifier".to_string()))?;
    if crl_key_identifier.as_bytes() != ark_key_identifier.0.as_bytes() {
        return Err(Error::CertChain(
            "AMD CRL authorityKeyIdentifier does not match the trusted ARK subjectKeyIdentifier"
                .to_string(),
        ));
    }
    if !crl_number_seen {
        return Err(Error::CertChain(
            "AMD CRL has no cRLNumber extension".to_string(),
        ));
    }

    for revoked in crl
        .tbs_cert_list
        .revoked_certificates
        .as_deref()
        .unwrap_or(&[])
    {
        for extension in revoked.crl_entry_extensions.as_deref().unwrap_or(&[]) {
            if extension.extn_id == CERTIFICATE_ISSUER_OID {
                return Err(Error::CertChain(
                    "AMD CRL contains an indirect-CRL certificateIssuer entry".to_string(),
                ));
            }
            if extension.critical {
                return Err(Error::CertChain(format!(
                    "AMD CRL entry carries unsupported critical extension {}",
                    extension.extn_id
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;
    use x509_cert::ext::Extension;

    // AMD production CRL #9, fetched 2026-08-14. Keeping the vendor-signed
    // DER makes these tests deterministic and exercises the exact production
    // RSA-PSS/extension profile without reaching AMD.
    const AMD_CRL_B64: &str = concat!(
        "MIIDdjCCASoCAQEwQQYJKoZIhvcNAQEKMDSgDzANBglghkgBZQMEAgIFAKEcMBoG",
        "CSqGSIb3DQEBCDANBglghkgBZQMEAgIFAKIDAgEwMHsxFDASBgNVBAsMC0VuZ2lu",
        "ZWVyaW5nMQswCQYDVQQGEwJVUzEUMBIGA1UEBwwLU2FudGEgQ2xhcmExCzAJBgNV",
        "BAgMAkNBMR8wHQYDVQQKDBZBZHZhbmNlZCBNaWNybyBEZXZpY2VzMRIwEAYDVQQD",
        "DAlBUkstR2Vub2EXDTI2MDcyODA0MDcyM1oXDTI2MDkwOTAwMDAwMFowFjAUAgMC",
        "AAEXDTIyMTAzMTEyMDAwMFqgLzAtMB8GA1UdIwQYMBaAFJ9d+f4N2PNa0DMaJe+B",
        "KU++MahbMAoGA1UdFAQDAgEJMEEGCSqGSIb3DQEBCjA0oA8wDQYJYIZIAWUDBAIC",
        "BQChHDAaBgkqhkiG9w0BAQgwDQYJYIZIAWUDBAICBQCiAwIBMAOCAgEAxU5bvzC6",
        "5iJST/mZBZCR+V+PcNqGs9HmAiaJdjAhmk24v2A0tHXRWWyX+tIKJqNN3CgVC+gY",
        "eFN4wds9pstIuGAl/KfbYAoivg3MvgMyutCbfF4apE0lDfyHiQv+MeCC1yi9dtie",
        "HMHNUAyHk0wJMZAUYNrSAOHJX/FpDp7rZZNRkaQHUUAHrJGps8hjwbdLSuvGLWC3",
        "t4v/bKM27CTLOBxqiaMffd4sYqQOpkhUUmq8rrh04/ZZVV1muwUvLYs/NQTBnJgW",
        "YSA7l0dnwwhdPwZss9bNxls4sFUEtKp/V29lkYiU2cX2voQFOi80XN3nw7UmR2RQ",
        "noXqV+JV85WqcZqUXLiVGRzxZHSyrQQhVxC8nkCW5L0mU0tH12v35Ll6jkycA33",
        "j7ZdMRwB9fqMCh2l0aht5UEWUwi+8G3wZa0TcLmarR/I7yKlFp/ByneLN0OnWRYt",
        "pmyEQdct+8t3HyycA5YSAWMRC73iG4yrKQtriVJemZDoFv6Zt3UdzMDAlCa2u1AY",
        "4zXNW2yun0Yg5neVR0AHzr/DSZ9dxcwHx8OophTc+dgEv7pLmtdAXI9TgEmD0o7",
        "G4Ay8ITvY05GrcAEhhljfY/gxgaIaZS1LayFShENxmPlZNvCAMzC0CAYNdIYllEh",
        "BJBUbgmRao3vVyxW+G3afB7WV9ggODF6H3e+Y=",
    );
    const THIS_UPDATE: u64 = 1_785_211_643;
    const NEXT_UPDATE: u64 = 1_788_912_000;

    fn fixture() -> (Vec<u8>, Certificate) {
        let crl = base64::engine::general_purpose::STANDARD
            .decode(AMD_CRL_B64)
            .unwrap();
        let (ark, _) = crate::sevsnp::resolve_chain_certs_der(None, None).unwrap();
        let ark = Certificate::from_der(&ark).unwrap();
        (crl, ark)
    }

    #[test]
    fn accepts_vendor_signed_complete_crl_inside_half_open_window() {
        let (crl, ark) = fixture();
        check_revocation_at(&crl, &ark, &[], THIS_UPDATE).unwrap();
        check_revocation_at(&crl, &ark, &[], NEXT_UPDATE - 1).unwrap();
    }

    #[test]
    fn rejects_before_this_update_and_at_next_update() {
        let (crl, ark) = fixture();
        let before = check_revocation_at(&crl, &ark, &[], THIS_UPDATE - 1).unwrap_err();
        assert!(before.to_string().contains("validity window"));
        let at_end = check_revocation_at(&crl, &ark, &[], NEXT_UPDATE).unwrap_err();
        assert!(at_end.to_string().contains("validity window"));
    }

    #[test]
    fn rejects_revoked_serial_and_tampered_signature() {
        let (mut crl, ark) = fixture();
        let revoked =
            check_revocation_at(&crl, &ark, &[&[0x02, 0x00, 0x01]], THIS_UPDATE).unwrap_err();
        assert!(revoked.to_string().contains("020001"));

        *crl.last_mut().unwrap() ^= 1;
        let tampered = check_revocation_at(&crl, &ark, &[], THIS_UPDATE).unwrap_err();
        assert!(
            tampered
                .to_string()
                .contains("signature verification failed")
        );
    }

    #[test]
    fn rejects_nonstandard_pss_parameters() {
        let (crl, _) = fixture();
        let mut crl = CertificateList::from_der(&crl).unwrap();
        let parameters = RsaPssParams::new::<Sha384>(32);
        crl.signature_algorithm.parameters = Some(der::Any::encode_from(&parameters).unwrap());
        let err = require_pss_sha384_parameters(&crl.signature_algorithm).unwrap_err();
        assert!(err.to_string().contains("48-byte salt"));
    }

    #[test]
    fn rejects_delta_scoped_and_unknown_critical_extensions() {
        let (crl, ark) = fixture();
        let parsed = CertificateList::from_der(&crl).unwrap();
        for (oid, critical, message) in [
            (DELTA_CRL_INDICATOR_OID, true, "delta CRL"),
            (ISSUING_DISTRIBUTION_POINT_OID, true, "scope-restricted"),
            (
                ObjectIdentifier::new_unwrap("1.3.6.1.4.1.55555.1"),
                true,
                "unsupported critical",
            ),
        ] {
            let mut candidate = parsed.clone();
            candidate
                .tbs_cert_list
                .crl_extensions
                .get_or_insert_default()
                .push(Extension {
                    extn_id: oid,
                    critical,
                    extn_value: der::asn1::OctetString::new(vec![0x05, 0x00]).unwrap(),
                });
            let err = verify_crl_profile(&ark, &candidate).unwrap_err();
            assert!(err.to_string().contains(message), "got: {err}");
        }
    }

    #[test]
    fn requires_ark_identity_and_crl_number_extensions() {
        let (crl, ark) = fixture();
        let parsed = CertificateList::from_der(&crl).unwrap();
        for (oid, message) in [
            (AuthorityKeyIdentifier::OID, "authorityKeyIdentifier"),
            (CrlNumber::OID, "cRLNumber"),
        ] {
            let mut candidate = parsed.clone();
            candidate
                .tbs_cert_list
                .crl_extensions
                .as_mut()
                .unwrap()
                .retain(|extension| extension.extn_id != oid);
            let err = verify_crl_profile(&ark, &candidate).unwrap_err();
            assert!(err.to_string().contains(message), "got: {err}");
        }
    }
}
