//! Offline verification of a GitHub artifact attestation: a Sigstore bundle
//! (v0.3) carrying a DSSE envelope over an in-toto Statement, signed by a
//! short-lived Fulcio certificate issued to a GitHub Actions workflow and
//! logged in Rekor as a `dsse` entry.
//!
//! Everything is checked against the committed Sigstore trusted root and
//! nothing else; there is no network access and no wall clock. Time enters
//! only as Rekor's `integratedTime`, which Rekor's signed entry timestamp
//! (SET) vouches for, and which must fall inside the Fulcio certificate's
//! validity and inside the validity windows the trusted root gives the CA and
//! the log key.
//!
//! What a pass proves:
//!
//! 1. The leaf certificate chains to a Fulcio CA of the trusted root that was
//!    valid when the leaf was issued, and its identity is exactly the caller's:
//!    OIDC issuer, SAN URI (the signing workflow at the exact ref), and the
//!    Fulcio extensions for the source repository, the source ref, the build
//!    signer and a GitHub-hosted runner.
//! 2. The DSSE envelope's one signature verifies under the leaf's key over
//!    `PAE(payloadType, payload)`.
//! 3. Rekor's one `dsse` 0.0.1 entry commits to this envelope (payload hash,
//!    signature and signer certificate), its SET verifies under a log key of
//!    the trusted root, and its inclusion proof reaches the stated root hash.
//!
//! Known gaps, shared with the updater's verifier of our own releases
//! (`docs/gaps.md`): the leaf's embedded SCT is not checked, and the Rekor
//! checkpoint signature over the inclusion proof's root hash is not checked
//! (the SET already binds the entry to a pinned log key).

use const_oid::ObjectIdentifier;
use der::asn1::Utf8StringRef;
use der::{Decode, Encode};
use serde::Deserialize;
use sha2::{Digest, Sha256, Sha384};
use signature::hazmat::PrehashVerifier;
use spki::DecodePublicKey;
use x509_cert::Certificate;
use x509_cert::ext::pkix::SubjectAltName;
use x509_cert::ext::pkix::name::GeneralName;

/// The bundle media type GitHub artifact attestations are served in.
pub(super) const BUNDLE_MEDIA_TYPE: &str = "application/vnd.dev.sigstore.bundle.v0.3+json";
/// The DSSE payload type of an in-toto Statement.
pub(super) const IN_TOTO_PAYLOAD_TYPE: &str = "application/vnd.in-toto+json";
/// GitHub Actions' OIDC issuer.
pub(super) const GITHUB_ACTIONS_ISSUER: &str = "https://token.actions.githubusercontent.com";

type Result<T> = std::result::Result<T, String>;

/// Who must have signed: every member is compared exactly.
pub(super) struct ExpectedIdentity<'a> {
    /// The SAN URI and Fulcio build-signer URI:
    /// `https://github.com/<repo>/<workflow path>@<ref>`.
    pub workflow_uri: &'a str,
    /// Fulcio's source-repository URI: `https://github.com/<repo>`.
    pub repository_uri: &'a str,
    /// Fulcio's source-repository ref: `refs/tags/<tag>`.
    pub source_ref: &'a str,
}

/// What a verified bundle signed, and where it was logged.
pub(super) struct Verified {
    /// The DSSE payload (the in-toto Statement's bytes).
    pub payload: Vec<u8>,
    pub rekor_log_index: u64,
    pub integrated_time: u64,
}

/// Verify `bundle` (the bundle JSON) against `trust` for `expected`.
pub(super) fn verify_bundle(
    bundle: &[u8],
    trust: &TrustedRoot,
    expected: &ExpectedIdentity<'_>,
) -> Result<Verified> {
    let bundle: Bundle =
        serde_json::from_slice(bundle).map_err(|e| format!("the bundle is not JSON: {e}"))?;
    if bundle.media_type != BUNDLE_MEDIA_TYPE {
        return Err(format!(
            "the bundle's mediaType is {:?}, not {BUNDLE_MEDIA_TYPE:?}",
            bundle.media_type
        ));
    }
    let envelope = bundle
        .dsse_envelope
        .as_ref()
        .ok_or("the bundle carries no dsseEnvelope; an artifact attestation is DSSE")?;
    let vm = &bundle.verification_material;
    let cert_der = b64(
        &vm.certificate
            .as_ref()
            .ok_or("the bundle's verificationMaterial carries no certificate")?
            .raw_bytes,
        "certificate.rawBytes",
    )?;

    // The leaf, its chain, and its identity.
    let leaf = cert::verify_chain_and_extract(&cert_der, &trust.fulcio_cas)?;
    cert::check_identity(&leaf, expected)?;

    // The envelope's one signature over PAE(payloadType, payload).
    if envelope.payload_type != IN_TOTO_PAYLOAD_TYPE {
        return Err(format!(
            "the DSSE payloadType is {:?}, not {IN_TOTO_PAYLOAD_TYPE:?}",
            envelope.payload_type
        ));
    }
    let [signature] = envelope.signatures.as_slice() else {
        return Err(format!(
            "the DSSE envelope carries {} signatures; an artifact attestation carries one",
            envelope.signatures.len()
        ));
    };
    let payload = b64(&envelope.payload, "dsseEnvelope.payload")?;
    let signature = b64(&signature.sig, "dsseEnvelope.signatures[0].sig")?;
    let pae = dsse_pae(&envelope.payload_type, &payload);
    cert::verify_ecdsa(&leaf.spki_der, leaf.key, &pae, &signature)
        .map_err(|e| format!("the DSSE signature does not verify: {e}"))?;

    // Rekor: one dsse entry about this envelope, logged and included.
    let [entry] = vm.tlog_entries.as_slice() else {
        return Err(format!(
            "the bundle carries {} transparency-log entries; an artifact attestation carries one",
            vm.tlog_entries.len()
        ));
    };
    if entry.kind_version.kind != "dsse" || entry.kind_version.version != "0.0.1" {
        return Err(format!(
            "the transparency-log entry is {} {}, not dsse 0.0.1",
            entry.kind_version.kind, entry.kind_version.version
        ));
    }
    let log_index: u64 = entry
        .log_index
        .parse()
        .map_err(|e| format!("tlog logIndex {:?}: {e}", entry.log_index))?;
    let integrated_time: i64 = entry
        .integrated_time
        .parse()
        .map_err(|e| format!("tlog integratedTime {:?}: {e}", entry.integrated_time))?;
    let body = b64(&entry.canonicalized_body, "canonicalizedBody")?;
    rekor::verify_dsse_body(&body, &payload, &signature, &cert_der)?;
    let log_id = b64_array::<32>(&entry.log_id.key_id, "tlog logId.keyId")?;
    let promise = entry
        .inclusion_promise
        .as_ref()
        .ok_or("the transparency-log entry carries no inclusionPromise (SET)")?;
    let set = b64(&promise.signed_entry_timestamp, "signedEntryTimestamp")?;
    let proof = entry
        .inclusion_proof
        .as_ref()
        .ok_or("the transparency-log entry carries no inclusionProof")?;
    let root_hash = b64_array::<32>(&proof.root_hash, "inclusionProof.rootHash")?;
    let tree_size: u64 = proof
        .tree_size
        .parse()
        .map_err(|e| format!("inclusionProof.treeSize {:?}: {e}", proof.tree_size))?;
    let proof_index: u64 = proof
        .log_index
        .parse()
        .map_err(|e| format!("inclusionProof.logIndex {:?}: {e}", proof.log_index))?;
    let hashes = proof
        .hashes
        .iter()
        .map(|h| b64_array::<32>(h, "inclusionProof.hashes[]"))
        .collect::<Result<Vec<_>>>()?;
    let key = trust
        .rekor_keys
        .iter()
        .find(|k| k.log_id == log_id)
        .ok_or_else(|| {
            format!(
                "no Rekor key of the trusted root has the entry's logId {}",
                hex::encode(log_id)
            )
        })?;
    if integrated_time < key.valid_from || key.valid_until.is_some_and(|end| integrated_time > end)
    {
        return Err(format!(
            "Rekor integratedTime {integrated_time} is outside the log key's validity in the \
             trusted root"
        ));
    }
    rekor::verify_set(
        key,
        &entry.canonicalized_body,
        integrated_time,
        log_index,
        &log_id,
        &set,
    )?;
    rekor::verify_inclusion(proof_index, &body, tree_size, &hashes, &root_hash)?;

    // The log's time is when the signing happened: inside the leaf's ten
    // minutes.
    let integrated_time = u64::try_from(integrated_time)
        .map_err(|_| format!("Rekor integratedTime {integrated_time} is negative"))?;
    if integrated_time < leaf.not_before || integrated_time > leaf.not_after {
        return Err(format!(
            "Rekor integratedTime {integrated_time} is outside the leaf certificate's validity \
             [{}, {}]",
            leaf.not_before, leaf.not_after
        ));
    }

    Ok(Verified {
        payload,
        rekor_log_index: log_index,
        integrated_time,
    })
}

/// DSSE pre-authentication encoding (PAEv1):
/// `"DSSEv1" SP LEN(type) SP type SP LEN(payload) SP payload`.
fn dsse_pae(payload_type: &str, payload: &[u8]) -> Vec<u8> {
    let mut pae = Vec::with_capacity(payload.len() + payload_type.len() + 32);
    pae.extend_from_slice(b"DSSEv1 ");
    pae.extend_from_slice(payload_type.len().to_string().as_bytes());
    pae.push(b' ');
    pae.extend_from_slice(payload_type.as_bytes());
    pae.push(b' ');
    pae.extend_from_slice(payload.len().to_string().as_bytes());
    pae.push(b' ');
    pae.extend_from_slice(payload);
    pae
}

// ---------------------------------------------------------------------------
// The bundle's JSON shape (the members read; others are ignored)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Bundle {
    media_type: String,
    verification_material: VerificationMaterial,
    #[serde(default)]
    dsse_envelope: Option<DsseEnvelope>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VerificationMaterial {
    #[serde(default)]
    certificate: Option<RawBytes>,
    #[serde(default)]
    tlog_entries: Vec<TlogEntry>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawBytes {
    raw_bytes: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DsseEnvelope {
    payload: String,
    payload_type: String,
    signatures: Vec<DsseSignature>,
}

#[derive(Deserialize)]
struct DsseSignature {
    sig: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TlogEntry {
    log_index: String,
    log_id: KeyId,
    kind_version: KindVersion,
    integrated_time: String,
    #[serde(default)]
    inclusion_promise: Option<InclusionPromise>,
    #[serde(default)]
    inclusion_proof: Option<InclusionProof>,
    canonicalized_body: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct KeyId {
    key_id: String,
}

#[derive(Deserialize)]
struct KindVersion {
    kind: String,
    version: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InclusionPromise {
    signed_entry_timestamp: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InclusionProof {
    log_index: String,
    root_hash: String,
    tree_size: String,
    #[serde(default)]
    hashes: Vec<String>,
}

// ---------------------------------------------------------------------------
// The Sigstore trusted root (`releases/trust/sigstore-trusted-root.json`)
// ---------------------------------------------------------------------------

/// The members of a Sigstore `TrustedRoot` this verifier uses: the Fulcio
/// CAs and the Rekor log keys, each with its validity window.
pub(super) struct TrustedRoot {
    fulcio_cas: Vec<FulcioCa>,
    rekor_keys: Vec<RekorKey>,
}

struct FulcioCa {
    /// Intermediate(s) then root, as the trusted root lists them.
    chain_der: Vec<Vec<u8>>,
    valid_from: i64,
    valid_until: Option<i64>,
}

struct RekorKey {
    log_id: [u8; 32],
    spki_der: Vec<u8>,
    key: KeyDetails,
    valid_from: i64,
    valid_until: Option<i64>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum KeyDetails {
    EcdsaP256Sha256,
    EcdsaP384Sha384,
    Ed25519,
}

impl TrustedRoot {
    pub(super) fn parse(json: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Raw {
            media_type: String,
            #[serde(default)]
            tlogs: Vec<RawTlog>,
            #[serde(default)]
            certificate_authorities: Vec<RawCa>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct RawCa {
            cert_chain: RawChain,
            valid_for: RawValidFor,
        }
        #[derive(Deserialize)]
        struct RawChain {
            certificates: Vec<RawBytes>,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct RawTlog {
            public_key: RawPublicKey,
            log_id: KeyId,
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct RawPublicKey {
            raw_bytes: String,
            key_details: String,
            valid_for: RawValidFor,
        }
        #[derive(Deserialize)]
        struct RawValidFor {
            start: String,
            #[serde(default)]
            end: Option<String>,
        }

        let raw: Raw = serde_json::from_slice(json)
            .map_err(|e| format!("the Sigstore trusted root is not readable: {e}"))?;
        if !raw
            .media_type
            .starts_with("application/vnd.dev.sigstore.trustedroot")
        {
            return Err(format!(
                "the Sigstore trusted root's mediaType is {:?}",
                raw.media_type
            ));
        }
        let fulcio_cas = raw
            .certificate_authorities
            .into_iter()
            .map(|ca| {
                Ok(FulcioCa {
                    chain_der: ca
                        .cert_chain
                        .certificates
                        .iter()
                        .map(|c| b64(&c.raw_bytes, "trusted root certificate"))
                        .collect::<Result<_>>()?,
                    valid_from: rfc3339_unix(&ca.valid_for.start)?,
                    valid_until: ca.valid_for.end.as_deref().map(rfc3339_unix).transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let rekor_keys = raw
            .tlogs
            .into_iter()
            .map(|t| {
                Ok(RekorKey {
                    log_id: b64_array::<32>(&t.log_id.key_id, "trusted root tlog keyId")?,
                    spki_der: b64(&t.public_key.raw_bytes, "trusted root tlog key")?,
                    key: match t.public_key.key_details.as_str() {
                        "PKIX_ECDSA_P256_SHA_256" => KeyDetails::EcdsaP256Sha256,
                        "PKIX_ECDSA_P384_SHA_384" => KeyDetails::EcdsaP384Sha384,
                        "PKIX_ED25519" => KeyDetails::Ed25519,
                        other => {
                            return Err(format!(
                                "the trusted root names an unknown key type {other:?}"
                            ));
                        }
                    },
                    valid_from: rfc3339_unix(&t.public_key.valid_for.start)?,
                    valid_until: t
                        .public_key
                        .valid_for
                        .end
                        .as_deref()
                        .map(rfc3339_unix)
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if fulcio_cas.is_empty() || rekor_keys.is_empty() {
            return Err("the Sigstore trusted root names no Fulcio CA or no Rekor key".into());
        }
        Ok(Self {
            fulcio_cas,
            rekor_keys,
        })
    }
}

/// `YYYY-MM-DDTHH:MM:SS[.fff]Z` as Unix seconds (UTC), by days-from-civil.
fn rfc3339_unix(s: &str) -> Result<i64> {
    let bad = || format!("malformed RFC 3339 timestamp {s:?}");
    let body = s.strip_suffix('Z').ok_or_else(bad)?;
    let (date, time) = body.split_once('T').ok_or_else(bad)?;
    let time = time.split('.').next().ok_or_else(bad)?;
    let numbers = |part: &str, sep: char| -> Result<Vec<i64>> {
        part.split(sep)
            .map(|n| n.parse::<i64>().map_err(|_| bad()))
            .collect()
    };
    let [year, month, day] = <[i64; 3]>::try_from(numbers(date, '-')?).map_err(|_| bad())?;
    let [hour, minute, second] = <[i64; 3]>::try_from(numbers(time, ':')?).map_err(|_| bad())?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || !(0..24).contains(&hour)
        || !(0..60).contains(&minute)
        || !(0..61).contains(&second)
    {
        return Err(bad());
    }
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Ok(days * 86_400 + hour * 3_600 + minute * 60 + second)
}

// ---------------------------------------------------------------------------
// The Fulcio certificate
// ---------------------------------------------------------------------------

mod cert {
    use super::*;

    const EC_PUBLIC_KEY: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
    const NIST_P256: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.3.1.7");
    const NIST_P384: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.132.0.34");
    const ECDSA_WITH_SHA256: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.2");
    const ECDSA_WITH_SHA384: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.3");
    // Fulcio's certificate extensions (sigstore/fulcio `docs/oid-info.md`);
    // the `.8` onward ones are DER UTF8String values.
    const ISSUER_V1: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.1");
    const ISSUER_V2: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.8");
    const BUILD_SIGNER_URI: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.9");
    const RUNNER_ENVIRONMENT: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.11");
    const SOURCE_REPOSITORY_URI: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.12");
    const SOURCE_REPOSITORY_REF: ObjectIdentifier =
        ObjectIdentifier::new_unwrap("1.3.6.1.4.1.57264.1.14");

    #[derive(Clone, Copy)]
    pub enum LeafKey {
        P256,
        P384,
    }

    pub struct Leaf {
        pub spki_der: Vec<u8>,
        pub key: LeafKey,
        pub not_before: u64,
        pub not_after: u64,
        cert: Certificate,
    }

    pub fn verify_chain_and_extract(leaf_der: &[u8], cas: &[FulcioCa]) -> Result<Leaf> {
        let cert = Certificate::from_der(leaf_der)
            .map_err(|e| format!("the Fulcio certificate does not parse: {e}"))?;
        let not_before = cert
            .tbs_certificate
            .validity
            .not_before
            .to_unix_duration()
            .as_secs();
        let not_after = cert
            .tbs_certificate
            .validity
            .not_after
            .to_unix_duration()
            .as_secs();
        let issued = i64::try_from(not_before).map_err(|_| "certificate time overflows")?;
        // Every CA of the trusted root valid at issuance is a candidate; the
        // leaf must chain to one of them.
        let mut last_error = String::from("no Fulcio CA of the trusted root was valid then");
        for ca in cas
            .iter()
            .filter(|ca| issued >= ca.valid_from && ca.valid_until.is_none_or(|end| issued <= end))
        {
            match walk_chain(&cert, &ca.chain_der) {
                Ok(()) => {
                    let spki = &cert.tbs_certificate.subject_public_key_info;
                    let key = leaf_key(spki)?;
                    return Ok(Leaf {
                        spki_der: spki
                            .to_der()
                            .map_err(|e| format!("re-encoding the leaf key: {e}"))?,
                        key,
                        not_before,
                        not_after,
                        cert,
                    });
                }
                Err(e) => last_error = e,
            }
        }
        Err(format!(
            "the certificate does not chain to the trusted root (issued at {not_before}): \
             {last_error}"
        ))
    }

    fn walk_chain(leaf: &Certificate, chain_der: &[Vec<u8>]) -> Result<()> {
        if chain_der.is_empty() {
            return Err("a Fulcio CA with no certificates".into());
        }
        let mut current = leaf.clone();
        for parent_der in chain_der {
            let parent = Certificate::from_der(parent_der)
                .map_err(|e| format!("a trusted-root certificate does not parse: {e}"))?;
            if current.tbs_certificate.issuer != parent.tbs_certificate.subject {
                return Err("an issuer name does not match its CA's subject".into());
            }
            verify_signed_by(&current, &parent)?;
            current = parent;
        }
        // The last certificate of the CA's chain is its self-signed root.
        verify_signed_by(&current, &current)
    }

    fn leaf_key(spki: &spki::SubjectPublicKeyInfoOwned) -> Result<LeafKey> {
        if spki.algorithm.oid != EC_PUBLIC_KEY {
            return Err(format!(
                "the leaf key's algorithm {} is not EC",
                spki.algorithm.oid
            ));
        }
        let curve: ObjectIdentifier = spki
            .algorithm
            .parameters
            .as_ref()
            .and_then(|p| p.decode_as().ok())
            .ok_or("the leaf key names no curve")?;
        match curve {
            c if c == NIST_P256 => Ok(LeafKey::P256),
            c if c == NIST_P384 => Ok(LeafKey::P384),
            other => Err(format!(
                "the leaf key's curve {other} is not P-256 or P-384"
            )),
        }
    }

    fn verify_signed_by(cert: &Certificate, signer: &Certificate) -> Result<()> {
        let tbs = cert
            .tbs_certificate
            .to_der()
            .map_err(|e| format!("re-encoding a TBSCertificate: {e}"))?;
        let sig = cert
            .signature
            .as_bytes()
            .ok_or("a certificate signature is not whole bytes")?;
        let spki = signer
            .tbs_certificate
            .subject_public_key_info
            .to_der()
            .map_err(|e| format!("re-encoding a CA key: {e}"))?;
        let oid = cert.signature_algorithm.oid;
        if oid == ECDSA_WITH_SHA256 {
            verify_p256(&spki, &Sha256::digest(&tbs), sig)
        } else if oid == ECDSA_WITH_SHA384 {
            verify_p384(&spki, &Sha384::digest(&tbs), sig)
        } else {
            Err(format!("a certificate is signed with {oid}, not ECDSA"))
        }
    }

    /// Verify an ECDSA signature over `message`, hashed as the leaf's curve
    /// pairs it (P-256 with SHA-256, P-384 with SHA-384).
    pub fn verify_ecdsa(spki: &[u8], key: LeafKey, message: &[u8], sig: &[u8]) -> Result<()> {
        match key {
            LeafKey::P256 => verify_p256(spki, &Sha256::digest(message), sig),
            LeafKey::P384 => verify_p384(spki, &Sha384::digest(message), sig),
        }
    }

    pub fn verify_p256(spki: &[u8], prehash: &[u8], sig: &[u8]) -> Result<()> {
        let key = p256::ecdsa::VerifyingKey::from_public_key_der(spki)
            .map_err(|e| format!("a P-256 key does not parse: {e}"))?;
        let sig = p256::ecdsa::Signature::from_der(sig)
            .map_err(|e| format!("a P-256 signature does not parse: {e}"))?;
        key.verify_prehash(prehash, &sig)
            .map_err(|e| format!("a P-256 signature does not verify: {e}"))
    }

    fn verify_p384(spki: &[u8], prehash: &[u8], sig: &[u8]) -> Result<()> {
        let key = p384::ecdsa::VerifyingKey::from_public_key_der(spki)
            .map_err(|e| format!("a P-384 key does not parse: {e}"))?;
        let sig = p384::ecdsa::Signature::from_der(sig)
            .map_err(|e| format!("a P-384 signature does not parse: {e}"))?;
        key.verify_prehash(prehash, &sig)
            .map_err(|e| format!("a P-384 signature does not verify: {e}"))
    }

    /// The leaf's identity, member by member, against `expected`.
    pub fn check_identity(leaf: &Leaf, expected: &ExpectedIdentity<'_>) -> Result<()> {
        let extensions = leaf
            .cert
            .tbs_certificate
            .extensions
            .as_deref()
            .unwrap_or_default();
        let one = |oid: ObjectIdentifier| -> Result<Option<&[u8]>> {
            let mut found = extensions.iter().filter(|e| e.extn_id == oid);
            let first = found.next().map(|e| e.extn_value.as_bytes());
            if found.next().is_some() {
                return Err(format!("the certificate repeats extension {oid}"));
            }
            Ok(first)
        };
        let utf8 = |oid: ObjectIdentifier, name: &str| -> Result<String> {
            let value = one(oid)?.ok_or_else(|| format!("the certificate states no {name}"))?;
            Utf8StringRef::from_der(value)
                .map(|s| s.as_str().to_owned())
                .map_err(|e| format!("the certificate's {name} does not parse: {e}"))
        };

        let issuer = match one(ISSUER_V2)? {
            Some(_) => utf8(ISSUER_V2, "OIDC issuer")?,
            None => {
                let raw = one(ISSUER_V1)?.ok_or("the certificate states no OIDC issuer")?;
                String::from_utf8(raw.to_vec())
                    .map_err(|_| "the certificate's OIDC issuer is not UTF-8")?
            }
        };
        if issuer != GITHUB_ACTIONS_ISSUER {
            return Err(format!(
                "the certificate's OIDC issuer is {issuer:?}, not {GITHUB_ACTIONS_ISSUER:?}"
            ));
        }

        let san = one(const_oid::db::rfc5280::ID_CE_SUBJECT_ALT_NAME)?
            .ok_or("the certificate has no subject alternative name")?;
        let san = SubjectAltName::from_der(san)
            .map_err(|e| format!("the subject alternative name does not parse: {e}"))?;
        let uris: Vec<String> = san
            .0
            .iter()
            .map(|name| match name {
                GeneralName::UniformResourceIdentifier(uri) => Ok(uri.to_string()),
                _ => Err("the subject alternative name holds a non-URI name".to_string()),
            })
            .collect::<Result<_>>()?;
        if uris != [expected.workflow_uri] {
            return Err(format!(
                "the certificate's identity is {uris:?}, not [{:?}]",
                expected.workflow_uri
            ));
        }

        for (oid, name, want) in [
            (BUILD_SIGNER_URI, "build signer URI", expected.workflow_uri),
            (
                SOURCE_REPOSITORY_URI,
                "source repository URI",
                expected.repository_uri,
            ),
            (
                SOURCE_REPOSITORY_REF,
                "source repository ref",
                expected.source_ref,
            ),
            (RUNNER_ENVIRONMENT, "runner environment", "github-hosted"),
        ] {
            let got = utf8(oid, name)?;
            if got != want {
                return Err(format!("the certificate's {name} is {got:?}, not {want:?}"));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Rekor: the dsse entry body, the SET, the inclusion proof
// ---------------------------------------------------------------------------

mod rekor {
    use super::*;

    #[derive(Deserialize)]
    struct Body {
        #[serde(rename = "apiVersion")]
        api_version: String,
        kind: String,
        spec: Spec,
    }

    #[derive(Deserialize)]
    struct Spec {
        #[serde(rename = "payloadHash")]
        payload_hash: Hash,
        signatures: Vec<BodySignature>,
    }

    #[derive(Deserialize)]
    struct Hash {
        algorithm: String,
        value: String,
    }

    #[derive(Deserialize)]
    struct BodySignature {
        /// Base64 of the DSSE signature.
        signature: String,
        /// Base64 of the signer certificate's PEM.
        verifier: String,
    }

    /// The entry body is about this envelope: its payload hash, its one
    /// signature and its signer certificate.
    pub fn verify_dsse_body(
        body: &[u8],
        payload: &[u8],
        signature: &[u8],
        leaf_der: &[u8],
    ) -> Result<()> {
        let body: Body = serde_json::from_slice(body)
            .map_err(|e| format!("the Rekor entry body is not a dsse body: {e}"))?;
        if body.kind != "dsse" || body.api_version != "0.0.1" {
            return Err(format!(
                "the Rekor entry body is {} {}, not dsse 0.0.1",
                body.kind, body.api_version
            ));
        }
        if body.spec.payload_hash.algorithm != "sha256"
            || body.spec.payload_hash.value != hex::encode(Sha256::digest(payload))
        {
            return Err("the Rekor entry's payload hash is not this envelope's payload".into());
        }
        let [entry_signature] = body.spec.signatures.as_slice() else {
            return Err("the Rekor entry body does not carry exactly one signature".into());
        };
        if b64(&entry_signature.signature, "Rekor dsse signature")? != signature {
            return Err("the Rekor entry's signature is not this envelope's".into());
        }
        let pem = b64(&entry_signature.verifier, "Rekor dsse verifier")?;
        let pem = pem::parse(&pem).map_err(|e| format!("the Rekor entry's verifier: {e}"))?;
        if pem.tag() != "CERTIFICATE" || pem.contents() != leaf_der {
            return Err("the Rekor entry's signer is not this bundle's certificate".into());
        }
        Ok(())
    }

    /// Rekor's signed entry timestamp over the canonical
    /// `{body, integratedTime, logID, logIndex}` (keys in byte order),
    /// `body` being the bundle's base64 text exactly as Rekor emitted it.
    pub fn verify_set(
        key: &RekorKey,
        body_b64: &str,
        integrated_time: i64,
        log_index: u64,
        log_id: &[u8; 32],
        set: &[u8],
    ) -> Result<()> {
        let message = format!(
            r#"{{"body":"{body_b64}","integratedTime":{integrated_time},"logID":"{}","logIndex":{log_index}}}"#,
            hex::encode(log_id)
        );
        match key.key {
            KeyDetails::EcdsaP256Sha256 => {
                cert::verify_p256(&key.spki_der, &Sha256::digest(message.as_bytes()), set)
                    .map_err(|e| format!("the Rekor SET: {e}"))
            }
            KeyDetails::EcdsaP384Sha384 | KeyDetails::Ed25519 => Err(
                "the entry's log key is not ECDSA P-256, which the rekor.sigstore.dev log uses"
                    .into(),
            ),
        }
    }

    fn leaf_hash(leaf: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([0x00]);
        h.update(leaf);
        h.finalize().into()
    }

    fn node_hash(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([0x01]);
        h.update(left);
        h.update(right);
        h.finalize().into()
    }

    /// RFC 6962 / RFC 9162 §2.1.3.2 inclusion-proof verification.
    pub fn verify_inclusion(
        index: u64,
        body: &[u8],
        tree_size: u64,
        hashes: &[[u8; 32]],
        root: &[u8; 32],
    ) -> Result<()> {
        if index >= tree_size {
            return Err(format!(
                "inclusion proof index {index} is not below the tree size {tree_size}"
            ));
        }
        let inner = u64::from(u64::BITS - (index ^ (tree_size - 1)).leading_zeros());
        let border = u64::from((index >> inner).count_ones());
        if hashes.len() as u64 != inner + border {
            return Err(format!(
                "the inclusion proof has {} hashes, not {}",
                hashes.len(),
                inner + border
            ));
        }
        let (inner_hashes, border_hashes) = hashes.split_at(inner as usize);
        let mut seed = leaf_hash(body);
        for (i, h) in inner_hashes.iter().enumerate() {
            seed = if (index >> i) & 1 == 0 {
                node_hash(&seed, h)
            } else {
                node_hash(h, &seed)
            };
        }
        for h in border_hashes {
            seed = node_hash(h, &seed);
        }
        if &seed != root {
            return Err("the inclusion proof does not reach the stated root hash".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------

fn b64(s: &str, field: &str) -> Result<Vec<u8>> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(s.as_bytes())
        .map_err(|e| format!("{field} is not base64: {e}"))
}

fn b64_array<const N: usize>(s: &str, field: &str) -> Result<[u8; N]> {
    let bytes = b64(s, field)?;
    bytes
        .as_slice()
        .try_into()
        .map_err(|_| format!("{field} is {} bytes, not {N}", bytes.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsse_pae_is_the_specified_encoding() {
        assert_eq!(
            dsse_pae("application/vnd.in-toto+json", b"{}"),
            b"DSSEv1 28 application/vnd.in-toto+json 2 {}"
        );
    }

    #[test]
    fn timestamps_are_read_as_utc_seconds() {
        assert_eq!(rfc3339_unix("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(rfc3339_unix("2021-01-12T11:53:27Z").unwrap(), 1_610_452_407);
        assert_eq!(
            rfc3339_unix("2022-12-31T23:59:59.999Z").unwrap(),
            1_672_531_199
        );
        for bad in ["2021-01-12T11:53:27", "2021-13-12T11:53:27Z", "x"] {
            assert!(rfc3339_unix(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn an_inclusion_proof_must_reach_its_root() {
        // A two-leaf tree: root = H(1 ‖ H(0 ‖ a) ‖ H(0 ‖ b)).
        let (a, b) = (b"a".as_slice(), b"b".as_slice());
        let (la, lb) = (rekor_leaf(a), rekor_leaf(b));
        let mut h = Sha256::new();
        h.update([0x01]);
        h.update(la);
        h.update(lb);
        let root: [u8; 32] = h.finalize().into();
        rekor::verify_inclusion(0, a, 2, &[lb], &root).unwrap();
        rekor::verify_inclusion(1, b, 2, &[la], &root).unwrap();
        assert!(rekor::verify_inclusion(0, b, 2, &[lb], &root).is_err());
        assert!(rekor::verify_inclusion(2, a, 2, &[lb], &root).is_err());
        assert!(rekor::verify_inclusion(0, a, 2, &[], &root).is_err());
    }

    fn rekor_leaf(data: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update([0x00]);
        h.update(data);
        h.finalize().into()
    }
}
