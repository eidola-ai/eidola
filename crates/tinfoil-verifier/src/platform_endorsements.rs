//! Intel TDX machine policy from the platform provider's signed platform
//! endorsements.
//!
//! Tinfoil publishes `platform-vX.Y.Z` releases of `tinfoilsh/cvmimage`
//! whose `platform-endorsements.json` (format
//! `https://tinfoil.sh/predicate/platform-endorsements/v2`) maps each
//! machine it operates to a named policy: TDX machines by the PPID of their
//! Intel PCK certificate, each policy giving the TDX module (MR_SEAM), the
//! exact TD_ATTRIBUTES and XFAM, the TEE_TCB_SVN floor, the QE vendor and the
//! floor on Intel's TCB evaluation data number. The workflow
//! `.github/workflows/platform-release.yml` signs the file as a GitHub
//! artifact attestation (Sigstore, Rekor-logged), with the file's own JSON
//! as the in-toto predicate.
//!
//! [`verify`] checks a committed copy of the file and its committed bundle
//! entirely offline, against the committed Sigstore trusted root: the bundle
//! verifies ([`sigstore`]), the signer is that workflow at exactly the named
//! tag in that repository on a GitHub-hosted runner, the Statement's one
//! subject is `platform-endorsements.json` with the file's SHA-256 (which
//! must also be the hash the caller recorded), and the signed predicate is
//! the file. [`VerifiedEndorsements::tdx_policy`] then derives a
//! [`TdxPolicy`] for one named policy: its fields, and as `ppids` every
//! machine the file maps to it. The PCK platform flags, which the
//! endorsements do not carry, are the caller's.
//!
//! This is pin-authoring code, run where a pin is made or re-derived
//! (`measure-enclave`, the gateway's build), never during a handshake; it is
//! behind the `platform-endorsements` feature.

mod sigstore;

use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};

use crate::Error;
use crate::measurement::{PckFlag, TdxPolicy};

/// The repository the endorsements are released from.
pub const REPOSITORY: &str = "tinfoilsh/cvmimage";
/// The workflow that signs them.
pub const WORKFLOW: &str = ".github/workflows/platform-release.yml";
/// The in-toto predicate type, and the file's own `format`.
pub const PREDICATE_TYPE: &str = "https://tinfoil.sh/predicate/platform-endorsements/v2";
/// The release asset, and the attestation's subject name.
pub const FILE_NAME: &str = "platform-endorsements.json";
/// The in-toto Statement type the attestation uses.
const STATEMENT_TYPE: &str = "https://in-toto.io/Statement/v1";

/// Whether `tag` is a platform release tag: `platform-vX.Y.Z`, each number
/// decimal without a leading zero (the pattern the release workflow itself
/// enforces, `PLATFORM_TAG_PATTERN`).
pub fn is_release_tag(tag: &str) -> bool {
    let Some(version) = tag.strip_prefix("platform-v") else {
        return false;
    };
    let parts: Vec<&str> = version.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.bytes().all(|b| b.is_ascii_digit())
                && (p.len() == 1 || !p.starts_with('0'))
                && p.len() <= 9
        })
}

/// A release tag's `(major, minor, patch)`, for ordering releases.
pub fn release_version(tag: &str) -> Option<(u32, u32, u32)> {
    if !is_release_tag(tag) {
        return None;
    }
    let mut parts = tag["platform-v".len()..]
        .split('.')
        .map(|p| p.parse::<u32>().ok());
    Some((parts.next()??, parts.next()??, parts.next()??))
}

/// The PCK certificate platform flags a pin states, which the endorsements
/// do not carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PckFlags {
    pub dynamic_platform: PckFlag,
    pub cached_keys: PckFlag,
    pub smt_enabled: PckFlag,
}

/// A verified endorsements file.
#[derive(Debug)]
pub struct VerifiedEndorsements {
    file: EndorsementsFile,
    /// The Rekor log index of the attestation.
    pub rekor_log_index: u64,
    /// When Rekor logged it (Unix seconds).
    pub integrated_time: u64,
}

/// Verify `file` (the release's `platform-endorsements.json`) and `bundle`
/// (its Sigstore bundle) against `trusted_root` (a Sigstore `TrustedRoot`)
/// as release `tag`, whose file the caller recorded as `expected_sha256`
/// (lowercase hex). Offline; every failure is
/// [`Error::PlatformEndorsements`].
pub fn verify(
    file: &[u8],
    bundle: &[u8],
    trusted_root: &[u8],
    tag: &str,
    expected_sha256: &str,
) -> Result<VerifiedEndorsements, Error> {
    let fail = |message: String| Error::PlatformEndorsements(format!("{tag}: {message}"));
    if !is_release_tag(tag) {
        return Err(fail("not a platform-vX.Y.Z release tag".into()));
    }
    let sha256 = hex::encode(Sha256::digest(file));
    if sha256 != expected_sha256 {
        return Err(fail(format!(
            "{FILE_NAME} hashes to {sha256}, but the recorded hash is {expected_sha256}"
        )));
    }

    let trust = sigstore::TrustedRoot::parse(trusted_root).map_err(fail)?;
    let workflow_uri = format!("https://github.com/{REPOSITORY}/{WORKFLOW}@refs/tags/{tag}");
    let repository_uri = format!("https://github.com/{REPOSITORY}");
    let source_ref = format!("refs/tags/{tag}");
    let verified = sigstore::verify_bundle(
        bundle,
        &trust,
        &sigstore::ExpectedIdentity {
            workflow_uri: &workflow_uri,
            repository_uri: &repository_uri,
            source_ref: &source_ref,
        },
    )
    .map_err(fail)?;

    // The signed Statement: about this file, by name and hash, with the
    // file itself as its predicate.
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Statement {
        #[serde(rename = "_type")]
        type_: String,
        subject: Vec<Subject>,
        #[serde(rename = "predicateType")]
        predicate_type: String,
        predicate: serde_json::Value,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Subject {
        name: String,
        digest: BTreeMap<String, String>,
    }
    let statement: Statement = serde_json::from_slice(&verified.payload).map_err(|e| {
        fail(format!(
            "the signed payload is not an in-toto Statement: {e}"
        ))
    })?;
    if statement.type_ != STATEMENT_TYPE || statement.predicate_type != PREDICATE_TYPE {
        return Err(fail(format!(
            "the attestation is a {:?} of {:?}, not a {STATEMENT_TYPE:?} of {PREDICATE_TYPE:?}",
            statement.type_, statement.predicate_type
        )));
    }
    let [subject] = statement.subject.as_slice() else {
        return Err(fail(format!(
            "the attestation has {} subjects, not one",
            statement.subject.len()
        )));
    };
    if subject.name != FILE_NAME
        || subject.digest.len() != 1
        || subject.digest.get("sha256") != Some(&sha256)
    {
        return Err(fail(format!(
            "the attestation's subject is {:?} {:?}, not {FILE_NAME:?} with sha256 {sha256}",
            subject.name, subject.digest
        )));
    }
    let as_value: serde_json::Value =
        serde_json::from_slice(file).map_err(|e| fail(format!("{FILE_NAME} is not JSON: {e}")))?;
    if as_value != statement.predicate {
        return Err(fail(format!(
            "{FILE_NAME} differs from the attestation's signed predicate"
        )));
    }

    let parsed: EndorsementsFile =
        serde_json::from_slice(file).map_err(|e| fail(format!("{FILE_NAME}: {e}")))?;
    if parsed.format != PREDICATE_TYPE {
        return Err(fail(format!(
            "{FILE_NAME}'s format is {:?}, not {PREDICATE_TYPE:?}",
            parsed.format
        )));
    }
    for (machine, policy) in &parsed.machines {
        if !parsed.policies.contains_key(policy) {
            return Err(fail(format!(
                "machine {machine} names policy {policy:?}, which the file does not define"
            )));
        }
    }
    Ok(VerifiedEndorsements {
        file: parsed,
        rekor_log_index: verified.rekor_log_index,
        integrated_time: verified.integrated_time,
    })
}

impl VerifiedEndorsements {
    /// The machine policy for the endorsements' policy `name`, which must be
    /// a TDX policy, with the pin's own PCK platform `flags`. `ppids` is
    /// every machine the file maps to `name`, in order, and must not be
    /// empty.
    pub fn tdx_policy(&self, name: &str, flags: PckFlags) -> Result<TdxPolicy, Error> {
        let fail =
            |message: String| Error::PlatformEndorsements(format!("policy {name:?}: {message}"));
        let policy = self
            .file
            .policies
            .get(name)
            .ok_or_else(|| fail("the endorsements define no such policy".into()))?;
        let tdx = match (policy.platform.as_str(), &policy.tdx, &policy.sev_snp) {
            ("tdx", Some(tdx), None) => tdx,
            (platform, _, _) => {
                return Err(fail(format!(
                    "a {platform:?} policy, with a {} block; only a TDX policy with a tdx block \
                     pins a TDX machine",
                    if policy.tdx.is_some() {
                        "tdx"
                    } else {
                        "non-tdx"
                    }
                )));
            }
        };
        for (field, value, bytes) in [
            ("qe_vendor_id", &tdx.qe_vendor_id, 16),
            ("minimum_tee_tcb_svn", &tdx.minimum_tee_tcb_svn, 16),
            ("mr_seam", &tdx.mr_seam, 48),
            ("td_attributes", &tdx.td_attributes, 8),
            ("xfam", &tdx.xfam, 8),
        ] {
            if !is_lower_hex(value, bytes) {
                return Err(fail(format!(
                    "{field} must be {} lowercase hex digits",
                    bytes * 2
                )));
            }
        }
        let ppids: Vec<String> = self
            .file
            .machines
            .iter()
            .filter(|(_, policy)| policy.as_str() == name)
            .map(|(machine, _)| {
                if is_lower_hex(machine, 16) {
                    Ok(machine.clone())
                } else {
                    Err(fail(format!(
                        "machine {machine:?} is not a PPID (32 lowercase hex digits)"
                    )))
                }
            })
            .collect::<Result<_, _>>()?;
        if ppids.is_empty() {
            return Err(fail("the endorsements map no machine to it".into()));
        }
        Ok(TdxPolicy {
            mr_seam: vec![tdx.mr_seam.clone()],
            td_attributes: tdx.td_attributes.clone(),
            xfam: tdx.xfam.clone(),
            minimum_tee_tcb_svn: tdx.minimum_tee_tcb_svn.clone(),
            minimum_tcb_evaluation_data_number: tdx.minimum_tcb_evaluation_data_number,
            qe_vendor_id: tdx.qe_vendor_id.clone(),
            ppids,
            dynamic_platform: flags.dynamic_platform,
            cached_keys: flags.cached_keys,
            smt_enabled: flags.smt_enabled,
        })
    }

    /// The names of the file's TDX policies, in order.
    pub fn tdx_policy_names(&self) -> Vec<&str> {
        self.file
            .policies
            .iter()
            .filter(|(_, p)| p.platform == "tdx")
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

// The file's shape. The top level and every TDX block refuse unknown
// members: a member Tinfoil adds to a TDX policy may be a constraint, and is
// reviewed rather than ignored. A SEV-SNP block is not read here. Maps refuse
// repeated keys, so no machine can be mapped to two policies by whichever
// reader keeps which duplicate.

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EndorsementsFile {
    format: String,
    #[serde(deserialize_with = "unique_keys")]
    machines: BTreeMap<String, String>,
    #[serde(deserialize_with = "unique_keys")]
    policies: BTreeMap<String, Policy>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    platform: String,
    #[serde(default)]
    tdx: Option<TdxBlock>,
    #[serde(default)]
    sev_snp: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TdxBlock {
    qe_vendor_id: String,
    minimum_tee_tcb_svn: String,
    mr_seam: String,
    td_attributes: String,
    xfam: String,
    minimum_tcb_evaluation_data_number: u32,
}

fn unique_keys<'de, D, V>(d: D) -> Result<BTreeMap<String, V>, D::Error>
where
    D: Deserializer<'de>,
    V: Deserialize<'de>,
{
    struct Visitor<V>(std::marker::PhantomData<V>);
    impl<'de, V: Deserialize<'de>> serde::de::Visitor<'de> for Visitor<V> {
        type Value = BTreeMap<String, V>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("an object with distinct keys")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut map: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, V>()? {
                if out.contains_key(&key) {
                    return Err(serde::de::Error::custom(format!("duplicate key {key:?}")));
                }
                out.insert(key, value);
            }
            Ok(out)
        }
    }
    d.deserialize_map(Visitor(std::marker::PhantomData))
}

fn is_lower_hex(s: &str, bytes: usize) -> bool {
    s.len() == bytes * 2 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}
