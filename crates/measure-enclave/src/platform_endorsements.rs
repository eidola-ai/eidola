//! An engine deployment's Intel TDX machine policy, from the platform
//! provider's signed platform endorsements vendored under
//! `releases/trust/platform-endorsements/<tag>/`.
//!
//! A deployment's `deployment.json` names one policy of one endorsements
//! release, with the hash of that release's file, and states the three PCK
//! platform flags the endorsements do not carry. [`tdx_policy`] verifies the
//! vendored file and its Sigstore bundle offline against the committed
//! Sigstore trusted root (`tinfoil_verifier::platform_endorsements::verify`)
//! and derives the policy from them: the gateway's build runs the same two
//! calls over the same committed bytes, so the pin is re-derived on every
//! build and nothing here reads the network.
//!
//! [`vendor`] is the one step that takes new bytes: the maintainer's update
//! recipe (`just engine-platform-endorsements <tag>`) downloads a release's
//! file and GitHub's attestations for its digest into an empty directory and
//! hands them here, and only a bundle that verifies exactly as the build
//! verifies it is written.

use anyhow::{Context, Result, bail, ensure};
use eidola_common::engine_deployment::deployment::{
    PckFlag as SidecarFlag, SIGSTORE_TRUSTED_ROOT, TdxMachines, platform_endorsement_files,
};
use serde::Deserialize;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use tinfoil_verifier::platform_endorsements::{self, PckFlags, VerifiedEndorsements};
use tinfoil_verifier::{PckFlag, TdxPolicy};

/// Reads a committed file by its path relative to the workspace root.
pub type ReadFile<'a> = dyn Fn(&str) -> Result<Vec<u8>> + 'a;

/// The TDX machine policy `machines` names, derived from the vendored
/// endorsements it names, which must verify.
pub fn tdx_policy(machines: &TdxMachines, read: &ReadFile<'_>) -> Result<TdxPolicy> {
    let verified = verify_vendored(&machines.release, &machines.sha256, read)?;
    Ok(verified.tdx_policy(
        &machines.policy,
        PckFlags {
            dynamic_platform: flag(machines.dynamic_platform),
            cached_keys: flag(machines.cached_keys),
            smt_enabled: flag(machines.smt_enabled),
        },
    )?)
}

/// Verify the vendored release `tag`, whose file must hash to `sha256`.
pub fn verify_vendored(
    tag: &str,
    sha256: &str,
    read: &ReadFile<'_>,
) -> Result<VerifiedEndorsements> {
    let (file, bundle) = platform_endorsement_files(tag);
    Ok(platform_endorsements::verify(
        &read(&file)?,
        &read(&bundle)?,
        &read(SIGSTORE_TRUSTED_ROOT)?,
        tag,
        sha256,
    )?)
}

fn flag(flag: SidecarFlag) -> PckFlag {
    match flag {
        SidecarFlag::True => PckFlag::True,
        SidecarFlag::False => PckFlag::False,
        SidecarFlag::Undefined => PckFlag::Undefined,
    }
}

/// What [`vendor`] wrote.
#[derive(Debug)]
pub struct Vendored {
    /// The endorsements file's SHA-256 (lowercase hex), for `deployment.json`.
    pub sha256: String,
    /// The files written, relative to the workspace root.
    pub files: (String, String),
    pub rekor_log_index: u64,
    /// Each TDX policy, with the number of machines mapped to it.
    pub tdx_policies: Vec<(String, usize)>,
}

/// Pick, from GitHub's attestations for a release file's digest
/// (`attestations`: the `GET /repos/{owner}/{repo}/attestations/sha256:<hex>`
/// response), the bundle that verifies `file` as release `tag`, and return
/// it with what it verified. Several releases can carry a byte-identical
/// file, and each signs it again under its own tag, so the response may hold
/// bundles for other tags; those fail verification here. Of several bundles
/// for this tag (a re-run release job signs twice), the first logged wins.
pub fn select_bundle(
    file: &[u8],
    attestations: &[u8],
    trusted_root: &[u8],
    tag: &str,
) -> Result<(Vec<u8>, VerifiedEndorsements)> {
    #[derive(Deserialize)]
    struct Response<'a> {
        #[serde(borrow)]
        attestations: Vec<Attestation<'a>>,
    }
    #[derive(Deserialize)]
    struct Attestation<'a> {
        #[serde(borrow)]
        bundle: &'a RawValue,
    }
    let response: Response<'_> =
        serde_json::from_slice(attestations).context("the attestations response")?;
    let sha256 = hex::encode(Sha256::digest(file));
    let mut refusals = Vec::new();
    let mut best: Option<(Vec<u8>, VerifiedEndorsements)> = None;
    for attestation in &response.attestations {
        // The bundle exactly as served, with a final newline.
        let mut bytes = attestation.bundle.get().as_bytes().to_vec();
        bytes.push(b'\n');
        match platform_endorsements::verify(file, &bytes, trusted_root, tag, &sha256) {
            Ok(verified) => {
                if best
                    .as_ref()
                    .is_none_or(|(_, b)| verified.rekor_log_index < b.rekor_log_index)
                {
                    best = Some((bytes, verified));
                }
            }
            Err(e) => refusals.push(e.to_string()),
        }
    }
    best.ok_or_else(|| {
        anyhow::anyhow!(
            "none of the {} attestations verifies {} as {tag}:\n  {}",
            response.attestations.len(),
            platform_endorsements::FILE_NAME,
            refusals.join("\n  ")
        )
    })
}

/// Write release `tag`'s endorsements `file` and the bundle among
/// `attestations` that verifies it, under the workspace root, then verify
/// what was written exactly as the build reads it.
pub fn vendor(
    workspace: &std::path::Path,
    tag: &str,
    file: &[u8],
    attestations: &[u8],
) -> Result<Vendored> {
    ensure!(
        platform_endorsements::is_release_tag(tag),
        "{tag:?} is not a platform-vX.Y.Z release tag"
    );
    let read =
        |path: &str| std::fs::read(workspace.join(path)).with_context(|| format!("reading {path}"));
    let trusted_root = read(SIGSTORE_TRUSTED_ROOT)?;
    let (bundle, _) = select_bundle(file, attestations, &trusted_root, tag)?;
    let files = platform_endorsement_files(tag);
    for (path, bytes) in [(&files.0, file), (&files.1, bundle.as_slice())] {
        let path = workspace.join(path);
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    }
    let sha256 = hex::encode(Sha256::digest(file));
    let verified = verify_vendored(tag, &sha256, &read)?;
    let tdx_policies = verified
        .tdx_policy_names()
        .into_iter()
        .map(|name| {
            let flags = PckFlags {
                dynamic_platform: PckFlag::Undefined,
                cached_keys: PckFlag::Undefined,
                smt_enabled: PckFlag::Undefined,
            };
            let machines = verified
                .tdx_policy(name, flags)
                .map(|p| p.ppids.len())
                .unwrap_or(0);
            (name.to_owned(), machines)
        })
        .collect();
    Ok(Vendored {
        sha256,
        files,
        rekor_log_index: verified.rekor_log_index,
        tdx_policies,
    })
}

/// The newest release tag vendored under the workspace's
/// `releases/trust/platform-endorsements/`, if any.
pub fn newest_vendored(workspace: &std::path::Path) -> Result<Option<String>> {
    let root =
        workspace.join(eidola_common::engine_deployment::deployment::PLATFORM_ENDORSEMENTS_ROOT);
    if !root.exists() {
        return Ok(None);
    }
    let mut newest: Option<((u32, u32, u32), String)> = None;
    for entry in std::fs::read_dir(&root).with_context(|| format!("listing {}", root.display()))? {
        let name = entry?.file_name();
        let name = name.to_str().context("a non-UTF-8 name")?;
        let Some(version) = platform_endorsements::release_version(name) else {
            bail!(
                "{} holds {name:?}, which is not a release tag",
                root.display()
            );
        };
        if newest.as_ref().is_none_or(|(v, _)| version > *v) {
            newest = Some((version, name.to_owned()));
        }
    }
    Ok(newest.map(|(_, name)| name))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use eidola_common::engine_deployment::deployment::PLATFORM_ENDORSEMENTS_ROOT;

    /// The release the known answers are about, and its file's hash. The
    /// tests read it from a fixture copy, so the vendored tree can rotate.
    const TAG: &str = "platform-v0.1.11";
    const SHA256: &str = "ba0ccfbe8fb8a92d77a072ffeab362dc4b67faa0c88e7114eee7939ad9dd9548";
    /// The one machine the release maps to `tdx-b300-prod`.
    const B300_PPID: &str = "9e2d1e4f1e26f3347d5d7ddd3841f9f4";

    fn workspace() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn fixtures() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/platform-endorsements")
    }

    /// Reads a committed file by its workspace path, except that a vendored
    /// release is read from the fixture releases instead.
    pub(crate) fn read(path: &str) -> Result<Vec<u8>> {
        let at = match path.strip_prefix(&format!("{PLATFORM_ENDORSEMENTS_ROOT}/")) {
            Some(release) => fixtures().join(release),
            None => workspace().join(path),
        };
        std::fs::read(at).with_context(|| format!("reading {path}"))
    }

    fn fixture(name: &str) -> Vec<u8> {
        std::fs::read(fixtures().join(name)).unwrap()
    }

    fn vendored() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let (file, bundle) = platform_endorsement_files(TAG);
        (
            read(&file).unwrap(),
            read(&bundle).unwrap(),
            read(SIGSTORE_TRUSTED_ROOT).unwrap(),
        )
    }

    fn verify(file: &[u8], bundle: &[u8], root: &[u8], tag: &str, sha: &str) -> String {
        platform_endorsements::verify(file, bundle, root, tag, sha)
            .unwrap_err()
            .to_string()
    }

    fn machines(policy: &str) -> TdxMachines {
        TdxMachines {
            release: TAG.into(),
            sha256: SHA256.into(),
            policy: policy.into(),
            dynamic_platform: SidecarFlag::True,
            cached_keys: SidecarFlag::True,
            smt_enabled: SidecarFlag::False,
        }
    }

    /// The committed release verifies offline against the committed trusted
    /// root, and the B300 policy derives exactly what the release states.
    #[test]
    fn the_vendored_release_verifies_and_derives_its_policy() {
        let (file, bundle, root) = vendored();
        let verified = platform_endorsements::verify(&file, &bundle, &root, TAG, SHA256).unwrap();
        assert_eq!(verified.rekor_log_index, 3_725_492_526);
        let policy = tdx_policy(&machines("tdx-b300-prod"), &read).unwrap();
        assert_eq!(
            policy,
            TdxPolicy {
                mr_seam: vec![
                    "aef4ed3e686fe6cf44c9a8d1cc63105443b558ac4f40c282abf78271e9a4586c\
                     50408c8584c7b43fd21edb736700ba5f"
                        .into()
                ],
                td_attributes: "0000001000000000".into(),
                xfam: "e702060000000000".into(),
                minimum_tee_tcb_svn: "03010200000000000000000000000000".into(),
                minimum_tcb_evaluation_data_number: 19,
                qe_vendor_id: "939a7233f79c4ca9940a0db3957f0607".into(),
                ppids: vec![B300_PPID.into()],
                dynamic_platform: PckFlag::True,
                cached_keys: PckFlag::True,
                smt_enabled: PckFlag::False,
            }
        );
        // A policy with several machines lists every one, in order.
        let h200 = tdx_policy(&machines("tdx-h200-prod"), &read).unwrap();
        assert_eq!(h200.ppids.len(), 7);
        assert!(h200.ppids.windows(2).all(|w| w[0] < w[1]));
        // The derived policy is one the attesting client compiles.
        tinfoil_verifier::validate_pins(&[tinfoil_verifier::AllowedMeasurement::tdx(
            tinfoil_verifier::TdxPin {
                mrtd: "00".repeat(48),
                mrconfigid: "00".repeat(48),
                policy,
            },
        )])
        .unwrap();
    }

    /// A policy the endorsements do not define, a SEV-SNP policy, and a
    /// recorded hash that is not the file's are each refused.
    #[test]
    fn a_policy_must_be_a_defined_tdx_policy() {
        let err = tdx_policy(&machines("tdx-b300-dev"), &read)
            .unwrap_err()
            .to_string();
        assert!(err.contains("define no such policy"), "{err}");
        let err = tdx_policy(&machines("amd-turin-prod"), &read)
            .unwrap_err()
            .to_string();
        assert!(err.contains("only a TDX policy"), "{err}");
        let mut wrong = machines("tdx-b300-prod");
        wrong.sha256 = "00".repeat(32);
        let err = tdx_policy(&wrong, &read).unwrap_err().to_string();
        assert!(err.contains("but the recorded hash is"), "{err}");
        let mut missing = machines("tdx-b300-prod");
        missing.release = "platform-v0.1.9".into();
        let err = tdx_policy(&missing, &read).unwrap_err().to_string();
        assert!(err.contains("platform-v0.1.9"), "{err}");
    }

    /// The tag pin: an authentic attestation of the byte-identical file,
    /// signed by the same workflow for another release (`platform-v0.1.10`
    /// carried the same `platform-endorsements.json`), is refused as
    /// `platform-v0.1.11`, and the committed bundle is refused as any other
    /// tag.
    #[test]
    fn an_attestation_for_another_tag_is_refused() {
        let (file, bundle, root) = vendored();
        let other_tag = fixture("platform-v0.1.10.sigstore.json");
        let err = verify(&file, &other_tag, &root, TAG, SHA256);
        assert!(
            err.contains("the certificate's identity is")
                && err.contains("platform-release.yml@refs/tags/platform-v0.1.10"),
            "{err}"
        );
        // That bundle is genuine: it verifies as its own tag.
        platform_endorsements::verify(&file, &other_tag, &root, "platform-v0.1.10", SHA256)
            .unwrap();
        let err = verify(&file, &bundle, &root, "platform-v0.1.12", SHA256);
        assert!(err.contains("the certificate's identity is"), "{err}");
        let err = verify(&file, &bundle, &root, "v0.1.11", SHA256);
        assert!(err.contains("not a platform-vX.Y.Z release tag"), "{err}");
    }

    /// The repository pin: a genuine GitHub artifact attestation from another
    /// repository of the same provider (`tinfoilsh/confidential-model-router`
    /// release v0.0.115) is refused on its signer's identity.
    #[test]
    fn an_attestation_from_another_repository_is_refused() {
        let (file, _, root) = vendored();
        let err = verify(
            &file,
            &fixture("confidential-model-router-v0.0.115.sigstore.json"),
            &root,
            TAG,
            SHA256,
        );
        assert!(
            err.contains("tinfoilsh/confidential-model-router/.github/workflows/"),
            "{err}"
        );
    }

    /// The file, the bundle and the recorded hash, each tampered with.
    #[test]
    fn tampering_with_any_input_is_refused() {
        let (file, bundle, root) = vendored();

        // The recorded hash is not the file's.
        let err = verify(&file, &bundle, &root, TAG, &"00".repeat(32));
        assert!(err.contains("but the recorded hash is"), "{err}");

        // The file edited (a raised floor lowered), with the hash recorded
        // for the edit: the signed subject is the original's.
        let text = String::from_utf8(file.clone()).unwrap();
        let edited = text.replacen(
            "\"minimum_tcb_evaluation_data_number\": 19",
            "\"minimum_tcb_evaluation_data_number\": 1",
            1,
        );
        assert_ne!(edited, text);
        let edited_sha = hex::encode(Sha256::digest(edited.as_bytes()));
        let err = verify(edited.as_bytes(), &bundle, &root, TAG, &edited_sha);
        assert!(err.contains("the attestation's subject is"), "{err}");

        // A machine added to a policy: the same.
        let added = text.replacen(
            "\"9e2d1e4f1e26f3347d5d7ddd3841f9f4\": \"tdx-b300-prod\",",
            "\"9e2d1e4f1e26f3347d5d7ddd3841f9f4\": \"tdx-b300-prod\",\n    \
             \"00000000000000000000000000000001\": \"tdx-b300-prod\",",
            1,
        );
        assert_ne!(added, text);
        let added_sha = hex::encode(Sha256::digest(added.as_bytes()));
        let err = verify(added.as_bytes(), &bundle, &root, TAG, &added_sha);
        assert!(err.contains("the attestation's subject is"), "{err}");

        // The bundle: the DSSE signature, the SET, the inclusion proof, the
        // log body and the certificate, each altered in one place.
        let json: serde_json::Value = serde_json::from_slice(&bundle).unwrap();
        // Untampered, the bundle re-serialized still verifies, so each
        // refusal below is the edit's.
        platform_endorsements::verify(
            &file,
            &serde_json::to_vec(&json).unwrap(),
            &root,
            TAG,
            SHA256,
        )
        .unwrap();
        type Edit = fn(&mut serde_json::Value);
        let cases: Vec<(&str, Edit, &str)> = vec![
            (
                "signature",
                |b| flip_b64(&mut b["dsseEnvelope"]["signatures"][0]["sig"]),
                "the DSSE signature does not verify",
            ),
            (
                "payload",
                |b| flip_b64(&mut b["dsseEnvelope"]["payload"]),
                "the DSSE signature does not verify",
            ),
            (
                "SET",
                |b| {
                    flip_b64(
                        &mut b["verificationMaterial"]["tlogEntries"][0]["inclusionPromise"]["signedEntryTimestamp"],
                    )
                },
                "the Rekor SET",
            ),
            (
                "inclusion proof",
                |b| {
                    flip_b64(
                        &mut b["verificationMaterial"]["tlogEntries"][0]["inclusionProof"]["hashes"]
                            [0],
                    )
                },
                "does not reach the stated root hash",
            ),
            (
                "certificate",
                |b| flip_b64(&mut b["verificationMaterial"]["certificate"]["rawBytes"]),
                "does not chain to the trusted root",
            ),
            (
                "log body",
                |b| flip_b64(&mut b["verificationMaterial"]["tlogEntries"][0]["canonicalizedBody"]),
                "",
            ),
            (
                "log index",
                |b| {
                    b["verificationMaterial"]["tlogEntries"][0]["logIndex"] = "1".into();
                },
                "the Rekor SET",
            ),
        ];
        for (name, edit, needle) in cases {
            let mut tampered = json.clone();
            edit(&mut tampered);
            let bytes = serde_json::to_vec(&tampered).unwrap();
            let err = verify(&file, &bytes, &root, TAG, SHA256);
            assert!(err.contains(needle), "{name}: {err}");
        }

        // A genuine, logged and included Rekor entry for another signing of
        // the same Statement (the `platform-v0.1.10` attestation's) in place
        // of this bundle's: the entry is about another signature and signer,
        // which the entry-body binding refuses before anything else.
        let other: serde_json::Value =
            serde_json::from_slice(&fixture("platform-v0.1.10.sigstore.json")).unwrap();
        let mut swapped = json.clone();
        swapped["verificationMaterial"]["tlogEntries"] =
            other["verificationMaterial"]["tlogEntries"].clone();
        let err = verify(
            &file,
            &serde_json::to_vec(&swapped).unwrap(),
            &root,
            TAG,
            SHA256,
        );
        assert!(err.contains("the Rekor entry"), "{err}");
    }

    fn flip_b64(s: &mut serde_json::Value) {
        use base64::Engine as _;
        let engine = base64::engine::general_purpose::STANDARD;
        let mut bytes = engine.decode(s.as_str().unwrap()).unwrap();
        let at = bytes.len() - 10;
        bytes[at] ^= 1;
        *s = engine.encode(bytes).into();
    }

    /// The update step selects, from an attestations response holding both
    /// releases' bundles for the same file, the one for the tag asked for.
    #[test]
    fn vendoring_selects_the_bundle_for_the_tag() {
        let (file, bundle, root) = vendored();
        let other = fixture("platform-v0.1.10.sigstore.json");
        let response = format!(
            r#"{{"attestations":[{{"bundle":{}}},{{"bundle":{}}}]}}"#,
            String::from_utf8(other.clone()).unwrap().trim_end(),
            String::from_utf8(bundle.clone()).unwrap().trim_end(),
        );
        let (selected, verified) = select_bundle(&file, response.as_bytes(), &root, TAG).unwrap();
        assert_eq!(selected, bundle);
        assert_eq!(verified.rekor_log_index, 3_725_492_526);
        let err = select_bundle(&file, response.as_bytes(), &root, "platform-v0.1.12")
            .unwrap_err()
            .to_string();
        assert!(err.contains("none of the 2 attestations"), "{err}");
    }

    #[test]
    fn release_tags_follow_the_release_workflows_pattern() {
        for good in ["platform-v0.1.11", "platform-v1.0.0", "platform-v10.20.30"] {
            assert!(platform_endorsements::is_release_tag(good), "{good}");
            assert!(
                eidola_common::engine_deployment::deployment::is_platform_release_tag(good),
                "{good}"
            );
        }
        for bad in [
            "platform-v0.1",
            "platform-v0.01.1",
            "v0.1.11",
            "platform-v0.1.11-rc1",
            "platform-v0.1.11/..",
            "platform-v.1.1",
        ] {
            assert!(!platform_endorsements::is_release_tag(bad), "{bad}");
            assert!(
                !eidola_common::engine_deployment::deployment::is_platform_release_tag(bad),
                "{bad}"
            );
        }
        assert!(
            platform_endorsements::release_version("platform-v0.1.11")
                < platform_endorsements::release_version("platform-v0.2.0")
        );
    }

    /// Every release under the vendored tree verifies as its tag, and there
    /// is one: the newest stays vendored while no deployment names it, as
    /// the baseline newer provider releases are compared against.
    #[test]
    fn every_vendored_release_verifies() {
        let read = |path: &str| {
            std::fs::read(workspace().join(path)).with_context(|| format!("reading {path}"))
        };
        let newest = newest_vendored(&workspace()).unwrap();
        assert!(
            newest.is_some(),
            "no release under {PLATFORM_ENDORSEMENTS_ROOT}"
        );
        let mut tags = Vec::new();
        for entry in std::fs::read_dir(workspace().join(PLATFORM_ENDORSEMENTS_ROOT)).unwrap() {
            let tag = entry.unwrap().file_name().into_string().unwrap();
            let (file, _) = platform_endorsement_files(&tag);
            let sha256 = hex::encode(Sha256::digest(read(&file).unwrap()));
            verify_vendored(&tag, &sha256, &read).unwrap_or_else(|e| panic!("{tag}: {e:#}"));
            tags.push(tag);
        }
        assert!(tags.contains(newest.as_ref().unwrap()));
    }
}
