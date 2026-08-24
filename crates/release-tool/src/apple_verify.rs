//! The mechanical half of the `apple_signature_reconstructs` claim.
//!
//! The engineer affirms that applying the published detached signature
//! material to the reproducible unsigned build yields exactly the shipped
//! signed artifact. That sentence is checkable, so `attest` checks it and
//! refuses to offer the claim when it does not hold — the same posture
//! `manifest_reproduced` already has, where the tool re-fetches and
//! re-compares the manifest rather than trusting that a verify pass
//! happened. A claim a person can affirm but nothing can check is the
//! failure mode the whole scheme exists to avoid.
//!
//! Containers are unpacked with the *installer's* extractor
//! ([`eidola_app_core::updater::install::unpack_zip`]) rather than a second
//! one written here: what the release officer proves reconstructs must be
//! unpacked under exactly the rules a user's client will apply.
//!
//! What is compared is the reconstructed tree against the shipped
//! container's contents, entry for entry — bytes and the executable bit.
//! The container is transport; the tree is what a user installs and what
//! macOS validates, and comparing trees also means this check needs no
//! zip *writer* and therefore cannot disagree with the shipping recipe
//! (`scripts/pack-shipping-zip.sh`) about how a directory becomes a file.
//! The hashes that reach the attestation are taken from the published
//! files exactly as published.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// The manifest row recording the reproducible unsigned macOS container.
///
/// The claim says "the reproducible unsigned build recorded in
/// artifact-manifest.json", so the input is bound to that row rather than
/// to whatever file the operator happened to pass. The row arrives with
/// manifest schema 3; until a release emits that schema there is nothing
/// to bind to, and the claim is not offerable — which is the manifest's
/// own accept-before-emit rotation doing the sequencing.
pub const MACOS_UNSIGNED_ZIP_KEY: &str = "eidola-gui-macos-universal-zip";

/// The three published objects this check composes.
pub struct AppleAssets {
    /// The reproducible unsigned container, whose hash the manifest records.
    pub unsigned_artifact: PathBuf,
    /// The detached signature material.
    pub signature_bundle: PathBuf,
    /// The signed container a browser downloads.
    pub shipped_artifact: PathBuf,
}

/// What the check established, in the form the attestation records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppleReconstruction {
    pub unsigned_artifact_sha256: String,
    pub signature_bundle_sha256: String,
    pub shipped_artifact_sha256: String,
    /// The bundle directory both containers hold, e.g. `Eidola.app`.
    pub bundle_name: String,
    /// `None` for a signature with no Developer ID behind it. Recorded as
    /// it was read, never defaulted: "this artifact names no team" is a
    /// fact a reader is entitled to.
    pub team_id: Option<String>,
    pub signing_identifier: String,
    pub hardened_runtime: bool,
}

/// Reconstruct the signed bundle from the unsigned build plus the detached
/// material, and require it to equal the shipped artifact's contents.
///
/// `scratch` must be an empty directory the caller owns; this leaves its
/// working trees there for the caller to drop.
pub fn verify_reconstruction(assets: &AppleAssets, scratch: &Path) -> Result<AppleReconstruction> {
    let unsigned_bytes = read_file(&assets.unsigned_artifact)?;
    let bundle_bytes = read_file(&assets.signature_bundle)?;
    let shipped_bytes = read_file(&assets.shipped_artifact)?;

    let reconstructed_root = scratch.join("reconstructed");
    let shipped_root = scratch.join("shipped");
    let envelope_root = scratch.join("envelope");
    for root in [&reconstructed_root, &shipped_root, &envelope_root] {
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
    }

    unpack(&unsigned_bytes, &reconstructed_root, "the unsigned build")?;
    unpack(&shipped_bytes, &shipped_root, "the shipped artifact")?;
    unpack(&bundle_bytes, &envelope_root, "the signature bundle")?;

    let bundle_name = sole_bundle_name(&reconstructed_root, "the unsigned build")?;
    let shipped_bundle_name = sole_bundle_name(&shipped_root, "the shipped artifact")?;
    if bundle_name != shipped_bundle_name {
        bail!(
            "the unsigned build holds `{bundle_name}` but the shipped artifact holds \
             `{shipped_bundle_name}` — these are not two forms of one application"
        );
    }

    let reconstructed = reconstructed_root.join(&bundle_name);
    eidola_apple::apply(&reconstructed, &envelope_root).with_context(|| {
        format!(
            "applying {} to {}",
            assets.signature_bundle.display(),
            assets.unsigned_artifact.display()
        )
    })?;

    compare_trees(&reconstructed, &shipped_root.join(&bundle_name))?;

    let facts = eidola_apple::inspect(&reconstructed)
        .context("reading back what the reconstructed bundle claims about its signature")?;

    Ok(AppleReconstruction {
        unsigned_artifact_sha256: sha256_hex(&unsigned_bytes),
        signature_bundle_sha256: sha256_hex(&bundle_bytes),
        shipped_artifact_sha256: sha256_hex(&shipped_bytes),
        bundle_name,
        team_id: facts.team_id,
        signing_identifier: facts.identifier,
        hardened_runtime: facts.hardened_runtime,
    })
}

/// The `sha256:`-prefixed hash a manifest records for one artifact key, as
/// a bare lowercase hex string.
pub fn manifest_recorded_sha256(manifest_bytes: &[u8], key: &str) -> Result<String> {
    let manifest: serde_json::Value =
        serde_json::from_slice(manifest_bytes).context("parsing artifact-manifest.json")?;
    let recorded = manifest
        .get("artifacts")
        .and_then(|artifacts| artifacts.get(key))
        .and_then(|entry| entry.get("sha256"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "artifact-manifest.json records no `sha256` for `{key}`. That row arrives \
                 with manifest schema 3; until a release emits it there is no recorded \
                 unsigned build for the claim to name (releases/README.md, \"Rotating \
                 document schema versions\")."
            )
        })?;
    recorded
        .strip_prefix("sha256:")
        .map(str::to_ascii_lowercase)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "artifact-manifest.json records `{key}` as `{recorded}`, not `sha256:…`"
            )
        })
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

fn unpack(bytes: &[u8], dest: &Path, label: &str) -> Result<()> {
    eidola_app_core::updater::install::unpack_zip(bytes, dest, label)
        .with_context(|| format!("unpacking {label}"))?;
    Ok(())
}

/// The one top-level directory a shipping container holds.
///
/// Derived rather than assumed, and required to be *sole*: a container
/// with a second top-level entry is not the shape the packer produces, and
/// picking one of them would be choosing which application to verify.
fn sole_bundle_name(root: &Path, label: &str) -> Result<String> {
    let mut names: Vec<String> = Vec::new();
    for entry in
        std::fs::read_dir(root).with_context(|| format!("reading {} in {label}", root.display()))?
    {
        let entry = entry?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|name| anyhow::anyhow!("{label} holds a non-UTF-8 entry `{name:?}`"))?;
        if !entry.file_type()?.is_dir() {
            bail!("{label} holds `{name}` at its top level, which is not a directory");
        }
        names.push(name);
    }
    match names.len() {
        1 => Ok(names.remove(0)),
        0 => bail!("{label} is empty"),
        n => {
            names.sort();
            bail!("{label} holds {n} top-level entries ({names:?}), expected exactly one bundle")
        }
    }
}

/// What a tree comparison records per path. Symbolic links never appear:
/// the extractor refuses them, and reconstruction creates none.
#[derive(Debug, PartialEq, Eq)]
enum Entry {
    Directory,
    File { sha256: String, executable: bool },
}

fn compare_trees(reconstructed: &Path, shipped: &Path) -> Result<()> {
    let left = collect_tree(reconstructed)?;
    let right = collect_tree(shipped)?;

    for (path, entry) in &left {
        match right.get(path) {
            None => bail!(
                "reconstruction produced `{}`, which the shipped artifact does not contain",
                path.display()
            ),
            Some(other) if other != entry => bail!(
                "reconstruction and the shipped artifact disagree about `{}`: {entry:?} vs {other:?}",
                path.display()
            ),
            Some(_) => {}
        }
    }
    if let Some(path) = right.keys().find(|path| !left.contains_key(*path)) {
        bail!(
            "the shipped artifact contains `{}`, which reconstruction did not produce",
            path.display()
        );
    }
    Ok(())
}

fn collect_tree(root: &Path) -> Result<BTreeMap<PathBuf, Entry>> {
    let mut out = BTreeMap::new();
    collect_into(root, Path::new(""), &mut out)?;
    Ok(out)
}

fn collect_into(
    absolute: &Path,
    relative: &Path,
    out: &mut BTreeMap<PathBuf, Entry>,
) -> Result<()> {
    for entry in std::fs::read_dir(absolute)
        .with_context(|| format!("reading directory {}", absolute.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let child = relative.join(entry.file_name());
        // `symlink_metadata` answers what the entry *is*, not what it
        // resolves to: a comparison that followed links would compare the
        // same target twice and call two different trees equal.
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("reading metadata for {}", path.display()))?;
        if metadata.is_symlink() {
            bail!(
                "`{}` is a symbolic link; neither the extractor nor reconstruction produces one",
                child.display()
            );
        }
        if metadata.is_dir() {
            out.insert(child.clone(), Entry::Directory);
            collect_into(&path, &child, out)?;
        } else {
            use std::os::unix::fs::PermissionsExt;
            let bytes = read_file(&path)?;
            out.insert(
                child,
                Entry::File {
                    sha256: sha256_hex(&bytes),
                    executable: metadata.permissions().mode() & 0o111 != 0,
                },
            );
        }
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(out, "{byte:02x}").expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    //! Exercised against the committed **ad-hoc**-signed round-trip
    //! fixture: `settled/` is the reproducible unsigned input, `detached/`
    //! the signature material, `signed/` the golden output. Nothing in the
    //! claim or in this check mentions Developer ID — an ad-hoc signature
    //! walks the identical three-step chain, and only the identity inside
    //! the signature changes when a certificate exists. That is what makes
    //! the whole layer testable before any key does.

    use super::*;

    use std::io::Write as _;

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/fixtures/apple-roundtrip/synthetic-universal")
    }

    /// Zip a tree the way a shipping container holds one: entries begin at
    /// the tree's children, so a tree holding `Fixture.app` produces
    /// `Fixture.app/…`.
    fn zip_tree(root: &Path) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        add_tree(&mut writer, root, Path::new(""));
        writer.finish().unwrap().into_inner()
    }

    fn add_tree(
        writer: &mut zip::ZipWriter<std::io::Cursor<Vec<u8>>>,
        absolute: &Path,
        relative: &Path,
    ) {
        use std::os::unix::fs::PermissionsExt;

        let mut entries: Vec<_> = std::fs::read_dir(absolute)
            .unwrap()
            .map(|entry| entry.unwrap())
            .collect();
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let child = relative.join(entry.file_name());
            let name = child.to_str().unwrap().to_string();
            let metadata = entry.metadata().unwrap();
            let options = zip::write::SimpleFileOptions::default()
                .unix_permissions(metadata.permissions().mode());
            if metadata.is_dir() {
                writer.add_directory(name, options).unwrap();
                add_tree(writer, &entry.path(), &child);
            } else {
                writer.start_file(name, options).unwrap();
                writer
                    .write_all(&std::fs::read(entry.path()).unwrap())
                    .unwrap();
            }
        }
    }

    struct Case {
        _scratch: tempfile::TempDir,
        assets: AppleAssets,
        work: PathBuf,
    }

    /// The three containers, written to disk, with each tree free to be
    /// perturbed first.
    fn case(mutate: impl FnOnce(&Path, &Path, &Path)) -> Case {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path();
        let unsigned = root.join("unsigned-tree");
        let signed = root.join("signed-tree");
        let detached = root.join("detached-tree");
        copy_tree(&fixtures().join("settled"), &unsigned);
        copy_tree(&fixtures().join("signed"), &signed);
        copy_tree(&fixtures().join("detached"), &detached);
        mutate(&unsigned, &signed, &detached);

        let assets = AppleAssets {
            unsigned_artifact: root.join("unsigned.zip"),
            signature_bundle: root.join("sigbundle.zip"),
            shipped_artifact: root.join("shipped.zip"),
        };
        std::fs::write(&assets.unsigned_artifact, zip_tree(&unsigned)).unwrap();
        std::fs::write(&assets.signature_bundle, zip_tree(&detached)).unwrap();
        std::fs::write(&assets.shipped_artifact, zip_tree(&signed)).unwrap();

        let work = root.join("work");
        std::fs::create_dir(&work).unwrap();
        Case {
            _scratch: scratch,
            assets,
            work,
        }
    }

    fn copy_tree(source: &Path, destination: &Path) {
        std::fs::create_dir_all(destination).unwrap();
        for entry in std::fs::read_dir(source).unwrap() {
            let entry = entry.unwrap();
            let target = destination.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).unwrap();
            }
        }
    }

    #[test]
    fn reconstruction_matches_the_shipped_artifact() {
        let case = case(|_, _, _| {});
        let result = verify_reconstruction(&case.assets, &case.work).unwrap();

        assert_eq!(result.bundle_name, "Fixture.app");
        assert_eq!(result.signing_identifier, "ai.eidola.fixture");
        assert_eq!(result.team_id, None, "an ad-hoc signature names no team");
        assert!(result.hardened_runtime);
        assert_eq!(
            result.shipped_artifact_sha256,
            sha256_hex(&std::fs::read(&case.assets.shipped_artifact).unwrap()),
            "the attested hash is the published file's own, not a rehash of anything"
        );
    }

    #[test]
    fn a_shipped_artifact_that_is_not_the_reconstruction_is_refused() {
        // One byte of the sealed resource file — nothing the signature
        // material would produce.
        let case = case(|_, signed, _| {
            let seal = signed.join("Fixture.app/Contents/_CodeSignature/CodeResources");
            let mut bytes = std::fs::read(&seal).unwrap();
            bytes[0] ^= 0xff;
            std::fs::write(&seal, bytes).unwrap();
        });
        let error = verify_reconstruction(&case.assets, &case.work)
            .expect_err("a shipped artifact the material does not produce must be refused");
        let message = format!("{error:?}");
        assert!(
            message.contains("disagree about") && message.contains("CodeResources"),
            "got: {message}"
        );
    }

    #[test]
    fn a_shipped_artifact_with_an_extra_file_is_refused() {
        let case = case(|_, signed, _| {
            std::fs::write(signed.join("Fixture.app/Contents/extra"), b"rider").unwrap();
        });
        let error = verify_reconstruction(&case.assets, &case.work)
            .expect_err("an unreconstructed file rides along unsigned unless this refuses");
        let message = format!("{error:?}");
        assert!(
            message.contains("which reconstruction did not produce"),
            "got: {message}"
        );
    }

    #[test]
    fn the_wrong_unsigned_build_is_refused_rather_than_corrupted() {
        let case = case(|unsigned, _, _| {
            std::fs::write(unsigned.join("Fixture.app/Contents/Info.plist"), b"other").unwrap();
        });
        let error = verify_reconstruction(&case.assets, &case.work)
            .expect_err("signature material pins the build it was detached from");
        let message = format!("{error:?}");
        assert!(message.contains("Info.plist"), "got: {message}");
    }

    #[test]
    fn a_container_holding_more_than_one_bundle_is_refused() {
        let case = case(|unsigned, _, _| {
            std::fs::create_dir(unsigned.join("Second.app")).unwrap();
        });
        let error = verify_reconstruction(&case.assets, &case.work)
            .expect_err("choosing between two bundles is not this check's decision");
        let message = format!("{error:?}");
        assert!(
            message.contains("expected exactly one bundle"),
            "got: {message}"
        );
    }

    #[test]
    fn a_manifest_without_the_unsigned_row_offers_no_claim() {
        let manifest = br#"{"schema_version": 2, "artifacts": {}}"#;
        let error = manifest_recorded_sha256(manifest, MACOS_UNSIGNED_ZIP_KEY).unwrap_err();
        let message = format!("{error}");
        assert!(message.contains("manifest schema 3"), "got: {message}");
    }

    #[test]
    fn a_recorded_row_reads_back_as_bare_hex() {
        let manifest = br#"{
            "schema_version": 3,
            "artifacts": {
                "eidola-gui-macos-universal-zip": {
                    "type": "file",
                    "sha256": "sha256:ABCD"
                }
            }
        }"#;
        assert_eq!(
            manifest_recorded_sha256(manifest, MACOS_UNSIGNED_ZIP_KEY).unwrap(),
            "abcd"
        );
    }
}
