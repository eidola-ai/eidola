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
/// artifact-manifest.json", so where the claim is affirmed the input is
/// bound to that row rather than to whatever file the operator happened to
/// pass. The row arrives with manifest schema 3; see [`AppleBinding`] for
/// what that means for a release that publishes the assets before any
/// claim about them exists.
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

/// What the check established, in the form the attestation records it —
/// and the bytes it established it about.
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
    /// Always `true`: [`require_signing_recipe`] refuses anything else.
    /// Kept as a field because it is read back off the reconstructed
    /// artifact rather than assumed, and printed as the receipt for that.
    pub hardened_runtime: bool,
    /// The two objects a release publishes, written out of the very bytes
    /// that were hashed above.
    ///
    /// Private with accessors, because a caller that can name the
    /// operator's original path can upload something else. Attestation is
    /// interactive — key inspection, an affirmation per claim, a device
    /// PIN — so minutes pass between the hash and the upload, and a path
    /// re-read then is not the file that was checked. Holding an
    /// `AppleReconstruction` is the evidence that these two files are the
    /// ones the hashes describe.
    staged_shipped_artifact: PathBuf,
    staged_signature_bundle: PathBuf,
}

impl AppleReconstruction {
    /// The signed container, staged from the verified bytes.
    pub fn staged_shipped_artifact(&self) -> &Path {
        &self.staged_shipped_artifact
    }

    /// The detached material, staged from the verified bytes.
    pub fn staged_signature_bundle(&self) -> &Path {
        &self.staged_signature_bundle
    }
}

/// Reconstruct the signed bundle from the unsigned build plus the detached
/// material, and require it to equal the shipped artifact's contents.
///
/// `scratch` must be an empty directory the caller owns; this leaves its
/// working trees and the staged publishable bytes there, so it must
/// outlive the upload.
pub fn verify_reconstruction(assets: &AppleAssets, scratch: &Path) -> Result<AppleReconstruction> {
    // The two containers a client fetches are bounded on the wire by the
    // installer's own ceiling, so a release published above it verifies
    // here and is refused by every client — the same "publishable in a
    // shape clients reject" failure the attestant-key pre-flight exists to
    // prevent. Checked against the installer's constant rather than a
    // number repeated here. The shipped artifact is deliberately not
    // bounded: no client fetches it, and inventing a limit a browser
    // download does not have would refuse a release nothing would reject.
    let unsigned_bytes = read_bounded(&assets.unsigned_artifact, "the unsigned build")?;
    let bundle_bytes = read_bounded(&assets.signature_bundle, "the signature bundle")?;
    let shipped_bytes = read_file(&assets.shipped_artifact)?;

    // Staged before anything else can fail, and from the bytes already in
    // hand rather than by copying the path a second time.
    let staged = scratch.join("publish");
    std::fs::create_dir_all(&staged).with_context(|| format!("creating {}", staged.display()))?;
    let staged_shipped_artifact = staged.join("shipped");
    let staged_signature_bundle = staged.join("signature");
    write_file(&staged_shipped_artifact, &shipped_bytes)?;
    write_file(&staged_signature_bundle, &bundle_bytes)?;

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
    require_signing_recipe(&facts)?;

    Ok(AppleReconstruction {
        unsigned_artifact_sha256: sha256_hex(&unsigned_bytes),
        signature_bundle_sha256: sha256_hex(&bundle_bytes),
        shipped_artifact_sha256: sha256_hex(&shipped_bytes),
        bundle_name,
        team_id: facts.team_id,
        signing_identifier: facts.identifier,
        hardened_runtime: facts.hardened_runtime,
        staged_shipped_artifact,
        staged_signature_bundle,
    })
}

/// The one signing-recipe property an installed client asserts rather
/// than reads.
///
/// Every Eidola macOS signature is made with `--options runtime`, so the
/// installer's plan states the hardened runtime as a fixed fact instead of
/// taking it from a document a coerced signer writes
/// (`crates/eidola-app-core/AGENTS.md`). Asserting on the client is only
/// safe if the release side cannot publish something that contradicts it —
/// and an artifact signed without the flag is invisible to every other
/// check here: it reconstructs perfectly and compares equal to what
/// shipped. Publishing it would leave a browser download without the
/// protection for good, and every self-update refused at staging. So this
/// is where the recipe is enforced, before publication is possible.
///
/// Deliberately just the one property, because it is exactly what
/// `install::ExpectedSignature` compares. Entitlements and the
/// notarization ticket are not in that comparison — the round-trip fixture
/// carries entitlements and no ticket — and refusing on them here would
/// invent a requirement no client applies.
fn require_signing_recipe(facts: &eidola_apple::SignatureFacts) -> Result<()> {
    if !facts.hardened_runtime {
        bail!(
            "the reconstructed bundle does not carry the hardened runtime, which every \
             Eidola macOS signature is made with and every installed client asserts \
             rather than reads. Publishing it would ship a browser download without that \
             protection and a self-update every client refuses at staging."
        );
    }
    Ok(())
}

/// The `sha256:`-prefixed hash a manifest records for one artifact key, as
/// a bare lowercase hex string.
///
/// `Ok(None)` means the manifest records no such row — a fact about the
/// manifest's schema, and a decision for the caller ([`AppleBinding`]).
/// `Err` is reserved for a row that exists and cannot be used, which is
/// wrong under every schema.
pub fn manifest_recorded_sha256(manifest_bytes: &[u8], key: &str) -> Result<Option<String>> {
    let manifest: serde_json::Value =
        serde_json::from_slice(manifest_bytes).context("parsing artifact-manifest.json")?;
    let Some(recorded) = manifest
        .get("artifacts")
        .and_then(|artifacts| artifacts.get(key))
        .and_then(|entry| entry.get("sha256"))
    else {
        return Ok(None);
    };
    let recorded = recorded.as_str().ok_or_else(|| {
        anyhow::anyhow!("artifact-manifest.json records `{key}`'s `sha256` as a non-string")
    })?;
    recorded
        .strip_prefix("sha256:")
        .map(|hex| Some(hex.to_ascii_lowercase()))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "artifact-manifest.json records `{key}` as `{recorded}`, not `sha256:…`"
            )
        })
}

/// What a release may say about its macOS signing outputs.
///
/// The two orderings this sits between move at different times, and the
/// releases in between are real: the claim arrives in the *binding*
/// templates one release after it is committed, while the manifest row
/// naming the unsigned build arrives with manifest schema 3, whose own
/// accept-before-emit rotation is independent. So a release can genuinely
/// have signing outputs worth publishing and nothing yet to claim about
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppleBinding {
    /// No signing outputs were supplied, and nothing asks for them.
    Absent,
    /// Published as data. The reconstruction was still checked — that is
    /// what an operator gets for supplying the inputs — but no claim is
    /// affirmed and no field is recorded, so nothing is bound to the
    /// manifest.
    PublishedOnly,
    /// Published *and* attested: the claim is affirmed, so the unsigned
    /// build must be the one the manifest records.
    Attested,
}

/// Decide what this release may do with the signing outputs, and refuse
/// every combination that would let a claim outrun its evidence.
///
/// The manifest binding is required **exactly where the claim is
/// affirmed**, because that is the sentence that names the manifest.
/// Requiring it unconditionally would mean the first release able to
/// publish these assets could not publish them; skipping it where the row
/// exists would ignore a check that costs nothing. So a present row is
/// always honoured and an absent row is only fatal to a claim.
pub fn apple_binding(
    claim_binds: bool,
    reconstruction: Option<&AppleReconstruction>,
    manifest_row: Option<&str>,
) -> Result<AppleBinding> {
    if let (Some(reconstruction), Some(recorded)) = (reconstruction, manifest_row)
        && reconstruction.unsigned_artifact_sha256 != recorded
    {
        bail!(
            "the unsigned build hashes to {}, but artifact-manifest.json records {recorded} \
             for `{MACOS_UNSIGNED_ZIP_KEY}` — the reconstruction was run against a build \
             this release did not measure",
            reconstruction.unsigned_artifact_sha256,
        );
    }

    if !claim_binds {
        return Ok(match reconstruction {
            Some(_) => AppleBinding::PublishedOnly,
            None => AppleBinding::Absent,
        });
    }

    if reconstruction.is_none() {
        bail!(
            "the binding templates declare the macOS reconstruction claim, so this release \
             cannot be attested without the signing outputs to check it against. Supply \
             --apple-unsigned-artifact, --apple-signature-bundle, and \
             --apple-shipped-artifact."
        );
    }
    if manifest_row.is_none() {
        bail!(
            "the macOS reconstruction claim names the unsigned build recorded in \
             artifact-manifest.json, but this manifest records no `{MACOS_UNSIGNED_ZIP_KEY}`. \
             That row arrives with manifest schema 3; the claim cannot be affirmed until a \
             release emits it (releases/README.md, \"Rotating document schema versions\")."
        );
    }
    Ok(AppleBinding::Attested)
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

/// Read a file the installer will later fetch over the wire, refusing one
/// larger than the installer's own ceiling.
fn read_bounded(path: &Path, label: &str) -> Result<Vec<u8>> {
    read_within(
        path,
        label,
        eidola_app_core::updater::install::MAX_CONTAINER_BYTES,
    )
}

/// The ceiling, applied the way the installer applies it: in two tiers,
/// with the authoritative one over the bytes actually in hand.
///
/// The installer refuses an oversized `Content-Length` early and *then*
/// bounds the body with a running count, because a claimed length is a
/// claim and the bytes are the fact. A stat is this side's claimed length:
/// it is sampled before the read, and the read is what gets staged and
/// published. Trusting it would leave the two sides applying the same
/// number under different rules, which is exactly what sharing the
/// constant was meant to prevent.
///
/// The difference is reachable without anyone being adversarial — the
/// operator owns this filesystem and races nobody. A file another process
/// is still writing (a signing job's artifact download that has not
/// finished) reports one length and yields another; a stream-like path
/// reports zero and yields everything it is fed. Refusing on a number that
/// is not a property of the release is the defect, not the race.
///
/// So the stat stays as the cheap early refusal it is good at, and the
/// count of bytes returned is what decides. One byte past the ceiling is
/// enough to know a file is over it, and is also the most this will ever
/// hold.
fn read_within(path: &Path, label: &str, max: u64) -> Result<Vec<u8>> {
    use std::io::Read as _;

    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;

    if let Ok(metadata) = file.metadata()
        && metadata.len() > max
    {
        bail!(oversized(label, max));
    }

    let mut bytes = Vec::new();
    file.by_ref()
        .take(max.saturating_add(1))
        .read_to_end(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    if bytes.len() as u64 > max {
        bail!(oversized(label, max));
    }
    Ok(bytes)
}

fn oversized(label: &str, max: u64) -> String {
    format!(
        "{label} is larger than the {max} bytes an installed client will download — \
         publishing it would produce a release every client refuses before it \
         reconstructs anything"
    )
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
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
    fn an_unrecorded_row_is_a_fact_about_the_manifest_not_an_error() {
        let manifest = br#"{"schema_version": 2, "artifacts": {}}"#;
        assert_eq!(
            manifest_recorded_sha256(manifest, MACOS_UNSIGNED_ZIP_KEY).unwrap(),
            None
        );
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
            Some("abcd".to_string())
        );
    }

    #[test]
    fn a_row_that_exists_and_cannot_be_used_is_loud() {
        // Wrong under every schema, so it must not read as "absent".
        for manifest in [
            br#"{"artifacts": {"eidola-gui-macos-universal-zip": {"sha256": "abcd"}}}"#.as_slice(),
            br#"{"artifacts": {"eidola-gui-macos-universal-zip": {"sha256": 7}}}"#.as_slice(),
        ] {
            assert!(manifest_recorded_sha256(manifest, MACOS_UNSIGNED_ZIP_KEY).is_err());
        }
    }

    // ── What a release may say about its signing outputs ─────────────

    const ROW: &str = "aaaa";

    fn reconstruction_hashing(unsigned: &str) -> AppleReconstruction {
        AppleReconstruction {
            unsigned_artifact_sha256: unsigned.to_string(),
            signature_bundle_sha256: "bbbb".into(),
            shipped_artifact_sha256: "cccc".into(),
            bundle_name: "Fixture.app".into(),
            team_id: None,
            signing_identifier: "ai.eidola.fixture".into(),
            hardened_runtime: true,
            staged_shipped_artifact: PathBuf::new(),
            staged_signature_bundle: PathBuf::new(),
        }
    }

    /// The transition release: signing outputs exist and are worth
    /// publishing, but the binding templates declare no claim about them
    /// *and* the emitted manifest predates the row that would bind the
    /// unsigned build. Requiring the row here would mean the first release
    /// able to publish these assets could not publish them.
    #[test]
    fn assets_publish_before_any_claim_or_manifest_row_exists() {
        let reconstruction = reconstruction_hashing(ROW);
        assert_eq!(
            apple_binding(false, Some(&reconstruction), None).unwrap(),
            AppleBinding::PublishedOnly
        );
    }

    #[test]
    fn a_present_row_is_honoured_even_with_no_claim_to_affirm() {
        let reconstruction = reconstruction_hashing("dddd");
        let error = apple_binding(false, Some(&reconstruction), Some(ROW)).unwrap_err();
        assert!(
            format!("{error}").contains("this release did not measure"),
            "a row that exists is checked whether or not anything is claimed"
        );
    }

    #[test]
    fn an_affirmed_claim_requires_the_manifest_row() {
        let reconstruction = reconstruction_hashing(ROW);
        assert_eq!(
            apple_binding(true, Some(&reconstruction), Some(ROW)).unwrap(),
            AppleBinding::Attested
        );

        let error = apple_binding(true, Some(&reconstruction), None).unwrap_err();
        let message = format!("{error}");
        assert!(message.contains("manifest schema 3"), "got: {message}");
    }

    #[test]
    fn an_affirmed_claim_requires_something_to_have_been_checked() {
        let error = apple_binding(true, None, Some(ROW)).unwrap_err();
        assert!(
            format!("{error}").contains("without the signing outputs"),
            "the claim may never outrun its evidence"
        );
    }

    #[test]
    fn no_inputs_and_no_claim_is_the_ordinary_release() {
        assert_eq!(
            apple_binding(false, None, None).unwrap(),
            AppleBinding::Absent
        );
        assert_eq!(
            apple_binding(false, None, Some(ROW)).unwrap(),
            AppleBinding::Absent
        );
    }

    // ── The two ceilings a client will later apply ───────────────────

    /// A container larger than the installer's wire bound would verify
    /// here and then be refused by every client before it reconstructed
    /// anything. Sparse: the refusal reads metadata, so no bytes are
    /// written or read.
    #[test]
    fn a_container_larger_than_a_client_will_fetch_is_refused() {
        let scratch = tempfile::tempdir().unwrap();
        let oversized = scratch.path().join("oversized.zip");
        let file = std::fs::File::create(&oversized).unwrap();
        file.set_len(eidola_app_core::updater::install::MAX_CONTAINER_BYTES + 1)
            .unwrap();
        drop(file);

        // Mapped to a length before unwrapping: a regression here means
        // the bytes *were* read, and the failure message must not be half
        // a gigabyte of them.
        let error = read_bounded(&oversized, "the unsigned build")
            .map(|bytes| bytes.len())
            .unwrap_err();
        let message = format!("{error}");
        assert!(
            message.contains("an installed client will download"),
            "got: {message}"
        );
    }

    /// The tier that matters: a path whose stat and whose bytes disagree.
    ///
    /// A named pipe reports a length of zero and yields whatever is
    /// written through it, so it is the disagreement without a race to
    /// arrange — the same shape as a file another process is still
    /// writing, which is the realistic case.
    #[cfg(unix)]
    #[test]
    fn the_ceiling_is_applied_to_the_bytes_not_to_a_sampled_length() {
        use std::io::Write as _;

        let scratch = tempfile::tempdir().unwrap();
        let pipe = scratch.path().join("pipe");
        assert!(
            std::process::Command::new("mkfifo")
                .arg(&pipe)
                .status()
                .unwrap()
                .success()
        );

        // Small enough to fit the pipe buffer, so the writer never blocks
        // and the reader's early exit cannot deadlock it.
        let writer = std::thread::spawn({
            let pipe = pipe.clone();
            move || {
                let mut handle = std::fs::OpenOptions::new().write(true).open(&pipe).unwrap();
                let _ = handle.write_all(&[0u8; 64]);
            }
        });

        assert_eq!(
            std::fs::metadata(&pipe).unwrap().len(),
            0,
            "the stat must understate, or this proves nothing"
        );
        let error = read_within(&pipe, "the unsigned build", 8)
            .map(|bytes| bytes.len())
            .unwrap_err();
        let _ = writer.join();

        let message = format!("{error}");
        assert!(
            message.contains("an installed client will download"),
            "got: {message}"
        );
    }

    /// The boundary the installer draws: it refuses only once a body has
    /// *exceeded* the ceiling, so exactly the ceiling is publishable.
    #[test]
    fn exactly_the_ceiling_is_allowed_and_one_byte_more_is_not() {
        let scratch = tempfile::tempdir().unwrap();
        let file = scratch.path().join("container");

        std::fs::write(&file, vec![0u8; 8]).unwrap();
        assert_eq!(
            read_within(&file, "the unsigned build", 8).unwrap().len(),
            8
        );

        std::fs::write(&file, vec![0u8; 9]).unwrap();
        assert!(
            read_within(&file, "the unsigned build", 8)
                .map(|bytes| bytes.len())
                .is_err()
        );
    }

    /// The signed container is not bounded: no client fetches it, so a
    /// limit here would refuse a release nothing would reject.
    #[test]
    fn the_browser_download_carries_no_client_ceiling() {
        let case = case(|_, _, _| {});
        let shipped = std::fs::metadata(&case.assets.shipped_artifact)
            .unwrap()
            .len();
        assert!(shipped > 0);
        // Nothing in the happy path consults a ceiling for it.
        verify_reconstruction(&case.assets, &case.work).unwrap();
    }

    // ── The recipe a client asserts rather than reads ────────────────

    /// The fixture's *unsigned input* is itself an ad-hoc signed bundle
    /// made without `--options runtime`, so detaching it against itself
    /// yields genuinely valid material whose reconstruction is missing
    /// exactly one thing. Everything else about this release is correct:
    /// the material applies cleanly and the result equals what shipped.
    #[test]
    fn a_reconstruction_without_the_hardened_runtime_is_refused() {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path();

        let unsigned = root.join("unsigned-tree");
        let shipped = root.join("signed-tree");
        copy_tree(&fixtures().join("settled"), &unsigned);
        copy_tree(&fixtures().join("settled"), &shipped);

        let detached = root.join("detached-tree");
        eidola_apple::detach(
            &shipped.join("Fixture.app"),
            &unsigned.join("Fixture.app"),
            &detached,
        )
        .unwrap();

        let assets = AppleAssets {
            unsigned_artifact: root.join("unsigned.zip"),
            signature_bundle: root.join("sigbundle.zip"),
            shipped_artifact: root.join("shipped.zip"),
        };
        std::fs::write(&assets.unsigned_artifact, zip_tree(&unsigned)).unwrap();
        std::fs::write(&assets.signature_bundle, zip_tree(&detached)).unwrap();
        std::fs::write(&assets.shipped_artifact, zip_tree(&shipped)).unwrap();

        let work = root.join("work");
        std::fs::create_dir(&work).unwrap();

        let error = verify_reconstruction(&assets, &work)
            .expect_err("a release the updater would refuse at staging must not be publishable");
        let message = format!("{error:?}");
        assert!(message.contains("hardened runtime"), "got: {message}");
    }

    #[test]
    fn the_recipe_check_is_scoped_to_what_a_client_compares() {
        let recipe = |hardened| eidola_apple::SignatureFacts {
            team_id: None,
            identifier: "ai.eidola.fixture".into(),
            hardened_runtime: hardened,
            // Neither of these is in `install::ExpectedSignature`, so
            // neither may be a reason to refuse: the round-trip fixture
            // carries entitlements and no ticket, and it is a valid
            // artifact.
            entitlements_sha256: Some("00".repeat(32)),
            has_notarization_ticket: false,
        };
        require_signing_recipe(&recipe(true)).unwrap();
        assert!(require_signing_recipe(&recipe(false)).is_err());
    }

    // ── The bytes that get published ─────────────────────────────────

    /// Attestation is interactive, so minutes pass between the check and
    /// the upload. Replacing the operator's files in that window must not
    /// change what gets published.
    #[test]
    fn the_published_bytes_are_the_verified_ones_after_the_inputs_change() {
        let case = case(|_, _, _| {});
        let result = verify_reconstruction(&case.assets, &case.work).unwrap();

        std::fs::write(&case.assets.shipped_artifact, b"swapped after the check").unwrap();
        std::fs::write(&case.assets.signature_bundle, b"swapped after the check").unwrap();

        assert_eq!(
            sha256_hex(&std::fs::read(result.staged_shipped_artifact()).unwrap()),
            result.shipped_artifact_sha256
        );
        assert_eq!(
            sha256_hex(&std::fs::read(result.staged_signature_bundle()).unwrap()),
            result.signature_bundle_sha256
        );
    }
}
