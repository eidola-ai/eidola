//! `release-tool attest <tag>` — interactively render every claim, prompt
//! the engineer to type `yes` to affirm each, sign with their attestant
//! key via `cosign sign-blob` (the key can be a local PEM, a PKCS#11
//! URI for a YubiKey / SmartCard, or any cosign-supported KMS URI), and
//! upload the resulting Sigstore Bundle v0.3 + attestation JSON +
//! release.json index to the GitHub release.
//!
//! Before any signing or upload, the underlying key's algorithm is
//! cross-checked against the updater's allowlist (ECDSA-P256,
//! ECDSA-P384, or Ed25519) via
//! [`eidola_app_core::updater::human_attestation::classify_attestant_spki_algorithm`].
//! RSA / ECDSA-P521 / other algorithms are rejected up front — cosign
//! would happily sign with them, but every client would reject the
//! resulting attestation, leaving a `latest`-marked release no one
//! can install.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use chrono::Utc;
use sha2::{Digest, Sha256};

use crate::trust;

pub struct Args {
    pub workspace_root: PathBuf,
    pub repo: String,
    pub tag: String,
    /// Cosign `--key` value: a local PEM path, a PKCS#11 URI
    /// (`pkcs11:slot-id=...;object=...`), or any KMS URI cosign
    /// supports. Passed through to cosign verbatim.
    pub cosign_key: String,
    pub attestant_id: String,
    pub attestant_name: String,
    pub jurisdiction: String,
}

pub fn run(args: Args) -> Result<()> {
    require_tool("gh")?;
    require_tool("cosign")?;
    require_tool("git")?;

    let trust = trust::load(&args.workspace_root)?;
    let templates_path = args
        .workspace_root
        .join("releases/schema/attestation-templates-v1.json");
    let templates = eidola_attestation::load_from_path(&templates_path)?;

    validate_attestant_id(&args.attestant_id)?;
    if args.attestant_name.trim().is_empty() {
        bail!("--attestant-name must not be empty");
    }
    if args.jurisdiction.trim().is_empty() {
        bail!("--jurisdiction must not be empty");
    }
    if args.cosign_key.trim().is_empty() {
        bail!("--cosign-key must not be empty (path to PEM, pkcs11: URI, or KMS URI)");
    }
    // Software-keyed cosign keys are passphrase-encrypted. PKCS#11 and
    // KMS-backed keys don't use COSIGN_PASSWORD — the device prompts
    // for its own PIN. We can't easily distinguish here, so we warn
    // only when the path looks like a local file and the env var is
    // unset.
    if is_local_key_path(&args.cosign_key) && std::env::var("COSIGN_PASSWORD").is_err() {
        bail!(
            "--cosign-key looks like a local file but COSIGN_PASSWORD is not set. \
             Local cosign keys are passphrase-encrypted; export COSIGN_PASSWORD \
             (or empty string for a passphrase-less key) and retry."
        );
    }

    let release_version = args
        .tag
        .strip_prefix('v')
        .ok_or_else(|| anyhow::anyhow!("tag `{}` must start with `v`", args.tag))?
        .to_string();
    let git_commit = git_rev_parse(&args.workspace_root, &args.tag)?;
    let prev_tag = previous_release_tag(&args.workspace_root, &args.tag)?;
    let previous_release_git_commit = git_rev_parse(&args.workspace_root, &prev_tag)?;

    // Print the resolved SHAs in the same format as `release-verify`'s
    // diff header, before any prompts. The `diff_reviewed` claim later
    // echoes these same SHAs inside the signed bytes; surfacing them up
    // front gives the engineer an early visual cross-check against the
    // verify scrollback so they can spot drift before affirming anything.
    println!();
    println!("== commits being attested ==");
    println!("  previous: {prev_tag}  →  {previous_release_git_commit}");
    println!("  this:     {tag}  →  {git_commit}", tag = args.tag);
    println!();
    println!(
        "Confirm these SHAs match what `release-verify` showed. They will be\n\
         recorded verbatim in the signed attestation."
    );

    let manifest_path = args.workspace_root.join("artifact-manifest.json");
    let artifact_manifest_sha256 = sha256_hex(&fs::read(&manifest_path)?);

    let privacy_doc_path = args.workspace_root.join("docs/privacy-guarantees.md");
    let privacy_guarantees_doc_sha256 = match fs::read(&privacy_doc_path) {
        Ok(bytes) => sha256_hex(&bytes),
        Err(_) => bail!(
            "`docs/privacy-guarantees.md` not found. \
             The `privacy_guarantees_not_weakened` claim references its sha256; \
             create the document and commit it before attesting."
        ),
    };

    let spki_der = fetch_cosign_pubkey_spki_der(&args.cosign_key)?;
    let key_fingerprint_sha256 = sha256_hex(&spki_der);

    // Pre-flight: reject keys the updater can't verify (RSA, ECDSA-P521, …)
    // before we sign and publish. Without this, cosign would happily sign
    // with, say, a YubiKey RSA-2048 slot or an awskms RSA key — the upload
    // and `--latest` would succeed, but every client's
    // `verify_blob_signature_with_spki` rejects anything outside
    // ECDSA-P256 / ECDSA-P384 / Ed25519, leaving us with a "latest"
    // release no one can install. Same allowlist, same OID dispatch as the
    // updater (single source of truth in `eidola_app_core`).
    let attestant_algorithm =
        eidola_app_core::updater::human_attestation::classify_attestant_spki_algorithm(&spki_der)
            .with_context(|| {
            format!(
                "the public key for `--cosign-key {}` is not a supported attestant key type. \
             The eidola updater only accepts ECDSA-P256, ECDSA-P384, or Ed25519; \
             refusing to sign and publish a release the updater would reject. \
             Re-issue the cosign / KMS / PKCS#11 key with one of those algorithms and retry.",
                args.cosign_key
            )
        })?;
    println!();
    println!(
        "  attestant key algorithm: {} (sha256 SPKI fingerprint: {key_fingerprint_sha256})",
        attestant_algorithm.name()
    );

    // Surface a loud warning if the signing key is not yet pinned. Expected
    // for the very first release (the seed); harmful for subsequent ones.
    let is_pinned = trust
        .trusted_attestant_fingerprints
        .iter()
        .any(|fp| fp.eq_ignore_ascii_case(&key_fingerprint_sha256));
    if !is_pinned {
        eprintln!();
        eprintln!(
            "WARNING: signing key fingerprint `{key_fingerprint_sha256}` is NOT yet in\n\
             `trusted_attestant_fingerprints` in releases/trust/trust-constants.json. Clients\n\
             will reject this attestation until you commit + ship a release that adds the\n\
             fingerprint (signed by an already-trusted key). This is expected for the very\n\
             first release; ignore if intentional."
        );
        eprintln!();
    }

    let attested_at = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

    // Build the bare attestation skeleton. We splice in attestant_statement
    // + claims after the engineer affirms them.
    let mut attestation = serde_json::json!({
        "schema_version": 1,
        "release_version": release_version,
        "git_commit": git_commit,
        "previous_release_git_commit": previous_release_git_commit,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "privacy_guarantees_doc_sha256": privacy_guarantees_doc_sha256,
        "attestant": {
            "id": args.attestant_id,
            "name": args.attestant_name,
            "key_fingerprint_sha256": key_fingerprint_sha256,
            "jurisdiction": args.jurisdiction,
        },
        "attested_at": attested_at,
    });

    let attestation_clone = attestation.clone();
    let mut roots: BTreeMap<&str, &serde_json::Value> = BTreeMap::new();
    roots.insert("attestation", &attestation_clone);

    let (preamble, _) = eidola_attestation::render(
        &templates.attestant_statement_template.template,
        &templates.attestant_statement_template.sources,
        &roots,
    )
    .context("rendering attestant_statement_template")?;

    println!();
    println!("=== Attestant statement ===");
    print_quoted(&preamble);
    println!();
    affirm_or_abort("Type 'yes' to affirm the above preamble under penalty of perjury")?;

    let mut claims_json = serde_json::Map::new();
    for (claim_id, claim) in &templates.claims {
        let (rendered, fields) =
            eidola_attestation::render(&claim.template, &claim.sources, &roots)
                .with_context(|| format!("rendering claim `{claim_id}`"))?;

        println!();
        println!("=== Claim: {claim_id} ===");
        print_quoted(&rendered);
        println!();
        affirm_or_abort(&format!("Type 'yes' to affirm `{claim_id}`"))?;

        let mut claim_obj = serde_json::Map::new();
        claim_obj.insert("statement".to_string(), serde_json::Value::String(rendered));
        if !fields.is_empty() {
            let fields_value: serde_json::Map<String, serde_json::Value> = fields
                .into_iter()
                .map(|(k, v)| (k, serde_json::Value::String(v)))
                .collect();
            claim_obj.insert(
                "fields".to_string(),
                serde_json::Value::Object(fields_value),
            );
        }
        claims_json.insert(claim_id.clone(), serde_json::Value::Object(claim_obj));
    }

    if claims_json.is_empty() {
        bail!("templates file contains no claims — refusing to sign an empty attestation");
    }

    attestation.as_object_mut().unwrap().insert(
        "attestant_statement".to_string(),
        serde_json::Value::String(preamble),
    );
    attestation
        .as_object_mut()
        .unwrap()
        .insert("claims".to_string(), serde_json::Value::Object(claims_json));

    let attestation_filename = format!("attestation-{}.json", args.attestant_id);
    let bundle_filename = format!("attestation-{}.bundle.json", args.attestant_id);

    let tmp = tempfile::tempdir().context("creating tempdir")?;
    let attestation_path = tmp.path().join(&attestation_filename);
    let bundle_path = tmp.path().join(&bundle_filename);

    // Pretty-printed canonical form with a trailing newline. The signature
    // is over these exact bytes; the verifier downloads these bytes and
    // re-hashes them, so any drift breaks verification.
    let serialized = serde_json::to_string_pretty(&attestation)? + "\n";
    fs::write(&attestation_path, serialized.as_bytes())
        .with_context(|| format!("writing {}", attestation_path.display()))?;

    println!();
    println!("=== Signing with cosign + uploading to Rekor ===");
    cosign_sign_blob(&args.cosign_key, &attestation_path, &bundle_path)?;
    let log_index = read_rekor_log_index(&bundle_path)?;
    println!("  ✓ rekor log_index = {log_index}");
    println!("      https://search.sigstore.dev/?logIndex={log_index}");

    println!();
    println!("=== Uploading attestation + bundle to release ===");
    let attestation_str = attestation_path.to_str().unwrap();
    let bundle_str = bundle_path.to_str().unwrap();
    gh_upload(&args.repo, &args.tag, &[attestation_str, bundle_str])?;

    println!();
    println!("=== Generating release.json ===");
    let assets = gh_list_assets(&args.repo, &args.tag)?;
    let manifest_url = asset_url(&assets, "artifact-manifest.json")?;
    let manifest_sigstore_url = asset_url(&assets, "artifact-manifest.json.sigstore")?;
    let att_url = asset_url(&assets, &attestation_filename)?;
    let att_bundle_url = asset_url(&assets, &bundle_filename)?;

    // `release.json` is a pure URL index. Policy fields (threshold, allowed
    // identities, allowed schema versions) live in the verifier's embedded
    // trust root so a forged index can't downgrade them — see
    // `docs/trust-root.md`.
    let release_json = serde_json::json!({
        "schema_version": 1,
        "version": release_version,
        "git_commit": git_commit,
        "git_tag": args.tag,
        "released_at": attested_at,
        "previous_release": {
            "version": previous_release_version(&args.workspace_root, &args.tag)?,
            "git_commit": previous_release_git_commit,
        },
        "artifact_manifest": {
            "url": manifest_url,
            "sigstore_bundle_url": manifest_sigstore_url,
        },
        "human_attestations": [{
            "attestant_id": args.attestant_id,
            "url": att_url,
            "bundle_url": att_bundle_url,
        }],
    });
    let release_json_path = tmp.path().join("release.json");
    let release_json_str = serde_json::to_string_pretty(&release_json)? + "\n";
    fs::write(&release_json_path, release_json_str.as_bytes())?;

    gh_upload(
        &args.repo,
        &args.tag,
        &[release_json_path.to_str().unwrap()],
    )?;

    println!();
    println!("=== Marking release as latest ===");
    let status = Command::new("gh")
        .args([
            "release", "edit", &args.tag, "--repo", &args.repo, "--latest",
        ])
        .status()
        .context("running `gh release edit --latest`")?;
    if !status.success() {
        bail!("`gh release edit --latest` failed");
    }

    println!();
    println!("Release {} attested and published as latest.", args.tag);
    Ok(())
}

fn require_tool(name: &str) -> Result<()> {
    let status = Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // Accept any tool whose `--version` invocation reached the binary;
    // we just need to confirm it's resolvable on PATH, not that it
    // implements a specific exit-code convention.
    if status.is_ok() {
        return Ok(());
    }
    bail!("required tool `{name}` not found on PATH");
}

fn validate_attestant_id(id: &str) -> Result<()> {
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        bail!("--attestant-id must be lowercase alphanumeric with dashes (e.g. `mike-prince`)");
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        use std::fmt::Write;
        write!(out, "{b:02x}").unwrap();
    }
    out
}

/// Extract the public key for `cosign_key` via `cosign public-key
/// --key <ref>` and return the PKIX SubjectPublicKeyInfo DER bytes.
/// Callers compute `sha256(spki_der)` to match
/// `TRUSTED_ATTESTANT_FINGERPRINTS` and pass the same bytes through
/// [`eidola_app_core::updater::human_attestation::classify_attestant_spki_algorithm`]
/// to verify the algorithm is on the updater's allowlist before signing.
fn fetch_cosign_pubkey_spki_der(cosign_key: &str) -> Result<Vec<u8>> {
    let out = Command::new("cosign")
        .args(["public-key", "--key", cosign_key])
        .output()
        .context("running `cosign public-key --key <ref>`")?;
    if !out.status.success() {
        bail!(
            "`cosign public-key --key {}` failed: {}",
            cosign_key,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let pem = String::from_utf8(out.stdout).context("cosign public-key output is not UTF-8")?;
    pem_public_key_to_spki_der(&pem)
}

/// Decode a single `-----BEGIN PUBLIC KEY-----` PEM block to its inner
/// PKIX SubjectPublicKeyInfo DER bytes. Mirror of the verifier-side
/// helper in `eidola_app_core::updater::human_attestation`.
fn pem_public_key_to_spki_der(pem: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    let trimmed = pem.trim();
    let body = trimmed
        .strip_prefix("-----BEGIN PUBLIC KEY-----")
        .and_then(|s| s.strip_suffix("-----END PUBLIC KEY-----"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cosign public-key output missing `-----BEGIN/END PUBLIC KEY-----` markers"
            )
        })?
        .trim();
    let stripped: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(stripped.as_bytes())
        .context("base64-decoding cosign public-key PEM body")
}

/// True if `cosign_key` looks like a local file path rather than a
/// PKCS#11/KMS URI. cosign's URI schemes use `:` as a scheme separator
/// (`pkcs11:`, `awskms:`, `gcpkms:`, `azurekms:`, `hashivault:`); a
/// local path doesn't.
fn is_local_key_path(cosign_key: &str) -> bool {
    !matches!(
        cosign_key.split(':').next().unwrap_or(""),
        "pkcs11" | "awskms" | "gcpkms" | "azurekms" | "hashivault"
    )
}

/// Run `cosign sign-blob` against `attestation_path`, writing the
/// Sigstore Bundle v0.3 to `bundle_path`. cosign handles SHA-256 of the
/// blob, signature, Rekor upload, and bundle serialization in one
/// shot. `--yes` skips the confirmation prompt; the engineer has
/// already affirmed every claim before we get here.
fn cosign_sign_blob(cosign_key: &str, attestation_path: &Path, bundle_path: &Path) -> Result<()> {
    let status = Command::new("cosign")
        .args([
            "sign-blob",
            "--yes",
            "--key",
            cosign_key,
            "--bundle",
            bundle_path.to_str().unwrap(),
            attestation_path.to_str().unwrap(),
        ])
        .status()
        .context("running `cosign sign-blob`")?;
    if !status.success() {
        bail!(
            "`cosign sign-blob` failed; ensure the cosign key reference `{}` is reachable and \
             COSIGN_PASSWORD / device PIN is correct",
            cosign_key
        );
    }
    Ok(())
}

/// Extract the Rekor `logIndex` from a Sigstore Bundle v0.3 file. Used
/// purely for the user-facing "log_index=N" message + Sigstore search
/// URL; the verifier re-derives this independently.
fn read_rekor_log_index(bundle_path: &Path) -> Result<u64> {
    let bytes = fs::read(bundle_path)
        .with_context(|| format!("reading bundle {}", bundle_path.display()))?;
    let bundle: serde_json::Value =
        serde_json::from_slice(&bytes).context("parsing cosign-emitted bundle as JSON")?;
    let log_index_str = bundle
        .get("verificationMaterial")
        .and_then(|v| v.get("tlogEntries"))
        .and_then(|v| v.get(0))
        .and_then(|v| v.get("logIndex"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "cosign bundle missing `verificationMaterial.tlogEntries[0].logIndex` string"
            )
        })?;
    log_index_str
        .parse::<u64>()
        .with_context(|| format!("parsing bundle logIndex `{log_index_str}` as u64"))
}

fn git_rev_parse(workspace_root: &Path, refname: &str) -> Result<String> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["rev-parse", &format!("{refname}^{{commit}}")])
        .output()
        .context("running `git rev-parse <ref>^{commit}`")?;
    if !out.status.success() {
        bail!(
            "`git rev-parse {refname}^{{commit}}` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

fn previous_release_version(workspace_root: &Path, tag: &str) -> Result<String> {
    let prev = previous_release_tag(workspace_root, tag)?;
    Ok(prev
        .strip_prefix('v')
        .ok_or_else(|| anyhow::anyhow!("prior tag `{prev}` does not start with `v`"))?
        .to_string())
}

fn previous_release_tag(workspace_root: &Path, tag: &str) -> Result<String> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["describe", "--tags", "--abbrev=0", &format!("{tag}^")])
        .output()
        .context("running `git describe --tags --abbrev=0 <tag>^`")?;
    if !out.status.success() {
        bail!(
            "no previous tag reachable from `{tag}` — `previous_release` in attestation \
             cannot be populated. (If this is the first release ever, manually edit the \
             attestation skeleton or change the templates.)\n{}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

fn print_quoted(s: &str) {
    for line in s.lines() {
        println!("    {line}");
    }
}

fn affirm_or_abort(prompt: &str) -> Result<()> {
    print!("{prompt}: ");
    io::stdout().flush()?;
    let mut input = String::new();
    io::stdin()
        .lock()
        .read_line(&mut input)
        .context("reading stdin")?;
    if input.trim() == "yes" {
        Ok(())
    } else {
        bail!(
            "attestation aborted (got `{}`, expected `yes`)",
            input.trim()
        );
    }
}

fn gh_upload(repo: &str, tag: &str, files: &[&str]) -> Result<()> {
    let mut cmd = Command::new("gh");
    cmd.args(["release", "upload", tag, "--repo", repo, "--clobber"]);
    for f in files {
        cmd.arg(f);
    }
    let status = cmd.status().context("running `gh release upload`")?;
    if !status.success() {
        bail!("`gh release upload` failed");
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct GhAsset {
    name: String,
    url: String,
}

#[derive(serde::Deserialize)]
struct GhReleaseAssetsView {
    assets: Vec<GhAsset>,
}

fn gh_list_assets(repo: &str, tag: &str) -> Result<Vec<GhAsset>> {
    let out = Command::new("gh")
        .args(["release", "view", tag, "--repo", repo, "--json", "assets"])
        .output()
        .context("running `gh release view`")?;
    if !out.status.success() {
        bail!(
            "`gh release view {tag}` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let parsed: GhReleaseAssetsView =
        serde_json::from_slice(&out.stdout).context("parsing `gh release view` JSON")?;
    Ok(parsed.assets)
}

fn asset_url(assets: &[GhAsset], name: &str) -> Result<String> {
    assets
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.url.clone())
        .ok_or_else(|| anyhow::anyhow!("expected asset `{name}` not found on release"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_is_lowercase_64() {
        let h = sha256_hex(b"hello");
        assert_eq!(h.len(), 64);
        assert!(
            h.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        );
    }

    #[test]
    fn validate_attestant_id_accepts_kebab_lowercase() {
        validate_attestant_id("mike-prince").unwrap();
        validate_attestant_id("a1-b2-c3").unwrap();
    }

    #[test]
    fn validate_attestant_id_rejects_uppercase_or_underscores() {
        assert!(validate_attestant_id("Mike").is_err());
        assert!(validate_attestant_id("mike_prince").is_err());
        assert!(validate_attestant_id("").is_err());
    }

    #[test]
    fn is_local_key_path_recognizes_kms_uris() {
        // KMS / hardware URIs — cosign reads these as its own key refs.
        assert!(!is_local_key_path("pkcs11:slot-id=0;object=eidola"));
        assert!(!is_local_key_path("awskms:///alias/eidola"));
        assert!(!is_local_key_path(
            "gcpkms://projects/x/locations/y/keyRings/z/cryptoKeys/q"
        ));
        assert!(!is_local_key_path(
            "azurekms://vault.vault.azure.net/keys/eidola"
        ));
        assert!(!is_local_key_path("hashivault://eidola"));
        // Local paths.
        assert!(is_local_key_path("cosign.key"));
        assert!(is_local_key_path("/home/mike/cosign.key"));
        assert!(is_local_key_path("./cosign.key"));
    }

    #[test]
    fn pem_public_key_to_spki_der_extracts_p256_spki() {
        // PEM SPKI from a real `cosign generate-key-pair` run. P-256
        // SPKI DER is 91 bytes — the same value the verifier's parallel
        // helper produces.
        let pem = "-----BEGIN PUBLIC KEY-----\n\
                   MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEJsEXjtQe9u/kRQ006UUEXIt4aY7u\n\
                   JI4fqwrk1qBM9GyGPqZYJrflz/dWImo3wdF17ZG3kmfSe/rCiQKL3x/unQ==\n\
                   -----END PUBLIC KEY-----\n";
        let der = pem_public_key_to_spki_der(pem).unwrap();
        assert_eq!(der.len(), 91);
    }

    #[test]
    fn pem_public_key_to_spki_der_rejects_certificate_block() {
        let pem = "-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n";
        assert!(pem_public_key_to_spki_der(pem).is_err());
    }
}
