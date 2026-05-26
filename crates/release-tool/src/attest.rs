//! `release-tool attest <tag>` — interactively render every claim, prompt
//! the engineer to type `yes` to affirm each, sign with their hardware-
//! backed SSH key via `ssh-keygen -Y sign`, post the signature to Sigstore
//! Rekor as a `hashedrekord` entry, and upload everything to the GitHub
//! release.

use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use base64::Engine;
use chrono::Utc;
use sha2::{Digest, Sha256};

use crate::trust;

/// Namespace mixed into the SSH signature (see PROTOCOL.sshsig). Stable;
/// changing it invalidates every prior signature.
const SSH_SIG_NAMESPACE: &str = "eidola-attestation@v1";

/// Public Rekor instance — same one Sigstore-rs's TrustedRoot points at.
/// If we ever run our own Rekor mirror this becomes a trust-root field.
const REKOR_BASE_URL: &str = "https://rekor.sigstore.dev";

pub struct Args {
    pub workspace_root: PathBuf,
    pub repo: String,
    pub tag: String,
    pub ssh_pubkey: PathBuf,
    pub attestant_id: String,
    pub attestant_name: String,
    pub jurisdiction: String,
}

pub fn run(args: Args) -> Result<()> {
    require_tool("gh")?;
    require_tool("ssh-keygen")?;
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
    if !args.ssh_pubkey.is_file() {
        bail!(
            "--ssh-pubkey path `{}` does not exist or is not a file",
            args.ssh_pubkey.display()
        );
    }
    if std::env::var("SSH_AUTH_SOCK").is_err() {
        bail!(
            "SSH_AUTH_SOCK is not set. Configure your SSH agent (e.g. Secretive) so \
             `ssh-keygen -Y sign` can reach the private key, then retry."
        );
    }

    let release_version = args
        .tag
        .strip_prefix('v')
        .ok_or_else(|| anyhow::anyhow!("tag `{}` must start with `v`", args.tag))?
        .to_string();
    let git_commit = git_rev_parse(&args.workspace_root, &args.tag)?;
    let previous_release_git_commit = previous_release_commit(&args.workspace_root, &args.tag)?;

    let manifest_path = args.workspace_root.join("artifact-manifest.json");
    let artifact_manifest_sha256 = sha256_hex(&fs::read(&manifest_path)?);

    let privacy_doc_path = args.workspace_root.join("PRIVACY-GUARANTEES.md");
    let privacy_guarantees_doc_sha256 = match fs::read(&privacy_doc_path) {
        Ok(bytes) => sha256_hex(&bytes),
        Err(_) => bail!(
            "`PRIVACY-GUARANTEES.md` not found at repo root. \
             The `no_known_privacy_weakening` claim references its sha256; \
             create the document and commit it before attesting."
        ),
    };

    let key_fingerprint_sha256 = compute_ssh_fingerprint(&args.ssh_pubkey)?;

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
    println!("=== Signing with SSH agent ===");
    let signature_path = ssh_sign(&attestation_path, &args.ssh_pubkey, SSH_SIG_NAMESPACE)?;
    println!("  signature → {}", signature_path.display());

    let attestation_hash_hex = sha256_hex(&fs::read(&attestation_path)?);
    let signature_bytes = fs::read(&signature_path)?;
    let pubkey_bytes = fs::read(&args.ssh_pubkey)?;

    println!();
    println!("=== Uploading to Rekor ({REKOR_BASE_URL}) ===");
    let rekor_entry =
        rekor_upload_hashedrekord(&attestation_hash_hex, &signature_bytes, &pubkey_bytes)?;
    println!(
        "  ✓ log_index={} uuid={}",
        rekor_entry.log_index, rekor_entry.uuid
    );

    // Build the bundle file the verifier reads — self-contained: the rekor
    // entry body holds the SSH signature + pubkey + signed file hash, plus
    // the inclusion proof + SignedEntryTimestamp.
    let bundle = serde_json::json!({
        "schema_version": 1,
        "rekor_log_entry": rekor_entry.raw,
    });
    let bundle_serialized = serde_json::to_string_pretty(&bundle)? + "\n";
    fs::write(&bundle_path, bundle_serialized.as_bytes())?;

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
        "policy": { "min_human_attestations": 1 },
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
    // Some tools (e.g. ssh-keygen) exit non-zero on --version; accept either
    // exit-code success or "the binary at least dispatched."
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

/// Compute `sha256(<wire-format pubkey bytes>)` for an OpenSSH `.pub` file.
/// The wire bytes are the base64-decoded middle field of the `.pub` line
/// (`ssh-<type> <base64-wire-format> [comment]`).
fn compute_ssh_fingerprint(pubkey_path: &Path) -> Result<String> {
    let content = fs::read_to_string(pubkey_path)
        .with_context(|| format!("reading {}", pubkey_path.display()))?;
    let line = content
        .lines()
        .next()
        .ok_or_else(|| anyhow::anyhow!("ssh pubkey file `{}` is empty", pubkey_path.display()))?;
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        bail!(
            "ssh pubkey file `{}` is malformed (expected `ssh-<type> <base64> [comment]`)",
            pubkey_path.display()
        );
    }
    let wire = base64::engine::general_purpose::STANDARD
        .decode(parts[1].as_bytes())
        .context("base64-decoding ssh pubkey wire data")?;
    Ok(sha256_hex(&wire))
}

/// Sign `file_to_sign` via `ssh-keygen -Y sign`, returning the path of the
/// `.sig` file the tool writes alongside the input.
fn ssh_sign(file_to_sign: &Path, pubkey: &Path, namespace: &str) -> Result<PathBuf> {
    let status = Command::new("ssh-keygen")
        .args([
            "-Y",
            "sign",
            "-n",
            namespace,
            "-f",
            pubkey.to_str().unwrap(),
        ])
        .arg(file_to_sign)
        .status()
        .context("running `ssh-keygen -Y sign`")?;
    if !status.success() {
        bail!(
            "ssh-keygen -Y sign failed; ensure SSH_AUTH_SOCK reaches an agent that holds \
             the matching private key for `{}`",
            pubkey.display()
        );
    }
    let mut sig_path = file_to_sign.as_os_str().to_owned();
    sig_path.push(".sig");
    Ok(PathBuf::from(sig_path))
}

/// POST a `hashedrekord` entry to Sigstore Rekor and return the parsed log
/// entry. Rekor identifies SSH-format keys automatically from the pubkey
/// content; we don't need to declare the format explicitly.
fn rekor_upload_hashedrekord(
    artifact_hash_hex: &str,
    signature_bytes: &[u8],
    pubkey_bytes: &[u8],
) -> Result<RekorLogEntry> {
    let body = serde_json::json!({
        "apiVersion": "0.0.1",
        "kind": "hashedrekord",
        "spec": {
            "data": {
                "hash": {
                    "algorithm": "sha256",
                    "value": artifact_hash_hex,
                }
            },
            "signature": {
                "content": base64::engine::general_purpose::STANDARD.encode(signature_bytes),
                "publicKey": {
                    "content": base64::engine::general_purpose::STANDARD.encode(pubkey_bytes),
                }
            }
        }
    });

    let url = format!("{REKOR_BASE_URL}/api/v1/log/entries");
    let response = reqwest::blocking::Client::new()
        .post(&url)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .json(&body)
        .send()
        .context("posting hashedrekord to Rekor")?;

    let status = response.status();
    let response_text = response.text().context("reading Rekor response body")?;
    if !status.is_success() {
        bail!("Rekor POST returned {status}: {response_text}");
    }

    let parsed: serde_json::Value = serde_json::from_str(&response_text)
        .with_context(|| format!("parsing Rekor response: {response_text}"))?;

    // Rekor returns `{ "<uuid>": { body, integratedTime, logID, logIndex,
    // verification: { inclusionProof, signedEntryTimestamp } } }`.
    let obj = parsed
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Rekor response is not an object"))?;
    let (uuid, entry) = obj
        .iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Rekor response is empty"))?;

    let log_index = entry
        .get("logIndex")
        .and_then(|v| v.as_i64())
        .ok_or_else(|| anyhow::anyhow!("Rekor response missing logIndex"))?;

    Ok(RekorLogEntry {
        uuid: uuid.clone(),
        log_index,
        raw: parsed,
    })
}

struct RekorLogEntry {
    uuid: String,
    log_index: i64,
    /// Verbatim Rekor response — embedded as-is in the bundle file so the
    /// verifier sees exactly what the log returned.
    raw: serde_json::Value,
}

fn git_rev_parse(workspace_root: &Path, refname: &str) -> Result<String> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["rev-parse", refname])
        .output()
        .context("running `git rev-parse`")?;
    if !out.status.success() {
        bail!(
            "`git rev-parse {refname}` failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

fn previous_release_commit(workspace_root: &Path, tag: &str) -> Result<String> {
    let prev = previous_release_tag(workspace_root, tag)?;
    git_rev_parse(workspace_root, &prev)
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
    fn ssh_fingerprint_matches_known_value() {
        // Construct a `.pub` file with a known wire-format payload.
        // sha256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let encoded = base64::engine::general_purpose::STANDARD.encode(b"hello");
        std::fs::write(tmp.path(), format!("ssh-ed25519 {encoded} comment\n")).unwrap();
        let fp = compute_ssh_fingerprint(tmp.path()).unwrap();
        assert_eq!(
            fp,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn ssh_fingerprint_rejects_malformed_pubkey() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "only-one-field\n").unwrap();
        assert!(compute_ssh_fingerprint(tmp.path()).is_err());
    }
}
