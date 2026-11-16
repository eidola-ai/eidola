//! Which release a tag follows, and what that release's clients pin.
//!
//! A release is checked by the clients of the release before it, so that
//! release decides what binds this one: the templates an attestation is
//! rendered from, the commit `previous_release` names (which the client's
//! continuity check compares with the commit it runs), and the CI identity
//! the manifest signature has to carry.
//!
//! The previous release of `vX` is the highest release version below `vX`
//! among the tags in `vX`'s history, compared by semantic-version
//! precedence. History rules out a release cut on another line; the version
//! rules out every tag that is not a release (`git describe` would happily
//! answer with any tag) and picks the right one of several tags on one
//! commit. `scripts/check-signing-identity.sh` applies the same rule in CI.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

/// The trust root as committed in the tree.
pub const TRUST_CONSTANTS_PATH: &str = "releases/trust/trust-constants.json";

/// The release `tag` follows, or `None` when no release precedes it.
pub fn previous_release_tag(workspace_root: &Path, tag: &str) -> Result<Option<String>> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["tag", "--merged", tag, "--list", "v*"])
        .output()
        .context("running `git tag --merged <tag>`")?;
    if !out.status.success() {
        bail!(
            "`git tag --merged {tag}` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    let listed = String::from_utf8(out.stdout).context("git tag output is not UTF-8")?;
    pick_previous(listed.lines(), tag)
}

/// The highest release tag in `tags` whose version is below `tag`'s.
fn pick_previous<'a>(tags: impl IntoIterator<Item = &'a str>, tag: &str) -> Result<Option<String>> {
    let Some(current) = release_version(tag) else {
        bail!("`{tag}` is not a release tag (vMAJOR.MINOR.PATCH[-PRE])");
    };
    Ok(tags
        .into_iter()
        .map(str::trim)
        .filter_map(|t| release_version(t).map(|v| (v, t)))
        .filter(|(v, _)| *v < current)
        .max_by(|(a, _), (b, _)| a.cmp(b))
        .map(|(_, t)| t.to_string()))
}

/// The version a release tag names. Build metadata has no precedence, so a
/// tag carrying it is not a release tag.
fn release_version(tag: &str) -> Option<semver::Version> {
    let version = semver::Version::parse(tag.strip_prefix('v')?).ok()?;
    version.build.is_empty().then_some(version)
}

/// The exact bytes of `path` as committed at `refname`.
pub fn git_show(workspace_root: &Path, refname: &str, path: &str) -> Result<Vec<u8>> {
    let out = Command::new("git")
        .current_dir(workspace_root)
        .args(["show", &format!("{refname}:{path}")])
        .output()
        .context("running `git show <ref>:<path>`")?;
    if !out.status.success() {
        bail!(
            "`git show {refname}:{path}` failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(out.stdout)
}

/// The CI identity pattern and issuer a release's manifest signature must
/// satisfy: those its clients pin, from the trust root committed at
/// `previous`.
pub fn ci_pins_at(workspace_root: &Path, previous: &str) -> Result<(String, String)> {
    let bytes = git_show(workspace_root, previous, TRUST_CONSTANTS_PATH)?;
    ci_pins_from(&bytes).with_context(|| format!("reading {TRUST_CONSTANTS_PATH} at `{previous}`"))
}

fn ci_pins_from(bytes: &[u8]) -> Result<(String, String)> {
    // Read as JSON rather than through the current schema's struct: the
    // file is as an older release committed it, and only these two fields
    // are needed from it.
    let value: serde_json::Value = serde_json::from_slice(bytes).context("parsing as JSON")?;
    let field = |name: &str| -> Result<String> {
        match value.get(name).and_then(serde_json::Value::as_str) {
            Some(s) if !s.is_empty() => Ok(s.to_string()),
            _ => bail!("no `{name}`"),
        }
    };
    Ok((
        field("expected_ci_identity_pattern")?,
        field("expected_ci_issuer")?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn previous(tags: &[&str], tag: &str) -> Option<String> {
        pick_previous(tags.iter().copied(), tag).unwrap()
    }

    #[test]
    fn the_previous_release_is_the_highest_version_below() {
        assert_eq!(
            previous(&["v0.1.10", "v0.1.9", "v0.1.2"], "v0.1.11").as_deref(),
            Some("v0.1.10")
        );
        assert_eq!(
            previous(&["v0.1.10", "v0.1.9"], "v0.1.10").as_deref(),
            Some("v0.1.9")
        );
    }

    #[test]
    fn tags_that_are_not_releases_are_ignored() {
        // `git describe` answers with whichever tag is nearest, and this
        // repository's history carries tags that are not releases.
        assert_eq!(
            previous(
                &["immut-test-1", "v0.0.6", "latest", "v0.0.5+build"],
                "v0.0.9"
            )
            .as_deref(),
            Some("v0.0.6")
        );
    }

    #[test]
    fn of_several_tags_on_one_commit_the_highest_is_the_release() {
        assert_eq!(
            previous(&["v0.0.7", "v0.0.8"], "v0.0.9").as_deref(),
            Some("v0.0.8")
        );
    }

    #[test]
    fn pre_releases_compare_by_precedence() {
        // Numeric identifiers compare as numbers: rc.10 follows rc.9.
        assert_eq!(
            previous(&["v1.0.0-rc.9", "v1.0.0-rc.10", "v0.9.0"], "v1.0.0").as_deref(),
            Some("v1.0.0-rc.10")
        );
        // And a pre-release sorts below its release.
        assert_eq!(
            previous(&["v1.0.0-rc.9", "v0.9.0", "v1.0.0"], "v1.0.0-rc.10").as_deref(),
            Some("v1.0.0-rc.9")
        );
    }

    #[test]
    fn the_first_release_has_none() {
        assert_eq!(previous(&["immut-test-1"], "v0.0.1"), None);
    }

    #[test]
    fn a_tag_that_is_not_a_release_is_refused() {
        assert!(pick_previous(["v0.0.1"], "immut-test-1").is_err());
    }

    #[test]
    fn ci_pins_are_read_from_an_older_trust_root() {
        let committed = br#"{
            "schema_version": 1,
            "expected_ci_identity_pattern": "https://github.com/o/r/.github/workflows/a.yml@refs/tags/v*",
            "expected_ci_issuer": "https://token.actions.githubusercontent.com",
            "update_discovery_url": "https://api.github.com/repos/o/r/releases/latest"
        }"#;
        assert_eq!(
            ci_pins_from(committed).unwrap(),
            (
                "https://github.com/o/r/.github/workflows/a.yml@refs/tags/v*".to_string(),
                "https://token.actions.githubusercontent.com".to_string()
            )
        );
        assert!(ci_pins_from(br#"{"expected_ci_issuer": "x"}"#).is_err());
    }
}
