//! The platform provider's platform-endorsements release tags,
//! `platform-vX.Y.Z`: the one parser of them.
//!
//! An engine deployment's sidecar names a release by its tag, which becomes a
//! path component of the vendored copy (`engine_deployment::deployment`), and
//! `tinfoil-verifier` accepts a release's Sigstore attestation only from the
//! provider's release workflow at exactly that tag. Both read the tag here,
//! as does every tool that orders vendored releases, so what a sidecar may
//! name, what may be verified, and which release is the newest cannot
//! disagree.

/// A release tag's `(major, minor, patch)`, which orders releases; `None`
/// unless `tag` is `platform-vX.Y.Z` with each number decimal, without a
/// leading zero and at most nine digits (the pattern the provider's release
/// workflow enforces, `PLATFORM_TAG_PATTERN`, bounded so each fits a `u32`).
/// A tag that parses is also a safe path component.
pub fn version(tag: &str) -> Option<(u32, u32, u32)> {
    let mut parts = tag.strip_prefix("platform-v")?.split('.');
    let mut number = || {
        let p = parts.next()?;
        let canonical = !p.is_empty()
            && p.len() <= 9
            && p.bytes().all(|b| b.is_ascii_digit())
            && (p.len() == 1 || !p.starts_with('0'));
        canonical.then(|| p.parse().ok()).flatten()
    };
    let version = (number()?, number()?, number()?);
    parts.next().is_none().then_some(version)
}

/// Whether `tag` is a platform release tag ([`version`] parses it).
pub fn is_tag(tag: &str) -> bool {
    version(tag).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tags_follow_the_release_workflows_pattern() {
        for (good, parsed) in [
            ("platform-v0.1.11", (0, 1, 11)),
            ("platform-v1.0.0", (1, 0, 0)),
            ("platform-v10.20.30", (10, 20, 30)),
            ("platform-v999999999.0.0", (999_999_999, 0, 0)),
        ] {
            assert_eq!(version(good), Some(parsed), "{good}");
            assert!(is_tag(good), "{good}");
        }
        for bad in [
            "platform-v0.1",
            "platform-v0.1.11.1",
            "platform-v0.01.1",
            "platform-v0.1.",
            "platform-v.1.1",
            "platform-v0.1.+1",
            "platform-v0.1.1000000000",
            "v0.1.11",
            "platform-0.1.11",
            "platform-v0.1.11-rc1",
            "platform-v0.1.11/..",
            "",
        ] {
            assert_eq!(version(bad), None, "{bad}");
            assert!(!is_tag(bad), "{bad}");
        }
    }

    #[test]
    fn versions_order_numerically() {
        assert!(version("platform-v0.1.11") > version("platform-v0.1.9"));
        assert!(version("platform-v0.2.0") > version("platform-v0.1.11"));
        assert!(version("platform-v1.0.0") > version("platform-v0.99.99"));
    }
}
