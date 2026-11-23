//! The mobile app bundle as `artifact-manifest.json` records it.
//!
//! ```json
//! "eidola-mobile-bundle": {
//!   "type": "bundle",
//!   "platform": "webview/wasm32",
//!   "sha256": "sha256:<the container's hash>",
//!   "digest": "sha256:<the bundle digest>",
//!   "files": [
//!     {"path": "boot.js", "size": 1234, "sha256": "sha256:<the file's hash>"},
//!     …
//!   ]
//! }
//! ```
//!
//! Three hashes, each answering a different question:
//!
//! - `sha256` is the published **container**, the one release asset the
//!   bundle is downloaded as (`release.json`'s index names one URL per
//!   artifact key). It is what `scripts/release-assets.sh` re-hashes before
//!   upload, and what the installer checks before it parses a byte.
//! - `files` lists **every file the shell will store**: its bundle path, its
//!   length and the hash of its bytes. The installer checks each file it
//!   unpacked against this list, so what reaches the shell is held to the
//!   signed manifest and not to the container's reading of itself.
//! - `digest` is the **bundle digest** (`mobile/shell-contract.md` → Bundle
//!   digest): the fingerprint the shell computes from the bytes it serves and
//!   shows in its own sheet. It is a function of `files` alone, and a row
//!   whose digest is not that function is malformed, so the value a person
//!   reads off their phone is the value they can find in the manifest.
//!
//! Every value is a function of source: the files are the reproducible
//! bundle build's output (`mobile/build-bundle.sh`), and the container is a
//! deterministic archive of them. Nothing here depends on a key.
//!
//! [`BundleRow::from_value`] is the one reader of this shape. The update
//! notification's claims check (`crate::updates::describe_artifact`) and the
//! installer both go through it, so a row the notification calls well
//! formed is a row the installer can act on.

use sha2::{Digest, Sha256};

use crate::updater::human_attestation::hex_encode;

/// The artifact key the bundle is recorded under.
pub const BUNDLE_KEY: &str = "eidola-mobile-bundle";
/// The artifact type a bundle row declares.
pub const BUNDLE_TYPE: &str = "bundle";
/// The platform a bundle row declares: it runs in a system webview, as
/// wasm32, on every mobile platform alike.
pub const BUNDLE_PLATFORM: &str = "webview/wasm32";

/// The most files a bundle may hold. Today's has five.
pub const MAX_BUNDLE_FILES: usize = 1024;

/// The most bytes a bundle may hold, and the most its container may be. The
/// container is held in memory while it is checked, on a phone, so this is
/// a bound on what a hostile server can make the app allocate as much as on
/// the bundle (today's is about ten megabytes, nearly all of it the wasm).
pub const MAX_BUNDLE_BYTES: u64 = 64 * 1024 * 1024;

/// The largest integer the shell contract carries (2^53 - 1): sizes beyond it
/// cannot cross the bridge.
const MAX_CONTRACT_INTEGER: u64 = (1 << 53) - 1;

/// The page every bundle boots from: the shell serves `/` as it, and treats a
/// bundle without it as one that failed its trial.
pub const BUNDLE_INDEX: &str = "index.html";

/// One file of the bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleFile {
    /// A bundle path (`shell-contract.md` → Paths).
    pub path: String,
    pub size: u64,
    /// Lowercase hex, without the `sha256:` prefix.
    pub sha256: String,
}

/// A well-formed bundle row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleRow {
    /// The container's hash: lowercase hex, without the prefix.
    pub container_sha256: String,
    /// The bundle digest: lowercase hex, without the prefix.
    pub digest: String,
    /// Every file, sorted by path, each path once.
    pub files: Vec<BundleFile>,
}

impl BundleRow {
    /// The total length of the bundle's files.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.size).sum()
    }

    /// Read a bundle row, holding it to the exact shape above. The message
    /// says what is wrong, for a claim or a refusal.
    pub fn from_value(entry: &serde_json::Value) -> Result<Self, String> {
        let obj = entry.as_object().ok_or("not an object")?;
        exact_keys(obj, &["digest", "files", "platform", "sha256", "type"], "")?;
        if obj.get("type").and_then(|v| v.as_str()) != Some(BUNDLE_TYPE) {
            return Err(format!("type is not `{BUNDLE_TYPE}`"));
        }
        if obj.get("platform").and_then(|v| v.as_str()) != Some(BUNDLE_PLATFORM) {
            return Err(format!("platform is not `{BUNDLE_PLATFORM}`"));
        }
        let container_sha256 = prefixed_sha256(obj.get("sha256"), "sha256")?;
        let digest = prefixed_sha256(obj.get("digest"), "digest")?;

        let listed = obj
            .get("files")
            .and_then(|v| v.as_array())
            .ok_or("files is not an array")?;
        if listed.is_empty() {
            return Err("files is empty".into());
        }
        if listed.len() > MAX_BUNDLE_FILES {
            return Err(format!(
                "files lists {} files, more than the {MAX_BUNDLE_FILES} a bundle may hold",
                listed.len()
            ));
        }
        let mut files = Vec::with_capacity(listed.len());
        let mut total: u64 = 0;
        for (i, file) in listed.iter().enumerate() {
            let at = format!("files[{i}]");
            let file = file
                .as_object()
                .ok_or_else(|| format!("{at} is not an object"))?;
            exact_keys(file, &["path", "sha256", "size"], &at)?;
            let path = file
                .get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("{at}.path is not a string"))?;
            if !is_bundle_path(path) {
                return Err(format!("{at}.path `{path}` is not a bundle path"));
            }
            let size = file
                .get("size")
                .and_then(|v| v.as_u64())
                .filter(|size| *size <= MAX_CONTRACT_INTEGER)
                .ok_or_else(|| format!("{at}.size is not a non-negative integer below 2^53"))?;
            let sha256 = prefixed_sha256(file.get("sha256"), &format!("{at}.sha256"))?;
            if let Some(previous) = files.last().map(|f: &BundleFile| f.path.as_str())
                && previous.as_bytes() >= path.as_bytes()
            {
                return Err(format!(
                    "{at}.path `{path}` is not after `{previous}`: files are sorted by path, \
                     each path once"
                ));
            }
            total = total.saturating_add(size);
            files.push(BundleFile {
                path: path.to_string(),
                size,
                sha256,
            });
        }
        if total > MAX_BUNDLE_BYTES {
            return Err(format!(
                "the files total {total} bytes, more than the {MAX_BUNDLE_BYTES} a bundle may hold"
            ));
        }
        if !files.iter().any(|f| f.path == BUNDLE_INDEX) {
            return Err(format!(
                "files has no `{BUNDLE_INDEX}`, which a bundle boots from"
            ));
        }
        let listed_digest =
            bundle_digest(files.iter().map(|f| (f.path.as_str(), f.sha256.as_str())));
        if listed_digest != digest {
            return Err(format!(
                "digest is sha256:{digest}, but the files listed digest to sha256:{listed_digest}"
            ));
        }
        Ok(Self {
            container_sha256,
            digest,
            files,
        })
    }
}

/// The bundle digest (`shell-contract.md` → Bundle digest) of files given as
/// `(path, lowercase hex sha256)`: the `sha256sum` line of each, in path
/// order, hashed together. The order the files are given in does not matter.
pub fn bundle_digest<'a>(files: impl IntoIterator<Item = (&'a str, &'a str)>) -> String {
    let mut lines: Vec<(&str, &str)> = files.into_iter().collect();
    lines.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut hasher = Sha256::new();
    for (path, sha256) in lines {
        hasher.update(sha256.as_bytes());
        hasher.update(b"  ");
        hasher.update(path.as_bytes());
        hasher.update(b"\n");
    }
    hex_encode(&hasher.finalize())
}

/// A bundle path (`shell-contract.md` → Paths): one or more segments joined
/// by `/`, each of `[A-Za-z0-9._-]`, none starting with `.`.
pub fn is_bundle_path(path: &str) -> bool {
    !path.is_empty()
        && path.split('/').all(|segment| {
            !segment.is_empty()
                && !segment.starts_with('.')
                && segment
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
        })
}

/// `sha256:` and 64 lowercase hex digits, as every hash in the manifest is
/// spelled; the hex without its prefix.
fn prefixed_sha256(value: Option<&serde_json::Value>, name: &str) -> Result<String, String> {
    value
        .and_then(|v| v.as_str())
        .and_then(|s| s.strip_prefix("sha256:"))
        .filter(|hex| {
            hex.len() == 64 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
        })
        .map(str::to_string)
        .ok_or_else(|| format!("{name} is not `sha256:` and 64 lowercase hex digits"))
}

/// The object holds exactly these keys: one added is a claim nobody
/// reviewed, one missing is a claim withdrawn.
fn exact_keys(
    obj: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    at: &str,
) -> Result<(), String> {
    let mut found: Vec<&str> = obj.keys().map(String::as_str).collect();
    found.sort_unstable();
    if found == allowed {
        return Ok(());
    }
    let place = if at.is_empty() {
        String::new()
    } else {
        format!("{at} ")
    };
    Err(format!(
        "{place}holds {}, where exactly {} are allowed",
        if found.is_empty() {
            "nothing".to_string()
        } else {
            found.join(", ")
        },
        allowed.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        hex_encode(&Sha256::digest(bytes))
    }

    pub(crate) fn row(files: &[(&str, &[u8])]) -> serde_json::Value {
        let listed: Vec<serde_json::Value> = files
            .iter()
            .map(|(path, bytes)| {
                serde_json::json!({
                    "path": path,
                    "size": bytes.len(),
                    "sha256": format!("sha256:{}", hex(bytes)),
                })
            })
            .collect();
        let digest = bundle_digest(
            files
                .iter()
                .map(|(path, bytes)| (*path, hex(bytes)))
                .collect::<Vec<_>>()
                .iter()
                .map(|(p, h)| (*p, h.as_str())),
        );
        serde_json::json!({
            "type": "bundle",
            "platform": "webview/wasm32",
            "sha256": format!("sha256:{}", "a".repeat(64)),
            "digest": format!("sha256:{digest}"),
            "files": listed,
        })
    }

    const FILES: &[(&str, &[u8])] = &[
        ("boot.js", b"boot"),
        ("index.html", b"<!doctype html>"),
        ("snippets/eidola-mobile-0/js/bridge.js", b"bridge"),
    ];

    #[test]
    fn a_well_formed_row_reads() {
        let row = BundleRow::from_value(&row(FILES)).unwrap();
        assert_eq!(row.files.len(), 3);
        assert_eq!(row.total_bytes(), 4 + 15 + 6);
        assert_eq!(row.container_sha256, "a".repeat(64));
    }

    #[test]
    fn the_digest_is_the_contract_s_sha256sum_digest() {
        // `printf 'a' > a; printf 'b' > d/b; find . -type f | sed 's|^\./||'
        // | LC_ALL=C sort | xargs sha256sum | sha256sum`, computed by hand
        // from the two lines below.
        let a = hex(b"a");
        let b = hex(b"b");
        let lines = format!("{a}  a\n{b}  d/b\n");
        assert_eq!(
            bundle_digest([("d/b", b.as_str()), ("a", a.as_str())]),
            hex(lines.as_bytes()),
            "paths sort by bytes and each line is `hash  path\\n`"
        );
    }

    #[test]
    fn a_digest_that_is_not_the_listing_s_is_refused() {
        let mut value = row(FILES);
        value["digest"] = format!("sha256:{}", "b".repeat(64)).into();
        let err = BundleRow::from_value(&value).unwrap_err();
        assert!(err.contains("digest to"), "{err}");
    }

    type Mutation = (&'static str, Box<dyn Fn(&mut serde_json::Value)>);

    #[test]
    fn every_shape_rule_holds() {
        let cases: Vec<Mutation> = vec![
            ("an extra field", Box::new(|v| v["signed"] = true.into())),
            (
                "a missing field",
                Box::new(|v| {
                    v.as_object_mut().unwrap().remove("digest");
                }),
            ),
            ("another type", Box::new(|v| v["type"] = "file".into())),
            (
                "another platform",
                Box::new(|v| v["platform"] = "ios/arm64".into()),
            ),
            (
                "an uppercase hash",
                Box::new(|v| v["sha256"] = format!("sha256:{}", "A".repeat(64)).into()),
            ),
            (
                "an unprefixed hash",
                Box::new(|v| v["sha256"] = "a".repeat(64).into()),
            ),
            (
                "an empty listing",
                Box::new(|v| v["files"] = serde_json::json!([])),
            ),
            (
                "a file field too many",
                Box::new(|v| v["files"][0]["mode"] = 420.into()),
            ),
            (
                "a path with a dot segment",
                Box::new(|v| v["files"][0]["path"] = "../boot.js".into()),
            ),
            (
                "a hidden file",
                Box::new(|v| v["files"][0]["path"] = ".boot.js".into()),
            ),
            (
                "an absolute path",
                Box::new(|v| v["files"][0]["path"] = "/boot.js".into()),
            ),
            (
                "an empty segment",
                Box::new(|v| v["files"][2]["path"] = "snippets//bridge.js".into()),
            ),
            (
                "a negative size",
                Box::new(|v| v["files"][0]["size"] = (-1).into()),
            ),
            (
                "a fractional size",
                Box::new(|v| v["files"][0]["size"] = 1.5.into()),
            ),
            (
                "a size past 2^53",
                Box::new(|v| v["files"][0]["size"] = (1u64 << 53).into()),
            ),
            (
                "unsorted files",
                Box::new(|v| {
                    let files = v["files"].as_array_mut().unwrap();
                    files.swap(0, 1);
                }),
            ),
            (
                "a repeated path",
                Box::new(|v| {
                    let first = v["files"][0].clone();
                    v["files"].as_array_mut().unwrap().insert(0, first);
                }),
            ),
            (
                "no index.html",
                Box::new(|v| v["files"][1]["path"] = "main.html".into()),
            ),
        ];
        for (what, mutate) in cases {
            let mut value = row(FILES);
            mutate(&mut value);
            assert!(
                BundleRow::from_value(&value).is_err(),
                "{what} must be refused: {value}"
            );
        }
    }

    #[test]
    fn the_determinism_gate_s_accepted_fixture_reads_here_too() {
        // `scripts/check-manifest-determinism.sh` computes the digest in
        // Python and accepts this file; the client must read the same row
        // the same way, or the gate and the client disagree on the shape.
        let bytes = include_bytes!(
            "../../../../../scripts/fixtures/manifest-determinism/good-schema-6.json"
        );
        let manifest: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        let row = BundleRow::from_value(&manifest["artifacts"][BUNDLE_KEY]).unwrap();
        assert_eq!(row.files.len(), 3);
        // And the one it refuses for its digest, this refuses too.
        let bytes = include_bytes!(
            "../../../../../scripts/fixtures/manifest-determinism/bad-bundle-digest.json"
        );
        let manifest: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        assert!(BundleRow::from_value(&manifest["artifacts"][BUNDLE_KEY]).is_err());
    }

    #[test]
    fn a_bundle_larger_than_the_bound_is_refused() {
        let mut value = row(FILES);
        value["files"][0]["size"] = (MAX_BUNDLE_BYTES + 1).into();
        let err = BundleRow::from_value(&value).unwrap_err();
        assert!(err.contains("more than"), "{err}");
    }

    #[test]
    fn bundle_paths_are_the_contract_s() {
        for ok in [
            "index.html",
            "a/b/c.js",
            "x-y_z.1",
            "snippets/eidola-mobile-0/js/bridge.js",
        ] {
            assert!(is_bundle_path(ok), "{ok}");
        }
        for bad in [
            "", "/", "a/", "/a", "a//b", ".a", "a/.b", "a b", "a\\b", "ä", "a/../b",
        ] {
            assert!(!is_bundle_path(bad), "{bad}");
        }
    }
}
