//! The grammar of an engine deployment's measured configuration, shared by
//! everything that reads one.
//!
//! An Eidola-hosted engine deployment is a `tinfoil-config.yml` whose
//! `eidola-server-engine` container boots from its environment. Three parties
//! read that config and must agree on what it means, so each rule here is the
//! one function they all call:
//!
//! - the node (`eidola-server-engine`), which boots from the values or
//!   refuses to;
//! - the gateway's build (`engine_trust/manifest.rs`), which refuses a pin
//!   for a config the node would refuse, or that it reads differently;
//! - `measure-enclave`, which turns the config into that pin.
//!
//! The Argon2id rule needs the `argon2` crate and sits behind this crate's
//! `argon2` feature: only the node and the gateway enable it, and both
//! already depend on `argon2` themselves (see the crate docs' dependency
//! rule).

/// Largest prefix-cache retention bound, in seconds: the engine core keeps
/// retention in milliseconds as a `u64`.
pub const MAX_CACHE_SECONDS: u64 = u64::MAX / 1000;

/// A prefix-cache retention bound (`EIDOLA_ENGINE_CACHE_IDLE_TTL_SECS`,
/// `EIDOLA_ENGINE_CACHE_MAX_AGE_SECS`): a positive whole number of seconds,
/// at most [`MAX_CACHE_SECONDS`].
pub fn parse_cache_seconds(text: &str) -> Option<u64> {
    text.parse::<u64>()
        .ok()
        .filter(|n| *n > 0 && *n <= MAX_CACHE_SECONDS)
}

/// The prefix-cache switch (`EIDOLA_ENGINE_PREFIX_CACHE`): exactly `true` or
/// `false`.
pub fn parse_prefix_cache(text: &str) -> Option<bool> {
    match text {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// The platform release a config's `cvm-version` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CvmVersion<'a> {
    /// The bare version (`0.15.0`; no leading `v`).
    pub version: &'a str,
    /// The release manifest's SHA-256, lowercase hex, when the config pins it
    /// inline (`0.15.0@sha256:<hex>`).
    pub manifest_sha256: Option<&'a str>,
}

/// Why a `cvm-version` value is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CvmVersionError {
    /// The version is empty or starts with `v`.
    NotBare,
    /// Something follows `@` other than `sha256:` and 64 lowercase hex
    /// digits.
    MalformedPin,
}

impl core::fmt::Display for CvmVersionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotBare => "cvm-version must be a bare version such as 0.15.0",
            Self::MalformedPin => {
                "malformed cvm-version pin: expected VERSION@sha256:<64 lowercase hex>"
            }
        })
    }
}

/// Parse a config's `cvm-version`: a bare version, optionally followed by
/// exactly one `@sha256:<64 lowercase hex>` release-manifest pin.
pub fn parse_cvm_version(raw: &str) -> Result<CvmVersion<'_>, CvmVersionError> {
    let (version, manifest_sha256) = match raw.split_once('@') {
        None => (raw, None),
        Some((version, pin)) => {
            let hex = pin
                .strip_prefix("sha256:")
                .filter(|h| {
                    h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
                })
                .ok_or(CvmVersionError::MalformedPin)?;
            (version, Some(hex))
        }
    };
    if version.is_empty() || version.starts_with('v') {
        return Err(CvmVersionError::NotBare);
    }
    Ok(CvmVersion {
        version,
        manifest_sha256,
    })
}

/// Why a gateway-token hash is refused.
#[cfg(feature = "argon2")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenHashError {
    /// Not a PHC string `argon2` can read.
    NotAPhcString,
    /// A PHC string for an algorithm other than Argon2id.
    NotArgon2id,
}

#[cfg(feature = "argon2")]
impl core::fmt::Display for TokenHashError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::NotAPhcString => "not a valid Argon2 hash string",
            Self::NotArgon2id => "must be an Argon2id hash",
        })
    }
}

/// Parse the gateway token's measured hash (`GATEWAY_TOKEN_HASH`): a PHC
/// string for Argon2id. The node verifies its token against the result.
#[cfg(feature = "argon2")]
pub fn parse_gateway_token_hash(hash: &str) -> Result<argon2::PasswordHash, TokenHashError> {
    let parsed = argon2::PasswordHash::new(hash).map_err(|_| TokenHashError::NotAPhcString)?;
    if parsed.algorithm.as_str() != "argon2id" {
        return Err(TokenHashError::NotArgon2id);
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_seconds_are_positive_and_fit_in_milliseconds() {
        assert_eq!(parse_cache_seconds("900"), Some(900));
        assert_eq!(
            parse_cache_seconds(&MAX_CACHE_SECONDS.to_string()),
            Some(MAX_CACHE_SECONDS)
        );
        for bad in [
            "0",
            "-1",
            "",
            " 900",
            "9.5",
            &(MAX_CACHE_SECONDS + 1).to_string(),
        ] {
            assert_eq!(parse_cache_seconds(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn the_prefix_cache_switch_is_exact() {
        assert_eq!(parse_prefix_cache("true"), Some(true));
        assert_eq!(parse_prefix_cache("false"), Some(false));
        for bad in ["True", "1", "yes", ""] {
            assert_eq!(parse_prefix_cache(bad), None);
        }
    }

    #[test]
    fn cvm_version_is_bare_with_at_most_one_inline_pin() {
        let pin = "ab".repeat(32);
        assert_eq!(
            parse_cvm_version("0.15.0"),
            Ok(CvmVersion {
                version: "0.15.0",
                manifest_sha256: None
            })
        );
        assert_eq!(
            parse_cvm_version(&format!("0.15.0@sha256:{pin}"))
                .unwrap()
                .manifest_sha256,
            Some(pin.as_str())
        );
        for (bad, error) in [
            ("".to_string(), CvmVersionError::NotBare),
            ("v0.15.0".to_string(), CvmVersionError::NotBare),
            (format!("@sha256:{pin}"), CvmVersionError::NotBare),
            (
                "0.15.0@sha256:AB".to_string(),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{}", pin.to_uppercase()),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha512:{pin}"),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{pin}@sha256:{pin}"),
                CvmVersionError::MalformedPin,
            ),
            (
                format!("0.15.0@sha256:{pin}x"),
                CvmVersionError::MalformedPin,
            ),
        ] {
            assert_eq!(parse_cvm_version(&bad), Err(error), "{bad:?}");
        }
    }

    #[cfg(feature = "argon2")]
    #[test]
    fn the_token_hash_is_a_whole_argon2id_phc_string() {
        let good = "$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0$aGFzaGhhc2hoYXNoaGFzaA";
        assert!(parse_gateway_token_hash(good).is_ok());
        assert_eq!(
            parse_gateway_token_hash("$argon2id$garbage").unwrap_err(),
            TokenHashError::NotAPhcString
        );
        assert_eq!(
            parse_gateway_token_hash(&good.replace("argon2id", "argon2i")).unwrap_err(),
            TokenHashError::NotArgon2id
        );
        assert_eq!(
            parse_gateway_token_hash("").unwrap_err(),
            TokenHashError::NotAPhcString
        );
    }
}
