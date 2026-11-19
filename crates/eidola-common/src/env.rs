//! Environment variables, read under a deployment's prefix.
//!
//! The platform hands each organisation-scoped secret to a container as a
//! variable of the secret's own name, so two processes that need different
//! values for one setting (the gateway's database role and a billing worker's)
//! cannot both read `DATABASE_URL`. `EIDOLA_ENV_PREFIX`, read unprefixed and
//! once at startup, names the prefix every other variable carries: with
//! `GATEWAY_` set, the gateway reads `GATEWAY_DATABASE_URL`, and the bare name
//! is never consulted. Unset or empty, the prefix is empty and every variable
//! is read by its own name.
//!
//! The gateway, the billing binary and the inference node read their
//! environment through this. The rule is also a contract with a measured
//! config: `engine_deployment` reads an engine deployment's env under the
//! prefix that env declares, by [`Env::name`], so the names a pin is checked
//! against are the names the node reads.

use std::fmt;

/// The variable that names the prefix. It is the one variable read bare.
pub const PREFIX_VAR: &str = "EIDOLA_ENV_PREFIX";

/// A process's environment prefix, validated.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Env {
    prefix: String,
}

/// A set `EIDOLA_ENV_PREFIX` that is not of the accepted form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidPrefix(String);

impl fmt::Display for InvalidPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for InvalidPrefix {}

impl Env {
    /// Read the prefix from the process environment.
    pub fn from_process() -> Result<Self, InvalidPrefix> {
        match std::env::var(PREFIX_VAR) {
            Ok(value) => Self::with_prefix(&value),
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(InvalidPrefix(format!("{PREFIX_VAR} is not valid UTF-8")))
            }
        }
    }

    /// Read the prefix through `lookup`, which answers for one variable name.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self, InvalidPrefix> {
        Self::with_prefix(&lookup(PREFIX_VAR).unwrap_or_default())
    }

    /// The environment under `prefix`: empty, or `^[A-Z][A-Z0-9_]*_$`.
    pub fn with_prefix(prefix: &str) -> Result<Self, InvalidPrefix> {
        if prefix.is_empty() || is_valid_prefix(prefix) {
            Ok(Self {
                prefix: prefix.to_string(),
            })
        } else {
            Err(InvalidPrefix(format!(
                "{PREFIX_VAR} must be upper-case ASCII letters, digits and underscores, \
                 starting with a letter and ending with an underscore \
                 (^[A-Z][A-Z0-9_]*_$); got {prefix:?}"
            )))
        }
    }

    /// The prefix in use; empty when none is set.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The variable this process reads for `name`.
    pub fn name(&self, name: &str) -> String {
        format!("{}{name}", self.prefix)
    }

    /// The value of `name` under the prefix, through `lookup`.
    pub fn lookup(&self, name: &str, lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
        lookup(&self.name(name))
    }

    /// The value of `name` under the prefix, from the process environment. A
    /// value that is not valid UTF-8 reads as unset.
    pub fn var(&self, name: &str) -> Option<String> {
        self.lookup(name, |name| std::env::var(name).ok())
    }
}

fn is_valid_prefix(prefix: &str) -> bool {
    let bytes = prefix.as_bytes();
    bytes.first().is_some_and(u8::is_ascii_uppercase)
        && bytes.last() == Some(&b'_')
        && bytes
            .iter()
            .all(|&b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn an_unset_or_empty_prefix_reads_every_variable_by_its_own_name() {
        for lookup in [
            vars(&[("DATABASE_URL", "bare")]),
            vars(&[(PREFIX_VAR, ""), ("DATABASE_URL", "bare")]),
        ] {
            let env = Env::from_lookup(&lookup).unwrap();
            assert_eq!(env.prefix(), "");
            assert_eq!(env.name("DATABASE_URL"), "DATABASE_URL");
            assert_eq!(env.lookup("DATABASE_URL", &lookup).as_deref(), Some("bare"));
        }
    }

    #[test]
    fn a_set_prefix_names_every_variable() {
        let lookup = vars(&[
            (PREFIX_VAR, "BILLING_WORKER_"),
            ("BILLING_WORKER_DATABASE_URL", "worker"),
            ("DATABASE_URL", "bare"),
        ]);
        let env = Env::from_lookup(&lookup).unwrap();
        assert_eq!(env.prefix(), "BILLING_WORKER_");
        assert_eq!(env.name("STRIPE_API_KEY"), "BILLING_WORKER_STRIPE_API_KEY");
        assert_eq!(
            env.lookup("DATABASE_URL", &lookup).as_deref(),
            Some("worker")
        );
    }

    #[test]
    fn a_set_prefix_never_falls_back_to_the_bare_name() {
        let lookup = vars(&[(PREFIX_VAR, "GATEWAY_"), ("DATABASE_URL", "bare")]);
        let env = Env::from_lookup(&lookup).unwrap();
        assert_eq!(env.lookup("DATABASE_URL", &lookup), None);
    }

    #[test]
    fn the_prefix_variable_itself_is_read_bare() {
        let lookup = vars(&[("GATEWAY_EIDOLA_ENV_PREFIX", "GATEWAY_")]);
        assert_eq!(Env::from_lookup(lookup).unwrap().prefix(), "");
    }

    #[test]
    fn a_prefix_must_be_upper_case_and_end_with_an_underscore() {
        for valid in ["A_", "GATEWAY_", "BILLING_RECEIVER_", "B2_", "A__", "X_1_"] {
            assert!(
                Env::with_prefix(valid).is_ok(),
                "{valid:?} should be accepted"
            );
        }
        for invalid in [
            "_", "GATEWAY", "gateway_", "Gateway_", "1A_", "_A_", "A-B_", "A B_", " A_", "A_ ",
            "É_", "A\n_",
        ] {
            let error = Env::with_prefix(invalid).expect_err(invalid);
            assert!(
                error.to_string().contains(PREFIX_VAR),
                "the error names the variable: {error}"
            );
        }
    }
}
