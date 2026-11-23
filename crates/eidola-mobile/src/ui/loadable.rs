//! `Loadable<T>` — the async cell every store snapshot lives in
//! (`crates/eidola-gui/STATE.md` → Loadable): a re-fetch over data keeps it
//! shown and marks it stale; `Loading` is initial-load-only and always means a
//! live task in the store's sibling slot; a failure keeps the prior value so a
//! surface shows old data plus the error rather than a blank page.

use eidola_app_core::error::AppError;

#[derive(Clone, Debug)]
pub enum Loadable<T> {
    NotLoaded,
    Loading,
    Loaded { value: T, stale: bool },
    Failed { error: AppError, prior: Option<T> },
}

impl<T> Loadable<T> {
    /// The cell as a request starts: data on screen stays (stale), nothing
    /// becomes `Loading`.
    pub fn to_loading(self) -> Self {
        match self {
            Self::Loaded { value, .. }
            | Self::Failed {
                prior: Some(value), ..
            } => Self::Loaded { value, stale: true },
            _ => Self::Loading,
        }
    }

    /// The cell once a request answered, on either arm: a failure keeps the
    /// value it had, so the surface owes a quiet retry rather than a blank.
    pub fn resolve(self, result: Result<T, AppError>) -> Self {
        match result {
            Ok(value) => Self::Loaded {
                value,
                stale: false,
            },
            Err(error) => Self::Failed {
                error,
                prior: self.into_value(),
            },
        }
    }

    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Loaded { value, .. }
            | Self::Failed {
                prior: Some(value), ..
            } => Some(value),
            _ => None,
        }
    }

    pub fn value_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Loaded { value, .. }
            | Self::Failed {
                prior: Some(value), ..
            } => Some(value),
            _ => None,
        }
    }

    pub fn into_value(self) -> Option<T> {
        match self {
            Self::Loaded { value, .. }
            | Self::Failed {
                prior: Some(value), ..
            } => Some(value),
            _ => None,
        }
    }

    pub fn error(&self) -> Option<&AppError> {
        match self {
            Self::Failed { error, .. } => Some(error),
            _ => None,
        }
    }

    /// Nothing to show yet and a request is (or should be) running.
    pub fn is_loading(&self) -> bool {
        matches!(self, Self::Loading | Self::NotLoaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> AppError {
        AppError::Internal {
            message: "x".into(),
        }
    }

    #[test]
    fn a_refresh_never_blanks_what_is_shown() {
        let cell = Loadable::Loaded {
            value: 1,
            stale: false,
        };
        assert!(matches!(
            cell.to_loading(),
            Loadable::Loaded {
                value: 1,
                stale: true
            }
        ));
        assert!(matches!(
            Loadable::<i32>::NotLoaded.to_loading(),
            Loadable::Loading
        ));
    }

    #[test]
    fn a_failure_keeps_the_prior_value() {
        let cell = Loadable::Loaded {
            value: 1,
            stale: true,
        }
        .resolve(Err(err()));
        assert_eq!(cell.value(), Some(&1));
        assert!(cell.error().is_some());
        let first = Loadable::<i32>::Loading.resolve(Err(err()));
        assert_eq!(first.value(), None);
    }
}
