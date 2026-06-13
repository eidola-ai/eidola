//! `Loadable<T>` — the universal async cell described in
//! `docs/architecture/state.md` ("Loadable — the universal async cell").
//!
//! A store exposes `Loadable<T>` snapshots; views match on them exhaustively.
//! The in-flight `Task` that drives a load lives in a *sibling* field on the
//! store, never inside the `Loadable` itself — so `Loadable` stays cheap to
//! clone and a view can read it without touching task ownership.
//!
//! State-machine rules baked in here (per the doctrine):
//!
//! - **`Loading` is for initial loads only.** A re-fetch over existing data
//!   keeps `Loaded { stale: true }` visible — re-fetches must never blank a
//!   page. `to_loading()` enforces this: it moves to `Loading` only from
//!   `NotLoaded` (or another `Loading`); over `Loaded`/`Failed{prior}` it
//!   marks the existing snapshot stale instead.
//! - **Every spinner maps to a live task.** A store must only ever hold
//!   `Loading` while its sibling task field is `Some`; `debug_assert`s in the
//!   stores guard that invariant.

use eidola_app_core::error::AppError;

/// The universal async cell. See the module docs and
/// `docs/architecture/state.md`.
#[derive(Clone, Debug, Default)]
pub enum Loadable<T> {
    /// Never requested. Render nothing or a quiet placeholder.
    #[default]
    NotLoaded,
    /// A real initial-load request is in flight. The driving `Task` lives in
    /// a sibling field on the store, not here, so `Loadable` stays cloneable.
    Loading,
    /// Data present. `stale` means an invalidation arrived after this
    /// snapshot was taken; a refresh task may already be replacing it.
    Loaded { value: T, stale: bool },
    /// The last attempt failed; `prior` may retain the previous snapshot so
    /// the UI can show old-data-plus-error rather than a blank page.
    Failed { error: AppError, prior: Option<T> },
}

impl<T> Loadable<T> {
    /// A freshly `Loaded` value (not stale).
    pub fn loaded(value: T) -> Self {
        Loadable::Loaded {
            value,
            stale: false,
        }
    }

    /// The current value, if one is present (`Loaded`, or `Failed` carrying a
    /// prior snapshot). `None` for `NotLoaded`/`Loading`/`Failed{prior:None}`.
    pub fn value(&self) -> Option<&T> {
        match self {
            Loadable::Loaded { value, .. } => Some(value),
            Loadable::Failed { prior, .. } => prior.as_ref(),
            _ => None,
        }
    }

    /// Mutable access to the current value, if one is present.
    pub fn value_mut(&mut self) -> Option<&mut T> {
        match self {
            Loadable::Loaded { value, .. } => Some(value),
            Loadable::Failed { prior, .. } => prior.as_mut(),
            _ => None,
        }
    }

    /// The error, if the cell is in the `Failed` state.
    pub fn error(&self) -> Option<&AppError> {
        match self {
            Loadable::Failed { error, .. } => Some(error),
            _ => None,
        }
    }

    /// True while an *initial* load is in flight (the `Loading` state). A
    /// stale re-fetch over existing data is **not** loading — it stays
    /// `Loaded { stale: true }`.
    pub fn is_loading(&self) -> bool {
        matches!(self, Loadable::Loading)
    }

    /// True for a `Loaded` snapshot that an invalidation has since marked
    /// stale (a refresh is presumably under way).
    pub fn is_stale(&self) -> bool {
        matches!(self, Loadable::Loaded { stale: true, .. })
    }

    /// True if a value is present (`Loaded`, or `Failed` with a prior).
    pub fn has_value(&self) -> bool {
        self.value().is_some()
    }

    /// Transition into the in-flight state for the start of a load, *without*
    /// blanking existing data:
    ///
    /// - `NotLoaded`/`Loading` → `Loading` (a genuine initial load).
    /// - `Loaded { value, .. }` → `Loaded { value, stale: true }` (a re-fetch
    ///   over data keeps the data visible; the spinner is the `stale` flag,
    ///   not a blank `Loading`).
    /// - `Failed { prior: Some(value), .. }` → `Loaded { value, stale: true }`
    ///   (retry over the retained snapshot keeps showing it).
    /// - `Failed { prior: None, .. }` → `Loading` (nothing to keep visible).
    ///
    /// Takes `self` by value (call sites do `self.x = self.x.to_loading()`),
    /// so the predecessor is consumed rather than cloned.
    #[must_use]
    pub fn to_loading(self) -> Self {
        match self {
            Loadable::NotLoaded | Loadable::Loading => Loadable::Loading,
            Loadable::Loaded { value, .. } => Loadable::Loaded { value, stale: true },
            Loadable::Failed {
                prior: Some(value), ..
            } => Loadable::Loaded { value, stale: true },
            Loadable::Failed { prior: None, .. } => Loadable::Loading,
        }
    }

    /// Resolve a completed request into the cell, preserving the prior value
    /// on failure so the UI can show old-data-plus-error. Consumes `self` so
    /// the predecessor's value can be moved into the `Failed { prior }` slot
    /// without cloning.
    #[must_use]
    pub fn resolve(self, result: Result<T, AppError>) -> Self {
        match result {
            Ok(value) => Loadable::loaded(value),
            Err(error) => Loadable::Failed {
                error,
                prior: self.into_value(),
            },
        }
    }

    /// Consume the cell and return the value it holds, if any.
    pub fn into_value(self) -> Option<T> {
        match self {
            Loadable::Loaded { value, .. } => Some(value),
            Loadable::Failed { prior, .. } => prior,
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err() -> AppError {
        AppError::Internal {
            message: "boom".into(),
        }
    }

    #[test]
    fn default_is_not_loaded() {
        let l: Loadable<i32> = Loadable::default();
        assert!(matches!(l, Loadable::NotLoaded));
        assert!(!l.is_loading());
        assert!(l.value().is_none());
    }

    #[test]
    fn initial_load_goes_through_loading() {
        let l: Loadable<i32> = Loadable::NotLoaded.to_loading();
        assert!(l.is_loading(), "an initial load must show Loading");
        assert!(l.value().is_none());
    }

    #[test]
    fn refetch_over_data_stays_loaded_stale_not_loading() {
        let l = Loadable::loaded(7).to_loading();
        // No blank Loading flash: the value stays visible, just marked stale.
        assert!(
            !l.is_loading(),
            "a re-fetch over data must not blank to Loading"
        );
        assert!(l.is_stale());
        assert_eq!(l.value(), Some(&7));
    }

    #[test]
    fn resolve_ok_clears_stale() {
        let l = Loadable::loaded(1).to_loading().resolve(Ok(2));
        assert!(!l.is_stale());
        assert_eq!(l.value(), Some(&2));
    }

    #[test]
    fn resolve_err_retains_prior() {
        let l = Loadable::loaded(5).resolve(Err(err()));
        assert_eq!(
            l.value(),
            Some(&5),
            "failure keeps the prior snapshot visible"
        );
        assert!(l.error().is_some());
    }

    #[test]
    fn resolve_err_from_empty_has_no_prior() {
        let l: Loadable<i32> = Loadable::Loading.resolve(Err(err()));
        assert!(l.value().is_none());
        assert!(l.error().is_some());
    }

    #[test]
    fn retry_over_failed_with_prior_keeps_data_visible() {
        // Failed-with-prior → to_loading() keeps the prior visible as stale,
        // never blanks to Loading.
        let failed = Loadable::loaded(9).resolve(Err(err()));
        let retry = failed.to_loading();
        assert!(!retry.is_loading());
        assert!(retry.is_stale());
        assert_eq!(retry.value(), Some(&9));
    }

    #[test]
    fn retry_over_failed_without_prior_goes_to_loading() {
        let failed: Loadable<i32> = Loadable::Loading.resolve(Err(err()));
        let retry = failed.to_loading();
        assert!(retry.is_loading());
    }
}
