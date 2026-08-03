//! "Open at login" — the opt-in auto-start (task 17, wave 3).
//!
//! Opt-in, never default (decided). On macOS this is a `SMAppService` login
//! item for the app bundle itself; the Linux analogue is `eidola service
//! install` enabling the systemd user unit, which wave 2 shipped and which
//! this module deliberately does not duplicate.
//!
//! **The system is the source of truth, not our config.** `SMAppService`
//! already stores the registration, the user can revoke it in System Settings
//! → General → Login Items, and a second copy in our database would be a
//! second answer to one question. So the toggle reads [`state`] and writes
//! [`set`]; nothing is persisted app-side. The cost is that a change made in
//! System Settings while a Settings window is open is not noticed until the
//! pane is reopened — the honest trade for having one answer.

/// What the system says about our login item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoginItemState {
    /// Off — the app does not start at login.
    Off,
    /// On.
    On,
    /// Registered, but waiting on the user in System Settings (this is what
    /// macOS reports after the user has revoked consent there).
    NeedsApproval,
    /// The system cannot manage this app as a login item — a non-macOS
    /// platform, a pre-Ventura system, or a build running outside a signed
    /// `.app` bundle (a bare `cargo run`). Never presented as "off": the
    /// difference between "you turned it off" and "this cannot be turned on"
    /// is exactly what a user needs to know.
    Unsupported,
}

impl LoginItemState {
    /// Whether the toggle reads as on. `NeedsApproval` counts: we *are*
    /// registered, and the switch showing off beside "approve it in System
    /// Settings" would tell the user to enable something already enabled.
    pub fn is_on(self) -> bool {
        matches!(self, Self::On | Self::NeedsApproval)
    }

    /// Whether the toggle can be operated at all.
    pub fn is_settable(self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// The muted line under the toggle.
    pub fn description(self) -> &'static str {
        match self {
            Self::Off => "Eidola stays out of the way until you open it.",
            Self::On => "Eidola starts with your session, ready in the menu bar.",
            Self::NeedsApproval => {
                "Turned on, but macOS is waiting for you — allow Eidola in System Settings → \
                 General → Login Items."
            }
            #[cfg(target_os = "macos")]
            Self::Unsupported => {
                "Unavailable — macOS manages login items only for an installed, signed app."
            }
            #[cfg(not(target_os = "macos"))]
            Self::Unsupported => {
                "Unavailable here — on Linux, `eidola service install` enables the user service."
            }
        }
    }
}

/// Map the raw `SMAppServiceStatus` value. Split out as a pure function so
/// the mapping is unit-tested on every platform, not only where the
/// framework exists.
///
/// `NotFound` means the system has no such service to speak of — the
/// unbundled dev build case — which is `Unsupported`, not `Off`.
pub fn state_from_status(raw: isize) -> LoginItemState {
    match raw {
        0 => LoginItemState::Off,           // NotRegistered
        1 => LoginItemState::On,            // Enabled
        2 => LoginItemState::NeedsApproval, // RequiresApproval
        _ => LoginItemState::Unsupported,   // NotFound, and anything future
    }
}

#[cfg(target_os = "macos")]
pub fn state() -> LoginItemState {
    // SAFETY: `mainAppService` takes no arguments and is safe to call from
    // any thread; the returned object is retained by objc2.
    let service = unsafe { objc2_service_management::SMAppService::mainAppService() };
    state_from_status(unsafe { service.status() }.0)
}

#[cfg(not(target_os = "macos"))]
pub fn state() -> LoginItemState {
    LoginItemState::Unsupported
}

/// Turn the login item on or off. Returns the system's own message on
/// refusal — a login item can be declined for reasons we must not guess at
/// (an unsigned bundle, a user-level denial), and inventing copy for them
/// would be worse than quoting macOS.
#[cfg(target_os = "macos")]
pub fn set(enabled: bool) -> Result<(), String> {
    // SAFETY: as above; both calls are the documented register/unregister
    // pair and return their failure through the `NSError` out-parameter,
    // which objc2 surfaces as a `Result`.
    let service = unsafe { objc2_service_management::SMAppService::mainAppService() };
    let result = unsafe {
        if enabled {
            service.registerAndReturnError()
        } else {
            service.unregisterAndReturnError()
        }
    };
    result.map_err(|e| e.localizedDescription().to_string())
}

#[cfg(not(target_os = "macos"))]
pub fn set(enabled: bool) -> Result<(), String> {
    let _ = enabled;
    Err("Opening at login is not available on this platform.".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_or_missing_service_is_never_reported_as_merely_off() {
        assert_eq!(state_from_status(0), LoginItemState::Off);
        assert_eq!(state_from_status(1), LoginItemState::On);
        assert_eq!(state_from_status(2), LoginItemState::NeedsApproval);
        // NotFound (3) and anything a future macOS adds.
        assert_eq!(state_from_status(3), LoginItemState::Unsupported);
        assert_eq!(state_from_status(97), LoginItemState::Unsupported);
    }

    #[test]
    fn awaiting_approval_still_reads_as_on() {
        assert!(LoginItemState::On.is_on());
        assert!(LoginItemState::NeedsApproval.is_on());
        assert!(!LoginItemState::Off.is_on());
        assert!(!LoginItemState::Unsupported.is_on());
        assert!(!LoginItemState::Unsupported.is_settable());
        assert!(LoginItemState::Off.is_settable());
    }
}
