//! "Open at login" — the opt-in auto-start (task 17, wave 3).
//!
//! Opt-in, never default (decided). On macOS this is a `SMAppService` login
//! item for the app bundle itself; the Linux analogue is `eidola service
//! install` enabling the systemd user unit, which wave 2 shipped and which
//! this module deliberately does not duplicate.
//!
//! **What login actually launches — and what it should (open question).**
//! `SMAppService.mainAppService` registers *the app bundle* as a login item,
//! and launchd launches it exactly as a Dock click would: the ordinary
//! windowed app, Dock icon and all. It takes **no arguments** — the class
//! exposes no way to pass any (see `SMAppService.h`), so `--windowless`, the
//! background/toolbar shape wave 3b made ⌘Q retire *into*, is unreachable
//! this way. Under the background-app model login ought to start that
//! background layer, not put a window in the user's face at every login.
//!
//! Nothing detects the difference at runtime, either:
//! `NSApplicationLaunchIsDefaultLaunchKey` distinguishes only file/print/
//! Service/state-restoration launches (a login launch *is* a "default"
//! launch, same as a Dock click), and the process shape is identical —
//! measured on an ordinary `open -a`: `ppid` is 1 and `XPC_SERVICE_NAME` is
//! `application.<bundle-id>.<hash>.<hash>`, because LaunchServices routes
//! every GUI launch through launchd. A timing or environment guess would
//! misfire on normal launches, which is worse than the current wrong.
//!
//! **The documented cure is a LaunchAgent**, not this class:
//! `SMAppService.agentServiceWithPlistName:` reads a plist from the bundle's
//! `Contents/Library/LaunchAgents`, and (per the header) accepts "the
//! standard launchd.plist keys" — including `ProgramArguments`, hence
//! `--windowless`. It is deliberately **not** done here: it moves the toggle
//! from System Settings' "Open at Login" to "Allow in the Background",
//! requires the plist to ship from both `package-gui-app.sh` and the Nix
//! bundle, needs a real signature to register at all (our ad-hoc dev build
//! reports `Unsupported`, so it cannot be exercised locally), wants a
//! migration for anyone already registered through `mainAppService`, and
//! re-opens the TCC-identity question task 17 settled by keeping one
//! app-bundle process. That is a product-and-packaging decision with a
//! logout/login verification cycle, not a code tidy. Until it is taken, the
//! copy below says what actually happens rather than what we intend.
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
            // Not "ready in the menu bar": `mainAppService` launches the
            // ordinary windowed app (see the module docs). Say what happens.
            Self::On => "Eidola opens with your session.",
            Self::NeedsApproval => {
                "Turned on, but macOS is waiting for you — allow Eidola in System Settings → \
                 General → Login Items."
            }
            #[cfg(target_os = "macos")]
            Self::Unsupported => {
                "Unavailable — this needs macOS 13 or later, and an installed, signed app."
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

/// Whether `SMAppService` exists in this process at all.
///
/// **This guard is load-bearing, not belt-and-braces.** `SMAppService` is
/// macOS 13+, and our binaries declare a floor of **macOS 11** (`otool -l` on
/// the packaged app: `LC_BUILD_VERSION … minos 11.0`) with no
/// `LSMinimumSystemVersion` in `Support/Info.plist` and no documented product
/// floor anywhere in the repo. `ServiceManagement.framework` itself exists on
/// 11/12, so the process links and launches — only the *class* is absent, and
/// objc2's class lookup **panics** on a miss (`objc2`'s `CachedClass::fetch`:
/// `panic!("class {name} could not be found")`). Without this check, opening
/// Settings → General on Monterey would take the whole app down from inside a
/// `Render`.
///
/// Asking the runtime for the class is the direct question, so no version
/// table (`objc2::available!`) is needed. **Removal trigger:** delete this and
/// call `SMAppService` unconditionally once the shipped floor is ≥ macOS 13 —
/// i.e. once `LSMinimumSystemVersion` says so *and* the release build's
/// `minos` agrees.
#[cfg(target_os = "macos")]
fn service_management_is_available() -> bool {
    objc2::runtime::AnyClass::get(c"SMAppService").is_some()
}

#[cfg(target_os = "macos")]
pub fn state() -> LoginItemState {
    if !service_management_is_available() {
        return LoginItemState::Unsupported;
    }
    // SAFETY: `mainAppService` takes no arguments and is safe to call from
    // any thread; the returned object is retained by objc2. The class is
    // present — checked immediately above.
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
    if !service_management_is_available() {
        // Both entry points are guarded, not just `state`: the switch is
        // inert while `Unsupported`, but a caller is a caller.
        return Err(
            "Opening at login needs macOS 13 or later — this Mac's system software is older."
                .to_string(),
        );
    }
    // SAFETY: as above; both calls are the documented register/unregister
    // pair and return their failure through the `NSError` out-parameter,
    // which objc2 surfaces as a `Result`. The class is present — checked
    // immediately above.
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

    /// The pre-Ventura guard has to *discriminate*, or it is a check that
    /// always says yes and protects nothing. Both halves are asserted: the
    /// real class resolves here (so the guard does not disable the feature on
    /// a supported Mac), and a name the runtime has never heard of does not
    /// (so a missing `SMAppService` really would be caught, rather than the
    /// lookup succeeding for everything).
    #[cfg(target_os = "macos")]
    #[test]
    fn the_availability_guard_answers_no_for_a_class_that_is_not_there() {
        use objc2::runtime::AnyClass;
        assert!(
            service_management_is_available(),
            "the test machine is macOS 13+, so the class must resolve"
        );
        assert!(AnyClass::get(c"SMAppServiceThatDoesNotExist").is_none());
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
