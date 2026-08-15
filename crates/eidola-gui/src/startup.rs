//! The startup-failure surface — what the app says when it cannot come up at
//! all.
//!
//! `AppCore::new` is fallible (`crates/eidola-app-core/AGENTS.md` → Database):
//! it takes the exclusive advisory lock on the single-writer local database, so
//! a second Eidola gets `AppError::DatabaseInUse` naming the holder's pid; the
//! config/data directories can fail to resolve; a database at an unknown schema
//! version is refused outright. **The error messages were already good; the
//! presentation was a crash.** Construction happened inside the closure passed
//! to `Application::run`, which on macOS runs inside AppKit's
//! `applicationDidFinishLaunching:` — an `extern "C"` frame a Rust panic cannot
//! unwind through, so the panic escalated to `panic_cannot_unwind` and the
//! process died with SIGABRT (measured: exit 134). In a packaged `.app` that is
//! a macOS crash report for a state the app understands perfectly well.
//!
//! So construction moves **before** anything gpui-shaped exists and a failure
//! is reported here instead: a native alert on macOS, stderr everywhere, and a
//! non-zero exit either way. Deliberately no retry loop and no waiting for the
//! other process to quit — the honest thing to do with "another Eidola has
//! this open" is to say so and stop.
//!
//! **The panic remains the fallback.** If the alert cannot be shown at all
//! (no AppKit to talk to), the failure stays loud rather than becoming a
//! silent exit from a process that never drew anything.

use eidola_app_core::error::AppError;

/// The alert's title and body for a construction failure.
///
/// The body is the error's own `Display` and nothing else: these strings are
/// written where the failure is understood (app-core's typed errors), and
/// re-wording them here would put a second, staler account of the same fact in
/// the one place a reader has no way to check it against. The title is the only
/// thing this layer decides, because "already open" and "cannot start" are
/// different situations to walk into — the first is a thing the reader can fix
/// in one move, and the message says how.
///
/// Every other variant shares one title rather than being enumerated: this must
/// render *any* construction error, including ones added after it was written,
/// and a match that has to be extended to stay honest is a match that will not
/// be.
pub fn dialog_text(error: &AppError) -> (&'static str, String) {
    let title = match error {
        AppError::DatabaseInUse { .. } => "Eidola is already open",
        _ => "Eidola can’t start",
    };
    (title, error.to_string())
}

/// Report a startup failure and end the process. Never returns.
///
/// stderr first, unconditionally: a terminal launch, a systemd unit and
/// `Console.app`'s unified log all read it, and it is the whole surface on
/// Linux for now (the service/journal world the `--windowless` mode lives in
/// makes stderr the honest channel there).
pub fn report_and_exit(error: &AppError) -> ! {
    let (title, message) = dialog_text(error);
    eprintln!("{title}: {message}");

    #[cfg(target_os = "macos")]
    if !macos::show_alert(title, &message) {
        // No alert and no window: the app would otherwise disappear between
        // the Dock bounce and nothing at all. Keep the loud failure.
        panic!("{title}: {message}");
    }

    std::process::exit(1);
}

#[cfg(target_os = "macos")]
mod macos {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSAlert, NSAlertStyle, NSApplication, NSApplicationActivationPolicy};
    use objc2_foundation::NSString;

    /// Show a modal alert with a single Quit button. Returns whether it was
    /// actually presented.
    ///
    /// `sharedApplication` creates `NSApp` if nothing has yet, and `runModal`
    /// runs its own modal loop — so this works before `Application::run` and
    /// without one, which is the entire point of reporting here. The activation
    /// policy is set to `Regular` so the alert comes up in front of whatever
    /// the reader was doing rather than behind it; there is no window to leave
    /// behind, because the process exits as soon as the button is pressed.
    pub(super) fn show_alert(title: &str, message: &str) -> bool {
        // No main-thread marker means we are not on the main thread, where
        // AppKit refuses to run a modal loop. `run_with` calls this from
        // `main`, so this is a guard rather than a case.
        let Some(mtm) = MainThreadMarker::new() else {
            return false;
        };
        let app = NSApplication::sharedApplication(mtm);
        app.setActivationPolicy(NSApplicationActivationPolicy::Regular);
        // `-[NSApplication activate]` is macOS 14+ and our binaries declare
        // `minos 11.0` (the bundle's own floor is 13; see `Support/Info.plist`
        // and `login_item.rs`) — objc2 cannot enforce an availability
        // attribute, so calling it would be an unrecognized selector on an
        // older system, crashing the very path that exists to avoid a crash.
        // The deprecated spelling is a harmless no-op on 14+ (Apple's note:
        // "will have no effect"), which is the right way round.
        // *Removal trigger: a shipped floor of macOS 14 in both the plist and
        // `minos`, after which this becomes `app.activate()`.*
        #[allow(deprecated)]
        app.activateIgnoringOtherApps(true);

        let alert = NSAlert::new(mtm);
        alert.setAlertStyle(NSAlertStyle::Critical);
        alert.setMessageText(&NSString::from_str(title));
        alert.setInformativeText(&NSString::from_str(message));
        alert.addButtonWithTitle(&NSString::from_str("Quit"));
        alert.runModal();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dialog_says_what_happened_in_the_errors_own_words() {
        let in_use = AppError::DatabaseInUse {
            pid: Some(4321),
            message: "another Eidola process (pid 4321) has this database open. \
                      Quit it and try again."
                .into(),
        };
        let (title, message) = dialog_text(&in_use);
        assert_eq!(title, "Eidola is already open");
        assert_eq!(
            message,
            in_use.to_string(),
            "the body is the typed error's own text, never a second telling of it"
        );

        // Any other construction failure still renders — the surface is for
        // the class, not for one variant.
        let (title, message) = dialog_text(&AppError::Config {
            message: "could not determine the Eidola config directory".into(),
        });
        assert_eq!(title, "Eidola can’t start");
        assert!(message.contains("config directory"), "got {message:?}");
    }
}
