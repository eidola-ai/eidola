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
use gpui::SharedString;

use crate::i18n::msg_in;

/// The alert's title and body for a construction failure, in `locale`.
///
/// **The body is the error's own `Display` and nothing else** — and that is
/// also why it is the one part of this surface that stays English: app-core is
/// locale-free by the layering rule (typed errors out, the presentation layer
/// localizes), and the payload a faithful translation would need is not in the
/// variant to read. `DatabaseInUse` carries a pid and a prose message with the
/// database path inside it, not the path itself, so a localized body would
/// either lose the path or restate a fact this layer cannot check. Re-wording it
/// here would put a second, staler account of the same failure in the one place
/// a reader has no way to compare it against.
///
/// The title is what this layer decides, because "already open" and "cannot
/// start" are different situations to walk into — the first is a thing the
/// reader can fix in one move, and the message says how. Every other variant
/// shares one title rather than being enumerated: this must render *any*
/// construction error, including ones added after it was written, and a match
/// that has to be extended to stay honest is a match that will not be.
pub fn dialog_text(locale: &str, error: &AppError) -> (SharedString, String) {
    let title = match error {
        AppError::DatabaseInUse { .. } => msg_in::startup_title_already_open(locale),
        _ => msg_in::startup_title_failed(locale),
    };
    (title, error.to_string())
}

/// The locale this surface speaks, resolved **without gpui**.
///
/// The whole path is pure: the stored preference is read straight from
/// `config.toml` (`Config::load_from` needs no `AppCore`, which matters here —
/// the failure being reported may be that the core could not be built at all),
/// then negotiated against the OS's preferred languages exactly as
/// `i18n::wire_config` does once there is an `App`. A config path that will not
/// resolve is not an error to report on top of an error: it simply leaves the
/// system's own languages to decide, which is what "no stored preference"
/// already means.
pub fn locale() -> &'static str {
    let stored = eidola_app_core::config::default_config_path()
        .map(|path| eidola_app_core::config::Config::load_from(&path))
        .and_then(|config| config.language().map(str::to_string));
    crate::i18n::resolve(stored.as_deref(), &crate::i18n::system_preferred())
}

/// Report a startup failure in `locale` and end the process. Never returns.
///
/// stderr first, unconditionally: a terminal launch, a systemd unit and
/// `Console.app`'s unified log all read it, and it is the whole surface on
/// Linux for now (the service/journal world the `--windowless` mode lives in
/// makes stderr the honest channel there).
pub fn report_and_exit(locale: &str, error: &AppError) -> ! {
    let (title, message) = dialog_text(locale, error);
    eprintln!("{title}: {message}");

    #[cfg(target_os = "macos")]
    if !macos::show_alert(&title, &message, &msg_in::startup_quit(locale)) {
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

    /// Show a modal alert with a single button, labelled `quit`. Returns
    /// whether it was actually presented.
    ///
    /// `sharedApplication` creates `NSApp` if nothing has yet, and `runModal`
    /// runs its own modal loop — so this works before `Application::run` and
    /// without one, which is the entire point of reporting here. The activation
    /// policy is set to `Regular` so the alert comes up in front of whatever
    /// the reader was doing rather than behind it; there is no window to leave
    /// behind, because the process exits as soon as the button is pressed.
    pub(super) fn show_alert(title: &str, message: &str, quit: &str) -> bool {
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
        alert.addButtonWithTitle(&NSString::from_str(quit));
        alert.runModal();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn in_use() -> AppError {
        AppError::DatabaseInUse {
            pid: Some(4321),
            message: "another Eidola process (pid 4321) has this database open. \
                      Quit it and try again."
                .into(),
        }
    }

    #[test]
    fn the_dialog_says_what_happened_in_the_errors_own_words() {
        let in_use = in_use();
        let (title, message) = dialog_text(crate::i18n::SOURCE_LOCALE, &in_use);
        assert_eq!(title, "Eidola is already open");
        assert_eq!(
            message,
            in_use.to_string(),
            "the body is the typed error's own text, never a second telling of it"
        );

        // Any other construction failure still renders — the surface is for
        // the class, not for one variant.
        let (title, message) = dialog_text(
            crate::i18n::SOURCE_LOCALE,
            &AppError::Config {
                message: "could not determine the Eidola config directory".into(),
            },
        );
        assert_eq!(title, "Eidola can\u{2019}t start");
        assert!(message.contains("config directory"), "got {message:?}");

        // The schema refusal is the second thing `open_app_core` can hand back
        // (it opens the database, not just the core), and its message is the
        // one that says what to do — so it must arrive intact.
        let (title, message) = dialog_text(
            crate::i18n::SOURCE_LOCALE,
            &AppError::Database {
                message: "your local Eidola database is from an incompatible build \
                          (schema v1; this build expects v6). \u{2026} delete your dev \
                          database and restart"
                    .into(),
            },
        );
        assert_eq!(title, "Eidola can\u{2019}t start");
        assert!(
            message.contains("incompatible build") && message.contains("delete your dev database"),
            "the schema refusal must reach the reader whole: {message:?}"
        );
    }

    /// **The pre-gpui surface is localized like any other.** It runs before
    /// `Application::run`, so it has no `App` to read the active locale from —
    /// the `msg_in` accessors take the tag instead, and the tag is resolved by a
    /// pure path that still answers when the core could not be built.
    ///
    /// The body stays the typed error's own words in every locale: app-core is
    /// locale-free, which is the layering, not an omission.
    #[test]
    fn the_titles_and_the_button_speak_the_readers_language() {
        let in_use = in_use();
        for (tag, already_open, cannot_start, quit) in [
            (
                "es",
                "Eidola ya est\u{e1} abierto",
                "Eidola no puede iniciarse",
                "Salir",
            ),
            (
                "fr",
                "Eidola est d\u{e9}j\u{e0} ouvert",
                "Eidola ne peut pas d\u{e9}marrer",
                "Quitter",
            ),
            (
                "zh-Hans",
                "Eidola \u{5df2}\u{5728}\u{8fd0}\u{884c}",
                "Eidola \u{65e0}\u{6cd5}\u{542f}\u{52a8}",
                "\u{9000}\u{51fa}",
            ),
            (
                "zh-Hant",
                "Eidola \u{5df2}\u{5728}\u{57f7}\u{884c}",
                "Eidola \u{7121}\u{6cd5}\u{555f}\u{52d5}",
                "\u{7d50}\u{675f}",
            ),
        ] {
            let (title, message) = dialog_text(tag, &in_use);
            assert_eq!(title, already_open, "{tag}");
            assert_eq!(
                message,
                in_use.to_string(),
                "{tag}: the body is app-core's, and app-core is locale-free"
            );
            let (title, _) = dialog_text(
                tag,
                &AppError::Config {
                    message: "no config directory".into(),
                },
            );
            assert_eq!(title, cannot_start, "{tag}");
            assert_eq!(msg_in::startup_quit(tag), quit, "{tag}");
        }

        // A tag this build does not ship answers in the source locale, exactly
        // as a locale change to one would be refused.
        assert_eq!(
            dialog_text("de", &in_use).0,
            "Eidola is already open",
            "an unshipped tag falls back rather than rendering an id"
        );
    }
}
