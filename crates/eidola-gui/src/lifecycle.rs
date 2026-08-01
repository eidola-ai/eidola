//! App lifecycle — how long the process lives, what closing a window means,
//! and what a reactivation does (task 17, wave 2).
//!
//! The app is becoming longer-lived than its windows: loaded local engines,
//! and soon continuous capture and an OpenAI-compatible proxy, all belong to
//! the *process*, not to whichever window happens to be open. This module
//! holds the small set of decisions that follow from that, kept out of
//! `lib.rs` so the lifecycle rules read in one place.
//!
//! ## gpui's seams at our pin (spike, task 17 wave 2)
//!
//! The spike question was whether gpui at our pinned commit lets every window
//! close without taking the process with it. It does, and the seams are all
//! first-class:
//!
//! - **[`gpui::QuitMode`]** (`Application::with_quit_mode` / `App::set_quit_mode`)
//!   decides whether the window-close trail calls `cx.quit()` when the last
//!   window goes. `QuitMode::Default` already resolves to *don't quit* on
//!   macOS and *quit* everywhere else — so macOS has outlived its windows for
//!   as long as we've been on this pin. We now name the mode explicitly
//!   rather than inheriting a `cfg!` from upstream, because on Linux it is a
//!   *launch-mode* decision (a `--windowless` service must not die when a
//!   visiting window closes), which no upstream default can make for us.
//! - **`Application::on_reopen`** (macOS `applicationShouldHandleReopen:`) is
//!   the Dock-click / Spotlight-relaunch signal. On Linux the callback is
//!   stored but nothing ever invokes it — there is no platform mechanism —
//!   which is exactly why "a second launch asks the running instance to open
//!   a window" is the wave-4 socket's job there and not something wave 2 can
//!   fake.
//! - **`App::on_app_quit`** runs (async, bounded by `gpui::SHUTDOWN_TIMEOUT`,
//!   200ms) from AppKit's `applicationWillTerminate:`. It is the only hook
//!   that runs on the macOS quit path at all: `[NSApp terminate:]` ends in
//!   `exit()`, so no Rust destructor downstream of `main` ever runs. That is
//!   what [`install_engine_shutdown`] exists for.
//! - **`App::on_window_closed`** is an observer, not a policy hook — it fires
//!   *before* gpui's own quit-on-empty check. We used to hang the Linux
//!   "quit with the last window" rule off it; that is now the `QuitMode`,
//!   which is the same behaviour expressed at the one seam that can also say
//!   "no".
//!
//! No fork, no upstream gap. Everything wave 2 needs is on the `Application`
//! builder or on `App`.

use gpui::{App, QuitMode};

use crate::stores::Stores;

/// Usage text for the GUI binary. Short by design — the GUI takes no real
/// configuration; `--windowless` is the service launch mode.
pub const USAGE: &str = "\
Eidola — a native client for confidential inference.

Usage: eidola-gui [OPTIONS]

Options:
      --windowless   Run with no window. The process hosts the app (loaded
                     local engines, background polling) and waits; on Linux
                     this is the systemd user-service mode — see
                     `eidola service install`.
  -h, --help         Print this help
";

/// How the process was launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaunchOptions {
    /// Start with no window and keep running with none. The process is the
    /// service; a window is a visitor.
    pub windowless: bool,
}

impl LaunchOptions {
    /// Parse the process arguments (excluding argv[0]).
    pub fn from_env() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        Self::parse(&args)
    }

    /// Parse an argument list. Deliberately lenient about anything it does
    /// not recognise: macOS hands launched apps flags of its own
    /// (`-psn_0_…` from LaunchServices, `-NSDocumentRevisionsDebugMode` from
    /// Xcode), and refusing to start over one of those would be a bug, not
    /// rigour.
    pub fn parse<S: AsRef<str>>(args: &[S]) -> Self {
        Self {
            windowless: args.iter().any(|a| a.as_ref() == "--windowless"),
        }
    }

    /// True when the argument list asks for the usage text.
    pub fn wants_help<S: AsRef<str>>(args: &[S]) -> bool {
        args.iter().any(|a| matches!(a.as_ref(), "--help" | "-h"))
    }
}

/// When the process should quit by itself.
///
/// Three cases, and the reasoning is different in each:
///
/// - **Windowless (either platform).** Never. The window-less process *is*
///   the running app; a window that opens and closes over it is a visitor,
///   and letting the last visitor leaving stop the service would defeat the
///   launch mode entirely.
/// - **macOS with windows.** Never — the platform idiom, and the reason
///   `on_reopen` exists. Quit is `⌘Q` / the menu, and it is a full shutdown
///   (decided): engines, everything.
/// - **Linux with windows.** With the last window closed. Until the wave-3
///   tray (best-effort, never load-bearing) and the wave-4 socket, a Linux
///   process with no window has no way to be reached or quit — a lingering
///   headless process would just be a leak. A user who *wants* the
///   long-lived process asks for it explicitly with `--windowless`.
pub fn quit_mode(opts: LaunchOptions) -> QuitMode {
    if opts.windowless || cfg!(target_os = "macos") {
        QuitMode::Explicit
    } else {
        QuitMode::LastWindowClosed
    }
}

/// Handle a reactivation (macOS Dock click, Spotlight relaunch of an
/// already-running app, or a second `open -a Eidola`).
///
/// Focus a window if there is one — the active window if the platform still
/// names one, else the most recently opened — and otherwise call
/// `open_when_empty`, which production wires to the Library: the app's table
/// of contents is the honest door back into a process that has been running
/// without a face. `cx.activate(true)` runs either way, because a reopen can
/// arrive while another app is foreground.
///
/// `open_when_empty` is a parameter rather than a direct call so the decision
/// is testable without `AppGlobal` installed.
pub fn reactivate(cx: &mut App, open_when_empty: impl FnOnce(&mut App)) {
    let target = cx.active_window().or_else(|| cx.windows().last().copied());
    match target {
        Some(handle) => {
            handle
                .update(cx, |_, window, _| window.activate_window())
                .ok();
        }
        None => open_when_empty(cx),
    }
    cx.activate(true);
}

/// Best-effort teardown of local inference engines when the app quits.
///
/// Quit is a full shutdown — engines, everything (decided) — and on macOS
/// nothing else delivers it: `cx.quit()` becomes `[NSApp terminate:]`, which
/// ends in `exit()`, so the tokio runtime is never dropped and the
/// `kill_on_drop` reaping that covers the Linux path (where the event loop
/// simply returns and `main` unwinds) never happens. Without this hook a
/// quit on macOS leaves every loaded `llama-server` running, holding
/// gigabytes, adopted by launchd.
///
/// Bounded by `gpui::SHUTDOWN_TIMEOUT` (200ms), so it is one round trip: the
/// whole sweep runs inside a single bridged future on the core's runtime.
/// Signalling an engine's supervisor is a channel send, and the supervisor is
/// already parked on it, so the `start_kill` that follows lands in
/// microseconds; the short sleep is insurance, not a wait.
///
/// This is best-effort by construction and says so: a quit that outruns the
/// budget still exits. A hard guarantee wants the OS to enforce it
/// (`PR_SET_PDEATHSIG` on Linux; macOS has no equivalent and would need an
/// explicit synchronous reap in app-core) — a follow-up, not wave 2.
pub fn install_engine_shutdown(stores: &Stores, cx: &mut App) {
    let core = stores.app_core();
    cx.on_app_quit(move |_cx| {
        let core = core.clone();
        async move {
            let Some(core) = core else { return };
            let _ = crate::bridge::bridge(core, |c| async move {
                let state = c.local_models_state().await?;
                for id in loaded_engine_ids(&state) {
                    let _ = c.unload_local_model(id).await;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        }
    })
    .detach();
}

/// Every currently-serving engine's selection id — the managed `local`
/// store's plus every `llamacpp` backend's. `Loading` entries are included:
/// a warming engine has a live subprocess too.
pub fn loaded_engine_ids(state: &eidola_app_core::LocalModelsState) -> Vec<String> {
    use eidola_app_core::LocalModelStatus as S;
    let running = |s: &S| matches!(s, S::Loaded { .. } | S::Loading);
    state
        .models
        .iter()
        .filter(|m| running(&m.status))
        .map(|m| m.id.clone())
        .chain(
            state
                .external
                .iter()
                .flat_map(|b| b.models.iter())
                .filter(|m| running(&m.status))
                .map(|m| m.id.clone()),
        )
        .collect()
}

/// Quit cleanly on `SIGTERM` / `SIGINT` (windowless mode).
///
/// A systemd user service is stopped with `SIGTERM`, and the default
/// disposition would kill the process outright — skipping the quit
/// observers, and with them the engine teardown above. So in windowless mode
/// we translate the signal into an ordinary `cx.quit()`: the signal is
/// awaited on the core's tokio runtime (the one runtime in the process that
/// can host `tokio::signal`) and handed to gpui through the same `oneshot`
/// bridge every other core → gpui hop uses.
///
/// Returns the gpui task; the caller detaches it (app-lifetime, like the bus
/// bridge). A missing core, or a platform that refuses the handler, leaves
/// the process on the default disposition — which still stops, just less
/// politely.
#[cfg(unix)]
pub fn install_signal_quit(stores: &Stores, cx: &mut App) -> gpui::Task<()> {
    let core = stores.app_core();
    cx.spawn(async move |cx: &mut gpui::AsyncApp| {
        let Some(core) = core else { return };
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        core.runtime().handle().spawn(async move {
            use tokio::signal::unix::{SignalKind, signal};
            let (Ok(mut term), Ok(mut int)) = (
                signal(SignalKind::terminate()),
                signal(SignalKind::interrupt()),
            ) else {
                return;
            };
            tokio::select! {
                _ = term.recv() => {}
                _ = int.recv() => {}
            }
            let _ = tx.send(());
        });
        if rx.await.is_ok() {
            cx.update(|cx| cx.quit());
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_recognizes_windowless_and_ignores_platform_flags() {
        assert!(!LaunchOptions::parse::<&str>(&[]).windowless);
        assert!(LaunchOptions::parse(&["--windowless"]).windowless);
        // LaunchServices / Xcode hand these to a launched .app; ignoring
        // them is the point, not an oversight.
        let noisy = ["-psn_0_1234567", "-NSDocumentRevisionsDebugMode", "YES"];
        assert!(!LaunchOptions::parse(&noisy).windowless);
        let mixed = ["-psn_0_1234567", "--windowless"];
        assert!(LaunchOptions::parse(&mixed).windowless);
    }

    #[test]
    fn help_is_recognized() {
        assert!(LaunchOptions::wants_help(&["--help"]));
        assert!(LaunchOptions::wants_help(&["-h"]));
        assert!(!LaunchOptions::wants_help(&["--windowless"]));
    }

    #[test]
    fn windowless_never_quits_on_its_own() {
        let windowless = LaunchOptions { windowless: true };
        assert_eq!(quit_mode(windowless), QuitMode::Explicit);
    }

    #[test]
    fn windowed_quit_mode_follows_the_platform_idiom() {
        let windowed = LaunchOptions::default();
        if cfg!(target_os = "macos") {
            assert_eq!(quit_mode(windowed), QuitMode::Explicit);
        } else {
            assert_eq!(quit_mode(windowed), QuitMode::LastWindowClosed);
        }
    }
}
