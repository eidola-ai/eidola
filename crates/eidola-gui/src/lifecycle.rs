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

use crate::stores::{BusBridge, Stores};

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
///   `on_reopen` exists. Leaving is `⌘Q`, which since wave 3b **retires the
///   app to the background** behind its status item rather than ending it
///   ([`crate::status_item::quit_or_retire`]); the full shutdown is the
///   status menu's own Quit.
/// - **Linux with windows.** With the last window closed. There is no tray
///   (decided): until the wave-4 socket, a Linux process with no window has
///   no way to be reached or quit — a lingering headless process would just
///   be a leak. A user who *wants* the long-lived process asks for it
///   explicitly with `--windowless`, which is what the systemd user unit
///   launches.
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
/// **"The most recent" is the platform's front-to-back stack, not slot order.**
/// `App::windows()` enumerates gpui's `SlotMap`, which reuses a closed window's
/// slot — so after a close-and-reopen its `.last()` is whatever landed in the
/// highest slot index, not the window the user saw last. `App::window_stack()`
/// is the real signal (macOS implements it as `MacWindow::ordered_windows()`,
/// front-to-back), and reactivation is a macOS-only event, so this is exactly
/// the platform that answers it. `windows().last()` stays as the deterministic
/// fallback for the platforms that return `None` — including the test platform,
/// which is why the behavior tests still exercise a real path.
///
/// `open_when_empty` is a parameter rather than a direct call so the decision
/// is testable without `AppGlobal` installed.
pub fn reactivate(cx: &mut App, open_when_empty: impl FnOnce(&mut App)) {
    let target = cx
        .active_window()
        .or_else(|| cx.window_stack().and_then(|stack| stack.first().copied()))
        .or_else(|| cx.windows().last().copied());
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

/// Close every open window, leaving the process running.
///
/// The window half of ⌘Q's retire-to-the-background (task 17 wave 3b;
/// `status_item::quit_or_retire` pairs it with the `Accessory` flip). It is
/// deliberately *not* `cx.quit()`: nothing on the quit path runs, so
/// [`install_engine_shutdown`] never fires and the loaded engines survive.
///
/// gpui drops a window as its own `update_window` unwinds (`Window::removed`
/// is read there), so the registry is genuinely empty when this returns — and
/// `QuitMode::Explicit`, which macOS uses, means emptying it quits nothing.
/// The handles are snapshotted first because the removal mutates the very
/// registry `App::windows` reads.
///
/// **Do not call this from inside a window's own update.** A window being
/// updated is still *listed* by `App::windows`, but `update_window` has taken
/// its slot for the duration, so `handle.update` on it fails — and this sweep
/// swallows that (there is nothing sensible to do with it) and moves on. The
/// window quietly skipped is exactly the one that dispatched, since
/// `App::dispatch_action` routes an action through the active window; that is
/// every action handler. The caller defers (`status_item::quit_or_retire`
/// does), which is why the failure is swallowed rather than reported.
pub fn close_all_windows(cx: &mut App) {
    for handle in cx.windows() {
        handle
            .update(cx, |_, window, _| window.remove_window())
            .ok();
    }
}

/// Best-effort teardown of local inference engines when the app quits.
///
/// The *full* shutdown is a full shutdown — engines, everything — and on
/// macOS nothing else delivers it. (⌘Q no longer reaches here: it retires the
/// app and leaves the engines up, which is the point. The full shutdown is
/// the status menu's Quit, or a `SIGTERM` in windowless mode.)
/// `cx.quit()` becomes `[NSApp terminate:]`, which
/// ends in `exit()`, so the tokio runtime is never dropped and the
/// `kill_on_drop` reaping that covers the Linux path (where the event loop
/// simply returns and `main` unwinds) never happens. Without this hook a
/// quit on macOS leaves every loaded `llama-server` running, holding
/// gigabytes, adopted by launchd.
///
/// **It enumerates the live engine registry, not a model snapshot.**
/// `AppCore::shutdown_engines` is synchronous, infallible and touches no
/// filesystem — which is the whole reason it exists rather than a loop over
/// `local_models_state`. That snapshot is *reconstructed by scanning* the
/// managed models directory and every `llamacpp` backend's directory (a
/// `read_dir`, a `stat` and a sidecar read per `.gguf`, plus a DB round trip
/// to list the backends), consulting the engine map only to decorate a file
/// it already found. So an engine whose backing `.gguf` was renamed or
/// deleted mid-session is *absent* from it while its subprocess is very much
/// alive, a `Result` means an unreadable directory could skip the sweep
/// entirely, and a large or slow directory could spend the whole budget on
/// I/O before killing anything. On a quit path none of that is acceptable.
///
/// Signalling is not reaping — the supervisor task owns the child — so the
/// hook yields briefly afterwards to let the runtime run them. The timer is on
/// gpui's *own* executor, not tokio's, which is what lets the tokio threads
/// make that progress while `App::shutdown` blocks the main thread.
///
/// **The grace is unconditional, not gated on how many engines were drained.**
/// `unload_local_model` removes an engine's registry entry and returns as soon
/// as it has *signalled* the supervisor — it does not await the child's exit.
/// So an unload immediately followed by a quit finds an empty registry, and a
/// count-gated grace would skip the yield precisely when a child is still
/// dying, letting AppKit's `exit()` orphan the engine that was one instant
/// from being reaped. 50ms is imperceptible on a quit and well inside the
/// budget; paying it on a no-engine quit is the cheaper mistake by far.
///
/// **The bus bridge is stopped first, before anything else.** During the grace
/// window below gpui keeps driving foreground tasks — including the
/// app-lifetime bridge that dispatches every `Change` into the stores — while
/// `App::shutdown` has already set `quitting`, so *any* dispatch lands in
/// `cx.spawn` and panics ("Can't spawn on main thread after on_app_quit").
/// Silencing the engine drain fixed one emitter and the engine supervisors a
/// second, but a model **download** reporting progress mid-grace was a third,
/// and every future emitter would be a fresh one. Closing the receiver is the
/// cure for the class; see [`stores::BusBridge::quiesce`] for the layering
/// (app-core keeps its own quiet-on-quit rules for the CLI's sake, and neither
/// layer relies on the other).
///
/// Bounded by `gpui::SHUTDOWN_TIMEOUT` (200ms), so this is best-effort by
/// construction and says so: a quit that outruns the budget still exits. A
/// hard guarantee wants the OS to enforce it (`PR_SET_PDEATHSIG` on Linux;
/// macOS has no equivalent and would need an explicit synchronous reap in
/// app-core) — a follow-up, not wave 2.
pub fn install_engine_shutdown(stores: &Stores, bridge: BusBridge, cx: &mut App) {
    let core = stores.app_core();
    cx.on_app_quit(move |cx: &mut App| {
        // Order is load-bearing: close the door before anything can knock.
        bridge.quiesce();
        let timer = cx.background_executor().timer(ENGINE_TEARDOWN_GRACE);
        let had_core = core.is_some();
        if let Some(core) = core.as_ref() {
            core.shutdown_engines();
        }
        async move {
            if had_core {
                timer.await;
            }
        }
    })
    .detach();
}

/// How long the quit hook yields so the runtime can reap the children it just
/// signalled. Comfortably inside `gpui::SHUTDOWN_TIMEOUT` (200ms).
const ENGINE_TEARDOWN_GRACE: std::time::Duration = std::time::Duration::from_millis(50);

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
