use gpui::{Action, actions};

/// Create a new space from a **specific** space template (the Space menu's
/// "New Space from Template ▸" submenu). Data-carrying (the template id), so the
/// submenu — rebuilt from the live template registry on `Change::Templates` —
/// dispatches one per template. `no_json`: dispatched only from
/// programmatically-built menu items (never a keybinding or JSON keymap), so it
/// needs no deserialization.
#[derive(Action, Clone, PartialEq, Eq)]
#[action(namespace = eidola, no_json)]
pub struct NewSpaceFromTemplate {
    pub template_id: String,
}

actions!(
    eidola,
    [
        /// Show the macOS-style Settings window. Bound to ⌘, on macOS.
        ///
        /// The settings window is a singleton: invoking this when it is
        /// already open brings the existing window to the front instead of
        /// creating a second one.
        OpenSettings,
        /// Open a new chat window. Each window owns its own space, so they
        /// are independent conversations sharing the same `Core`. Bound to
        /// ⌘N on macOS.
        NewSpace,
        /// Show the Library window — the table of contents of past spaces.
        /// Bound to ⌘L on macOS. Singleton, like Settings: re-invoking
        /// raises the existing window.
        OpenLibrary,
        /// Show the Record window — the raw local trail of attestations,
        /// requests, and spending. Bound to ⇧⌘L on macOS (sibling of the
        /// Library's ⌘L). Singleton, like Settings and Library.
        OpenRecord,
        /// Close the focused window (chat or settings). Bound to ⌘W on
        /// macOS. Closing the last window does not quit the app, and (since
        /// task 17 wave 3b) does not even retire it — the Dock icon and menu
        /// bar stay, so ⌘N still works.
        CloseWindow,
        /// The app's own ⌘Q. **On macOS with a status item this retires the
        /// app to the background** — every window closes, the Dock indicator
        /// goes, and the process, the status item and the loaded engines keep
        /// running (task 17 wave 3b). Without a status item, and everywhere
        /// off macOS, it is the full shutdown it has always been. The
        /// decision is `status_item::quit_intent`.
        Quit,
        /// End the process — the wave-2 teardown, engines included. Raised
        /// only by the status menu's "Quit Eidola" (which carries its own ⌘Q
        /// key equivalent), because that is the door that means it. Bound to
        /// no keystroke: ⌘Q belongs to [`Quit`].
        QuitApp,
        /// Show the About panel.
        About,
        /// Open the onboarding window — the from-scratch "Get Started" flow
        /// (account creation / linking / adding credit). Lives in the Eidola
        /// menu, and opens automatically at launch when no account is
        /// configured. Singleton, like Settings.
        GetStarted,
        /// **Quote** the current selection inside a post into the active
        /// draft: attach a reference edge and inject its `{{ embed N }}`
        /// marker at the caret, so the passage renders as a quote block while
        /// composing. Lives in the Edit menu; registered per-`SpaceView` and
        /// only while a quotable post selection exists, so the menu item greys
        /// out with nothing selected (the `CloseWindow`/`ToggleInspector`
        /// pattern, with the extra selection condition).
        Quote,
        /// **Quote in Reply** — the same quote, but into a *new* reply draft
        /// on the quoted post, so the answer branches where the passage is.
        /// Same registration and greying as [`Quote`].
        QuoteInReply,
        /// **Quote in Another Conversation…** — the cross-space arm (task 37).
        /// Opens a destination picker over the Library's conversations; the
        /// chosen one is named in a visibility statement the reader confirms,
        /// because quoting *copies* the passage to that audience. Same
        /// registration and greying as [`Quote`].
        QuoteElsewhere,
        /// Show the Updates window (singleton, like Settings) and run a
        /// manual update check. Lives in the Eidola menu directly under
        /// "About Eidola" — the standard macOS placement.
        CheckForUpdates,
        /// Hide the application (macOS App menu standard, ⌘H).
        Hide,
        /// Hide all other applications (macOS App menu standard, ⌥⌘H).
        HideOthers,
        /// Unhide all hidden applications (macOS App menu standard).
        ShowAll,
        /// Minimize the focused window (macOS Window menu standard, ⌘M).
        Minimize,
        /// Zoom the focused window (macOS Window menu standard).
        Zoom,
        /// Show or hide the focused space window's **inspector** — the
        /// per-space settings panel that splits the window (task 26). Space
        /// menu; ⌥⌘I (the Xcode/Finder convention). Registered per-`SpaceView`
        /// like `CloseWindow`, so macOS greys it when no space window is open;
        /// the space itself carries no visual toggle by design.
        ToggleInspector,
        /// Open **Find in Conversation** — the ⌘F bar over the visible
        /// branch. Edit menu; registered per-`SpaceView` like
        /// `ToggleInspector`, so macOS greys it with no space window open.
        /// With a bar already open it re-focuses the query field.
        FindInSpace,
        /// Toggle **gpui's element inspector** — the development overlay, not
        /// the product's. Bound to ⌥⇧⌘I (it gave up ⌥⌘I to the space
        /// inspector above). Requires the `inspector` feature on `gpui`
        /// (enabled in `Cargo.toml`); the rich element/style editor UI comes
        /// from `gpui-component`'s inspector renderer, also feature-gated.
        ToggleElementInspector,
        /// **Post** the composer's draft — the common gesture. The space's
        /// participants decide who responds (notify policies drive one
        /// streaming turn per responder). The composer's ⌘↩ reaches this via
        /// the editor's `PressEnter` event; the action itself stays
        /// dispatchable (the Post affordance, tests, future menu items).
        Send,
        /// Post the composer's draft **quietly** — save without notifying
        /// anyone (⌘⇧↩ / the ⌥-revealed "Post quietly" affordance).
        PostOnly,
        /// Reset the base type scale to Actual Size (1.0). View menu; ⌘0 on
        /// macOS / Ctrl+0 elsewhere.
        ActualSize,
        /// Step the base type scale up one rung. View menu; ⌘+ / Ctrl++.
        ZoomIn,
        /// Step the base type scale down one rung. View menu; ⌘- / Ctrl+-.
        ZoomOut,
    ]
);

/// Platform-aware chord labels for user-visible copy (empty states, hints,
/// menu items). macOS uses the symbol register (⌘N, ⇧⌘L, ⌥); Linux spells
/// the chords out (Ctrl+N, Ctrl+Shift+L, Alt) — the desktop convention.
pub(crate) fn primary_chord(key: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("⌘{key}")
    } else {
        format!("Ctrl+{key}")
    }
}

/// The primary+alt chord label: ⌥⌘I on macOS, Ctrl+Alt+I elsewhere.
pub(crate) fn primary_alt_chord(key: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("⌥⌘{key}")
    } else {
        format!("Ctrl+Alt+{key}")
    }
}

/// The primary+shift chord label: ⇧⌘L on macOS, Ctrl+Shift+L elsewhere.
pub(crate) fn primary_shift_chord(key: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("⇧⌘{key}")
    } else {
        format!("Ctrl+Shift+{key}")
    }
}
