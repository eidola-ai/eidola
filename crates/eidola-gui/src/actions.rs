use gpui::actions;

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
        /// macOS. Closing the last chat window does not quit the app —
        /// that's ⌘Q.
        CloseWindow,
        /// Quit the application.
        Quit,
        /// Show the About panel.
        About,
        /// Open the onboarding window — the from-scratch "Get Started" flow
        /// (account creation / linking / adding credit). Lives in the Eidola
        /// menu, and opens automatically at launch when no account is
        /// configured. Singleton, like Settings.
        GetStarted,
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
        /// Toggle the gpui element inspector on the focused window. Bound to
        /// ⌘⌥I. Requires the `inspector` feature on `gpui` (enabled in
        /// `Cargo.toml`); the rich element/style editor UI comes from
        /// `gpui-component`'s inspector renderer, also feature-gated.
        ToggleInspector,
        /// Post the composer's draft **and** request a response — the common
        /// gesture. The composer's ⌘↩ reaches this via the editor's
        /// `PressEnter` event; the action itself stays dispatchable (the Ask
        /// affordance, tests, future menu items).
        Send,
        /// Post the composer's draft **without** requesting a response — the
        /// save side of the save-vs-request split (⌘⇧↩ / the ⌥-revealed
        /// Post affordance).
        PostOnly,
        /// Toggle the request panel anchored to the composer's action gutter
        /// (model selection; the home of per-request config). Bound to ⌥⌘M
        /// in the `SpaceView` key context; clicking the model chip is the
        /// pointer path to the same state.
        ToggleModelPicker,
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

/// The primary+shift chord label: ⇧⌘L on macOS, Ctrl+Shift+L elsewhere.
pub(crate) fn primary_shift_chord(key: &str) -> String {
    if cfg!(target_os = "macos") {
        format!("⇧⌘{key}")
    } else {
        format!("Ctrl+Shift+{key}")
    }
}

/// The display name of the Option/Alt modifier.
pub(crate) fn alt_name() -> &'static str {
    if cfg!(target_os = "macos") {
        "⌥"
    } else {
        "Alt"
    }
}
