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
        /// Close the focused window (chat or settings). Bound to ⌘W on
        /// macOS. Closing the last chat window does not quit the app —
        /// that's ⌘Q.
        CloseWindow,
        /// Quit the application.
        Quit,
        /// Show the About panel.
        About,
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
    ]
);
