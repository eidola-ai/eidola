use gpui::actions;

actions!(
    eidola,
    [
        /// Show the macOS-style Settings window. Bound to ⌘, on macOS.
        OpenSettings,
        /// Quit the application.
        Quit,
        /// Show the About panel.
        About,
    ]
);
