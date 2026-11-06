//! The app's commands. Menus, key bindings and handlers all name these.

use std::path::PathBuf;

use gpui::{Action, actions};

actions!(
    notepad,
    [
        About,
        Hide,
        HideOthers,
        ShowAll,
        Quit,
        NewDocument,
        Open,
        ClearRecentDocuments,
        CloseWindow,
        Save,
        SaveAs,
        RevertToSaved,
        ShowInFinder,
        ActualSize,
        ZoomIn,
        ZoomOut,
        ToggleFullScreen,
        Minimize,
        Zoom,
    ]
);

/// File ▸ Open Recent ▸ <document>. Carries its path, so it is built per
/// entry rather than bound to a key.
#[derive(Action, Clone, PartialEq, Eq)]
#[action(namespace = notepad, no_json)]
pub struct OpenRecent {
    pub path: PathBuf,
}
