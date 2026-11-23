//! Where a screen can send the reader, and how the system's back gesture
//! reaches the app.
//!
//! The screens form one stack over the Library (or over the first run, when
//! it opens at launch): a screen asks to go [`NavEvent::Back`], to another
//! screen ([`NavEvent::To`]), or [`NavEvent::Home`] to the Library alone.
//!
//! **Android's back gesture** goes back in the page's session history, which
//! the shell hands to the webview when it has somewhere to go back to
//! (`mobile/shell-contract.md` → Screen and keyboard). So each screen pushed
//! on the stack pushes one history entry (`js/nav.js`), and the gesture's
//! `popstate` pops it here; a back the app takes itself (a chevron) pops its
//! history entry quietly. iOS has no system back gesture in a webview; the
//! chevron is the way back there.

/// A screen beyond the Library.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dest {
    Settings,
    Account,
    Wallet,
    Purchase,
    Appearance,
    Updates,
    About,
    Licences,
    /// The first run's account steps, from Settings: create, link, or not.
    SetUpAccount,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NavEvent {
    Back,
    To(Dest),
    /// The first run is done: the Library, alone.
    Home,
}

thread_local! {
    static BACK: std::cell::RefCell<(
        tokio::sync::mpsc::UnboundedSender<()>,
        Option<tokio::sync::mpsc::UnboundedReceiver<()>>,
    )> = {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        std::cell::RefCell::new((tx, Some(rx)))
    };
}

/// The system's back gesture, as the app's one listener receives it (taken
/// once, by the window's root view).
pub fn system_back_events() -> Option<tokio::sync::mpsc::UnboundedReceiver<()>> {
    BACK.with(|b| b.borrow_mut().1.take())
}

/// The system's back gesture arrived (`popstate` for an entry this app
/// pushed).
pub fn system_back() {
    BACK.with(|b| {
        let _ = b.borrow().0.send(());
    });
}

thread_local! {
    static FOREGROUND: std::cell::RefCell<(
        tokio::sync::mpsc::UnboundedSender<()>,
        Option<tokio::sync::mpsc::UnboundedReceiver<()>>,
    )> = {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        std::cell::RefCell::new((tx, Some(rx)))
    };
}

/// The page came back to the foreground (`visibilitychange` to visible:
/// the reader returned from another app, the browser after a checkout among
/// them), as the app's one listener receives it.
pub fn foreground_events() -> Option<tokio::sync::mpsc::UnboundedReceiver<()>> {
    FOREGROUND.with(|f| f.borrow_mut().1.take())
}

/// The page became visible again.
pub fn came_to_foreground() {
    FOREGROUND.with(|f| {
        let _ = f.borrow().0.send(());
    });
}

/// The page's session history, mirroring the stack (wasm32 only; natively
/// nothing has a back gesture).
pub mod history {
    /// A screen was pushed.
    pub fn pushed() {
        #[cfg(target_family = "wasm")]
        crate::web::nav_push();
    }

    /// The app popped a screen itself.
    pub fn popped() {
        #[cfg(target_family = "wasm")]
        crate::web::nav_popped();
    }

    /// The stack went back to its root.
    pub fn reset() {
        #[cfg(target_family = "wasm")]
        crate::web::nav_reset();
    }
}
