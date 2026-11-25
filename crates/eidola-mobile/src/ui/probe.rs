//! Probes — one annotation for assistive technology and for UI tests, as in
//! the desktop app (`crates/eidola-gui/src/probe.rs`).
//!
//! `el.id(..).probe(name, role, label)` sets the element's role and accessible
//! label (its accessible name is the localized `label`; the probe `name` is a
//! stable slash-scoped selector that never localizes). **gpui's web backend
//! builds no accessibility tree yet**, so on the phone these attributes reach
//! no screen reader today; they are written now so the tree is complete the
//! day the backend exposes one.
//!
//! In a development bundle (feature `dev`) each probe also records the bounds
//! it painted at, and [`flush`] mirrors the frame's probes into the page as
//! invisible, labelled elements that let touches through
//! (`js/probes.js`): the UI tests find them by name in the webview's own
//! accessibility tree (XCUITest, UIAutomator) and tap their centres, which
//! land on the canvas beneath. A release bundle carries none of it.

#[cfg(feature = "dev")]
use gpui::{ParentElement, Styled};
use gpui::{Role, SharedString, StatefulInteractiveElement};

/// A probed element's role and label. Bounded on
/// [`StatefulInteractiveElement`], so an element must carry an id first.
#[cfg(not(feature = "dev"))]
pub trait Probe: StatefulInteractiveElement + Sized {
    fn probe(
        self,
        _name: impl Into<SharedString>,
        role: Role,
        label: impl Into<SharedString>,
    ) -> Self {
        self.role(role).aria_label(label.into())
    }
}

#[cfg(not(feature = "dev"))]
impl<T: StatefulInteractiveElement + Sized> Probe for T {}

#[cfg(feature = "dev")]
pub trait Probe: StatefulInteractiveElement + ParentElement + Sized {
    fn probe(
        self,
        name: impl Into<SharedString>,
        role: Role,
        label: impl Into<SharedString>,
    ) -> Self {
        let label = label.into();
        let name = name.into();
        let role_name = format!("{role:?}");
        self.role(role).aria_label(label.clone()).child(
            gpui::canvas(
                move |bounds, _, _| {
                    mirror::record(name.clone(), role_name.clone(), label.clone(), bounds)
                },
                |_, _, _, _| {},
            )
            .absolute()
            .top_0()
            .left_0()
            .size_full(),
        )
    }
}

#[cfg(feature = "dev")]
impl<T: StatefulInteractiveElement + ParentElement + Sized> Probe for T {}

/// Start a frame's record (the root view's render).
pub fn begin_frame() {
    #[cfg(feature = "dev")]
    mirror::begin_frame();
}

/// Mirror the last frame's probes into the page (a development bundle only).
pub fn flush() {
    #[cfg(feature = "dev")]
    mirror::flush();
}

#[cfg(feature = "dev")]
pub mod mirror {
    use std::cell::RefCell;

    use gpui::{Bounds, Pixels, SharedString};

    #[derive(Clone, Debug, serde::Serialize)]
    pub struct Entry {
        pub name: String,
        pub role: String,
        pub label: String,
        pub x: f32,
        pub y: f32,
        pub w: f32,
        pub h: f32,
    }

    thread_local! {
        static FRAME: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
        static LAST: RefCell<String> = const { RefCell::new(String::new()) };
    }

    pub fn begin_frame() {
        FRAME.with(|f| f.borrow_mut().clear());
    }

    pub fn record(name: SharedString, role: String, label: SharedString, bounds: Bounds<Pixels>) {
        FRAME.with(|f| {
            let mut f = f.borrow_mut();
            f.retain(|e| e.name != name.as_ref());
            f.push(Entry {
                name: name.to_string(),
                role,
                label: label.to_string(),
                x: f32::from(bounds.origin.x),
                y: f32::from(bounds.origin.y),
                w: f32::from(bounds.size.width),
                h: f32::from(bounds.size.height),
            });
        });
    }

    /// This frame's entries (also what the native tests read).
    pub fn entries() -> Vec<Entry> {
        FRAME.with(|f| f.borrow().clone())
    }

    pub fn flush() {
        let json = serde_json::to_string(&entries()).unwrap_or_default();
        let changed = LAST.with(|last| {
            let mut last = last.borrow_mut();
            if *last == json {
                false
            } else {
                *last = json.clone();
                true
            }
        });
        if changed {
            #[cfg(target_family = "wasm")]
            crate::web::sync_probes(&json);
        }
    }
}
