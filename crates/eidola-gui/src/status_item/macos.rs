//! The AppKit half of the status item — `NSStatusItem` + `NSMenu` through
//! `objc2`, no gpui fork.
//!
//! gpui has no status-item API at our pin and does not need one: a status
//! menu is an AppKit menu, not a gpui window, so it is built directly against
//! the framework. (gpui's own platform layer uses the older `cocoa`/`objc`
//! crates; both talk to the same Objective-C runtime, so the two coexist
//! without conflict.)
//!
//! ## What re-enters gpui, and what does not
//!
//! A menu **command** re-enters gpui synchronously through an [`AsyncApp`] —
//! exactly what gpui does from its own `handleGPUIMenuItem:` callback, and
//! safe for the same reason: AppKit invokes it from the run loop after the
//! menu has closed, never from inside a gpui `App::update`.
//!
//! `menuNeedsUpdate:` deliberately does **not**. It runs while AppKit is
//! opening the menu, and re-entering gpui there would put a `borrow_mut` of
//! the app cell inside an AppKit callback whose nesting we do not control.
//! Instead the rows are mirrored into an `Rc<RefCell<…>>` by a gpui observer
//! on the `LocalModelsStore` — refreshed by the store's own notify, so it is
//! the store's truth and not a snapshot taken at install time — and
//! `menuNeedsUpdate:` only materialises `NSMenuItem`s from it. Rebuilding at
//! open rather than on every notify also keeps a download's ~2.5 Hz progress
//! stream out of AppKit entirely.

use std::cell::RefCell;
use std::rc::Rc;

use gpui::{App, AsyncApp};
use objc2::rc::Retained;
use objc2::runtime::{ProtocolObject, Sel};
use objc2::{DefinedClass, MainThreadMarker, MainThreadOnly, define_class, sel};
use objc2_app_kit::{
    NSApplication, NSApplicationActivationPolicy, NSImage, NSMenu, NSMenuDelegate, NSMenuItem,
    NSStatusBar, NSStatusItem, NSVariableStatusItemLength,
};
use objc2_foundation::{NSObject, NSObjectProtocol, NSString};

use super::{ActivationPolicy, StatusCommand, StatusRow, menu_rows, policy_for};
use crate::lifecycle::LaunchOptions;
use crate::stores::Stores;

/// The status item's menu-bar glyph. A lattice of cells — the enclave, and
/// quiet enough to sit in a menu bar all day. Falls back to a text title if
/// the symbol is unavailable (it is macOS 11+; the fallback is what keeps
/// this from silently rendering an invisible item).
const GLYPH_SYMBOL: &str = "circle.hexagongrid";
const GLYPH_FALLBACK: &str = "Eidola";

/// The rows the next `menuNeedsUpdate:` will build, kept current by the
/// store observer.
type RowMirror = Rc<RefCell<Vec<StatusRow>>>;

struct Ivars {
    app: AsyncApp,
    rows: RowMirror,
}

define_class!(
    // SAFETY:
    // - `NSObject` imposes no subclassing requirements.
    // - The class does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[thread_kind = MainThreadOnly]
    #[name = "EidolaStatusMenuTarget"]
    #[ivars = Ivars]
    struct Target;

    impl Target {
        /// Every command item's action. The command travels as the item's
        /// tag rather than in a captured closure, so the menu can be rebuilt
        /// freely without re-registering anything.
        #[unsafe(method(eidolaStatusMenuCommand:))]
        fn command(&self, sender: &NSMenuItem) {
            let Some(command) = StatusCommand::from_tag(sender.tag()) else {
                return;
            };
            // `AsyncApp::update` panics if the app has been released or is
            // already borrowed. Neither is reachable here: the app owns the
            // status item (so it outlives every click), and AppKit invokes
            // this from the run loop once the menu has closed, never from
            // inside a gpui `App::update`. This is the same synchronous
            // re-entry gpui performs from its own `handleGPUIMenuItem:`.
            self.ivars().app.update(|cx| dispatch(command, cx));
        }
    }

    unsafe impl NSObjectProtocol for Target {}

    unsafe impl NSMenuDelegate for Target {
        #[unsafe(method(menuNeedsUpdate:))]
        fn menu_needs_update(&self, menu: &NSMenu) {
            // SAFETY: the delegate is main-thread-only, so AppKit only ever
            // calls this on the main thread.
            let mtm = unsafe { MainThreadMarker::new_unchecked() };
            rebuild(menu, &self.ivars().rows.borrow(), self, mtm);
        }
    }
);

impl Target {
    fn new(app: AsyncApp, rows: RowMirror, mtm: MainThreadMarker) -> Retained<Self> {
        let this = Self::alloc(mtm).set_ivars(Ivars { app, rows });
        unsafe { objc2::msg_send![super(this), init] }
    }
}

/// Everything the status item owns, held for the life of the process.
///
/// `NSMenu` holds its delegate **weakly**, so the `Target` has to be retained
/// here or the menu would stop updating (and the command action would message
/// a freed object) the moment the local went out of scope.
struct StatusItemGlobal {
    _item: Retained<NSStatusItem>,
    _target: Retained<Target>,
    /// The last policy actually handed to AppKit, so a repeated open/close
    /// does not re-assert it (each `setActivationPolicy:` is a real state
    /// change to the window server).
    applied: ActivationPolicy,
}

impl gpui::Global for StatusItemGlobal {}

pub fn install(stores: &Stores, opts: LaunchOptions, cx: &mut App) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let rows: RowMirror = Rc::new(RefCell::new(menu_rows(
        stores.local_models.read(cx).state().value(),
    )));
    let Some((item, target)) = build(rows.clone(), cx, mtm) else {
        // No door was built, so none is claimed: without the global,
        // `policy_for` sees `status_item_present = false` and the app stays
        // `Regular` for the session — the wave-2 shape, which is a perfectly
        // good app. This early return is the whole reason the flip's promise
        // holds by construction rather than by AppKit trivia.
        return;
    };

    cx.set_global(StatusItemGlobal {
        _item: item,
        _target: target,
        // Not the real policy yet — `apply` below is what asserts it, and it
        // compares against this to decide whether to speak to AppKit at all.
        applied: ActivationPolicy::Regular,
    });

    // Keep the mirror current. App-lifetime observation, the same sanctioned
    // detach as the template-driven menu rebuild in `lib.rs`.
    cx.observe(&stores.local_models, move |store, cx| {
        *rows.borrow_mut() = menu_rows(store.read(cx).state().value());
    })
    .detach();

    // The other half of the flip: gpui removes the window from its registry
    // *before* firing this, so the count is already the post-close one.
    cx.on_window_closed(|cx, _| window_did_close(cx)).detach();

    // A windowed launch is about to open its first window — declaring that
    // here rather than settling at zero avoids a visible Dock-icon flicker
    // (the onboarding branch opens its window a spawned read later, which is
    // long enough for an Accessory→Regular round trip to be seen).
    apply(policy_for(usize::from(!opts.windowless), true), cx);
}

/// Build the status item and its menu, or hand back `None` if what came out
/// would not be a door the user can find.
///
/// **A status item nobody can see is worse than no status item**, because the
/// Accessory flip is gated on this returning `Some`. Two checks, both real:
///
/// - **`button` is `None`** when the item was made with the deprecated custom
///   `view` property. We never call `setView:`, so this is not reachable on
///   today's path (measured: `button=true` on a live launch) — it is here so
///   that a later change which *does* reach it degrades to Regular instead of
///   silently making the app invisible.
/// - **`isVisible` is false** when the user has previously hidden the item.
///   Visibility is persisted under the autosave name, which is exactly what
///   makes this live rather than dead: today `behavior` stays at its default
///   (measured: `0`, i.e. no `RemovalAllowed`, so nothing can set it false),
///   but the day removal is allowed, a removed item comes back hidden on the
///   next launch and must not be counted as a door.
///
/// **The residual, accepted and undetectable:** a menu bar with no room does
/// not draw the item, and AppKit reports nothing — `isVisible` stays true
/// (measured, at creation and three seconds later). It is not permanent (the
/// item reappears as other items go away) and it does not strand the app:
/// Spotlight / `open -a Eidola` / Finder reach the running process and
/// `lifecycle::reactivate` opens a window, which flips it back to Regular
/// (measured: an Accessory, window-less instance went to `Foreground` on
/// `open -a`, same pid). So the failure mode is "hard to see", not
/// "unquittable".
fn build(
    rows: RowMirror,
    cx: &mut App,
    mtm: MainThreadMarker,
) -> Option<(Retained<NSStatusItem>, Retained<Target>)> {
    let target = Target::new(cx.to_async(), rows.clone(), mtm);

    let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str("Eidola"));
    // Deterministic enablement: the info rows are labels and must stay grey,
    // the commands must stay live even with no window key (AppKit's
    // auto-enabling walks the responder chain and would disable both).
    menu.setAutoenablesItems(false);
    menu.setDelegate(Some(ProtocolObject::from_ref(&*target)));
    rebuild(&menu, &rows.borrow(), &target, mtm);

    let bar = NSStatusBar::systemStatusBar();
    let item = bar.statusItemWithLength(NSVariableStatusItemLength);
    // Without an autosave name macOS forgets where the user dragged the item.
    item.setAutosaveName(Some(&NSString::from_str("EidolaStatusItem")));

    let Some(button) = item.button(mtm) else {
        bar.removeStatusItem(&item);
        return None;
    };
    match NSImage::imageWithSystemSymbolName_accessibilityDescription(
        &NSString::from_str(GLYPH_SYMBOL),
        Some(&NSString::from_str("Eidola")),
    ) {
        Some(image) => {
            // Template images take the menu bar's own tint, light or dark.
            image.setTemplate(true);
            button.setImage(Some(&image));
        }
        None => button.setTitle(&NSString::from_str(GLYPH_FALLBACK)),
    }
    item.setMenu(Some(&menu));

    if !item.isVisible() {
        bar.removeStatusItem(&item);
        return None;
    }
    Some((item, target))
}

pub fn window_will_open(cx: &mut App) {
    if !cx.has_global::<StatusItemGlobal>() {
        return;
    }
    // The window being opened is not in `cx.windows()` yet; count it.
    apply(policy_for(cx.windows().len() + 1, true), cx);
}

pub fn window_did_close(cx: &mut App) {
    if !cx.has_global::<StatusItemGlobal>() {
        return;
    }
    apply(policy_for(cx.windows().len(), true), cx);
}

/// Hand a policy to AppKit, if it isn't already the one in force.
///
/// **`Accessory → Regular` needs the activate.** AppKit brings the menu bar
/// up for the newly-regular app only once it is frontmost, and a window
/// opened from a background Accessory process does not get there on its own —
/// leaving a window on screen under some other app's menu bar.
fn apply(policy: ActivationPolicy, cx: &mut App) {
    let Some(global) = cx.try_global::<StatusItemGlobal>() else {
        return;
    };
    if global.applied == policy {
        return;
    }
    let was = global.applied;
    cx.global_mut::<StatusItemGlobal>().applied = policy;

    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    NSApplication::sharedApplication(mtm).setActivationPolicy(match policy {
        ActivationPolicy::Regular => NSApplicationActivationPolicy::Regular,
        ActivationPolicy::Accessory => NSApplicationActivationPolicy::Accessory,
    });
    if was == ActivationPolicy::Accessory && policy == ActivationPolicy::Regular {
        cx.activate(true);
    }
}

/// Route a status-menu command onto the path that already exists for it —
/// the menu is a second door, never a second implementation. Quit in
/// particular dispatches the very action ⌘Q does, so wave 2's engine
/// teardown runs.
fn dispatch(command: StatusCommand, cx: &mut App) {
    match command {
        StatusCommand::Open => crate::reactivate_app(cx),
        StatusCommand::NewSpace => cx.dispatch_action(&crate::actions::NewSpace),
        StatusCommand::Quit => cx.dispatch_action(&crate::actions::Quit),
    }
}

/// Materialise the rows as `NSMenuItem`s, replacing whatever was there.
fn rebuild(menu: &NSMenu, rows: &[StatusRow], target: &Target, mtm: MainThreadMarker) {
    menu.removeAllItems();
    for row in rows {
        let item = match row {
            StatusRow::Separator => NSMenuItem::separatorItem(mtm),
            StatusRow::Info(text) => {
                let item = new_item(&NSString::from_str(text), None, mtm);
                item.setEnabled(false);
                item
            }
            StatusRow::Command(command) => {
                let item = new_item(
                    &NSString::from_str(command.title()),
                    Some(sel!(eidolaStatusMenuCommand:)),
                    mtm,
                );
                item.setTag(command.tag());
                // SAFETY: the target outlives the menu (both are owned by
                // `StatusItemGlobal`, which lives for the process).
                unsafe { item.setTarget(Some(target)) };
                item.setEnabled(true);
                item
            }
        };
        menu.addItem(&item);
    }
}

fn new_item(title: &NSString, action: Option<Sel>, mtm: MainThreadMarker) -> Retained<NSMenuItem> {
    // SAFETY: a plain designated initializer; the selector (when given) is
    // implemented by `Target`, which is the only object we ever target.
    unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            title,
            action,
            &NSString::from_str(""),
        )
    }
}
