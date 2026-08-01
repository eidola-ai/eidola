//! The focus model — what is focusable, what the ring looks like, and when it
//! shows. Wave B of the accessibility program (`work/tasks/12`).
//!
//! **One internal focus model, two audiences.** The same focus state feeds the
//! app-drawn ring (keyboard-heavy users with no assistive technology enabled —
//! and, on Linux, the *entire* keyboard story, since there is no Full Keyboard
//! Access equivalent) and the AccessKit tree (VoiceOver / FKA track it, and
//! FKA's focus/press actions route back in). Both come from one annotation:
//! [`crate::probe::Probe::probe`] applies the focus attributes alongside the
//! role and label it already applied.
//!
//! **Focusability is derived from the role**, not declared per call site. That
//! keeps the single-annotation doctrine — a probe stays three arguments — and
//! makes the classification reviewable in one place instead of 166:
//!
//! - [`is_tab_stop`] — the interactive roles. Focusable *and* in the Tab order.
//! - [`is_focusable`] — the above plus [`Role::Article`] (a post): focusable so
//!   the ring and the AccessKit focus report work, but **not** a Tab stop —
//!   posts are the space tree's arrow-key surface, and Tab is the
//!   between-region device (see [`crate::space_view`]'s keyboard map).
//! - everything else — landmarks (`Main` / `Navigation` / `Region` / `List` /
//!   `TabList` / `Group` / `Menu`), static readouts (`Label`), alerts and
//!   headings — is a container or a readout, never a focus target.
//!
//! **An element that delegates its focus is never a focus target.**
//! Role-derivation classifies the *role*, but many of our probes annotate
//! something that is not itself the control: a shrink-wrapped `div` around a
//! `gpui-component` widget (which carries no a11y annotations at our pin), or a
//! row of a roving-focus list whose container owns the keyboard. That
//! distinction is not cosmetic: **gpui's Enter/Space activation invokes only
//! the focused element's own click listeners** (`div.rs` registers the whole
//! keyboard-click block only when that element has some), so a focusable
//! wrapper is a tab stop that can never be activated — it rings, swallows a
//! Tab, and does nothing, with the real control one Tab further on. (VoiceOver
//! is unaffected: `Window::handle_a11y_action`'s `Action::Click` fallback
//! synthesizes a mouse press at the node's centre, which hit-tests through to
//! the widget. So the failure is exactly Tab+Enter, not AT activation.) Those
//! elements use [`crate::probe::Probe::probe_delegating`], and for a wrapper
//! the shape depends on what is inside:
//!
//! - **`Button` / `Checkbox`** track a focus handle with `tab_stop(true)`, draw
//!   their own ring, and own the `on_click`. `probe_delegating` applies no focus
//!   attributes at all and Tab lands on the widget, which activates.
//! - **`Switch`** tracks no focus handle at all, so there is nothing inside to
//!   reach. Those wrappers keep the ordinary `probe` (they *are* the tab stop
//!   and wear the ring) and **hoist the activation** — an `on_click` on the
//!   wrapper. It cannot double-fire on a pointer click, because `Switch`
//!   handles the press in `on_mouse_down` and stops propagation, and gpui
//!   bubbles mouse listeners innermost-first.
//! - **`Input` / `MarkdownEditor`** is the model's one remaining hole, which is
//!   why **`Role::TextInput` is excluded from [`is_focusable`] outright**: a
//!   tab stop on the wrapper would land focus on a div that cannot type, mere
//!   focusability would let a click on its padding steal focus from the field,
//!   and no hoist exists — typing needs the field's own handle. Reaching one by
//!   Tab needs per-site plumbing plus an upstream fix (`Input` applies
//!   `.tab_index(…)` to the element rather than to the tracked handle, where
//!   gpui reads it, so it is not a tab stop today). The composer is the one
//!   that matters and it is covered by design rather than by Tab: a space opens
//!   with it focused, and task 38 routes any printable character to it from
//!   anywhere in the window.
//!
//! **The ring is `:focus-visible`, and gpui owns the condition.** `Window`
//! tracks the last input modality (`KeyDown` → keyboard, `MouseDown` /
//! `MouseMove` → mouse) and `InteractiveElement::focus_visible` applies its
//! style only while the element is focused *and* the last input was a key. So
//! a mouse user never sees a ring, and we never had to invent the rule.
//!
//! **The ring is drawn as a box shadow**, which is what lets one annotation
//! ring 166 differently-shaped elements: a shadow is not part of layout (so
//! nothing shifts when it appears) and gpui paints it with **the element's own
//! corner radii**, so a pill chip rings as a pill and a square row rings
//! square, with no radius argument at any call site. Two shadows stack: the
//! outer one is the accent ring, the inner one is a paper-colored spacer that
//! punches the 1px gap between the element edge and the ring. Drop shadows
//! paint before the element's background, in insertion order, so the spacer
//! covers the inner rim of the ring beneath it.

use std::sync::RwLock;

use gpui::{BoxShadow, Hsla, Pixels, Role, px};

/// Ring thickness. Mike's call (candidate A, the hairline) — present, never
/// glowing; the IDE register rather than the Aqua halo.
pub const RING_WIDTH: Pixels = px(1.5);

/// How far outside the element's own bounds the ring's inner edge sits.
pub const RING_OFFSET: Pixels = px(1.);

/// Ring alpha on day paper.
pub const RING_ALPHA_DAY: f32 = 0.55;

/// Ring alpha on night paper. Luminance contrast on a dark ground needs more
/// signal; night raises the **alpha**, never the thickness — one device, one
/// weight, everywhere.
pub const RING_ALPHA_NIGHT: f32 = 0.70;

/// The two colors the ring is built from, snapshotted out of the theme.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RingColors {
    /// The ring itself — the theme accent (`ring`) at the day/night alpha.
    pub ring: Hsla,
    /// The paper the 1px gap is punched in — the theme background.
    pub paper: Hsla,
}

impl Default for RingColors {
    fn default() -> Self {
        // A neutral mid-grey standing in for an un-installed theme (a bare
        // test `App`). Never seen in production: `theme::install` seeds the
        // real pair before the first window paints.
        Self {
            ring: gpui::hsla(0., 0., 0.5, RING_ALPHA_DAY),
            paper: gpui::hsla(0., 0., 1., 1.),
        }
    }
}

/// The live ring colors. Written by `theme::apply` / `apply_fixed` (the only
/// paths that ever change the palette) and read by `probe_inner`, which has no
/// `App` to ask — a probe is a pure element decoration, and threading a
/// context through 166 call sites to fetch two colors would be the wrong
/// trade. Uncontended `RwLock` read, once per probed element per frame.
static RING: RwLock<Option<RingColors>> = RwLock::new(None);

/// Record the palette's ring colors. Called from `theme::apply*`.
pub fn set_ring_colors(colors: RingColors) {
    *RING.write().unwrap() = Some(colors);
}

/// The current ring colors (the neutral default before a theme is installed).
pub fn ring_colors() -> RingColors {
    RING.read().unwrap().unwrap_or_default()
}

/// The ring, as a box-shadow stack. Outer = the accent ring
/// (`RING_OFFSET + RING_WIDTH` of spread); inner = the paper spacer
/// (`RING_OFFSET` of spread) painted over it, leaving a 1px gap between the
/// element's edge and the ring.
pub fn ring_shadows(colors: RingColors) -> Vec<BoxShadow> {
    ring_shadows_at(colors, RING_OFFSET)
}

/// [`ring_shadows`] with the gap widened. One caller: a post's reading column
/// (see [`crate::space_view::post`]), which the ring frames rather than hugs —
/// prose has no chrome of its own to sit against, so a 1px gap reads as an
/// underline on the first line rather than as a ring.
pub fn ring_shadows_at(colors: RingColors, offset: Pixels) -> Vec<BoxShadow> {
    vec![
        BoxShadow::new(px(0.), px(0.), colors.ring).spread_radius(offset + RING_WIDTH),
        BoxShadow::new(px(0.), px(0.), colors.paper).spread_radius(offset),
    ]
}

/// How far outside a **post's reading column** its ring sits. A post row is
/// full-bleed — its own bounds run to both window edges, where a surrounding
/// ring degenerates into two horizontal rules with its sides off-screen — so
/// the space view rings the reading column instead, and gives it room to read
/// as a frame around the prose.
pub const POST_RING_OFFSET: Pixels = px(6.);

/// Whether a role names an element that takes part in Tab navigation.
///
/// Interactive affordances only. See the module docs for why `TextInput` is
/// absent and why `Article` is focusable without being a stop.
pub fn is_tab_stop(role: Role) -> bool {
    matches!(
        role,
        Role::Button
            | Role::DefaultButton
            | Role::Link
            | Role::Tab
            | Role::CheckBox
            | Role::RadioButton
            | Role::Switch
            | Role::ListItem
            | Role::ListBoxOption
            | Role::MenuItem
            | Role::MenuItemCheckBox
            | Role::MenuItemRadio
            | Role::Slider
            | Role::DisclosureTriangle
    )
}

/// Whether a role names an element that can hold focus at all — every tab
/// stop, plus the post article, which the space tree's arrow keys focus
/// directly.
///
/// A post is focusable but **does not wear the automatic ring**: the row is
/// full-bleed, so a ring on its own bounds is two window-wide rules. The space
/// view draws the post's ring around its reading column instead
/// ([`POST_RING_OFFSET`]), which is also why it needs no `:focus-visible`
/// guard — tree focus arrives only from the keyboard, by construction.
pub fn is_focusable(role: Role) -> bool {
    is_tab_stop(role) || matches!(role, Role::Article)
}

/// Tab-order region indices. Tab order is **explicitly grouped per region**
/// (Wave A's landmarks are the groups) rather than following paint order:
/// each landmark container declares [`TabRegion::tab_region`], which gives
/// gpui's tab-stop tree a two-level path — region first, then paint order
/// within the region. Renumbering a region is one constant here; adding an
/// affordance inside one never renumbers anything.
///
/// **Why the numbers are negative.** gpui orders tab stops by path, and a
/// shorter path always sorts before a longer one that shares its prefix — so
/// an *ungrouped* affordance (path `[0]`) precedes every member of a group at
/// index `0` or above (path `[0, …]`), whatever its paint order. Making the
/// content regions negative therefore puts them *ahead* of the ungrouped
/// floating chrome, which is what the reading order wants: in a space window
/// the conversation comes before the composer, the notices and the pickers,
/// and the minimap — a table of contents, not a way in — comes last.
///
/// So the full order in any window is: `NAV`, `MAIN`, the window's ungrouped
/// floating chrome, `AUX`. The numbers are sparse so a region can be inserted
/// between two without touching the rest.
pub mod region {
    /// A window's primary navigation — the Settings nav band, the Record
    /// section strip, the Backends tab strip.
    pub const NAV: isize = -20;
    /// The window's main content.
    pub const MAIN: isize = -10;
    /// Secondary navigation that follows the content — the space window's
    /// minimap.
    pub const AUX: isize = 10;
}

/// Declare an element a **tab-order region**: its own place in the window's
/// order, with its children numbered from zero inside it.
///
/// `tab_stop(false)` is the point of the helper — [`gpui::InteractiveElement::tab_index`]
/// makes an element a stop as a side effect, and a landmark container is not
/// somewhere Tab should ever land.
pub trait TabRegion: gpui::InteractiveElement + Sized {
    /// See [`region`] for the indices and the ordering rule.
    fn tab_region(self, index: isize) -> Self {
        self.tab_index(index).tab_group().tab_stop(false)
    }
}

impl<T: gpui::InteractiveElement + Sized> TabRegion for T {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interactive_roles_are_tab_stops_and_containers_are_not() {
        for role in [
            Role::Button,
            Role::Link,
            Role::Tab,
            Role::CheckBox,
            Role::ListItem,
            Role::ListBoxOption,
            Role::MenuItem,
            Role::Slider,
        ] {
            assert!(is_tab_stop(role), "{role:?} should be a tab stop");
            assert!(is_focusable(role));
        }
        for role in [
            Role::Main,
            Role::Navigation,
            Role::Region,
            Role::List,
            Role::ListBox,
            Role::TabList,
            Role::Group,
            Role::Menu,
            Role::Label,
            Role::Alert,
            Role::Heading,
        ] {
            assert!(!is_tab_stop(role), "{role:?} should not be a tab stop");
            assert!(!is_focusable(role), "{role:?} should not be focusable");
        }
    }

    #[test]
    fn a_post_is_focusable_but_not_a_tab_stop() {
        // Posts are the arrow-key surface; Tab moves between regions.
        assert!(is_focusable(Role::Article));
        assert!(!is_tab_stop(Role::Article));
    }

    #[test]
    fn a_text_input_wrapper_is_neither() {
        // The field inside owns its focus — see the module docs.
        assert!(!is_focusable(Role::TextInput));
        assert!(!is_tab_stop(Role::TextInput));
    }

    #[test]
    fn the_region_order_puts_ungrouped_chrome_between_main_and_aux() {
        // The subtle, breakable half of the tab-order design: gpui sorts a
        // shorter tab path before a longer one sharing its prefix, so an
        // *ungrouped* affordance (path `[0]`) precedes every member of a group
        // at index >= 0. Content regions must therefore be negative, or the
        // space window's floating composer would tab before its conversation.
        const { assert!(region::NAV < region::MAIN, "navigation leads") };
        const { assert!(region::MAIN < 0, "main precedes ungrouped chrome") };
        const { assert!(region::AUX > 0, "auxiliary navigation follows it") };
    }

    #[test]
    fn the_ring_is_a_gap_then_a_hairline() {
        let colors = RingColors::default();
        let shadows = ring_shadows(colors);
        assert_eq!(shadows.len(), 2);
        // Outer first (painted behind), spacer second (painted over it).
        assert_eq!(shadows[0].spread_radius, RING_OFFSET + RING_WIDTH);
        assert_eq!(shadows[0].color, colors.ring);
        assert_eq!(shadows[1].spread_radius, RING_OFFSET);
        assert_eq!(shadows[1].color, colors.paper);
        // Never blurred, never offset — a ring, not a shadow.
        assert!(shadows.iter().all(|s| s.blur_radius == px(0.)));
        assert!(shadows.iter().all(|s| !s.inset));
    }
}
