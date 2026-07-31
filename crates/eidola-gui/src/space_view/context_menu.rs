//! The post context menu — the pointer route to the verbs that already exist
//! on the keyboard and in the Edit menu (task 28).
//!
//! A right-click inside any of the space's `MarkdownEditor`s opens a small
//! popover at the pointer. What it offers depends on what is under it, and on
//! nothing else:
//!
//! - **A read-only post body** (settled posts, a streaming reply): *Select All*
//!   always, and — only while that editor actually holds a selection — *Copy*,
//!   plus *Quote* / *Quote in Reply* when the selection is a quotable one.
//! - **An editable editor** (the composer, an inline Edit session): *Cut*,
//!   *Copy*, *Paste*, *Select All* — with Cut and Copy offered only while
//!   something is selected. The house rule is that an affordance appears when
//!   it is actionable (the per-post verbs hide mid-stream for the same reason:
//!   "dead verbs would lie"), and a *greyed* row would be the worse option
//!   here — gpui exposes no `aria_disabled` at this pin, so a dimmed item
//!   would read to assistive technology as a live one that does nothing.
//!
//! **Nothing here is a parallel path.** The clipboard verbs run the editor's
//! own commands through the additive `MarkdownEditorState::perform` seam (the
//! same code the ⌘X/⌘C/⌘V/⌘A keymap reaches, minus the responder chain, which
//! a read-only post body isn't on), and the quote verbs call
//! [`SpaceView::quote`] / [`SpaceView::quote_in_reply`] — the very handlers the
//! Edit menu dispatches — so the two surfaces cannot drift.
//!
//! **The quotable selection is refreshed at open time.** The editor places the
//! caret before it reports the gesture (a press outside the selection collapses
//! to it), and `SelectionChanged` is delivered *after* this callback returns —
//! so the menu re-resolves the post's selection synchronously rather than
//! reading a `post_selection` that is one event behind.
//!
//! Dismissal follows the band-menu pattern: click-out, Escape, or a choice.

use gpui::{
    AnyElement, Context, Entity, InteractiveElement, IntoElement, ParentElement, Pixels, Point,
    SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};
use gpui_component::{ActiveTheme, v_flex};
use gpui_markdown_editor::{EditorCommand, MarkdownEditorState};

use crate::probe::Probe as _;

use super::SpaceView;

/// Which editor a context menu was opened over, and therefore which verbs it
/// offers. The `Post` arm carries the tree node id so the quote verbs can
/// resolve the selection back to a persisted generation + block.
#[derive(Clone)]
pub(crate) enum ContextTarget {
    /// A read-only post body. `node_id` is `None` for a streaming reply (there
    /// is no persisted post to quote yet).
    Post { node_id: Option<SharedString> },
    /// An editable editor: the composer, or an inline Edit session.
    Editable,
}

/// The open context menu: where it sits, which editor it acts on, and the two
/// facts (selection present, selection quotable) resolved at open time.
#[derive(Clone)]
pub(crate) struct PostContextMenu {
    pub(crate) position: Point<Pixels>,
    pub(crate) editor: Entity<MarkdownEditorState>,
    pub(crate) target: ContextTarget,
    pub(crate) has_selection: bool,
    pub(crate) quotable: bool,
}

/// One menu row: its label, its probe slug, and what it runs. Every row that
/// is built is live — see the module docs on why nothing is greyed.
struct Item {
    slug: &'static str,
    label: &'static str,
    action: ItemAction,
}

#[derive(Copy, Clone)]
enum ItemAction {
    Command(EditorCommand),
    Quote,
    QuoteInReply,
}

/// The menu's width — wide enough for "Quote in Reply" at the UI size, narrow
/// enough to read as a menu rather than a panel.
const MENU_WIDTH: Pixels = px(184.);

impl SpaceView {
    /// Open the context menu over `editor` at `position`. Called from every
    /// editor's `on_context_menu` prop.
    ///
    /// **Deferred by one turn, and it has to be.** The editor invokes the
    /// callback from *inside* its own `update` (the right-mouse-down handler,
    /// where it has just placed the caret), so resolving the selection here
    /// would re-enter that entity and panic. By the time the deferred body
    /// runs the update has completed and the caret it placed is readable —
    /// which is also what makes the quotable-selection re-resolve honest.
    pub(crate) fn open_context_menu(
        &mut self,
        position: Point<Pixels>,
        editor: Entity<MarkdownEditorState>,
        target: ContextTarget,
        cx: &mut Context<Self>,
    ) {
        let this = cx.entity();
        cx.defer(move |cx: &mut gpui::App| {
            this.update(cx, |this, cx| {
                this.open_context_menu_now(position, editor, target, cx)
            });
        });
    }

    fn open_context_menu_now(
        &mut self,
        position: Point<Pixels>,
        editor: Entity<MarkdownEditorState>,
        target: ContextTarget,
        cx: &mut Context<Self>,
    ) {
        // Any other transient popover yields to it (one open thing at a time).
        self.band_menu = None;
        self.highlight_picker = None;

        let has_selection = !editor.read(cx).selection().is_collapsed();
        // Re-resolve the post's quotable selection *now*: the editor just
        // moved the caret and its `SelectionChanged` has not been delivered
        // yet, so `post_selection` is a frame behind.
        let quotable = match &target {
            ContextTarget::Post {
                node_id: Some(node_id),
            } => {
                self.note_body_selection(node_id, cx);
                self.post_selection.is_some()
            }
            _ => false,
        };

        self.context_menu = Some(PostContextMenu {
            position,
            editor,
            target,
            has_selection,
            quotable,
        });
        cx.notify();
    }

    /// Close the menu if one is open. Returns whether anything closed, so the
    /// composer's Escape handler can consume the key.
    pub(crate) fn close_context_menu(&mut self, cx: &mut Context<Self>) -> bool {
        let was_open = self.context_menu.is_some();
        self.context_menu = None;
        if was_open {
            cx.notify();
        }
        was_open
    }

    /// The rows the open menu offers, in order.
    fn context_menu_items(menu: &PostContextMenu) -> Vec<Item> {
        let mut items = Vec::new();
        match menu.target {
            ContextTarget::Editable => {
                if menu.has_selection {
                    items.push(Item {
                        slug: "cut",
                        label: "Cut",
                        action: ItemAction::Command(EditorCommand::Cut),
                    });
                    items.push(Item {
                        slug: "copy",
                        label: "Copy",
                        action: ItemAction::Command(EditorCommand::Copy),
                    });
                }
                items.push(Item {
                    slug: "paste",
                    label: "Paste",
                    action: ItemAction::Command(EditorCommand::Paste),
                });
                items.push(Item {
                    slug: "select-all",
                    label: "Select All",
                    action: ItemAction::Command(EditorCommand::SelectAll),
                });
            }
            ContextTarget::Post { .. } => {
                // Cut and Paste never appear here: they are not things a
                // read-only body can ever do.
                if menu.has_selection {
                    items.push(Item {
                        slug: "copy",
                        label: "Copy",
                        action: ItemAction::Command(EditorCommand::Copy),
                    });
                }
                if menu.quotable {
                    items.push(Item {
                        slug: "quote",
                        label: "Quote",
                        action: ItemAction::Quote,
                    });
                    items.push(Item {
                        slug: "quote-in-reply",
                        label: "Quote in Reply",
                        action: ItemAction::QuoteInReply,
                    });
                }
                items.push(Item {
                    slug: "select-all",
                    label: "Select All",
                    action: ItemAction::Command(EditorCommand::SelectAll),
                });
            }
        }
        items
    }

    /// Run a chosen row and close the menu.
    fn run_context_item(
        &mut self,
        action: ItemAction,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let Some(menu) = self.context_menu.take() else {
            return;
        };
        match action {
            ItemAction::Command(command) => {
                menu.editor
                    .update(cx, |e, cx| e.perform(command, window, cx));
            }
            ItemAction::Quote => self.quote(&crate::actions::Quote, window, cx),
            ItemAction::QuoteInReply => {
                self.quote_in_reply(&crate::actions::QuoteInReply, window, cx)
            }
        }
        cx.notify();
    }

    /// Test seam: open the read-only menu over post `node_id` without
    /// synthesizing a pointer press (the probe suite renders headlessly).
    #[doc(hidden)]
    pub fn open_context_menu_for_test(&mut self, node_id: &str, cx: &mut Context<Self>) {
        let id = SharedString::from(node_id.to_string());
        let Some(editor) = self.bodies.get(&id).cloned() else {
            return;
        };
        self.open_context_menu_now(
            gpui::point(px(120.), px(160.)),
            editor,
            ContextTarget::Post { node_id: Some(id) },
            cx,
        );
    }

    /// Test seam: the labels the open menu offers, in order (`None` when no
    /// menu is open). The menu builds no dead rows, so this list *is* what the
    /// surface affords.
    #[doc(hidden)]
    pub fn context_menu_items_for_test(&self) -> Option<Vec<String>> {
        self.context_menu.as_ref().map(|menu| {
            Self::context_menu_items(menu)
                .into_iter()
                .map(|i| i.label.to_string())
                .collect()
        })
    }

    /// Test seam: activate the open menu's row with probe slug `slug`, exactly
    /// as a click on it would. Returns whether such a row was offered.
    #[doc(hidden)]
    pub fn activate_context_item_for_test(
        &mut self,
        slug: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> bool {
        let Some(action) = self.context_menu.as_ref().and_then(|menu| {
            Self::context_menu_items(menu)
                .into_iter()
                .find(|i| i.slug == slug)
                .map(|i| i.action)
        }) else {
            return false;
        };
        self.run_context_item(action, window, cx);
        true
    }

    /// The popover itself — the picker's quiet register (a bordered `popover`
    /// card of small rows), anchored at the pointer and clamped into the
    /// window so a right-click near an edge still shows every row.
    pub(crate) fn render_context_menu(
        &self,
        window: &Window,
        cx: &Context<Self>,
    ) -> Option<AnyElement> {
        let menu = self.context_menu.as_ref()?;
        let items = Self::context_menu_items(menu);
        if items.is_empty() {
            return None;
        }
        let theme = cx.theme();

        // Row height + the card's own padding, so the clamp knows the size
        // before layout does.
        let height = px(items.len() as f32 * 24.0 + 8.0);
        let size = crate::chrome::content_size(window);
        let left = menu
            .position
            .x
            .min(size.width - MENU_WIDTH - px(8.))
            .max(px(8.));
        let top = menu
            .position
            .y
            .min(size.height - height - px(8.))
            .max(px(8.));

        let mut col = v_flex()
            .id("space-context-menu")
            .probe(
                "space/context-menu",
                gpui::Role::Menu,
                match menu.target {
                    ContextTarget::Editable => "Edit menu",
                    ContextTarget::Post { .. } => "Post menu",
                },
            )
            .occlude()
            .absolute()
            .left(left)
            .top(top)
            .w(MENU_WIDTH)
            .p_1()
            .gap_0p5()
            .rounded_md()
            .border_1()
            .border_color(theme.border)
            .bg(theme.popover)
            .on_mouse_down_out(cx.listener(|this, _, _, cx| {
                this.close_context_menu(cx);
            }));

        for item in items {
            let action = item.action;
            col = col.child(
                div()
                    .id(SharedString::from(format!(
                        "space-context-menu-{}",
                        item.slug
                    )))
                    .probe(
                        format!("space/context-menu/{}", item.slug),
                        gpui::Role::MenuItem,
                        item.label,
                    )
                    .w_full()
                    .px_1p5()
                    .py_0p5()
                    .rounded_sm()
                    .text_sm()
                    .text_color(theme.popover_foreground)
                    .cursor_pointer()
                    .hover(|s| s.bg(theme.muted))
                    .on_click(cx.listener(move |this, _, window, cx| {
                        this.run_context_item(action, window, cx);
                    }))
                    .child(item.label),
            );
        }
        Some(col.into_any_element())
    }
}
