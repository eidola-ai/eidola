//! One document window: the editor on Eidola's writing surface, and the
//! document's life on disk — dirty tracking, save, Save As, revert, the
//! unsaved-changes sheet, and noticing another program changed the file.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::SystemTime;

use eidola_gui::probe::Probe;
use eidola_gui::space_view::{BODY_MAX_WIDTH, POST_PAD_Y, TITLE_BAR_RESERVE, layout, prose_style};
use gpui::{
    AppContext, Context, Entity, ExternalPaths, FocusHandle, InteractiveElement, IntoElement,
    MouseButton, ParentElement, PromptButton, PromptLevel, Render, ScrollHandle, SharedString,
    StatefulInteractiveElement, Styled, Subscription, Task, Window, canvas, div, point,
    prelude::FluentBuilder, px,
};
use gpui_component::{ActiveTheme, menu::ContextMenuExt};
use gpui_markdown_editor::{
    Copy, Cut, EditorCommand, MarkdownEditor, MarkdownEditorEvent, MarkdownEditorState, Paste,
    SelectAll,
};

use crate::actions::{CloseWindow, RevertToSaved, Save, SaveAs, ShowInFinder};
use crate::file::{self, Format};

/// Space kept between the caret and the edge of the visible page when an
/// edit or a caret move scrolls it into view.
const CARET_MARGIN: f32 = 24.;
/// How close to the top or bottom of the page a selection drag has to come
/// before the page starts scrolling under it, and the fastest it scrolls.
const DRAG_SCROLL_BAND: f32 = 48.;
const DRAG_SCROLL_MAX_SPEED: f32 = 28.;

/// Where the editor's first line sits in the page: under the title band, one
/// post-pad down — exactly where a docked composer's text begins in Eidola.
fn text_top() -> f32 {
    (TITLE_BAR_RESERVE + POST_PAD_Y).as_f32()
}

/// What a new window opens on.
pub enum Contents {
    Untitled(usize),
    File {
        path: PathBuf,
        text: String,
        format: Format,
    },
}

pub struct DocumentView {
    editor: Entity<MarkdownEditorState>,
    focus: FocusHandle,
    path: Option<PathBuf>,
    /// The ordinal an unsaved document is named by ("Untitled 2").
    untitled: usize,
    format: Format,
    /// The buffer as it stands on disk, so undoing back to it reads clean.
    saved_text: String,
    dirty: bool,
    /// When this process last read or wrote the file.
    disk_stamp: Option<SystemTime>,
    /// The accessible value: the text as it stood when the editor last
    /// settled (opened, saved, or lost focus) — never per keystroke, which
    /// would have a screen reader re-read the whole note on every character.
    settled_text: SharedString,
    scroll: ScrollHandle,
    reveal_caret: Rc<Cell<bool>>,
    /// An unsaved-changes sheet is up, so a second ⌘W does not stack another.
    reviewing: bool,
    /// The external-change check is waiting on the reader.
    asking_about_disk: bool,
    _subscriptions: Vec<Subscription>,
}

impl DocumentView {
    pub fn new(contents: Contents, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let (path, untitled, text, format) = match contents {
            Contents::Untitled(n) => (None, n, String::new(), Format::default()),
            Contents::File { path, text, format } => (Some(path), 0, text, format),
        };
        let editor = cx.new(|cx| {
            let mut state = MarkdownEditorState::new(window, cx).default_value(text.clone());
            state.set_image_base_dir(path.as_deref().and_then(parent_dir), cx);
            state
        });

        let subscriptions = vec![
            cx.subscribe_in(&editor, window, Self::on_editor_event),
            cx.observe_window_activation(window, |this, window, cx| {
                if window.is_window_active() {
                    crate::note_front_bounds(window.bounds(), cx);
                    crate::follow_eidola_appearance(cx);
                    this.check_disk(window, cx);
                }
            }),
            cx.observe_window_bounds(window, |_, window, cx| {
                if window.is_window_active() {
                    crate::note_front_bounds(window.bounds(), cx);
                }
            }),
        ];

        let this = cx.entity().downgrade();
        window.on_window_should_close(cx, move |window, cx| {
            this.update(cx, |this, cx| this.should_close(window, cx))
                .unwrap_or(true)
        });

        let view = Self {
            focus: cx.focus_handle(),
            disk_stamp: path.as_deref().and_then(file::modified),
            path,
            untitled,
            format,
            settled_text: text.clone().into(),
            saved_text: text,
            dirty: false,
            editor,
            scroll: ScrollHandle::new(),
            reveal_caret: Rc::new(Cell::new(false)),
            reviewing: false,
            asking_about_disk: false,
            _subscriptions: subscriptions,
        };
        view.sync_window(window);
        view
    }

    pub fn editor(&self) -> &Entity<MarkdownEditorState> {
        &self.editor
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// An untitled document nobody has typed in — the one a newly opened
    /// file may take the place of, as in every Mac document app.
    pub fn is_pristine(&self, cx: &gpui::App) -> bool {
        self.path.is_none() && !self.dirty && self.editor.read(cx).value().is_empty()
    }

    pub fn name(&self) -> String {
        file::display_name(self.path.as_deref(), self.untitled)
    }

    pub fn focus_editor(&self, window: &mut Window, cx: &mut gpui::App) {
        let handle = self.editor.read(cx).focus_handle.clone();
        window.focus(&handle, cx);
    }

    /// Replace this window's document with a file — the pristine-untitled
    /// case of opening.
    pub fn load(
        &mut self,
        path: PathBuf,
        text: String,
        format: Format,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.disk_stamp = file::modified(&path);
        self.editor.update(cx, |editor, cx| {
            editor.set_image_base_dir(parent_dir(&path), cx);
            editor.set_value(text.clone(), cx);
        });
        self.path = Some(path);
        self.untitled = 0;
        self.format = format;
        self.settled_text = text.clone().into();
        self.saved_text = text;
        self.dirty = false;
        self.scroll.set_offset(point(px(0.), px(0.)));
        self.sync_window(window);
        cx.notify();
    }

    fn on_editor_event(
        &mut self,
        editor: &Entity<MarkdownEditorState>,
        event: &MarkdownEditorEvent,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            MarkdownEditorEvent::Change => {
                self.reveal_caret.set(true);
                let dirty = editor.read(cx).value() != self.saved_text;
                if dirty != self.dirty {
                    self.dirty = dirty;
                    self.sync_window(window);
                }
                cx.notify();
            }
            MarkdownEditorEvent::SelectionChanged => {
                self.reveal_caret.set(true);
                cx.notify();
            }
            MarkdownEditorEvent::Blur => {
                self.settled_text = editor.read(cx).value().to_string().into();
                cx.notify();
            }
            MarkdownEditorEvent::Focus | MarkdownEditorEvent::PressEnter { .. } => {}
        }
    }

    /// Bring the platform window in line with the document: its title (the
    /// Window menu, the window switcher and VoiceOver's window chooser all
    /// read it), the close button's edited dot, and the accessibility
    /// document path.
    fn sync_window(&self, window: &mut Window) {
        window.set_window_title(&self.name());
        window.set_window_edited(self.dirty);
        window.set_document_path(self.path.as_deref());
    }

    // ---------------------------------------------------------------------
    // Saving
    // ---------------------------------------------------------------------

    /// Save the document — to its own path, or (untitled, or `save_as`) to
    /// one the reader picks. Resolves `true` once the bytes are on disk.
    pub fn save(
        &mut self,
        save_as: bool,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Task<bool> {
        let text = self.editor.read(cx).value().to_string();
        let format = self.format;
        let target = self.path.clone().filter(|_| !save_as);
        let directory = self
            .path
            .as_deref()
            .and_then(parent_dir)
            .or_else(dirs::document_dir)
            .or_else(dirs::home_dir)
            .unwrap_or_else(|| PathBuf::from("/"));
        let suggested = match &self.path {
            Some(_) => self.name(),
            None => format!("{}.md", self.name()),
        };

        cx.spawn_in(window, async move |this, cx| {
            let path = match target {
                Some(path) => path,
                None => {
                    let Ok(chosen) =
                        cx.update(|_, cx| cx.prompt_for_new_path(&directory, Some(&suggested)))
                    else {
                        return false;
                    };
                    match chosen.await {
                        Ok(Ok(Some(path))) => file::with_default_extension(path),
                        _ => return false,
                    }
                }
            };
            let bytes = file::encode(&text, format);
            let written = path.clone();
            let result = cx
                .background_executor()
                .spawn(async move { file::write_atomically(&written, &bytes) })
                .await;
            this.update_in(cx, |this, window, cx| match result {
                Ok(()) => {
                    this.did_save(path, text, window, cx);
                    true
                }
                Err(error) => {
                    let message = format!("The document “{}” could not be saved.", this.name());
                    alert(
                        window,
                        cx,
                        PromptLevel::Critical,
                        &message,
                        Some(&error.to_string()),
                    );
                    false
                }
            })
            .unwrap_or(false)
        })
    }

    fn did_save(
        &mut self,
        path: PathBuf,
        text: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.disk_stamp = file::modified(&path);
        if self.path.as_deref() != Some(path.as_path()) {
            self.editor.update(cx, |editor, cx| {
                editor.set_image_base_dir(parent_dir(&path), cx);
            });
        }
        crate::note_recent(&path, cx);
        self.path = Some(path);
        self.untitled = 0;
        self.dirty = self.editor.read(cx).value() != text;
        self.settled_text = text.clone().into();
        self.saved_text = text;
        self.sync_window(window);
        cx.notify();
    }

    fn revert(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.dirty || self.path.is_none() {
            return;
        }
        let message = format!(
            "Do you want to revert to the most recently saved version of “{}”?",
            self.name()
        );
        let answer = window.prompt(
            PromptLevel::Warning,
            &message,
            Some("Your current changes will be lost."),
            &[PromptButton::ok("Revert"), PromptButton::cancel("Cancel")],
            cx,
        );
        cx.spawn_in(window, async move |this, cx| {
            if answer.await == Ok(0) {
                this.update_in(cx, |this, window, cx| this.reload_from_disk(window, cx))
                    .ok();
            }
        })
        .detach();
    }

    /// Read the file again and show it, dropping whatever the buffer held.
    fn reload_from_disk(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(path) = self.path.clone() else {
            return;
        };
        match std::fs::read(&path).map(|bytes| file::decode(&bytes)) {
            Ok(Ok((text, format))) => {
                let offset = self.scroll.offset();
                self.load(path, text, format, window, cx);
                self.scroll.set_offset(offset);
            }
            Ok(Err(file::DecodeError::NotUtf8)) => {
                self.disk_stamp = file::modified(&path);
                alert(
                    window,
                    cx,
                    PromptLevel::Warning,
                    &format!("“{}” is no longer a UTF-8 text file.", self.name()),
                    Some("The version in this window was kept."),
                );
            }
            Err(error) => {
                alert(
                    window,
                    cx,
                    PromptLevel::Warning,
                    &format!("“{}” could not be read.", self.name()),
                    Some(&error.to_string()),
                );
            }
        }
    }

    /// Another program — Obsidian, a sync client, `git` — may have written
    /// the file while it sat open here. Checked when the window comes to the
    /// front: an unedited document just follows the disk; an edited one asks.
    fn check_disk(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.asking_about_disk || self.reviewing {
            return;
        }
        let Some(path) = self.path.clone() else {
            return;
        };
        let Some(stamp) = file::modified(&path) else {
            // Gone (moved, deleted). The window keeps the text, and Save
            // writes it back to where it was.
            return;
        };
        if Some(stamp) == self.disk_stamp {
            return;
        }
        if !self.dirty {
            self.reload_from_disk(window, cx);
            return;
        }
        self.asking_about_disk = true;
        let message = format!("“{}” was changed by another application.", self.name());
        let answer = window.prompt(
            PromptLevel::Warning,
            &message,
            Some("Do you want to keep the version in this window, or reload the one on disk?"),
            &[
                PromptButton::new("Keep This Version"),
                PromptButton::new("Reload"),
            ],
            cx,
        );
        cx.spawn_in(window, async move |this, cx| {
            let reload = answer.await == Ok(1);
            this.update_in(cx, |this, window, cx| {
                this.asking_about_disk = false;
                if reload {
                    this.reload_from_disk(window, cx);
                } else {
                    // Asked once per change on disk, not on every activation.
                    this.disk_stamp = Some(stamp);
                }
            })
            .ok();
        })
        .detach();
    }

    // ---------------------------------------------------------------------
    // Closing
    // ---------------------------------------------------------------------

    /// Resolve the document's unsaved changes before it goes away: `true`
    /// once it is saved or the reader chose to discard, `false` if they
    /// cancelled. A clean document resolves `true` at once.
    pub fn review_unsaved(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Task<bool> {
        if !self.dirty {
            return Task::ready(true);
        }
        if self.reviewing {
            return Task::ready(false);
        }
        self.reviewing = true;
        window.activate_window();
        let message = format!(
            "Do you want to save the changes you made to “{}”?",
            self.name()
        );
        let answer = window.prompt(
            PromptLevel::Warning,
            &message,
            Some("Your changes will be lost if you don’t save them."),
            &[
                PromptButton::ok("Save"),
                PromptButton::new("Don’t Save"),
                PromptButton::cancel("Cancel"),
            ],
            cx,
        );
        cx.spawn_in(window, async move |this, cx| {
            let settled = match answer.await {
                Ok(0) => {
                    match this.update_in(cx, |this, window, cx| this.save(false, window, cx)) {
                        Ok(save) => save.await,
                        Err(_) => false,
                    }
                }
                Ok(1) => true,
                _ => false,
            };
            this.update(cx, |this, _| this.reviewing = false).ok();
            settled
        })
    }

    /// File ▸ Close and ⌘W.
    fn close(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let review = self.review_unsaved(window, cx);
        cx.spawn_in(window, async move |_, cx| {
            if review.await {
                cx.update(|window, _| window.remove_window()).ok();
            }
        })
        .detach();
    }

    /// The window's close button. A clean document closes; an edited one
    /// keeps the window while the sheet asks.
    fn should_close(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        if !self.dirty {
            return true;
        }
        self.close(window, cx);
        false
    }

    // ---------------------------------------------------------------------
    // Rendering
    // ---------------------------------------------------------------------

    /// While a selection drag holds the pointer near the top or bottom of
    /// the page, scroll under it and keep extending the selection — the
    /// editor lays out at full height, so the page is what has to move.
    fn autoscroll_selection(&self, window: &mut Window, cx: &mut Context<Self>) {
        if !self.editor.read(cx).is_selecting() {
            return;
        }
        let bounds = self.scroll.bounds();
        let mouse = window.mouse_position().y.as_f32();
        let top = bounds.top().as_f32() + TITLE_BAR_RESERVE.as_f32();
        let bottom = bounds.bottom().as_f32();
        let speed = if mouse < top + DRAG_SCROLL_BAND {
            -((top + DRAG_SCROLL_BAND - mouse) / DRAG_SCROLL_BAND).min(1.)
        } else if mouse > bottom - DRAG_SCROLL_BAND {
            ((mouse - (bottom - DRAG_SCROLL_BAND)) / DRAG_SCROLL_BAND).min(1.)
        } else {
            return;
        };
        let offset = self.scroll.offset();
        let max = self.scroll.max_offset().y.as_f32();
        let next = (offset.y.as_f32() - speed * DRAG_SCROLL_MAX_SPEED).clamp(-max, 0.);
        if (next - offset.y.as_f32()).abs() < 0.5 {
            return;
        }
        self.scroll.set_offset(point(offset.x, px(next)));
        let position = window.mouse_position();
        self.editor
            .update(cx, |editor, cx| editor.drag_extend_to(position, cx));
        window.request_animation_frame();
    }

    fn render_title_band(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme();
        let bg = theme.background;
        let muted = theme.muted_foreground;
        let mut title = self.name();
        if self.dirty {
            title.push_str(" — Edited");
        }
        let band = eidola_gui::chrome::round_top_client_corners(div(), window)
            .id("notepad-title-band")
            .absolute()
            .top_0()
            .left_0()
            .right_0()
            .h(TITLE_BAR_RESERVE)
            .flex()
            .items_center()
            .justify_center()
            .bg(gpui::linear_gradient(
                180.,
                gpui::linear_color_stop(bg, 0.0),
                gpui::linear_color_stop(bg.opacity(0.0), 1.0),
            ))
            .child(
                div()
                    .max_w(BODY_MAX_WIDTH)
                    .px_4()
                    .truncate()
                    .text_xs()
                    .text_color(muted)
                    .child(title),
            );
        eidola_gui::titlebar::make_draggable(band, "notepad-title-band", window, cx)
    }
}

impl Render for DocumentView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.autoscroll_selection(window, cx);

        let theme = cx.theme();
        let (bg, fg) = (theme.background, theme.foreground);
        let viewport = eidola_gui::chrome::content_size(window);
        let bottom_pad = POST_PAD_Y.as_f32();
        // The editor reaches the bottom of the page however short the note,
        // so a click anywhere below the last line puts the caret at the end.
        let fill = (viewport.height.as_f32() - text_top() - bottom_pad).max(0.);

        let editor = self.editor.clone();
        let menu_editor = self.editor.clone();
        let editor_focus = self.editor.read(cx).focus_handle.clone();
        let reveal = self.reveal_caret.clone();
        let scroll = self.scroll.clone();
        let label: SharedString = self.name().into();

        let body = div()
            .id("notepad-body")
            .w_full()
            .max_w(BODY_MAX_WIDTH)
            .probe_value(
                "notepad/document",
                gpui::Role::TextInput,
                label,
                self.settled_text.clone(),
            )
            .child(
                MarkdownEditor::new(&editor)
                    .style(prose_style(cx))
                    .min_height(px(fill))
                    // Registering the gesture is what has the editor place
                    // the caret under a right-click before the menu opens.
                    .on_context_menu(|_, _, _| {}),
            )
            // Reveal the caret after an edit or a caret move. A later sibling
            // of the editor, so its paint reads this frame's layout.
            .child(
                canvas(
                    |_, _, _| {},
                    move |_, _, window, cx| {
                        if !reveal.get() {
                            return;
                        }
                        let Some((top, bottom)) = editor.read(cx).caret_content_y() else {
                            return;
                        };
                        reveal.set(false);
                        let caret_top = text_top() + top.as_f32();
                        let caret_bottom = text_top() + bottom.as_f32();
                        let offset = scroll.offset();
                        let viewport = scroll.bounds().size.height.as_f32();
                        let max = scroll.max_offset().y.as_f32();
                        let next =
                            caret_offset(caret_top, caret_bottom, viewport, offset.y.as_f32(), max);
                        if (next - offset.y.as_f32()).abs() > 0.5 {
                            scroll.set_offset(point(offset.x, px(next)));
                            window.refresh();
                        }
                    },
                )
                .w_full()
                .h(px(0.)),
            )
            .context_menu(move |menu, _, cx| {
                let editor = menu_editor.read(cx);
                let can = |command| editor.can_perform(command, cx);
                menu.action_context(editor_focus.clone())
                    .when(can(EditorCommand::Cut), |m| m.menu("Cut", Box::new(Cut)))
                    .when(can(EditorCommand::Copy), |m| m.menu("Copy", Box::new(Copy)))
                    .when(can(EditorCommand::Paste), |m| {
                        m.menu("Paste", Box::new(Paste))
                    })
                    .separator()
                    .menu("Select All", Box::new(SelectAll))
            });

        let page = div()
            .id("notepad-page")
            .size_full()
            .overflow_y_scroll()
            .track_scroll(&self.scroll)
            .child(
                div()
                    .w_full()
                    .flex()
                    .justify_center()
                    .pt(px(text_top()))
                    .pb(POST_PAD_Y)
                    .px(layout::COMPACT_PAGE_INSET)
                    .child(body),
            );

        let has_path = self.path.is_some();
        let can_revert = has_path && self.dirty;
        let drop_wash = theme.selection.opacity(0.25);

        eidola_gui::chrome::round_client_corners(div(), window)
            .id("notepad-document")
            .key_context("NotepadDocument")
            .track_focus(&self.focus)
            .relative()
            .size_full()
            .bg(bg)
            .text_color(fg)
            .on_action(cx.listener(|this, _: &Save, window, cx| {
                this.save(false, window, cx).detach();
            }))
            .on_action(cx.listener(|this, _: &SaveAs, window, cx| {
                this.save(true, window, cx).detach();
            }))
            .on_action(cx.listener(|this, _: &CloseWindow, window, cx| {
                this.close(window, cx);
            }))
            .when(can_revert, |el| {
                el.on_action(cx.listener(|this, _: &RevertToSaved, window, cx| {
                    this.revert(window, cx);
                }))
            })
            .when(has_path, |el| {
                el.on_action(cx.listener(|this, _: &ShowInFinder, _, cx| {
                    if let Some(path) = &this.path {
                        cx.reveal_path(path);
                    }
                }))
            })
            // A press in the margins beside the column still means "write
            // here": hand the keyboard back to the editor when nothing else
            // holds it. Something that does — an open context menu — keeps it.
            .on_mouse_down(
                MouseButton::Left,
                cx.listener(|this, _, window, cx| {
                    let idle = window
                        .focused(cx)
                        .is_none_or(|focused| focused == this.focus);
                    if idle {
                        this.focus_editor(window, cx);
                    }
                }),
            )
            .drag_over::<ExternalPaths>(move |style, _, _, _| style.bg(drop_wash))
            .on_drop(cx.listener(|_, paths: &ExternalPaths, _, cx| {
                crate::open_paths(paths.paths().to_vec(), cx);
            }))
            .child(page)
            .child(eidola_gui::scrollbar::vertical(
                "notepad-scrollbar",
                &self.scroll,
                window,
            ))
            .child(self.render_title_band(window, cx))
    }
}

/// The page offset that brings a caret spanning `caret_top..caret_bottom`
/// (page coordinates) into the visible page — below the title band, a margin
/// clear of either edge — moving as little as it can. Offsets are gpui's:
/// zero at the top, negative further down.
fn caret_offset(caret_top: f32, caret_bottom: f32, viewport: f32, current: f32, max: f32) -> f32 {
    let view_top = -current;
    let visible_top = view_top + TITLE_BAR_RESERVE.as_f32();
    let mut next_top = view_top;
    if caret_top - CARET_MARGIN < visible_top {
        next_top = caret_top - CARET_MARGIN - TITLE_BAR_RESERVE.as_f32();
    } else if caret_bottom + CARET_MARGIN > view_top + viewport {
        next_top = caret_bottom + CARET_MARGIN - viewport;
    }
    -next_top.clamp(0., max.max(0.))
}

/// Tell the reader something that needs no answer: a sheet with an OK button.
pub(crate) fn alert(
    window: &mut Window,
    cx: &mut gpui::App,
    level: PromptLevel,
    message: &str,
    detail: Option<&str>,
) {
    // Nothing waits on the answer, so the receiver is dropped; the sheet
    // stays up until the reader dismisses it.
    drop(window.prompt(level, message, detail, &["OK"], cx));
}

fn parent_dir(path: &Path) -> Option<PathBuf> {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_caret_already_in_view_moves_nothing() {
        assert_eq!(caret_offset(300., 320., 800., -100., 2000.), -100.);
    }

    #[test]
    fn a_caret_below_the_page_scrolls_just_far_enough() {
        // Viewport 800 at the top of the page; caret at 900..920.
        assert_eq!(
            caret_offset(900., 920., 800., 0., 2000.),
            -(920. + CARET_MARGIN - 800.)
        );
    }

    #[test]
    fn a_caret_above_the_page_clears_the_title_band() {
        let reserve = TITLE_BAR_RESERVE.as_f32();
        let next = caret_offset(500., 520., 800., -600., 2000.);
        assert_eq!(-next, 500. - CARET_MARGIN - reserve);
    }

    #[test]
    fn the_offset_never_leaves_the_page() {
        assert_eq!(caret_offset(10., 30., 800., -50., 2000.), 0.);
        assert_eq!(caret_offset(5000., 5020., 800., 0., 2000.), -2000.);
    }
}
