//! A conversation on the phone: the transcript as one column of posts, the
//! notices, and the composer.
//!
//! **One branch at a time.** A space is a tree (replies branch); the phone
//! shows the path from the first post to the newest answer, and where a post
//! has several replies, the reply shown carries a small "‹ 1 of 2 ›" switch.
//! Writing continues the branch on screen, and **Reply** under any post starts
//! a branch from it.
//!
//! **Posts render as the desktop app renders them**: each body is the
//! markdown editor, read-only, in the shared prose style; a streaming answer
//! is the same editor growing. The transcript scrolls with momentum and
//! **follows the tail** while an answer streams, for a reader parked at the
//! end; a reader who scrolled up is left where they are until they come back
//! down.

use std::collections::{HashMap, HashSet};

use gpui::{
    AnyElement, AppContext as _, Context, Entity, EventEmitter, FollowMode,
    InteractiveElement as _, IntoElement, ListAlignment, ListState, MouseButton,
    ParentElement as _, Render, Role, SharedString, StatefulInteractiveElement as _, Styled as _,
    Subscription, Window, div, list, prelude::FluentBuilder as _, px,
};
use gpui_component::{ActiveTheme as _, Icon};
use gpui_kit_assets::IconName;
use gpui_markdown_editor::{MarkdownEditor, MarkdownEditorState};

use super::branch::{self, Row, RowKey};
use super::composer::{Composer, ComposerEvent, ReplyTarget};
use super::conversation::{Author, Conversation, ConversationEvent, Post};
use super::errors::error_copy;
use super::i18n::msg;
use super::loadable::Loadable;
use super::probe::Probe as _;
use super::stores::Stores;
use super::theme;
use super::{GUTTER, HEADER_HEIGHT, TOUCH, fmt_when};

#[derive(Clone, Debug)]
pub enum ConversationViewEvent {
    Back,
}

pub struct ConversationView {
    stores: Stores,
    conversation: Entity<Conversation>,
    composer: Entity<Composer>,
    list: ListState,
    rows: Vec<Row>,
    /// A post body editor per row, with the text it was last given.
    bodies: HashMap<RowKey, (Entity<MarkdownEditorState>, SharedString)>,
    /// The reply chosen where a post has several, by parent (`None`: the
    /// first post). Absent: the newest.
    chosen: HashMap<Option<String>, String>,
    /// The reasoning disclosures the reader opened.
    reasoning_open: HashSet<RowKey>,
    _subscriptions: Vec<Subscription>,
}

impl EventEmitter<ConversationViewEvent> for ConversationView {}

impl ConversationView {
    pub fn new(
        stores: Stores,
        conversation: Entity<Conversation>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let composer = cx.new(|cx| Composer::new(window, cx));
        let list = ListState::new(0, ListAlignment::Top, px(1200.));
        list.set_follow_mode(FollowMode::Tail);
        let subscriptions = vec![
            cx.observe(&conversation, |this, _, cx| this.rebuild(cx)),
            cx.observe(&stores.spaces, |_, _, cx| cx.notify()),
            cx.subscribe_in(&conversation, window, Self::on_conversation_event),
            cx.subscribe_in(&composer, window, Self::on_composer_event),
        ];
        let mut this = Self {
            stores,
            conversation,
            composer,
            list,
            rows: Vec::new(),
            bodies: HashMap::new(),
            chosen: HashMap::new(),
            reasoning_open: HashSet::new(),
            _subscriptions: subscriptions,
        };
        this.rebuild(cx);
        // A new conversation opens ready to write.
        if this.rows.is_empty() {
            this.composer.update(cx, |c, cx| c.focus(window, cx));
        }
        this
    }

    pub fn conversation(&self) -> &Entity<Conversation> {
        &self.conversation
    }

    fn on_conversation_event(
        &mut self,
        _: &Entity<Conversation>,
        event: &ConversationEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            ConversationEvent::Posted {
                action_id,
                reply_to,
            } => {
                // Follow the branch the reader just wrote into.
                self.chosen.insert(reply_to.clone(), action_id.clone());
                self.composer.update(cx, |c, cx| c.set_holding(false, cx));
                self.list.set_follow_mode(FollowMode::Tail);
            }
            ConversationEvent::SubmitFailed { text, reply_to } => {
                let target = reply_to.as_ref().map(|id| ReplyTarget {
                    action_id: id.clone(),
                    author: self.author_of(id, cx),
                });
                self.composer.update(cx, |c, cx| {
                    c.set_holding(false, cx);
                    c.restore(text, target, cx);
                });
            }
            ConversationEvent::Landed { seq, action_id } => {
                if self.reasoning_open.remove(&RowKey::Turn(*seq)) {
                    self.reasoning_open.insert(RowKey::Post(action_id.clone()));
                }
            }
            ConversationEvent::Delta { seq } => {
                if let Some(ix) = self.rows.iter().position(|r| r.key() == RowKey::Turn(*seq)) {
                    self.refresh_turn_row(ix, cx);
                }
            }
        }
        cx.notify();
    }

    fn on_composer_event(
        &mut self,
        _: &Entity<Composer>,
        event: &ComposerEvent,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            ComposerEvent::Send { text, reply_to } => {
                // Writing continues the branch on screen, or answers the post
                // the reader chose to reply to.
                let reply_to = reply_to
                    .as_ref()
                    .map(|t| t.action_id.clone())
                    .or_else(|| self.branch_end());
                let accepted = self
                    .conversation
                    .update(cx, |c, cx| c.submit(text.clone(), reply_to.clone(), cx));
                if accepted {
                    self.composer.update(cx, |c, cx| {
                        c.sent(cx);
                        c.set_holding(true, cx);
                    });
                    self.list.set_follow_mode(FollowMode::Tail);
                }
            }
            ComposerEvent::Focused => {
                self.list.set_follow_mode(FollowMode::Tail);
            }
        }
        cx.notify();
    }

    /// The action id at the end of the visible branch.
    fn branch_end(&self) -> Option<String> {
        branch::tip(&self.rows)
    }

    fn author_of(&self, action_id: &str, cx: &gpui::App) -> SharedString {
        self.rows
            .iter()
            .find_map(|r| match r {
                Row::Post { post, .. } if post.action_id == action_id => {
                    Some(author_name(&post.author, cx))
                }
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Recompute the visible branch and tell the list what changed.
    fn rebuild(&mut self, cx: &mut Context<Self>) {
        let rows = self.visible_rows(cx);
        let old: Vec<RowKey> = self.rows.iter().map(Row::key).collect();
        let new: Vec<RowKey> = rows.iter().map(Row::key).collect();
        let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
        let suffix = old[prefix..]
            .iter()
            .rev()
            .zip(new[prefix..].iter().rev())
            .take_while(|(a, b)| a == b)
            .count();
        if old != new {
            self.list
                .splice(prefix..old.len() - suffix, new.len() - prefix - suffix);
        }
        // Rows whose content changed in place measure again.
        for (ix, (row, before)) in rows.iter().zip(self.rows.iter()).enumerate() {
            if row.key() == before.key() && row_text(row) != row_text(before) {
                self.list.remeasure_items(ix..ix + 1);
            }
        }
        let live: HashSet<RowKey> = new.iter().cloned().collect();
        self.bodies.retain(|k, _| live.contains(k));
        self.rows = rows;
        cx.notify();
    }

    fn refresh_turn_row(&mut self, ix: usize, cx: &mut Context<Self>) {
        let rows = self.visible_rows(cx);
        if rows.len() == self.rows.len() {
            self.rows = rows;
            self.list.remeasure_items(ix..ix + 1);
        } else {
            self.rebuild(cx);
        }
    }

    /// The branch on screen: from the first post, at each fork the chosen
    /// reply (else the newest); then the reader's post while it saves, and
    /// the answers streaming to whatever the branch ends with.
    fn visible_rows(&self, cx: &gpui::App) -> Vec<Row> {
        let conversation = self.conversation.read(cx);
        let posts: &[Post] = conversation
            .transcript()
            .value()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        branch::visible(
            posts,
            &self.chosen,
            conversation.pending(),
            conversation.turns(),
            |participant| conversation.label_of(participant),
        )
    }

    fn body(
        &mut self,
        key: &RowKey,
        text: &SharedString,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Entity<MarkdownEditorState> {
        if let Some((editor, seeded)) = self.bodies.get_mut(key) {
            if seeded != text {
                *seeded = text.clone();
                let text = text.to_string();
                editor.update(cx, |e, cx| e.set_value(text, cx));
            }
            return editor.clone();
        }
        let editor = cx.new(|cx| {
            let mut e = MarkdownEditorState::new(window, cx);
            e.set_value(text.to_string(), cx);
            e
        });
        self.bodies
            .insert(key.clone(), (editor.clone(), text.clone()));
        editor
    }

    fn switch_branch(&mut self, parent: Option<String>, to: usize, cx: &mut Context<Self>) {
        let posts = self
            .conversation
            .read(cx)
            .transcript()
            .value()
            .cloned()
            .unwrap_or_default();
        let siblings = branch::replies(&posts, &parent);
        if let Some(post) = siblings.get(to) {
            self.chosen.insert(parent, post.action_id.clone());
            self.list.set_follow_mode(FollowMode::Normal);
            self.rebuild(cx);
        }
    }

    fn render_row(&mut self, ix: usize, window: &mut Window, cx: &mut Context<Self>) -> AnyElement {
        let Some(row) = self.rows.get(ix).cloned() else {
            return div().into_any_element();
        };
        let theme = cx.theme().clone();
        let key = row.key();
        let style = theme::prose_style(cx);
        match row {
            Row::Post { post, fork } => {
                let editor = self.body(&key, &post.text, window, cx);
                let author = author_name(&post.author, cx);
                let action_id = post.action_id.clone();
                let reply_author = author.clone();
                let reasoning_open = self.reasoning_open.contains(&key);
                div()
                    .id(("post", ix))
                    .probe(
                        format!("conversation/post/{ix}"),
                        Role::Article,
                        format!("{author}, {}", fmt_when(post.created_at)),
                    )
                    .px(GUTTER)
                    .pt(row_top(ix))
                    .pb(px(4.))
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .children(fork.map(|(pos, count, parent)| {
                        branch_switch(ix, pos, count, parent, &theme, cx)
                    }))
                    .child(byline(author.clone(), post.created_at, &theme))
                    .children(post.reasoning.clone().map(|reasoning| {
                        reasoning_disclosure(ix, &key, reasoning_open, false, reasoning, &theme, cx)
                    }))
                    .child(MarkdownEditor::new(&editor).style(style).disabled(true))
                    .child(
                        div().flex().child(
                            div()
                                .id(("reply", ix))
                                .probe(
                                    format!("conversation/post/{ix}/reply"),
                                    Role::Button,
                                    msg::conversation_reply_label(cx, reply_author.to_string()),
                                )
                                .min_h(TOUCH)
                                .min_w(TOUCH)
                                .px(px(4.))
                                .flex()
                                .items_center()
                                .gap_1()
                                .text_size(px(14.))
                                .text_color(theme.muted_foreground)
                                .on_mouse_down(MouseButton::Left, keep_keyboard)
                                .on_click(cx.listener(move |this, _, window, cx| {
                                    let target = ReplyTarget {
                                        action_id: action_id.clone(),
                                        author: reply_author.clone(),
                                    };
                                    this.composer.update(cx, |c, cx| {
                                        c.set_reply(Some(target), cx);
                                        c.focus(window, cx);
                                    });
                                    window.request_virtual_keyboard();
                                }))
                                .child(Icon::new(IconName::Reply).size(px(16.)))
                                .child(msg::conversation_reply(cx)),
                        ),
                    )
                    .into_any_element()
            }
            Row::Pending {
                seq,
                text,
                created_at,
                ..
            } => {
                let editor = self.body(&key, &text, window, cx);
                div()
                    .id(("pending", ix))
                    .probe(
                        format!("conversation/pending/{seq}"),
                        Role::Article,
                        msg::conversation_you(cx),
                    )
                    .px(GUTTER)
                    .pt(row_top(ix))
                    .pb(px(12.))
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .opacity(0.7)
                    .child(byline(msg::conversation_you(cx), created_at, &theme))
                    .child(MarkdownEditor::new(&editor).style(style).disabled(true))
                    .into_any_element()
            }
            Row::Turn {
                seq,
                author,
                reasoning,
                content,
            } => {
                let editor = self.body(&key, &content, window, cx);
                let open = self.reasoning_open.contains(&key);
                let waiting = content.is_empty() && reasoning.is_empty();
                div()
                    .id(("turn", ix))
                    // A streaming answer carries no content value: a screen
                    // reader would re-read it whole at every token.
                    .probe(
                        format!("conversation/streaming/{seq}"),
                        Role::Group,
                        author.clone().unwrap_or_default(),
                    )
                    .px(GUTTER)
                    .pt(row_top(ix))
                    .pb(px(12.))
                    .flex()
                    .flex_col()
                    .gap(px(4.))
                    .children(author.map(|a| byline(a, 0, &theme)))
                    .when(!reasoning.is_empty(), |d| {
                        d.child(reasoning_disclosure(
                            ix,
                            &key,
                            open,
                            true,
                            reasoning.clone(),
                            &theme,
                            cx,
                        ))
                    })
                    .when(waiting, |d| {
                        d.child(
                            div()
                                .text_size(px(14.))
                                .text_color(theme.muted_foreground)
                                .child(msg::conversation_waiting(cx)),
                        )
                    })
                    .when(!content.is_empty(), |d| {
                        d.child(MarkdownEditor::new(&editor).style(style).disabled(true))
                    })
                    .into_any_element()
            }
        }
    }

    fn render_notices(&self, cx: &mut Context<Self>) -> Option<AnyElement> {
        let conversation = self.conversation.read(cx);
        let theme = cx.theme().clone();
        if let Some(error) = conversation.error().cloned() {
            let can_retry = conversation.can_retry();
            return Some(
                div()
                    .id("conversation-error")
                    .probe("conversation/error", Role::Alert, error_copy(&error, cx))
                    .mx(px(12.))
                    .mb(px(6.))
                    .p(px(12.))
                    .rounded_lg()
                    .border_1()
                    .border_color(theme.danger)
                    .bg(theme.background)
                    .flex()
                    .flex_col()
                    .gap_2()
                    .child(div().text_size(px(15.)).child(error_copy(&error, cx)))
                    .child(
                        div()
                            .flex()
                            .gap_2()
                            .justify_end()
                            .child(notice_verb(
                                "conversation/dismiss",
                                msg::conversation_dismiss(cx),
                                false,
                                &theme,
                                cx.listener(|this, _, _, cx| {
                                    this.conversation.update(cx, |c, cx| c.dismiss_error(cx))
                                }),
                            ))
                            .when(can_retry, |d| {
                                d.child(notice_verb(
                                    "conversation/retry",
                                    msg::conversation_retry(cx),
                                    true,
                                    &theme,
                                    cx.listener(|this, _, _, cx| {
                                        this.conversation.update(cx, |c, cx| c.retry(cx));
                                        this.list.set_follow_mode(FollowMode::Tail);
                                    }),
                                ))
                            }),
                    )
                    .into_any_element(),
            );
        }
        if let Some(depth) = conversation.cascade_paused() {
            return Some(
                div()
                    .id("conversation-cascade")
                    .probe(
                        "conversation/cascade",
                        Role::Alert,
                        msg::conversation_cascade_paused(cx, depth),
                    )
                    .mx(px(12.))
                    .mb(px(6.))
                    .flex()
                    .items_center()
                    .gap_2()
                    .text_size(px(14.))
                    .text_color(theme.muted_foreground)
                    .child(
                        div()
                            .flex_1()
                            .child(msg::conversation_cascade_paused(cx, depth)),
                    )
                    .child(notice_verb(
                        "conversation/cascade-dismiss",
                        msg::conversation_dismiss(cx),
                        false,
                        &theme,
                        cx.listener(|this, _, _, cx| {
                            this.conversation
                                .update(cx, |c, cx| c.dismiss_cascade_notice(cx))
                        }),
                    ))
                    .into_any_element(),
            );
        }
        None
    }
}

impl Render for ConversationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let theme = cx.theme().clone();
        let conversation = self.conversation.read(cx);
        let id = conversation.id().to_string();
        let transcript = conversation.transcript().clone();
        let gone = conversation.gone();
        let title = self
            .stores
            .spaces
            .read(cx)
            .info(&id)
            .and_then(|s| s.title.clone())
            .map(SharedString::from)
            .unwrap_or_else(|| msg::conversation_untitled(cx));

        let header = div()
            .h(HEADER_HEIGHT)
            .flex_none()
            .flex()
            .items_center()
            .gap_1()
            .px(px(4.))
            .border_b_1()
            .border_color(theme.border)
            .child(
                div()
                    .id("conversation-back")
                    .probe(
                        "conversation/back",
                        Role::Button,
                        msg::conversation_back(cx),
                    )
                    .size(TOUCH)
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded_full()
                    .active(|s| s.bg(theme.secondary))
                    .on_click(cx.listener(|_, _, _, cx| cx.emit(ConversationViewEvent::Back)))
                    .child(Icon::new(IconName::ChevronLeft).size(px(26.))),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .font_family(eidola_theme::FONT_FAMILY)
                    .text_size(px(19.))
                    .child(title),
            )
            .child(div().size(TOUCH).flex_none());

        let body: AnyElement = match (&gone, &transcript) {
            (Some(_), _) => quiet(msg::conversation_gone(cx), &theme).into_any_element(),
            (None, Loadable::Failed { prior: None, .. }) => div()
                .flex_1()
                .flex()
                .flex_col()
                .items_center()
                .justify_center()
                .gap_3()
                .child(quiet(msg::conversation_load_failed(cx), &theme))
                .child(notice_verb(
                    "conversation/transcript/retry",
                    msg::conversation_retry(cx),
                    true,
                    &theme,
                    cx.listener(|this, _, _, cx| this.conversation.update(cx, |c, cx| c.load(cx))),
                ))
                .into_any_element(),
            (None, Loadable::NotLoaded | Loadable::Loading) => div().flex_1().into_any_element(),
            _ if self.rows.is_empty() => {
                quiet(msg::conversation_empty(cx), &theme).into_any_element()
            }
            _ => list(
                self.list.clone(),
                cx.processor(|this, ix, window, cx| this.render_row(ix, window, cx)),
            )
            .flex_1()
            .size_full()
            .into_any_element(),
        };

        let stale = matches!(transcript, Loadable::Failed { prior: Some(_), .. });
        let notices = self.render_notices(cx);

        div()
            .size_full()
            .flex()
            .flex_col()
            .child(header)
            .when(stale && gone.is_none(), |d| {
                d.child(
                    div()
                        .id("conversation-refresh")
                        .probe(
                            "conversation/transcript/refresh-retry",
                            Role::Button,
                            msg::conversation_refresh_failed(cx),
                        )
                        .mx(px(12.))
                        .mt(px(6.))
                        .min_h(TOUCH)
                        .flex()
                        .items_center()
                        .justify_center()
                        .rounded_lg()
                        .bg(theme.secondary)
                        .text_size(px(14.))
                        .text_color(theme.secondary_foreground)
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.conversation.update(cx, |c, cx| c.load(cx))
                        }))
                        .child(msg::conversation_refresh_failed(cx)),
                )
            })
            // A tap on the conversation puts the keyboard away (the field keeps
            // its focus and draft); its controls keep it.
            .child(
                div()
                    .id("conversation-body")
                    .flex_1()
                    .min_h_0()
                    .flex()
                    .flex_col()
                    .on_mouse_down(MouseButton::Left, |_, window, _| {
                        window.dismiss_virtual_keyboard()
                    })
                    .child(body),
            )
            .children(notices)
            // No composer over a space that does not exist.
            .when(gone.is_none(), |d| d.child(self.composer.clone()))
    }
}

/// A row's top padding: the first row stands further from the header's rule.
fn row_top(ix: usize) -> gpui::Pixels {
    if ix == 0 { px(28.) } else { px(18.) }
}

/// A control's press: it takes no focus and leaves the keyboard as it is.
fn keep_keyboard(_: &gpui::MouseDownEvent, window: &mut Window, cx: &mut gpui::App) {
    window.prevent_default();
    cx.stop_propagation();
}

fn author_name(author: &Author, cx: &gpui::App) -> SharedString {
    match author {
        Author::Reader => msg::conversation_you(cx),
        Author::Named(label) => label.clone(),
    }
}

fn row_text(row: &Row) -> (SharedString, SharedString) {
    match row {
        Row::Post { post, .. } => (
            post.text.clone(),
            post.reasoning.clone().unwrap_or_default(),
        ),
        Row::Pending { text, .. } => (text.clone(), SharedString::default()),
        Row::Turn {
            reasoning, content, ..
        } => (content.clone(), reasoning.clone()),
    }
}

fn byline(author: SharedString, at: i64, theme: &gpui_component::Theme) -> impl IntoElement {
    div()
        .flex()
        .items_baseline()
        .gap_2()
        .text_size(px(13.))
        .child(div().text_color(theme.foreground).child(author))
        .when(at > 0, |d| {
            d.child(div().text_color(theme.muted_foreground).child(fmt_when(at)))
        })
}

#[allow(clippy::too_many_arguments)]
fn reasoning_disclosure(
    ix: usize,
    key: &RowKey,
    open: bool,
    live: bool,
    reasoning: SharedString,
    theme: &gpui_component::Theme,
    cx: &mut Context<ConversationView>,
) -> impl IntoElement {
    let label = if live && !open {
        msg::conversation_thinking(cx)
    } else if open {
        msg::conversation_hide_thinking(cx)
    } else {
        msg::conversation_show_thinking(cx)
    };
    let key = key.clone();
    div()
        .flex()
        .flex_col()
        .gap_1()
        .child(
            div()
                .id(("thinking", ix))
                .probe(
                    format!("conversation/row/{ix}/thinking"),
                    Role::Button,
                    label.clone(),
                )
                .aria_expanded(open)
                .min_h(TOUCH)
                .flex()
                .items_center()
                .gap_1()
                .text_size(px(14.))
                .text_color(theme.muted_foreground)
                .on_mouse_down(MouseButton::Left, keep_keyboard)
                .on_click(cx.listener(move |this, _, _, cx| {
                    if !this.reasoning_open.remove(&key) {
                        this.reasoning_open.insert(key.clone());
                    }
                    this.list.remeasure_items(ix..ix + 1);
                    cx.notify();
                }))
                .child(
                    Icon::new(if open {
                        IconName::ChevronDown
                    } else {
                        IconName::ChevronRight
                    })
                    .size(px(16.)),
                )
                .child(label),
        )
        .when(open, |d| {
            d.child(
                div()
                    .pl(px(10.))
                    .border_l_2()
                    .border_color(theme.border)
                    .text_size(px(14.))
                    .line_height(px(21.))
                    .text_color(theme.muted_foreground)
                    .child(reasoning),
            )
        })
}

fn branch_switch(
    ix: usize,
    pos: usize,
    count: usize,
    parent: Option<String>,
    theme: &gpui_component::Theme,
    cx: &mut Context<ConversationView>,
) -> impl IntoElement {
    let previous = parent.clone();
    let next = parent;
    div()
        .flex()
        .items_center()
        .gap_1()
        .text_size(px(13.))
        .text_color(theme.muted_foreground)
        .child(
            div()
                .id(("branch-previous", ix))
                .probe(
                    format!("conversation/post/{ix}/branch-previous"),
                    Role::Button,
                    msg::conversation_branch_previous(cx),
                )
                .size(TOUCH)
                .flex()
                .items_center()
                .justify_center()
                .when(pos == 0, |d| d.opacity(0.3))
                .on_mouse_down(MouseButton::Left, keep_keyboard)
                .on_click(cx.listener(move |this, _, _, cx| {
                    if pos > 0 {
                        this.switch_branch(previous.clone(), pos - 1, cx);
                    }
                }))
                .child(Icon::new(IconName::ChevronLeft).size(px(18.))),
        )
        .child(msg::conversation_branch(cx, (pos + 1) as i64, count as i64))
        .child(
            div()
                .id(("branch-next", ix))
                .probe(
                    format!("conversation/post/{ix}/branch-next"),
                    Role::Button,
                    msg::conversation_branch_next(cx),
                )
                .size(TOUCH)
                .flex()
                .items_center()
                .justify_center()
                .when(pos + 1 >= count, |d| d.opacity(0.3))
                .on_mouse_down(MouseButton::Left, keep_keyboard)
                .on_click(cx.listener(move |this, _, _, cx| {
                    if pos + 1 < count {
                        this.switch_branch(next.clone(), pos + 1, cx);
                    }
                }))
                .child(Icon::new(IconName::ChevronRight).size(px(18.))),
        )
}

fn notice_verb(
    name: &'static str,
    label: SharedString,
    primary: bool,
    theme: &gpui_component::Theme,
    on_click: impl Fn(&gpui::ClickEvent, &mut Window, &mut gpui::App) + 'static,
) -> impl IntoElement {
    div()
        .id(name)
        .probe(name, Role::Button, label.clone())
        .min_h(TOUCH)
        .px(px(16.))
        .flex()
        .items_center()
        .rounded_full()
        .text_size(px(15.))
        .when(primary, |d| {
            d.bg(theme.primary).text_color(theme.primary_foreground)
        })
        .when(!primary, |d| {
            d.bg(theme.secondary).text_color(theme.secondary_foreground)
        })
        .on_mouse_down(MouseButton::Left, keep_keyboard)
        .on_click(on_click)
        .child(label)
}

fn quiet(line: SharedString, theme: &gpui_component::Theme) -> impl IntoElement {
    div()
        .flex_1()
        .flex()
        .items_center()
        .justify_center()
        .px(GUTTER)
        .font_family(eidola_theme::FONT_FAMILY)
        .text_size(px(19.))
        .text_color(theme.muted_foreground)
        .text_center()
        .child(line)
}
