//! The app's one window: booting, the two screens, and the frame they sit in.
//!
//! **The bundle confirms after it has drawn.** Boot opens the profile
//! ([`crate::boot::prepare`]) while this window shows "Opening…"; once the
//! Library has been drawn, the next frame confirms the bundle with the shell
//! (`bundle.confirm`). A bundle that boots but cannot draw never confirms, and
//! the shell falls back at the next launch (`mobile/shell-contract.md` →
//! Provisional activation). A boot that fails shows why, and never confirms.
//!
//! **The frame avoids everything that covers the page**: the notch, the status
//! bar, the home indicator and the software keyboard. The webview reports the
//! safe areas (`env(safe-area-inset-*)`) and the keyboard (the visual
//! viewport), and `Window::fully_visible_bounds` is their intersection; the
//! screens are laid out inside it while the paper fills the whole viewport.
//! (`insets().ime` stays zero on the web, so the visual viewport is what
//! carries the keyboard.) On Android the shell already pads the webview by
//! the system bars and the keyboard, so the same rule leaves nothing more to
//! avoid.

use std::rc::Rc;

use eidola_app_core::AppCore;
use futures_util::future::LocalBoxFuture;
use gpui::{
    AppContext as _, Context, Entity, InteractiveElement as _, IntoElement, ParentElement as _,
    Render, SharedString, Styled as _, Subscription, Task, Window, div, px,
};
use gpui_component::ActiveTheme as _;

use super::conversation_view::{ConversationView, ConversationViewEvent};
use super::i18n::{self, msg};
use super::library::{LibraryEvent, LibraryView};
use super::probe;
use super::stores::{self, Stores};
use super::theme;

/// Confirms the bundle with its shell (`bundle.confirm`).
pub type Confirm = Box<dyn FnOnce() -> LocalBoxFuture<'static, Result<(), String>>>;

/// What a successful boot hands the interface.
pub struct Booted {
    pub core: AppCore,
    pub confirm: Confirm,
}

/// The boot, as the window's first task runs it: the shell, the profile, the
/// database. An error is the text the failure screen shows.
pub type Boot = LocalBoxFuture<'static, Result<Booted, String>>;

enum Screen {
    Library(Entity<LibraryView>),
    Conversation(Entity<ConversationView>),
}

enum AppState {
    Starting,
    Failed(SharedString),
    Ready { stores: Stores, screen: Screen },
}

pub struct AppView {
    state: AppState,
    boot_task: Option<Task<()>>,
    /// Taken by the first frame that shows the booted app.
    confirm: Option<Confirm>,
    confirm_task: Option<Task<()>>,
    pending_navigation: Option<Navigation>,
    _screen: Option<Subscription>,
}

impl AppView {
    pub fn new(boot: Boot, preferred_languages: Vec<String>, cx: &mut Context<Self>) -> Self {
        let boot_task = cx.spawn(async move |this, cx| {
            let result = boot.await;
            this.update(cx, |this, cx| this.booted(result, preferred_languages, cx))
                .ok();
        });
        Self {
            state: AppState::Starting,
            boot_task: Some(boot_task),
            confirm: None,
            confirm_task: None,
            pending_navigation: None,
            _screen: None,
        }
    }

    fn booted(
        &mut self,
        result: Result<Booted, String>,
        preferred_languages: Vec<String>,
        cx: &mut Context<Self>,
    ) {
        self.boot_task = None;
        match result {
            Err(reason) => {
                log::error!("boot failed: {reason}");
                self.state = AppState::Failed(reason.into());
            }
            Ok(booted) => {
                let core = Rc::new(booted.core);
                let config = core.config_state();
                theme::set_settings(theme::settings_of(&config), cx);
                i18n::apply(
                    i18n::resolve(config.language.as_deref(), &preferred_languages),
                    cx,
                );
                theme::start_clock(cx);
                let stores = Stores::new(core, cx);
                stores::install_bus_bridge(&stores, preferred_languages, cx);
                self.confirm = Some(booted.confirm);
                self.state = AppState::Ready {
                    screen: Screen::Library(cx.new(|cx| LibraryView::new(stores.clone(), cx))),
                    stores,
                };
                self.subscribe_screen(cx);
            }
        }
        cx.notify();
    }

    fn subscribe_screen(&mut self, cx: &mut Context<Self>) {
        let AppState::Ready { screen, .. } = &self.state else {
            return;
        };
        self._screen = Some(match screen {
            Screen::Library(view) => cx.subscribe(view, |this, _, event: &LibraryEvent, cx| {
                this.pending_navigation = Some(Navigation::Open(event.clone()));
                cx.notify();
            }),
            Screen::Conversation(view) => cx.subscribe(
                view,
                |this, _, event: &ConversationViewEvent, cx| match event {
                    ConversationViewEvent::Back => {
                        this.pending_navigation = Some(Navigation::Back);
                        cx.notify();
                    }
                },
            ),
        });
    }
}

/// A navigation asked for by a screen, carried out at the next render (where
/// there is a window to build the next screen's editors in).
enum Navigation {
    Open(LibraryEvent),
    Back,
}

impl AppView {
    fn navigate(&mut self, navigation: Navigation, window: &mut Window, cx: &mut Context<Self>) {
        let AppState::Ready { stores, screen } = &mut self.state else {
            return;
        };
        let stores = stores.clone();
        match navigation {
            Navigation::Open(event) => {
                let conversation = stores.spaces.update(cx, |spaces, cx| match event {
                    LibraryEvent::Open(id) => spaces.open(&id, cx),
                    LibraryEvent::New => spaces.create(cx),
                });
                let view =
                    cx.new(|cx| ConversationView::new(stores.clone(), conversation, window, cx));
                *screen = Screen::Conversation(view);
            }
            Navigation::Back => {
                if let Screen::Conversation(view) = screen {
                    let conversation = view.read(cx).conversation().clone();
                    stores
                        .spaces
                        .update(cx, |spaces, cx| spaces.left(&conversation, cx));
                }
                window.dismiss_virtual_keyboard();
                *screen = Screen::Library(cx.new(|cx| LibraryView::new(stores.clone(), cx)));
            }
        }
        self.subscribe_screen(cx);
    }
}

impl Render for AppView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        probe::begin_frame();
        window.on_next_frame(|_, _| probe::flush());
        if let Some(navigation) = self.pending_navigation.take() {
            self.navigate(navigation, window, cx);
        }

        let background = cx.theme().background;
        let viewport = window.viewport_size();
        // The viewport reads 0×0 on the first frame in a webview: draw the
        // paper and ask for another frame rather than laying out into nothing.
        if viewport.width <= px(0.) || viewport.height <= px(0.) {
            window.request_animation_frame();
            return div()
                .id("app")
                .size_full()
                .bg(background)
                .into_any_element();
        }
        let visible = window.fully_visible_bounds();

        let content = match &self.state {
            AppState::Starting => status(msg::app_starting(cx), None, cx).into_any_element(),
            AppState::Failed(reason) => status(
                msg::app_start_failed(cx),
                Some(msg::app_start_failed_detail(cx, reason.to_string())),
                cx,
            )
            .into_any_element(),
            AppState::Ready { screen, .. } => {
                // The first frame that shows the booted app has been drawn by
                // the time the next one starts: confirm then.
                if let Some(confirm) = self.confirm.take() {
                    let this = cx.entity().downgrade();
                    window.on_next_frame(move |_, cx| {
                        this.update(cx, |this, cx| {
                            this.confirm_task =
                                Some(cx.spawn(async move |_, _| match confirm().await {
                                    Ok(()) => log::info!("bundle confirmed after its first frame"),
                                    Err(e) => log::error!("bundle.confirm failed: {e}"),
                                }));
                        })
                        .ok();
                    });
                }
                match screen {
                    Screen::Library(view) => view.clone().into_any_element(),
                    Screen::Conversation(view) => view.clone().into_any_element(),
                }
            }
        };

        div()
            .id("app")
            .size_full()
            .relative()
            .bg(background)
            .text_color(cx.theme().foreground)
            .child(
                div()
                    .absolute()
                    .left(visible.left())
                    .top(visible.top())
                    .w(visible.size.width)
                    .h(visible.size.height)
                    .child(content),
            )
            .into_any_element()
    }
}

fn status(
    line: SharedString,
    detail: Option<SharedString>,
    cx: &Context<AppView>,
) -> impl IntoElement {
    let muted = cx.theme().muted_foreground;
    div()
        .size_full()
        .flex()
        .flex_col()
        .items_center()
        .justify_center()
        .gap_2()
        .px(px(24.))
        .child(
            div()
                .font_family(eidola_theme::FONT_FAMILY)
                .text_size(px(19.))
                .text_center()
                .child(line),
        )
        .children(detail.map(|d| {
            div()
                .text_size(px(14.))
                .text_color(muted)
                .text_center()
                .child(d)
        }))
}
