//! The app's one window: booting, the stack of screens (`super::nav`), and
//! the frame they sit in.
//!
//! **The bundle confirms after it has drawn.** Boot opens the profile
//! ([`crate::boot::prepare`]) while this window shows "Opening…"; once the
//! first screen (the Library, or the first run) has been drawn, the next
//! frame confirms the bundle with the shell
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
    AnyElement, AppContext as _, Context, Entity, InteractiveElement as _, IntoElement,
    ParentElement as _, Render, SharedString, Styled as _, Subscription, Task, Window, div, px,
};
use gpui_component::ActiveTheme as _;

use super::about::{AboutView, LicencesView};
use super::account_view::{AccountView, WalletView};
use super::appearance::AppearanceView;
use super::conversation_view::{ConversationView, ConversationViewEvent};
use super::host::Host;
use super::i18n::{self, msg};
use super::library::{LibraryEvent, LibraryView};
use super::nav::{self, Dest, NavEvent};
use super::onboarding::{OnboardingStart, OnboardingView};
use super::probe;
use super::purchase::PurchaseView;
use super::settings::SettingsView;
use super::stores::{self, Stores};
use super::theme;
use super::updates_view::UpdatesView;

/// Confirms the bundle with its shell (`bundle.confirm`).
pub type Confirm = Box<dyn FnOnce() -> LocalBoxFuture<'static, Result<(), String>>>;

/// What a successful boot hands the interface.
pub struct Booted {
    pub core: AppCore,
    pub host: Rc<dyn Host>,
    pub confirm: Confirm,
}

/// The boot, as the window's first task runs it: the shell, the profile, the
/// database. An error is the text the failure screen shows.
pub type Boot = LocalBoxFuture<'static, Result<Booted, String>>;

/// One screen of the stack.
enum Screen {
    Library(Entity<LibraryView>),
    Conversation(Entity<ConversationView>),
    Onboarding(Entity<OnboardingView>),
    Settings(Entity<SettingsView>),
    Account(Entity<AccountView>),
    Wallet(Entity<WalletView>),
    Purchase(Entity<PurchaseView>),
    Appearance(Entity<AppearanceView>),
    Updates(Entity<UpdatesView>),
    About(Entity<AboutView>),
    Licences(Entity<LicencesView>),
}

impl Screen {
    fn element(&self) -> AnyElement {
        match self {
            Screen::Library(v) => v.clone().into_any_element(),
            Screen::Conversation(v) => v.clone().into_any_element(),
            Screen::Onboarding(v) => v.clone().into_any_element(),
            Screen::Settings(v) => v.clone().into_any_element(),
            Screen::Account(v) => v.clone().into_any_element(),
            Screen::Wallet(v) => v.clone().into_any_element(),
            Screen::Purchase(v) => v.clone().into_any_element(),
            Screen::Appearance(v) => v.clone().into_any_element(),
            Screen::Updates(v) => v.clone().into_any_element(),
            Screen::About(v) => v.clone().into_any_element(),
            Screen::Licences(v) => v.clone().into_any_element(),
        }
    }
}

enum AppState {
    Starting,
    Failed(SharedString),
    /// Booted; the first screen waits for the account check.
    Ready {
        stores: Stores,
        /// The screen stack, the root first. Empty until the first screen
        /// is chosen.
        stack: Vec<Screen>,
    },
}

pub struct AppView {
    state: AppState,
    boot_task: Option<Task<()>>,
    /// Taken by the first frame that shows the booted app.
    confirm: Option<Confirm>,
    confirm_task: Option<Task<()>>,
    pending_navigation: Vec<Navigation>,
    /// One subscription per screen on the stack.
    subscriptions: Vec<Subscription>,
    _back: Option<Task<()>>,
    _foreground: Option<Task<()>>,
    _first_screen: Option<Subscription>,
}

impl AppView {
    pub fn new(
        boot: Boot,
        preferred_languages: Vec<String>,
        _window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let boot_task = cx.spawn(async move |this, cx| {
            let result = boot.await;
            this.update(cx, |this, cx| this.booted(result, preferred_languages, cx))
                .ok();
        });
        // The system's back gesture (Android), one listener for the app.
        let back = nav::system_back_events().map(|mut events| {
            cx.spawn(async move |this, cx| {
                while events.recv().await.is_some() {
                    if this
                        .update(cx, |this, cx| {
                            this.pending_navigation.push(Navigation::SystemBack);
                            cx.notify();
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
        });
        // Back from the browser after a checkout: the balance is read again.
        let foreground = nav::foreground_events().map(|mut events| {
            cx.spawn(async move |this, cx| {
                while events.recv().await.is_some() {
                    let returned = this.update(cx, |this, cx| {
                        if let AppState::Ready { stores, .. } = &this.state {
                            stores
                                .account
                                .update(cx, |a, cx| a.returned_to_foreground(cx));
                        }
                    });
                    if returned.is_err() {
                        break;
                    }
                }
            })
        });
        Self {
            state: AppState::Starting,
            boot_task: Some(boot_task),
            confirm: None,
            confirm_task: None,
            pending_navigation: Vec::new(),
            subscriptions: Vec::new(),
            _back: back,
            _foreground: foreground,
            _first_screen: None,
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
                let stores = Stores::new(core, booted.host, cx);
                stores::install_bus_bridge(&stores, preferred_languages, cx);
                self.confirm = Some(booted.confirm);
                // The first screen: the first run when there is no account
                // and the reader has not chosen to go on without one (the
                // Eidola backend is still enabled), else the Library. Waits
                // for the registry read, so nothing flashes.
                let account = stores.account.clone();
                self._first_screen = Some(cx.observe(&account, |this, _, cx| {
                    this.choose_first_screen(cx);
                }));
                self.state = AppState::Ready {
                    stores,
                    stack: Vec::new(),
                };
                self.choose_first_screen(cx);
            }
        }
        cx.notify();
    }

    fn choose_first_screen(&mut self, cx: &mut Context<Self>) {
        let AppState::Ready { stores, stack } = &self.state else {
            return;
        };
        if !stack.is_empty() {
            self._first_screen = None;
            return;
        }
        let account = stores.account.read(cx);
        let first_run = match account.eidola_enabled() {
            l if l.is_loading() => return,
            // A failed read errs toward showing the first run, which the
            // reader can leave.
            l => !account.has_account() && l.value().copied().unwrap_or(true),
        };
        self._first_screen = None;
        self.pending_navigation
            .push(Navigation::First { first_run });
        cx.notify();
    }

    fn subscribe(&mut self, screen: &Screen, cx: &mut Context<Self>) -> Subscription {
        fn nav(this: &mut AppView, event: NavEvent, cx: &mut Context<AppView>) {
            this.pending_navigation.push(Navigation::Nav(event));
            cx.notify();
        }
        match screen {
            Screen::Library(v) => cx.subscribe(v, |this, _, event: &LibraryEvent, cx| {
                this.pending_navigation.push(match event {
                    LibraryEvent::Settings => Navigation::Nav(NavEvent::To(Dest::Settings)),
                    other => Navigation::Open(other.clone()),
                });
                cx.notify();
            }),
            Screen::Conversation(v) => cx.subscribe(
                v,
                |this, _, event: &ConversationViewEvent, cx| match event {
                    ConversationViewEvent::Back => nav(this, NavEvent::Back, cx),
                },
            ),
            Screen::Onboarding(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Settings(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Account(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Wallet(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Purchase(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Appearance(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Updates(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::About(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
            Screen::Licences(v) => cx.subscribe(v, |this, _, e: &NavEvent, cx| nav(this, *e, cx)),
        }
    }
}

/// A navigation asked for by a screen, carried out at the next render (where
/// there is a window to build the next screen's editors in).
enum Navigation {
    /// The first screen, once the account check answered.
    First {
        first_run: bool,
    },
    Open(LibraryEvent),
    Nav(NavEvent),
    /// Android's back gesture: its history entry is already gone.
    SystemBack,
}

impl AppView {
    fn build(
        &self,
        dest: Dest,
        stores: &Stores,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Screen {
        let s = stores.clone();
        match dest {
            Dest::Settings => Screen::Settings(cx.new(|cx| SettingsView::new(s, cx))),
            Dest::Account => Screen::Account(cx.new(|cx| AccountView::new(s, cx))),
            Dest::Wallet => Screen::Wallet(cx.new(|cx| WalletView::new(s, cx))),
            Dest::Purchase => Screen::Purchase(cx.new(|cx| PurchaseView::new(s, cx))),
            Dest::Appearance => Screen::Appearance(cx.new(|cx| AppearanceView::new(s, cx))),
            Dest::Updates => Screen::Updates(cx.new(|cx| UpdatesView::new(s, cx))),
            Dest::About => Screen::About(cx.new(|cx| AboutView::new(s, cx))),
            Dest::Licences => Screen::Licences(cx.new(|cx| LicencesView::new(s, cx))),
            Dest::SetUpAccount => Screen::Onboarding(
                cx.new(|cx| OnboardingView::new(s, OnboardingStart::Choose, window, cx)),
            ),
            Dest::AcceptTerms => Screen::Onboarding(
                cx.new(|cx| OnboardingView::new(s, OnboardingStart::AcceptTerms, window, cx)),
            ),
        }
    }

    fn push(&mut self, screen: Screen, cx: &mut Context<Self>) {
        let subscription = self.subscribe(&screen, cx);
        if let AppState::Ready { stack, .. } = &mut self.state {
            if !stack.is_empty() {
                nav::history::pushed();
            }
            stack.push(screen);
            self.subscriptions.push(subscription);
        }
    }

    /// Pop the top screen (never the root). A conversation left behind is
    /// handed to the store, which discards it if nothing was written.
    fn pop(&mut self, window: &mut Window, cx: &mut Context<Self>) -> bool {
        let AppState::Ready { stores, stack } = &mut self.state else {
            return false;
        };
        if stack.len() <= 1 {
            return false;
        }
        let top = stack.pop();
        self.subscriptions.pop();
        if let Some(Screen::Conversation(view)) = top {
            let conversation = view.read(cx).conversation().clone();
            stores
                .spaces
                .update(cx, |spaces, cx| spaces.left(&conversation, cx));
        }
        window.dismiss_virtual_keyboard();
        true
    }

    fn navigate(&mut self, navigation: Navigation, window: &mut Window, cx: &mut Context<Self>) {
        let AppState::Ready { stores, stack } = &self.state else {
            return;
        };
        let stores = stores.clone();
        let root_is_library = matches!(stack.first(), Some(Screen::Library(_)));
        match navigation {
            Navigation::First { first_run } => {
                let screen = if first_run {
                    Screen::Onboarding(cx.new(|cx| {
                        OnboardingView::new(stores.clone(), OnboardingStart::Welcome, window, cx)
                    }))
                } else {
                    Screen::Library(cx.new(|cx| LibraryView::new(stores.clone(), cx)))
                };
                self.push(screen, cx);
            }
            Navigation::Open(event) => {
                let conversation = stores.spaces.update(cx, |spaces, cx| match event {
                    LibraryEvent::Open(id) => spaces.open(&id, cx),
                    _ => spaces.create(cx),
                });
                let view =
                    cx.new(|cx| ConversationView::new(stores.clone(), conversation, window, cx));
                self.push(Screen::Conversation(view), cx);
            }
            Navigation::Nav(NavEvent::To(dest)) => {
                let screen = self.build(dest, &stores, window, cx);
                self.push(screen, cx);
            }
            Navigation::Nav(NavEvent::Back) => {
                if self.pop(window, cx) {
                    nav::history::popped();
                }
            }
            Navigation::SystemBack => {
                self.pop(window, cx);
            }
            Navigation::Nav(NavEvent::Home) => {
                if root_is_library {
                    while self.pop(window, cx) {}
                    nav::history::reset();
                } else {
                    // The first run was the root: the Library replaces it.
                    while self.pop(window, cx) {}
                    nav::history::reset();
                    if let AppState::Ready { stack, .. } = &mut self.state {
                        stack.clear();
                    }
                    self.subscriptions.clear();
                    window.dismiss_virtual_keyboard();
                    let library = cx.new(|cx| LibraryView::new(stores.clone(), cx));
                    self.push(Screen::Library(library), cx);
                }
            }
        }
    }
}

impl Render for AppView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        probe::begin_frame();
        window.on_next_frame(|_, _| probe::flush());
        for navigation in std::mem::take(&mut self.pending_navigation) {
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
            AppState::Ready { stack, .. } if stack.is_empty() => {
                status(msg::app_starting(cx), None, cx).into_any_element()
            }
            AppState::Ready { stack, .. } => {
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
                stack
                    .last()
                    .map(Screen::element)
                    .unwrap_or_else(|| div().into_any_element())
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
