//! The phone interface, on gpui: one window, a stack of screens over the
//! Library, the desktop app's look and voice adapted to a phone.
//!
//! - [`app`]: boot, the screen stack ([`nav`]), the frame that avoids the
//!   notch and keyboard, and the bundle's confirmation after its first frame.
//! - [`stores`] / [`conversation`] / [`account_store`] / [`update_store`]:
//!   the domain state (the Library index, one entity per space, the account
//!   and wallet, the self-update) and the bus bridge that keeps it true.
//! - [`host`]: what the interface asks of the shell and the page.
//! - [`library`], [`conversation_view`], [`composer`]: the conversations.
//! - [`onboarding`], [`settings`], [`account_view`], [`purchase`],
//!   [`appearance`], [`updates_view`], [`about`]: the first run, and
//!   everything else a reader needs; [`widgets`] their shared pieces.
//! - [`theme`], [`i18n`]: the shared Circadian look and the compile-time
//!   locales, wired to this app.
//! - [`errors`]: typed errors become words at render.
//! - [`probe`]: roles and labels, and the development bundle's probe mirror.
//!
//! The interface is target-independent; only its entry point (`crate::web`)
//! is wasm32.

pub mod about;
pub mod account_store;
pub mod account_view;
pub mod app;
pub mod appearance;
pub mod branch;
pub mod composer;
pub mod conversation;
pub mod conversation_view;
pub mod errors;
pub mod host;
pub mod i18n;
pub mod library;
pub mod loadable;
pub mod nav;
pub mod onboarding;
pub mod probe;
pub mod purchase;
pub mod settings;
pub mod stores;
pub mod theme;
pub mod update_store;
pub mod updates_view;
pub mod widgets;

use gpui::{App, AppContext as _, Pixels, SharedString, WindowOptions, px};

/// Side margin of the reading column and the screens' chrome.
pub const GUTTER: Pixels = px(20.);
/// A screen's header bar.
pub const HEADER_HEIGHT: Pixels = px(52.);
/// The smallest touch target (Apple's 44 pt; Android asks 48 dp of the shell's
/// density-scaled pixels, which 44 CSS px exceeds).
pub const TOUCH: Pixels = px(44.);

// The icons the interface draws, embedded: the web build of the icon crate
// would otherwise fetch them from a CDN, and the shell allows no other origin.
gpui_kit_assets::icon_assets!(
    pub Icons,
    [
        ArrowUp,
        Check,
        ChevronDown,
        ChevronLeft,
        ChevronRight,
        Close,
        Eye,
        EyeOff,
        Minus,
        Plus,
        Reply,
        Settings,
        SquarePen
    ]
);

/// Initialise the libraries, the theme and the source locale. Before the
/// window opens.
pub fn init(cx: &mut App) {
    gpui_component::init(cx);
    gpui_markdown_editor::init(cx);
    theme::install(cx);
}

/// Open the app's window over `boot`.
pub fn open_window(boot: app::Boot, preferred_languages: Vec<String>, cx: &mut App) {
    let opened = cx.open_window(WindowOptions::default(), |window, cx| {
        theme::observe_window_appearance(window);
        let view = cx.new(|cx| app::AppView::new(boot, preferred_languages, window, cx));
        cx.new(|cx| gpui_component::Root::new(view, window, cx))
    });
    if let Err(e) = opened {
        log::error!("failed to open the window: {e}");
    }
}

/// When a post or a conversation was last active, in the reader's local
/// time: the time of day for today, else the date. Numbers only, so no words
/// to localize.
pub fn fmt_when(ms: i64) -> SharedString {
    let (y, mo, d, h, mi) = local_parts(ms);
    let (ty, tmo, td, _, _) = local_parts(conversation::now_ms());
    if (y, mo, d) == (ty, tmo, td) {
        format!("{h:02}:{mi:02}").into()
    } else {
        format!("{y:04}-{mo:02}-{d:02}").into()
    }
}

/// (year, month, day, hour, minute) of a unix-millis time, local on wasm32
/// and UTC natively.
fn local_parts(ms: i64) -> (i64, i64, i64, i64, i64) {
    #[cfg(target_family = "wasm")]
    {
        let date = js_sys::Date::new(&wasm_bindgen::JsValue::from_f64(ms as f64));
        (
            i64::from(date.get_full_year()),
            i64::from(date.get_month()) + 1,
            i64::from(date.get_date()),
            i64::from(date.get_hours()),
            i64::from(date.get_minutes()),
        )
    }
    #[cfg(not(target_family = "wasm"))]
    {
        let secs = ms.div_euclid(1000);
        let days = secs.div_euclid(86_400);
        let rem = secs.rem_euclid(86_400);
        // Civil from days (Howard Hinnant's algorithm).
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = yoe + era * 400 + i64::from(m <= 2);
        (y, m, d, rem / 3600, rem % 3600 / 60)
    }
}
