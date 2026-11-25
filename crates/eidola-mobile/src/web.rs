//! The webview side: the bridge's transport over `js/bridge.js`, and the
//! bundle's entry points — `start` (the app), `startDev` (a development
//! bundle) and `selfTest`.

use std::cell::RefCell;
use std::collections::HashMap;

use tokio::sync::oneshot;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::boot::{self, BootOptions};
use crate::client::{BoxFuture, ShellClient, Transport};
use crate::protocol;
use crate::ui;

#[wasm_bindgen(module = "/js/bridge.js")]
extern "C" {
    fn platform() -> Option<String>;
    #[wasm_bindgen(js_name = postIos, catch)]
    fn post_ios(text: &str) -> Result<js_sys::Promise, JsValue>;
    #[wasm_bindgen(js_name = openAndroid, catch)]
    fn open_android(on_reply: &Closure<dyn FnMut(String)>) -> Result<(), JsValue>;
    #[wasm_bindgen(js_name = postAndroid, catch)]
    fn post_android(text: &str) -> Result<(), JsValue>;
    #[wasm_bindgen(js_name = preferredLanguages)]
    fn preferred_languages_json() -> String;
}

#[wasm_bindgen(module = "/js/page.js")]
extern "C" {
    #[wasm_bindgen(js_name = reloadPage)]
    fn reload_page();
    /// A file of this document's own origin (the bundle the shell served),
    /// as text.
    #[wasm_bindgen(js_name = fetchText, catch)]
    fn fetch_text(path: &str) -> Result<js_sys::Promise, JsValue>;
    #[wasm_bindgen(js_name = watchVisibility)]
    fn watch_visibility(on_visible: &Closure<dyn FnMut()>);
}

#[wasm_bindgen(module = "/js/nav.js")]
extern "C" {
    #[wasm_bindgen(js_name = navInit)]
    fn nav_init(on_back: &Closure<dyn FnMut()>);
    #[wasm_bindgen(js_name = navPush)]
    fn nav_push_js();
    #[wasm_bindgen(js_name = navPopped)]
    fn nav_popped_js();
    #[wasm_bindgen(js_name = navReset)]
    fn nav_reset_js();
}

pub(crate) fn nav_push() {
    nav_push_js();
}

pub(crate) fn nav_popped() {
    nav_popped_js();
}

pub(crate) fn nav_reset() {
    nav_reset_js();
}

/// Listen for the system's back gesture, and for the page coming back to the
/// foreground, for the page's lifetime.
fn listen_for_back() {
    let on_back = Closure::<dyn FnMut()>::new(crate::ui::nav::system_back);
    nav_init(&on_back);
    on_back.forget();
    let on_visible = Closure::<dyn FnMut()>::new(crate::ui::nav::came_to_foreground);
    watch_visibility(&on_visible);
    on_visible.forget();
}

/// The page's own calls, for the interface's host.
fn page() -> ui::host::Page {
    ui::host::Page {
        reload: Box::new(reload_page),
        fetch_text: Box::new(|path| {
            let promise = fetch_text(path);
            Box::pin(async move {
                let promise = promise.map_err(|e| describe(&e))?;
                let text = JsFuture::from(promise).await.map_err(|e| describe(&e))?;
                text.as_string()
                    .ok_or_else(|| "the file is not text".to_owned())
            })
        }),
    }
}

#[cfg(feature = "dev")]
#[wasm_bindgen(module = "/js/probes.js")]
extern "C" {
    #[wasm_bindgen(js_name = syncProbes)]
    fn sync_probes_js(json: &str);
}

/// Mirror the frame's probes into the page (`crate::ui::probe`).
#[cfg(feature = "dev")]
pub(crate) fn sync_probes(json: &str) {
    sync_probes_js(json);
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(text: &str);
    #[wasm_bindgen(js_namespace = console, js_name = error)]
    fn log_error(text: &str);
    #[cfg(feature = "self-test")]
    #[wasm_bindgen(js_name = setTimeout)]
    fn set_timeout(f: &js_sys::Function, ms: u32);
}

#[cfg(feature = "self-test")]
fn sleep_ms(ms: u32) -> BoxFuture<'static, ()> {
    let promise = js_sys::Promise::new(&mut |resolve, _| set_timeout(&resolve, ms));
    let timer = JsFuture::from(promise);
    Box::pin(async move {
        let _ = timer.await;
    })
}

fn describe(value: &JsValue) -> String {
    value
        .as_string()
        .or_else(|| {
            value
                .dyn_ref::<js_sys::Error>()
                .map(|e| String::from(e.message()))
        })
        .unwrap_or_else(|| format!("{value:?}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Platform {
    Ios,
    Android,
}

/// The shell's bridge. iOS answers each message with its reply; Android
/// replies on one port, matched to requests here by id.
#[derive(Clone, Copy)]
pub struct WebTransport {
    platform: Platform,
}

type Waiter = oneshot::Sender<Result<String, String>>;
type OnReply = Closure<dyn FnMut(String)>;

thread_local! {
    static WAITING: RefCell<HashMap<u64, Waiter>> = RefCell::new(HashMap::new());
    static ANDROID_REPLIES: RefCell<Option<OnReply>> = const { RefCell::new(None) };
}

fn on_android_reply(text: String) {
    let Some(id) = protocol::reply_id(&text) else {
        log_error("shell bridge: a reply without an id was dropped");
        return;
    };
    if let Some(waiter) = WAITING.with(|w| w.borrow_mut().remove(&id)) {
        // The caller may have stopped waiting; its reply goes nowhere.
        let _ = waiter.send(Ok(text));
    }
}

impl WebTransport {
    /// The bridge of the shell this page runs in, if it runs in one.
    pub fn connect() -> Result<Self, String> {
        match platform().as_deref() {
            Some("ios") => Ok(Self {
                platform: Platform::Ios,
            }),
            Some("android") => {
                let replies = OnReply::new(on_android_reply);
                open_android(&replies).map_err(|e| describe(&e))?;
                ANDROID_REPLIES.with(|slot| *slot.borrow_mut() = Some(replies));
                Ok(Self {
                    platform: Platform::Android,
                })
            }
            _ => Err("this page is not running in an Eidola shell".into()),
        }
    }
}

impl Transport for WebTransport {
    fn call(&self, id: u64, request: String) -> BoxFuture<'static, Result<String, String>> {
        match self.platform {
            Platform::Ios => match post_ios(&request) {
                Ok(promise) => {
                    let reply = JsFuture::from(promise);
                    Box::pin(async move {
                        let text = reply.await.map_err(|e| describe(&e))?;
                        text.as_string()
                            .ok_or_else(|| "the reply is not a string".to_owned())
                    })
                }
                Err(e) => {
                    let message = describe(&e);
                    Box::pin(async move { Err(message) })
                }
            },
            Platform::Android => {
                let (waiter, reply) = oneshot::channel();
                WAITING.with(|w| w.borrow_mut().insert(id, waiter));
                if let Err(e) = post_android(&request) {
                    WAITING.with(|w| w.borrow_mut().remove(&id));
                    let message = describe(&e);
                    return Box::pin(async move { Err(message) });
                }
                Box::pin(async move {
                    reply
                        .await
                        .unwrap_or_else(|_| Err("the bridge dropped the request".into()))
                })
            }
        }
    }
}

#[cfg(feature = "self-test")]
fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| log_error(&format!("panic: {info}"))));
}

/// The webview's preferred languages (which follow the device's), most
/// preferred first.
fn preferred_languages() -> Vec<String> {
    serde_json::from_str(&preferred_languages_json()).unwrap_or_default()
}

/// Start the app on its shell: open the window at once ("Opening…") and boot
/// behind it; the interface confirms the bundle once it has drawn
/// (`crate::ui::app`). Returns as soon as the window is up; a boot that fails
/// is shown, not thrown.
#[wasm_bindgen]
pub fn start() -> Result<(), JsValue> {
    launch(None)
}

/// A development bundle's start: the same app, with the default agent seated
/// on the development server `config` (`dev-backend.json`) names.
#[cfg(feature = "dev")]
#[wasm_bindgen(js_name = startDev)]
pub fn start_dev(config: String) -> Result<(), JsValue> {
    let config: crate::dev::DevConfig =
        serde_json::from_str(&config).map_err(|e| JsValue::from_str(&e.to_string()))?;
    launch(Some(config))
}

#[cfg(feature = "dev")]
type DevConfig = crate::dev::DevConfig;
#[cfg(not(feature = "dev"))]
type DevConfig = std::convert::Infallible;

fn launch(dev: Option<DevConfig>) -> Result<(), JsValue> {
    // Panics and gpui's log to the console.
    gpui_platform::web_init();
    listen_for_back();
    let transport = WebTransport::connect().map_err(|e| JsValue::from_str(&e))?;
    let boot: ui::app::Boot = Box::pin(async move {
        #[cfg(feature = "dev")]
        let options = BootOptions {
            extra_tls_roots: dev
                .as_ref()
                .map(|d| d.extra_tls_roots())
                .unwrap_or_default(),
            ..BootOptions::default()
        };
        #[cfg(not(feature = "dev"))]
        let options = BootOptions::default();
        let prepared = boot::prepare(ShellClient::new(transport), options)
            .await
            .map_err(|e| e.to_string())?;
        log(&format!(
            "booted on the {} shell {} (contract {})",
            prepared.hello.platform, prepared.hello.shell, prepared.hello.contract
        ));
        #[cfg(feature = "dev")]
        if let Some(config) = &dev {
            crate::dev::seed(&prepared.core, config)
                .await
                .map_err(|e| format!("development server: {e}"))?;
        }
        #[cfg(not(feature = "dev"))]
        let _ = dev;
        let client = prepared.client;
        let host = ui::host::ShellHost::new(client.clone(), prepared.hello, page());
        #[cfg(feature = "dev")]
        let host = match dev.as_ref().and_then(|d| d.update_fixtures.clone()) {
            Some(url) => host.with_update_check(crate::dev::fixture_update_check(url)),
            None => host,
        };
        let host: std::rc::Rc<dyn ui::host::Host> = std::rc::Rc::new(host);
        let confirm: ui::app::Confirm = Box::new(move || {
            Box::pin(async move {
                boot::confirm(&client)
                    .await
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            })
        });
        Ok(ui::app::Booted {
            core: prepared.core,
            host,
            confirm,
        })
    });
    let languages = preferred_languages();
    // The single-threaded web platform, with the browser's own fonts allowed
    // as a fallback for CJK as well as emoji: the bundle's faces are Latin,
    // and a reader writing Chinese must see Chinese.
    let platform = std::rc::Rc::new(gpui_web::WebPlatform::new_with_backend_and_font_fallback(
        false,
        gpui_web::WebBackendPreference::Auto,
        gpui_web::CanvasFontFallback::EmojiAndCjk,
    ));
    gpui::Application::with_platform(platform)
        .with_assets(ui::Icons)
        .run(move |cx| {
            ui::init(cx);
            ui::open_window(boot, languages, cx);
        });
    Ok(())
}

/// The headless self-test (`crate::selftest`): `config` is the bundle's
/// `selftest.json`, `webview` what the page found in the webview's own
/// storage. Resolves to the report's JSON text.
#[cfg(feature = "self-test")]
#[wasm_bindgen(js_name = selfTest)]
pub async fn self_test(config: String, webview: String) -> Result<String, JsValue> {
    install_panic_hook();
    let transport = WebTransport::connect().map_err(|e| JsValue::from_str(&e))?;
    let env = crate::selftest::Env {
        now_ms: js_sys::Date::now,
        log: |line: &str| log(line),
        sleep_ms,
        webview,
    };
    crate::selftest::run(ShellClient::new(transport), &config, env)
        .await
        .map_err(|e| JsValue::from_str(&e))
}
