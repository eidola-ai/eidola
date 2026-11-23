//! The webview side: the bridge's transport over `js/bridge.js`, and the
//! bundle's entry point.

use std::cell::RefCell;
use std::collections::HashMap;

use tokio::sync::oneshot;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::boot::{self, BootOptions, Booted};
use crate::client::{BoxFuture, ShellClient, Transport};
use crate::protocol;

#[wasm_bindgen(module = "/js/bridge.js")]
extern "C" {
    fn platform() -> Option<String>;
    #[wasm_bindgen(js_name = postIos, catch)]
    fn post_ios(text: &str) -> Result<js_sys::Promise, JsValue>;
    #[wasm_bindgen(js_name = openAndroid, catch)]
    fn open_android(on_reply: &Closure<dyn FnMut(String)>) -> Result<(), JsValue>;
    #[wasm_bindgen(js_name = postAndroid, catch)]
    fn post_android(text: &str) -> Result<(), JsValue>;
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

thread_local! {
    /// The booted app, alive for the document's lifetime.
    static APP: RefCell<Option<Booted<WebTransport>>> = const { RefCell::new(None) };
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| log_error(&format!("panic: {info}"))));
}

/// Boot the app on its shell and confirm the bundle. Resolves to a line for
/// the page to show; rejects with the reason the boot failed.
#[wasm_bindgen]
pub async fn start() -> Result<String, JsValue> {
    install_panic_hook();
    let transport = WebTransport::connect().map_err(|e| JsValue::from_str(&e))?;
    let booted = boot::boot(ShellClient::new(transport), BootOptions::default())
        .await
        .map_err(|e| {
            let message = format!("boot failed: {e}");
            log_error(&message);
            JsValue::from_str(&message)
        })?;
    let status = format!(
        "booted on the {} shell {} (contract {}); bundle {} {}, {}",
        booted.hello.platform,
        booted.hello.shell,
        booted.hello.contract,
        booted.bundle.serving.version,
        booted
            .bundle
            .serving
            .digest
            .as_deref()
            .unwrap_or("(digest pending)"),
        if booted.bundle.confirmed {
            "confirmed"
        } else {
            "on trial"
        },
    );
    log(&status);
    APP.with(|app| *app.borrow_mut() = Some(booted));
    Ok(status)
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
