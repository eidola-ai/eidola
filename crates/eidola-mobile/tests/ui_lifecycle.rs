//! The interface's domain lifecycle under gpui's test app, over a real core
//! answering from an OpenAI-compatible mock: navigation never cancels a
//! conversation's work, a reader who comes back joins the running entity, and
//! a conversation reopened while its disposal is being decided is read once
//! the disposal keeps it; and linking an account is the account store's
//! operation, whose refused check rolls the credentials back with no screen
//! holding it.

use std::rc::Rc;
use std::time::Duration;

use eidola_app_core::AppCore;
use eidola_mobile::dev::{DevConfig, seed};
use eidola_mobile::ui::conversation::Conversation;
use eidola_mobile::ui::stores::Stores;
use gpui::{Entity, TestAppContext};
use wiremock::matchers::{body_partial_json, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// The mock model: a streamed answer held back by `delay`, so a test can act
/// while the turn is in flight; any unstreamed request (a title) answers at
/// once.
struct Model {
    server: MockServer,
    // The runtime the mock serves on, kept for the test's lifetime.
    _runtime: tokio::runtime::Runtime,
}

impl Model {
    fn start(delay: Duration) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let server = runtime.block_on(async {
            let server = MockServer::start().await;
            let chunk = |delta: serde_json::Value, finish: Option<&str>| {
                format!(
                    "data: {}\n\n",
                    serde_json::json!({
                        "id": "m", "object": "chat.completion.chunk", "created": 1, "model": "stub",
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    })
                )
            };
            let body = [
                chunk(serde_json::json!({"role": "assistant"}), None),
                chunk(
                    serde_json::json!({"reasoning_content": "Thinking it over."}),
                    None,
                ),
                chunk(serde_json::json!({"content": "An answer."}), None),
                chunk(serde_json::json!({}), Some("stop")),
                "data: [DONE]\n\n".to_string(),
            ]
            .concat();
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .and(body_partial_json(serde_json::json!({"stream": true})))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_raw(body, "text/event-stream")
                        .set_delay(delay),
                )
                .mount(&server)
                .await;
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id": "t", "object": "chat.completion", "created": 1, "model": "stub",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "A title"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })))
                .mount(&server)
                .await;
            server
        });
        Self {
            server,
            _runtime: runtime,
        }
    }
}

/// A host with no shell behind it: these tests reach none of its calls.
struct NoShell {
    hello: eidola_mobile::protocol::Hello,
}

impl NoShell {
    fn new() -> Self {
        Self {
            hello: eidola_mobile::protocol::Hello {
                contract: 2,
                platform: "test".into(),
                shell: "none".into(),
                methods: vec![],
            },
        }
    }
}

impl eidola_mobile::ui::host::Host for NoShell {
    fn hello(&self) -> &eidola_mobile::protocol::Hello {
        &self.hello
    }
    fn open_external(&self, _: String) -> Unanswered<()> {
        Box::pin(async { Err("no shell".into()) })
    }
    fn bundle_state(&self) -> Unanswered<eidola_app_core::updater::mobile::BundleState> {
        Box::pin(async { Err("no shell".into()) })
    }
    fn check_update(
        &self,
        _: Rc<AppCore>,
    ) -> futures_util::future::LocalBoxFuture<
        'static,
        Result<
            eidola_app_core::updater::mobile::BundleUpdate,
            eidola_app_core::updater::ReleaseCheckError,
        >,
    > {
        Box::pin(async {
            Err(eidola_app_core::updater::ReleaseCheckError::Unreachable(
                "no shell".into(),
            ))
        })
    }
    fn install(
        &self,
        _: Rc<AppCore>,
        _: eidola_app_core::updater::mobile::PendingBundleUpdate,
    ) -> futures_util::future::LocalBoxFuture<
        'static,
        Result<eidola_app_core::updater::mobile::ActivatedBundle, eidola_app_core::error::AppError>,
    > {
        unreachable!("no update is offered")
    }
    fn revert(
        &self,
    ) -> futures_util::future::LocalBoxFuture<
        'static,
        Result<eidola_app_core::updater::mobile::BundleState, eidola_app_core::error::AppError>,
    > {
        unreachable!("nothing reverts")
    }
    fn reload(&self) {}
    fn bundle_text(&self, _: &'static str) -> Unanswered<String> {
        Box::pin(async { Err("no shell".into()) })
    }
}

type Unanswered<T> = futures_util::future::LocalBoxFuture<'static, Result<T, String>>;

/// A real core over a temporary profile, its default agent on the mock.
fn stores(cx: &mut TestAppContext, model: &Model) -> (Stores, tempfile::TempDir) {
    // The core's runtime is real: declare it to the test scheduler.
    cx.executor().allow_parking();
    let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
    let dir = tempfile::tempdir().unwrap();
    let core = AppCore::new(dir.path().to_path_buf(), dir.path().join("data")).unwrap();
    let config = DevConfig {
        base_url: model.server.uri(),
        model: "stub".into(),
        appearance: None,
        tint: None,
        update_fixtures: None,
        eidola: None,
    };
    core.runtime().block_on(seed(&core, &config)).unwrap();
    let stores = cx.update(|cx| Stores::new(Rc::new(core), Rc::new(NoShell::new()), cx));
    (stores, dir)
}

fn wait_until(
    cx: &mut TestAppContext,
    what: &str,
    mut done: impl FnMut(&mut TestAppContext) -> bool,
) {
    for _ in 0..600 {
        cx.run_until_parked();
        if done(cx) {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("timed out waiting until {what}");
}

fn posts(conversation: &Entity<Conversation>, cx: &mut TestAppContext) -> usize {
    conversation.read_with(cx, |c, _| c.transcript().value().map_or(0, Vec::len))
}

/// A new conversation, committed.
fn created(stores: &Stores, cx: &mut TestAppContext) -> Entity<Conversation> {
    let conversation = stores.spaces.update(cx, |s, cx| s.create(cx));
    wait_until(cx, "the space is created", |cx| {
        conversation.read_with(cx, |c, _| !c.is_deciding())
    });
    conversation
}

/// Back while the post is still saving: the save's plan is followed and the
/// answer comes, though no view shows the conversation any more.
#[gpui::test]
fn leaving_while_a_post_saves_still_gets_it_answered(cx: &mut TestAppContext) {
    let model = Model::start(Duration::from_millis(200));
    let (stores, _dir) = stores(cx, &model);
    let conversation = created(&stores, cx);
    let id = conversation.read_with(cx, |c, _| c.id().to_string());

    // Send, and leave in the same breath: the save is in flight.
    stores.spaces.update(cx, |spaces, cx| {
        conversation.update(cx, |c, cx| assert!(c.submit("Hello".into(), None, cx)));
        assert!(conversation.read(cx).is_saving());
        spaces.left(&conversation, cx);
    });
    drop(conversation);

    // Its work settles and the store lets it go.
    wait_until(cx, "the left conversation settles", |cx| {
        stores.spaces.read_with(cx, |s, _| !s.is_held(&id))
    });
    // Reopened, it holds the post and its answer.
    let reopened = stores.spaces.update(cx, |s, cx| s.open(&id, cx));
    wait_until(cx, "the answer is read", |cx| posts(&reopened, cx) == 2);
}

/// Back while an answer streams, then back again: the reader joins the same
/// entity, its turn still running, and the answer lands there.
#[gpui::test]
fn coming_back_joins_the_answer_still_streaming(cx: &mut TestAppContext) {
    let model = Model::start(Duration::from_millis(1500));
    let (stores, _dir) = stores(cx, &model);
    let conversation = created(&stores, cx);
    let id = conversation.read_with(cx, |c, _| c.id().to_string());
    let entity = conversation.entity_id();

    conversation.update(cx, |c, cx| assert!(c.submit("Hello".into(), None, cx)));
    wait_until(cx, "the answer is streaming", |cx| {
        conversation.read_with(cx, |c, _| !c.turns().is_empty())
    });
    stores.spaces.update(cx, |s, cx| s.left(&conversation, cx));
    drop(conversation);
    cx.run_until_parked();
    assert!(stores.spaces.read_with(cx, |s, _| s.is_held(&id)));

    let again = stores.spaces.update(cx, |s, cx| s.open(&id, cx));
    assert_eq!(again.entity_id(), entity, "a second entity for one space");
    assert!(
        again.read_with(cx, |c, _| !c.turns().is_empty()),
        "the running turn was lost"
    );
    wait_until(cx, "the answer lands", |cx| {
        posts(&again, cx) == 2 && again.read_with(cx, |c, _| c.turns().is_empty())
    });
    wait_until(cx, "the hold is released", |cx| {
        stores.spaces.read_with(cx, |s, _| !s.is_held(&id))
    });
}

/// Reopened while the disposal of its last visit is still deciding, a space
/// with posts is read once the disposal keeps it.
#[gpui::test]
fn a_conversation_reopened_during_its_disposal_is_read_when_kept(cx: &mut TestAppContext) {
    let model = Model::start(Duration::from_millis(0));
    let (stores, _dir) = stores(cx, &model);
    let conversation = created(&stores, cx);
    let id = conversation.read_with(cx, |c, _| c.id().to_string());
    conversation.update(cx, |c, cx| assert!(c.submit("Hello".into(), None, cx)));
    wait_until(cx, "the exchange settles", |cx| {
        posts(&conversation, cx) == 2 && conversation.read_with(cx, |c, _| !c.is_busy())
    });

    // Leave (the disposal starts deciding) and come straight back.
    stores.spaces.update(cx, |s, cx| s.left(&conversation, cx));
    drop(conversation);
    let reopened = stores.spaces.update(cx, |s, cx| s.open(&id, cx));
    assert!(reopened.read_with(cx, |c, _| c.is_deciding()));
    wait_until(cx, "the kept conversation is read", |cx| {
        !reopened.read_with(cx, |c, _| c.is_deciding()) && posts(&reopened, cx) == 2
    });
}

/// A link whose balance check is refused, started with nothing but the store
/// holding it (the onboarding screen it came from may be gone): the
/// credentials it set are rolled back and the refusal waits for whoever looks.
#[gpui::test]
fn a_refused_link_is_rolled_back_with_no_screen_holding_it(cx: &mut TestAppContext) {
    let model = Model::start(Duration::ZERO);
    let (stores, _dir) = stores(cx, &model);
    let core = stores.core.clone();
    // Nothing answers here: the check fails.
    core.runtime()
        .block_on(core.set_base_url("https://127.0.0.1:1".into()))
        .unwrap();

    assert!(
        stores
            .account
            .update(cx, |a, cx| a.link("acct_test".into(), "secret".into(), cx))
    );
    // Neither another link nor a creation starts while it runs.
    assert!(stores.account.update(cx, |a, cx| !a.link(
        "acct_other".into(),
        "secret".into(),
        cx
    )));
    assert!(stores.account.update(cx, |a, cx| !a.create(Vec::new(), cx)));

    wait_until(cx, "the link settles", |cx| {
        stores.account.read_with(cx, |a, _| !a.linking())
    });
    assert!(
        stores
            .account
            .read_with(cx, |a, _| matches!(a.link_outcome(), Some(Err(_)))),
        "the refusal is not reported"
    );
    let config = core.config_state();
    assert!(
        !config.has_account && !config.has_account_secret,
        "a refused link left its credentials in the profile"
    );
}
