//! End-to-end tests for `AppCore::chat` / `chat_stream` against an in-process
//! mock upstream (see `chat_harness`). These pay down two waves of debt:
//!
//! * **Wave-3 failure-path bus emissions** — the exit-point table in
//!   `tests/bus.rs` was asserted-by-inspection only. Each typed-failure test
//!   here asserts BOTH the returned error AND the emitted `Change`s, turning a
//!   table row into an executed test.
//! * **Wave-1 auto-provisioning** — `ensure_spendable_credential` had only a
//!   pure-decision unit test; `auto_provisioning_*` here drives the empty-wallet
//!   + funded-account path all the way through a successful chat.
//!
//! Several tests also pin the **exact rendered upstream bytes** (roles +
//! `#<handle> · <label>` headers + the system message) via `chat_bodies()` —
//! see `single_agent_thread_renders_alternating_roles_with_headers`,
//! `branch_reply_sends_only_its_branch_context`,
//! `regenerate_sends_only_upstream_context_at_current_versions`, and
//! `model_emitted_header_is_stripped_before_persisting`. The multi-agent role
//! split lives in `tests/participants_orchestration.rs`.
//!
//! Determinism: the mock is in-process over loopback HTTP with no real network
//! and no attestation handshake (the `with_test_http_client` seam). The whole
//! suite runs in well under a second.

mod chat_harness;

use chat_harness::{
    ChatBehavior, DEFAULT_AGENT_LABEL, HUMAN_LABEL, MODEL, MockConfig, MockServer, RefundMode,
    Stamps, THREAD_MAP_NOTE, THREAD_MAP_TOOLS_NOTE, TRAILING_BLOCK_NOTE, flat_messages, map_entry,
    roster, system_message, system_message_with, thread_map, trailing, with_account,
};
use eidola_app_core::changes::{Change, ChangeEvent};
use eidola_app_core::error::AppError;
use eidola_app_core::{AppCore, ChatStreamEvent};

/// The generic system prompt the seeded default template agent carries (mirrors
/// `db::DEFAULT_AGENT_SYSTEM_PROMPT`). Participant-aware turns prepend it as the
/// leading `system` message, so context assertions include it (Participants v1).
const SEEDED_SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer clearly and concisely.";

// ---------------------------------------------------------------------------
// Harness: run an async test body on a dedicated OS thread.
//
// `AppCore` owns its own multi-thread tokio runtime; dropping it while another
// runtime is active on the same thread panics. We run each test body — and the
// `AppCore` Drop — on a plain OS thread that builds the mock + core inside its
// own runtime (`AppCore::runtime().block_on`). Mirrors `tests/bus.rs`.
// ---------------------------------------------------------------------------

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// Drain all currently-available bus messages (non-blocking).
fn drain(rx: &mut tokio::sync::broadcast::Receiver<ChangeEvent>) -> Vec<Change> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(c) => out.push(c.change),
            Err(tokio::sync::broadcast::error::TryRecvError::Empty) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Closed) => break,
            Err(tokio::sync::broadcast::error::TryRecvError::Lagged(n)) => {
                panic!("test receiver lagged by {n}");
            }
        }
    }
    out
}

fn space_changes(changes: &[Change]) -> Vec<&Change> {
    changes
        .iter()
        .filter(|c| matches!(c, Change::Space(_)))
        .collect()
}

/// The id of the one space this core holds — what a `chat`/`chat_stream` with
/// no space id created on its way to a failure. Panics if the space was not
/// persisted, which is the property these failure tests are asserting: the
/// saved thought survives the turn that failed around it.
fn only_space(core: &AppCore) -> String {
    let spaces = core
        .runtime()
        .block_on(core.list_spaces(false))
        .expect("list spaces");
    assert_eq!(
        spaces.len(),
        1,
        "expected exactly one space to have been created"
    );
    spaces[0].id.clone()
}

/// The most recently active space — [`only_space`] for a core that has more
/// than one, `list_spaces` being ordered by `last_activity_at DESC`.
fn latest_space(core: &AppCore) -> String {
    let spaces = core
        .runtime()
        .block_on(core.list_spaces(false))
        .expect("list spaces");
    spaces.first().expect("a space was created").id.clone()
}

/// Build a mock + a core wired to it (see `chat_harness::core_for`). Callers
/// add an account via `with_account` when they want the auto-provisioning path.
fn setup(config: MockConfig) -> (MockServer, AppCore, tempfile::TempDir) {
    chat_harness::core_for(config)
}

/// The shared constructor is the only outer request shape app-core may send.
/// Read every input from the harness-captured body, so a field added at either
/// dispatch call site makes this exact-body comparison fail.
fn assert_shared_request_body(body: &serde_json::Value, expected_stream: bool) {
    let model = body["model"].as_str().expect("wire model");
    let messages = body["messages"].as_array().expect("messages array");
    let max_completion_tokens = u32::try_from(
        body["max_completion_tokens"]
            .as_u64()
            .expect("completion-token limit"),
    )
    .expect("completion-token limit fits u32");
    let tools = body
        .get("tools")
        .and_then(|tools| tools.as_array())
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if expected_stream {
        assert_eq!(
            body.get("stream").and_then(|stream| stream.as_bool()),
            Some(true),
            "streaming dispatch must set `stream: true`"
        );
    } else {
        assert!(
            body.get("stream").is_none(),
            "blocking dispatch must omit `stream`"
        );
    }
    let include_usage = body
        .get("stream_options")
        .and_then(|options| options.get("include_usage"))
        .and_then(|include_usage| include_usage.as_bool())
        == Some(true);

    assert_eq!(
        body,
        &eidola_common::chat_completion_request_body(
            model,
            messages,
            max_completion_tokens,
            tools,
            expected_stream,
            include_usage,
        ),
        "dispatch must send exactly the shared chat request body"
    );
    eidola_server::types::test_chat_completion_request_is_accepted(body.clone())
        .expect("captured body must satisfy the server's strict request type");
}

// ===========================================================================
// Happy path — blocking chat
// ===========================================================================

#[test]
fn blocking_chat_persists_and_emits() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("How do tides work?".into(), MODEL.into(), None))
            .expect("chat should succeed");

        // Returned usage + charge.
        assert_eq!(result.input_tokens, Some(11));
        assert_eq!(result.output_tokens, Some(5));
        assert!(result.credits_charged > 0);
        assert_eq!(result.content, "Hello from the mock.");

        let changes = drain(&mut rx);
        // SpaceIndex (new space + auto-title), Space(id), Wallet, Record all
        // emitted on success. Wallet appears twice: once at spend start, once
        // on final success.
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(
            changes.contains(&Change::Space(result.space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(changes.contains(&Change::Record), "got {changes:?}");

        // Persistence: space, user + assistant turns, request row.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(result.space_id.clone()))
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "How do tides work?");
        assert_eq!(messages[1].role, "assistant");
        assert_eq!(messages[1].content, "Hello from the mock.");

        let spaces = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");
        assert_eq!(spaces.len(), 1);
        assert_eq!(spaces[0].title.as_deref(), Some("How do tides work?"));

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].path, "/v1/chat/completions");
        assert_eq!(requests[0].response_status, Some(200));

        // The inline refund recovered a successor credential — no recovery hit.
        assert_eq!(mock.refund_hits(), 0);
        assert_eq!(mock.chat_hits(), 1);

        // Participants v1: the fresh chat space was instantiated from the
        // default template, so it carries the shared human "You" and the
        // template's agent (model = MODEL = DEFAULT_MODEL) from birth.
        let participants = core
            .runtime()
            .block_on(core.list_space_participants(result.space_id.clone()))
            .expect("participants");
        assert_eq!(
            participants.len(),
            2,
            "the shared human + one agent; got {participants:?}"
        );
        let human = participants
            .iter()
            .find(|p| p.kind == "human")
            .expect("human participant");
        assert_eq!(human.label, "User");
        let agent = participants
            .iter()
            .find(|p| p.kind == "agent")
            .expect("agent participant");
        assert_eq!(agent.model_ref.as_deref(), Some(MODEL));

        // The persisted actions carry real participant identities: the user
        // turn is the human, the inference is the responding agent.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(result.space_id.clone()))
            .expect("tree");
        let user_post = tree
            .iter()
            .find(|n| n.action_type == "user_input")
            .expect("user post");
        assert_eq!(user_post.participant.kind, "human");
        assert_eq!(user_post.participant.label, "User");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("inference post");
        assert_eq!(inference.participant.kind, "agent");
        assert!(!inference.action_id.is_empty());
    });
}

/// **The eagerly-created space's first message is an ordinary post.**
///
/// A client that creates the space when its window opens
/// (`create_space_with_id`) then sends into a space that already exists, so
/// the send path instantiates nothing: exactly one space before and after, the
/// row is the one the client named, and the only listing change the turn
/// causes is the auto-title. This is the property that lets the GUI address a
/// new conversation by a real id from its first frame.
#[test]
fn a_first_message_into_a_pre_created_space_instantiates_nothing() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        let space_id = eidola_app_core::new_space_id();
        core.runtime()
            .block_on(core.create_space_with_id(space_id.clone(), None))
            .expect("the space is created when its window opens");
        assert_eq!(only_space(&core), space_id);

        let mut rx = core.subscribe_changes();
        let result = core
            .runtime()
            .block_on(core.chat(
                "First question".into(),
                MODEL.into(),
                Some(space_id.clone()),
            ))
            .expect("the first message is an ordinary post into an existing space");
        assert_eq!(result.space_id, space_id, "no second space was minted");
        assert_eq!(
            only_space(&core),
            space_id,
            "the send path instantiates nothing"
        );

        // `SpaceIndex` here is the auto-title, not a creation: the space was
        // already in the listing before the turn ran.
        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id.clone())));
        let titled = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");
        assert_eq!(titled[0].title.as_deref(), Some("First question"));

        // One user turn and one reply, in the space the client named.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "user");
    });
}

#[test]
fn blocking_chat_into_existing_space_does_not_emit_space_index_again() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        // First turn creates + auto-titles the space.
        let first = core
            .runtime()
            .block_on(core.chat("First question".into(), MODEL.into(), None))
            .expect("first chat");

        let mut rx = core.subscribe_changes();
        // Second turn into the same (titled) space.
        let second = core
            .runtime()
            .block_on(core.chat(
                "Second question".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
            ))
            .expect("second chat");
        assert_eq!(second.space_id, first.space_id);

        let changes = drain(&mut rx);
        // Not a new space and not auto-titled → no SpaceIndex this time.
        assert!(
            !changes.contains(&Change::SpaceIndex),
            "second turn into a titled space must not emit SpaceIndex; got {changes:?}"
        );
        assert!(changes.contains(&Change::Space(first.space_id.clone())));
        assert!(changes.contains(&Change::Record));

        // Four messages now (2 turns × user+assistant).
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(first.space_id))
            .expect("messages");
        assert_eq!(messages.len(), 4);
    });
}

#[test]
fn blocking_chat_recovers_refund_when_no_inline_refund() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkBlockingNoInlineRefund,
            ..MockConfig::default()
        });
        with_account(&core);

        let result = core
            .runtime()
            .block_on(core.chat("hello".into(), MODEL.into(), None))
            .expect("chat should succeed");
        assert_eq!(result.content, "Hello from the mock.");

        // No inline refund → the recovery endpoint was consulted.
        assert!(mock.refund_hits() >= 1);

        // A successor credential exists and is active/spendable.
        let creds = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet");
        assert!(
            !creds.is_empty(),
            "a recovered successor credential should be active"
        );
    });
}

// ===========================================================================
// Happy path — streaming chat
// ===========================================================================

#[test]
fn streaming_chat_delivers_deltas_and_persists() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();

        let result = core.runtime().block_on(async {
            // Collect events concurrently with the stream completing.
            let collector = async {
                let mut content = String::new();
                let mut reasoning = String::new();
                while let Some(ev) = events_rx.recv().await {
                    match ev {
                        ChatStreamEvent::ContentDelta(t) => content.push_str(&t),
                        ChatStreamEvent::ReasoningDelta(t) => reasoning.push_str(&t),
                    }
                }
                (content, reasoning)
            };
            let chat = core.chat_stream("stream me".into(), MODEL.into(), None, tx);
            let (res, (content, reasoning)) = tokio::join!(chat, collector);
            (res, content, reasoning)
        });

        let (res, content, reasoning) = result;
        let res = res.expect("stream should complete");

        assert_eq!(content, "Hello from the stream.");
        assert_eq!(reasoning, "thinking…");
        assert_eq!(res.content, "Hello from the stream.");
        assert_eq!(res.input_tokens, Some(11));
        assert_eq!(res.output_tokens, Some(5));

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Space(res.space_id.clone())));
        assert!(changes.contains(&Change::Wallet));
        assert!(changes.contains(&Change::Record));

        // Streaming always goes through the recovery endpoint for its refund.
        assert!(mock.refund_hits() >= 1);

        let messages = core
            .runtime()
            .block_on(core.get_space_messages(res.space_id))
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content, "Hello from the stream.");
    });
}

#[test]
fn blocking_and_streaming_dispatch_use_the_shared_request_body() {
    run(|| {
        {
            let (mock, core, _dir) = setup(MockConfig::default());
            with_account(&core);

            core.runtime()
                .block_on(core.chat("blocking body".into(), MODEL.into(), None))
                .expect("blocking chat succeeds");

            let body = mock.chat_bodies().pop().expect("blocking request");
            assert_shared_request_body(&body, false);
        }

        {
            let (mock, core, _dir) = setup(MockConfig {
                chat: ChatBehavior::OkStreaming,
                ..MockConfig::default()
            });
            with_account(&core);
            let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();

            let result = core.runtime().block_on(async {
                let drain = async { while events_rx.recv().await.is_some() {} };
                let (result, ()) = tokio::join!(
                    core.chat_stream("streaming body".into(), MODEL.into(), None, tx),
                    drain,
                );
                result
            });
            result.expect("streaming chat succeeds");

            let body = mock.chat_bodies().pop().expect("streaming request");
            assert_shared_request_body(&body, true);
        }

        {
            let (mock, core, _dir) = setup(MockConfig {
                chat: ChatBehavior::ToolRoundsBlocking(1),
                ..MockConfig::default()
            });
            with_account(&core);
            with_echo_tool(&core);

            core.runtime()
                .block_on(core.chat("blocking tool body".into(), MODEL.into(), None))
                .expect("blocking tool chat succeeds");

            let body = mock
                .chat_bodies()
                .pop()
                .expect("blocking tool follow-up request");
            assert_shared_request_body(&body, false);
        }

        {
            let (mock, core, _dir) = setup(MockConfig {
                chat: ChatBehavior::ToolRoundsStreaming(1),
                ..MockConfig::default()
            });
            with_account(&core);
            with_echo_tool(&core);
            let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();

            let result = core.runtime().block_on(async {
                let drain = async { while events_rx.recv().await.is_some() {} };
                let (result, ()) = tokio::join!(
                    core.chat_stream("streaming tool body".into(), MODEL.into(), None, tx),
                    drain,
                );
                result
            });
            result.expect("streaming tool chat succeeds");

            let body = mock
                .chat_bodies()
                .pop()
                .expect("streaming tool follow-up request");
            assert_shared_request_body(&body, true);
        }
    });
}

/// A streamed `ReasoningDelta` is **persisted** as a `thinking` content block
/// on the inference, ahead of its `text` block — so reopening a space still
/// shows the thinking disclosure instead of losing it with the process.
/// It must **not** leak into the readable transcript or the upstream context:
/// both context queries join only `text` blocks.
#[test]
fn streamed_reasoning_persists_as_a_thinking_block() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let res = core.runtime().block_on(async {
            let drainer = async { while events_rx.recv().await.is_some() {} };
            let chat = core.chat_stream("stream me".into(), MODEL.into(), None, tx);
            let (res, ()) = tokio::join!(chat, drainer);
            res
        });
        let res = res.expect("stream should complete");

        // Read it back the way the GUI does — through the post tree.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(res.space_id.clone()))
            .expect("space tree");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("the persisted response");

        let kinds: Vec<&str> = inference
            .blocks
            .iter()
            .map(|b| b.block_type.as_str())
            .collect();
        assert_eq!(
            kinds,
            vec!["thinking", "text"],
            "thinking is persisted before the answer"
        );
        assert_eq!(inference.blocks[0].text.as_deref(), Some("thinking…"));
        assert_eq!(
            inference.blocks[1].text.as_deref(),
            Some("Hello from the stream.")
        );

        // The flat transcript (and therefore the upstream context, which shares
        // the text-only block filter) sees the answer, never the thinking.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(res.space_id))
            .expect("messages");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content, "Hello from the stream.");
    });
}

/// The blocking transport recovers reasoning from the aggregated
/// `message.reasoning_content` and persists it identically — the two
/// transports must not disagree about what a reopened space shows.
#[test]
fn blocking_reasoning_persists_as_a_thinking_block() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        let res = core
            .runtime()
            .block_on(core.chat("hello".into(), MODEL.into(), None))
            .expect("chat should succeed");

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(res.space_id))
            .expect("space tree");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("the persisted response");
        let kinds: Vec<&str> = inference
            .blocks
            .iter()
            .map(|b| b.block_type.as_str())
            .collect();
        assert_eq!(kinds, vec!["thinking", "text"]);
        assert_eq!(inference.blocks[0].text.as_deref(), Some("thinking…"));
    });
}

/// A persisted `thinking` block never reaches the model: the next turn's
/// upstream context carries the prior answer's text and nothing else.
#[test]
fn persisted_thinking_is_not_sent_upstream() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let (tx, mut rx1) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let first = core.runtime().block_on(async {
            let drainer = async { while rx1.recv().await.is_some() {} };
            let chat = core.chat_stream("first".into(), MODEL.into(), None, tx);
            let (res, ()) = tokio::join!(chat, drainer);
            res
        });
        let space_id = first.expect("first turn").space_id;

        let (tx2, mut rx2) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let second = core.runtime().block_on(async {
            let drainer = async { while rx2.recv().await.is_some() {} };
            let chat = core.chat_stream("second".into(), MODEL.into(), Some(space_id.clone()), tx2);
            let (res, ()) = tokio::join!(chat, drainer);
            res
        });
        second.expect("second turn");

        let body = mock
            .chat_bodies()
            .last()
            .cloned()
            .expect("the second turn's request body");
        let text = body.to_string();
        assert!(
            text.contains("Hello from the stream."),
            "the prior answer is in context: {text}"
        );
        assert!(
            !text.contains("thinking…"),
            "the prior turn's thinking must never be replayed upstream: {text}"
        );
    });
}

// ===========================================================================
// Auto-provisioning (wave-1 debt)
// ===========================================================================

#[test]
fn auto_provisioning_empty_wallet_funded_account_succeeds() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        // Wallet starts empty.
        let before = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet");
        assert!(before.is_empty(), "wallet should start empty");

        let mut rx = core.subscribe_changes();
        let result = core
            .runtime()
            .block_on(core.chat("provision me".into(), MODEL.into(), None))
            .expect("chat should auto-provision and succeed");
        assert_eq!(result.content, "Hello from the mock.");

        // Allocation emits Wallet + Account transparently; chat then emits its
        // own Space/Wallet/Record. Account is only emitted by the allocate path.
        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Account),
            "auto-allocation should emit Account; got {changes:?}"
        );
        assert!(changes.contains(&Change::Wallet));

        // After the spend, a successor credential remains.
        let after = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet");
        assert!(
            !after.is_empty(),
            "a successor credential should remain after the spend"
        );
    });
}

// ===========================================================================
// Typed failure: pre-space errors leave zero durable trace, emit nothing
// ===========================================================================

// ===========================================================================
// Regenerate (Revise mode) — a new agent generation of an inference's item
// ===========================================================================

#[test]
fn regenerate_replaces_answer_with_a_new_generation() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        let result = core
            .runtime()
            .block_on(core.chat("How do tides work?".into(), MODEL.into(), None))
            .expect("chat should succeed");

        // The inference action id is reachable via the Record (spend trail:
        // credential → request → action).
        let trail = core
            .runtime()
            .block_on(core.spend_trail(10, 0))
            .expect("spend trail");
        let inference_action_id = trail
            .iter()
            .find_map(|e| e.action_id.clone())
            .expect("spend trail carries the inference action id");

        // Regenerate: a new generation of the SAME inference item (Revise),
        // a second real spend.
        let regen = core
            .runtime()
            .block_on(core.regenerate(inference_action_id, MODEL.into()))
            .expect("regenerate should succeed");
        assert_eq!(regen.space_id, result.space_id);
        assert!(regen.credits_charged > 0);

        // The default view shows the regenerated answer in place — still two
        // messages (user + current answer), not three: Revise replaces in the
        // default view, where Reply would have appended a sibling.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(result.space_id))
            .expect("messages");
        assert_eq!(
            messages.len(),
            2,
            "regenerate replaces the answer, not appends; got {messages:?}"
        );
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[1].role, "assistant");

        // Two model calls → two inference requests recorded (each generation is
        // its own costed inference).
        assert_eq!(mock.chat_hits(), 2);
        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 2, "two inference requests recorded");
    });
}

#[test]
fn regenerate_sends_only_upstream_context_at_current_versions() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        // Two full turns: u1 -> i1 -> u2 -> i2.
        let first = core
            .runtime()
            .block_on(core.chat("How do tides work?".into(), MODEL.into(), None))
            .expect("first chat");
        core.runtime()
            .block_on(core.chat(
                "And why two per day?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
            ))
            .expect("second chat");

        // Locate the first user post and the FIRST inference via the tree.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(first.space_id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 4, "u1, i1, u2, i2; got {tree:#?}");
        let u1 = tree[0].action_id.clone();
        let u1_item = tree[0].item_id.clone();
        assert_eq!(tree[1].action_type, "inference");
        let i1 = tree[1].action_id.clone();

        // The header stamp names the post's *current generation*, so the
        // pre-edit bytes need the pre-edit snapshot.
        let before = Stamps::of(&core, &first.space_id);

        // Edit the upstream question, then regenerate the FIRST answer.
        core.runtime()
            .block_on(core.edit_post(u1, "How do tides work? Explain it for a sailor.".into()))
            .expect("edit");
        core.runtime()
            .block_on(core.regenerate(i1, MODEL.into()))
            .expect("regenerate");

        // The regenerate call (third request) must see ONLY its upstream
        // thread — the edited question at its most recent version — never the
        // downstream turn (u2/i2) or its own prior output (i1). Rendered bytes
        // are pinned: system prompt + protocol note, then the edited post under
        // its human author's header.
        let after = Stamps::of(&core, &first.space_id);
        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 3, "two chats + one regenerate");
        assert_eq!(
            flat_messages(&bodies[2]),
            vec![
                (
                    "system".to_string(),
                    system_message(Some(SEEDED_SYSTEM_PROMPT), DEFAULT_AGENT_LABEL)
                ),
                (
                    "user".to_string(),
                    after.headed(
                        &u1_item,
                        HUMAN_LABEL,
                        "How do tides work? Explain it for a sailor."
                    )
                )
            ],
            "regenerate context = system prompt + upstream only, most-recent versions"
        );

        // The handle is derived from the ITEM id, so the edit did not change
        // it: the bytes the model already read keep naming the same post. The
        // *stamp* does move, because an edit is a new generation and the stamp
        // dates the text it heads — the honest answer, and still byte-stable:
        // it moves only when the body it heads moves anyway.
        assert_eq!(
            flat_messages(&bodies[0])[1].1,
            before.headed(&u1_item, HUMAN_LABEL, "How do tides work?"),
            "the pre-edit rendering carries the same handle"
        );

        // Sanity: the second chat (a Reply on the linear spine) saw the full
        // thread — its target's inclusive ancestry is the whole conversation —
        // plus the leading system message (u1, i1, u2 + system = 4).
        let second_messages = bodies[1]["messages"].as_array().expect("messages").len();
        assert_eq!(
            second_messages, 4,
            "a linear reply sees the full thread + system prompt"
        );
    });
}

/// Quoted references: a post carrying a `{{ embed N }}` marker sends the
/// referenced passage upstream **attributed** — handle, author, annotation —
/// in that post's message, and a reference the body never embedded rides in a
/// trailing footnote block rather than reaching the model as nothing at all.
/// Unmapped markers stay literal (honest degradation, mirroring the editor).
#[test]
fn upstream_context_expands_embed_markers_into_quotes() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // A source post, then a post quoting a range of its text block.
        let source = core
            .runtime()
            .block_on(core.post(
                "The mitochondria is the powerhouse of the cell".into(),
                None,
            ))
            .expect("source post");
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(source.space_id.clone()))
            .expect("tree");
        let block_id = tree[0].blocks[0].id.clone();
        // The body carries a structural marker (expands), an unmapped marker
        // (stays literal), and a fence-defused marker of the SAME mapped
        // ordinal (stays literal — the editor renders it literal, so the
        // wire must too; expansion is structural, not line-based). Reference 2
        // has no marker anywhere — the human reads it in the footnote rail, so
        // the model must read it too.
        let posted = core
            .runtime()
            .block_on(
                core.post_with_references(
                    "What does this mean?\n\n{{ embed 1 }}\n\nAnd {{ embed 9 }} is unmapped.\n\n\
                 ```\n\n{{ embed 1 }}\n\n```"
                        .into(),
                    Some(source.space_id.clone()),
                    None,
                    vec![
                        eidola_app_core::ReferenceSpec {
                            antecedent_action_id: source.action_id.clone(),
                            content_block_id: Some(block_id.clone()),
                            range_start: Some(24),
                            range_end: Some(34), // "powerhouse"
                            annotation: None,
                        },
                        eidola_app_core::ReferenceSpec {
                            antecedent_action_id: source.action_id.clone(),
                            content_block_id: Some(block_id),
                            range_start: Some(4),
                            range_end: Some(16), // "mitochondria"
                            annotation: Some("the organelle itself".into()),
                        },
                    ],
                ),
            )
            .expect("post with reference");

        // Request a response to the quoting post.
        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(async {
                let collector = async { while events_rx.recv().await.is_some() {} };
                let respond = core.respond_stream(
                    posted.space_id.clone(),
                    MODEL.into(),
                    posted.action_id.clone(),
                    tx,
                );
                let (res, ()) = tokio::join!(respond, collector);
                res
            })
            .expect("respond_stream should succeed");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 1);
        let contents: Vec<String> = bodies[0]["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .map(|m| m["content"].as_str().unwrap_or_default().to_string())
            .collect();
        // system + u1 + u2 (the quoting post, expanded).
        assert_eq!(
            contents.len(),
            3,
            "system + two user turns; got {contents:?}"
        );
        let quoted = format!(
            "#{} · {HUMAN_LABEL}",
            eidola_app_core::post_handle(&source.item_id)
        );
        let stamps = Stamps::of(&core, &posted.space_id);
        assert_eq!(
            contents[2],
            stamps.headed(
                &posted.item_id,
                HUMAN_LABEL,
                &format!(
                    "What does this mean?\n\n[1] {quoted}\n> powerhouse\n\n\
                     And {{{{ embed 9 }}}} is unmapped.\n\n\
                     ```\n\n{{{{ embed 1 }}}}\n\n```\n\n\
                     Passages this post quotes:\n\
                     [2] {quoted} — the organelle itself\n> mitochondria"
                )
            ),
            "the embedded passage expands attributed, in place; the un-embedded one is \
             footnoted; unmapped and fence-defused markers go upstream literal"
        );
    });
}

/// The second route to a marker-less reference (the first is a draft whose
/// marker was deleted): `edit_post` replicates every surviving reference edge
/// onto the new generation without consulting the new body. The human keeps
/// reading the passage in the footnote rail, so the model keeps receiving it —
/// as a footnote, since the edited body no longer embeds it.
#[test]
fn a_reference_whose_marker_an_edit_removed_still_reaches_the_model() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let source = core
            .runtime()
            .block_on(core.post("Tides come from the moon's pull.".into(), None))
            .expect("source post");
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(source.space_id.clone()))
            .expect("tree");
        let block_id = tree[0].blocks[0].id.clone();
        let quoting = core
            .runtime()
            .block_on(core.post_with_references(
                "Is this right?\n\n{{ embed 1 }}".into(),
                Some(source.space_id.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: source.action_id.clone(),
                    content_block_id: Some(block_id),
                    range_start: Some(20),
                    range_end: Some(31), // "moon's pull"
                    annotation: None,
                }],
            ))
            .expect("post with reference");

        // The edit drops the marker and keeps the edge.
        let edited = core
            .runtime()
            .block_on(core.edit_post(quoting.action_id.clone(), "Is this right?".into()))
            .expect("edit");

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(async {
                let collector = async { while events_rx.recv().await.is_some() {} };
                let respond = core.respond_stream(
                    edited.space_id.clone(),
                    MODEL.into(),
                    edited.action_id.clone(),
                    tx,
                );
                let (res, ()) = tokio::join!(respond, collector);
                res
            })
            .expect("respond_stream should succeed");

        let bodies = mock.chat_bodies();
        let last = flat_messages(&bodies[0]).pop().expect("the edited post");
        let stamps = Stamps::of(&core, &edited.space_id);
        assert_eq!(
            last.1,
            stamps.headed(
                &edited.item_id,
                HUMAN_LABEL,
                &format!(
                    "Is this right?\n\nPassages this post quotes:\n[1] #{} · {HUMAN_LABEL}\n\
                     > moon's pull",
                    eidola_app_core::post_handle(&source.item_id)
                )
            ),
            "an edge the edited body no longer embeds is footnoted, not dropped"
        );
    });
}

/// A range-less reference is a **backlink** — a pointer to a post, not a quote
/// of one (`ReferenceSpec`'s range fields are "both present or both absent").
/// It has no passage to stand in for a marker, and reporting it as a quote
/// whose range broke would be a plain untruth about an ordinary edge.
#[test]
fn a_reference_that_names_no_range_reads_as_a_backlink_not_a_broken_quote() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let source = core
            .runtime()
            .block_on(core.post("Tides come from the moon's pull.".into(), None))
            .expect("source post");
        let posted = core
            .runtime()
            .block_on(core.post_with_references(
                "See also.\n\n{{ embed 1 }}".into(),
                Some(source.space_id.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: source.action_id.clone(),
                    content_block_id: None,
                    range_start: None,
                    range_end: None,
                    annotation: Some("for context".into()),
                }],
            ))
            .expect("a range-less reference is a backlink, and is allowed");

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(async {
                let collector = async { while events_rx.recv().await.is_some() {} };
                let respond = core.respond_stream(
                    posted.space_id.clone(),
                    MODEL.into(),
                    posted.action_id.clone(),
                    tx,
                );
                let (res, ()) = tokio::join!(respond, collector);
                res
            })
            .expect("respond_stream should succeed");

        let last = flat_messages(&mock.chat_bodies()[0])
            .pop()
            .expect("the quoting post");
        let stamps = Stamps::of(&core, &posted.space_id);
        assert_eq!(
            last.1,
            stamps.headed(
                &posted.item_id,
                HUMAN_LABEL,
                &format!(
                    "See also.\n\n{{{{ embed 1 }}}}\n\n\
                     Passages this post quotes:\n\
                     [1] #{} · {HUMAN_LABEL} — for context\n\
                     (referenced without quoting a passage)",
                    eidola_app_core::post_handle(&source.item_id)
                )
            ),
            "a backlink names its target and says it quoted nothing"
        );
    });
}

/// A per-space label overridden to **empty** is a documented state — on an
/// override column `NULL` inherits and `''` means "override to empty", and
/// `set_space_participant_override` deliberately allows it. Every byline has to
/// survive it: the quoted post keeps its handle, with nothing dangling after
/// the header separator.
#[test]
fn a_quote_whose_author_has_no_name_here_keeps_a_clean_handle() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let source = core
            .runtime()
            .block_on(core.post("Tides come from the moon's pull.".into(), None))
            .expect("source post");
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(source.space_id.clone()))
            .expect("tree");
        let posted = core
            .runtime()
            .block_on(core.post_with_references(
                "Is this right?\n\n{{ embed 1 }}".into(),
                Some(source.space_id.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: source.action_id.clone(),
                    content_block_id: Some(tree[0].blocks[0].id.clone()),
                    range_start: Some(20),
                    range_end: Some(31), // "moon's pull"
                    annotation: None,
                }],
            ))
            .expect("post with reference");

        core.runtime()
            .block_on(core.set_space_participant_override(
                source.space_id.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                eidola_app_core::ParticipantOverride {
                    label: Some(Some(String::new())),
                    model_ref: None,
                    system_prompt: None,
                    notify_policy: None,
                },
            ))
            .expect("an empty override is a real state");

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(async {
                let collector = async { while events_rx.recv().await.is_some() {} };
                let respond = core.respond_stream(
                    posted.space_id.clone(),
                    MODEL.into(),
                    posted.action_id.clone(),
                    tx,
                );
                let (res, ()) = tokio::join!(respond, collector);
                res
            })
            .expect("respond_stream should succeed");

        let sent = flat_messages(&mock.chat_bodies()[0])
            .into_iter()
            .map(|(_, c)| c)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            sent.contains(&format!(
                "[1] #{}\n> moon's pull",
                eidola_app_core::post_handle(&source.item_id)
            )),
            "the handle stands alone when this space gives its author no name: {sent}"
        );
        assert!(
            !sent.contains(&format!(
                "[1] #{} · ",
                eidola_app_core::post_handle(&source.item_id)
            )),
            "a byline never trails the header separator into nothing: {sent}"
        );
    });
}

/// A quote names a **concrete generation**. Once that generation is superseded,
/// the item's handle opens the *tip* — different text under the byline of the
/// excerpt beside it — so the handle is withheld and the quote renders as an
/// earlier version, the same shape a cross-space quote takes. An address that
/// resolves to something other than what the model is reading is worse than no
/// address.
#[test]
fn a_quote_of_a_since_edited_post_is_not_offered_its_handle() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let source = core
            .runtime()
            .block_on(core.post("Tides come from the moon's pull.".into(), None))
            .expect("source post");
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(source.space_id.clone()))
            .expect("tree");
        let quoting = core
            .runtime()
            .block_on(core.post_with_references(
                "Is this right?\n\n{{ embed 1 }}".into(),
                Some(source.space_id.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: source.action_id.clone(),
                    content_block_id: Some(tree[0].blocks[0].id.clone()),
                    range_start: Some(20),
                    range_end: Some(31), // "moon's pull"
                    annotation: None,
                }],
            ))
            .expect("post with reference");

        // The quoted post is edited: the passage stays what it was, the handle
        // now opens something else.
        core.runtime()
            .block_on(core.edit_post(
                source.action_id.clone(),
                "Tides come from the sun as well.".into(),
            ))
            .expect("edit");

        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(async {
                let collector = async { while events_rx.recv().await.is_some() {} };
                let respond = core.respond_stream(
                    quoting.space_id.clone(),
                    MODEL.into(),
                    quoting.action_id.clone(),
                    tx,
                );
                let (res, ()) = tokio::join!(respond, collector);
                res
            })
            .expect("respond_stream should succeed");

        let sent = flat_messages(&mock.chat_bodies()[0])
            .into_iter()
            .map(|(_, c)| c)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            sent.contains(&format!(
                "[1] {HUMAN_LABEL} (a post outside this space, or an earlier version)\n\
                 > moon's pull"
            )),
            "a superseded generation is named as one, not addressed: {sent}"
        );
        assert!(
            !sent.contains(&format!(
                "[1] #{} · ",
                eidola_app_core::post_handle(&source.item_id)
            )),
            "the item handle opens the edited text, so it may not be offered: {sent}"
        );
    });
}

#[test]
fn branch_reply_sends_only_its_branch_context() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // A linear spine, streamed: u1 -> i1 -> u2 -> i2.
        let (tx, _rx1) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let first = core
            .runtime()
            .block_on(core.chat_stream("How do tides work?".into(), MODEL.into(), None, tx))
            .expect("first turn");
        let (tx, _rx2) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(core.chat_stream(
                "And why two per day?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
                tx,
            ))
            .expect("second turn");

        // Branch off the FIRST answer (u2 already replies to i1, so this
        // reply forks the thread there).
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(first.space_id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 4, "u1, i1, u2, i2; got {tree:#?}");
        assert_eq!(tree[1].action_type, "inference");
        let u1_item = tree[0].item_id.clone();
        let i1_item = tree[1].item_id.clone();
        let i1 = tree[1].action_id.clone();

        let (tx, _rx3) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(core.chat_stream_reply(
                "What about spring tides?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
                Some(i1),
                tx,
            ))
            .expect("branch reply");
        let branch_item = item_id_of(&core, &first.space_id, "What about spring tides?");

        // The branch ask (third request) must see ONLY its own branch: the
        // ancestry of the new post — never the sibling turn (u2/i2). Exact
        // bytes: every message headed, and the responding agent's own prior
        // answer (and only it) rendered `assistant`.
        //
        // The sibling turn is *named* rather than shown: the space now branches
        // at i1, so the turn carries a trailing thread map (task 21). That is
        // the whole point — the model learns the branch exists without its
        // bytes entering the trunk.
        let u2_item = item_id_of(&core, &first.space_id, "And why two per day?");
        let stamps = Stamps::of(&core, &first.space_id);
        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 3, "two spine turns + one branch reply");
        assert_eq!(
            flat_messages(&bodies[2]),
            vec![
                (
                    "system".to_string(),
                    system_message_with(
                        Some(SEEDED_SYSTEM_PROMPT),
                        DEFAULT_AGENT_LABEL,
                        &[TRAILING_BLOCK_NOTE, THREAD_MAP_NOTE, THREAD_MAP_TOOLS_NOTE]
                    )
                ),
                (
                    "user".to_string(),
                    stamps.headed(&u1_item, HUMAN_LABEL, "How do tides work?")
                ),
                (
                    "assistant".to_string(),
                    stamps.headed(&i1_item, DEFAULT_AGENT_LABEL, "Hello from the stream.")
                ),
                (
                    "user".to_string(),
                    stamps.headed(&branch_item, HUMAN_LABEL, "What about spring tides?")
                ),
                (
                    "user".to_string(),
                    trailing(
                        Some(&roster(&[
                            (HUMAN_LABEL, "human", false),
                            (DEFAULT_AGENT_LABEL, "agent", true),
                        ])),
                        Some(&thread_map(&[(
                            format!("at #{}", eidola_app_core::post_handle(&i1_item)),
                            vec![map_entry(
                                &u2_item,
                                HUMAN_LABEL,
                                "2 posts",
                                "just now",
                                Some("1 post"),
                                "And why two per day?",
                            )],
                        )])),
                        &eidola_app_core::post_handle(&branch_item),
                    )
                ),
            ],
            "branch reply context = system prompt + the branch's ancestry + the trailing block"
        );
    });
}

// ===========================================================================
// Thread map (task 21)
// ===========================================================================

/// Take one streaming turn, optionally branching at `reply_to`. The
/// thread-map tests all build multi-post fixtures, so this keeps them readable.
fn turn(
    core: &AppCore,
    prompt: &str,
    space: Option<String>,
    reply_to: Option<String>,
) -> eidola_app_core::ChatResult {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
    core.runtime()
        .block_on(core.chat_stream_reply(prompt.to_string(), MODEL.into(), space, reply_to, tx))
        .unwrap_or_else(|e| panic!("turn {prompt:?} failed: {e}"))
}

/// The linear common case sends **no** map, no map note, and no `tools` field.
/// This is the pin that keeps the overwhelming majority of turns (and their
/// upstream prefix caches) untouched by the thread-map feature. (It is no
/// longer a claim of byte-*identity* with pre-task-21: task 64 added the
/// identity line to every space. The rule it still enforces is that a feature's
/// bytes appear with the feature and are stable after — `AGENTS.md` → Thread
/// map.)
#[test]
fn a_linear_space_sends_no_thread_map_and_no_tools() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let first = turn(&core, "How do tides work?", None, None);
        turn(
            &core,
            "And why two per day?",
            Some(first.space_id.clone()),
            None,
        );

        for body in mock.chat_bodies() {
            assert!(
                body.get("tools").is_none(),
                "a linear space attaches no tools: {body}"
            );
            let msgs = flat_messages(&body);
            assert_eq!(
                msgs[0].1,
                system_message(Some(SEEDED_SYSTEM_PROMPT), DEFAULT_AGENT_LABEL),
                "the system message is untouched without a map"
            );
            assert!(
                msgs.iter().all(|(_, c)| !c.contains("<thread-map>")),
                "no map block anywhere: {msgs:#?}"
            );
        }
    });
}

/// A branched space's turn carries the map as its **last** message, naming
/// every branch the spine does not contain: the fork it hangs off (by handle),
/// the branch's author, post count, recency, and opening line. Dormant branches
/// get short entries — never omission.
#[test]
fn a_branched_space_appends_a_thread_map_as_the_last_message() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // u1 -> i1 -> u2 -> i2, then a branch off i1.
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        turn(&core, "And why two per day?", Some(space.clone()), None);

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let i1 = tree[1].action_id.clone();
        let i1_item = tree[1].item_id.clone();
        let i2 = tree[3].action_id.clone();

        turn(
            &core,
            "What about spring tides?",
            Some(space.clone()),
            Some(i1),
        );
        let branch_item = item_id_of(&core, &space, "What about spring tides?");

        // Now take a turn back on the ORIGINAL spine (replying to i2), so the
        // branch is what the turn cannot see.
        turn(&core, "Anything else?", Some(space.clone()), Some(i2));
        let last_item = item_id_of(&core, &space, "Anything else?");

        let bodies = mock.chat_bodies();
        let msgs = flat_messages(&bodies[3]);

        // The trailing volatile block is the LAST message — the placement
        // decision. Everything above it is conversation; inside it the map
        // comes last, because it ends with the `Respond to #h.` pointer.
        assert_eq!(
            msgs.last().expect("a message").clone(),
            (
                "user".to_string(),
                trailing(
                    Some(&roster(&[
                        (HUMAN_LABEL, "human", false),
                        (DEFAULT_AGENT_LABEL, "agent", true),
                    ])),
                    Some(&thread_map(&[(
                        format!("at #{}", eidola_app_core::post_handle(&i1_item)),
                        vec![map_entry(
                            &branch_item,
                            HUMAN_LABEL,
                            // The branch's own ask plus the answer it drew.
                            "2 posts",
                            "just now",
                            // One of which is this responder's own.
                            Some("1 post"),
                            "What about spring tides?",
                        )],
                    )])),
                    &eidola_app_core::post_handle(&last_item),
                ),
            ),
        );
        assert_eq!(
            msgs[0].1,
            system_message_with(
                Some(SEEDED_SYSTEM_PROMPT),
                DEFAULT_AGENT_LABEL,
                &[TRAILING_BLOCK_NOTE, THREAD_MAP_NOTE, THREAD_MAP_TOOLS_NOTE]
            ),
            "the map note and the tools note both join the system message"
        );
        assert!(
            bodies[3].get("tools").is_some(),
            "the eidola backend is offered the navigation tools like any other: {}",
            bodies[3]
        );
    });
}

/// The eidola backend goes through exactly the same gate as every other: a
/// branched space's turn advertises the three navigation tools and tells the
/// model they exist. Nothing about the kind is consulted — capability is
/// learned, and an endpoint that has never refused the field is offered it.
#[test]
fn a_branched_eidola_turn_advertises_the_navigation_tools() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // u1 -> i1, then a branch off i1 so the next turn on the spine forks.
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let i1 = tree[1].action_id.clone();
        core.runtime()
            .block_on(core.post_reply(
                "What about spring tides?".into(),
                Some(space.clone()),
                Some(i1.clone()),
            ))
            .expect("branch post");

        turn(&core, "Anything else?", Some(space.clone()), Some(i1));

        let bodies = mock.chat_bodies();
        let branched = bodies.last().expect("the branched turn's request");
        let advertised: Vec<String> = branched["tools"]
            .as_array()
            .unwrap_or_else(|| panic!("a branched eidola turn advertises tools: {branched}"))
            .iter()
            .map(|t| t["function"]["name"].as_str().expect("a name").to_string())
            .collect();
        assert_eq!(advertised, ["list_branches", "read_thread", "read_post"]);
        assert!(
            flat_messages(branched)[0].1.contains(THREAD_MAP_TOOLS_NOTE),
            "…and the model is told how to use them: {}",
            flat_messages(branched)[0].1
        );

        // The first turn predates the branch, so it is untouched: the pin that
        // keeps linear eidola spaces byte-identical.
        assert!(
            bodies[0].get("tools").is_none(),
            "a linear turn still attaches nothing: {}",
            bodies[0]
        );
    });
}

/// **The map's you-participated annotation (task 33).** A branch the
/// responding participant has posted in says so, with its own post count; a
/// branch it has never touched says nothing. The map is the volatile tail, so
/// this per-participant segment costs no shared prefix — and it is the
/// retrieval prompt: a model does not descend unless told there is something
/// of its own down there.
#[test]
fn the_map_says_which_branches_the_responder_posted_in() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // Spine: u1 -> i1 -> u2 -> i2.
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        turn(&core, "And why two per day?", Some(space.clone()), None);

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let i1 = tree[1].action_id.clone();
        let i1_item = tree[1].item_id.clone();
        let i2 = tree[3].action_id.clone();

        // Branch A off i1: asked, so the responder answered in it.
        turn(
            &core,
            "What about spring tides?",
            Some(space.clone()),
            Some(i1.clone()),
        );
        let answered_branch = item_id_of(&core, &space, "What about spring tides?");

        // Branch B off i1: saved only (`post_reply` makes no request), so the
        // responder has never posted there.
        core.runtime()
            .block_on(core.post_reply(
                "And neap tides?".into(),
                Some(space.clone()),
                Some(i1.clone()),
            ))
            .expect("saved branch");
        let quiet_branch = item_id_of(&core, &space, "And neap tides?");

        // A turn back on the original spine sees both branches in its map.
        turn(&core, "Anything else?", Some(space.clone()), Some(i2));
        let last_item = item_id_of(&core, &space, "Anything else?");

        let msgs = flat_messages(mock.chat_bodies().last().expect("a request"));
        assert_eq!(
            msgs.last().expect("a message").clone(),
            (
                "user".to_string(),
                trailing(
                    Some(&roster(&[
                        (HUMAN_LABEL, "human", false),
                        (DEFAULT_AGENT_LABEL, "agent", true),
                    ])),
                    Some(&thread_map(&[(
                        format!("at #{}", eidola_app_core::post_handle(&i1_item)),
                        vec![
                            map_entry(
                                &answered_branch,
                                HUMAN_LABEL,
                                "2 posts",
                                "just now",
                                // The answer in that branch is the responder's.
                                Some("1 post"),
                                "What about spring tides?",
                            ),
                            map_entry(
                                &quiet_branch,
                                HUMAN_LABEL,
                                "1 post",
                                "just now",
                                None,
                                "And neap tides?",
                            ),
                        ],
                    )])),
                    &eidola_app_core::post_handle(&last_item),
                ),
            ),
        );
    });
}

/// The cache invariant, as byte equality.
///
/// Two things are pinned, and they are different strengths on purpose:
///
/// * **The conversation trunk is unconditionally identical** across every turn
///   in the space — the posts up to the fork are the same bytes in the same
///   order whether the turn is on branch A, on branch B, or on a linear space
///   that has not branched yet. Nothing a branch does may move a shared byte.
/// * **The full prefix, system message included, is identical across sibling
///   branches** once both see a branched space. The system message flips
///   exactly once per space — when the space first branches and the map note
///   joins it — and is byte-stable from then on, which is why the two sibling
///   turns below share it and the pre-branch turn does not.
///
/// All the per-turn volatility (which branches exist, how recently they moved)
/// is in the trailing map, where recompute is cheap by construction.
#[test]
fn sibling_branch_turns_send_identical_trunk_bytes() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // Trunk: u1 -> i1. Then two sibling branches off i1.
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let i1 = tree[1].action_id.clone();

        let branch_a = turn(
            &core,
            "What about spring tides?",
            Some(space.clone()),
            Some(i1.clone()),
        );
        turn(
            &core,
            "And what about neap tides?",
            Some(space.clone()),
            Some(i1.clone()),
        );

        // A third turn, back on branch A now that B exists — the genuine
        // sibling of branch B's turn (both see the same branched space).
        turn(
            &core,
            "Go on.",
            Some(space.clone()),
            branch_a.response_action_id.clone(),
        );

        let bodies = mock.chat_bodies();
        let a = flat_messages(&bodies[1]); // branch A forked a linear space
        let b = flat_messages(&bodies[2]); // branch B, space now branched
        let c = flat_messages(&bodies[3]); // branch A continues, space branched

        // The conversation trunk — the posts before the fork — is the same
        // bytes in all three, including the pre-branch turn.
        assert_eq!(&a[1..3], &b[1..3], "posts before the fork: A vs B");
        assert_eq!(
            &a[1..3],
            &c[1..3],
            "posts before the fork: A vs A-continued"
        );

        // Sibling turns over the branched space share the WHOLE prefix,
        // system message included.
        assert_eq!(
            &b[..3],
            &c[..3],
            "sibling branches share every byte up to the fork, system message included"
        );

        // The one-time flip: A's turn predates the branch, so it carries
        // neither the map note nor a map.
        assert_eq!(
            a[0].1,
            system_message(Some(SEEDED_SYSTEM_PROMPT), DEFAULT_AGENT_LABEL),
            "the pre-branch turn's system message is the pre-task-21 one"
        );
        assert!(
            a.iter().all(|(_, c)| !c.contains("<thread-map>")),
            "branch A forked a linear space: no map yet"
        );

        // And in both branched turns the map rides the LAST message — all the
        // volatility at the tail — which closes with the response pointer.
        for msgs in [&b, &c] {
            let last = &msgs.last().expect("a message").1;
            assert!(
                last.contains("</thread-map>"),
                "the map rides the last message: {msgs:#?}"
            );
            assert!(
                last.trim_end().ends_with('.') && last.contains("\n\nRespond to #"),
                "which closes with the response pointer: {last}"
            );
            assert_eq!(
                msgs.iter()
                    .filter(|(_, c)| c.contains("</thread-map>"))
                    .count(),
                1,
                "exactly one map block"
            );
        }
    });
}

// ===========================================================================
// Navigation tools (task 21)
//
// `prepare_turn` attaches the navigation tools when the space has branches AND
// the endpoint has not been observed to reject a `tools` field — no backend
// kind is consulted. These run over an `openai` backend so the tool round-trip
// is exercised without a spend in the way; the eidola path takes the identical
// gate (`a_branched_eidola_turn_advertises_the_navigation_tools`) and its spend
// interaction is pinned separately (`a_rejected_tools_field_on_a_spending_
// backend_refunds_the_failed_hold`).
// ===========================================================================

/// An `openai` backend pointed at the mock — no account, no spend.
fn external_backend(core: &AppCore, base_url: &str) {
    core.runtime()
        .block_on(core.add_backend(eidola_app_core::NewBackend {
            id: "ext".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: String::new(),
            base_url: Some(base_url.to_string()),
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        }))
        .expect("add backend");
}

/// The three navigation tools are advertised on a branched space's turn, and a
/// round of real calls — including a deliberately stale handle — round-trips
/// through the task-20 loop and comes back to the model as tool *results*.
#[test]
fn navigation_tools_round_trip_through_the_turn_loop() {
    run(|| {
        let script = chat_harness::tool_script();
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolScript,
            tool_script: script.clone(),
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        let model = "qwen3-8b@ext";

        // u1 -> i1 -> u2 -> i2, plus a branch post off i1 (saved, not asked —
        // `post_reply` makes no request, so the mock's script stays untouched).
        let first = core
            .runtime()
            .block_on(core.chat("How do tides work?".into(), model.into(), None))
            .expect("first turn");
        let space = first.space_id.clone();
        core.runtime()
            .block_on(core.chat(
                "And why two per day?".into(),
                model.into(),
                Some(space.clone()),
            ))
            .expect("second turn");
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let i1 = tree[1].action_id.clone();
        let u2_item = tree[2].item_id.clone();
        let i2_item = tree[3].item_id.clone();
        let i1_item = tree[1].item_id.clone();
        // The branch post quotes the post it branches off, so `read_thread`'s
        // window has to render an embed marker (task 63): a model descending
        // into a branch reads the passage, not the marker.
        let quoted = tree[1].blocks[0].text.clone().expect("i1 text");
        let i1_label = tree[1].participant.label.clone();
        let branch = core
            .runtime()
            .block_on(core.post_with_references(
                "What about spring tides?\n\n{{ embed 1 }}".into(),
                Some(space.clone()),
                Some(i1),
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: tree[1].action_id.clone(),
                    content_block_id: Some(tree[1].blocks[0].id.clone()),
                    range_start: Some(0),
                    range_end: Some(quoted.len() as i64),
                    annotation: None,
                }],
            ))
            .expect("branch post");

        // Now ask on the branch. The space forks at i1, so the map is present
        // and the tools attach.
        *script.lock().unwrap() = vec![
            ("list_branches".into(), "{}".into()),
            (
                "read_thread".into(),
                serde_json::json!({
                    "handle": format!("#{}", eidola_app_core::post_handle(&u2_item)),
                    "limit": 1,
                })
                .to_string(),
            ),
            (
                "read_post".into(),
                serde_json::json!({ "handle": eidola_app_core::post_handle(&i2_item) }).to_string(),
            ),
            (
                "read_post".into(),
                serde_json::json!({ "handle": "#zzzzzzz" }).to_string(),
            ),
            (
                "read_thread".into(),
                serde_json::json!({
                    "handle": eidola_app_core::post_handle(&branch.item_id),
                })
                .to_string(),
            ),
        ];
        let result = core
            .runtime()
            .block_on(core.chat("Tell me more.".into(), model.into(), Some(space.clone())))
            .expect("the tool round-trip must not fail the turn");
        assert_eq!(result.content, chat_harness::TOOL_FINAL_CONTENT);

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 4, "two fixture turns + two rounds");

        // Round 1 advertises exactly the three navigation tools.
        let advertised: Vec<String> = bodies[2]["tools"]
            .as_array()
            .expect("tools array")
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(advertised, ["list_branches", "read_thread", "read_post"]);
        assert!(
            flat_messages(&bodies[2])[0]
                .1
                .contains(THREAD_MAP_TOOLS_NOTE),
            "the tools note only joins the system message when the tools attach"
        );

        // Round 2 replays the results the model reads.
        let stamps = Stamps::of(&core, &space);
        let results: Vec<String> = flat_messages(&bodies[3])
            .into_iter()
            .filter(|(role, _)| role == "tool")
            .map(|(_, c)| c)
            .collect();
        assert_eq!(results.len(), 5, "one result per call, in order");

        // list_branches: the whole space's structure, not just the map's.
        assert!(
            results[0].starts_with("1 fork point in this space."),
            "{}",
            results[0]
        );
        assert!(
            results[0].contains(&format!("at #{}", eidola_app_core::post_handle(&i1_item))),
            "{}",
            results[0]
        );

        // read_thread: a bounded window, rendered in the task-19 header format,
        // stating honestly what it did not show.
        let u2_handle = eidola_app_core::post_handle(&u2_item);
        assert!(
            results[1].starts_with(&format!("Thread from #{u2_handle} — 2 posts, showing 1–1")),
            "{}",
            results[1]
        );
        assert!(
            results[1].contains(&stamps.headed(&u2_item, HUMAN_LABEL, "And why two per day?")),
            "one rendering path — the exact wire header format: {}",
            results[1]
        );
        assert!(
            results[1].ends_with("1 post not shown — call read_thread again with offset=1."),
            "{}",
            results[1]
        );

        // read_post: one post in full.
        assert!(
            results[2].starts_with(&format!("#{} · ", eidola_app_core::post_handle(&i2_item))),
            "{}",
            results[2]
        );
        assert!(
            results[2].contains(chat_harness::TOOL_FINAL_CONTENT),
            "{}",
            results[2]
        );

        // A stale handle is answered honestly IN THE RESULT — the map is a
        // snapshot, and reading a stale one must not burn the turn.
        assert!(results[3].contains("`#zzzzzzz`"), "{}", results[3]);
        assert!(results[3].contains("snapshot"), "{}", results[3]);
        assert!(
            !results[3].starts_with("error:"),
            "a stale handle is an answer, not a tool error: {}",
            results[3]
        );

        // read_thread renders a post's quotes exactly as read_post does —
        // attributed and in place, never a literal marker.
        assert!(
            results[4].contains(&stamps.headed(
                &branch.item_id,
                HUMAN_LABEL,
                &format!(
                    "What about spring tides?\n\n[1] #{} · {}\n> {quoted}",
                    eidola_app_core::post_handle(&i1_item),
                    i1_label,
                )
            )),
            "{}",
            results[4]
        );
        assert!(
            !results[4].contains("{{ embed"),
            "no literal marker survives into a tool result: {}",
            results[4]
        );
    });
}

/// A post carries **one** byline. The transcript renders the author's
/// effective label — the space's own override where there is one — and so must
/// every tool result, or the same participant answers to two names in one turn.
#[test]
fn a_per_space_rename_reaches_the_tools_as_well_as_the_transcript() {
    run(|| {
        let script = chat_harness::tool_script();
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolScript,
            tool_script: script.clone(),
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        let space = branched_external_space(&core);
        core.runtime()
            .block_on(core.set_space_participant_override(
                space.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                eidola_app_core::ParticipantOverride {
                    label: Some(Some("Skipper".into())),
                    model_ref: None,
                    system_prompt: None,
                    notify_policy: None,
                },
            ))
            .expect("override");

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let u1_item = tree[0].item_id.clone();
        *script.lock().unwrap() = vec![(
            "read_post".into(),
            serde_json::json!({ "handle": eidola_app_core::post_handle(&u1_item) }).to_string(),
        )];
        core.runtime()
            .block_on(core.chat(
                "Tell me more.".into(),
                "qwen3-8b@ext".into(),
                Some(space.clone()),
            ))
            .expect("turn");

        let stamps = Stamps::of(&core, &space);
        let bodies = mock.chat_bodies();
        let round1 = flat_messages(&bodies[bodies.len() - 2]);
        assert!(
            round1.iter().any(|(_, c)| c.starts_with(&stamps.headed(
                &u1_item,
                "Skipper",
                "How do tides work?"
            ))),
            "the transcript renames the author: {round1:?}"
        );
        let result = flat_messages(bodies.last().expect("a follow-up round"))
            .into_iter()
            .rfind(|(role, _)| role == "tool")
            .expect("a tool result")
            .1;
        assert!(
            result.starts_with(&stamps.headed(&u1_item, "Skipper", "How do tides work?")),
            "and so does read_post — one byline per post: {result}"
        );
    });
}

/// Build a branched fixture space over an `openai` backend and return
/// `(space_id, the action to reply to)`. Two posts and a branch post, so the
/// next turn carries a map (and would carry tools).
fn branched_external_space(core: &AppCore) -> String {
    let first = core
        .runtime()
        .block_on(core.chat("How do tides work?".into(), "qwen3-8b@ext".into(), None))
        .expect("first turn");
    let space = first.space_id.clone();
    core.runtime()
        .block_on(core.chat(
            "And why two per day?".into(),
            "qwen3-8b@ext".into(),
            Some(space.clone()),
        ))
        .expect("second turn");
    let tree = core
        .runtime()
        .block_on(core.get_space_tree(space.clone()))
        .expect("tree");
    core.runtime()
        .block_on(core.post_reply(
            "What about spring tides?".into(),
            Some(space.clone()),
            Some(tree[1].action_id.clone()),
        ))
        .expect("branch post");
    space
}

/// Backend *kind* cannot establish tool-calling capability: a generic
/// OpenAI-compatible endpoint — or a llama.cpp server whose model template's
/// tool block does not render — speaks chat completions perfectly well and
/// rejects only the `tools` field. Since attachment is automatic on branch,
/// that would otherwise mean "branching your conversation breaks every turn"
/// with no opt-out. Instead the turn withdraws the tools it added and retries
/// the round once, keeping the map.
#[test]
fn a_backend_that_rejects_tools_degrades_to_a_toolless_retry() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectTools,
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        let space = branched_external_space(&core);

        let result = core
            .runtime()
            .block_on(core.chat(
                "Tell me more.".into(),
                "qwen3-8b@ext".into(),
                Some(space.clone()),
            ))
            .expect("the turn must survive a backend that rejects tools");
        assert_eq!(result.content, "Hello from the mock.");

        let bodies = mock.chat_bodies();
        assert_eq!(
            bodies.len(),
            4,
            "two fixture turns + the rejected try + the retry"
        );
        assert!(
            bodies[2].get("tools").is_some(),
            "the first try offers the navigation tools"
        );
        assert!(
            bodies[3].get("tools").is_none(),
            "the retry withdraws them: {}",
            bodies[3]
        );
        // The map is not part of the degrade — it rides the messages array and
        // is unaffected by the tools field.
        for body in [&bodies[2], &bodies[3]] {
            let msgs = flat_messages(body);
            assert!(
                msgs.last().expect("a message").1.contains("</thread-map>"),
                "the map survives the degrade: {msgs:#?}"
            );
        }
    });
}

/// **The rejection shape the deployed server actually produces.**
///
/// An Eidola server too old to know the `tools` field fails it in the body
/// extractor, and axum renders that rejection as a **plain-text 422** — not
/// JSON. A client that decided "server error vs transport error" by whether the
/// body parses would file this as a transport error and never degrade, so every
/// branched turn against such a server would fail outright. The status is what
/// classifies the response; the body is just the message.
#[test]
fn a_plain_text_rejection_still_degrades_to_a_toolless_retry() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectToolsUnparseable,
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        let space = branched_external_space(&core);

        let result = core
            .runtime()
            .block_on(core.chat(
                "Tell me more.".into(),
                "qwen3-8b@ext".into(),
                Some(space.clone()),
            ))
            .expect("an unparseable rejection body must still degrade, not fail the turn");
        assert_eq!(result.content, "Hello from the mock.");

        let bodies = mock.chat_bodies();
        assert_eq!(
            bodies.len(),
            4,
            "two fixture turns + the rejected try + the retry"
        );
        assert!(bodies[2].get("tools").is_some(), "{}", bodies[2]);
        assert!(bodies[3].get("tools").is_none(), "{}", bodies[3]);

        // The rejected exchange is recorded, and the plain-text body survives
        // into the Record rather than becoming a JSON-parse complaint.
        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        let rejected = requests
            .iter()
            .find(|r| r.response_status == Some(422))
            .expect("the rejected attempt's request row");
        let detail = core
            .runtime()
            .block_on(core.request_detail(rejected.id.clone()))
            .expect("detail")
            .expect("the row exists");
        assert_eq!(
            String::from_utf8_lossy(&detail.response_body.unwrap_or_default()),
            chat_harness::UNKNOWN_FIELD_REJECTION,
            "the endpoint's own words are kept verbatim"
        );

        // And it is remembered, so the next branched turn skips the probe.
        core.runtime()
            .block_on(core.chat(
                "And more.".into(),
                "qwen3-8b@ext".into(),
                Some(space.clone()),
            ))
            .expect("second turn");
        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 5, "the second turn costs ONE request");
        assert!(
            bodies[4].get("tools").is_none(),
            "the rejection was memoized: {}",
            bodies[4]
        );
    });
}

/// The streaming twin classifies by status *before* reading the body, so it
/// never had the blocking path's ordering problem — but the GUI is the
/// streaming caller, so the behaviour is pinned rather than assumed.
#[test]
fn a_plain_text_rejection_degrades_on_the_streaming_path_too() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectToolsUnparseable,
            ..MockConfig::default()
        });
        with_account(&core);

        // Branch an eidola space with streamed turns, then ask on the fork.
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        turn(&core, "And why two per day?", Some(space.clone()), None);
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        core.runtime()
            .block_on(core.post_reply(
                "What about spring tides?".into(),
                Some(space.clone()),
                Some(tree[1].action_id.clone()),
            ))
            .expect("branch post");

        let result = turn(&core, "Tell me more.", Some(space.clone()), None);
        assert_eq!(result.content, "Hello from the stream.");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 4, "two fixture turns + rejected try + retry");
        assert!(bodies[2].get("tools").is_some(), "{}", bodies[2]);
        assert!(bodies[3].get("tools").is_none(), "{}", bodies[3]);

        // Nothing stranded on the spend path either.
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        assert!(
            lifecycle.iter().all(|c| c.state != "spending"),
            "no hold outlives the streamed degrade; got {lifecycle:?}"
        );
    });
}

/// Build a branched fixture space over the **eidola** backend — the spending
/// one — and return its id. Same shape as `branched_external_space`: two
/// linear turns (which attach no tools, so a tools-rejecting mock answers them
/// normally) plus a saved branch off the first answer, so the next turn on the
/// tail forks and attaches the navigation tools.
fn branched_eidola_space(core: &AppCore) -> String {
    let first = core
        .runtime()
        .block_on(core.chat("How do tides work?".into(), MODEL.into(), None))
        .expect("first turn");
    let space = first.space_id.clone();
    core.runtime()
        .block_on(core.chat(
            "And why two per day?".into(),
            MODEL.into(),
            Some(space.clone()),
        ))
        .expect("second turn");
    let tree = core
        .runtime()
        .block_on(core.get_space_tree(space.clone()))
        .expect("tree");
    core.runtime()
        .block_on(core.post_reply(
            "What about spring tides?".into(),
            Some(space.clone()),
            Some(tree[1].action_id.clone()),
        ))
        .expect("branch post");
    space
}

/// **The degrade-on-rejection retry, on a backend that spends.**
///
/// Every round holds its own credential: the ACT protocol binds a spend proof
/// to one request and one charge, so a retry can never re-present the hold the
/// rejected attempt burned. This pins the whole accounting of that, end to end:
///
/// * the rejected attempt's hold is **settled** — its 500 carries no inline
///   refund, so the round recovers one and its credential reaches `spent`;
/// * the retry holds a **different** credential, re-estimated over the array it
///   is actually going to send, which is *smaller* because the withdrawn tool
///   schemas are no longer being charged for;
/// * nothing is stranded (no credential left `spending`) and nothing is lost —
///   with the mock refunding each charge in full, the wallet's spendable total
///   is exactly what it was before the turn;
/// * one answer is persisted, not two.
#[test]
fn a_rejected_tools_field_on_a_spending_backend_settles_both_holds() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectTools,
            ..MockConfig::default()
        });
        with_account(&core);
        let space = branched_eidola_space(&core);

        let credits_before: i64 = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet")
            .iter()
            .map(|c| c.credits)
            .sum();
        let refunds_before = mock.refund_hits();
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("Tell me more.".into(), MODEL.into(), Some(space.clone())))
            .expect("the turn must survive a server that rejects the tools field");
        assert_eq!(result.content, "Hello from the mock.");

        // The rejected attempt is an emission point of its own (bus.rs's
        // degrade row): its request row lands, and the retry's fresh hold is a
        // wallet change the moment the credential flips to `spending`.
        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space.clone())));

        // --- the wire: one rejected attempt, one toolless retry -------------
        let bodies = mock.chat_bodies();
        assert_eq!(
            bodies.len(),
            4,
            "two fixture turns + the rejected try + the retry"
        );
        assert!(
            bodies[2].get("tools").is_some(),
            "the eidola backend is offered the tools first: {}",
            bodies[2]
        );
        assert!(
            bodies[3].get("tools").is_none(),
            "the retry withdraws them: {}",
            bodies[3]
        );
        assert_eq!(
            flat_messages(&bodies[2]),
            flat_messages(&bodies[3]),
            "only the tools field is withdrawn — the messages, map included, are the same bytes"
        );

        // Each attempt presented its own ACT token: a hold is bound to one
        // request, so reusing the rejected one would have been the bug.
        let auths = mock.chat_auth_values();
        assert!(
            auths[2].is_some() && auths[3].is_some(),
            "both attempts spend: {auths:?}"
        );
        assert_ne!(
            auths[2], auths[3],
            "the retry presents a fresh spend proof, not the rejected attempt's"
        );

        // --- the accounting -------------------------------------------------
        // The 500 carries no inline refund, so the rejected attempt's hold had
        // to be recovered through `/v1/credentials/refund`.
        assert!(
            mock.refund_hits() > refunds_before,
            "the failed attempt recovered its refund rather than abandoning the credential"
        );

        let answer = result
            .response_action_id
            .clone()
            .expect("the retry persisted the answer");
        let trail = core
            .runtime()
            .block_on(core.spend_trail(50, 0))
            .expect("spend trail");
        // The rejected attempt is the one spend with no action: the inference
        // is deliberately not persisted when the round is about to be retried.
        let rejected = trail
            .iter()
            .find(|e| e.action_id.is_none())
            .expect("the rejected attempt's request row");
        let retry = trail
            .iter()
            .find(|e| e.action_id.as_deref() == Some(answer.as_str()))
            .expect("the retry's request row");

        assert_ne!(
            rejected.credential_nonce, retry.credential_nonce,
            "the retry acquired a fresh hold"
        );
        for (what, entry) in [("the rejected attempt", rejected), ("the retry", retry)] {
            assert_eq!(
                entry.credential_state, "spent",
                "{what}'s credential is settled, not left holding: {}",
                entry.credential_nonce
            );
        }
        let rejected_hold = rejected.spend_amount.expect("the rejected attempt held");
        let retry_hold = retry.spend_amount.expect("the retry held");
        assert!(
            retry_hold < rejected_hold,
            "the retry re-estimates over the array it will actually send, so its hold \
             drops by the withdrawn tool schemas: {retry_hold} vs {rejected_hold}"
        );
        assert_eq!(
            result.credits_charged,
            rejected_hold + retry_hold,
            "the reported charge is the sum of the holds the turn actually took"
        );

        // Nothing stranded: no credential is still holding.
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        assert!(
            lifecycle.iter().all(|c| c.state != "spending"),
            "no hold outlives the turn; got {lifecycle:?}"
        );
        // Nothing lost: the mock refunds each charge in full, so a settled pair
        // of holds leaves the spendable total exactly where it was. A leaked
        // hold or a double-charge would show up here as a shortfall.
        let credits_after: i64 = core
            .runtime()
            .block_on(core.wallet_credentials())
            .expect("wallet")
            .iter()
            .map(|c| c.credits)
            .sum();
        assert_eq!(
            credits_after, credits_before,
            "both holds came back — the failed attempt cost the wallet nothing"
        );

        // One answer, not two: the rejected attempt wrote no inference, so the
        // retry could claim the turn's item identity.
        assert_eq!(
            trail
                .iter()
                .filter(|e| e.action_type.as_deref() == Some("inference"))
                .count(),
            3,
            "one inference per turn — two fixture turns and this one: {trail:?}"
        );
    });
}

/// **A hold is never abandoned to take another one.**
///
/// The rejected attempt settles its own hold, but settlement is a network call
/// and can fail. Acquiring the retry's hold overwrites the only in-memory
/// handle to the materials that mint the first one's successor — so proceeding
/// anyway would leave that credential `spending`, its face value locked out of
/// the wallet until an explicit recovery, and would do it while spending a
/// *second* credential against an endpoint that just failed to settle the
/// first. `begin_next_round` refuses instead, and the turn ends honestly.
#[test]
fn a_retry_will_not_take_a_hold_while_the_last_one_is_unsettled() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectTools,
            refund: RefundMode::Fail,
            ..MockConfig::default()
        });
        with_account(&core);
        let space = branched_eidola_space(&core);

        let holding_before = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle")
            .iter()
            .filter(|c| c.state == "spending")
            .count();
        let requests_before = mock.chat_bodies().len();

        let err = core
            .runtime()
            .block_on(core.chat("Tell me more.".into(), MODEL.into(), Some(space.clone())))
            .expect_err("an unsettleable hold must end the turn, not be abandoned");
        assert!(
            matches!(err, AppError::Credential { .. }),
            "the turn names the settlement failure: {err}"
        );

        // The retry never reached the wire: refusing happens *before* the hold.
        assert_eq!(
            mock.chat_bodies().len() - requests_before,
            1,
            "exactly one attempt — the rejected one"
        );
        // And exactly one new hold exists, not two. This is the whole finding:
        // a second `acquire_spend` here would have stranded the first.
        let holding_after = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle")
            .iter()
            .filter(|c| c.state == "spending")
            .count();
        assert_eq!(
            holding_after - holding_before,
            1,
            "the failed turn took one hold, not two"
        );
    });
}

/// The degrade is remembered for the process, so the wasted request happens at
/// most once per model rather than on every branched turn. It is an in-process
/// observation, not a config surface and not persisted — the capability flag
/// stays genuinely deferred.
#[test]
fn a_backend_observed_to_reject_tools_is_not_offered_them_again() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectTools,
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        let space = branched_external_space(&core);

        for prompt in ["Tell me more.", "And more."] {
            core.runtime()
                .block_on(core.chat(prompt.into(), "qwen3-8b@ext".into(), Some(space.clone())))
                .expect("turn succeeds");
        }

        let bodies = mock.chat_bodies();
        assert_eq!(
            bodies.len(),
            5,
            "two fixture turns + (rejected try + retry) + ONE request for the second turn"
        );
        assert!(
            bodies[4].get("tools").is_none(),
            "the second branched turn never offers tools again: {}",
            bodies[4]
        );
        assert!(
            flat_messages(&bodies[4])
                .last()
                .expect("a message")
                .1
                .contains("</thread-map>"),
            "…and still carries the map"
        );
    });
}

/// **The turn loop's true worst case, stated exactly.**
///
/// A turn issues at most `MAX_TURN_ROUNDS` *model rounds* plus at most **one**
/// tool-capability probe — the rejected attempt the degrade retries. The probe
/// is not a round: it carries no model output, and the retry re-runs the same
/// round with the same messages. That the extra is exactly one, and that the
/// cap still binds where it always did, is what this pins.
///
/// One probe per turn is structural: `should_degrade_tools` fires only on
/// round 1 with the turn's own tools attached, and withdrawing them clears
/// that latch, so a second degrade is unrepresentable.
#[test]
fn a_degraded_turn_costs_one_extra_request_and_no_more() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectAutoToolsThenToolRounds,
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        // A consumer tool survives the withdrawal, so the retry is accepted and
        // the loop runs on to its cap — the only way to reach the worst case.
        with_echo_tool(&core);

        // Build the fork with `post` alone: this mock answers every accepted
        // request with a tool call, so any fixture *turn* would run to the cap.
        let post = core
            .runtime()
            .block_on(core.post("How do tides work?".into(), None))
            .expect("post");
        let space = post.space_id.clone();
        for prompt in ["What about spring tides?", "And neap tides?"] {
            core.runtime()
                .block_on(core.post_reply(
                    prompt.into(),
                    Some(space.clone()),
                    Some(post.action_id.clone()),
                ))
                .expect("branch post");
        }
        let fixture_requests = mock.chat_bodies().len();

        let err = core
            .runtime()
            .block_on(core.chat(
                "Tell me more.".into(),
                "qwen3-8b@ext".into(),
                Some(space.clone()),
            ))
            .expect_err("the model never stops asking, so the round cap binds");
        assert!(
            matches!(err, AppError::ToolLoop { .. }),
            "the cap still ends the turn after a degrade: {err:?}"
        );

        let turn_requests = mock.chat_bodies().len() - fixture_requests;
        assert_eq!(
            turn_requests,
            eidola_app_core::MAX_TURN_ROUNDS + 1,
            "MAX_TURN_ROUNDS rounds plus exactly one probe"
        );

        let bodies = mock.chat_bodies();
        let turn = &bodies[fixture_requests..];
        assert!(
            turn[0]["tools"]
                .as_array()
                .expect("the probe advertises tools")
                .iter()
                .any(|t| t["function"]["name"] == "list_branches"),
            "request 1 is the probe: {}",
            turn[0]
        );
        // Every later request carries only the consumer's tool — the degrade
        // happened once, and nothing re-attached what it withdrew.
        for (i, body) in turn[1..].iter().enumerate() {
            let names: Vec<&str> = body["tools"]
                .as_array()
                .expect("the consumer's tool stays")
                .iter()
                .map(|t| t["function"]["name"].as_str().expect("a name"))
                .collect();
            assert_eq!(names, ["echo"], "request {} after the probe", i + 2);
        }
    });
}

/// **What the memo is keyed by.** A backend is a host, not a capability: the
/// eidola catalog and a llama.cpp install both serve many models, and whether a
/// `tools` field renders is a property of the model's chat template. So an
/// observed rejection is remembered against the one endpoint that produced it —
/// a sibling model on the same backend is still offered tools, and would
/// otherwise have lost them to a neighbour's answer.
#[test]
fn one_models_tool_rejection_leaves_its_siblings_armed() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::RejectToolsForModel("qwen3-8b"),
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);
        let space = branched_external_space(&core);

        // The refusing model: rejected, then a toolless retry, then remembered.
        for prompt in ["Tell me more.", "And more."] {
            core.runtime()
                .block_on(core.chat(prompt.into(), "qwen3-8b@ext".into(), Some(space.clone())))
                .expect("the refusing model's turns still succeed");
        }
        // A different model on the SAME backend.
        core.runtime()
            .block_on(core.chat(
                "What do you think?".into(),
                "mistral-7b@ext".into(),
                Some(space.clone()),
            ))
            .expect("the sibling model's turn succeeds");

        let bodies = mock.chat_bodies();
        assert_eq!(
            bodies.len(),
            6,
            "two fixture turns + (rejected try + retry) + one remembered turn + the sibling's"
        );
        assert!(
            bodies[4].get("tools").is_none(),
            "the refusing model is not offered them again: {}",
            bodies[4]
        );
        assert!(
            bodies[5].get("tools").is_some(),
            "its sibling on the same backend still is: {}",
            bodies[5]
        );
    });
}

/// A failure that is *not* attributable to the tools field must not silently
/// downgrade the backend forever. The memo records only when the toolless
/// retry actually succeeds — which is the evidence that the field was the
/// cause.
#[test]
fn a_rejection_unrelated_to_tools_does_not_downgrade_the_backend() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        external_backend(&core, &mock.base_url);

        // Build the branched fixture with `post` alone — no HTTP, so the
        // always-500 mock doesn't interfere with the setup.
        let post = core
            .runtime()
            .block_on(core.post("How do tides work?".into(), None))
            .expect("post");
        let space = post.space_id.clone();
        core.runtime()
            .block_on(core.post_reply(
                "What about spring tides?".into(),
                Some(space.clone()),
                Some(post.action_id.clone()),
            ))
            .expect("branch post");
        core.runtime()
            .block_on(core.post_reply(
                "And neap tides?".into(),
                Some(space.clone()),
                Some(post.action_id.clone()),
            ))
            .expect("second branch post");

        // The space forks, so the turn attaches tools; the endpoint 500s for
        // an unrelated reason, so the toolless retry fails too.
        let err = core
            .runtime()
            .block_on(core.chat(
                "Tell me more.".into(),
                "qwen3-8b@ext".into(),
                Some(space.clone()),
            ))
            .expect_err("the turn fails honestly");
        assert!(
            matches!(err, AppError::Server { status: 500, .. }),
            "{err:?}"
        );

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2, "the try and the one toolless retry");
        assert!(bodies[0].get("tools").is_some());
        assert!(bodies[1].get("tools").is_none());
    });
}

/// The navigation tool names are reserved at the registration seam.
///
/// Without this, a consumer's `read_thread` would register fine, work on every
/// linear turn, and then be silently replaced by the built-in the moment a
/// space branched — the advertised schema and the executed implementation
/// diverging on exactly the turns the feature exists for. The turn layers its
/// tools onto the registry snapshot and `ToolRegistry::register` is
/// last-write-wins, so the collision has to be refused where it is made.
#[test]
fn registering_a_reserved_navigation_tool_name_is_refused() {
    struct Impostor(&'static str);
    impl eidola_app_core::tools::Tool for Impostor {
        fn name(&self) -> &str {
            self.0
        }
        fn description(&self) -> &str {
            "a consumer's own tool"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        fn call<'a>(
            &'a self,
            _arguments: serde_json::Value,
        ) -> eidola_app_core::tools::ToolFuture<'a> {
            Box::pin(async move { Ok(String::new()) })
        }
    }

    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        for name in eidola_app_core::tools::RESERVED_TOOL_NAMES {
            let err = core
                .register_tool(std::sync::Arc::new(Impostor(name)))
                .expect_err("a reserved name must be refused");
            assert!(
                matches!(&err, AppError::NotConfigured { message } if message.contains(name)),
                "{err:?}"
            );
        }
        assert!(
            core.registered_tools().is_empty(),
            "nothing was registered: {:?}",
            core.registered_tools()
        );
        // Any other name still registers exactly as before.
        core.register_tool(std::sync::Arc::new(Impostor("summarize")))
            .expect("an unreserved name registers");
        assert_eq!(core.registered_tools(), vec!["summarize".to_string()]);
    });
}

/// The single-agent linear thread — the common case — rendered upstream:
/// alternating `user` / `assistant`, every message carrying its uniform
/// `#<handle> · <label>` header, and one leading `system` message (the agent's
/// effective prompt + the header protocol note). This is the byte-shape pin
/// for the case that must stay equivalent-modulo-headers to the pre-role-split
/// rendering.
#[test]
fn single_agent_thread_renders_alternating_roles_with_headers() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let (tx, _rx1) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let first = core
            .runtime()
            .block_on(core.chat_stream("How do tides work?".into(), MODEL.into(), None, tx))
            .expect("first turn");
        let (tx, _rx2) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        core.runtime()
            .block_on(core.chat_stream(
                "And why two per day?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
                tx,
            ))
            .expect("second turn");

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(first.space_id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 4, "u1, i1, u2, i2; got {tree:#?}");
        let stamps = Stamps::of(&core, &first.space_id);

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2);

        // Turn 1: system + the single human post.
        assert_eq!(
            flat_messages(&bodies[0]),
            vec![
                (
                    "system".to_string(),
                    system_message(Some(SEEDED_SYSTEM_PROMPT), DEFAULT_AGENT_LABEL)
                ),
                (
                    "user".to_string(),
                    stamps.headed(&tree[0].item_id, HUMAN_LABEL, "How do tides work?")
                ),
            ]
        );

        // Turn 2: the whole spine, the responding agent's own answer as
        // `assistant`, the human's two posts as `user`.
        assert_eq!(
            flat_messages(&bodies[1]),
            vec![
                (
                    "system".to_string(),
                    system_message(Some(SEEDED_SYSTEM_PROMPT), DEFAULT_AGENT_LABEL)
                ),
                (
                    "user".to_string(),
                    stamps.headed(&tree[0].item_id, HUMAN_LABEL, "How do tides work?")
                ),
                (
                    "assistant".to_string(),
                    stamps.headed(
                        &tree[1].item_id,
                        DEFAULT_AGENT_LABEL,
                        "Hello from the stream."
                    )
                ),
                (
                    "user".to_string(),
                    stamps.headed(&tree[2].item_id, HUMAN_LABEL, "And why two per day?")
                ),
            ]
        );
    });
}

/// Strip-on-receipt: a model that mimics the visible header scaffolding has
/// that leading line removed before anything is persisted — the durable post,
/// the returned `ChatResult`, **and the live stream** all carry the bare
/// answer. The stream is included because the alternative is a header that
/// shows while a reply arrives and vanishes when it lands (task 46, bug 1).
#[test]
fn model_emitted_header_is_stripped_before_persisting() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            // The streaming mock's content is header-shaped (see
            // `ChatBehavior::OkStreamingWithHeader`).
            chat: ChatBehavior::OkStreamingWithHeader,
            ..MockConfig::default()
        });
        with_account(&core);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let result = core
            .runtime()
            .block_on(async {
                let collector = async {
                    let mut seen = String::new();
                    while let Some(ev) = rx.recv().await {
                        if let ChatStreamEvent::ContentDelta(d) = ev {
                            seen.push_str(&d);
                        }
                    }
                    seen
                };
                let chat = core.chat_stream("hi".into(), MODEL.into(), None, tx);
                let (res, streamed) = tokio::join!(chat, collector);
                res.map(|r| (r, streamed))
            })
            .expect("stream");
        let (result, streamed) = result;

        // What the reader watched arrive is what lands durably.
        assert_eq!(
            streamed, "Hello from the stream.",
            "the header never reaches the caller either"
        );
        assert_eq!(result.content, "Hello from the stream.");

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(result.space_id.clone()))
            .expect("tree");
        let answer = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("inference post");
        let text_block = answer
            .blocks
            .iter()
            .find(|b| b.block_type == "text")
            .expect("text block");
        assert_eq!(
            text_block.text.as_deref(),
            Some("Hello from the stream."),
            "the header-shaped first line never reaches the durable trail"
        );
    });
}

/// The real shape of the same defect: a token stream chops the mimicked header
/// across deltas (mid-handle, mid-separator, and with the blank line attached
/// to the first body token). The incremental strip must still show the reader
/// exactly the persisted text, and must not hold body text back once the first
/// line is decided.
#[test]
fn a_header_split_across_deltas_is_stripped_from_the_stream() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreamingWithSplitHeader,
            ..MockConfig::default()
        });
        with_account(&core);

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let (result, deltas) = core
            .runtime()
            .block_on(async {
                let collector = async {
                    let mut deltas: Vec<String> = Vec::new();
                    while let Some(ev) = rx.recv().await {
                        if let ChatStreamEvent::ContentDelta(d) = ev {
                            deltas.push(d);
                        }
                    }
                    deltas
                };
                let chat = core.chat_stream("hi".into(), MODEL.into(), None, tx);
                let (res, deltas) = tokio::join!(chat, collector);
                res.map(|r| (r, deltas))
            })
            .expect("stream");

        assert_eq!(
            deltas.concat(),
            "Hello from the stream.",
            "the header is gone from the stream however it was chopped up"
        );
        assert_eq!(result.content, "Hello from the stream.");
        assert!(
            !deltas.is_empty(),
            "the body still arrives as deltas, not as one flush at the end"
        );
    });
}

/// The item id of the current-generation post whose text block equals `text`.
fn item_id_of(core: &AppCore, space_id: &str, text: &str) -> String {
    core.runtime()
        .block_on(core.get_space_tree(space_id.to_string()))
        .expect("tree")
        .into_iter()
        .find(|n| {
            n.blocks
                .iter()
                .any(|b| b.text.as_deref() == Some(text) && b.block_type == "text")
        })
        .unwrap_or_else(|| panic!("no post with text {text:?}"))
        .item_id
}

// Post-first contract (wave 5.2b): chat = post + run_turn. The post persists
// the thought BEFORE the response request can fail on funding, so a NoAccount /
// InsufficientBalance failure now leaves the saved post behind (and emits it),
// while root() still routes onboarding. (Replaces the pre-inversion
// "leaves_zero_trace" tests; decision #1 made concrete.)

#[test]
fn no_account_persists_post_then_fails_routing_to_onboarding() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig::default());
        // No account, empty wallet: the post persists first, then the response
        // request fails with NoAccount.
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("hi".into(), MODEL.into(), None))
            .expect_err("should fail with NoAccount");
        // The typed error routes onboarding, and the post survived the funding
        // failure.
        assert!(matches!(err, AppError::NoAccount), "got {err:?}");
        let space_id = only_space(&core);

        // The saved thought is emitted (Space + SpaceIndex from post) and durable.
        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Space(space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Record),
            "no Record before any request; got {changes:?}"
        );

        let spaces = core
            .runtime()
            .block_on(core.list_spaces(true))
            .expect("spaces");
        assert_eq!(spaces.len(), 1, "the post persisted; got {spaces:?}");
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "hi");
    });
}

#[test]
fn insufficient_balance_persists_post_then_fails_routing_to_plans() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            balance: 1, // cannot cover the charge
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("hi".into(), MODEL.into(), None))
            .expect_err("should fail with InsufficientBalance");
        assert!(
            matches!(err, AppError::InsufficientBalance { .. }),
            "got {err:?}"
        );
        let space_id = only_space(&core);

        let changes = drain(&mut rx);
        // The post emitted Space + SpaceIndex; the request never spent, so no
        // Record/Wallet.
        assert!(
            !space_changes(&changes).is_empty(),
            "post should emit Space(id); got {changes:?}"
        );
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Record),
            "no Record on a pre-spend failure; got {changes:?}"
        );

        let spaces = core
            .runtime()
            .block_on(core.list_spaces(true))
            .expect("spaces");
        assert_eq!(
            spaces.len(),
            1,
            "the post persisted; got {space_id} {spaces:?}"
        );
    });
}

// ===========================================================================
// Typed failure: network error after send (connection dropped)
// ===========================================================================

#[test]
fn network_error_after_send_emits_user_turn_and_keeps_the_post() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::DropBeforeResponse,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("dropped".into(), MODEL.into(), None))
            .expect_err("should fail on dropped connection");

        // Wrapped with the persisted space id (the user turn committed).
        let space_id = only_space(&core);
        // Underlying error is a transport/network error, not a server error.
        assert!(matches!(err, AppError::Network { .. }), "got {:?}", err);

        let changes = drain(&mut rx);
        // User turn committed → Space(id) + SpaceIndex (new space) emitted.
        // Wallet was emitted at spend start. No Record (request row not written).
        assert!(
            changes.contains(&Change::Space(space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Record),
            "no Record before request row; got {changes:?}"
        );

        // The committed user turn is durable and discoverable by the wrapped id.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    });
}

// ===========================================================================
// Typed failure: non-2xx response with error body
// ===========================================================================

#[test]
fn non_2xx_emits_record_and_space_and_persists_request_row() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx should fail");

        let space_id = only_space(&core);
        match err {
            AppError::Server { status, .. } => assert_eq!(status, 500),
            other => panic!("expected Server(500), got {other:?}"),
        }

        let changes = drain(&mut rx);
        // The full set: Wallet (spend start), Space(id), SpaceIndex (new space),
        // and Record (request row committed in the non-2xx arm).
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id.clone())));
        assert!(changes.contains(&Change::SpaceIndex));
        assert!(changes.contains(&Change::Wallet));

        // Request row persisted with the 500 status.
        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].response_status, Some(500));
    });
}

#[test]
fn streaming_non_2xx_emits_record_and_space() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(503),
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, _events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let err = core
            .runtime()
            .block_on(core.chat_stream("boom".into(), MODEL.into(), None, tx))
            .expect_err("non-2xx stream should fail");

        let space_id = only_space(&core);
        match err {
            AppError::Server { status, .. } => assert_eq!(status, 503),
            other => panic!("expected Server(503), got {other:?}"),
        }

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id)));
        assert!(changes.contains(&Change::SpaceIndex));
        assert!(changes.contains(&Change::Wallet));

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].response_status, Some(503));
    });
}

// ===========================================================================
// Typed failure: mid-SSE abort (server closes the stream mid-events)
// ===========================================================================

#[test]
fn mid_sse_abort_emits_user_turn_and_keeps_the_post() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::StreamingMidAbort,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, _events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let err = core
            .runtime()
            .block_on(core.chat_stream("stream me".into(), MODEL.into(), None, tx))
            .expect_err("mid-stream abort should fail");

        let space_id = only_space(&core);
        assert!(matches!(err, AppError::Network { .. }), "got {:?}", err);

        let changes = drain(&mut rx);
        // User turn committed before the stream began reading → Space + SpaceIndex
        // + Wallet, but no Record (request row not written on mid-stream failure).
        assert!(
            changes.contains(&Change::Space(space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::SpaceIndex), "got {changes:?}");
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(
            !changes.contains(&Change::Record),
            "no Record on mid-SSE abort; got {changes:?}"
        );

        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    });
}

// ===========================================================================
// Re-request (`respond_stream`): request a response to an already-persisted
// post without re-posting — the retry entry point after a failed ask.
// It is the `run_turn_stream(Reply)` half of `chat_stream` with no leading
// `post`, so it shares that path's exit points / emissions (see tests/bus.rs);
// these assert the distinguishing behavior: no duplicated user turn, and no
// `SpaceIndex` (which is `post`'s concern, and `respond_stream` never posts).
// ===========================================================================

#[test]
fn respond_stream_requests_response_without_reposting() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // A saved user post with no reply (exactly what a failed ask leaves).
        let posted = core
            .runtime()
            .block_on(core.post("Hello, what is your name?".into(), None))
            .expect("post should save the user turn");

        // Subscribe *after* the post so only the re-request's emissions are drained.
        let mut rx = core.subscribe_changes();
        let (tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();

        let res = core.runtime().block_on(async {
            let collector = async {
                let mut content = String::new();
                while let Some(ev) = events_rx.recv().await {
                    if let ChatStreamEvent::ContentDelta(t) = ev {
                        content.push_str(&t);
                    }
                }
                content
            };
            let respond = core.respond_stream(
                posted.space_id.clone(),
                MODEL.into(),
                posted.action_id.clone(),
                tx,
            );
            let (res, content) = tokio::join!(respond, collector);
            (res.expect("re-request should stream a reply"), content)
        });
        let (res, content) = res;

        assert_eq!(res.space_id, posted.space_id);
        assert_eq!(content, "Hello from the stream.");
        assert_eq!(mock.chat_hits(), 1, "exactly one model call");

        // The saved user turn was answered in place — user + assistant, NOT a
        // second user turn (respond_stream does not re-post).
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(posted.space_id))
            .expect("messages");
        assert_eq!(
            messages.len(),
            2,
            "re-request answers the existing post, no duplicate; got {messages:?}"
        );
        assert_eq!(messages[0].role, "user");
        assert_eq!(messages[0].content, "Hello, what is your name?");
        assert_eq!(messages[1].role, "assistant");

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Space(res.space_id.clone())));
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(
            !changes.contains(&Change::SpaceIndex),
            "respond_stream never posts, so it never emits SpaceIndex; got {changes:?}"
        );
    });
}

#[test]
fn respond_stream_failure_keeps_single_post() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(503),
            ..MockConfig::default()
        });
        with_account(&core);

        let posted = core
            .runtime()
            .block_on(core.post("Hello, what is your name?".into(), None))
            .expect("post should save the user turn");

        let mut rx = core.subscribe_changes();
        let (tx, _events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();

        let err = core
            .runtime()
            .block_on(core.respond_stream(
                posted.space_id.clone(),
                MODEL.into(),
                posted.action_id.clone(),
                tx,
            ))
            .expect_err("non-2xx re-request should fail");

        match err {
            AppError::Server { status, .. } => assert_eq!(status, 503),
            other => panic!("expected Server(503), got {other:?}"),
        }

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(
            changes.contains(&Change::Space(posted.space_id.clone())),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Wallet), "got {changes:?}");
        assert!(
            !changes.contains(&Change::SpaceIndex),
            "no SpaceIndex on a re-request (post's concern); got {changes:?}"
        );

        // No assistant turn persisted, and still exactly one user turn.
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(posted.space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");
    });
}

// ===========================================================================
// Setup failures: a `prepare_turn` failure (e.g. the `/v1/models` fetch the
// PR #218 screenshot failed on) happens after `post` has already committed the
// user's turn, so the saved thought survives it and exactly one space stands —
// never a stranded second one. Both transports share the path, so both are
// asserted.
// ===========================================================================

#[test]
fn streaming_setup_failure_keeps_single_space() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            // The chat behavior never matters — we fail earlier, in prepare_turn's
            // `/v1/models` fetch.
            models_status: Some(503),
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        let (tx, _events_rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        // A blank window (space_id None) — post persists the space, then the
        // models fetch fails inside prepare_turn.
        core.runtime()
            .block_on(core.chat_stream("Hello, what is your name?".into(), MODEL.into(), None, tx))
            .expect_err("models-fetch failure should fail the turn");

        // The user post survived (post ran before prepare_turn) — exactly one
        // space with exactly one user turn, retryable rather than stranded.
        let space_id = only_space(&core);
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id.clone()))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // post emitted Space+SpaceIndex; the setup failure itself emits nothing
        // further (no request row, no spend).
        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Space(space_id.clone())));
        assert!(changes.contains(&Change::SpaceIndex));
        assert!(
            !changes.contains(&Change::Record),
            "no request row on a models-fetch failure; got {changes:?}"
        );
        assert!(
            !changes.contains(&Change::Wallet),
            "no spend started before the models fetch; got {changes:?}"
        );
    });
}

#[test]
fn blocking_setup_failure_leaves_the_saved_post() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            models_status: Some(503),
            ..MockConfig::default()
        });
        with_account(&core);

        // The blocking `chat` shares the same prepare_turn call.
        core.runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("models-fetch failure should fail the turn");
        let space_id = only_space(&core);
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id))
            .expect("messages");
        assert_eq!(messages.len(), 1, "user turn persisted; no assistant reply");
    });
}

// ===========================================================================
// Refund-recovery variants: succeed vs fail on the non-2xx path
// ===========================================================================

#[test]
fn non_2xx_with_refund_recovery_emits_wallet_for_successor() {
    run(|| {
        // Pre-fund the wallet with a large credential so the spend does NOT
        // auto-provision (keeps Account out of the picture) and the only Wallet
        // emissions come from spend-start + recovered successor.
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            refund: RefundMode::Succeed,
            ..MockConfig::default()
        });
        with_account(&core);

        let _ = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx fails");

        // The non-2xx arm consulted the recovery endpoint and minted a successor.
        assert!(mock.refund_hits() >= 1);
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        // At least one credential should now be active (the recovered successor).
        assert!(
            lifecycle.iter().any(|c| c.state == "active"),
            "a recovered successor should be active; got {lifecycle:?}"
        );
    });
}

#[test]
fn non_2xx_with_failed_refund_recovery_still_errors_and_emits_record() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            refund: RefundMode::Fail,
            ..MockConfig::default()
        });
        with_account(&core);
        let mut rx = core.subscribe_changes();

        core.runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx fails");
        let space_id = only_space(&core);

        // Recovery was attempted but failed (500 from the refund endpoint).
        assert!(mock.refund_hits() >= 1);

        let changes = drain(&mut rx);
        // Even with no recovered successor, the non-2xx arm still emits the
        // request-row Record + Space + SpaceIndex.
        assert!(changes.contains(&Change::Record), "got {changes:?}");
        assert!(changes.contains(&Change::Space(space_id)));
        assert!(changes.contains(&Change::SpaceIndex));

        // The spending credential never recovered → it stays in `spending`.
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        assert!(
            lifecycle.iter().any(|c| c.state == "spending"),
            "the unspent credential should remain in spending; got {lifecycle:?}"
        );
    });
}

// ===========================================================================
// Tool-calling turns (task 20)
//
// The turn is a bounded agentic loop: the model either answers or asks for
// tools, and a tool round persists a `tool_call` / `tool_result` pair, appends
// the results to the in-flight messages array, and runs again — at most
// `MAX_TURN_ROUNDS` model requests per turn.
//
// `get_space_tree` filters trace action types out of the render by design, so
// these assertions read the raw rows through `test_space_actions`.
// ===========================================================================

use chat_harness::{TOOL_FINAL_CONTENT, TOOL_NAME, tool_call_id, tool_result_text};
use eidola_app_core::tools::EchoTool;

/// Register the harness's echo tool — the only thing that makes a turn send
/// `tools` at all.
fn with_echo_tool(core: &AppCore) {
    core.register_tool(std::sync::Arc::new(EchoTool))
        .expect("echo is not a reserved name");
}

/// A turn whose registry is empty must send **no** `tools` field: the whole
/// point of the omission is that a registry-less install's requests stay
/// byte-identical to the pre-tools shape (upstream prefix caches, pinned-bytes
/// tests).
#[test]
fn a_turn_with_no_registered_tools_sends_no_tools_field() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);

        core.runtime()
            .block_on(core.chat("hello".into(), MODEL.into(), None))
            .expect("chat succeeds");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 1);
        assert!(
            bodies[0].get("tools").is_none(),
            "an empty registry must not add a `tools` key: {}",
            bodies[0]
        );
        assert!(bodies[0].get("tool_choice").is_none());
    });
}

/// Registering a tool adds exactly one OpenAI function schema to the request.
#[test]
fn a_registered_tool_is_advertised_as_a_function_schema() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig::default());
        with_account(&core);
        with_echo_tool(&core);

        core.runtime()
            .block_on(core.chat("hello".into(), MODEL.into(), None))
            .expect("chat succeeds");

        let bodies = mock.chat_bodies();
        let tools = bodies[0]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], TOOL_NAME);
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    });
}

/// The canonical two-round turn (blocking): round 1 asks for a tool, round 2
/// answers. Pins the persisted rows — types, threading, content — and the
/// exact shape of the follow-up request.
#[test]
fn two_round_blocking_turn_persists_tool_call_and_result_actions() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);
        let mut rx = core.subscribe_changes();

        let result = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("tool turn succeeds");

        assert_eq!(result.content, TOOL_FINAL_CONTENT);
        assert_eq!(mock.chat_hits(), 2, "one HTTP request per round");

        // --- persisted rows ------------------------------------------------
        let actions = core
            .runtime()
            .block_on(core.test_space_actions(result.space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(
            kinds,
            vec!["user_input", "tool_call", "tool_result", "inference"],
            "the round's trace is persisted between the post and the answer"
        );

        let post = &actions[0];
        let call = &actions[1];
        let tool_result = &actions[2];
        let inference = &actions[3];

        // Threading: the trace chains off the post, and the inference replies
        // to the post — never to the trace. That is what makes the trace
        // collapse out of `get_space_tree` without orphaning the answer.
        assert_eq!(call.reply_to.as_deref(), Some(post.id.as_str()));
        assert_eq!(tool_result.reply_to.as_deref(), Some(call.id.as_str()));
        assert_eq!(inference.reply_to.as_deref(), Some(post.id.as_str()));

        // The tool_call action carries one `tool_use` block naming the tool,
        // its call id, and the raw argument string.
        let blocks: Vec<&str> = call.blocks.iter().map(|b| b.block_type.as_str()).collect();
        assert_eq!(blocks, vec!["tool_use"]);
        assert_eq!(call.blocks[0].tool_name.as_deref(), Some(TOOL_NAME));
        assert_eq!(
            call.blocks[0].tool_call_id.as_deref(),
            Some(tool_call_id(1).as_str())
        );
        assert_eq!(
            call.blocks[0].data.as_deref(),
            Some(r#"{"text":"round-1"}"#)
        );
        // Each round is its own priced request, so each round's action carries
        // its own hold.
        assert!(call.credits_consumed.unwrap_or(0) > 0);
        assert_eq!(call.status, "complete");

        // The tool_result action carries the echoed text keyed by call id.
        assert_eq!(tool_result.blocks.len(), 1);
        assert_eq!(tool_result.blocks[0].block_type, "tool_result");
        assert_eq!(
            tool_result.blocks[0].tool_call_id.as_deref(),
            Some(tool_call_id(1).as_str())
        );
        assert_eq!(
            tool_result.blocks[0].text_content.as_deref(),
            Some(tool_result_text(1).as_str())
        );
        // Tools run locally — nothing was purchased for them.
        assert_eq!(tool_result.credits_consumed, None);
        assert_eq!(tool_result.model, None);

        // --- the follow-up request ------------------------------------------
        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2);
        let msgs = bodies[1]["messages"].as_array().expect("messages");
        let last_two = &msgs[msgs.len() - 2..];
        assert_eq!(last_two[0]["role"], "assistant");
        assert_eq!(last_two[0]["content"], serde_json::Value::Null);
        // The call object is replayed verbatim, ids and all.
        assert_eq!(last_two[0]["tool_calls"][0]["id"], tool_call_id(1));
        assert_eq!(last_two[0]["tool_calls"][0]["function"]["name"], TOOL_NAME);
        assert_eq!(last_two[1]["role"], "tool");
        assert_eq!(last_two[1]["tool_call_id"], tool_call_id(1));
        assert_eq!(last_two[1]["content"], tool_result_text(1));
        // Tool messages are raw — a tool result is not a post and must not
        // wear a `#<handle> · <label>` header.
        assert!(
            !last_two[1]["content"]
                .as_str()
                .unwrap_or_default()
                .starts_with('#'),
        );
        // Both rounds advertise the same tool set.
        assert_eq!(bodies[1]["tools"].as_array().map(|a| a.len()), Some(1));

        // The rendered thread shows only the two posts — the trace collapses.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(result.space_id.clone()))
            .expect("tree");
        let tree_kinds: Vec<&str> = tree.iter().map(|n| n.action_type.as_str()).collect();
        assert_eq!(tree_kinds, vec!["user_input", "inference"]);

        let changes = drain(&mut rx);
        assert!(changes.contains(&Change::Space(result.space_id.clone())));
        assert!(changes.contains(&Change::Record));
        assert!(changes.contains(&Change::Wallet));
    });
}

/// Each round acquires its **own** ACT hold: the protocol consumes a
/// credential per request (the spend proof is bound to that credential and
/// that charge), so a hold cannot be reused. Two rounds ⇒ two distinct
/// credential nonces on the request rows, and `credits_charged` is their sum.
#[test]
fn each_tool_round_acquires_its_own_hold() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let result = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("tool turn succeeds");

        // Two `PrivateToken` headers, and they differ — each round proved a
        // fresh spend.
        let auths: Vec<String> = mock
            .chat_auth_values()
            .into_iter()
            .map(|a| a.expect("every eidola round sends an ACT header"))
            .collect();
        assert_eq!(auths.len(), 2);
        assert!(auths.iter().all(|a| a.starts_with("PrivateToken token=")));
        assert_ne!(auths[0], auths[1], "each round proves its own spend");

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        let chat_rows: Vec<_> = requests
            .iter()
            .filter(|r| r.path == "/v1/chat/completions")
            .collect();
        assert_eq!(chat_rows.len(), 2, "one request row per round");
        let nonces: std::collections::HashSet<_> = chat_rows
            .iter()
            .filter_map(|r| r.credential_nonce.clone())
            .collect();
        assert_eq!(nonces.len(), 2, "each round spent a different credential");

        // The reported charge is the sum of the rounds' holds.
        let actions = core
            .runtime()
            .block_on(core.test_space_actions(result.space_id.clone()))
            .expect("raw actions");
        let per_round: i64 = actions
            .iter()
            .filter_map(|a| a.credits_consumed)
            .sum::<i64>();
        assert_eq!(result.credits_charged, per_round);
    });
}

/// The contract's chargeable prompt tokens for a recorded wire body.
///
/// Calls `eidola_common::prompt_charge` — **the** walk, the same one the
/// server runs over the request it receives. Before the walk was
/// consolidated this function was a hand-maintained replica of the server's
/// version; now it is the request-shaped adapter only, so the test measures
/// what the server will actually charge rather than what a copy believes.
fn contract_prompt_tokens(body: &serde_json::Value) -> u64 {
    let messages = body["messages"].as_array().expect("messages array");
    let tools = body
        .get("tools")
        .and_then(|t| t.as_array())
        .map(Vec::as_slice);
    eidola_common::prompt_charge(messages, tools).chargeable_prompt_tokens()
}

/// **The pricing regression for the primary (eidola) backend.** Every round's
/// hold is computed over the exact bytes that round put on the wire —
/// including the advertised `tools` schema and the assistant's replayed
/// `tool_calls`, which the pre-task-25 contract counted as zero. Recomputing
/// the server's side of the contract from the recorded bodies must reproduce
/// each hold exactly, which is what makes hold ≥ charge structural for a tool
/// turn rather than merely likely.
///
/// The harness prices the model at 1 credit per token, scale factor 1, so
/// `hold = chargeable_prompt_tokens + max_completion_tokens` — the credits on
/// the rows are directly comparable to the contract's token counts.
#[test]
fn a_tool_rounds_hold_covers_the_tool_bytes_it_sends() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let result = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("tool turn succeeds");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2, "two rounds");

        // The second round genuinely carries the shapes that used to be free.
        assert!(bodies[1]["tools"].as_array().is_some_and(|t| !t.is_empty()));
        assert!(
            bodies[1]["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m.get("tool_calls").is_some()),
            "round 2 must replay the assistant's tool call"
        );

        // Hold ≥ charge, exactly: the sum of the holds equals the sum of the
        // server's recomputation over the same bodies.
        let expected: u64 = bodies
            .iter()
            .map(|b| {
                contract_prompt_tokens(b) + b["max_completion_tokens"].as_u64().expect("ceiling")
            })
            .sum();
        assert_eq!(
            result.credits_charged as u64, expected,
            "each round's hold must equal the contract over its own wire bytes"
        );

        // And the extension is load-bearing: the old content-only walk of
        // round 2 would have under-counted the very bytes that were sent.
        let mut content_only = eidola_common::PromptCharge::new();
        for message in bodies[1]["messages"].as_array().unwrap() {
            content_only.add_message(
                message
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.len() as u64)
                    .unwrap_or(0),
            );
        }
        assert!(
            content_only.chargeable_prompt_tokens() < contract_prompt_tokens(&bodies[1]),
            "the tools schema and replayed tool call must contribute chargeable bytes"
        );
    });
}

/// The context assembly is the **ordered composition of the prompt**, so a
/// replayed trace occupies the position it actually occupied on the wire —
/// between the post it answered and the answer it produced — not a lump at the
/// end. Only the rounds the *current* turn generated come last, because that is
/// where they were appended live.
#[test]
fn the_context_assembly_records_the_order_the_messages_were_sent_in() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let first = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("turn 1");
        let second = core
            .runtime()
            .block_on(core.chat(
                "and now?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
            ))
            .expect("turn 2");

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(first.space_id.clone()))
            .expect("raw actions");
        let id_of = |n: usize| actions[n].id.clone();
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "user_input",
                "tool_call",
                "tool_result",
                "inference",
                "user_input",
                "inference"
            ]
        );
        let (post1, call, result, inference1, post2) =
            (id_of(0), id_of(1), id_of(2), id_of(3), id_of(4));

        // Turn 1: the post it answered, then the round it ran (appended live).
        assert_eq!(
            core.runtime()
                .block_on(core.test_context_assembly(inference1.clone()))
                .expect("assembly"),
            vec![post1.clone(), call.clone(), result.clone()],
        );

        // Turn 2: the same round, now *replayed*, sits where it was sent —
        // before the answer it produced, not after the whole conversation.
        assert_eq!(
            core.runtime()
                .block_on(
                    core.test_context_assembly(
                        second.response_action_id.clone().expect("an answer")
                    )
                )
                .expect("assembly"),
            vec![post1, call, result, inference1, post2],
        );

        // …which is exactly the order of the posts and traces on the wire.
        let last = mock.chat_bodies().last().expect("turn 2's request").clone();
        let shape: Vec<String> = last["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .skip(1) // the system message is not an action
            .map(|m| match (m["role"].as_str(), m.get("tool_calls")) {
                (Some("assistant"), Some(_)) => "tool_call".to_string(),
                (Some("tool"), _) => "tool_result".to_string(),
                (Some(r), _) => r.to_string(),
                _ => "?".to_string(),
            })
            .collect();
        assert_eq!(
            shape,
            vec!["user", "tool_call", "tool_result", "assistant", "user"],
        );
    });
}

/// A regeneration replays nothing of the generation it replaces — not the
/// answer (`get_upstream_context` withholds it), and not the working that
/// produced it. A trace belongs to the turn whose post you can see, and a
/// `Revise` turn is precisely the one that cannot see that post.
#[test]
fn regenerate_does_not_replay_the_traces_of_the_generation_it_replaces() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let first = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("tool turn succeeds");
        core.runtime()
            .block_on(core.regenerate(
                first.response_action_id.clone().expect("an answer"),
                MODEL.into(),
            ))
            .expect("regenerate succeeds");

        let bodies = mock.chat_bodies();
        let regen = bodies.last().expect("the regenerate request");
        let msgs = regen["messages"].as_array().expect("messages");
        assert_eq!(
            msgs.iter()
                .map(|m| m["role"].as_str().unwrap_or_default())
                .collect::<Vec<_>>(),
            vec!["system", "user"],
            "only the post being answered: {regen}"
        );
        assert!(msgs.iter().all(|m| m.get("tool_calls").is_none()));
    });
}

/// The same pricing regression for a **replayed** trace (task 33): a later
/// turn's hold is computed over its own wire bytes, so the trace rounds it
/// reads back are charged by exactly the walk the server will run over the
/// same array — the task-25 tool-entry rule applies to a replayed call
/// identically to a live one, because it is the same function over the same
/// bytes.
#[test]
fn a_later_turns_hold_covers_the_traces_it_replays() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let first = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("turn 1");
        let second = core
            .runtime()
            .block_on(core.chat(
                "and now?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
            ))
            .expect("turn 2");

        let bodies = mock.chat_bodies();
        let last = bodies.last().expect("turn 2's request");
        assert!(
            last["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m.get("tool_calls").is_some()),
            "turn 2 replays turn 1's round: {last}"
        );

        assert_eq!(
            second.credits_charged as u64,
            contract_prompt_tokens(last) + last["max_completion_tokens"].as_u64().expect("ceiling"),
            "the hold equals the contract over the bytes actually sent"
        );

        // The replayed call and its result really are chargeable bytes — a
        // content-only walk of the same array under-counts.
        let mut content_only = eidola_common::PromptCharge::new();
        for message in last["messages"].as_array().unwrap() {
            content_only.add_message(
                message
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(|s| s.len() as u64)
                    .unwrap_or(0),
            );
        }
        assert!(content_only.chargeable_prompt_tokens() < contract_prompt_tokens(last));
    });
}

/// The SSE twin: a tool call assembled from four partial deltas (function name
/// in two pieces, arguments in two) drives exactly the same loop.
#[test]
fn streamed_tool_call_is_assembled_across_deltas() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsStreaming(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let (tx, mut rx_events) = tokio::sync::mpsc::unbounded_channel();
        let result = core
            .runtime()
            .block_on(core.chat_stream("use the tool".into(), MODEL.into(), None, tx))
            .expect("streaming tool turn succeeds");

        assert_eq!(result.content, TOOL_FINAL_CONTENT);
        assert_eq!(mock.chat_hits(), 2);

        // The tool round is an invisible pause: the caller only ever sees the
        // final round's deltas (the tool round emitted no content).
        let mut streamed = String::new();
        while let Ok(ev) = rx_events.try_recv() {
            if let ChatStreamEvent::ContentDelta(d) = ev {
                streamed.push_str(&d);
            }
        }
        assert_eq!(streamed, TOOL_FINAL_CONTENT);

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(result.space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(
            kinds,
            vec!["user_input", "tool_call", "tool_result", "inference"]
        );
        // The name and arguments were reassembled from their pieces.
        assert_eq!(actions[1].blocks[0].tool_name.as_deref(), Some(TOOL_NAME));
        assert_eq!(
            actions[1].blocks[0].data.as_deref(),
            Some(r#"{"text":"round-1"}"#)
        );
        assert_eq!(
            actions[2].blocks[0].text_content.as_deref(),
            Some(tool_result_text(1).as_str())
        );
    });
}

/// A model that keeps asking for tools hits the round cap. The turn ends with
/// a typed `ToolLoop` error wrapped with the space id — never a silently
/// truncated answer — and every round it did run stays persisted.
#[test]
fn round_cap_ends_the_turn_honestly_with_the_rounds_persisted() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            // Never stops asking.
            chat: ChatBehavior::ToolRoundsBlocking(u64::MAX),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("loop forever".into(), MODEL.into(), None))
            .expect_err("the round cap must fail the turn");

        let space_id = only_space(&core);
        assert!(
            matches!(err, AppError::ToolLoop { .. }),
            "expected a typed ToolLoop error, got {err:?}"
        );

        // Exactly the cap's worth of model requests, no more.
        assert_eq!(mock.chat_hits(), eidola_app_core::MAX_TURN_ROUNDS as u64);

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        // The post, then (cap - 1) executed rounds, then the capped round's
        // tool_call — whose tools were deliberately NOT executed, because their
        // results could never be sent.
        let mut expected = vec!["user_input"];
        for _ in 1..eidola_app_core::MAX_TURN_ROUNDS {
            expected.push("tool_call");
            expected.push("tool_result");
        }
        expected.push("tool_call");
        assert_eq!(kinds, expected);
        assert!(
            !kinds.contains(&"inference"),
            "no answer was produced, so no inference may be persisted"
        );

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Space(space_id)),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");
    });
}

/// **A settled hold announces itself even when the turn then fails.**
///
/// Settlement is a durable commit — the credential leaves `spending` the
/// instant its successor is written — and the crate's rule is that every
/// durable commit emits. The place that is easy to get wrong is the
/// last-chance settlement inside `begin_next_round`: it commits, and then the
/// budget check right below it can end the turn. A caller-side emission would
/// never run, and every subscriber would go on showing a credential as
/// spending that has in fact been spent, until something else refreshed it.
/// So the emission belongs to the write, not to any caller.
///
/// The fixture is a **transient** settlement failure: round 1's own recovery
/// 500s (leaving the hold unsettled), `begin_next_round`'s retry succeeds, and
/// the budget then binds — commit, then failure, in that order.
#[test]
fn a_settlement_that_lands_before_a_failed_round_still_emits_wallet() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(u64::MAX),
            // Round 1's in-round recovery fails; the next attempt — which is
            // `begin_next_round`'s — succeeds.
            refund: RefundMode::FailFirst(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        // Round 1's exact hold, so the budget admits it and refuses round 2.
        core.runtime()
            .block_on(core.test_chat_with_budget("budget me".into(), MODEL.into(), None, i64::MAX))
            .expect_err("the probe hits the round cap");
        let probe_space = only_space(&core);
        let round1 = core
            .runtime()
            .block_on(core.test_space_actions(probe_space))
            .expect("raw actions")
            .into_iter()
            .find(|a| a.action_type == "tool_call")
            .and_then(|a| a.credits_consumed)
            .expect("round 1 recorded its hold");

        let mut rx = core.subscribe_changes();
        let err = core
            .runtime()
            .block_on(core.test_chat_with_budget("budget me".into(), MODEL.into(), None, round1))
            .expect_err("round 2 must exceed the budget");
        assert!(
            matches!(err, AppError::Credential { .. }),
            "the budget refusal, not a settlement refusal: {err:?}"
        );

        // Exactly one hold was acquired (round 1's — the turn died before
        // round 2's), so a single `Wallet` would be the spend-start alone.
        // The second is the settlement that landed just before the refusal.
        let wallets = drain(&mut rx)
            .into_iter()
            .filter(|c| matches!(c, Change::Wallet))
            .count();
        assert_eq!(
            wallets, 2,
            "spend start, then the successor credential's commit"
        );

        // And the durable half agrees: nothing is left holding.
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("lifecycle");
        assert!(
            lifecycle.iter().all(|c| c.state != "spending"),
            "the retried settlement really did land; got {lifecycle:?}"
        );
    });
}

/// The per-turn `budget` is checked **per round**: round 1 fits, round 2's
/// estimate over the grown messages array does not. The turn fails with a
/// typed error and round 1 stays durable.
#[test]
fn budget_exceeded_mid_loop_fails_with_the_first_round_persisted() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(u64::MAX),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);
        let mut rx = core.subscribe_changes();

        // Find round 1's exact estimate by running an unbudgeted single-round
        // turn against the same mock pricing, then budget exactly that: round 1
        // fits, round 2 (a strictly longer messages array) cannot.
        core.runtime()
            .block_on(core.test_chat_with_budget("budget me".into(), MODEL.into(), None, i64::MAX))
            .expect_err("the probe also hits the cap eventually");
        let probe_space = only_space(&core);
        let round1 = core
            .runtime()
            .block_on(core.test_space_actions(probe_space))
            .expect("raw actions")
            .into_iter()
            .find(|a| a.action_type == "tool_call")
            .and_then(|a| a.credits_consumed)
            .expect("round 1 recorded its hold");

        let hits_before = mock.chat_hits();
        drain(&mut rx);

        let err = core
            .runtime()
            .block_on(core.test_chat_with_budget("budget me".into(), MODEL.into(), None, round1))
            .expect_err("round 2 must exceed the budget");
        // The probe above left a space of its own, so this is the newer one.
        let space_id = latest_space(&core);
        let message = err.to_string();
        assert!(
            matches!(err, AppError::Credential { .. }),
            "expected the budget refusal, got {err:?}"
        );
        assert!(message.contains("budget"), "got {message}");

        // Only round 1 was sent; the budget stopped round 2 before the wire.
        assert_eq!(mock.chat_hits(), hits_before + 1);

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(
            kinds,
            vec!["user_input", "tool_call", "tool_result"],
            "the round that did run is durable"
        );

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Space(space_id)),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");
    });
}

/// A tool call with no `id` is structurally unusable — there is nothing to
/// execute and nothing that can be written as a `tool_use` block (the schema
/// requires the id). The turn fails honestly, records the raw exchange, and
/// does not panic.
#[test]
fn structurally_malformed_tool_call_fails_the_turn_honestly() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolCallMalformed,
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("break it".into(), MODEL.into(), None))
            .expect_err("a malformed tool call must fail the turn");
        let space_id = only_space(&core);
        assert!(
            matches!(err, AppError::ToolLoop { .. }),
            "expected a typed ToolLoop error, got {err:?}"
        );
        assert_eq!(mock.chat_hits(), 1, "the loop stops at the bad round");

        // No action could be written, but the raw exchange is still in the
        // Record (attached to no action).
        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(kinds, vec!["user_input"]);

        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        assert_eq!(
            requests
                .iter()
                .filter(|r| r.path == "/v1/chat/completions")
                .count(),
            1,
            "the unusable round's raw exchange is recorded (attached to no action, \
             since only the post exists)"
        );

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Space(space_id)),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");
    });
}

/// Arguments that aren't valid JSON are a *model* mistake, not a turn failure:
/// the loop reports the parse error back as the tool result and carries on, so
/// the model can correct itself on the next round.
#[test]
fn invalid_tool_arguments_are_reported_back_to_the_model() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolCallBadArguments,
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let result = core
            .runtime()
            .block_on(core.chat("bad args".into(), MODEL.into(), None))
            .expect("the turn survives a model argument mistake");
        assert_eq!(result.content, TOOL_FINAL_CONTENT);
        assert_eq!(mock.chat_hits(), 2);

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(result.space_id.clone()))
            .expect("raw actions");
        let tool_result = actions
            .iter()
            .find(|a| a.action_type == "tool_result")
            .expect("a tool_result action");
        // The failure is visible in the trail…
        assert_eq!(tool_result.status, "error");
        let text = tool_result.blocks[0]
            .text_content
            .clone()
            .unwrap_or_default();
        assert!(
            text.starts_with("error: arguments are not valid JSON"),
            "got {text}"
        );
        // …and it is exactly what the model was shown.
        let bodies = mock.chat_bodies();
        let msgs = bodies[1]["messages"].as_array().expect("messages");
        assert_eq!(msgs[msgs.len() - 1]["content"], text);
    });
}

/// An unknown tool name is likewise a model mistake, reported as a tool error
/// rather than failing the turn — this is the path a registry-mismatch takes.
#[test]
fn unknown_tool_name_is_reported_back_to_the_model() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        // Deliberately register nothing… but then the request would carry no
        // `tools` and the mock would still call one. That is exactly the
        // registry-mismatch case: a model calling a tool we don't have.
        let result = core
            .runtime()
            .block_on(core.chat("call something".into(), MODEL.into(), None))
            .expect("the turn survives an unknown tool");
        assert_eq!(result.content, TOOL_FINAL_CONTENT);
        assert_eq!(mock.chat_hits(), 2);

        let bodies = mock.chat_bodies();
        assert!(bodies[0].get("tools").is_none());
        let msgs = bodies[1]["messages"].as_array().expect("messages");
        let last = msgs[msgs.len() - 1]["content"].as_str().unwrap_or_default();
        assert!(last.starts_with("error: unknown tool `echo`"), "got {last}");
    });
}

/// **First-person traces (task 33).** A later turn of the *same* participant
/// reads its own prior tool rounds back, inline where they happened — between
/// the post that prompted them and the answer they produced — in the wire shape
/// the model emitted them in: an `assistant` message with `content: null` and
/// its `tool_calls`, then one `tool` message per result.
///
/// (The other half of the rule — that another participant's traces never
/// appear — is `another_participants_traces_never_reach_this_turn` in
/// `tests/participants_orchestration.rs`, which needs two agents.)
#[test]
fn a_later_turn_replays_its_own_tool_rounds_inline() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let first = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("first turn succeeds");
        // Rounds 1-2 are the first turn; the mock answers plainly from here on
        // (its scripted tool round is spent).
        core.runtime()
            .block_on(core.chat(
                "and now?".into(),
                MODEL.into(),
                Some(first.space_id.clone()),
            ))
            .expect("second turn succeeds");

        let bodies = mock.chat_bodies();
        let last = bodies.last().expect("a third request");
        let msgs = last["messages"].as_array().expect("messages");
        let roles: Vec<&str> = msgs
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "tool", "assistant", "user"],
            "the round sits between the post it answered and the answer it produced"
        );

        // The replayed call: canonical OpenAI shape, carrying the persisted
        // id, name and raw arguments.
        assert_eq!(msgs[2]["content"], serde_json::Value::Null);
        let calls = msgs[2]["tool_calls"].as_array().expect("tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], tool_call_id(1));
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], TOOL_NAME);
        assert_eq!(calls[0]["function"]["arguments"], r#"{"text":"round-1"}"#);

        // The replayed result: raw, keyed by call id — a tool result is not a
        // post and wears no header.
        assert_eq!(msgs[3]["tool_call_id"], tool_call_id(1));
        assert_eq!(msgs[3]["content"], tool_result_text(1));

        // The answer the round produced follows it, as an ordinary headed post.
        let answer_item = item_id_of(&core, &first.space_id, TOOL_FINAL_CONTENT);
        let stamps = Stamps::of(&core, &first.space_id);
        assert_eq!(
            msgs[4]["content"],
            stamps.headed(&answer_item, DEFAULT_AGENT_LABEL, TOOL_FINAL_CONTENT)
        );

        // Exactly once — a trace is attributed to the turn that produced it,
        // not re-emitted by every later turn that replayed it.
        assert_eq!(
            msgs.iter()
                .filter(|m| m.get("tool_calls").is_some())
                .count(),
            1
        );
    });
}

/// The trunk of one participant's branch is **append-only**, byte for byte:
/// every request's `messages` array is a literal prefix of every later one's,
/// tool rounds included. That is the no-elision invariant — traces stay inline
/// where they happened until an explicit consolidation event, so nothing above
/// the tail ever moves and an upstream prefix cache keeps its whole hit.
#[test]
fn a_participants_trunk_is_append_only_across_turns() {
    run(|| {
        let script = chat_harness::tool_script();
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolScript,
            tool_script: script.clone(),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        // Three turns, the first two with a tool round each (the script is
        // consumed by the next request, so it is refilled per turn).
        *script.lock().unwrap() = vec![(TOOL_NAME.to_string(), r#"{"text":"first"}"#.to_string())];
        let first = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("turn 1");
        let space = first.space_id.clone();

        *script.lock().unwrap() = vec![(TOOL_NAME.to_string(), r#"{"text":"second"}"#.to_string())];
        core.runtime()
            .block_on(core.chat("again".into(), MODEL.into(), Some(space.clone())))
            .expect("turn 2");

        core.runtime()
            .block_on(core.chat("and now?".into(), MODEL.into(), Some(space.clone())))
            .expect("turn 3");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 5, "two 2-round turns and one plain turn");

        // Compare serialized messages: an explicit byte statement, not a
        // structural one.
        let serialized = |body: &serde_json::Value| -> Vec<String> {
            body["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .map(|m| m.to_string())
                .collect()
        };
        let arrays: Vec<Vec<String>> = bodies.iter().map(serialized).collect();
        for (i, window) in arrays.windows(2).enumerate() {
            let (earlier, later) = (&window[0], &window[1]);
            assert!(
                later.len() >= earlier.len(),
                "request {} shrank the context: {} → {}",
                i + 1,
                earlier.len(),
                later.len()
            );
            assert_eq!(
                &later[..earlier.len()],
                &earlier[..],
                "request {} moved a byte of the trunk",
                i + 1
            );
        }
        // And the traces really are in there (the invariant would hold
        // vacuously otherwise).
        let last = arrays.last().expect("a last request");
        assert_eq!(
            last.iter()
                .filter(|m| m.contains(r#""tool_calls""#))
                .count(),
            2,
            "both prior rounds are still inline: {last:#?}"
        );
    });
}

/// A turn that ended on a trace (the round cap) leaves a `tool_result` as the
/// literal newest action in the space. The next post must still continue the
/// **thread**, not hang off that trace — `db::last_action_in_space` returns the
/// last *post*, so the space keeps one root instead of silently growing a
/// second.
#[test]
fn a_post_after_a_trace_ending_turn_threads_under_the_last_post() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(u64::MAX),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        core.runtime()
            .block_on(core.chat("loop forever".into(), MODEL.into(), None))
            .expect_err("the round cap fails the turn");
        let space_id = only_space(&core);

        core.runtime()
            .block_on(core.post("still there?".into(), Some(space_id.clone())))
            .expect("a follow-up post");

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space_id.clone()))
            .expect("raw actions");
        let first_post = actions
            .iter()
            .find(|a| a.action_type == "user_input")
            .expect("the original post");
        let follow_up = actions
            .iter()
            .rfind(|a| a.action_type == "user_input")
            .expect("the follow-up post");
        assert_ne!(first_post.id, follow_up.id);
        assert_eq!(
            follow_up.reply_to.as_deref(),
            Some(first_post.id.as_str()),
            "the follow-up continues the thread, not the abandoned trace"
        );

        // One thread, two posts — the trace stays invisible.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space_id))
            .expect("tree");
        let kinds: Vec<&str> = tree.iter().map(|n| n.action_type.as_str()).collect();
        assert_eq!(kinds, vec!["user_input", "user_input"]);
        assert!(tree.iter().all(|n| !n.is_branch));
    });
}

/// A `tool_calls` value that is present and non-null but **not an array** is
/// structurally unusable — there is nothing to execute and nothing that can be
/// written as a `tool_use` block. It must take the same honest `ToolLoop` exit
/// as a call with no id, never be read as "the model requested no tools" and
/// persisted as a successful (empty) answer.
#[test]
fn non_array_tool_calls_fails_the_turn_rather_than_answering_empty() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolCallsNotAnArray,
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);
        let mut rx = core.subscribe_changes();

        let err = core
            .runtime()
            .block_on(core.chat("break it".into(), MODEL.into(), None))
            .expect_err("a non-array tool_calls must fail the turn");
        let space_id = only_space(&core);
        assert!(
            matches!(err, AppError::ToolLoop { .. }),
            "expected a typed ToolLoop error, got {err:?}"
        );
        assert_eq!(mock.chat_hits(), 1);

        // Critically: no `inference` was persisted. The bug this pins would
        // have committed an empty answer as a successful turn.
        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(kinds, vec!["user_input"]);

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Space(space_id)),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");
    });
}

/// The streaming twin: a `delta.tool_calls` that is present and non-null but
/// not an array takes the same exit.
#[test]
fn non_array_streamed_tool_calls_fails_the_turn() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolCallsNotAnArrayStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);
        let mut rx = core.subscribe_changes();

        let (tx, _rx_events) = tokio::sync::mpsc::unbounded_channel();
        let err = core
            .runtime()
            .block_on(core.chat_stream("break it".into(), MODEL.into(), None, tx))
            .expect_err("a non-array delta.tool_calls must fail the turn");
        let space_id = only_space(&core);
        assert!(
            matches!(err, AppError::ToolLoop { .. }),
            "expected a typed ToolLoop error, got {err:?}"
        );
        assert_eq!(mock.chat_hits(), 1);

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space_id.clone()))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(kinds, vec!["user_input"]);

        let changes = drain(&mut rx);
        assert!(
            changes.contains(&Change::Space(space_id)),
            "got {changes:?}"
        );
        assert!(changes.contains(&Change::Record), "got {changes:?}");
    });
}

/// …but an explicit `"tool_calls": null` means "no tools" and stays an
/// ordinary success. Some providers always emit the key; rejecting that would
/// break every turn against them.
#[test]
fn explicitly_null_tool_calls_is_an_ordinary_completion() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolCallsNull,
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let result = core
            .runtime()
            .block_on(core.chat("hello".into(), MODEL.into(), None))
            .expect("a null tool_calls is not an error");
        assert_eq!(result.content, chat_harness::TOOL_FINAL_CONTENT);

        let actions = core
            .runtime()
            .block_on(core.test_space_actions(result.space_id))
            .expect("raw actions");
        let kinds: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(kinds, vec!["user_input", "inference"]);
    });
}

/// Streamed tool calls honor the same provider-field preservation contract as
/// blocking ones: the follow-up assistant message replays every field the
/// provider sent, not just the canonical `id` / `type` / `function` triple.
///
/// Also pins the merge rule for fields that arrive more than once — **last
/// non-null wins**, shallow, at both levels — and that the streaming-only
/// `index` framing key is not replayed.
#[test]
fn streamed_tool_calls_preserve_provider_specific_fields() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsStreamingWithExtras(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let (tx, _rx_events) = tokio::sync::mpsc::unbounded_channel();
        let result = core
            .runtime()
            .block_on(core.chat_stream("use the tool".into(), MODEL.into(), None, tx))
            .expect("streaming tool turn succeeds");
        assert_eq!(result.content, chat_harness::TOOL_FINAL_CONTENT);

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2);
        let msgs = bodies[1]["messages"].as_array().expect("messages");
        let call = &msgs[msgs.len() - 2]["tool_calls"][0];

        // Canonical fields still assembled from their fragments.
        assert_eq!(call["id"], tool_call_id(1));
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], TOOL_NAME);
        assert_eq!(call["function"]["arguments"], r#"{"text":"round-1"}"#);

        // Provider fields survive — including a nested object, which no
        // concatenating merge could reproduce.
        assert_eq!(
            call["trace"]["span"], "s1",
            "a structured provider field must be replayed intact: {call}"
        );
        // Restated fields take the latest value (last-wins), at both levels.
        assert_eq!(
            call["provider_tag"], "beta",
            "a restated top-level provider field is last-wins: {call}"
        );
        assert_eq!(
            call["function"]["cache_key"], "k2",
            "a restated function-level provider field is last-wins: {call}"
        );

        // `index` is the SSE fragment key, not part of the assembled call.
        assert!(
            call.get("index").is_none(),
            "the streaming framing key must not be replayed: {call}"
        );
    });
}

// ===========================================================================
// Trace visibility (task 34) — the parallel read behind the space UI's
// disclosure. These exercise the real SQL end to end; the grouping rules
// themselves are unit-tested in `lib.rs`.
// ===========================================================================

#[test]
fn space_traces_anchor_a_turns_rounds_on_the_answer_it_produced() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(1),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        let result = core
            .runtime()
            .block_on(core.chat("use the tool".into(), MODEL.into(), None))
            .expect("tool turn succeeds");
        let inference = result
            .response_action_id
            .clone()
            .expect("the turn produced a post");

        let traces = core
            .runtime()
            .block_on(core.space_traces(result.space_id.clone()))
            .expect("trace read");
        assert_eq!(traces.len(), 1, "one turn, one disclosure: {traces:?}");
        let trace = &traces[0];
        assert_eq!(
            trace.anchor_action_id, inference,
            "the disclosure hangs under the answer, not the ask"
        );
        assert!(!trace.unanswered);
        assert_eq!(trace.entries.len(), 1);
        match &trace.entries[0] {
            eidola_app_core::TraceEntry::Tool {
                name,
                arguments,
                result,
                request_id,
                ..
            } => {
                assert_eq!(name, TOOL_NAME);
                assert_eq!(arguments, r#"{"text":"round-1"}"#);
                assert_eq!(result.as_deref(), Some(tool_result_text(1).as_str()));
                assert!(
                    request_id.is_some(),
                    "the round links through to its own raw exchange in the Record"
                );
            }
            other => panic!("expected a tool round, got {other:?}"),
        }

        // A later turn of the same participant replays its own rounds (task
        // 33) and records them in its assembly too — the round must still
        // render once, under the turn that ran it.
        let second = core
            .runtime()
            .block_on(core.chat(
                "and again".into(),
                MODEL.into(),
                Some(result.space_id.clone()),
            ))
            .expect("second turn");
        let traces = core
            .runtime()
            .block_on(core.space_traces(result.space_id.clone()))
            .expect("trace read");
        let anchors: Vec<&str> = traces.iter().map(|t| t.anchor_action_id.as_str()).collect();
        assert!(
            anchors.contains(&inference.as_str()),
            "the first turn keeps its own rounds: {anchors:?}"
        );
        assert_eq!(
            anchors.iter().filter(|a| **a == inference).count(),
            1,
            "and they are not duplicated onto the replaying turn: {anchors:?}"
        );
        assert!(second.response_action_id.is_some());
    });
}

#[test]
fn space_traces_anchor_a_capped_turn_on_the_post_it_answered() {
    run(|| {
        // The gap case: the round cap ends the turn with no inference at all,
        // so its trace has no answer to hang under — it belongs to the post it
        // was answering, which is exactly where the disclosure makes the
        // non-event visible.
        let (_mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::ToolRoundsBlocking(u64::MAX),
            ..MockConfig::default()
        });
        with_account(&core);
        with_echo_tool(&core);

        core.runtime()
            .block_on(core.chat("loop forever".into(), MODEL.into(), None))
            .expect_err("the round cap ends the turn");
        let space_id = only_space(&core);

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space_id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 1, "only the human post survives the cap");
        let post = tree[0].action_id.clone();

        let traces = core
            .runtime()
            .block_on(core.space_traces(space_id))
            .expect("trace read");
        assert_eq!(traces.len(), 1, "one disclosure: {traces:?}");
        assert_eq!(
            traces[0].anchor_action_id, post,
            "with no answer to hang under, it hangs under the ask"
        );
        assert!(traces[0].unanswered, "and says the turn left no post");
        assert!(
            traces[0].entries.len() > 1,
            "every round it ran is listed: {:?}",
            traces[0].entries
        );
    });
}

// ===========================================================================
// Identity, roster, and post timestamps (task 64)
//
// A model was never told which participant it is, nor who else is present.
// The two halves are split by volatility: identity is static per participant
// and rides the system message, in every space; the roster changes with
// membership and rides the trailing block, where recompute is free.
// ===========================================================================

/// The identity line is present in **every** space — a two-party linear one
/// included — and sits between the charter and the notes. That position is the
/// claim: identity governs what follows it.
#[test]
fn every_turn_states_which_participant_the_model_is() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        turn(&core, "How do tides work?", None, None);

        let system = flat_messages(mock.chat_bodies().last().expect("a request"))[0]
            .1
            .clone();
        assert_eq!(
            system,
            system_message(Some(SEEDED_SYSTEM_PROMPT), DEFAULT_AGENT_LABEL)
        );
        // Stated positionally, so a reordering fails here rather than silently:
        // charter, then identity, then the protocol note.
        assert_eq!(
            system.split("\n\n").collect::<Vec<_>>(),
            vec![
                SEEDED_SYSTEM_PROMPT,
                &format!("You are \"{DEFAULT_AGENT_LABEL}\" in this conversation."),
                chat_harness::HEADER_PROTOCOL_NOTE,
            ],
            "the identity line sits after the charter and before the notes"
        );
    });
}

/// A per-space label override renames the participant, and the identity line
/// follows it — the effective label is what the model's own posts are headed
/// with, so the two cannot disagree.
#[test]
fn the_identity_line_names_the_participants_effective_label() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let first = turn(&core, "How do tides work?", None, None);

        let agent = core
            .runtime()
            .block_on(core.list_space_participants(first.space_id.clone()))
            .expect("participants")
            .into_iter()
            .find(|p| p.kind == "agent")
            .expect("the seeded agent")
            .id;
        core.runtime()
            .block_on(core.update_space_participant(
                agent,
                eidola_app_core::ParticipantUpdate {
                    label: Some("Navigator".into()),
                    ..Default::default()
                },
                eidola_app_core::ExpectedScope::Any,
            ))
            .expect("rename");

        turn(&core, "And why two per day?", Some(first.space_id), None);
        assert_eq!(
            flat_messages(mock.chat_bodies().last().expect("a request"))[0].1,
            system_message(Some(SEEDED_SYSTEM_PROMPT), "Navigator"),
            "the identity line flips once, with the rename"
        );
    });
}

/// The roster gate, both ways. A two-participant linear space is not
/// multi-party, so it carries **no** roster and its turn has no trailing
/// message at all. Adding a third participant flips it on — and it stays
/// byte-identical across the turns that follow, because membership order is
/// stable and nothing else feeds it.
#[test]
fn the_roster_appears_only_once_a_space_is_multi_party() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();

        // Two participants, no branches: no trailing message whatsoever.
        let msgs = flat_messages(mock.chat_bodies().last().expect("a request"));
        assert_eq!(msgs.len(), 2, "system + the one post: {msgs:#?}");
        assert!(
            msgs.iter()
                .all(|(_, c)| !c.contains("Participants in this conversation:")),
            "a two-party linear space carries no roster: {msgs:#?}"
        );

        // A third participant joins.
        core.runtime()
            .block_on(core.add_space_participant(
                space.clone(),
                eidola_app_core::NewParticipant {
                    label: "Ada".into(),
                    model_ref: Some(MODEL.into()),
                    system_prompt: None,
                    notify_policy: "explicit".into(),
                },
            ))
            .expect("add agent");

        turn(&core, "And why two per day?", Some(space.clone()), None);
        let asked = item_id_of(&core, &space, "And why two per day?");
        let members = roster(&[
            (HUMAN_LABEL, "human", false),
            (DEFAULT_AGENT_LABEL, "agent", true),
            ("Ada", "agent", false),
        ]);
        // The roster-only shape — a linear space with three participants — is
        // still a trailing metadata message, so it closes with the same
        // `Respond to #h.` pointer the map shape does. Without it the roster
        // was the last thing the model read with nothing marking it as
        // metadata, and models answered the roster instead of the post (Codex
        // review, PR #294).
        let expected = trailing(Some(&members), None, &eidola_app_core::post_handle(&asked));
        let after_join = flat_messages(mock.chat_bodies().last().expect("a request"));
        assert_eq!(
            after_join.last().expect("a message").clone(),
            ("user".to_string(), expected.clone()),
            "the third member flips the roster on, in the trailing block"
        );
        assert_eq!(
            after_join[0].1,
            system_message_with(
                Some(SEEDED_SYSTEM_PROMPT),
                DEFAULT_AGENT_LABEL,
                &[TRAILING_BLOCK_NOTE]
            ),
            "and the system message frames it as metadata — with no map note, \
             because this space has no map"
        );

        // …and stays byte-identical while the membership does. (The pointer
        // moves with the post being answered, which is the one part of the
        // trailing message that is *supposed* to be volatile.)
        turn(&core, "Anything else?", Some(space.clone()), None);
        let asked_again = item_id_of(&core, &space, "Anything else?");
        assert_eq!(
            flat_messages(mock.chat_bodies().last().expect("a request"))
                .last()
                .expect("a message")
                .clone(),
            (
                "user".to_string(),
                trailing(
                    Some(&members),
                    None,
                    &eidola_app_core::post_handle(&asked_again)
                )
            ),
            "stable within a membership: the roster's bytes move only when it does"
        );
    });
}

/// The other arm of the gate: a **branched** two-party space is multi-party in
/// the sense that matters — there is structure the model cannot see — so the
/// roster rides beside the map, the map second. The `Respond to #h.` pointer
/// closes the whole message, after both.
#[test]
fn a_branch_turns_the_roster_on_in_a_two_party_space() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        let i1 = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree")[1]
            .action_id
            .clone();
        turn(&core, "And why two per day?", Some(space.clone()), None);
        turn(
            &core,
            "What about spring tides?",
            Some(space.clone()),
            Some(i1),
        );

        let trailing_block = flat_messages(mock.chat_bodies().last().expect("a request"))
            .last()
            .expect("a message")
            .1
            .clone();
        let expected_roster = roster(&[
            (HUMAN_LABEL, "human", false),
            (DEFAULT_AGENT_LABEL, "agent", true),
        ]);
        assert!(
            trailing_block.starts_with(&expected_roster),
            "the roster opens the trailing block: {trailing_block}"
        );
        assert!(
            trailing_block.contains("</thread-map>"),
            "the map follows it: {trailing_block}"
        );
        let branch_item = item_id_of(&core, &space, "What about spring tides?");
        assert!(
            trailing_block.ends_with(&format!(
                "</thread-map>\n\nRespond to #{}.",
                eidola_app_core::post_handle(&branch_item)
            )),
            "and the response pointer closes the message, after both: {trailing_block}"
        );
    });
}

/// The header stamp is absolute, RFC 3339 UTC at seconds precision, and — the
/// property that matters — **byte-stable for a given post**: the same post
/// re-read on a later turn is the same header bytes, which is what lets the
/// trunk stay identical across turns.
#[test]
fn post_headers_carry_a_stable_absolute_stamp() {
    run(|| {
        let (mock, core, _dir) = setup(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let first = turn(&core, "How do tides work?", None, None);
        let space = first.space_id.clone();
        let u1 = item_id_of(&core, &space, "How do tides work?");

        let stamps = Stamps::of(&core, &space);
        let header = flat_messages(mock.chat_bodies().last().expect("a request"))[1]
            .1
            .lines()
            .next()
            .expect("a header line")
            .to_string();
        assert_eq!(
            header,
            format!(
                "#{} · {HUMAN_LABEL} · {}",
                eidola_app_core::post_handle(&u1),
                eidola_app_core::post_stamp(stamps.at(&u1))
            )
        );
        // Shape, spelled out rather than derived: 20 characters, `Z`-suffixed,
        // seconds precision, no offset and no fractional part.
        let stamp = header.rsplit(" · ").next().expect("a stamp");
        assert_eq!(stamp.len(), 20, "{stamp}");
        assert!(stamp.ends_with('Z') && stamp.contains('T'), "{stamp}");
        assert!(!stamp.contains('.') && !stamp.contains('+'), "{stamp}");

        // Two more turns re-read the same post. Its header must not drift —
        // a relative stamp would move on every one of them.
        turn(&core, "And why two per day?", Some(space.clone()), None);
        turn(&core, "Anything else?", Some(space), None);
        for body in mock.chat_bodies() {
            for (_, content) in flat_messages(&body) {
                let first_line = content.lines().next().unwrap_or_default();
                if first_line.starts_with(&format!("#{}", eidola_app_core::post_handle(&u1))) {
                    assert_eq!(first_line, header, "the stamp must not drift between turns");
                }
            }
        }
    });
}
