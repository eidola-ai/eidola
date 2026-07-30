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
    THREAD_MAP_NOTE, THREAD_MAP_TOOLS_NOTE, flat_messages, headed, map_entry, system_message,
    system_message_with, thread_map, with_account,
};
use eidola_app_core::changes::Change;
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
fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    loop {
        match rx.try_recv() {
            Ok(c) => out.push(c),
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

/// Build a mock + a core wired to it (see `chat_harness::core_for`). Callers
/// add an account via `with_account` when they want the auto-provisioning path.
fn setup(config: MockConfig) -> (MockServer, AppCore, tempfile::TempDir) {
    chat_harness::core_for(config)
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
            "You + one agent; got {participants:?}"
        );
        let human = participants
            .iter()
            .find(|p| p.kind == "human")
            .expect("human participant");
        assert_eq!(human.label, "You");
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
        assert_eq!(user_post.participant.label, "You");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("inference post");
        assert_eq!(inference.participant.kind, "agent");
        assert!(!inference.action_id.is_empty());
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
        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 3, "two chats + one regenerate");
        assert_eq!(
            flat_messages(&bodies[2]),
            vec![
                (
                    "system".to_string(),
                    system_message(Some(SEEDED_SYSTEM_PROMPT))
                ),
                (
                    "user".to_string(),
                    headed(
                        &u1_item,
                        HUMAN_LABEL,
                        "How do tides work? Explain it for a sailor."
                    )
                )
            ],
            "regenerate context = system prompt + upstream only, most-recent versions"
        );

        // The handle is derived from the ITEM id, so the edit did not change
        // it: the bytes the model already read keep naming the same post.
        assert_eq!(
            flat_messages(&bodies[0])[1].1,
            headed(&u1_item, HUMAN_LABEL, "How do tides work?"),
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

/// Quoted references (wave 1): a post carrying a `{{ embed N }}` marker sends
/// the referenced passage upstream as a markdown blockquote in that post's
/// message — the model reads what was quoted, never the opaque marker.
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
        // wire must too; expansion is structural, not line-based).
        let posted = core
            .runtime()
            .block_on(
                core.post_with_references(
                    "What does this mean?\n\n{{ embed 1 }}\n\nAnd {{ embed 9 }} is unmapped.\n\n\
                 ```\n\n{{ embed 1 }}\n\n```"
                        .into(),
                    Some(source.space_id.clone()),
                    None,
                    vec![eidola_app_core::ReferenceSpec {
                        antecedent_action_id: source.action_id.clone(),
                        content_block_id: Some(block_id),
                        range_start: Some(24),
                        range_end: Some(34), // "powerhouse"
                        annotation: None,
                    }],
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
        assert_eq!(
            contents[2],
            headed(
                &posted.item_id,
                HUMAN_LABEL,
                "What does this mean?\n\n> powerhouse\n\nAnd {{ embed 9 }} is unmapped.\n\n\
                 ```\n\n{{ embed 1 }}\n\n```"
            ),
            "structural marker expands; unmapped and fence-defused markers go upstream literal"
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
        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 3, "two spine turns + one branch reply");
        assert_eq!(
            flat_messages(&bodies[2]),
            vec![
                (
                    "system".to_string(),
                    system_message_with(Some(SEEDED_SYSTEM_PROMPT), &[THREAD_MAP_NOTE])
                ),
                (
                    "user".to_string(),
                    headed(&u1_item, HUMAN_LABEL, "How do tides work?")
                ),
                (
                    "assistant".to_string(),
                    headed(&i1_item, DEFAULT_AGENT_LABEL, "Hello from the stream.")
                ),
                (
                    "user".to_string(),
                    headed(&branch_item, HUMAN_LABEL, "What about spring tides?")
                ),
                (
                    "user".to_string(),
                    thread_map(
                        &[(
                            format!("at #{}", eidola_app_core::post_handle(&i1_item)),
                            vec![map_entry(
                                &u2_item,
                                HUMAN_LABEL,
                                "2 posts",
                                "just now",
                                "And why two per day?",
                            )],
                        )],
                        &eidola_app_core::post_handle(&branch_item),
                    )
                ),
            ],
            "branch reply context = system prompt + the branch's ancestry + the trailing map"
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

/// The linear common case sends **no** map, no map note, and no `tools` field —
/// byte-identical to what it sent before task 21. This is the pin that keeps
/// the overwhelming majority of turns (and their upstream prefix caches)
/// untouched by the feature.
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
                system_message(Some(SEEDED_SYSTEM_PROMPT)),
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

        // The map is the LAST message — the placement decision. Everything
        // above it is conversation.
        assert_eq!(
            msgs.last().expect("a message").clone(),
            (
                "user".to_string(),
                thread_map(
                    &[(
                        format!("at #{}", eidola_app_core::post_handle(&i1_item)),
                        vec![map_entry(
                            &branch_item,
                            HUMAN_LABEL,
                            // The branch's own ask plus the answer it drew.
                            "2 posts",
                            "just now",
                            "What about spring tides?",
                        )],
                    )],
                    &eidola_app_core::post_handle(&last_item),
                ),
            ),
        );
        assert_eq!(
            msgs[0].1,
            system_message_with(Some(SEEDED_SYSTEM_PROMPT), &[THREAD_MAP_NOTE]),
            "the map note joins the system message; the tools note does not \
             (the eidola backend cannot carry a `tools` field yet)"
        );
        assert!(
            bodies[3].get("tools").is_none(),
            "no tools on the eidola path until task 25: {}",
            bodies[3]
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
            system_message(Some(SEEDED_SYSTEM_PROMPT)),
            "the pre-branch turn's system message is the pre-task-21 one"
        );
        assert!(
            a.iter().all(|(_, c)| !c.contains("<thread-map>")),
            "branch A forked a linear space: no map yet"
        );

        // And in both branched turns the map is the LAST message — all the
        // volatility at the tail.
        for msgs in [&b, &c] {
            assert!(
                msgs.last()
                    .expect("a message")
                    .1
                    .starts_with("<thread-map>"),
                "the map is the last message: {msgs:#?}"
            );
            assert_eq!(
                msgs.iter()
                    .filter(|(_, c)| c.starts_with("<thread-map>"))
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
// These run over an `openai` backend rather than the eidola one, because that
// is the real production gate: `prepare_turn` attaches the navigation tools
// only when the space has branches AND the backend can carry a `tools` field,
// and the Eidola server rejects unknown body fields today (see
// `backend_accepts_tools`). Testing through the gate rather than around it is
// what makes the eidola-path assertion above meaningful.
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
        core.runtime()
            .block_on(core.post_reply(
                "What about spring tides?".into(),
                Some(space.clone()),
                Some(i1),
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
        let results: Vec<String> = flat_messages(&bodies[3])
            .into_iter()
            .filter(|(role, _)| role == "tool")
            .map(|(_, c)| c)
            .collect();
        assert_eq!(results.len(), 4, "one result per call, in order");

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
            results[1].contains(&headed(&u2_item, HUMAN_LABEL, "And why two per day?")),
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

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2);

        // Turn 1: system + the single human post.
        assert_eq!(
            flat_messages(&bodies[0]),
            vec![
                (
                    "system".to_string(),
                    system_message(Some(SEEDED_SYSTEM_PROMPT))
                ),
                (
                    "user".to_string(),
                    headed(&tree[0].item_id, HUMAN_LABEL, "How do tides work?")
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
                    system_message(Some(SEEDED_SYSTEM_PROMPT))
                ),
                (
                    "user".to_string(),
                    headed(&tree[0].item_id, HUMAN_LABEL, "How do tides work?")
                ),
                (
                    "assistant".to_string(),
                    headed(
                        &tree[1].item_id,
                        DEFAULT_AGENT_LABEL,
                        "Hello from the stream."
                    )
                ),
                (
                    "user".to_string(),
                    headed(&tree[2].item_id, HUMAN_LABEL, "And why two per day?")
                ),
            ]
        );
    });
}

/// Strip-on-receipt: a model that mimics the visible header scaffolding has
/// that leading line removed before anything is persisted — the durable post
/// and the returned `ChatResult` both carry the bare answer.
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

        // The deltas reach the caller verbatim (the emission contract is
        // untouched); the *persisted* and *reported* text is stripped.
        assert!(
            streamed.starts_with('#'),
            "the raw deltas are forwarded as received: {streamed:?}"
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
        // root() still routes onboarding; the error now carries the persisted
        // space id (the post survived the funding failure).
        assert!(matches!(err.root(), AppError::NoAccount), "got {err:?}");
        let space_id = err
            .chat_space_id()
            .expect("post persisted → space id carried")
            .to_string();

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
            matches!(err.root(), AppError::InsufficientBalance { .. }),
            "got {err:?}"
        );
        let space_id = err
            .chat_space_id()
            .expect("post persisted → space id carried")
            .to_string();

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
fn network_error_after_send_emits_user_turn_and_wraps_space_id() {
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
        let space_id = err
            .chat_space_id()
            .expect("network-error arm must carry the space id")
            .to_string();
        // Underlying error is a transport/network error, not a server error.
        assert!(
            matches!(err.root(), AppError::Network { .. }),
            "got {:?}",
            err.root()
        );

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

        let space_id = err
            .chat_space_id()
            .expect("space id on non-2xx")
            .to_string();
        match err.root() {
            AppError::Server { status, .. } => assert_eq!(*status, 500),
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

        let space_id = err.chat_space_id().expect("space id").to_string();
        match err.root() {
            AppError::Server { status, .. } => assert_eq!(*status, 503),
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
fn mid_sse_abort_emits_user_turn_and_wraps_space_id() {
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

        let space_id = err
            .chat_space_id()
            .expect("mid-SSE abort must carry the space id")
            .to_string();
        assert!(
            matches!(err.root(), AppError::Network { .. }),
            "got {:?}",
            err.root()
        );

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
fn respond_stream_failure_wraps_space_id_and_keeps_single_post() {
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

        // Wrapped with the (already-known) space id so the GUI routes it the
        // same way a failed ask is routed.
        assert_eq!(
            err.chat_space_id().expect("space id"),
            posted.space_id,
            "failure carries the space id"
        );
        match err.root() {
            AppError::Server { status, .. } => assert_eq!(*status, 503),
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
// Setup-failure wrapping: a `prepare_turn` failure (e.g. the `/v1/models`
// fetch the PR #218 screenshot failed on) happens *before* the turn's inline
// `wrap` closure — it must still carry the persisted space id, or a blank
// GUI space can't adopt its id (Retry suppressed; a follow-up strands a
// second space). Both transports share the fix (the wrapped prepare_turn
// call), so both are asserted.
// ===========================================================================

#[test]
fn streaming_setup_failure_wraps_space_id_and_keeps_single_space() {
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
        let err = core
            .runtime()
            .block_on(core.chat_stream("Hello, what is your name?".into(), MODEL.into(), None, tx))
            .expect_err("models-fetch failure should fail the turn");

        // The persisted space id is carried even though the failure was pre-wrap.
        let space_id = err
            .chat_space_id()
            .expect("setup failure must carry the space id")
            .to_string();

        // The user post survived (post ran before prepare_turn) — exactly one
        // space with exactly one user turn, retryable rather than stranded.
        let spaces = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");
        assert_eq!(spaces.len(), 1, "one space, not a stranded pair");
        assert_eq!(spaces[0].id, space_id);
        let messages = core
            .runtime()
            .block_on(core.get_space_messages(space_id.clone()))
            .expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, "user");

        // post emitted Space+SpaceIndex; the pre-wrap setup failure itself emits
        // nothing further (no request row, no spend).
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
fn blocking_setup_failure_wraps_space_id() {
    run(|| {
        let (_mock, core, _dir) = setup(MockConfig {
            models_status: Some(503),
            ..MockConfig::default()
        });
        with_account(&core);

        // The blocking `chat` shares the same wrapped prepare_turn call.
        let err = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("models-fetch failure should fail the turn");
        let space_id = err
            .chat_space_id()
            .expect("blocking setup failure must carry the space id")
            .to_string();
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

        let err = core
            .runtime()
            .block_on(core.chat("boom".into(), MODEL.into(), None))
            .expect_err("non-2xx fails");
        let space_id = err.chat_space_id().expect("space id").to_string();

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
    core.register_tool(std::sync::Arc::new(EchoTool));
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

        let space_id = err.chat_space_id().expect("space id carried").to_string();
        assert!(
            matches!(err.root(), AppError::ToolLoop { .. }),
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
        let probe = core
            .runtime()
            .block_on(core.test_chat_with_budget("budget me".into(), MODEL.into(), None, i64::MAX))
            .expect_err("the probe also hits the cap eventually");
        let probe_space = probe.chat_space_id().expect("space id").to_string();
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
        let space_id = err.chat_space_id().expect("space id carried").to_string();
        let message = err.to_string();
        assert!(
            matches!(err.root(), AppError::Credential { .. }),
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
        let space_id = err.chat_space_id().expect("space id carried").to_string();
        assert!(
            matches!(err.root(), AppError::ToolLoop { .. }),
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

/// A later turn's context never replays a prior turn's tool traffic: the
/// upstream renderer keeps only post-bearing action types, so the second ask
/// sees the two posts and nothing else.
#[test]
fn a_later_turns_context_does_not_replay_tool_traffic() {
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
        let roles: Vec<String> = last["messages"]
            .as_array()
            .expect("messages")
            .iter()
            .map(|m| m["role"].as_str().unwrap_or_default().to_string())
            .collect();
        assert!(
            !roles.contains(&"tool".to_string()),
            "a later turn must not replay tool results: {roles:?}"
        );
        assert!(
            last["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m.get("tool_calls").is_none()),
            "a later turn must not replay tool calls"
        );
        assert_eq!(roles, vec!["system", "user", "assistant", "user"]);
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

        let err = core
            .runtime()
            .block_on(core.chat("loop forever".into(), MODEL.into(), None))
            .expect_err("the round cap fails the turn");
        let space_id = err.chat_space_id().expect("space id").to_string();

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
        let space_id = err.chat_space_id().expect("space id carried").to_string();
        assert!(
            matches!(err.root(), AppError::ToolLoop { .. }),
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
        let space_id = err.chat_space_id().expect("space id carried").to_string();
        assert!(
            matches!(err.root(), AppError::ToolLoop { .. }),
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
