//! Participants v1 — wave 2 orchestration tests, on the in-process
//! mock-upstream harness (`chat_harness`). These exercise the new turn path
//! end-to-end with *real* credential crypto:
//!
//! * **Notify-set fan-out planning** — `submit` + `plan_notifications` over a
//!   multi-participant space, asserting the notify policy predicate (`all` /
//!   `human` / `explicit`) and author exclusion, for both a human-authored and
//!   an agent-authored triggering post.
//! * **Participant-aware context** — an explicit participant's effective system
//!   prompt is prepended as the leading `system` message, and an edit to that
//!   prompt is honored on the next turn (effective config resolved per turn).
//! * **Participant-aware rendering** — the role split: in a multi-agent space
//!   another agent's post reaches the responder as `user` (headed with its
//!   author's label), and only the responder's own prior posts are
//!   `assistant`. Exact rendered bytes are pinned.
//! * **Cascade guard** — a chain of consecutive agent posts reaches the space's
//!   `cascade_limit`, and `plan_notifications` returns the paused marker;
//!   explicit asks (`respond_stream_as`) bypass the guard.
//! * **ACT provisioning queue** — two concurrent turns each obtain their own
//!   real, spendable credential (the wallet-level serialization prevents two
//!   turns from double-booking one credential); the wait predicate fast-fails
//!   with `InsufficientBalance` when no in-flight credential could ever cover
//!   the charge, and waits/succeeds when a covering one is mid-spend.
//! * **Space consistency** — `plan_notifications`, `respond_stream_as`, and a
//!   `post_reply` `reply_to` all reject an action from another space (no
//!   cross-space context / reply edges); an out-of-space participant is
//!   rejected too.

mod chat_harness;

use chat_harness::{
    ChatBehavior, HUMAN_LABEL, MODEL, MockConfig, RefundMode, flat_messages, headed,
    system_message, tool_result_text, with_account,
};
use eidola_app_core::error::AppError;
use eidola_app_core::tools::EchoTool;
use eidola_app_core::{
    AppCore, ChatResult, ChatStreamEvent, NewParticipant, NotificationPlan, ParticipantUpdate,
};

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// The participant id of the space's agent with the given label.
fn agent_id(core: &AppCore, space: &str, label: &str) -> String {
    core.runtime()
        .block_on(core.list_space_participants(space.to_string()))
        .expect("participants")
        .into_iter()
        .find(|p| p.label == label)
        .unwrap_or_else(|| panic!("no participant labelled {label}"))
        .id
}

/// Drive a streaming turn as `participant`, replying to `target`. Returns the
/// terminal `ChatResult`.
fn drive_as(
    core: &AppCore,
    space: &str,
    participant: &str,
    target: &str,
) -> Result<ChatResult, AppError> {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
    core.runtime().block_on(core.respond_stream_as(
        space.to_string(),
        participant.to_string(),
        target.to_string(),
        tx,
    ))
}

fn add_agent(
    core: &AppCore,
    space: &str,
    label: &str,
    model: &str,
    policy: &str,
    prompt: Option<&str>,
) {
    core.runtime()
        .block_on(core.add_space_participant(
            space.to_string(),
            NewParticipant {
                label: label.to_string(),
                model_ref: Some(model.to_string()),
                system_prompt: prompt.map(str::to_string),
                notify_policy: policy.to_string(),
            },
        ))
        .expect("add agent");
}

// ===========================================================================
// Notify-set fan-out planning
// ===========================================================================

#[test]
fn submit_plans_notify_set_by_policy_for_human_post() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;

        // The space is born with You (human) + the seeded default agent
        // (policy 'human', model MODEL). Add two more agents.
        add_agent(&core, &space, "All-Agent", MODEL, "all", None);
        add_agent(&core, &space, "Explicit-Agent", MODEL, "explicit", None);

        let result = core
            .runtime()
            .block_on(core.submit("How do tides work?".into(), Some(space.clone()), None))
            .expect("submit");

        // A human post at depth 0: 'human' and 'all' fire, 'explicit' never,
        // the human "You" is not an agent.
        let turns = match result.plan {
            NotificationPlan::Turns(t) => t,
            other => panic!("expected Turns, got {other:?}"),
        };
        let mut labels: Vec<String> = turns
            .iter()
            .map(|t| {
                core.runtime()
                    .block_on(core.list_space_participants(space.clone()))
                    .unwrap()
                    .into_iter()
                    .find(|p| p.id == t.participant_id)
                    .unwrap()
                    .label
            })
            .collect();
        labels.sort();
        assert_eq!(
            labels,
            vec!["All-Agent".to_string(), default_agent_label()],
            "human post notifies 'human' + 'all' agents only"
        );
        // Every planned turn replies to the posted user turn at cascade depth 1.
        for t in &turns {
            assert_eq!(t.target_action_id, result.post.action_id);
            assert_eq!(t.cascade_depth, 1);
        }
    });
}

#[test]
fn plan_on_agent_post_only_fires_all_policy() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        add_agent(&core, &space, "All-Agent", MODEL, "all", None);
        add_agent(&core, &space, "Explicit-Agent", MODEL, "explicit", None);

        // A human post, then an agent response (authored by the seeded default
        // agent via the model-picker compat path).
        let posted = core
            .runtime()
            .block_on(core.post("hi".into(), Some(space.clone())))
            .expect("post");
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
        let reply = core
            .runtime()
            .block_on(core.respond_stream(
                space.clone(),
                MODEL.into(),
                posted.action_id.clone(),
                tx,
            ))
            .expect("agent reply");
        let inference_id = reply.response_action_id.expect("inference action id");

        // Planning on the agent-authored post: 'human' no longer fires (author
        // is an agent), only 'all'. The author (default agent) is excluded, so
        // the notify set is exactly {All-Agent}, at cascade depth 2.
        let plan = core
            .runtime()
            .block_on(core.plan_notifications(space.clone(), inference_id))
            .expect("plan");
        let turns = match plan {
            NotificationPlan::Turns(t) => t,
            other => panic!("expected Turns, got {other:?}"),
        };
        assert_eq!(
            turns.len(),
            1,
            "only the 'all' agent fires on an agent post"
        );
        assert_eq!(
            turns[0].participant_id,
            agent_id(&core, &space, "All-Agent")
        );
        assert_eq!(turns[0].cascade_depth, 2);
    });
}

// ===========================================================================
// Participant-aware context assembly
// ===========================================================================

#[test]
fn explicit_participant_prepends_effective_system_prompt() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        add_agent(
            &core,
            &space,
            "Pirate",
            MODEL,
            "explicit",
            Some("You are a pirate."),
        );
        let pirate = agent_id(&core, &space, "Pirate");

        let posted = core
            .runtime()
            .block_on(core.post("ahoy".into(), Some(space.clone())))
            .expect("post");
        drive_as(&core, &space, &pirate, &posted.action_id).expect("turn");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 1);
        assert_eq!(
            flat_messages(&bodies[0]),
            vec![
                (
                    "system".to_string(),
                    system_message(Some("You are a pirate."))
                ),
                (
                    "user".to_string(),
                    headed(&posted.item_id, HUMAN_LABEL, "ahoy")
                ),
            ]
        );
    });
}

/// **First-person traces (task 33), the scoping half.** A trace is the private
/// record of how one participant reached a conclusion; the post is its
/// distillation. So Ada's tool round reaches Ada's next turn and reaches
/// nobody else — Bo, answering on the very same branch, sees Ada's *post* and
/// no trace of how she got there.
#[test]
fn another_participants_traces_never_reach_this_turn() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::ToolRoundsStreaming(1),
            ..MockConfig::default()
        });
        with_account(&core);
        core.register_tool(std::sync::Arc::new(EchoTool))
            .expect("echo is not a reserved name");
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        add_agent(&core, &space, "Ada", MODEL, "explicit", Some("I am Ada."));
        add_agent(&core, &space, "Bo", MODEL, "explicit", Some("I am Bo."));
        let ada = agent_id(&core, &space, "Ada");
        let bo = agent_id(&core, &space, "Bo");

        // Ada answers with a tool round (requests 1-2), Bo answers Ada
        // (request 3), then Ada answers Bo (request 4).
        let posted = core
            .runtime()
            .block_on(core.post("what say you both?".into(), Some(space.clone())))
            .expect("post");
        let ada_post = drive_as(&core, &space, &ada, &posted.action_id)
            .expect("ada turn")
            .response_action_id
            .expect("ada's post");
        let bo_post = drive_as(&core, &space, &bo, &ada_post)
            .expect("bo turn")
            .response_action_id
            .expect("bo's post");
        drive_as(&core, &space, &ada, &bo_post).expect("ada's second turn");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 4, "ada (2 rounds), bo, ada");
        let roles = |body: &serde_json::Value| -> Vec<String> {
            body["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .map(|m| m["role"].as_str().unwrap_or_default().to_string())
                .collect()
        };

        // Bo's turn: Ada's answer, none of Ada's working.
        assert_eq!(
            roles(&bodies[2]),
            vec!["system", "user", "user"],
            "another participant's trace is invisible: {}",
            bodies[2]
        );
        assert!(
            bodies[2]["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|m| m.get("tool_calls").is_none()),
            "no replayed calls either: {}",
            bodies[2]
        );

        // Ada's next turn: her own round is right back where it happened —
        // after the post she answered, before the answer it produced.
        assert_eq!(
            roles(&bodies[3]),
            vec!["system", "user", "assistant", "tool", "assistant", "user"],
        );
        assert_eq!(bodies[3]["messages"][3]["content"], tool_result_text(1));
    });
}

/// The role split, on the case it exists for: in a multi-agent space, the
/// *other* agent's post reaches the responder as `user` — with a header naming
/// its author — while only the responder's own prior post is `assistant`.
/// Before the split, every `inference` rendered `assistant`, so a model was
/// shown another agent's words as its own.
#[test]
fn other_agents_posts_render_as_user_with_a_header() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        add_agent(&core, &space, "Ada", MODEL, "explicit", Some("I am Ada."));
        add_agent(&core, &space, "Bo", MODEL, "explicit", Some("I am Bo."));
        let ada = agent_id(&core, &space, "Ada");
        let bo = agent_id(&core, &space, "Bo");

        // Human post → Ada answers → Bo answers Ada → Ada answers Bo.
        let posted = core
            .runtime()
            .block_on(core.post("what say you both?".into(), Some(space.clone())))
            .expect("post");
        let ada_turn = drive_as(&core, &space, &ada, &posted.action_id).expect("ada turn");
        let ada_post = ada_turn.response_action_id.expect("ada's post");
        let bo_turn = drive_as(&core, &space, &bo, &ada_post).expect("bo turn");
        let bo_post = bo_turn.response_action_id.expect("bo's post");
        drive_as(&core, &space, &ada, &bo_post).expect("ada's second turn");

        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let item_of = |action_id: &str| {
            tree.iter()
                .find(|n| n.action_id == action_id)
                .unwrap_or_else(|| panic!("no post {action_id}"))
                .item_id
                .clone()
        };

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 3, "ada, bo, ada");

        // Bo's turn: the human post AND Ada's post are both `user` — Ada's
        // headed with her label, so Bo can tell who said what.
        assert_eq!(
            flat_messages(&bodies[1]),
            vec![
                ("system".to_string(), system_message(Some("I am Bo."))),
                (
                    "user".to_string(),
                    headed(&posted.item_id, HUMAN_LABEL, "what say you both?")
                ),
                (
                    "user".to_string(),
                    headed(&item_of(&ada_post), "Ada", "Hello from the stream.")
                ),
            ],
            "another agent's post is `user`, never `assistant`"
        );

        // Ada's second turn: her own earlier post is `assistant`; Bo's is
        // `user`. Same rows, different point of view.
        assert_eq!(
            flat_messages(&bodies[2]),
            vec![
                ("system".to_string(), system_message(Some("I am Ada."))),
                (
                    "user".to_string(),
                    headed(&posted.item_id, HUMAN_LABEL, "what say you both?")
                ),
                (
                    "assistant".to_string(),
                    headed(&item_of(&ada_post), "Ada", "Hello from the stream.")
                ),
                (
                    "user".to_string(),
                    headed(&item_of(&bo_post), "Bo", "Hello from the stream.")
                ),
            ],
            "only the responder's own prior post is `assistant`"
        );
    });
}

#[test]
fn effective_system_prompt_edit_is_honored_next_turn() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        add_agent(
            &core,
            &space,
            "Bard",
            MODEL,
            "explicit",
            Some("First prompt."),
        );
        let bard = agent_id(&core, &space, "Bard");

        let p1 = core
            .runtime()
            .block_on(core.post("one".into(), Some(space.clone())))
            .expect("post");
        drive_as(&core, &space, &bard, &p1.action_id).expect("turn 1");

        // Edit the agent's own system prompt, then take another turn.
        core.runtime()
            .block_on(core.update_space_participant(
                bard.clone(),
                ParticipantUpdate {
                    system_prompt: Some(Some("Second prompt.".into())),
                    ..Default::default()
                },
            ))
            .expect("edit prompt");
        let p2 = core
            .runtime()
            .block_on(core.post("two".into(), Some(space.clone())))
            .expect("post 2");
        drive_as(&core, &space, &bard, &p2.action_id).expect("turn 2");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 2);
        assert_eq!(
            bodies[0]["messages"][0]["content"],
            system_message(Some("First prompt."))
        );
        assert_eq!(
            bodies[1]["messages"][0]["content"],
            system_message(Some("Second prompt."))
        );
    });
}

// ===========================================================================
// Cascade guard
// ===========================================================================

#[test]
fn cascade_pauses_at_limit_and_explicit_ask_bypasses() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

        // A template with cascade_limit 2 (fewer turns to reach the pause) whose
        // single agent auto-responds ('all'), made the default so a new space
        // instantiates it.
        let tmpl = core
            .runtime()
            .block_on(core.create_template(
                "Cascade".into(),
                2,
                vec![eidola_app_core::NewTemplateParticipant {
                    label: "Chatter".into(),
                    model_ref: Some(MODEL.into()),
                    notify_policy: "all".into(),
                    ..Default::default()
                }],
            ))
            .expect("template");
        core.set_default_template(tmpl.id.clone()).expect("default");
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        let chatter = agent_id(&core, &space, "Chatter");

        // u1 (human) → i1 (depth 1) → i2 (depth 2).
        let u1 = core
            .runtime()
            .block_on(core.post("start".into(), Some(space.clone())))
            .expect("post");
        let i1 = drive_as(&core, &space, &chatter, &u1.action_id)
            .expect("i1")
            .response_action_id
            .expect("i1 id");
        let i2 = drive_as(&core, &space, &chatter, &i1)
            .expect("i2")
            .response_action_id
            .expect("i2 id");

        // Planning on i1 (depth 1 < limit 2) is not paused; on i2 (depth 2) it
        // pauses at the limit.
        assert!(
            matches!(
                core.runtime()
                    .block_on(core.plan_notifications(space.clone(), i1.clone()))
                    .expect("plan i1"),
                NotificationPlan::Turns(_)
            ),
            "depth below the limit must plan turns"
        );
        match core
            .runtime()
            .block_on(core.plan_notifications(space.clone(), i2.clone()))
            .expect("plan i2")
        {
            NotificationPlan::Paused { depth, limit } => {
                assert_eq!(depth, 2);
                assert_eq!(limit, 2);
            }
            other => panic!("expected Paused at the cascade limit, got {other:?}"),
        }

        // An explicit ask bypasses the guard entirely — it runs past the limit.
        drive_as(&core, &space, &chatter, &i2).expect("explicit ask past the limit runs");
    });
}

// ===========================================================================
// ACT provisioning queue — concurrent turns
// ===========================================================================

#[test]
fn concurrent_turns_each_get_their_own_credential() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        with_account(&core);

        // Pre-seed exactly ONE active credential large enough to cover either
        // turn's charge. Without the provisioning queue two concurrent turns
        // would both grab this single credential and double-book it (recording
        // the same nonce on both request rows); the queue makes the first turn
        // flip it to `spending` inside the lock, so the second must allocate a
        // fresh one — two distinct credentials.
        core.runtime()
            .block_on(core.account_allocate(50_000))
            .expect("seed credential");

        // Two concurrent blocking chats (each posts + runs a turn), driven on
        // the core's own runtime so they race the wallet-level provisioning
        // queue. Both must obtain their own real, spendable credential.
        let (a, b) = core.runtime().block_on(async {
            tokio::join!(
                core.chat("first question".into(), MODEL.into(), None),
                core.chat("second question".into(), MODEL.into(), None),
            )
        });
        let a = a.expect("first turn");
        let b = b.expect("second turn");
        assert!(a.credits_charged > 0 && b.credits_charged > 0);

        // Each chat produced exactly one request row; the two rows must carry
        // two DISTINCT credential nonces — the queue's guarantee that two
        // concurrent turns never double-book one credential.
        let requests = core
            .runtime()
            .block_on(core.list_requests(10, 0))
            .expect("requests");
        let nonces: Vec<String> = requests
            .iter()
            .filter_map(|r| r.credential_nonce.clone())
            .collect();
        assert_eq!(nonces.len(), 2, "both turns recorded a spending credential");
        assert_ne!(
            nonces[0], nonces[1],
            "concurrent turns must spend two distinct credentials"
        );

        // Two chats each hit the mock once.
        assert_eq!(mock.chat_hits(), 2);
    });
}

#[test]
fn provisioning_fast_fails_when_no_covering_in_flight() {
    run(|| {
        // A tiny static balance (< a normal turn's charge) and a refund that
        // never lands, so the first turn leaves a credential mid-spend whose
        // face value cannot cover the *second* (huge-prompt) turn's charge.
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            refund: RefundMode::Fail,
            balance: 20_000,
            ..MockConfig::default()
        });
        with_account(&core);

        // First turn: a small prompt. It allocates a 20_000 credential, spends
        // it, and — refund failing — leaves it in the `spending` state.
        core.runtime()
            .block_on(core.chat("hi".into(), MODEL.into(), None))
            .expect("first turn succeeds (refund failure is best-effort)");

        // Second turn: a huge prompt whose charge far exceeds both the static
        // balance (20_000) and the in-flight credential's face value (20_000).
        // No allocation is possible and the mid-spend credential — even fully
        // refunded — could never cover it, so provisioning must fail FAST with
        // InsufficientBalance, not burn the 30s bounded wait.
        let huge = "x".repeat(200_000);
        let started = std::time::Instant::now();
        let err = core
            .runtime()
            .block_on(core.chat(huge, MODEL.into(), None))
            .expect_err("second turn cannot be funded");
        let elapsed = started.elapsed();

        assert!(
            matches!(err.root(), AppError::InsufficientBalance { available, .. } if *available == 20_000),
            "expected an immediate InsufficientBalance, got {:?}",
            err.root()
        );
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "must not burn the provisioning wait when no in-flight credential can cover the charge \
             (took {elapsed:?})"
        );
    });
}

#[test]
fn provisioning_waits_and_succeeds_when_covering_in_flight() {
    run(|| {
        // Static balance too small to allocate ANY turn's charge, plus one
        // pre-seeded large credential. Two concurrent turns: the first spends
        // the seeded credential (flipping it to `spending`); the second cannot
        // allocate (balance too small) but sees a covering in-flight credential,
        // waits for its refund recovery, and is funded by the resulting
        // successor. Both must succeed — the covering in-flight case works.
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            balance: 100,
            ..MockConfig::default()
        });
        with_account(&core);

        // Seed one large active credential (the explicit allocate path issues
        // the requested amount regardless of the reported balance).
        core.runtime()
            .block_on(core.account_allocate(1_000_000))
            .expect("seed a large credential");

        let (a, b) = core.runtime().block_on(async {
            tokio::join!(
                core.chat("first".into(), MODEL.into(), None),
                core.chat("second".into(), MODEL.into(), None),
            )
        });
        // Neither turn could allocate a fresh credential (balance is 100), so the
        // second was necessarily funded by the first's recovered refund — the
        // covering-in-flight wait path.
        a.expect("first turn");
        b.expect("second turn funded by the recovered refund");
    });
}

// ===========================================================================
// Space consistency — an action must belong to the supplied space
// ===========================================================================

#[test]
fn plan_notifications_rejects_action_from_another_space() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let a = core.runtime().block_on(core.create_space(None)).unwrap().id;
        let b = core.runtime().block_on(core.create_space(None)).unwrap().id;
        let post_b = core
            .runtime()
            .block_on(core.post("in B".into(), Some(b.clone())))
            .expect("post in B");

        // Planning space A over a post that lives in space B must be rejected.
        assert!(
            core.runtime()
                .block_on(core.plan_notifications(a, post_b.action_id))
                .is_err(),
            "plan_notifications must reject an action from another space"
        );
    });
}

#[test]
fn respond_stream_as_rejects_target_and_participant_from_another_space() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        let a = core.runtime().block_on(core.create_space(None)).unwrap().id;
        let b = core.runtime().block_on(core.create_space(None)).unwrap().id;

        let post_a = core
            .runtime()
            .block_on(core.post("in A".into(), Some(a.clone())))
            .expect("post in A");
        let post_b = core
            .runtime()
            .block_on(core.post("in B".into(), Some(b.clone())))
            .expect("post in B");
        let agent_a = agent_id(&core, &a, &default_agent_label());
        let agent_b = agent_id(&core, &b, &default_agent_label());

        // A target from another space is rejected (cross-space context + edge).
        assert!(
            drive_as(&core, &a, &agent_a, &post_b.action_id).is_err(),
            "respond_stream_as must reject a target from another space"
        );
        // A participant from another space is rejected (not a member here).
        assert!(
            drive_as(&core, &a, &agent_b, &post_a.action_id).is_err(),
            "respond_stream_as must reject a participant from another space"
        );
    });
}

#[test]
fn post_reply_rejects_reply_to_from_another_space() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let a = core.runtime().block_on(core.create_space(None)).unwrap().id;
        let b = core.runtime().block_on(core.create_space(None)).unwrap().id;
        let post_b = core
            .runtime()
            .block_on(core.post("in B".into(), Some(b.clone())))
            .expect("post in B");

        assert!(
            core.runtime()
                .block_on(core.post_reply("reply".into(), Some(a.clone()), Some(post_b.action_id),))
                .is_err(),
            "post_reply must reject a reply_to from another space"
        );
        // The rejected reply left no trace in space A (validated before any write).
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(a))
            .expect("tree");
        assert!(
            tree.is_empty(),
            "a rejected cross-space reply must not persist"
        );
    });
}

/// The default label the seeded agent gets from `DEFAULT_MODEL` (mirrors
/// `db::default_agent_label`). MODEL == DEFAULT_MODEL in the harness.
fn default_agent_label() -> String {
    // `gemma4-31b` → `Gemma4 31b` (see db::default_agent_label).
    "Gemma4 31b".to_string()
}

// ===========================================================================
// Override-here vs edit-everywhere (the wave-3 GUI fork's app-core surface)
// ===========================================================================

#[test]
fn referenced_global_override_and_edit_everywhere_fork() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;

        // "You" is a REFERENCED global; its `reference` detail is populated with
        // the shared base config and (initially) no overrides.
        let you = core
            .runtime()
            .block_on(core.list_space_participants(space.clone()))
            .unwrap()
            .into_iter()
            .find(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
            .expect("You is a member");
        assert_eq!(you.source, "referenced");
        let reference = you.reference.clone().expect("referenced detail present");
        assert_eq!(reference.base_label, you.label, "no override yet");
        assert!(reference.override_label.is_none(), "no override yet");

        // Override-here: this space only. Effective label changes; base does not.
        core.runtime()
            .block_on(core.set_space_participant_override(
                space.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                eidola_app_core::ParticipantOverride {
                    label: Some(Some("Me".to_string())),
                    ..Default::default()
                },
            ))
            .expect("override");
        let you = core
            .runtime()
            .block_on(core.list_space_participants(space.clone()))
            .unwrap()
            .into_iter()
            .find(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
            .unwrap();
        assert_eq!(you.label, "Me", "effective label is the override");
        let reference = you.reference.clone().unwrap();
        assert_eq!(reference.override_label.as_deref(), Some("Me"));
        assert_ne!(reference.base_label, "Me", "the shared global is untouched");

        // Revert the override (inner None) → back to inherited.
        core.runtime()
            .block_on(core.set_space_participant_override(
                space.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                eidola_app_core::ParticipantOverride {
                    label: Some(None),
                    ..Default::default()
                },
            ))
            .expect("revert override");
        let you = core
            .runtime()
            .block_on(core.list_space_participants(space.clone()))
            .unwrap()
            .into_iter()
            .find(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
            .unwrap();
        assert_eq!(you.label, reference.base_label, "reverted to the base");
        assert!(you.reference.unwrap().override_label.is_none());

        // Edit-everywhere: writes the shared global's own config.
        core.runtime()
            .block_on(core.update_space_participant(
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                ParticipantUpdate {
                    label: Some("Myself".to_string()),
                    ..Default::default()
                },
            ))
            .expect("edit everywhere");
        let you = core
            .runtime()
            .block_on(core.list_space_participants(space.clone()))
            .unwrap()
            .into_iter()
            .find(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
            .unwrap();
        assert_eq!(you.label, "Myself", "the shared base changed");
        assert_eq!(you.reference.unwrap().base_label, "Myself");
    });
}

// ===========================================================================
// Quoted references through the composer CTA path (wave 2)
// ===========================================================================

/// The GUI's Post-with-pending-references path end-to-end:
/// `submit_with_references` saves the post with its reference edges and plans
/// notifications; driving the planned turn sends the quoting post upstream
/// with its `{{ embed N }}` marker expanded into the quoted passage — so a
/// quote created in the composer reaches the model as real context.
#[test]
fn submit_with_references_carries_the_quote_to_the_wire() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            ..MockConfig::default()
        });
        with_account(&core);

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

        let result = core
            .runtime()
            .block_on(core.submit_with_references(
                "What does this mean?\n\n{{ embed 1 }}".into(),
                Some(source.space_id.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: source.action_id.clone(),
                    content_block_id: Some(block_id),
                    range_start: Some(24),
                    range_end: Some(34), // "powerhouse"
                    annotation: None,
                }],
            ))
            .expect("submit with references");

        // The seeded default agent (policy 'human') planned a turn on the post.
        let turns = match result.plan {
            NotificationPlan::Turns(t) => t,
            other => panic!("expected Turns, got {other:?}"),
        };
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].target_action_id, result.post.action_id);
        drive_as(
            &core,
            &source.space_id,
            &turns[0].participant_id,
            &turns[0].target_action_id,
        )
        .expect("driven turn");

        let bodies = mock.chat_bodies();
        assert_eq!(bodies.len(), 1);
        let contents: Vec<String> = bodies[0]["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .map(|m| m["content"].as_str().unwrap_or_default().to_string())
            .collect();
        let expected = headed(
            &result.post.item_id,
            HUMAN_LABEL,
            "What does this mean?\n\n> powerhouse",
        );
        assert!(
            contents.contains(&expected),
            "the marker expands into the quoted passage under the post's header; got {contents:?}"
        );
    });
}

// ===========================================================================
// May-decline router (task 22)
//
// The router is just another HTTP call, so the harness scripts it exactly like
// a turn: `RouterBehavior` answers requests whose wire model is
// `chat_harness::ROUTER_MODEL`, leaving `ChatBehavior` free to script the turns
// in the same test. The router model is registered as a *local* engine pointed
// at the mock, which is both the intended production shape (a local router is
// free) and what makes "no spend, no Wallet emission" a real assertion.
//
// Whether the router decides *well* is a judgment question — an offline eval
// over a golden set of (post, cards, slice) → expected notify set, scored
// against candidate models and prompts. That is deliberately not here: real
// model output is not deterministic across machines (Metal vs CPU kernels), so
// CI never gates on it.
// ===========================================================================

/// Point the space's `router_model` at a "loaded" local engine served by the
/// mock, so the router call is a real HTTP round-trip on the zero-spend path.
fn enable_router(core: &AppCore, mock: &chat_harness::MockServer, space: &str) {
    core.test_register_loaded_local_model("local", chat_harness::ROUTER_SLUG, mock.port());
    core.runtime()
        .block_on(
            core.set_space_router_model(space.to_string(), Some(chat_harness::ROUTER_MODEL.into())),
        )
        .expect("set router model");
}

/// The chat requests the mock saw that were *router* calls.
fn router_calls(mock: &chat_harness::MockServer) -> Vec<serde_json::Value> {
    mock.chat_bodies()
        .into_iter()
        .filter(|b| b["model"] == chat_harness::ROUTER_MODEL)
        .collect()
}

fn drain(
    rx: &mut tokio::sync::broadcast::Receiver<eidola_app_core::changes::Change>,
) -> Vec<eidola_app_core::changes::Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
    }
    out
}

/// A space with two auto-notified agents and a human post in it. Returns
/// `(space_id, post_action_id)`.
fn space_with_two_candidates(core: &AppCore) -> (String, String) {
    let space = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("space")
        .id;
    // Born with You + the seeded default agent (policy 'human'); add a second
    // agent so the router has a genuine choice to make.
    add_agent(core, &space, "All-Agent", MODEL, "all", None);
    let post = core
        .runtime()
        .block_on(core.post("How do tides work?".into(), Some(space.clone())))
        .expect("post");
    (space, post.action_id)
}

/// The **unrefined** mechanical set — the router is never consulted.
fn mechanical_turns(core: &AppCore, space: &str, post: &str) -> Vec<eidola_app_core::PlannedTurn> {
    match core
        .runtime()
        .block_on(core.mechanical_notification_plan(space.to_string(), post.to_string()))
        .expect("mechanical plan")
    {
        NotificationPlan::Turns(t) => t,
        other => panic!("expected Turns, got {other:?}"),
    }
}

/// The plan production actually drives — `AppCore::plan_notifications`, which
/// plans **and** refines. There is deliberately no public way to get an
/// unrefined plan for driving: a cascade re-plans on every hop, and one
/// unrefined hop would notify agents the router already filtered out.
fn planned(core: &AppCore, space: &str, post: &str) -> NotificationPlan {
    core.runtime()
        .block_on(core.plan_notifications(space.to_string(), post.to_string()))
        .expect("plan")
}

#[test]
fn router_selection_drops_the_participants_it_did_not_choose() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            router: chat_harness::RouterBehavior::Reply(r#"{"notify": [1]}"#.into()),
            ..MockConfig::default()
        });
        let (space, post) = space_with_two_candidates(&core);
        enable_router(&core, &mock, &space);

        let mechanical = mechanical_turns(&core, &space, &post);
        assert_eq!(mechanical.len(), 2, "both policies fire on a human post");

        let mut rx = core.subscribe_changes();
        let refined = planned(&core, &space, &post);

        // Exactly the first candidate survives — and it survives *unchanged*,
        // cascade_depth included: the router filters the set, it never
        // rewrites the turns (PlannedTurn equality covers the depth math).
        assert_eq!(
            refined,
            NotificationPlan::Turns(vec![mechanical[0].clone()]),
            "the router drops candidate 2 and leaves candidate 1 untouched"
        );

        // A declined participant costs nothing: no turn was driven, so no post
        // and no spend — and the router itself is local, so no Wallet either.
        assert_eq!(router_calls(&mock).len(), 1, "exactly one router call");
        let events = drain(&mut rx);
        assert!(
            !events
                .iter()
                .any(|c| matches!(c, eidola_app_core::changes::Change::Wallet)),
            "a local router spends nothing; got {events:?}"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_messages(space.clone()))
                .expect("messages")
                .len(),
            1,
            "only the human post — nobody was driven"
        );
    });
}

#[test]
fn router_can_choose_nobody() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            router: chat_harness::RouterBehavior::Reply(r#"{"notify": []}"#.into()),
            ..MockConfig::default()
        });
        let (space, post) = space_with_two_candidates(&core);
        enable_router(&core, &mock, &space);

        let mechanical = mechanical_turns(&core, &space, &post);
        assert_eq!(mechanical.len(), 2);
        assert_eq!(
            planned(&core, &space, &post),
            NotificationPlan::Turns(Vec::new()),
            "an empty selection is a valid decision: zero turns"
        );
    });
}

#[test]
fn an_unreachable_router_degrades_to_the_mechanical_set() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            router: chat_harness::RouterBehavior::Fail(500),
            ..MockConfig::default()
        });
        let (space, post) = space_with_two_candidates(&core);
        enable_router(&core, &mock, &space);

        let mechanical = mechanical_turns(&core, &space, &post);
        assert_eq!(
            planned(&core, &space, &post),
            NotificationPlan::Turns(mechanical),
            "a post is never blocked on the router: the failure mode is extra \
             notifications, not lost ones"
        );
    });
}

#[test]
fn unusable_router_output_degrades_to_the_mechanical_set() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            router: chat_harness::RouterBehavior::Reply(
                "I reckon the second one should take this.".into(),
            ),
            ..MockConfig::default()
        });
        let (space, post) = space_with_two_candidates(&core);
        enable_router(&core, &mock, &space);

        let mechanical = mechanical_turns(&core, &space, &post);
        assert_eq!(
            planned(&core, &space, &post),
            NotificationPlan::Turns(mechanical),
            "prose instead of JSON degrades rather than guessing"
        );
    });
}

#[test]
fn an_unset_router_model_is_never_invoked() {
    run(|| {
        // The default: no `router_model` on the space — the feature is off.
        let (mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        let (space, post) = space_with_two_candidates(&core);

        let mechanical = mechanical_turns(&core, &space, &post);
        assert_eq!(
            planned(&core, &space, &post),
            NotificationPlan::Turns(mechanical)
        );
        assert!(
            router_calls(&mock).is_empty(),
            "off means off: not one HTTP call"
        );

        // And the same through `submit`, the production path.
        let result = core
            .runtime()
            .block_on(core.submit("and another thing".into(), Some(space), None))
            .expect("submit");
        assert!(matches!(result.plan, NotificationPlan::Turns(_)));
        assert!(router_calls(&mock).is_empty());
    });
}

#[test]
fn a_paused_plan_never_reaches_the_router() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            router: chat_harness::RouterBehavior::Reply(r#"{"notify": []}"#.into()),
            ..MockConfig::default()
        });
        with_account(&core);

        let tmpl = core
            .runtime()
            .block_on(core.create_template(
                "Cascade".into(),
                1,
                vec![eidola_app_core::NewTemplateParticipant {
                    label: "Chatter".into(),
                    model_ref: Some(MODEL.into()),
                    notify_policy: "all".into(),
                    ..Default::default()
                }],
            ))
            .expect("template");
        core.set_default_template(tmpl.id.clone()).expect("default");
        let space = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        enable_router(&core, &mock, &space);
        let chatter = agent_id(&core, &space, "Chatter");

        let u1 = core
            .runtime()
            .block_on(core.post("start".into(), Some(space.clone())))
            .expect("post");
        let i1 = drive_as(&core, &space, &chatter, &u1.action_id)
            .expect("i1")
            .response_action_id
            .expect("i1 id");

        // Depth 1 == the limit: the cascade guard speaks first and the router
        // is not consulted at all.
        let plan = planned(&core, &space, &i1);
        assert!(matches!(plan, NotificationPlan::Paused { .. }));
        assert!(
            router_calls(&mock).is_empty(),
            "the guard short-circuits before any router call"
        );
    });
}

#[test]
fn an_explicit_ask_bypasses_the_router_entirely() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            // The router would notify nobody…
            router: chat_harness::RouterBehavior::Reply(r#"{"notify": []}"#.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let (space, post) = space_with_two_candidates(&core);
        enable_router(&core, &mock, &space);
        let agent = agent_id(&core, &space, "All-Agent");

        // …but an explicit ask consults no guards and no router.
        let result = drive_as(&core, &space, &agent, &post).expect("explicit ask runs");
        assert!(result.declined.is_none());
        assert!(result.response_action_id.is_some());
        assert!(
            router_calls(&mock).is_empty(),
            "respond_stream_as never routes through the router"
        );
    });
}

#[test]
fn a_remote_routers_hold_is_settled_when_the_body_read_fails() {
    run(|| {
        // The request is accepted (so the credential is already `spending`)
        // and then the connection drops before the body arrives. The
        // refinement must degrade *and* settle the hold: leaving it stranded
        // would make the very next turn burn its bounded provisioning wait on
        // a refund that is never coming.
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            router: chat_harness::RouterBehavior::DropBody,
            ..MockConfig::default()
        });
        with_account(&core);
        let (space, post) = space_with_two_candidates(&core);
        // A *remote* router: the eidola backend, so the call really spends.
        core.runtime()
            .block_on(core.set_space_router_model(
                space.clone(),
                Some(chat_harness::ROUTER_REMOTE_MODEL.into()),
            ))
            .expect("set remote router model");

        let mechanical = mechanical_turns(&core, &space, &post);
        assert_eq!(
            planned(&core, &space, &post),
            NotificationPlan::Turns(mechanical),
            "a body-read failure degrades like any other router failure"
        );

        assert!(
            mock.refund_hits() >= 1,
            "the hold must be settled through the recovery endpoint"
        );
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("wallet lifecycle");
        assert!(
            !lifecycle.iter().any(|c| c.state == "spending"),
            "no credential may be left stranded in `spending`; got {lifecycle:?}"
        );
    });
}

// ===========================================================================
// Agent-side decline checkpoint (task 22)
// ===========================================================================

#[test]
fn a_declining_agent_writes_a_decision_and_suppresses_the_post() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::DeclineStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        core.register_tool(eidola_app_core::decline::decline_tool())
            .expect("decline is not a reserved name");

        let (space, post) = space_with_two_candidates(&core);
        let agent = agent_id(&core, &space, "All-Agent");

        let mut rx = core.subscribe_changes();
        let result = drive_as(&core, &space, &agent, &post).expect("declined turn still succeeds");

        // The would-be post is suppressed — and the decision id is kept OUT of
        // `response_action_id`, so a caller cannot mistake it for a fresh post
        // and cascade off it.
        let declined = result.declined.clone().expect("the turn declined");
        assert_eq!(declined.reason, chat_harness::DECLINE_REASON);
        assert!(result.content.is_empty(), "no post content");
        assert_eq!(
            result.response_action_id, None,
            "a decline produced no post"
        );

        // …and the thread still holds only the human post: `decision` is not a
        // post-bearing action type, so it collapses out of the render.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 1, "only the human post; got {tree:?}");

        // The decision IS in the trail, with the post as its antecedent and the
        // reason as its text.
        let actions = core
            .runtime()
            .block_on(core.test_space_actions(space.clone()))
            .expect("raw actions");
        let types: Vec<&str> = actions.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(
            types,
            vec!["user_input", "tool_call", "tool_result", "decision"],
            "the round's trace is kept and the decision is appended"
        );
        let decision = actions.last().expect("decision row");
        assert_eq!(decision.id, declined.action_id);
        assert_eq!(
            decision.reply_to.as_deref(),
            Some(post.as_str()),
            "a decision hangs off the post it declines, not the trace chain"
        );
        assert!(
            decision
                .blocks
                .iter()
                .any(|b| b.text_content.as_deref() == Some(chat_harness::DECLINE_REASON)),
            "the reason is persisted; got {:?}",
            decision.blocks
        );

        // Emissions: Space (new state to render) + Wallet (the round spent) +
        // Record (request rows). No SpaceIndex — nothing was posted.
        let events = drain(&mut rx);
        use eidola_app_core::changes::Change;
        assert!(
            events
                .iter()
                .any(|c| matches!(c, Change::Space(s) if *s == space))
        );
        assert!(events.iter().any(|c| matches!(c, Change::Record)));
        assert!(
            !events.iter().any(|c| matches!(c, Change::SpaceIndex)),
            "a decline adds no item to the listing"
        );

        // Defense in depth: even a caller that reached for the decision id
        // anyway gets no cascade — a `decision` is not a post, and planning
        // over one yields zero turns rather than notifying anybody.
        assert_eq!(
            core.runtime()
                .block_on(core.plan_notifications(space.clone(), declined.action_id.clone()))
                .expect("plan over the decision"),
            NotificationPlan::Turns(Vec::new()),
            "planning never cascades off a non-post action"
        );
    });
}

#[test]
fn the_decline_name_alone_is_not_a_decline_without_the_tool_registered() {
    run(|| {
        // Same scripted `decline` call, but the registry never got the tool:
        // the model gets an ordinary unknown-tool result and the turn goes on
        // to answer normally.
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::DeclineBlocking,
            ..MockConfig::default()
        });
        with_account(&core);
        let (space, post) = space_with_two_candidates(&core);
        let agent = agent_id(&core, &space, "All-Agent");

        let result = core.runtime().block_on(core.respond_stream_as(
            space.clone(),
            agent,
            post,
            tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>().0,
        ));
        // The turn does *not* end in a decline. (It runs another round against
        // the same behaviour and eventually hits the round cap, which is a
        // turn failure — the point is only that it is never a silent decline.)
        match result {
            Ok(r) => assert!(r.declined.is_none(), "no decline without registration"),
            // (the `Err` arm is checked below)
            Err(e) => assert!(
                matches!(e.root(), AppError::ToolLoop { .. }),
                "expected the ordinary tool loop to run on, got {e:?}"
            ),
        }
    });
}

#[test]
fn space_traces_put_a_decline_in_the_gap_under_the_post_it_answered() {
    run(|| {
        // Task 34's read over task 22's decision: the turn wrote no post, so
        // its disclosure hangs under the post it declined — the non-event is
        // visible where the answer would have been, named for the agent that
        // made it.
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::DeclineStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        core.register_tool(eidola_app_core::decline::decline_tool())
            .expect("decline is not a reserved name");

        let (space, post) = space_with_two_candidates(&core);
        let agent = agent_id(&core, &space, "All-Agent");
        let result = drive_as(&core, &space, &agent, &post).expect("declined turn still succeeds");
        let declined = result.declined.clone().expect("the turn declined");

        let traces = core
            .runtime()
            .block_on(core.space_traces(space.clone()))
            .expect("trace read");
        assert_eq!(traces.len(), 1, "one turn, one disclosure: {traces:?}");
        let trace = &traces[0];
        assert_eq!(
            trace.anchor_action_id, post,
            "the decline hangs under the post it answered"
        );
        assert!(trace.unanswered);
        assert_eq!(
            trace.participant_label, "All-Agent",
            "and names whose non-event it was"
        );
        // The rounds it ran stay with it, and the decision is the last word.
        assert!(matches!(
            trace.entries.first(),
            Some(eidola_app_core::TraceEntry::Tool { .. })
        ));
        match trace.entries.last() {
            Some(eidola_app_core::TraceEntry::Declined { action_id, reason }) => {
                assert_eq!(*action_id, declined.action_id);
                assert_eq!(reason.as_deref(), Some(chat_harness::DECLINE_REASON));
            }
            other => panic!("expected the decision last, got {other:?}"),
        }
    });
}

#[test]
fn two_declines_by_one_agent_on_one_post_stay_separate_disclosures() {
    run(|| {
        // A reader asks the same agent again after it bowed out. Two turns
        // ran, and each one's rounds and decision belong to *it* — merging
        // them would hide how many times the agent was asked and answered
        // with silence, which is exactly what the disclosure exists to show.
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::DeclineStreaming,
            ..MockConfig::default()
        });
        with_account(&core);
        core.register_tool(eidola_app_core::decline::decline_tool())
            .expect("decline is not a reserved name");

        let (space, post) = space_with_two_candidates(&core);
        let agent = agent_id(&core, &space, "All-Agent");
        let first = drive_as(&core, &space, &agent, &post).expect("first turn declines");
        let second = drive_as(&core, &space, &agent, &post).expect("second turn declines");
        let d1 = first.declined.expect("declined").action_id;
        let d2 = second.declined.expect("declined").action_id;
        assert_ne!(d1, d2, "two turns, two decisions");

        let traces = core
            .runtime()
            .block_on(core.space_traces(space.clone()))
            .expect("trace read");
        assert_eq!(traces.len(), 2, "two turns, two disclosures: {traces:?}");
        for trace in &traces {
            assert_eq!(trace.anchor_action_id, post);
            assert!(trace.unanswered);
            assert_eq!(trace.participant_label, "All-Agent");
            assert_eq!(
                trace
                    .entries
                    .iter()
                    .filter(|e| matches!(e, eidola_app_core::TraceEntry::Declined { .. }))
                    .count(),
                1,
                "each disclosure carries exactly its own turn's decision"
            );
        }
        let decisions: Vec<&str> = traces
            .iter()
            .flat_map(|t| t.entries.iter())
            .filter_map(|e| match e {
                eidola_app_core::TraceEntry::Declined { action_id, .. } => Some(action_id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(decisions, vec![d1.as_str(), d2.as_str()]);
        assert_ne!(traces[0].id, traces[1].id, "and each turn has its own id");
    });
}
