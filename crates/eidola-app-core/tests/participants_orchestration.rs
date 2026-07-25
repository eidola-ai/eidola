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

use chat_harness::{ChatBehavior, MODEL, MockConfig, RefundMode, with_account};
use eidola_app_core::error::AppError;
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
        let messages = bodies[0]["messages"].as_array().expect("messages");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are a pirate.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "ahoy");
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
        assert_eq!(bodies[0]["messages"][0]["content"], "First prompt.");
        assert_eq!(bodies[1]["messages"][0]["content"], "Second prompt.");
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
        let quoting = contents
            .iter()
            .find(|c| c.starts_with("What does this mean?"))
            .expect("the quoting post reaches the wire");
        assert_eq!(
            quoting, "What does this mean?\n\n> powerhouse",
            "the marker expands into the quoted passage"
        );
    });
}
