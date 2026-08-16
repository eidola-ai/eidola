//! The turn driver for agent-spawned sub-spaces.
//!
//! A delegated room has no window, so app-core gives it its turns. What is
//! pinned here:
//!
//! * **A room with nobody watching still works.** The brief is planned off, the
//!   helper answers, and the walk continues until the plan is empty.
//! * **Every stop is reported, on one channel.** Concluding, pausing at the
//!   cascade guard, spending the delegation's turn budget and failing a turn all
//!   arrive in the parent as the owner's post carrying the room's last word as a
//!   quoted reference — with an annotation saying which of the four it was.
//! * **The reference is attached by the driver, not chosen by the model**, and
//!   it resolves and renders through the one rendering every quoted passage
//!   takes.
//! * **The budget is a property of the delegation, not of the session.** It is
//!   counted from rows, so a restart cannot reset the meter.
//! * **Ownership is unambiguous.** The consumer-facing planning door answers no
//!   turns for a delegated room, so a window cannot drive one alongside the
//!   driver; an ordinary conversation is untouched.
//! * **Archival stops it**, at the room and at the parent, and neither leaves a
//!   report about work somebody closed.

mod chat_harness;

use chat_harness::{ChatBehavior, MockConfig, MockServer, flat_messages};
use eidola_app_core::error::AppError;
use eidola_app_core::{
    AppCore, ExpectedScope, MAX_DELEGATION_TURNS, NewParticipant, NotificationPlan,
    ParticipantUpdate, PostNode, SpawnedSubspace,
};

/// These turns run over an `openai` backend, so none of them needs a
/// credential — the driver's own behaviour is what is under test, not billing.
const MODEL: &str = "qwen3-8b@ext";

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

fn setup() -> (MockServer, AppCore, tempfile::TempDir) {
    let (mock, core, dir) = chat_harness::core_for(MockConfig {
        chat: ChatBehavior::OkStreaming,
        ..MockConfig::default()
    });
    add_backend(&core, &mock);
    (mock, core, dir)
}

fn add_backend(core: &AppCore, mock: &MockServer) {
    add_backend_at(core, &mock.base_url);
}

fn add_backend_at(core: &AppCore, base_url: &str) {
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

fn space(core: &AppCore) -> String {
    core.runtime()
        .block_on(core.create_space(None))
        .expect("space")
        .id
}

fn shared_agent(core: &AppCore, space_id: &str, label: &str) -> String {
    let added = core
        .runtime()
        .block_on(core.add_space_participant(
            space_id.to_string(),
            NewParticipant {
                label: label.to_string(),
                model_ref: Some(MODEL.to_string()),
                system_prompt: None,
                notify_policy: "human".into(),
            },
        ))
        .expect("add agent");
    core.runtime()
        .block_on(core.promote_participant(added.id.clone(), None, None))
        .expect("promote");
    added.id
}

/// A parent conversation with something in it: the report attaches to the
/// owner's own last post there, and a parent with no posts at all has nothing
/// for a reply to hang from.
fn parent_with_a_post(core: &AppCore) -> String {
    let parent = space(core);
    core.runtime()
        .block_on(core.post("What do the tide tables say?".into(), Some(parent.clone())))
        .expect("post");
    parent
}

fn spawn(core: &AppCore, parent: &str, owner: &str, participants: Vec<String>) -> SpawnedSubspace {
    core.runtime()
        .block_on(core.spawn_subspace(
            parent.to_string(),
            owner.to_string(),
            "Check the tide tables for Friday.".to_string(),
            participants,
            vec![],
            None,
        ))
        .expect("spawn")
}

fn drive(core: &AppCore, space_id: &str) -> Result<(), AppError> {
    core.runtime()
        .block_on(core.test_drive_subspace(space_id.to_string()))
}

fn tree(core: &AppCore, space_id: &str) -> Vec<PostNode> {
    core.runtime()
        .block_on(core.get_space_tree(space_id.to_string()))
        .expect("tree")
}

fn inference_count(nodes: &[PostNode]) -> usize {
    nodes
        .iter()
        .filter(|n| n.action_type == "inference")
        .count()
}

/// The owner's reports in the parent: the posts there that carry a reference
/// edge, oldest first.
fn reports(core: &AppCore, parent: &str) -> Vec<PostNode> {
    tree(core, parent)
        .into_iter()
        .filter(|n| !n.references.is_empty())
        .collect()
}

fn report(core: &AppCore, parent: &str) -> Option<PostNode> {
    reports(core, parent).into_iter().next()
}

/// A core whose upstream **outlives it**: the mock runs on a runtime of its
/// own, because a mock started on the core's runtime dies with the core being
/// restarted. Returns the mock's runtime (which the caller keeps alive), the
/// mock, the first core, and the profile directory both cores share.
fn restartable() -> (
    tokio::runtime::Runtime,
    MockServer,
    AppCore,
    tempfile::TempDir,
) {
    let mock_rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("mock runtime");
    let mock = mock_rt.block_on(chat_harness::start(MockConfig {
        chat: ChatBehavior::OkStreaming,
        ..MockConfig::default()
    }));
    let (_unused, core, dir) = chat_harness::core_for(MockConfig {
        chat: ChatBehavior::OkStreaming,
        ..MockConfig::default()
    });
    add_backend_at(&core, &mock.base_url);
    (mock_rt, mock, core, dir)
}

/// Ask a specific participant to respond — the door a human watching a
/// delegated room still has, and the only way to post into one before the join
/// affordance ships.
fn ask(core: &AppCore, space_id: &str, participant: &str, target: &str) -> String {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    core.runtime()
        .block_on(core.respond_stream_as(
            space_id.to_string(),
            participant.to_string(),
            target.to_string(),
            tx,
        ))
        .expect("the explicit ask runs")
        .response_action_id
        .expect("the ask wrote a post")
}

// ===========================================================================
// Driving a room nobody is watching
// ===========================================================================

/// The whole point of the wave: a room with no human in it and no window on it
/// takes its turns anyway, and stops when there is nothing left to plan.
#[test]
fn a_delegated_room_takes_its_turns_with_nobody_watching() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper.clone()]);

        let requests_before = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("the room is driven");

        let room = tree(&core, &out.space.id);
        assert_eq!(room.len(), 2, "brief + the helper's answer: {room:?}");
        assert_eq!(room[0].action_type, "brief");
        assert_eq!(room[1].action_type, "inference");
        assert_eq!(room[1].participant.label, "Surveyor");

        // Two requests: the helper's turn, and the owner's report.
        assert_eq!(
            mock.chat_bodies().len() - requests_before,
            2,
            "one driven turn and one report"
        );
    });
}

/// The room's own cascade guard still governs it, and pausing there is a
/// terminal outcome like any other — reported, not silently dropped.
#[test]
fn a_room_that_pauses_at_its_cascade_guard_says_so() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        // The brief is agent-authored, so the room opens one hop in; a limit of
        // 1 pauses on the brief itself.
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        drive(&core, &out.space.id).expect("the room is driven");

        assert_eq!(
            tree(&core, &out.space.id).len(),
            1,
            "the guard pauses before any turn runs"
        );
        let report = report(&core, &parent).expect("a paused room still reports");
        let annotation = report.references[0]
            .annotation
            .clone()
            .expect("the edge says how the room ended");
        assert!(
            annotation.contains("reply limit"),
            "the pause is named honestly: {annotation}"
        );
    });
}

// ===========================================================================
// The report
// ===========================================================================

/// The report is a turn for the owning agent in the **parent**, replying to the
/// owner's own last word there, carrying the delegated room's last post as a
/// quoted reference the driver attached — and the model is shown that passage
/// before it writes.
#[test]
fn a_finished_delegation_reports_back_with_the_rooms_last_word_attached() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        // Give the owner a post of its own in the parent, so the report has the
        // spine position it is supposed to take rather than the fallback.
        let owner_post = ask(
            &core,
            &parent,
            &owner,
            &tree(&core, &parent)[0].action_id.clone(),
        );
        let out = spawn(&core, &parent, &owner, vec![helper]);

        drive(&core, &out.space.id).expect("the room is driven");

        let room = tree(&core, &out.space.id);
        let last_word = room.last().expect("the room has a last post");
        let report = report(&core, &parent).expect("the delegation is reported");

        assert_eq!(
            report.participant.label, "Navigator",
            "the owning agent writes the report"
        );
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(owner_post.as_str()),
            "the report replies to the owner's own last post, keeping one spine"
        );
        assert_eq!(report.references.len(), 1, "one attached passage");
        let reference = &report.references[0];
        assert_eq!(reference.ordinal, 1, "ordinal 0 is the reply edge's");
        assert_eq!(reference.antecedent_action_id, last_word.action_id);
        let snippet = reference
            .snippet
            .clone()
            .expect("the passage resolves through the one rendering");
        assert!(!snippet.is_empty());
        assert_eq!(
            reference.annotation.as_deref(),
            Some("the delegated conversation ran to a stop"),
            "the edge carries what ended the room"
        );

        // And the model wrote with the passage in front of it: the report
        // request carries the attached block, attributed and quoted.
        let body = mock
            .chat_bodies()
            .last()
            .cloned()
            .expect("a report request");
        let messages = flat_messages(&body);
        let attached = messages
            .iter()
            .find(|(_, content)| content.contains("Attached to your reply"))
            .map(|(_, content)| content.clone())
            .expect("the attached block reaches the model");
        assert!(
            attached.contains("[1] Surveyor"),
            "attributed as the source room names its author: {attached}"
        );
        assert!(
            attached.contains(&format!("> {snippet}")),
            "quoted as a passage: {attached}"
        );
        assert!(
            attached.contains("ran to a stop"),
            "and says how the room ended: {attached}"
        );
    });
}

/// A room whose last post cannot be answered still gets its report — the
/// failure arm is the same channel, wearing the failure's own words.
#[test]
fn a_failed_turn_is_reported_rather_than_swallowed() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper.clone()]);

        // Point the helper at a backend that does not exist. It still plans (it
        // has a model), and its turn fails at preparation.
        core.runtime()
            .block_on(core.update_space_participant(
                helper,
                ParticipantUpdate {
                    model_ref: Some(Some("nowhere@missing".into())),
                    ..ParticipantUpdate::default()
                },
                ExpectedScope::Global,
            ))
            .expect("repoint the helper");

        drive(&core, &out.space.id).expect("a failed turn is not a failed drive");

        assert_eq!(
            tree(&core, &out.space.id).len(),
            1,
            "the failing turn wrote no post"
        );
        let report = report(&core, &parent).expect("failure is information, not silence");
        let annotation = report.references[0].annotation.clone().unwrap_or_default();
        assert!(
            annotation.starts_with("a turn in the delegated conversation failed:"),
            "the failure is named: {annotation}"
        );
        assert!(
            annotation.contains("missing"),
            "and says what went wrong in the error's own words: {annotation}"
        );
    });
}

/// An archived **parent** takes no new work, so the report meets the same gate
/// every other turn meets. The delegation's last word then stays unreported,
/// which is what a later run reads as "still outstanding".
#[test]
fn a_report_into_an_archived_parent_is_refused_and_stays_outstanding() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        core.runtime()
            .block_on(core.archive_space(parent.clone()))
            .expect("archive the parent");

        let err = drive(&core, &out.space.id).expect_err("a closed parent takes no report");
        match &err {
            AppError::SpaceArchived { space_id } => assert_eq!(space_id, &parent),
            other => panic!("expected SpaceArchived, got {other:?}"),
        }
        assert!(
            report(&core, &parent).is_none(),
            "nothing was written into the closed conversation"
        );
        // The room itself was still driven — it is live, and archival of the
        // parent is not archival of it.
        assert_eq!(inference_count(&tree(&core, &out.space.id)), 1);
    });
}

// ===========================================================================
// Guards
// ===========================================================================

/// The seat, depth and room guards bound the roster; this one bounds the work.
/// A room whose cascade guard is wide open runs exactly
/// [`MAX_DELEGATION_TURNS`] turns and then stops — and says so.
#[test]
fn a_delegation_stops_at_its_turn_budget_and_says_so() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        // Wide enough that the cascade guard never fires: the budget is what is
        // being teeth-checked.
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1_000))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let out = spawn(&core, &parent, &owner, vec![a, b]);

        drive(&core, &out.space.id).expect("the room is driven");

        assert_eq!(
            inference_count(&tree(&core, &out.space.id)) as i64,
            MAX_DELEGATION_TURNS,
            "the budget is the ceiling, exactly"
        );
        let report = report(&core, &parent).expect("a spent budget is reported");
        let annotation = report.references[0].annotation.clone().unwrap_or_default();
        assert!(
            annotation.contains(&format!("all {MAX_DELEGATION_TURNS} of the turns")),
            "budget exhaustion is information, not silence: {annotation}"
        );
    });
}

/// A **restart picks a delegation back up**. Nothing records that a room was
/// left mid-flight; the driver's startup sweep enumerates the live delegated
/// rooms and each armed driver decides for itself, from the rows, whether there
/// is anything outstanding.
#[test]
fn a_restart_picks_up_a_delegation_left_mid_flight() {
    run(|| {
        let (mock_rt, mock, core, dir) = restartable();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);
        // The driver was never started here, so the room was opened and left.
        assert_eq!(tree(&core, &out.space.id).len(), 1);

        drop(core);
        let core = chat_harness::reopen_core(&dir, &mock.base_url);
        core.start_subspace_driver();

        wait_until(&core, || !reports(&core, &parent).is_empty());
        assert_eq!(
            inference_count(&tree(&core, &out.space.id)),
            1,
            "the helper answered after the restart"
        );
        drop(mock_rt);
    });
}

/// **The meter is the rows, not the process.** A restart that reset it would
/// hand every delegation a fresh budget every time the app came back; counting
/// turns that were persisted is what makes that unrepresentable.
#[test]
fn a_spent_budget_survives_a_restart() {
    run(|| {
        let (mock_rt, mock, core, dir) = restartable();
        let parent = parent_with_a_post(&core);
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1_000))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let out = spawn(&core, &parent, &owner, vec![a.clone(), b]);
        drive(&core, &out.space.id).expect("the room is driven");
        let spent = inference_count(&tree(&core, &out.space.id));
        assert_eq!(spent as i64, MAX_DELEGATION_TURNS);

        // A new process, over the same profile, with none of the first one's
        // memory.
        drop(core);
        let core = chat_harness::reopen_core(&dir, &mock.base_url);

        // Something posts into the room, which makes its last word unreported
        // and the delegation outstanding again — the continuation path.
        let room = tree(&core, &out.space.id);
        let last = room.last().expect("a last post").action_id.clone();
        ask(&core, &out.space.id, &a, &last);
        let after_ask = mock.chat_bodies().len();

        drive(&core, &out.space.id).expect("the room is picked back up");

        assert_eq!(
            mock.chat_bodies().len() - after_ask,
            1,
            "the report, and not one driven turn more"
        );
        let report = report(&core, &parent).expect("the room reports again");
        let annotation = report.references[0].annotation.clone().unwrap_or_default();
        assert!(
            annotation.contains("of the turns"),
            "still over budget after the restart: {annotation}"
        );
        drop(mock_rt);
    });
}

/// Archival is what stops new work everywhere, and a delegated room is not an
/// exception: a driver armed on an archived room does nothing at all, and
/// reports nothing about a conversation somebody closed.
#[test]
fn the_driver_never_drives_an_archived_room() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);
        core.runtime()
            .block_on(core.archive_space(out.space.id.clone()))
            .expect("archive the room");

        let requests_before = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("an archived room is a no-op, not an error");

        assert_eq!(
            mock.chat_bodies().len(),
            requests_before,
            "no turn, and no report"
        );
        assert_eq!(
            tree(&core, &out.space.id).len(),
            1,
            "the brief, and nothing after it"
        );
        assert!(report(&core, &parent).is_none());
    });
}

// ===========================================================================
// Ownership
// ===========================================================================

/// **One driver per room.** The planning door a consumer can reach answers no
/// turns for a delegated room, so a window open on one cannot cascade it
/// alongside the driver — while the mechanical computation still shows the
/// turns are there, which is what proves the refusal is a policy rather than an
/// empty room.
#[test]
fn a_consumer_cannot_plan_turns_in_a_delegated_room() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper.clone()]);

        let mechanical =
            core.runtime()
                .block_on(core.mechanical_notification_plan(
                    out.space.id.clone(),
                    out.brief_action_id.clone(),
                ))
                .expect("plan");
        assert!(
            matches!(&mechanical, NotificationPlan::Turns(t) if t.len() == 1),
            "there is work in the room: {mechanical:?}"
        );

        let consumer = core
            .runtime()
            .block_on(core.plan_notifications(out.space.id.clone(), out.brief_action_id.clone()))
            .expect("plan");
        assert!(
            matches!(&consumer, NotificationPlan::Turns(t) if t.is_empty()),
            "and it is not the consumer's to drive: {consumer:?}"
        );

        // An ordinary conversation is untouched by the rule.
        let ordinary = space(&core);
        let agent = shared_agent(&core, &ordinary, "Ada");
        core.runtime()
            .block_on(core.set_space_participant_override(
                ordinary.clone(),
                agent,
                eidola_app_core::ParticipantOverride {
                    notify_policy: Some(Some("all".into())),
                    ..Default::default()
                },
            ))
            .expect("override");
        let posted = core
            .runtime()
            .block_on(core.post("Anyone home?".into(), Some(ordinary.clone())))
            .expect("post");
        let plan = core
            .runtime()
            .block_on(core.plan_notifications(ordinary, posted.action_id))
            .expect("plan");
        assert!(
            matches!(&plan, NotificationPlan::Turns(t) if !t.is_empty()),
            "an ordinary conversation still plans for its consumer: {plan:?}"
        );
    });
}

// ===========================================================================
// Continuation, through the live supervisor
// ===========================================================================

/// The bus is what arms a room, so **continuation needs no mechanism of its
/// own**: a spawn wakes the room, and a post into a room that has already
/// stopped wakes it again. Driven through the real supervisor rather than the
/// awaited seam, because the arming is the thing under test.
#[test]
fn a_post_into_a_stopped_room_starts_it_again() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper.clone()]);

        // The spawn's own `Change::Space` is what arms it.
        wait_until(&core, || !reports(&core, &parent).is_empty());
        let first = reports(&core, &parent).remove(0);

        // Now post into the stopped room. Nothing tells the driver about it
        // except the change the post itself emits.
        let last = tree(&core, &out.space.id)
            .last()
            .expect("a last post")
            .action_id
            .clone();
        ask(&core, &out.space.id, &helper, &last);

        wait_until(&core, || reports(&core, &parent).len() == 2);
        let second = reports(&core, &parent).pop().expect("a second report");
        assert_ne!(
            second.references[0].antecedent_action_id, first.references[0].antecedent_action_id,
            "the second report names what has been said since"
        );
    });
}

/// Poll for a condition the background driver brings about. Generous, because
/// what is being asserted is that it happens at all, not how fast.
fn wait_until(core: &AppCore, mut done: impl FnMut() -> bool) {
    for _ in 0..300 {
        if done() {
            return;
        }
        core.runtime().block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        });
    }
    panic!("the driver never got there");
}
