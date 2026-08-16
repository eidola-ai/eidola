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
    AppCore, DelegationEnd, DelegationFailure, ExpectedScope, MAX_ATTEMPTS_PER_TAIL,
    MAX_DELEGATION_TURNS, NewParticipant, NotificationPlan, ParticipantUpdate, PostNode,
    SpawnedSubspace,
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

/// A spawn with no anchor — a caller with no turn behind it, which is every
/// direct use of the API today.
fn spawn(core: &AppCore, parent: &str, owner: &str, participants: Vec<String>) -> SpawnedSubspace {
    spawn_from(core, parent, owner, participants, None)
}

/// A spawn that names the post in the parent it is being opened from — what a
/// turn-scoped caller supplies, and what the report attaches beneath.
fn spawn_from(
    core: &AppCore,
    parent: &str,
    owner: &str,
    participants: Vec<String>,
    anchor: Option<&str>,
) -> SpawnedSubspace {
    core.runtime()
        .block_on(core.spawn_subspace(
            parent.to_string(),
            owner.to_string(),
            "Check the tide tables for Friday.".to_string(),
            participants,
            vec![],
            None,
            anchor.map(str::to_string),
        ))
        .expect("spawn")
}

fn drive(core: &AppCore, space_id: &str) -> Result<(), AppError> {
    core.runtime()
        .block_on(core.test_drive_subspace(space_id.to_string()))
}

/// A walk of the kind the startup sweep arms — which never waits for an
/// anchor's answer.
fn drive_as_sweep(core: &AppCore, space_id: &str) -> Result<(), AppError> {
    core.runtime()
        .block_on(core.test_drive_subspace_armed(space_id.to_string(), false))
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
        assert!(
            matches!(
                report.references[0].delegation_end,
                Some(DelegationEnd::Paused { depth: 1, limit: 1 })
            ),
            "the pause is carried as a value, not a sentence: {:?}",
            report.references[0].delegation_end
        );
        assert_eq!(
            report.references[0].annotation, None,
            "and it is not left standing where a person's own note is rendered"
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
            reference.delegation_end,
            Some(DelegationEnd::Concluded),
            "the edge carries what ended the room, typed"
        );
        assert_eq!(reference.annotation, None, "and says nothing as a person");

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
        let (mock, core, _dir) = setup();
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
        assert_eq!(
            report.references[0].delegation_end,
            Some(DelegationEnd::TurnFailed {
                reason: DelegationFailure::Configuration
            }),
            "the failure is reported as a bounded category"
        );
        // And the error's own words never leave this process: nothing in the
        // owner's prompt names the backend, the URL or the message.
        let body = mock
            .chat_bodies()
            .last()
            .cloned()
            .expect("a report request");
        let sent = serde_json::to_string(&body).expect("body");
        assert!(
            !sent.contains("nowhere") && !sent.contains("missing"),
            "the raw failure text must not reach another model's context: {sent}"
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

/// **A post that arrives while the driver is working gets its turn.** It is on
/// nobody's frontier — the walk never saw it — and by the time the walk ends a
/// driven turn has very likely written something newer, so re-deriving "the
/// tail" would look straight past it: the report would name the driven tail,
/// the room would read as reported, and the question would sit there answered
/// by nobody. Staged exactly: the walk is stopped after its first driven turn
/// and the post is committed there.
#[test]
fn a_post_arriving_mid_walk_is_not_walked_past() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        let mut window = core.test_open_cascade_window();
        let outsider = std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &out.space.id));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches the cascade window");
            // Somebody asks a question in the room, mid-walk. Driven turns will
            // write newer posts after this one, which is what hides it.
            let brief = out.brief_action_id.clone();
            let outsider = ask(&core, &out.space.id, &owner, &brief);
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
            outsider
        });

        // It was planned off: something in the room replies to it.
        let room = tree(&core, &out.space.id);
        assert!(
            room.iter()
                .any(|n| n.parent_action_id.as_deref() == Some(outsider.as_str())),
            "the post that arrived mid-walk was answered: {:?}",
            room.iter()
                .map(|n| (n.action_id.clone(), n.parent_action_id.clone()))
                .collect::<Vec<_>>()
        );
        assert!(
            report(&core, &parent).is_some(),
            "and the delegation still reported"
        );
    });
}

/// **The line between "already here" and "arrived while I worked" is drawn
/// before the first read, not after it.** A walk's opening reads are several
/// awaits wide; a post committing inside them is not the tail the walk started
/// from, and a boundary taken afterwards would put it before the refill's
/// window too — belonging to neither, with its own change event spent on a walk
/// already under way. Staged exactly: the walk is stopped between reading the
/// room's last post and deciding anything from it, and the post is committed
/// there.
#[test]
fn a_post_landing_between_a_walks_own_reads_is_not_lost() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        let mut window = core.test_open_entry_window();
        let outsider = std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &out.space.id));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches the entry window");
            let brief = out.brief_action_id.clone();
            let outsider = ask(&core, &out.space.id, &owner, &brief);
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
            outsider
        });

        let room = tree(&core, &out.space.id);
        assert!(
            room.iter()
                .any(|n| n.parent_action_id.as_deref() == Some(outsider.as_str())),
            "the post that landed between the reads was answered: {:?}",
            room.iter()
                .map(|n| (n.action_id.clone(), n.parent_action_id.clone()))
                .collect::<Vec<_>>()
        );
    });
}

// ===========================================================================
// Where the report attaches
// ===========================================================================

/// **The report lands on the branch the work was asked for on**, beneath the
/// owning agent's own answer there — not wherever that agent happened to speak
/// last. The two are the same thing only when nothing else is going on in the
/// parent, which is exactly the assumption a delegation cannot make.
#[test]
fn a_report_attaches_beneath_the_owners_answer_to_the_post_it_was_asked_on() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");

        // The post the work is asked for on, and the owner's answer to it.
        let asked = tree(&core, &parent)[0].action_id.clone();
        let answer = ask(&core, &parent, &owner, &asked);
        // …and then the owner says something else, elsewhere in the parent,
        // *after* that answer. This is the post the old rule would have picked.
        let aside = core
            .runtime()
            .block_on(core.post("Meanwhile:".into(), Some(parent.clone())))
            .expect("post");
        let later = ask(&core, &parent, &owner, &aside.action_id);

        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));
        drive(&core, &out.space.id).expect("the room is driven");

        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(answer.as_str()),
            "beneath the owner's answer to the post it was asked on"
        );
        assert_ne!(
            report.parent_action_id.as_deref(),
            Some(later.as_str()),
            "and not beneath whatever it said most recently"
        );
    });
}

/// **A delegation whose spawning turn has not answered yet waits**, rather than
/// planting its report as the first reply and leaving the agent's own answer
/// indented beneath it.
#[test]
fn a_report_waits_for_the_answer_it_belongs_under() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // Spawned from a post the owner has not answered — the state a spawn
        // made mid-turn is in until that turn persists its answer.
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        drive(&core, &out.space.id).expect("the room is driven");
        assert!(
            report(&core, &parent).is_none(),
            "it waits rather than reporting into the wrong place"
        );
        assert_eq!(
            inference_count(&tree(&core, &out.space.id)),
            1,
            "the room's own work still happened"
        );

        // The answer lands, and the next arm delivers.
        let answer = ask(&core, &parent, &owner, &asked);
        drive(&core, &out.space.id).expect("the room is driven again");
        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(report.parent_action_id.as_deref(), Some(answer.as_str()));
    });
}

/// **The wait registers before it asks, so an answer landing in between is not
/// lost.** The room's registration is what a change in the parent looks for;
/// asking first and registering second leaves a window in which the answer
/// commits, its change finds nothing registered, and the registration that
/// follows waits for a wake-up that has already gone by — the room then never
/// reports at all. Staged exactly: the walk is stopped inside that window and
/// the answer is committed there.
#[test]
fn an_answer_landing_inside_the_anchor_window_is_not_lost() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        let mut window = core.test_open_anchor_window();
        let answer = std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &out.space.id));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches the anchor window");
            // The spawning turn's answer commits *here* — after the room has
            // registered, before it asks.
            let answer = ask(&core, &parent, &owner, &asked);
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
            answer
        });

        let report = report(&core, &parent).expect("the delegation is reported, not lost");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(answer.as_str()),
            "and beneath the answer that landed in the window"
        );
    });
}

/// **An unrelated wake does not spend the wait.** The room is registered
/// against its *parent*, and a parent is an ordinary conversation where
/// anything at all can happen — so a wait that gave up on the first wake would
/// attach the report to the anchor because somebody said something else, and
/// the real answer, landing a moment later, would become the report's sibling
/// instead of the post it belongs under.
#[test]
fn an_unrelated_post_in_the_parent_does_not_spend_the_wait() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        drive(&core, &out.space.id).expect("the room is driven");
        assert!(report(&core, &parent).is_none(), "it waits");

        // Something unrelated happens in the parent, twice, and wakes the room
        // both times. The answer it is waiting for is still not there.
        for line in ["Meanwhile:", "And another thing:"] {
            core.runtime()
                .block_on(core.post(line.into(), Some(parent.clone())))
                .expect("post");
            drive(&core, &out.space.id).expect("the room is woken");
            assert!(
                report(&core, &parent).is_none(),
                "an unrelated post is not the answer, so the wait stands"
            );
        }

        // The answer arrives, and only then does it report — beneath it.
        let answer = ask(&core, &parent, &owner, &asked);
        drive(&core, &out.space.id).expect("the room reports");
        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(answer.as_str()),
            "beneath the answer, not beside it"
        );
    });
}

/// **A room that can never run again stops being waited on.** An anchor wait is
/// a standing registration against the parent, and every post there wakes every
/// room registered under it — so a room that has been archived and stays
/// registered makes its parent's every post pay for it, for as long as the
/// process lives.
#[test]
fn an_archived_room_stops_waiting_on_its_parent() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        drive(&core, &out.space.id).expect("the room is driven");
        assert_eq!(
            core.test_rooms_awaiting(parent.clone()),
            vec![out.space.id.clone()],
            "it is registered against its parent while it waits"
        );

        core.runtime()
            .block_on(core.archive_space(out.space.id.clone()))
            .expect("archive the room");
        let requests = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("an archived room is a no-op");

        assert!(
            core.test_rooms_awaiting(parent.clone()).is_empty(),
            "and it lets go the moment it can never run again"
        );
        assert_eq!(mock.chat_bodies().len(), requests, "having done nothing");
    });
}

/// **Falling behind the bus is not the same as starting fresh.** A process that
/// has just started can say nothing is in flight, and stop waiting on that
/// basis; a process recovering from a bus it lagged cannot — an owner's turn
/// may be running at that very moment. So lag recovery arms as an ordinary
/// signal, the wait survives it, and the answer still lands where it belongs.
#[test]
fn recovering_from_a_lagged_bus_does_not_end_a_wait() {
    run(|| {
        let (mock, core, _dir) = setup();
        // Started before the spawn, so the room is armed by the spawn's own
        // change — a startup sweep is a fresh process and would rightly not
        // wait at all.
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        wait_until(&core, || {
            core.test_rooms_awaiting(parent.clone()) == vec![out.space.id.clone()]
        });
        assert!(report(&core, &parent).is_none(), "it waits");

        // The bus overflows while the owner's turn is still in flight, and the
        // supervisor re-asks the question of every live room. A recovery that
        // helped itself to the sweep's premise would deliver the report here,
        // against the anchor, while the answer was still on its way.
        let before = mock.chat_bodies().len();
        core.runtime().block_on(core.test_recover_from_lag());
        settle(&core);
        assert_eq!(
            mock.chat_bodies().len(),
            before,
            "the recovery started nothing"
        );
        assert!(
            report(&core, &parent).is_none(),
            "the wait survives the recovery"
        );

        // The turn lands, and the report goes where it always belonged.
        let answer = ask(&core, &parent, &owner, &asked);
        wait_until(&core, || report(&core, &parent).is_some());
        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(answer.as_str()),
            "beneath the answer, not beside it"
        );
    });
}

/// **A wait ends by outliving the turn it is waiting on.** The spawning turn
/// can fail after the room was opened — no reply to the anchor is ever coming,
/// and nothing durable says so — so the wait sets its own alarm when it begins.
/// Unrelated parent traffic still never spends it; the clock does.
#[test]
fn a_wait_that_outlives_any_turn_reports_against_the_anchor() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.test_set_anchor_wait_grace(std::time::Duration::from_millis(150));
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // Opened from a post the owner never answers — the shape a spawning
        // turn that failed after the spawn leaves behind.
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        // Nothing else ever happens in the parent: the alarm is the only thing
        // that can end this.
        wait_until(&core, || report(&core, &parent).is_some());
        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(asked.as_str()),
            "attached to the post it was asked on, there being no answer to sit beneath"
        );
        assert!(
            core.test_rooms_awaiting(parent).is_empty(),
            "and it is no longer waiting on anything"
        );
        assert_eq!(out.space.parent_action_id.as_deref(), Some(asked.as_str()));
    });
}

/// The other half of the wait: an answer that is never coming. A walk armed by
/// the startup sweep reports against the post the work was asked on, because at
/// that moment nothing can still be in flight.
#[test]
fn a_sweep_never_waits_for_an_answer_that_is_not_coming() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        drive_as_sweep(&core, &out.space.id).expect("the room is driven");

        let report = report(&core, &parent).expect("the delegation is reported anyway");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(asked.as_str()),
            "attached to the post it was asked on, there being no answer to sit beneath"
        );
    });
}

/// A quoted passage is attributed by the space it was written in, and that has
/// to survive the author being retired between saying it and its being
/// reported — the record retirement promises to leave alone.
#[test]
fn a_retired_author_is_still_named_in_the_report() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn_from(&core, &parent, &owner, vec![helper.clone()], None);

        // The helper answers, and is then retired before the report goes out.
        let brief = out.brief_action_id.clone();
        ask(&core, &out.space.id, &helper, &brief);
        core.runtime()
            .block_on(core.retire_participant(helper))
            .expect("retire the helper");

        drive(&core, &out.space.id).expect("the room is driven");

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
            "a retired author keeps the name they wrote under: {attached}"
        );
    });
}

/// **Re-wording a report does not orphan its delegation.** A regeneration is a
/// new generation of the same turn, and the finding it quotes and the ending it
/// records are facts of the delegation rather than of the sentence — so they
/// travel with it, exactly as an edit's references do. Without that the visible
/// footnote disappears while the driver goes on believing the room reported.
#[test]
fn regenerating_a_report_keeps_its_finding_and_its_ending() {
    run(|| {
        // The driver streams and `regenerate` does not, so this one test needs
        // an upstream that answers both twins.
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkEitherTransport,
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);
        drive(&core, &out.space.id).expect("the room is driven");

        let before = report(&core, &parent).expect("the delegation is reported");
        let quoted = before.references[0].antecedent_action_id.clone();
        let requests_before = mock.chat_bodies().len();

        core.runtime()
            .block_on(core.regenerate(before.action_id.clone(), MODEL.to_string()))
            .expect("regenerate the report");

        // The visible footnote survives, on the generation a reader now sees.
        let after = report(&core, &parent).expect("the report still carries its finding");
        assert_ne!(
            after.action_id, before.action_id,
            "it really is a new generation"
        );
        assert_eq!(after.references.len(), 1);
        assert_eq!(after.references[0].antecedent_action_id, quoted);
        assert_eq!(after.references[0].ordinal, 1, "at the ordinal it had");
        assert_eq!(
            after.references[0].delegation_end,
            Some(DelegationEnd::Concluded),
            "and the ending with it"
        );
        assert!(
            after.references[0].snippet.is_some(),
            "the passage still resolves"
        );

        // The regenerating model was shown what it was re-wording: `Revise`
        // withholds the generation being replaced, so without the attachment
        // the finding would be absent from its context entirely.
        let body = mock
            .chat_bodies()
            .get(requests_before)
            .cloned()
            .expect("the regeneration's request");
        let messages = flat_messages(&body);
        assert!(
            messages
                .iter()
                .any(|(_, c)| c.contains("Attached to your reply") && c.contains("[1] Surveyor")),
            "the finding is in front of the turn re-wording it: {messages:?}"
        );

        // And the driver still reads the room as reported — off the generation
        // a reader can see, not a superseded one.
        let requests = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("a second walk");
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "nothing is re-reported and nothing is re-billed"
        );
        assert_eq!(reports(&core, &parent).len(), 1, "still one report");
    });
}

/// **Every branch's finding comes back, not whichever was written last.** A
/// delegated room fans out — each helper answers the brief and each branch runs
/// down to a post nothing follows — and quoting one of those would report one
/// helper while the room settled as reported, so nothing would ever go back for
/// the rest.
#[test]
fn a_fan_out_reports_every_branchs_finding() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        // Two replies deep is the brief plus one answer each, so both branches
        // reach the room's reply limit — separately, which is the point.
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 2))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let out = spawn(&core, &parent, &owner, vec![a, b]);

        drive(&core, &out.space.id).expect("the room is driven");

        let room = tree(&core, &out.space.id);
        let findings: Vec<String> = room
            .iter()
            .filter(|n| n.action_type == "inference")
            .map(|n| n.action_id.clone())
            .collect();
        assert_eq!(findings.len(), 2, "both helpers answered: {room:?}");

        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(report.references.len(), 2, "both findings are attached");
        let quoted: std::collections::BTreeSet<String> = report
            .references
            .iter()
            .map(|r| r.antecedent_action_id.clone())
            .collect();
        assert_eq!(
            quoted,
            findings
                .iter()
                .cloned()
                .collect::<std::collections::BTreeSet<_>>(),
            "and they are the two branches' own tips"
        );
        let ordinals: Vec<i64> = report.references.iter().map(|r| r.ordinal).collect();
        assert_eq!(ordinals, vec![1, 2], "at ordinals 1..=N");
        assert!(
            report.references.iter().all(|r| r.snippet.is_some()),
            "each passage resolves"
        );

        // Both reach the model that writes the report.
        let body = mock
            .chat_bodies()
            .last()
            .cloned()
            .expect("a report request");
        let attached = flat_messages(&body)
            .into_iter()
            .find(|(_, c)| c.contains("Attached to your reply"))
            .map(|(_, c)| c)
            .expect("the attached block");
        assert!(
            attached.contains("[1] ") && attached.contains("[2] "),
            "both passages are in front of the turn: {attached}"
        );
        assert!(
            attached.contains("Surveyor") && attached.contains("Pilot"),
            "each attributed to the helper who found it: {attached}"
        );

        // And the room settles: its current last word is among what was quoted.
        let requests = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("a second walk");
        assert_eq!(mock.chat_bodies().len(), requests, "nothing is re-reported");
        assert_eq!(reports(&core, &parent).len(), 1);
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
        assert_eq!(
            report.references[0].delegation_end,
            Some(DelegationEnd::BudgetSpent {
                limit: MAX_DELEGATION_TURNS
            }),
            "budget exhaustion is information, not silence"
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
        assert!(
            matches!(
                report.references[0].delegation_end,
                Some(DelegationEnd::BudgetSpent { .. })
            ),
            "still over budget after the restart"
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

/// **A report is a notification, not an invitation.** The decline checkpoint is
/// withdrawn from its registry snapshot: declining one would write a decision
/// instead of the post the reference edge rides, and the delegation would
/// vanish without a word.
#[test]
fn a_report_turn_is_not_offered_the_decline_checkpoint() {
    run(|| {
        let (mock, core, _dir) = setup();
        core.register_tool(eidola_app_core::decline::decline_tool())
            .expect("register decline");
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        drive(&core, &out.space.id).expect("the room is driven");

        let bodies = mock.chat_bodies();
        let driven = &bodies[bodies.len() - 2];
        let report = bodies.last().expect("a report request");
        assert!(
            tool_names(driven).contains(&"decline".to_string()),
            "an ordinary driven turn still carries it: {:?}",
            tool_names(driven)
        );
        assert!(
            !tool_names(report).contains(&"decline".to_string()),
            "the report does not: {:?}",
            tool_names(report)
        );
    });
}

/// And the withdrawal reaches the **fallback** registry, not just the turn's
/// own snapshot. A backend that rejects a `tools` field degrades a turn back to
/// exactly what the consumer registered, so a withdrawal applied only to the
/// snapshot would hand the checkpoint back on the retry — and here that is
/// observable end to end: with `decline` the sole registration, a report whose
/// fallback is properly empty sends no `tools` field and lands, while one that
/// still carried the checkpoint would be rejected a second time and deliver
/// nothing at all.
#[test]
fn the_withdrawal_reaches_the_registry_a_degrade_falls_back_to() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::RejectTools,
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        core.register_tool(eidola_app_core::decline::decline_tool())
            .expect("register decline");
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        drive(&core, &out.space.id).expect("the room is driven");

        assert!(
            report(&core, &parent).is_some(),
            "the report gets through the degrade"
        );
        assert!(
            mock.chat_bodies()
                .iter()
                .all(|b| !tool_names(b).contains(&"decline".to_string())
                    || !flat_messages(b)
                        .iter()
                        .any(|(_, c)| c.contains("Attached to your reply"))),
            "and never advertises the checkpoint on the way"
        );
    });
}

/// The tool names a request advertises.
fn tool_names(body: &serde_json::Value) -> Vec<String> {
    body.get("tools")
        .and_then(|t| t.as_array())
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// **A room that cannot be worked on is not billed for forever.** A failing
/// turn writes no post, so nothing about the room changes and its own failure
/// arms it again; with the upstream down the report fails too, so nothing is
/// ever marked. The retry meter is what closes that circuit — a handful of
/// attempts against one unchanged last word, then silence until something
/// external happens.
#[test]
fn a_room_that_keeps_failing_is_not_retried_forever() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        // Arm it far more often than the meter allows, recording what each
        // arm cost.
        let mut sent = Vec::new();
        for _ in 0..8 {
            let _ = drive(&core, &out.space.id);
            sent.push(mock.chat_bodies().len());
        }

        assert!(sent[0] > 0, "it does try — a blip has to be able to heal");
        let spent = sent[MAX_ATTEMPTS_PER_TAIL as usize - 1];
        assert_eq!(
            *sent.last().expect("eight arms"),
            spent,
            "and it stops once the meter is spent: {sent:?}"
        );
        assert!(
            report(&core, &parent).is_none(),
            "nothing was delivered, which is the truth"
        );
        // The other half of the discrimination — a post from outside is a
        // different last word and starts the count over — cannot be staged
        // here, because a post into the room needs a turn and the upstream is
        // what is down. `a_spent_budget_survives_a_restart` walks that arm on a
        // healthy upstream.
    });
}

/// **A report that fails with nothing else going on retries itself.** Most
/// failures are re-armed for free by the turn that failed writing into the room
/// — but a room that drives nothing writes nothing, so nothing announces
/// anything, and a report that failed there would leave the delegation
/// unreported with no event left in the world to pick it up. An owner-only room
/// is exactly that shape: its plan is empty from the first post.
#[test]
fn a_failed_report_with_no_turn_behind_it_still_tries_again() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        // No helpers: the room's plan is empty from the brief onwards, so it
        // drives nothing and writes nothing.
        let out = spawn(&core, &parent, &owner, vec![]);

        let reports_attempted = |core: &AppCore| {
            let _ = core;
            mock.chat_bodies()
                .iter()
                .filter(|b| {
                    flat_messages(b)
                        .iter()
                        .any(|(_, c)| c.contains("Attached to your reply"))
                })
                .count()
        };
        // One walk is two requests (a 500 costs the tool-capability degrade its
        // one retry), so anything past two is a walk that armed itself.
        wait_until(&core, || reports_attempted(&core) > 2);
        let settled = reports_attempted(&core);
        assert!(
            settled <= 2 * MAX_ATTEMPTS_PER_TAIL as usize,
            "and it is bounded by the meter: {settled} attempts"
        );
        assert_eq!(
            tree(&core, &out.space.id).len(),
            1,
            "the room drove nothing"
        );
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

/// Let whatever the driver was going to do, happen. Generous enough that a
/// mock-backed walk is long finished — which is what makes "nothing happened"
/// mean something.
fn settle(core: &AppCore) {
    for _ in 0..25 {
        core.runtime().block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        });
    }
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
