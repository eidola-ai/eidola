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

use chat_harness::{
    ChatBehavior, MockConfig, MockServer, ROUTER_MODEL, ROUTER_SLUG, RouterBehavior, flat_messages,
};
use eidola_app_core::error::AppError;
use eidola_app_core::{
    AppCore, DelegationEnd, DelegationFailure, ExpectedScope, MAX_ATTEMPTS_PER_TAIL,
    MAX_CONCURRENT_WALKS, MAX_DELEGATION_TURNS, NewParticipant, NotificationPlan,
    ParticipantUpdate, PostNode, SpawnRefusal, SpawnedSubspace,
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

/// **Closing a conversation stops the delegation it was running, before the
/// report gate ever comes into it.** An archived parent would refuse the report
/// at the gate every turn meets — but that state is no longer one a room can
/// rest in: the archival closes the rooms beneath it in its own transaction, so
/// what the driver finds is its *own* archival, which is a no-op rather than a
/// failure and leaves no report about work somebody closed. The parent gate
/// stays where it is, now as the belt to that brace: it is what a walk already
/// past the room's liveness read would still meet.
#[test]
fn closing_a_conversation_stops_the_delegation_it_was_running() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        core.runtime()
            .block_on(core.archive_space(parent.clone()))
            .expect("archive the conversation");
        assert!(
            core.runtime()
                .block_on(core.test_space_archived(out.space.id.clone()))
                .expect("archived?"),
            "the delegation is closed with the conversation it serves"
        );

        let requests = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("a closed room is a no-op, not a failure");
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "no turn is driven and no report is attempted"
        );
        assert!(
            report(&core, &parent).is_none(),
            "nothing was written into the closed conversation"
        );
        assert_eq!(
            inference_count(&tree(&core, &out.space.id)),
            0,
            "and the room itself took no turn"
        );
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

/// **A regeneration mid-walk is one wording, not two.** An edit or
/// regeneration that lands while a sibling branch is walking leaves the old
/// id on the frontier; planning it would bill replies against wording the
/// transcript has hidden, and the refill would then also return the
/// replacement. The walk remaps each queued id to the item's visible tip
/// before planning, so the new generation is the one post.
#[test]
fn a_regeneration_mid_walk_is_planned_once_at_its_visible_tip() {
    run(|| {
        // The driver streams and `regenerate` does not.
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkEitherTransport,
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let out = spawn(&core, &parent, &owner, vec![a, b]);
        let room = out.space.id.clone();
        core.runtime()
            .block_on(core.add_global_participant(
                room.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("the reader joins so they can regenerate");

        let mut window = core.test_open_cascade_window();
        let (old, tip) = std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &room));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches its window");
            let first = tree(&core, &room)
                .into_iter()
                .find(|n| n.action_type == "inference")
                .expect("the first helper has answered");
            let old = first.action_id.clone();
            core.runtime()
                .block_on(core.regenerate(old.clone(), MODEL.to_string()))
                .expect("regenerate the first helper's answer");
            let tip = tree(&core, &room)
                .into_iter()
                .find(|n| n.action_type == "inference")
                .expect("the regenerated answer is visible")
                .action_id;
            assert_ne!(tip, old, "it really is a new generation");
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
            (old, tip)
        });

        let room_tree = tree(&core, &room);
        let replies_to_tip = room_tree
            .iter()
            .filter(|n| n.parent_action_id.as_deref() == Some(tip.as_str()))
            .count();
        // The human's regenerate re-authors the item under a seated member who
        // already has that model — here the owner — so both helpers reply to
        // the new wording. Planning the superseded generation as well would
        // add a third child from the helper who originally wrote it.
        assert_eq!(
            replies_to_tip,
            2,
            "the visible generation is planned once — not superseded {old} \
             and then replacement {tip}: {:?}",
            room_tree
                .iter()
                .map(|n| {
                    (
                        n.participant.label.clone(),
                        n.action_id.clone(),
                        n.parent_action_id.clone(),
                        n.generation,
                    )
                })
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

/// **A wait's own alarm is not flattened into a wake-up.** An arm carries a
/// licence, and the licence that matters most arrives exactly when the room is
/// busy: the grace comes due while a walk is under way. Recorded as "something
/// happened", it is lost — the next pass runs on the arm the walk started with,
/// the alarm was one-shot, the wait is not fresh so nothing schedules another,
/// and a delegation whose spawning turn failed waits until the process
/// restarts. Staged exactly: the walk is held inside the anchor window until
/// the clock it set has run out.
#[test]
fn a_grace_arriving_mid_walk_still_ends_the_wait() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.test_set_anchor_wait_grace(std::time::Duration::from_millis(150));
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();

        // The window is open before the spawn, so the walk the spawn arms
        // registers its wait — setting the alarm — and is then held there.
        let mut window = core.test_open_anchor_window();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));
        let resume = core
            .runtime()
            .block_on(window.recv())
            .expect("the walk reaches the anchor window");

        // The alarm comes due *here*, on a room whose driver is running: it can
        // only be recorded, and what is recorded is what the next pass runs on.
        core.runtime().block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(400)).await;
        });
        // Later passes are not held.
        drop(window);
        resume.send(()).expect("the walk resumes");

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

/// **The startup sweep does not claim a room this process opened.** Its whole
/// licence to stop waiting is that no turn which could answer these rooms'
/// anchors is still running — true of rooms an earlier run left behind, whose
/// turns died with it, and false of a room *this* process spawned, because a
/// spawn happens inside its owner's turn. Claiming it there reports against the
/// anchor while the owner's answer is still on its way, and the corrective
/// signal cannot take it back: the merge is strongest-wins, correctly.
#[test]
fn a_sweep_does_not_claim_a_room_this_process_opened() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // Opened from a post its owner has not answered — the state every
        // spawn is in until the turn it happened inside persists its answer.
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        // The driver starts *after* the spawn, so the sweep's enumeration finds
        // this room. It is still this process's room, and the owner's turn is
        // still the thing being waited for.
        core.start_subspace_driver();
        settle(&core);
        assert!(
            report(&core, &parent).is_none(),
            "the sweep armed it as an ordinary signal, so it holds out for the \
             owner's answer instead of planting the report against the anchor"
        );
        assert_eq!(
            core.test_rooms_awaiting(parent.clone()),
            vec![out.space.id.clone()],
            "and it is registered against its parent, waiting"
        );

        // The owner's turn lands, and the report goes where it always belonged.
        let answer = ask(&core, &parent, &owner, &asked);
        wait_until(&core, || report(&core, &parent).is_some());
        assert_eq!(
            report(&core, &parent)
                .expect("the delegation is reported")
                .parent_action_id
                .as_deref(),
            Some(answer.as_str()),
            "beneath the answer, not beside it"
        );
    });
}

/// **A failed enumeration is retried, with its provenance intact.** The
/// startup sweep is the only recovery a pre-existing room has — nothing else
/// will ever raise a signal for it — so an enumeration error merely logged
/// leaves every such room at its brief for the life of the process. And the
/// spawn record must survive the failure: consumed by an attempt that then
/// errored, it would have the retry claiming the sweep's licence for exactly
/// the rooms it is false about — the ones this process opened.
#[test]
fn a_failed_sweep_enumeration_retries_and_keeps_its_provenance() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // This process's own room, opened from a post its owner has not
        // answered — the state every spawn is in until its turn persists one.
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        // The sweep's first enumeration fails.
        core.test_fail_next_subspace_enumerations(1);
        core.start_subspace_driver();

        // The retry still arms the room — and still as this process's own, so
        // its walk registers a wait instead of reporting against the anchor on
        // the sweep's licence.
        wait_until(&core, || {
            core.test_rooms_awaiting(parent.clone())
                .contains(&out.space.id)
        });
        assert!(
            report(&core, &parent).is_none(),
            "armed as a signal: the wait holds for the owner's answer"
        );

        let answer = ask(&core, &parent, &owner, &asked);
        wait_until(&core, || report(&core, &parent).is_some());
        assert_eq!(
            report(&core, &parent)
                .expect("the delegation is reported")
                .parent_action_id
                .as_deref(),
            Some(answer.as_str()),
            "beneath the answer, not against the anchor"
        );
    });
}

/// **A backlog arrives as a trickle, not a stampede.** Arming is per room and
/// the registry only stops one room being walked twice — so a start that finds
/// a previous run's rooms outstanding put every walk on the runtime at once,
/// each of them a chain of billed turns. The bound is on execution and nothing
/// else: a room whose turn has not come keeps its registry entry, so arms still
/// merge into it.
#[test]
fn no_more_than_the_bound_of_rooms_is_walked_at_once() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");

        // Every walk stops at its first read and stays there, which is what
        // makes "in flight" countable: one handle arrives per walk that ran.
        let mut window = core.test_open_entry_window();
        core.start_subspace_driver();
        let rooms: Vec<String> = (0..MAX_CONCURRENT_WALKS + 2)
            .map(|_| spawn(&core, &parent, &owner, vec![helper.clone()]).space.id)
            .collect();
        assert!(rooms.len() > MAX_CONCURRENT_WALKS);

        let mut held = Vec::new();
        for n in 0..MAX_CONCURRENT_WALKS {
            held.push(
                core.runtime()
                    .block_on(window.recv())
                    .unwrap_or_else(|| panic!("walk {n} runs")),
            );
        }
        // And no more, while those hold the permits.
        assert!(
            core.runtime()
                .block_on(async {
                    tokio::time::timeout(std::time::Duration::from_millis(400), window.recv()).await
                })
                .is_err(),
            "a room past the bound waits its turn rather than starting"
        );

        // Letting one go admits exactly one more — the queue drains, it does
        // not stall.
        held.remove(0).send(()).expect("a walk resumes");
        assert!(
            core.runtime()
                .block_on(async {
                    tokio::time::timeout(std::time::Duration::from_secs(2), window.recv()).await
                })
                .expect("the next walk starts once a permit is free")
                .is_some()
        );
    });
}

/// **A room that is closed lets go of its parent, whichever door closed it.**
/// A waiting room holds a standing registration against its parent, and every
/// post there wakes every room registered under it — so a room archived out
/// from under its own wait makes that parent's every post pay for a walk that
/// can only no-op. Retirement is the door that reaches a room this way without
/// touching its parent, so nothing about the parent can announce it: the door
/// has to end the wait where it ends the room.
#[test]
fn retiring_an_owner_releases_the_wait_its_room_was_holding() {
    run(|| {
        let (mock, core, _dir) = setup();
        // Long enough that the alarm cannot be what releases this.
        core.test_set_anchor_wait_grace(std::time::Duration::from_secs(300));
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // Opened from a post the owner has not answered, so the room waits.
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        wait_until(&core, || {
            core.test_rooms_awaiting(parent.clone()) == vec![out.space.id.clone()]
        });

        let requests = mock.chat_bodies().len();
        core.runtime()
            .block_on(core.retire_participant(owner.clone()))
            .expect("retire the owner");
        settle(&core);

        assert!(
            core.runtime()
                .block_on(core.test_space_archived(out.space.id.clone()))
                .expect("archived?"),
            "the room is closed with its owner"
        );
        assert!(
            core.test_rooms_awaiting(parent.clone()).is_empty(),
            "and it stops waiting on a parent it can never report to — without \
             the grace, which is what the room would otherwise hold out for"
        );
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "nothing was driven on the way"
        );
        assert!(report(&core, &parent).is_none(), "and nothing was reported");
    });
}

/// **A delegation beneath a closed conversation stops, and lets go.** Archiving
/// a room archives what it delegated (`subspaces.rs`), and each closed room
/// announces itself — so a nested delegation waiting on its now-archived parent
/// is woken, reads its own archival, and takes its registration back out. Left
/// live it would have retried a report the archived parent refuses, forever.
#[test]
fn a_delegation_beneath_a_closed_conversation_stops_and_lets_go() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let room = spawn(&core, &parent, &owner, vec![helper.clone()]);
        // The helper opens a room of its own from the brief, before answering
        // it — so its report is waiting on an answer that has not been written.
        let brief = tree(&core, &room.space.id)[0].action_id.clone();
        let nested = core
            .runtime()
            .block_on(core.spawn_subspace(
                room.space.id.clone(),
                helper.clone(),
                "A sub-errand of my own.".to_string(),
                vec![],
                vec![],
                None,
                Some(brief),
            ))
            .expect("spawn");

        drive(&core, &nested.space.id).expect("the nested room is driven");
        assert_eq!(
            core.test_rooms_awaiting(room.space.id.clone()),
            vec![nested.space.id.clone()],
            "it is waiting for the answer its report belongs under"
        );

        // The human closes the conversation the whole tree hangs from.
        core.runtime()
            .block_on(core.archive_space(parent.clone()))
            .expect("archive");
        assert!(
            core.runtime()
                .block_on(core.test_space_archived(nested.space.id.clone()))
                .expect("archived?"),
            "the nested delegation is closed with everything above it"
        );

        let requests = mock.chat_bodies().len();
        drive(&core, &nested.space.id).expect("an archived room is a no-op");
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "it neither works nor reports into a room somebody closed"
        );
        assert!(
            core.test_rooms_awaiting(room.space.id.clone()).is_empty(),
            "and the registration it held against its parent is released"
        );
        assert!(
            reports(&core, &room.space.id).is_empty(),
            "nothing was reported into the closed room"
        );
    });
}

/// **An owner that has left the parent strands nothing.** Its every report
/// would act as a participant that is no longer in that conversation and be
/// refused for it — the room outstanding forever, retrying against its meter,
/// holding a live-room slot, and deaf to the one signal that changed
/// (`Change::Participants`, which no delegated room listens for). Closing those
/// rooms in the departure's own transaction is what makes the state
/// unreachable rather than handled.
#[test]
fn an_owners_departure_from_the_parent_leaves_no_room_retrying() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);

        core.runtime()
            .block_on(core.remove_space_participant(parent.clone(), owner.clone()))
            .expect("the owner leaves the conversation");

        let requests = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("a closed room is a no-op, not a failure");
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "nothing is driven and no report is attempted"
        );
        assert!(
            report(&core, &parent).is_none(),
            "and nothing was written into the conversation it left"
        );
    });
}

/// **A wait tried nothing — but a failure before the wait tried something.**
/// Giving the attempt back on every `Waiting` handed a dead upstream an
/// unbounded retry: a broken helper turn with an unanswered anchor ends in
/// exactly that outcome, and the release reopened the one circuit the meter
/// exists to close, at the one exit that gives its claim away.
#[test]
fn a_wait_after_a_failed_turn_spends_its_attempt() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // Opened from a post the owner has not answered: the walk's report ends
        // in a wait, and reaches it without an upstream call of its own — so
        // every request counted here is the helper's failing turn.
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        let mut sent = Vec::new();
        for _ in 0..8 {
            drive(&core, &out.space.id).expect("a walk that ends in a wait is not an error");
            sent.push(mock.chat_bodies().len());
        }

        assert!(sent[0] > 0, "it does try — a blip has to be able to heal");
        assert_eq!(
            *sent.last().expect("eight arms"),
            sent[MAX_ATTEMPTS_PER_TAIL as usize - 1],
            "and it stops once the meter is spent, rather than retrying a dead \
             upstream for as long as the anchor goes unanswered: {sent:?}"
        );
        assert!(
            core.test_rooms_awaiting(parent).contains(&out.space.id),
            "the delegation is still outstanding, which is the truth"
        );
    });
}

/// The other half of that discrimination: a wait with no failure behind it
/// spends nothing, however often it comes round. A parent is an ordinary
/// conversation and anything can move it, so charging the room an attempt per
/// wake would spend its whole allowance on unrelated posts and then refuse the
/// walk the real answer finally arms.
#[test]
fn a_wait_that_follows_no_failure_spends_nothing() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        // Far more waits than the meter would allow if one counted.
        for _ in 0..(MAX_ATTEMPTS_PER_TAIL + 3) {
            drive(&core, &out.space.id).expect("the room waits");
            assert!(report(&core, &parent).is_none(), "it is still waiting");
        }

        // The answer arrives, and the walk it arms is not refused.
        let answer = ask(&core, &parent, &owner, &asked);
        drive(&core, &out.space.id).expect("the room reports");
        assert_eq!(
            report(&core, &parent)
                .expect("the delegation is reported")
                .parent_action_id
                .as_deref(),
            Some(answer.as_str()),
            "beneath the answer it was holding out for"
        );
    });
}

/// **A spent meter must not stand between a decided ending and its delivery.**
/// Three failed walks against one last word close the room to further walking,
/// correctly — each drove billed turns. But with the anchor unanswered every
/// one of them held its report back, so the ending was decided and never told;
/// and when the answer finally landed, the arm it raised met the spent meter
/// and exited. The meter bounds *work*, and delivering a decision is not work.
#[test]
fn a_spent_meter_still_delivers_the_ending_its_work_decided() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper.clone()], Some(&asked));

        // The helper's turns fail at preparation — a backend that is not there
        // — so each walk decides a failure and then has nowhere to report it.
        core.runtime()
            .block_on(core.update_space_participant(
                helper,
                ParticipantUpdate {
                    model_ref: Some(Some("nowhere@missing".into())),
                    ..ParticipantUpdate::default()
                },
                ExpectedScope::Global,
            ))
            .expect("point the helper at nothing");

        for _ in 0..(MAX_ATTEMPTS_PER_TAIL + 2) {
            drive(&core, &out.space.id).expect("a walk that ends in a wait is not an error");
            assert!(
                report(&core, &parent).is_none(),
                "there is nowhere to report until the anchor is answered"
            );
        }

        // The answer lands. The walking allowance is long spent — and the
        // delegation still owes the parent its outcome.
        let answer = ask(&core, &parent, &owner, &asked);
        let requests = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("the ending is delivered");

        let report = report(&core, &parent).expect("the failure is reported, not swallowed");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(answer.as_str()),
            "beneath the answer it was holding out for"
        );
        assert!(
            matches!(
                report.references[0].delegation_end,
                Some(DelegationEnd::TurnFailed { .. })
            ),
            "carrying the ending the spent work decided: {:?}",
            report.references[0].delegation_end
        );
        assert_eq!(
            mock.chat_bodies().len() - requests,
            1,
            "and it cost exactly the one report — no plan, no re-driven turn"
        );

        // Delivered once and done: the room is settled, not re-reported.
        let after = mock.chat_bodies().len();
        drive(&core, &out.space.id).expect("a settled room is a no-op");
        assert_eq!(mock.chat_bodies().len(), after);
        assert_eq!(reports(&core, &parent).len(), 1);
    });
}

/// **A spent meter with nothing to deliver stops being waited on.** The wait
/// is registered before the anchor is asked, so a lookup that then errors
/// leaves a registration standing and remembers nothing. Once the meter
/// refuses, every later arm hits the same refuse — keeping the wait would
/// make every post in the parent pay for a walk that can only no-op. A
/// decided failure still keeps its wait (`a_wait_after_a_failed_turn_spends_its_attempt`);
/// this is the path that decided nothing.
#[test]
fn a_spent_meter_with_nothing_to_deliver_releases_its_wait() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::Non2xx(500),
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        core.test_fail_next_anchor_lookups(MAX_ATTEMPTS_PER_TAIL);
        for _ in 0..MAX_ATTEMPTS_PER_TAIL {
            drive(&core, &out.space.id).expect_err("the lookup fails after the wait is registered");
        }

        drive(&core, &out.space.id).expect("the meter is spent, the walk is refused");
        assert!(
            core.test_rooms_awaiting(parent.clone()).is_empty(),
            "the wait is released — nothing remains to deliver, so the parent \
             is not paying for a walk that can only no-op"
        );
        assert!(report(&core, &parent).is_none(), "and nothing was reported");
    });
}

/// **An ending already decided is delivered, never re-derived.** A walk that
/// ends while its anchor is unanswered keeps what it decided and hands back
/// its claim — so a wake before the answer lands would otherwise claim afresh
/// and re-run the cascade over an unchanged tail: a router inference per wake
/// where the room routes, and a re-derived ending whose leaves are just the
/// tail, overwriting the branch tips the walk that did the work collected.
#[test]
fn a_wake_while_an_ending_waits_delivers_it_without_rewalking() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        // Two replies deep — the brief plus one answer each — so the walk ends
        // with a branch tip per helper.
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 2))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![a, b], Some(&asked));

        drive(&core, &out.space.id).expect("the walk runs");
        assert!(report(&core, &parent).is_none(), "it waits for the answer");
        assert_eq!(
            inference_count(&tree(&core, &out.space.id)),
            2,
            "both helpers answered"
        );

        // The parent moves without answering, twice. Each wake finds the
        // ending already decided: nothing to walk, nothing to spend.
        let requests = mock.chat_bodies().len();
        for line in ["Meanwhile:", "And another thing:"] {
            core.runtime()
                .block_on(core.post(line.into(), Some(parent.clone())))
                .expect("post");
            drive(&core, &out.space.id).expect("the room is woken");
        }
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "a wake with a decided ending asks nothing and drives nothing"
        );

        // And the report carries the decided ending's leaves — every branch
        // tip, not a re-derived walk's tail.
        let answer = ask(&core, &parent, &owner, &asked);
        drive(&core, &out.space.id).expect("the room reports");
        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report.parent_action_id.as_deref(),
            Some(answer.as_str()),
            "beneath the answer it was holding out for"
        );
        assert_eq!(
            report.references.len(),
            2,
            "both branch tips survive the wakes in between"
        );
    });
}

/// **A walk that stops early still carries what arrived while it walked.** The
/// refill used to belong to the concluding path alone, so a post landing during
/// a walk that ran out of budget was on nobody's frontier and in nobody's
/// leaves: the report named the driven tail, the room read as reported on that
/// word, and the arrival sat there unserved *and* unquoted in a room nothing
/// would walk again. It cannot be answered — that is what a spent budget means
/// — but it can be reported, which is the difference between a question the
/// owner can act on and one nobody ever sees.
#[test]
fn a_budget_stopped_walk_reports_what_arrived_while_it_walked() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        // Wide enough that the cascade guard never fires: the budget is what
        // stops this walk.
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1_000))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let out = spawn(&core, &parent, &owner, vec![a, b]);
        let room = out.space.id.clone();
        core.runtime()
            .block_on(core.add_global_participant(
                room.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("the reader joins the room they are watching");

        // The reader's question lands after the walk has started driving.
        let mut window = core.test_open_cascade_window();
        let arrival = std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &room));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches its window");
            let arrival = core
                .runtime()
                .block_on(core.post("But what about Friday?".into(), Some(room.clone())))
                .expect("a post into the room")
                .action_id;
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
            arrival
        });

        let report = report(&core, &parent).expect("a spent budget is reported");
        assert_eq!(
            report.references[0].delegation_end,
            Some(DelegationEnd::BudgetSpent {
                limit: MAX_DELEGATION_TURNS
            }),
            "the room stopped on its budget"
        );
        assert!(
            report
                .references
                .iter()
                .any(|r| r.antecedent_action_id == arrival),
            "and the question that arrived while it walked is in the report, \
             unanswered but not lost"
        );

        // And the room really is settled — nothing walks it again holding an
        // invisible question.
        assert!(
            core.runtime()
                .block_on(core.get_space_tree(room.clone()))
                .expect("tree")
                .iter()
                .any(|n| n.action_id == arrival),
            "the arrival is a post of the room like any other"
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

/// **Re-wording a report under a new author does not un-settle it.** A
/// regeneration can mint a fresh agent when no seated member matches the
/// picked model (`TurnSelector::Model`), and the edges travel with the item
/// — so a settlement read that asked the tip's `participant_id` would treat
/// the owner's own report as somebody else's quote and the next walk would
/// post a duplicate. The origin generation is who opened the item.
#[test]
fn regenerating_a_report_under_a_new_author_still_settles() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkEitherTransport,
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        // A second backend on the same upstream, so the regeneration's model
        // matches nobody already seated and a new agent is minted.
        core.runtime()
            .block_on(core.add_backend(eidola_app_core::NewBackend {
                id: "ext2".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: String::new(),
                base_url: Some(mock.base_url.clone()),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            }))
            .expect("add the second backend");
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);
        drive(&core, &out.space.id).expect("the room is driven");

        let before = report(&core, &parent).expect("the delegation is reported");
        core.runtime()
            .block_on(core.regenerate(before.action_id.clone(), EXT2_MODEL.to_string()))
            .expect("regenerate under a model nobody seated has");

        let after = report(&core, &parent).expect("the report is still visible");
        assert_ne!(
            after.action_id, before.action_id,
            "it really is a new generation"
        );
        assert_ne!(
            after.participant.label, before.participant.label,
            "and a new author — the mint, not the owner re-wording themselves"
        );

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

/// A model ref routed through the test's second backend, `ext2` — whatever
/// behavior that test gave its mock (a 500 on every chat via
/// [`failing_backend`], or a decline). Point a participant here, induce the
/// turn, and point it back.
const EXT2_MODEL: &str = "qwen3-8b@ext2";

fn failing_backend(core: &AppCore) -> MockServer {
    let broken = core.runtime().block_on(chat_harness::start(MockConfig {
        chat: ChatBehavior::Non2xx(500),
        ..MockConfig::default()
    }));
    core.runtime()
        .block_on(core.add_backend(eidola_app_core::NewBackend {
            id: "ext2".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: String::new(),
            base_url: Some(broken.base_url.clone()),
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        }))
        .expect("add the failing backend");
    broken
}

fn set_model(core: &AppCore, participant: &str, model_ref: &str) {
    core.runtime()
        .block_on(core.update_space_participant(
            participant.to_string(),
            ParticipantUpdate {
                model_ref: Some(Some(model_ref.into())),
                ..ParticipantUpdate::default()
            },
            ExpectedScope::Global,
        ))
        .expect("point the participant");
}

/// **A failed regeneration must not settle the room.** Regenerating a report
/// against a failing upstream persists a **current `status = 'error'`**
/// generation carrying the report's replicated edges, and the transcript hides
/// it — so the parent shows no report, while a lifecycle read that counted any
/// current edge kept the room settled: durably, across restarts, the paid
/// result never recreated. Settlement reads by the transcript's own
/// visibility predicate, so a report the parent cannot show does not settle
/// anything.
#[test]
fn a_failed_regeneration_does_not_settle_the_room() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);
        drive(&core, &out.space.id).expect("the room is driven");
        let delivered = report(&core, &parent).expect("the delegation is reported");

        let _broken = failing_backend(&core);
        set_model(&core, &owner, EXT2_MODEL);
        core.runtime()
            .block_on(core.regenerate(delivered.action_id.clone(), EXT2_MODEL.into()))
            .expect_err("the regeneration fails upstream");
        assert!(
            reports(&core, &parent).is_empty(),
            "the error generation is current, so the parent shows no report"
        );

        // Pointed back at a working upstream, the next walk re-reports: the
        // room's last word is quoted by nothing the parent shows, so the
        // delegation is outstanding again.
        set_model(&core, &owner, MODEL);
        drive(&core, &out.space.id).expect("the room is re-walked");
        assert_eq!(
            reports(&core, &parent).len(),
            1,
            "the paid result is recreated rather than lost behind a hidden error"
        );
    });
}

/// **A failed regeneration announces the rooms it hid.** The settlement read
/// makes the room outstanding again the moment the error generation stands in
/// front of its report — but outstanding is not armed: the parent's own
/// emission maps back to no settled room (its wait ended at delivery, and the
/// supervisor's cache knows the parent as ordinary), so without an
/// announcement the paid result stays hidden until a restart's sweep. The
/// failing turn therefore announces every room its replicated edges quoted,
/// and the running driver takes it from there — no manual drive in this test.
#[test]
fn a_failed_regeneration_arms_the_room_it_unsettled() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let _out = spawn(&core, &parent, &owner, vec![helper]);
        wait_until(&core, || report(&core, &parent).is_some());

        let delivered = report(&core, &parent).expect("the delegation is reported");
        let _broken = failing_backend(&core);
        set_model(&core, &owner, EXT2_MODEL);
        core.runtime()
            .block_on(core.regenerate(delivered.action_id.clone(), EXT2_MODEL.into()))
            .expect_err("the regeneration fails upstream");
        set_model(&core, &owner, MODEL);

        // The announcement arms the room; the driver re-walks and re-reports
        // on its own. (The driver may catch the owner still pointed at the
        // failing upstream for its first attempt — the retry meter absorbs
        // that, which is what it is for.)
        wait_until(&core, || reports(&core, &parent).len() == 1);
    });
}

/// **A hidden answer is not an answer.** The report waits for the owner's
/// reply to the anchor, and a reply whose regeneration failed leaves a
/// current `status = 'error'` tip the transcript hides — beneath which a
/// report would render at the conversation root, attached to a word nobody
/// can see. The anchor lookup reads by the transcript's predicate, so the
/// wait holds until a visible answer exists.
#[test]
fn a_report_does_not_attach_beneath_a_hidden_answer() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        drive(&core, &out.space.id).expect("the walk runs");
        assert!(report(&core, &parent).is_none(), "it waits for the answer");

        // The answer lands — and then a failed regeneration hides it.
        let answer = ask(&core, &parent, &owner, &asked);
        let _broken = failing_backend(&core);
        set_model(&core, &owner, EXT2_MODEL);
        core.runtime()
            .block_on(core.regenerate(answer, EXT2_MODEL.into()))
            .expect_err("the regeneration fails upstream");
        set_model(&core, &owner, MODEL);

        drive(&core, &out.space.id).expect("the room is woken");
        assert!(
            report(&core, &parent).is_none(),
            "a superseded answer is not attached beneath — the wait holds"
        );

        // A fresh, visible answer ends the wait the ordinary way.
        let answer2 = ask(&core, &parent, &owner, &asked);
        drive(&core, &out.space.id).expect("the room reports");
        assert_eq!(
            report(&core, &parent)
                .expect("the delegation is reported")
                .parent_action_id
                .as_deref(),
            Some(answer2.as_str()),
            "beneath the visible answer"
        );
    });
}

/// **A regenerated answer is still the owner's reply.** A successful
/// regeneration can mint a new agent when no seated member matches the
/// picked model, and asking the tip's `participant_id` would miss it: the
/// wait would run out to grace and the report would land on the anchor as a
/// sibling of a word the parent already shows. The origin generation is who
/// opened the item; the tip is what the parent shows.
#[test]
fn a_regenerated_answer_is_still_the_owners_reply() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkEitherTransport,
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        core.runtime()
            .block_on(core.add_backend(eidola_app_core::NewBackend {
                id: "ext2".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: String::new(),
                base_url: Some(mock.base_url.clone()),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            }))
            .expect("add the second backend");
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let out = spawn_from(&core, &parent, &owner, vec![helper], Some(&asked));

        drive(&core, &out.space.id).expect("the walk runs");
        assert!(report(&core, &parent).is_none(), "it waits for the answer");

        let origin = ask(&core, &parent, &owner, &asked);
        core.runtime()
            .block_on(core.regenerate(origin.clone(), EXT2_MODEL.to_string()))
            .expect("regenerate under a model nobody seated has");
        let tip = tree(&core, &parent)
            .into_iter()
            .find(|n| n.parent_action_id.as_deref() == Some(asked.as_str()))
            .expect("the regenerated answer is visible");
        assert_ne!(tip.action_id, origin, "it really is a new generation");
        assert_ne!(
            tip.participant.label, "Navigator",
            "and a new author — the mint, not the owner re-wording themselves"
        );

        drive(&core, &out.space.id).expect("the room reports");
        assert_eq!(
            report(&core, &parent)
                .expect("the delegation is reported")
                .parent_action_id
                .as_deref(),
            Some(tip.action_id.as_str()),
            "beneath the regenerated answer, not the anchor"
        );
    });
}
#[test]
fn a_spawn_cannot_anchor_to_a_hidden_tip() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let asked = tree(&core, &parent)[0].action_id.clone();
        let answer = ask(&core, &parent, &owner, &asked);

        let _broken = failing_backend(&core);
        set_model(&core, &owner, EXT2_MODEL);
        core.runtime()
            .block_on(core.regenerate(answer.clone(), EXT2_MODEL.into()))
            .expect_err("the regeneration fails upstream");
        set_model(&core, &owner, MODEL);

        let error_tip = core
            .runtime()
            .block_on(core.test_space_actions(parent.clone()))
            .expect("actions")
            .into_iter()
            .find(|a| a.status == "error")
            .expect("the failed regeneration left a hidden tip")
            .id;

        for (what, id) in [
            ("superseded wording", &answer),
            ("hidden error tip", &error_tip),
        ] {
            match core
                .runtime()
                .block_on(core.spawn_subspace(
                    parent.clone(),
                    owner.clone(),
                    "Check the tide tables.".into(),
                    vec![],
                    vec![],
                    None,
                    Some(id.clone()),
                ))
                .expect_err("a hidden generation is not an anchor")
            {
                AppError::SpawnRefused {
                    refusal: SpawnRefusal::AnchorNotInParent { action_id },
                } => {
                    assert_eq!(&action_id, id, "{what}");
                }
                other => panic!("{what}: expected AnchorNotInParent, got {other:?}"),
            }
        }
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(parent.clone()))
                .expect("list")
                .is_empty(),
            "a refused spawn leaves nothing behind"
        );

        // The human post is still visible, and is still a valid anchor.
        let out = spawn_from(&core, &parent, &owner, vec![], Some(&asked));
        assert_eq!(out.space.parent_action_id.as_deref(), Some(asked.as_str()));
    });
}

/// **A branch every responder declines is still a finding.** A decline writes
/// a decision, not a post — so nothing follows the post, nothing re-plans
/// from it, and a leaf test shaped on the *plan* ("no turns") rather than on
/// what the turns produced left it neither leaf nor parent: dropped from the
/// report, and, when it is the room's newest word, absent from the quoted set
/// — so the room read as outstanding and re-billed the same declines on
/// every arm until a second report happened to quote the tail.
#[test]
fn a_branch_whose_responders_all_decline_is_still_reported() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.register_tool(eidola_app_core::decline::decline_tool())
            .expect("register the decline checkpoint");
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let answerer = shared_agent(&core, &parent, "Surveyor");
        let decliner = shared_agent(&core, &parent, "Pilot");
        // The decliner's upstream declines every turn it is offered.
        let declining = core.runtime().block_on(chat_harness::start(MockConfig {
            chat: ChatBehavior::DeclineStreaming,
            ..MockConfig::default()
        }));
        core.runtime()
            .block_on(core.add_backend(eidola_app_core::NewBackend {
                id: "ext2".into(),
                kind: eidola_app_core::BackendKind::OpenAi,
                display_name: String::new(),
                base_url: Some(declining.base_url.clone()),
                api_key: None,
                models_dir: None,
                model_overrides: None,
                engine_path: None,
                auto_start: true,
            }))
            .expect("add the declining backend");
        set_model(&core, &decliner, EXT2_MODEL);
        let out = spawn(&core, &parent, &owner, vec![answerer, decliner]);

        drive(&core, &out.space.id).expect("the room is driven");

        // One answer was written; the decliner declined the brief and then
        // declined that answer — which makes the answer a branch tip nothing
        // will follow, and the finding the parent is owed.
        let room = tree(&core, &out.space.id);
        let findings: Vec<String> = room
            .iter()
            .filter(|n| n.action_type == "inference")
            .map(|n| n.action_id.clone())
            .collect();
        assert_eq!(
            findings.len(),
            1,
            "one answer, every other turn declined: {room:?}"
        );
        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report
                .references
                .iter()
                .map(|r| r.antecedent_action_id.clone())
                .collect::<Vec<_>>(),
            findings,
            "the declined-into-silence branch tip is what the report quotes"
        );

        // And the room is settled — reported on its actual last word, not on
        // a fallback that left the tail unquoted and the room re-armable.
        drive(&core, &out.space.id).expect("a settled room is a no-op");
        assert_eq!(reports(&core, &parent).len(), 1);
    });
}

/// **An old alarm cannot expire a new wait.** The grace alarm outlives the
/// wait that set it, and a room can wait twice: an answer arrives (ending
/// wait one), is hidden by a failed regeneration, and a continuation post
/// opens a second wait. An alarm that fired without asking which wait set it
/// would expire the second on the remainder of the first's clock and attach
/// the report to the anchor prematurely.
#[test]
fn an_old_alarms_grace_does_not_expire_a_new_wait() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.test_set_anchor_wait_grace(std::time::Duration::from_millis(2000));
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let asked = tree(&core, &parent)[0].action_id.clone();
        // Wait one begins on the driver's own walk; its alarm is ticking.
        let out = spawn_from(&core, &parent, &owner, vec![helper.clone()], Some(&asked));
        wait_until(&core, || {
            core.test_rooms_awaiting(parent.clone())
                .contains(&out.space.id)
        });

        // The answer arrives; the report lands beneath it; wait one is over.
        let answer = ask(&core, &parent, &owner, &asked);
        wait_until(&core, || report(&core, &parent).is_some());
        let first_report = report(&core, &parent)
            .expect("the delegation is reported")
            .action_id
            .clone();

        // A failed regeneration hides the answer, and a post into the room
        // re-opens the delegation — wait two, under a grace far beyond this
        // test, while wait one's alarm is still in flight.
        let _broken = failing_backend(&core);
        set_model(&core, &owner, EXT2_MODEL);
        core.runtime()
            .block_on(core.regenerate(answer, EXT2_MODEL.into()))
            .expect_err("the regeneration fails upstream");
        set_model(&core, &owner, MODEL);
        core.test_set_anchor_wait_grace(std::time::Duration::from_secs(3600));
        ask(&core, &out.space.id, &helper, &out.brief_action_id);
        wait_until(&core, || {
            core.test_rooms_awaiting(parent.clone())
                .contains(&out.space.id)
        });

        // Wait one's alarm comes due here. It must answer only for its own
        // wait: the second report stays held, rather than landing against the
        // anchor on the remainder of a clock that was never its own.
        core.runtime().block_on(async {
            tokio::time::sleep(std::time::Duration::from_millis(2600)).await;
        });
        assert_eq!(
            reports(&core, &parent).len(),
            1,
            "the stale alarm did not expire the new wait"
        );
        assert!(
            core.test_rooms_awaiting(parent.clone())
                .contains(&out.space.id),
            "which goes on waiting for a visible answer"
        );

        // A fresh answer ends wait two the ordinary way.
        let answer2 = ask(&core, &parent, &owner, &asked);
        wait_until(&core, || reports(&core, &parent).len() == 2);
        // By identity, not tree order: the first report re-roots once its
        // answer is hidden (the recorded rendering residual), which shuffles
        // a depth-first flattening.
        let second = reports(&core, &parent)
            .into_iter()
            .find(|r| r.action_id != first_report)
            .expect("the continuation is reported");
        assert_eq!(
            second.parent_action_id.as_deref(),
            Some(answer2.as_str()),
            "beneath the visible answer"
        );
    });
}

/// **A replicated attachment is not asked for permission, and is still asked
/// what it may quote.** The two are different questions: quoting copies, so
/// re-asking permission at a re-wording would rewrite history — but *what may
/// be quoted at all* is an audience boundary, and a rendering is the act that
/// crosses it. An edge written below the create gate can name a post's
/// `thinking` block, which every other read path withholds; carried forward on
/// a regeneration it would be expanded straight into the model's context.
#[test]
fn a_replicated_attachment_cannot_carry_a_hidden_block_into_the_regeneration() {
    run(|| {
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

        let report = report(&core, &parent).expect("the delegation is reported");
        // The helper's answer carries the model's reasoning as a `thinking`
        // block at ordinal 0 — durable, render-side, and withheld from every
        // wire. The seam quotes an antecedent's *first* block, so this edge
        // names exactly that.
        let answer = tree(&core, &out.space.id)
            .last()
            .expect("the helper answered")
            .action_id
            .clone();
        core.runtime()
            .block_on(core.test_insert_unvalidated_reference(report.action_id.clone(), answer, 2))
            .expect("stage an edge below the gate");

        let requests = mock.chat_bodies().len();
        let err = core
            .runtime()
            .block_on(core.regenerate(report.action_id.clone(), MODEL.to_string()))
            .expect_err("a regeneration carrying that edge is refused");
        assert!(
            matches!(err, AppError::NotConfigured { .. }),
            "refused for what the edge names, not for who is writing it: {err:?}"
        );
        assert_eq!(
            mock.chat_bodies().len(),
            requests,
            "and refused before the turn acts — no request, no spend"
        );
        assert!(
            mock.chat_bodies().iter().all(|body| !flat_messages(body)
                .iter()
                .any(|(_, c)| c.contains("thinking…"))),
            "no model has been shown another turn's reasoning"
        );
    });
}

/// **A caller cannot forge a delegation ending.** `annotation` holds either a
/// person's note about their quote or this crate's own record of how a
/// delegated conversation ended, and every reader tells them apart by the
/// reserved prefix alone — so a supplied note claiming it would vanish from
/// every surface that shows a note and appear on every surface that shows an
/// ending. The refusal is on the prefix, so a note that merely mentions it is
/// an ordinary note.
#[test]
fn a_supplied_annotation_cannot_claim_the_ending_namespace() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let space = space(&core);
        let posted = core
            .runtime()
            .block_on(core.post("What do the tide tables say?".into(), Some(space.clone())))
            .expect("post");
        let block = tree(&core, &space)[0].blocks[0].id.clone();
        let quote = |annotation: &str| {
            core.runtime().block_on(core.post_with_references(
                "About that:".into(),
                Some(space.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: posted.action_id.clone(),
                    content_block_id: Some(block.clone()),
                    range_start: Some(0),
                    range_end: Some(4),
                    annotation: Some(annotation.to_string()),
                }],
            ))
        };

        let err = quote("eidola:delegation/concluded")
            .expect_err("the reserved namespace is not a caller's to write");
        assert!(
            matches!(err, AppError::NotConfigured { .. }),
            "a typed refusal, not a silent rewrite: {err:?}"
        );
        assert_eq!(
            tree(&core, &space).len(),
            1,
            "and it leaves no trace — the post is refused whole"
        );

        // A note that mentions the prefix without claiming it stays a note, and
        // reads back as one.
        quote("not an eidola:delegation/concluded marker, just prose").expect("an ordinary note");
        let quoting = tree(&core, &space)
            .into_iter()
            .find(|n| !n.references.is_empty())
            .expect("the quoting post");
        assert_eq!(
            quoting.references[0].annotation.as_deref(),
            Some("not an eidola:delegation/concluded marker, just prose")
        );
        assert_eq!(
            quoting.references[0].delegation_end, None,
            "and nothing reads it as an ending"
        );
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

/// **A wide walk reports every tip it reached.** The seat guard bounds a room's
/// roster and never bounded its frontier: every seat is notify-all, so one
/// post's fan-out puts an answer from each of them on it, and a walk stopped by
/// the budget can be holding many more tips than the room has agents. Dropping
/// the oldest of those was the worst outcome available — the newest tip alone
/// settles the room, so the branches that were dropped were dropped for good.
#[test]
fn a_wide_walk_reports_every_tip_and_not_a_roster_sized_tail() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        // Wide enough that the cascade guard never fires: what is under test is
        // what the walk is holding when the budget stops it.
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1_000))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        // A full room: every seat is notify-all, so one answer wakes all the
        // others and the frontier grows by the roster on every hop.
        let seats: Vec<String> = (1..=eidola_app_core::MAX_SUBAGENTS_PER_SPAWN)
            .map(|n| shared_agent(&core, &parent, &format!("Surveyor {n}")))
            .collect();
        let out = spawn(&core, &parent, &owner, seats);

        drive(&core, &out.space.id).expect("the room is driven");

        let report = report(&core, &parent).expect("a spent budget is reported");
        assert!(
            report.references.len() > eidola_app_core::MAX_SUBAGENTS_PER_SPAWN as usize,
            "the frontier outgrew the roster and every tip on it came back: {} attached",
            report.references.len()
        );
        // Ordinals are still `1..=N` over the whole set, and every passage
        // resolves — the edges describe their own posts whatever the prompt did
        // with them.
        let ordinals: Vec<i64> = report.references.iter().map(|r| r.ordinal).collect();
        assert_eq!(
            ordinals,
            (1..=report.references.len() as i64).collect::<Vec<_>>()
        );
        assert!(report.references.iter().all(|r| r.snippet.is_some()));
        assert!(
            report.references.iter().all(|r| r.delegation_end
                == Some(DelegationEnd::BudgetSpent {
                    limit: MAX_DELEGATION_TURNS
                })),
            "each carries the ending the walk stopped at"
        );

        // And the room settles: its own last word is among what was quoted, so
        // nothing is walked or billed again.
        let quoted: std::collections::BTreeSet<String> = report
            .references
            .iter()
            .map(|r| r.antecedent_action_id.clone())
            .collect();
        let last = tree(&core, &out.space.id)
            .last()
            .expect("a last post")
            .action_id
            .clone();
        assert!(quoted.contains(&last), "the room reads as reported");
    });
}

/// **A burst of findings is bounded as a block, and what is not shown is
/// named.** Clipping each passage bounds one entry and says nothing about how
/// many there are — the walk attaches a tip per branch it reached, and posts
/// into a room somebody is watching make those branches. Unbounded, the block
/// crowds out the conversation the report is written into and at the far end
/// overflows the request, taking the delivery with it. So the block has a
/// ceiling, the edges are all written regardless, and the model is told what it
/// is not being shown rather than left to assume it has read everything.
#[test]
fn an_over_budget_burst_still_delivers_and_says_what_it_withheld() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        // Nobody is seated beside the owner, and the owner answers only when
        // spoken to explicitly — so every post below is a branch nothing
        // follows, which is exactly what a tip is.
        let out = spawn(&core, &parent, &owner, vec![]);
        let room = out.space.id.clone();
        core.runtime()
            .block_on(core.set_space_participant_override(
                room.clone(),
                owner.clone(),
                eidola_app_core::ParticipantOverride {
                    notify_policy: Some(Some("explicit".into())),
                    ..Default::default()
                },
            ))
            .expect("the owner answers only an explicit ask");
        core.runtime()
            .block_on(core.add_global_participant(
                room.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("the reader joins the room they are watching");

        // A burst that lands while the walk is in its opening reads: each one
        // is a tip the report has to carry, and each is long enough that a
        // dozen of them cannot fit in one block.
        let long = |n: usize| format!("SOUNDING {n}. {}", "detail that runs on. ".repeat(120));
        let mut window = core.test_open_entry_window();
        std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &room));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches its opening window");
            for n in 1..=12 {
                core.runtime()
                    .block_on(core.post(long(n), Some(room.clone())))
                    .expect("a post into the room");
            }
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
        });

        let report = report(&core, &parent).expect("the delegation is reported");
        assert_eq!(
            report.references.len(),
            13,
            "every tip is an edge on the answer — the brief and all twelve posts"
        );
        assert!(
            report.references.iter().all(|r| r.snippet.is_some()),
            "and every one of them resolves for the reader"
        );

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
            attached.contains("SOUNDING 1."),
            "the oldest findings are the ones shown: {attached}"
        );
        assert!(
            !attached.contains("SOUNDING 12."),
            "and the block stops rather than running on forever"
        );
        assert!(
            attached.contains("not shown here") && attached.contains("further passages"),
            "the model is told its view is partial: {attached}"
        );
        assert!(
            attached.contains("a reader can open each one"),
            "and told the passages travel with its post anyway: {attached}"
        );
        assert!(
            attached.len() < 20_000,
            "the block is bounded, not merely trimmed: {} bytes",
            attached.len()
        );
    });
}

/// **A long finding is elided in the prompt and kept whole on the edge.** A
/// passage is a range its author chose with nothing bounding its length, and an
/// attached set is one entry wide per branch the walk reached — so one finding
/// could fill the report turn's context and leave the ones beside it saying
/// nothing. The prompt is what is budgeted; the record is not, because the edge
/// has to describe its passage exactly for the reader's own footnote rail.
#[test]
fn a_long_finding_is_elided_in_the_prompt_and_whole_on_the_edge() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        // A brief long enough to be over any per-passage budget, with a
        // distinctive head, middle and tail. Nobody is seated beside the owner,
        // so the room concludes on its brief and the brief is the finding.
        let brief = format!(
            "OPENING LINE.\n\n{}\n\nCLOSING LINE.",
            "middle padding that nobody needs to read. ".repeat(200)
        );
        let out = core
            .runtime()
            .block_on(core.spawn_subspace(
                parent.clone(),
                owner.clone(),
                brief.clone(),
                vec![],
                vec![],
                None,
                None,
            ))
            .expect("spawn");

        drive(&core, &out.space.id).expect("the room is driven");

        let report = report(&core, &parent).expect("the delegation is reported");
        let quoted = &report.references[0];
        assert_eq!(
            quoted.range_end,
            Some(brief.len() as i64),
            "the edge names the whole passage"
        );
        assert_eq!(
            quoted.snippet.as_deref(),
            Some(brief.as_str()),
            "and a reader's rail resolves it whole"
        );

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
            attached.len() < brief.len(),
            "the prompt is not the whole passage: {} vs {}",
            attached.len(),
            brief.len()
        );
        assert!(
            attached.contains("OPENING LINE.") && attached.contains("CLOSING LINE."),
            "both ends are paid for before the bulk is: {attached}"
        );
        assert!(
            attached.contains('…'),
            "and the cut is marked rather than silent: {attached}"
        );
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

/// **Empty router hops still spend the budget.** A router that selects nobody
/// bills an inference and writes no row, so `turns_taken_in_space` would stay
/// still while a burst of watched-room posts paid for an unbounded number of
/// planning calls. Each `plan_and_refine` in the walk counts against the same
/// ceiling persisted turns do. The first hop runs without a router so the
/// cascade window is reachable; the router is armed inside it.
#[test]
fn empty_router_hops_still_spend_the_delegation_budget() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            router: RouterBehavior::Reply(r#"{"notify": []}"#.into()),
            ..MockConfig::default()
        });
        add_backend(&core, &mock);
        let parent = parent_with_a_post(&core);
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1_000))
            .expect("cascade limit");
        let owner = shared_agent(&core, &parent, "Navigator");
        let a = shared_agent(&core, &parent, "Surveyor");
        let b = shared_agent(&core, &parent, "Pilot");
        let out = spawn(&core, &parent, &owner, vec![a, b]);
        let room = out.space.id.clone();
        core.runtime()
            .block_on(core.add_global_participant(
                room.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("the reader joins the room they are watching");

        let mut window = core.test_open_cascade_window();
        std::thread::scope(|scope| {
            let walking = scope.spawn(|| drive(&core, &room));
            let resume = core
                .runtime()
                .block_on(window.recv())
                .expect("the walk reaches its window");
            // Armed *inside* the window: a router on the brief that selects
            // nobody would drive no turn, and this window would never open.
            core.test_register_loaded_local_model("local", ROUTER_SLUG, mock.port());
            core.runtime()
                .block_on(core.set_space_router_model(room.clone(), Some(ROUTER_MODEL.into())))
                .expect("set the room's router");
            for i in 0..(MAX_DELEGATION_TURNS + 8) {
                core.runtime()
                    .block_on(core.post(format!("And what about day {i}?"), Some(room.clone())))
                    .expect("a post into the room");
            }
            resume.send(()).expect("the walk resumes");
            walking.join().expect("the walk finishes").expect("driven");
        });

        let report = report(&core, &parent).expect("a spent budget is reported");
        assert_eq!(
            report.references[0].delegation_end,
            Some(DelegationEnd::BudgetSpent {
                limit: MAX_DELEGATION_TURNS
            }),
            "planning hops bind the budget, not only persisted turns"
        );
        let router_calls = mock
            .chat_bodies()
            .iter()
            .filter(|b| b["model"] == ROUTER_MODEL)
            .count();
        assert!(
            router_calls as i64 <= MAX_DELEGATION_TURNS,
            "the router is not billed past the ceiling: {router_calls} calls"
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

/// Combined `chat` is a cascade, and a live sub-space's cascade belongs to
/// the driver. Joining lets the human post; the combined door still refuses,
/// and the running driver still takes the turn that post arms.
#[test]
fn a_combined_ask_does_not_double_drive_a_delegated_room() {
    run(|| {
        let (_mock, core, _dir) = setup();
        core.start_subspace_driver();
        let parent = parent_with_a_post(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(&core, &parent, &owner, vec![helper]);
        let sub = out.space.id.clone();
        wait_until(&core, || tree(&core, &sub).len() == 2);

        core.runtime()
            .block_on(core.add_global_participant(
                sub.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("join");

        let err = core
            .runtime()
            .block_on(core.chat("Anyone home?".into(), MODEL.into(), Some(sub.clone())))
            .expect_err("combined chat would double-drive");
        match err {
            AppError::DrivenConversation { space_id } => assert_eq!(space_id, sub),
            other => panic!("expected DrivenConversation, got {other:?}"),
        }

        core.runtime()
            .block_on(core.post("Anyone home?".into(), Some(sub.clone())))
            .expect("a member still posts");
        wait_until(&core, || {
            tree(&core, &sub)
                .iter()
                .any(|n| n.action_type == "user_input")
                && tree(&core, &sub).len() >= 4
        });
        let room = tree(&core, &sub);
        assert!(
            room.iter().any(|n| n.action_type == "user_input"),
            "the joined post landed: {room:?}"
        );
        assert!(
            room.iter().filter(|n| n.action_type == "inference").count() >= 2,
            "the driver answered the brief and the human's post: {room:?}"
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
