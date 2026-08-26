//! Agent-spawned sub-spaces: the data model and the spawn door.
//!
//! A sub-space is a room an agent opens to delegate work — `parent_space_id`
//! set, no human member, opened by a brief the owning agent writes. What is
//! pinned here:
//!
//! * **The door writes a whole sub-space or nothing.** Space row, memberships,
//!   brief and capability snapshot commit together; every refusal leaves zero
//!   durable trace.
//! * **No human in the room, and the human reads it anyway.** The roster the
//!   models see carries no human row, and the viewer-gated reads open the
//!   space to a human viewer regardless — oversight without membership. A
//!   notebook stays shut, which is what keeps the bypass about oversight.
//! * **The guards have teeth at the exact boundary**: depth 3 vs 4, eight live
//!   sub-spaces vs nine, and archiving one frees a slot.
//! * **Attenuation is monotonic down a chain.** A spawner cannot mint a
//!   capability it lacks, *including* one its grandparent holds and its parent
//!   does not — because the only source a spawn may draw from is its own
//!   parent's snapshot.
//! * **A sub-space is never pristine**, so neither disposal door can take it.
//! * **The brief is a post**: it notifies, it renders, and it reaches a
//!   sub-agent's turn as the thing being answered.

mod chat_harness;

use chat_harness::{ChatBehavior, MockConfig, MockServer, flat_messages};
use eidola_app_core::changes::{Change, ChangeEvent};
use eidola_app_core::error::AppError;
use eidola_app_core::{
    AppCore, MAX_LIVE_SUBSPACES_PER_OWNER, MAX_SPAWN_DEPTH, MAX_SUBAGENTS_PER_SPAWN,
    NewParticipant, NotificationPlan, SpawnRefusal, SpawnedSubspace,
};

/// The external backend's model — these tests never need a credential, so the
/// turns they do drive run over an `openai` backend.
const MODEL: &str = "qwen3-8b@ext";

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

fn setup() -> (MockServer, AppCore, tempfile::TempDir) {
    // **Answers whichever transport asked.** Several tests here drive real
    // turns, and a driven turn always streams — pointed at the blocking-only
    // behaviour those read a completion body as a stream, found no `data:`
    // frames, and passed on an empty answer nothing complained about. Surfaced
    // by the unterminated-stream rule.
    let (mock, core, dir) = chat_harness::core_for(MockConfig {
        chat: ChatBehavior::OkEitherTransport,
        ..MockConfig::default()
    });
    core.runtime()
        .block_on(core.add_backend(eidola_app_core::NewBackend {
            id: "ext".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: String::new(),
            base_url: Some(mock.base_url.clone()),
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        }))
        .expect("add backend");
    (mock, core, dir)
}

/// A parent conversation **with something said in it** — which is what one
/// always is at a spawn, since a spawn happens inside a turn answering a post.
/// A conversation with nothing in it has nowhere for a report to land and the
/// spawn door refuses it (`SpawnRefusal::NothingToReportTo`), so an empty one
/// would be testing a refusal rather than the thing under test.
fn space(core: &AppCore) -> String {
    let id = core
        .runtime()
        .block_on(core.create_space(None))
        .expect("space")
        .id;
    core.runtime()
        .block_on(core.post("What do the tide tables say?".into(), Some(id.clone())))
        .expect("post");
    id
}

/// A **shared** agent taking part in `space` — the only kind of participant
/// that can own a sub-space, because a space-owned one cannot be referenced
/// into another space at all.
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

fn spawn(
    core: &AppCore,
    parent: &str,
    owner: &str,
    brief: &str,
    participants: Vec<String>,
    capabilities: Vec<String>,
) -> Result<SpawnedSubspace, AppError> {
    spawn_from(core, parent, owner, brief, participants, capabilities, None)
}

/// A spawn that names the post in the parent it is being opened from — what a
/// turn-scoped caller supplies, and what the eventual report attaches to.
fn spawn_from(
    core: &AppCore,
    parent: &str,
    owner: &str,
    brief: &str,
    participants: Vec<String>,
    capabilities: Vec<String>,
    parent_action_id: Option<&str>,
) -> Result<SpawnedSubspace, AppError> {
    core.runtime().block_on(core.spawn_subspace(
        parent.to_string(),
        owner.to_string(),
        brief.to_string(),
        participants,
        capabilities,
        None,
        parent_action_id.map(str::to_string),
    ))
}

fn refusal(err: AppError) -> SpawnRefusal {
    match err {
        AppError::SpawnRefused { refusal } => refusal,
        other => panic!("expected a spawn refusal, got {other:?}"),
    }
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<ChangeEvent>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c.change);
    }
    out
}

fn listed(core: &AppCore, id: &str) -> bool {
    core.runtime()
        .block_on(core.list_spaces(true))
        .expect("spaces")
        .iter()
        .any(|s| s.id == id)
}

// ===========================================================================
// The door
// ===========================================================================

#[test]
fn a_spawn_opens_a_titled_room_with_no_human_in_it() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 2))
            .expect("cascade limit");

        let mut rx = core.subscribe_changes();
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables for Friday.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");

        // The Library lists it like any other conversation, titled from the
        // brief's own opening line — a reader has no prompt of their own to
        // recognize it by.
        assert!(listed(&core, &out.space.id), "sub-spaces are Library rows");
        assert_eq!(
            out.space.title.as_deref(),
            Some("Check the tide tables for Friday.")
        );
        assert_eq!(out.space.parent_space_id, parent);
        assert_eq!(out.space.owner_participant_id, owner);

        // Settings are inherited from the parent: a delegation never gets a
        // looser runaway guard than the room it was delegated from.
        let settings = core
            .runtime()
            .block_on(core.space_settings(out.space.id.clone()))
            .expect("settings");
        assert_eq!(settings.cascade_limit, 2);

        // The roster: the owner, the sub-agent, and nobody else. **No human.**
        let roster = core
            .runtime()
            .block_on(core.list_space_participants(out.space.id.clone()))
            .expect("roster");
        assert_eq!(roster.len(), 2, "owner + sub-agent only: {roster:?}");
        assert!(
            roster.iter().all(|p| p.kind != "human"),
            "a sub-space has no human member: {roster:?}"
        );
        let owner_row = roster.iter().find(|p| p.id == owner).expect("owner seated");
        assert_eq!(owner_row.role, "owner");
        // Written, not inherited — see
        // `a_notify_all_owner_is_quiet_among_its_helpers_and_answers_a_human`.
        assert_eq!(owner_row.notify_policy, "human");
        let helper_row = roster.iter().find(|p| p.id == helper).expect("sub-agent");
        assert_eq!(helper_row.role, "member");
        // The one override a spawn writes, and the reason a human-less room is
        // not inert: the seeded `human` policy fires only on a human's post,
        // and there will never be one here.
        assert_eq!(helper_row.notify_policy, "all");
        assert_eq!(
            helper_row
                .reference
                .as_ref()
                .and_then(|r| r.override_notify_policy.as_deref()),
            Some("all"),
            "the override is per-membership; the agent's global row is untouched"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.list_space_participants(parent.clone()))
                .expect("parent roster")
                .into_iter()
                .find(|p| p.id == helper)
                .expect("still in the parent")
                .notify_policy,
            "human",
            "and its membership of the parent is exactly as it was"
        );

        // The brief is the room's first post, authored by the owning agent.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(out.space.id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 1, "one post: {tree:?}");
        assert_eq!(tree[0].action_id, out.brief_action_id);
        assert_eq!(tree[0].action_type, "brief");
        assert_eq!(tree[0].participant.kind, "agent");
        assert_eq!(tree[0].participant.label, "Navigator");
        let raw = core
            .runtime()
            .block_on(core.test_space_actions(out.space.id.clone()))
            .expect("actions");
        assert_eq!(raw.len(), 1);
        assert_eq!(raw[0].participant_id, owner, "the owning agent authored it");

        // One emission per thing the spawn wrote; the parent is untouched, so
        // nothing is said about it.
        let seen = drain(&mut rx);
        assert!(seen.contains(&Change::SpaceIndex), "{seen:?}");
        assert!(
            seen.contains(&Change::Space(out.space.id.clone())),
            "{seen:?}"
        );
        assert!(seen.contains(&Change::Participants), "{seen:?}");
        assert!(
            !seen.contains(&Change::Space(parent.clone())),
            "the parent gained nothing: {seen:?}"
        );

        // And the reads a driver navigates by.
        let owned = core
            .runtime()
            .block_on(core.live_subspaces_owned_by(owner.clone()))
            .expect("owned");
        assert_eq!(owned.len(), 1);
        assert_eq!(owned[0].id, out.space.id);
        let children = core
            .runtime()
            .block_on(core.subspaces_of(parent.clone()))
            .expect("children");
        assert_eq!(children.len(), 1);
        let relation = core
            .runtime()
            .block_on(core.subspace(out.space.id.clone()))
            .expect("relation")
            .expect("it is a sub-space");
        assert_eq!(relation.parent_space_id, parent);
        assert_eq!(relation.owner_participant_id, owner);
        assert!(
            core.runtime()
                .block_on(core.subspace(parent.clone()))
                .expect("relation")
                .is_none(),
            "an ordinary space is not a sub-space"
        );
    });
}

#[test]
fn a_refused_spawn_writes_nothing_at_all() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        // A parent that has been closed takes no new work either — and that is
        // reachable rather than theoretical: a turn already past
        // `prepare_turn` when the archival landed runs to completion by
        // design, so the owner's spawn call is still to come. Arranged before
        // the count below, which is about what the *refusals* leave behind.
        let closed = space(&core);
        let closed_owner = shared_agent(&core, &closed, "Bystander");
        assert!(
            core.runtime()
                .block_on(core.archive_space(closed.clone()))
                .expect("archive")
        );
        let before = core
            .runtime()
            .block_on(core.list_spaces(true))
            .unwrap()
            .len();

        let mut rx = core.subscribe_changes();
        assert_eq!(
            refusal(spawn(&core, &parent, &owner, "   ", vec![], vec![]).unwrap_err()),
            SpawnRefusal::EmptyBrief
        );
        assert_eq!(
            refusal(
                spawn(
                    &core,
                    "no-such-space",
                    &owner,
                    "Do a thing.",
                    vec![],
                    vec![]
                )
                .unwrap_err()
            ),
            SpawnRefusal::UnknownParent {
                space_id: "no-such-space".into()
            }
        );
        assert_eq!(
            refusal(
                spawn(&core, &closed, &closed_owner, "Do a thing.", vec![], vec![]).unwrap_err()
            ),
            SpawnRefusal::ParentArchived {
                space_id: closed.clone()
            }
        );
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(closed))
                .unwrap()
                .is_empty(),
            "and nothing was minted under it"
        );
        // The shared human is not an agent, and Eidola takes part in nothing.
        assert!(matches!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    eidola_app_core::HUMAN_PARTICIPANT_ID,
                    "Do a thing.",
                    vec![],
                    vec![],
                )
                .unwrap_err()
            ),
            SpawnRefusal::SpawnerNotEligible { .. }
        ));
        // A space-owned agent cannot be referenced into another space, so it
        // cannot own one either.
        let unshared = core
            .runtime()
            .block_on(core.add_space_participant(
                parent.clone(),
                NewParticipant {
                    label: "Local".into(),
                    model_ref: Some(MODEL.into()),
                    system_prompt: None,
                    notify_policy: "human".into(),
                },
            ))
            .expect("add")
            .id;
        assert!(matches!(
            refusal(spawn(&core, &parent, &unshared, "Do a thing.", vec![], vec![]).unwrap_err()),
            SpawnRefusal::SpawnerNotEligible { .. }
        ));
        // …and cannot be invited into one.
        assert!(matches!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    &owner,
                    "Do a thing.",
                    vec![unshared.clone()],
                    vec![],
                )
                .unwrap_err()
            ),
            SpawnRefusal::ParticipantNotEligible { .. }
        ));
        // A retired agent is refused on both sides.
        let retired = shared_agent(&core, &parent, "Departing");
        core.runtime()
            .block_on(core.retire_participant(retired.clone()))
            .expect("retire");
        assert!(matches!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    &owner,
                    "Do a thing.",
                    vec![retired.clone()],
                    vec![],
                )
                .unwrap_err()
            ),
            SpawnRefusal::ParticipantNotEligible { .. }
        ));

        assert_eq!(
            core.runtime()
                .block_on(core.list_spaces(true))
                .unwrap()
                .len(),
            before,
            "not one of those refusals left a space behind"
        );
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(parent.clone()))
                .unwrap()
                .is_empty()
        );
        assert!(
            !drain(&mut rx)
                .iter()
                .any(|c| matches!(c, Change::SpaceIndex)),
            "and none of them announced anything"
        );
    });
}

// ===========================================================================
// The guards
// ===========================================================================

#[test]
fn the_depth_guard_admits_the_limit_and_refuses_one_past_it() {
    run(|| {
        assert_eq!(
            MAX_SPAWN_DEPTH, 3,
            "the boundary these assertions are about"
        );
        let (_mock, core, _dir) = setup();
        let root = space(&core);
        let owner = shared_agent(&core, &root, "Navigator");

        // Depths 1, 2, 3 — each spawned from the one before, by an owner that
        // is a member of it (the owner is a member of every room it opens).
        let mut parent = root.clone();
        let mut deepest = String::new();
        for depth in 1..=MAX_SPAWN_DEPTH {
            let out = spawn(
                &core,
                &parent,
                &owner,
                &format!("Level {depth}."),
                vec![],
                vec![],
            )
            .unwrap_or_else(|e| panic!("depth {depth} must be admitted: {e}"));
            parent = out.space.id.clone();
            deepest = out.space.id;
        }

        let err =
            refusal(spawn(&core, &deepest, &owner, "One too far.", vec![], vec![]).unwrap_err());
        assert_eq!(
            err,
            SpawnRefusal::TooDeep {
                depth: MAX_SPAWN_DEPTH + 1,
                limit: MAX_SPAWN_DEPTH,
            }
        );
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(deepest))
                .unwrap()
                .is_empty(),
            "the refused level left nothing"
        );
    });
}

#[test]
fn an_owner_holds_the_live_limit_and_archiving_frees_a_slot() {
    run(|| {
        assert_eq!(
            MAX_LIVE_SUBSPACES_PER_OWNER, 8,
            "the boundary these assertions are about"
        );
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");

        let mut ids = Vec::new();
        for n in 1..=MAX_LIVE_SUBSPACES_PER_OWNER {
            ids.push(
                spawn(
                    &core,
                    &parent,
                    &owner,
                    &format!("Errand {n}."),
                    vec![],
                    vec![],
                )
                .unwrap_or_else(|e| panic!("sub-space {n} must be admitted: {e}"))
                .space
                .id,
            );
        }
        assert_eq!(
            refusal(spawn(&core, &parent, &owner, "One too many.", vec![], vec![]).unwrap_err()),
            SpawnRefusal::TooManyLiveSubspaces {
                live: MAX_LIVE_SUBSPACES_PER_OWNER,
                limit: MAX_LIVE_SUBSPACES_PER_OWNER,
            }
        );

        // Archiving is the stated remedy, so it has to actually free one.
        core.runtime()
            .block_on(core.archive_space(ids[0].clone()))
            .expect("archive");
        assert_eq!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(owner.clone()))
                .unwrap()
                .len() as i64,
            MAX_LIVE_SUBSPACES_PER_OWNER - 1
        );
        spawn(&core, &parent, &owner, "The freed slot.", vec![], vec![])
            .expect("archiving frees a slot");

        // The count is per owner, not per space: a second agent starts fresh.
        let other = shared_agent(&core, &parent, "Surveyor");
        spawn(&core, &parent, &other, "A different owner.", vec![], vec![])
            .expect("another owner has its own allowance");

        // …and the parent's own history keeps the archived one.
        assert_eq!(
            core.runtime()
                .block_on(core.subspaces_of(parent))
                .unwrap()
                .len() as i64,
            MAX_LIVE_SUBSPACES_PER_OWNER + 2
        );
    });
}

/// **An owner leaving a conversation closes what it was running for it.** The
/// two rules about that one membership are not in tension: an owner cannot
/// leave the *room* (nothing can grant that membership back), and an owner
/// leaving the *parent* takes its delegations there with it — every report they
/// could still write is a turn in a conversation the writer is no longer part
/// of, refused for that reason, forever, against a meter nobody reads and a
/// live-room slot nobody gets back. A room under a conversation nobody left is
/// untouched: leaving one says nothing about another.
#[test]
fn an_owner_leaving_a_conversation_closes_the_delegations_it_ran_for_it() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let elsewhere = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        // The same agent is also at work in another conversation.
        core.runtime()
            .block_on(core.add_global_participant(
                elsewhere.clone(),
                owner.clone(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("join the other conversation");

        let here = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn")
        .space
        .id;
        // A room the helper opened from *that* room — beneath what is closing.
        let nested = spawn(
            &core,
            &here,
            &helper,
            "A sub-errand of my own.",
            vec![],
            vec![],
        )
        .expect("spawn")
        .space
        .id;
        let other = spawn(
            &core,
            &elsewhere,
            &owner,
            "A different errand.",
            vec![],
            vec![],
        )
        .expect("spawn")
        .space
        .id;

        let mut rx = core.subscribe_changes();
        assert!(
            core.runtime()
                .block_on(core.remove_space_participant(parent.clone(), owner.clone()))
                .expect("an ordinary departure from an ordinary conversation")
        );

        let archived = |id: &str| -> bool {
            core.runtime()
                .block_on(core.test_space_archived(id.to_string()))
                .expect("archived?")
        };
        assert!(
            archived(&here),
            "the delegation it was running for this conversation is closed"
        );
        assert!(
            archived(&nested),
            "and what that room had delegated onward goes with it"
        );
        assert!(
            !archived(&other),
            "a room under a conversation nobody left is untouched"
        );
        assert!(
            !archived(&parent),
            "and the conversation itself is not archived by somebody leaving it"
        );

        // The quota those rooms held is released; the one elsewhere still counts.
        assert_eq!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(owner.clone()))
                .unwrap()
                .len(),
            1,
            "only the room under the conversation it is still in"
        );

        // Each closed room announces itself — what releases a delegation
        // waiting on one of them as its parent — beside the roster and listing.
        let seen = drain(&mut rx);
        for id in [&here, &nested] {
            assert!(
                seen.contains(&Change::Space(id.clone())),
                "each closed room announces itself: {seen:?}"
            );
        }
        assert!(seen.contains(&Change::Participants), "{seen:?}");
        assert!(seen.contains(&Change::SpaceIndex), "{seen:?}");

        // The owner is out of the parent, and the room's own ownership record
        // is untouched — archival is not a departure.
        assert!(
            core.runtime()
                .block_on(core.list_space_participants(parent.clone()))
                .expect("roster")
                .iter()
                .all(|p| p.id != owner),
            "it really left the conversation"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.subspace(here.clone()))
                .unwrap()
                .expect("still a sub-space")
                .owner_participant_id,
            owner,
            "and the closed room still records who was answerable for it"
        );
    });
}

/// **Closing a conversation closes what it delegated, all the way down.**
///
/// A delegation exists to serve the room it was opened from. Left live under an
/// archived one it can never finish: its report is a turn in that archived
/// parent, which is refused at the gate every turn meets, so the room keeps
/// being armed, keeps failing to report, holds a live-room slot of an owner
/// nobody retired, and waits on an anchor no post will ever answer — with no
/// un-archive door anywhere to get it back out. The rule is therefore held at
/// the archival's own write, at every depth and whoever owns what it reaches.
#[test]
fn archiving_a_conversation_closes_the_delegations_beneath_it() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");

        // parent → room → nested → deepest, each level opened by whichever
        // agent is standing in the room above it.
        let room = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn")
        .space
        .id;
        let nested = spawn(
            &core,
            &room,
            &helper,
            "A sub-errand of my own.",
            vec![owner.clone()],
            vec![],
        )
        .expect("spawn")
        .space
        .id;
        let deepest = spawn(
            &core,
            &nested,
            &owner,
            "And one below that.",
            vec![],
            vec![],
        )
        .expect("spawn")
        .space
        .id;
        // A delegation under a conversation nobody is closing: the descent must
        // not reach it.
        let untouched = space(&core);
        let bystander = shared_agent(&core, &untouched, "Bystander");
        let elsewhere = spawn(
            &core,
            &untouched,
            &bystander,
            "Somewhere else.",
            vec![],
            vec![],
        )
        .expect("spawn")
        .space
        .id;

        let mut rx = core.subscribe_changes();
        assert!(
            core.runtime()
                .block_on(core.archive_space(parent.clone()))
                .expect("archive")
        );

        let archived = |id: &str| -> bool {
            core.runtime()
                .block_on(core.test_space_archived(id.to_string()))
                .expect("archived?")
        };
        assert!(archived(&parent), "the conversation itself");
        assert!(archived(&room), "the delegation it was running");
        assert!(
            archived(&nested),
            "and the one that room delegated onward, owned by another agent"
        );
        assert!(archived(&deepest), "at every depth, not just the first");
        assert!(
            !archived(&elsewhere),
            "and nothing under a conversation that was not closed"
        );

        // The slots those rooms held are released with them — derived from
        // liveness, so nothing had to remember to do it.
        assert!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(owner.clone()))
                .unwrap()
                .is_empty(),
            "the closed rooms stop counting against the agent that opened them"
        );
        assert!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(helper.clone()))
                .unwrap()
                .is_empty(),
            "the other owner's room is closed too — its purpose went with the room above it"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(bystander.clone()))
                .unwrap()
                .len(),
            1,
            "and an owner whose conversation nobody closed keeps its room"
        );

        // Every closed room announces itself, which is what releases a
        // delegation registered against one of them as its parent; the Library
        // hears once.
        let seen = drain(&mut rx);
        for id in [&parent, &room, &nested, &deepest] {
            assert!(
                seen.contains(&Change::Space(id.clone())),
                "each closed room announces itself: {seen:?}"
            );
        }
        assert!(seen.contains(&Change::SpaceIndex), "{seen:?}");
        assert!(
            !seen.contains(&Change::Space(elsewhere.clone())),
            "and a room nothing happened to says nothing: {seen:?}"
        );

        // Archival is still a visibility choice: the transcripts survive.
        assert!(listed(&core, &nested));
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_tree(nested.clone()))
                .expect("tree")
                .len(),
            1,
            "the brief is still there to read"
        );
    });
}

// ===========================================================================
// Attenuation — the adversarial arm
// ===========================================================================

#[test]
fn a_spawner_cannot_mint_a_capability_it_does_not_hold() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let root = space(&core);
        let owner = shared_agent(&core, &root, "Navigator");

        // Nothing in the world holds a capability, so every request is a
        // request for something that cannot be passed down.
        assert_eq!(
            refusal(
                spawn(
                    &core,
                    &root,
                    &owner,
                    "Run it in a sandbox.",
                    vec![],
                    vec!["sandbox".into()],
                )
                .unwrap_err()
            ),
            SpawnRefusal::CapabilityNotHeld {
                name: "sandbox".into()
            }
        );
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(root.clone()))
                .unwrap()
                .is_empty(),
            "a refused grant leaves no room behind"
        );

        // Given the capability, it travels — with its configuration copied
        // verbatim, because a grant is copied and never composed by the asker.
        core.runtime()
            .block_on(core.test_grant_space_capability(
                root.clone(),
                "sandbox".into(),
                "{\"cpus\":1}".into(),
            ))
            .expect("seed");
        let child = spawn(
            &core,
            &root,
            &owner,
            "Run it in a sandbox.",
            vec![],
            vec!["sandbox".into()],
        )
        .expect("a held capability travels");
        assert_eq!(child.capabilities, vec!["sandbox".to_string()]);
        let held = core
            .runtime()
            .block_on(core.space_capabilities(child.space.id.clone()))
            .expect("capabilities");
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].name, "sandbox");
        assert_eq!(
            held[0].config, "{\"cpus\":1}",
            "the config is the parent's, byte for byte"
        );

        // A capability the parent holds and the child was not given is not
        // silently present.
        core.runtime()
            .block_on(core.test_grant_space_capability(root.clone(), "search".into(), "{}".into()))
            .expect("seed");
        let narrowed = spawn(&core, &root, &owner, "Just the sandbox.", vec![], vec![])
            .expect("narrowing is always allowed");
        assert!(
            core.runtime()
                .block_on(core.space_capabilities(narrowed.space.id))
                .unwrap()
                .is_empty(),
            "absence of a row is absence of the capability"
        );
    });
}

#[test]
fn attenuation_holds_down_a_chain_a_grandparent_cannot_reach_through() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let a = space(&core);
        let owner = shared_agent(&core, &a, "Navigator");
        core.runtime()
            .block_on(core.test_grant_space_capability(a.clone(), "sandbox".into(), "{}".into()))
            .expect("seed");

        // A spawns B **without** the capability — narrowing, which is always
        // allowed.
        let b = spawn(&core, &a, &owner, "Delegate one level.", vec![], vec![])
            .expect("spawn B")
            .space
            .id;
        assert!(
            core.runtime()
                .block_on(core.space_capabilities(b.clone()))
                .unwrap()
                .is_empty()
        );

        // B's owner now asks for what B's *grandparent* holds. The only source
        // a spawn may draw from is its own parent's snapshot, so there is
        // nothing here to draw on — no walk up the chain to get wrong.
        assert_eq!(
            refusal(
                spawn(
                    &core,
                    &b,
                    &owner,
                    "Delegate two levels.",
                    vec![],
                    vec!["sandbox".into()],
                )
                .unwrap_err()
            ),
            SpawnRefusal::CapabilityNotHeld {
                name: "sandbox".into()
            }
        );

        // And the honest chain: A → B' with it, B' → C with it.
        let b2 = spawn(
            &core,
            &a,
            &owner,
            "Delegate one level, with the sandbox.",
            vec![],
            vec!["sandbox".into()],
        )
        .expect("spawn B'")
        .space
        .id;
        let c = spawn(
            &core,
            &b2,
            &owner,
            "Delegate two levels, with the sandbox.",
            vec![],
            vec!["sandbox".into()],
        )
        .expect("what B' holds, C may be given")
        .space
        .id;
        assert_eq!(
            core.runtime()
                .block_on(core.space_capabilities(c))
                .unwrap()
                .len(),
            1
        );
    });
}

// ===========================================================================
// The reaper
// ===========================================================================

#[test]
fn a_spawned_subspace_is_never_pristine() {
    run(|| {
        let (mock, core, dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let sub = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("spawn")
        .space
        .id;

        // The brief commits in the spawning transaction, so the space has an
        // action before it has existed for anyone: the disposal's first leg
        // refuses it, whatever anything else says.
        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine(sub.clone()))
                .expect("discard"),
            "the last window closing must not take a delegated conversation"
        );
        assert!(listed(&core, &sub));

        // And the parent is doubly safe: it has actions of its own, and a
        // space with a sub-space is refused by the predicate's own clause.
        assert!(
            !core
                .runtime()
                .block_on(core.discard_if_pristine(parent.clone()))
                .expect("discard")
        );

        // The startup sweep is the same predicate over every space.
        let path = dir.path().to_path_buf();
        drop(mock);
        drop(core);
        let _ = rustls::crypto::CryptoProvider::install_default(rustls_rustcrypto::provider());
        let next = AppCore::new(path.clone(), path.join("data")).expect("reopen");
        assert!(
            listed(&next, &sub),
            "the sweep leaves a delegated conversation exactly where it was"
        );
        assert_eq!(
            next.runtime()
                .block_on(next.subspaces_of(parent))
                .unwrap()
                .len(),
            1,
            "with its parent link intact"
        );
    });
}

// ===========================================================================
// Reading a room you are not in
// ===========================================================================

#[test]
fn a_human_reads_a_subspace_without_being_in_it() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("spawn");

        // The room has no human member — and the human resolves a post in it
        // anyway. Oversight is the justification: the rooms an agent opens on
        // the reader's behalf must not be the one thing the reader cannot see.
        let located = core
            .runtime()
            .block_on(core.action_location(
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                out.brief_action_id.clone(),
            ))
            .expect("a human may read it")
            .expect("some location");
        assert_eq!(located.1, out.space.id);

        // The agent's own gate is untouched: a shared agent that takes no part
        // in the sub-space is refused exactly as before.
        let outsider = shared_agent(&core, &parent, "Bystander");
        let err = core
            .runtime()
            .block_on(core.action_location(outsider.clone(), out.brief_action_id.clone()))
            .expect_err("membership still decides for an agent");
        assert!(matches!(err, AppError::NotAParticipant { .. }), "{err:?}");

        // The other viewer-gated read is the inbound index, and it takes the
        // same rule from the same place: a human sees the backlink a
        // sub-space's post made, an agent that is not in that room does not.
        let quoted = core
            .runtime()
            .block_on(core.post("The tide question.".into(), Some(parent.clone())))
            .expect("post")
            .action_id;
        core.runtime()
            .block_on(core.test_insert_unvalidated_reference(
                out.brief_action_id.clone(),
                quoted.clone(),
                1,
            ))
            .expect("edge");
        let human_sees = core
            .runtime()
            .block_on(core.references_to_visible_to(
                quoted.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            ))
            .expect("inbound");
        assert_eq!(
            human_sees
                .iter()
                .map(|r| r.action_id.as_str())
                .collect::<Vec<_>>(),
            vec![out.brief_action_id.as_str()]
        );
        assert!(
            core.runtime()
                .block_on(core.references_to_visible_to(quoted, outsider))
                .expect("inbound")
                .is_empty(),
            "an agent outside the sub-space is told nothing about it"
        );
    });
}

// ===========================================================================
// The brief is a post
// ===========================================================================

#[test]
fn the_brief_notifies_the_sub_agents_and_reaches_their_turn_as_a_post() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables for Friday.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");

        // A brief is a post, so planning off it is planning off a post — and
        // the sub-agent's `all` override is what makes it fire in a room with
        // no human to post. Asked of the mechanical computation, because the
        // consumer-facing door deliberately answers no turns for a delegated
        // room — those belong to app-core's own driver (see
        // `tests/subspace_driver.rs`).
        let plan =
            core.runtime()
                .block_on(core.mechanical_notification_plan(
                    out.space.id.clone(),
                    out.brief_action_id.clone(),
                ))
                .expect("plan");
        let turns = match plan {
            NotificationPlan::Turns(t) => t,
            other => panic!("expected turns, got {other:?}"),
        };
        assert_eq!(
            turns.len(),
            1,
            "the sub-agent, and not its owner: {turns:?}"
        );
        assert_eq!(turns[0].participant_id, helper);

        // Drive it, and the brief arrives as the post being answered.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        core.runtime()
            .block_on(core.respond_stream_as(
                out.space.id.clone(),
                helper.clone(),
                out.brief_action_id.clone(),
                tx,
            ))
            .expect("the turn runs");

        let bodies = mock.chat_bodies();
        let last = bodies.last().expect("a request was sent");
        let messages = flat_messages(last);
        assert!(
            messages.iter().any(|(role, content)| role == "user"
                && content.contains("Check the tide tables for Friday.")
                && content.contains("Navigator")),
            "the brief reaches the sub-agent as another participant's post: {messages:?}"
        );

        // And the answer lands in the sub-space beside it.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(out.space.id))
            .expect("tree");
        assert_eq!(tree.len(), 2, "brief + answer: {tree:?}");
        assert_eq!(tree[0].action_type, "brief");
        assert_eq!(tree[1].action_type, "inference");
    });
}

/// A brief is authored by an **agent**, so the room opens one cascade hop in —
/// and the cascade limit it opens under is the parent's. That is the intended
/// composition of two inherited facts, not an accident: a delegation is bounded
/// by the budget of the conversation that ordered it, counted from its first
/// word.
#[test]
fn a_brief_is_agent_authored_so_the_room_opens_one_cascade_hop_in() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        core.runtime()
            .block_on(core.set_space_cascade_limit(parent.clone(), 1))
            .expect("cascade limit");

        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables for Friday.",
            vec![helper],
            vec![],
        )
        .expect("spawn");

        match core
            .runtime()
            .block_on(core.mechanical_notification_plan(out.space.id, out.brief_action_id))
            .expect("plan")
        {
            NotificationPlan::Paused { depth, limit } => {
                assert_eq!((depth, limit), (1, 1), "the brief itself is the first hop");
            }
            other => panic!("a cascade limit of 1 must pause on the brief, got {other:?}"),
        }
    });
}

/// **A delegation is opened *from* a post, and that post has to be one here.**
/// The anchor is what the eventual report attaches to, so one naming another
/// conversation would answer a conversation nobody asked — and it is decided at
/// the write like every other guard.
#[test]
fn a_spawn_cannot_anchor_to_a_post_in_another_conversation() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let elsewhere = space(&core);
        let foreign = core
            .runtime()
            .block_on(core.post("Not here.".into(), Some(elsewhere)))
            .expect("post");

        match refusal(
            spawn_from(
                &core,
                &parent,
                &owner,
                "Check the tide tables.",
                vec![],
                vec![],
                Some(&foreign.action_id),
            )
            .expect_err("a foreign anchor is refused"),
        ) {
            SpawnRefusal::AnchorNotInParent { action_id } => {
                assert_eq!(action_id, foreign.action_id);
            }
            other => panic!("expected AnchorNotInParent, got {other:?}"),
        }
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(parent))
                .expect("list")
                .is_empty(),
            "a refused spawn leaves nothing behind"
        );

        // And an anchor that *is* a post here is kept, so the driver can find
        // it later.
        let parent2 = space(&core);
        let owner2 = shared_agent(&core, &parent2, "Pilot");
        let here = core
            .runtime()
            .block_on(core.post("Here.".into(), Some(parent2.clone())))
            .expect("post");
        let out = spawn_from(
            &core,
            &parent2,
            &owner2,
            "Check the tide tables.",
            vec![],
            vec![],
            Some(&here.action_id),
        )
        .expect("spawn");
        assert_eq!(
            out.space.parent_action_id.as_deref(),
            Some(here.action_id.as_str())
        );
        assert_eq!(
            core.runtime()
                .block_on(core.subspace(out.space.id))
                .expect("read")
                .expect("a sub-space")
                .parent_action_id
                .as_deref(),
            Some(here.action_id.as_str()),
            "and it is durable"
        );
    });
}

/// **A delegation is opened from a post the parent currently shows.** A
/// superseded wording is still a row in that conversation, and a failed
/// regeneration's hidden `error` tip is its current generation — either would
/// let the room run and then attach its report to something the transcript
/// does not render. The spawn asks the transcript's own predicate, so only
/// the current, terminal, visible post is an anchor.
#[test]
fn a_spawn_cannot_anchor_to_a_superseded_wording() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let original = core
            .runtime()
            .block_on(core.post("Check Friday.".into(), Some(parent.clone())))
            .expect("post");
        let edited = core
            .runtime()
            .block_on(core.edit_post(original.action_id.clone(), "Check Saturday.".into()))
            .expect("edit");
        assert_ne!(
            edited.action_id, original.action_id,
            "an edit mints a new generation"
        );

        match refusal(
            spawn_from(
                &core,
                &parent,
                &owner,
                "Check the tide tables.",
                vec![],
                vec![],
                Some(&original.action_id),
            )
            .expect_err("a superseded wording is not an anchor"),
        ) {
            SpawnRefusal::AnchorNotInParent { action_id } => {
                assert_eq!(action_id, original.action_id);
            }
            other => panic!("expected AnchorNotInParent, got {other:?}"),
        }
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(parent.clone()))
                .expect("list")
                .is_empty(),
            "a refused spawn leaves nothing behind"
        );

        let out = spawn_from(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
            Some(&edited.action_id),
        )
        .expect("the current wording is an anchor");
        assert_eq!(
            out.space.parent_action_id.as_deref(),
            Some(edited.action_id.as_str())
        );
    });
}

/// **A delegation that could never be reported is refused before it costs
/// anything.** The asymmetry is the whole argument: a refused spawn writes
/// nothing and spends nothing, while a room that runs its turns and then finds
/// no post in its parent to reply to has spent real money on work nobody will
/// ever be told about.
#[test]
fn a_spawn_with_nowhere_to_report_back_to_is_refused() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = core
            .runtime()
            .block_on(core.create_space(None))
            .expect("space")
            .id;
        let owner = shared_agent(&core, &parent, "Navigator");

        assert!(
            matches!(
                refusal(
                    spawn(
                        &core,
                        &parent,
                        &owner,
                        "Check the tide tables.",
                        vec![],
                        vec![]
                    )
                    .expect_err("a conversation with nothing in it is refused")
                ),
                SpawnRefusal::NothingToReportTo
            ),
            "an empty parent has nowhere for a report to land"
        );
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(parent.clone()))
                .expect("list")
                .is_empty(),
            "and the refusal costs nothing"
        );

        // One post is all it takes: that post is what the report replies to.
        core.runtime()
            .block_on(core.post("Look into the tides.".into(), Some(parent.clone())))
            .expect("post");
        spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("now there is somewhere to report back to");
    });
}

// ===========================================================================
// The owner membership is structural
// ===========================================================================

/// Ownership is one membership row, so ordinary roster work must not be able
/// to end it or add a second one. Both guards ride the write — a spawn and a
/// removal are separate transactions, and the roster a caller read a moment
/// ago is exactly what a concurrent spawn changes.
#[test]
fn a_subspace_owner_cannot_leave_and_cannot_be_joined_by_a_second_owner() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();

        // The owner cannot be taken out of the room it opened.
        let err = core
            .runtime()
            .block_on(core.remove_space_participant(sub.clone(), owner.clone()))
            .expect_err("the owner membership is structural");
        assert!(
            err.to_string().contains("Navigator") && err.to_string().contains("archive"),
            "the refusal names the agent and the remedy: {err}"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.subspace(sub.clone()))
                .unwrap()
                .expect("still a sub-space")
                .owner_participant_id,
            owner,
            "and ownership is exactly where it was"
        );

        // An ordinary member still leaves — the guard is about the owner, not
        // about sub-spaces being frozen.
        assert!(
            core.runtime()
                .block_on(core.remove_space_participant(sub.clone(), helper.clone()))
                .expect("an ordinary membership ends normally")
        );

        // And nobody else can be granted ownership of it.
        let outsider = shared_agent(&core, &parent, "Interloper");
        for (who, how) in [
            (outsider.clone(), "a stranger"),
            (helper.clone(), "a member that left"),
        ] {
            let err = core
                .runtime()
                .block_on(core.grant_space_membership(
                    sub.clone(),
                    who.clone(),
                    eidola_app_core::MembershipRole::Owner,
                ))
                .unwrap_err();
            assert!(
                err.to_string().contains("not as its owner"),
                "{how} must be refused ownership: {err}"
            );
        }
        // The same door with an ordinary role is unaffected.
        core.runtime()
            .block_on(core.grant_space_membership(
                sub.clone(),
                outsider.clone(),
                eidola_app_core::MembershipRole::Member,
            ))
            .expect("joining as a member is ordinary work");
        // …as is the other join door, which takes the role too.
        let err = core
            .runtime()
            .block_on(core.add_global_participant(
                sub.clone(),
                helper.clone(),
                Some(eidola_app_core::MembershipRole::Owner),
            ))
            .expect_err("the other join door carries the same guard");
        assert!(
            err.to_string().contains("not as its owner"),
            "and says the same thing: {err}"
        );

        // Through all of that, one owner and it is the original.
        assert_eq!(
            core.runtime()
                .block_on(core.subspace(sub.clone()))
                .unwrap()
                .expect("still a sub-space")
                .owner_participant_id,
            owner
        );
        assert_eq!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(owner.clone()))
                .unwrap()
                .len(),
            1,
            "so the quota still counts it against the agent that opened it"
        );
        assert!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(outsider))
                .unwrap()
                .is_empty()
        );

        // An ordinary space has no such rule: 'owner' is a descriptive role
        // there, and this guard must not have leaked into every roster.
        core.runtime()
            .block_on(core.grant_space_membership(
                parent.clone(),
                helper,
                eidola_app_core::MembershipRole::Owner,
            ))
            .expect("an ordinary space's roles are unconstrained");
    });
}

// ===========================================================================
// The other guards
// ===========================================================================

#[test]
fn a_room_seats_the_subagent_limit_and_refuses_one_more() {
    run(|| {
        assert_eq!(
            MAX_SUBAGENTS_PER_SPAWN, 8,
            "the boundary these assertions are about"
        );
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let mut agents: Vec<String> = (0..MAX_SUBAGENTS_PER_SPAWN)
            .map(|n| shared_agent(&core, &parent, &format!("Panelist {n}")))
            .collect();

        spawn(
            &core,
            &parent,
            &owner,
            "A full panel.",
            agents.clone(),
            vec![],
        )
        .expect("the limit itself is admitted");

        agents.push(shared_agent(&core, &parent, "One too many"));
        assert_eq!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    &owner,
                    "A panel one too large.",
                    agents.clone(),
                    vec![],
                )
                .unwrap_err()
            ),
            SpawnRefusal::TooManySubagents {
                requested: MAX_SUBAGENTS_PER_SPAWN + 1,
                limit: MAX_SUBAGENTS_PER_SPAWN,
            }
        );
        assert_eq!(
            core.runtime()
                .block_on(core.subspaces_of(parent))
                .unwrap()
                .len(),
            1,
            "and the refused room was never opened"
        );
    });
}

/// A spawn reports work scheduled, so it must not seat a participant that can
/// never take a turn. The configuration that decides it is the **base** one —
/// a spawn copies no overrides, so a model supplied by an override in the
/// parent does not travel to the child.
#[test]
fn an_agent_with_no_model_of_its_own_is_refused_a_seat() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");

        // A shared agent with no model at all.
        let mute = core
            .runtime()
            .block_on(core.add_space_participant(
                parent.clone(),
                NewParticipant {
                    label: "Mute".into(),
                    model_ref: None,
                    system_prompt: None,
                    notify_policy: "human".into(),
                },
            ))
            .expect("add")
            .id;
        core.runtime()
            .block_on(core.promote_participant(mute.clone(), None, None))
            .expect("promote");

        assert_eq!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    &owner,
                    "Ask the mute one.",
                    vec![mute.clone()],
                    vec![],
                )
                .unwrap_err()
            ),
            SpawnRefusal::NoModelConfigured {
                label: "Mute".into()
            }
        );

        // A per-space override in the parent is not the child's configuration,
        // so it does not make the agent eligible: overrides do not travel.
        core.runtime()
            .block_on(core.set_space_participant_override(
                parent.clone(),
                mute.clone(),
                eidola_app_core::ParticipantOverride {
                    model_ref: Some(Some(MODEL.into())),
                    ..Default::default()
                },
            ))
            .expect("override here");
        assert_eq!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    &owner,
                    "Ask the mute one again.",
                    vec![mute.clone()],
                    vec![],
                )
                .unwrap_err()
            ),
            SpawnRefusal::NoModelConfigured {
                label: "Mute".into()
            },
            "the child sees the base config, so the parent's override cannot vouch for it"
        );

        // Giving it a model of its own — the thing the child will actually see
        // — makes it eligible.
        core.runtime()
            .block_on(core.update_space_participant(
                mute.clone(),
                eidola_app_core::ParticipantUpdate {
                    model_ref: Some(Some(MODEL.into())),
                    ..Default::default()
                },
                eidola_app_core::ExpectedScope::Global,
            ))
            .expect("edit everywhere");
        spawn(
            &core,
            &parent,
            &owner,
            "Now it can answer.",
            vec![mute],
            vec![],
        )
        .expect("a model of its own is what it needed");

        // The owner is held to the same rule, for the same reason: it has to
        // answer in the room it opened.
        let voiceless = core
            .runtime()
            .block_on(core.add_space_participant(
                parent.clone(),
                NewParticipant {
                    label: "Voiceless".into(),
                    model_ref: None,
                    system_prompt: None,
                    notify_policy: "human".into(),
                },
            ))
            .expect("add")
            .id;
        core.runtime()
            .block_on(core.promote_participant(voiceless.clone(), None, None))
            .expect("promote");
        assert_eq!(
            refusal(
                spawn(
                    &core,
                    &parent,
                    &voiceless,
                    "I cannot speak.",
                    vec![],
                    vec![]
                )
                .unwrap_err()
            ),
            SpawnRefusal::NoModelConfigured {
                label: "Voiceless".into()
            }
        );
    });
}

/// A brief that yields no opening line still has to be findable in the
/// Library: it carries no snippet either (a brief is not what the listing's
/// fallback text reads), so a titleless row would be a blank line in a list.
///
/// It takes the **owner's label alone**. A title is persisted, and a persisted
/// string never passes through the presentation layer's translations — so a
/// sentence written here would be English in every language. A name is not
/// copy: it is locale-neutral by nature and says the one thing a reader
/// scanning the Library needs, which is whose room this is.
#[test]
fn a_brief_with_no_openable_line_still_names_its_room() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");

        let out = spawn(&core, &parent, &owner, "###", vec![], vec![]).expect("spawn");
        assert_eq!(out.space.title.as_deref(), Some("Navigator"));
        let row = core
            .runtime()
            .block_on(core.list_spaces(true))
            .expect("spaces")
            .into_iter()
            .find(|s| s.id == out.space.id)
            .expect("listed");
        assert_eq!(row.title.as_deref(), Some("Navigator"));
        assert!(
            row.snippet.is_none(),
            "which is exactly why the title has to be there: {row:?}"
        );

        // A brief with an ordinary opening line is still named by its own
        // words — the fallback is a fallback.
        let named = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("spawn");
        assert_eq!(named.space.title.as_deref(), Some("Check the tide tables."));
    });
}

// ===========================================================================
// Writing into a room you were not asked into
// ===========================================================================

/// Reading a sub-space is oversight; **speaking in one is joining it**. The
/// membership is written in the post's own transaction, so the roster the
/// models read is true of the transcript they read from that post onward — no
/// window in which somebody the roster omits has said something, and no
/// separate step a reader has to know to take first.
#[test]
fn a_humans_first_post_into_a_subspace_joins_them_to_it() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let mut rx = core.subscribe_changes();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();
        let roster = core
            .runtime()
            .block_on(core.list_space_participants(sub.clone()))
            .expect("roster");
        assert!(
            roster.iter().all(|p| p.kind != "human"),
            "the room starts with no human in it: {roster:?}"
        );
        let _ = drain(&mut rx);

        let posted = core
            .runtime()
            .block_on(core.post("Actually, check Saturday.".into(), Some(sub.clone())))
            .expect("speaking in a delegated room is how a reader joins it");
        assert_eq!(posted.space_id, sub);

        let roster = core
            .runtime()
            .block_on(core.list_space_participants(sub.clone()))
            .expect("roster");
        let human = roster
            .iter()
            .find(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
            .expect("the reader is in the room they just spoke in");
        assert_eq!(
            human.role, "member",
            "and joins as a member — `owner` is the delegation's own structural role: {roster:?}"
        );
        // The post and the membership are one commit, so the transcript never
        // carries a word by somebody the roster does not.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(sub.clone()))
            .unwrap();
        assert_eq!(tree.len(), 2, "brief + the reader's post: {tree:?}");
        let changes = drain(&mut rx);
        assert!(
            changes.iter().any(|c| matches!(c, Change::Participants)),
            "a roster changed, and every surface that renders one has to re-read: {changes:?}"
        );

        // Saying a second thing writes no second membership and announces
        // nothing about the roster — an invalidation about nothing is noise.
        core.runtime()
            .block_on(core.post("And Sunday.".into(), Some(sub.clone())))
            .expect("post");
        let changes = drain(&mut rx);
        assert!(
            !changes.iter().any(|c| matches!(c, Change::Participants)),
            "the join is idempotent and silent once it holds: {changes:?}"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.list_space_participants(sub.clone()))
                .expect("roster")
                .len(),
            2,
            "owner + the reader, once"
        );

        // `chat` is still refused, and now for the one reason that actually
        // stands in the way: the room drives its own cascade. Posting is the
        // door, and it is open.
        let err = core
            .runtime()
            .block_on(core.chat(
                "Actually, check Saturday.".into(),
                MODEL.into(),
                Some(sub.clone()),
            ))
            .expect_err("a delegated room's cascade is the driver's");
        assert!(
            matches!(err, AppError::DrivenConversation { .. }),
            "{err:?}"
        );

        // The human's own conversations are untouched.
        core.runtime()
            .block_on(core.post("Ordinary.".into(), Some(parent.clone())))
            .expect("posting where you are a member is unchanged");
        assert_eq!(
            core.runtime()
                .block_on(core.list_space_participants(parent))
                .expect("roster")
                .iter()
                .filter(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
                .count(),
            1,
            "and joins nobody twice"
        );
    });
}

/// **Combined post-and-turn is a cascade, and a live sub-space's cascade
/// belongs to the driver.** `chat` / `chat_stream` would post and then drive
/// the same notify-all seat the driver will take — twice the spend. Joining
/// takes the membership gate out of the way so this is the kind rule that
/// answers; posting itself remains, because that is what a joined reader
/// does and what arms the room.
#[test]
fn a_combined_ask_in_a_live_subspace_is_refused() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();
        core.runtime()
            .block_on(core.add_global_participant(
                sub.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("join");

        let mut rx = core.subscribe_changes();
        let _ = drain(&mut rx);

        let chat_err = core
            .runtime()
            .block_on(core.chat("Anyone home?".into(), MODEL.into(), Some(sub.clone())))
            .expect_err("combined chat would double-drive the room");
        match &chat_err {
            AppError::DrivenConversation { space_id } => assert_eq!(space_id, &sub),
            other => panic!("expected DrivenConversation, got {other:?}"),
        }

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let stream_err = core
            .runtime()
            .block_on(core.chat_stream("Anyone home?".into(), MODEL.into(), Some(sub.clone()), tx))
            .expect_err("combined stream would double-drive the room");
        match &stream_err {
            AppError::DrivenConversation { space_id } => assert_eq!(space_id, &sub),
            other => panic!("expected DrivenConversation, got {other:?}"),
        }

        assert!(
            drain(&mut rx).is_empty(),
            "a refused combined ask writes nothing and emits nothing"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_tree(sub.clone()))
                .expect("tree")
                .len(),
            1,
            "the room still holds only its brief"
        );

        // Posting is still the joined reader's door — the driver takes the
        // cascade from there.
        core.runtime()
            .block_on(core.post("Now I can speak.".into(), Some(sub)))
            .expect("a member posts");
    });
}

// ===========================================================================
// A brief is not something to edit or regenerate
// ===========================================================================

/// Both writes claim a post's item: an edit appends a `user_input` generation,
/// a regeneration an `inference` one. Aimed at a brief, either would replace
/// the contract a room is working from with a different kind of thing — and
/// the regeneration would pay a model to guess at text it is not shown.
#[test]
fn a_brief_can_be_neither_edited_nor_regenerated() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![],
            vec![],
        )
        .expect("spawn");

        // Two independent gates stand in front of these verbs, and this test is
        // about the inner one: joining first takes the membership gate out of
        // the way so the *kind* rule is what answers.
        // (`an_unjoined_reader_cannot_spend_or_change_what_is_already_there`
        // covers the outer one.)
        core.runtime()
            .block_on(core.add_global_participant(
                out.space.id.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("join");

        for err in [
            core.runtime()
                .block_on(core.edit_post(out.brief_action_id.clone(), "Check Saturday.".into()))
                .expect_err("a brief is not the human's post to edit"),
            core.runtime()
                .block_on(core.regenerate(out.brief_action_id.clone(), MODEL.into()))
                .expect_err("a brief was not inferred, so there is nothing to try again"),
        ] {
            assert!(matches!(err, AppError::WrongPostKind { .. }), "{err:?}");
        }

        // Zero trace: one generation, still a brief, still the same words.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(out.space.id))
            .expect("tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].action_type, "brief");
        assert_eq!(tree[0].generation_count, 1);

        // And the ordinary cases still work: a human edits their own post, and
        // an agent's answer is regenerable.
        let mine = core
            .runtime()
            .block_on(core.post("Mine.".into(), Some(parent.clone())))
            .expect("post");
        core.runtime()
            .block_on(core.edit_post(mine.action_id, "Mine, edited.".into()))
            .expect("a human's own post is editable");
        // Regenerating a human post is refused for the mirror reason.
        let mine2 = core
            .runtime()
            .block_on(core.post("Mine again.".into(), Some(parent)))
            .expect("post");
        assert!(
            matches!(
                core.runtime()
                    .block_on(core.regenerate(mine2.action_id, MODEL.into()))
                    .expect_err("a human post was never inferred"),
                AppError::WrongPostKind { .. }
            ),
            "the rule is symmetric"
        );
    });
}

/// **A capability can only be minted by a spawn.**
///
/// The attenuation model is a claim about *writes*: absence of a row is
/// absence of the capability, so a second writer anywhere would make every
/// check upstream of it decorative. Two things are therefore pinned lexically,
/// where the writes live rather than in a reviewer's memory —
///
/// 1. the test seam that seeds a capability is compiled out of release builds,
///    not merely hidden from the documentation (`#[doc(hidden)]` gates docs and
///    nothing else, so without the cfg a release-built dependent could link it
///    and hand itself a grant no parent held); and
/// 2. no other function in the module writes the table.
#[test]
fn only_a_spawn_can_mint_a_capability() {
    let source = include_str!("../src/db.rs");
    let production = source
        .split_once("\n#[cfg(test)]\nmod tests {")
        .map(|(before, _)| before)
        .expect("db.rs ends in its test module");

    assert!(
        production.contains(
            "#[doc(hidden)]\n#[cfg(any(test, feature = \"test-support\"))]\npub(crate) async fn \
             test_insert_space_capability("
        ),
        "the capability seam must be compiled out of release builds *and* crate-private: it \
         mints the one thing the attenuation gate exists to make unmintable, and `test-support` \
         is a feature a downstream crate's dev-dependencies may enable"
    );

    // Every function containing a write against the table, by the same
    // owner-tracking scan the stamp ledger uses.
    let mut current = "<file scope>";
    let mut writers: std::collections::BTreeSet<&str> = Default::default();
    for line in production.lines() {
        let trimmed = line.trim_start();
        for head in [
            "pub(crate) async fn ",
            "pub async fn ",
            "async fn ",
            "pub(crate) fn ",
            "pub fn ",
            "fn ",
        ] {
            if let Some(rest) = trimmed.strip_prefix(head) {
                let len = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .count();
                let start = line.len() - rest.len();
                current = &line[start..start + len];
                break;
            }
        }
        if line.contains("INTO space_capability")
            || line.contains("UPDATE space_capability")
            || line.contains("DELETE FROM space_capability")
        {
            writers.insert(current);
        }
    }
    assert_eq!(
        writers.into_iter().collect::<Vec<_>>(),
        vec!["spawn_subspace_tx_body", "test_insert_space_capability"],
        "a capability may be written by the spawn and by the test seam, and by nothing else"
    );
}

/// **Retirement completes the set of write boundaries around ownership.** The
/// leave is refused and a second owner is refused, but retirement writes
/// neither of those rows: it soft-removes the participant and leaves its
/// memberships standing, after which every membership *question* answers "no".
/// A live sub-space left behind is therefore a Library row whose owner is
/// named by the ownership read and absent from its own roster, whose planning
/// can never reach that agent again, and which still holds a live-room quota
/// nothing can spend — and no door can repair it, because a replacement owner
/// is (correctly) refused. So the retirement archives it, in its own
/// transaction, exactly as it archives the notebook and for the same reason:
/// the room existed only because of that agent.
#[test]
fn retiring_an_agent_archives_the_rooms_it_owned_and_keeps_their_transcripts() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");

        let live = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");
        let already = spawn(&core, &parent, &owner, "An older errand.", vec![], vec![])
            .expect("spawn")
            .space
            .id;
        core.runtime()
            .block_on(core.archive_space(already.clone()))
            .expect("archive");
        let archived_at = |core: &AppCore, id: &str| -> Option<i64> {
            core.runtime()
                .block_on(core.list_spaces(true))
                .expect("spaces")
                .into_iter()
                .find(|s| s.id == id)
                .expect("listed")
                .archived_at
        };
        let already_at = archived_at(&core, &already).expect("archived");

        // A room the *other* agent spawned from the owner's room — its owner is
        // not being retired, so it is nobody else's to close.
        let nested = spawn(
            &core,
            &live.space.id,
            &helper,
            "A sub-errand of my own.",
            vec![],
            vec![],
        )
        .expect("spawn")
        .space
        .id;

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.retire_participant(owner.clone()))
            .expect("retire");

        // The room it owned is archived — in the same transaction, so no
        // ownerless-live-room state ever existed to be observed.
        assert!(
            archived_at(&core, &live.space.id).is_some(),
            "a room whose owner is gone is not left live"
        );
        assert!(
            !core
                .runtime()
                .block_on(core.list_spaces(false))
                .unwrap()
                .iter()
                .any(|s| s.id == live.space.id),
            "so the Library stops offering it"
        );
        // The quota it held is released with it, and the ownership question
        // dissolves rather than being answered wrongly.
        assert!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(owner.clone()))
                .unwrap()
                .is_empty(),
            "a retired agent holds no live rooms"
        );

        // **No live sub-space anywhere has an owner that is not one of its own
        // live members.** This is the invariant the whole write-boundary set
        // exists for, asked of every room at once rather than of the one this
        // test happens to have retired.
        for room in core
            .runtime()
            .block_on(core.subspaces_of(parent.clone()))
            .unwrap()
            .into_iter()
            .chain(
                core.runtime()
                    .block_on(core.subspaces_of(live.space.id.clone()))
                    .unwrap(),
            )
            .filter(|r| r.archived_at.is_none())
        {
            let roster: Vec<String> = core
                .runtime()
                .block_on(core.list_space_participants(room.id.clone()))
                .expect("roster")
                .into_iter()
                .map(|p| p.id)
                .collect();
            assert!(
                roster.contains(&room.owner_participant_id),
                "live room {} reports an owner that is not in its roster: {roster:?}",
                room.id
            );
        }

        // Archival is a visibility choice, not a deletion: the transcript
        // survives whole, and the human can still read it — neither membership
        // nor the read bypass filters on archival.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(live.space.id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].action_type, "brief");
        assert_eq!(
            core.runtime()
                .block_on(core.action_location(
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                    live.brief_action_id.clone(),
                ))
                .expect("a human may still read it")
                .expect("some location")
                .1,
            live.space.id
        );

        // **The other agent's room goes with it.** Its owner is not being
        // retired, but its *purpose* is: a delegation exists to serve the
        // conversation above it, and that conversation is now closed — so its
        // report would be a turn the archived parent refuses, leaving it
        // outstanding forever against a meter nobody reads. The relation
        // survives intact; only its liveness ends.
        assert!(
            archived_at(&core, &nested).is_some(),
            "a delegation beneath a closed room is closed with it"
        );
        let relation = core
            .runtime()
            .block_on(core.subspace(nested.clone()))
            .unwrap()
            .expect("still a sub-space");
        assert_eq!(relation.parent_space_id, live.space.id);
        assert_eq!(relation.owner_participant_id, helper);
        assert!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(helper.clone()))
                .unwrap()
                .is_empty(),
            "and the slot it held against its own owner's quota is released"
        );

        // An already-archived room is not archived twice: its timestamp is
        // where it was, which is what the write's own rows-affected count is
        // read from.
        assert_eq!(archived_at(&core, &already), Some(already_at));

        // Emissions: the roster changed, and so did the Library — because a
        // sub-space *is* a Library row, unlike the notebook this same
        // transaction also archived. And **every room it closed is announced**,
        // the notebook included: a room can be opened from one, so a wait can
        // be registered against one, and the announcement is what wakes it.
        let seen = drain(&mut rx);
        assert!(seen.contains(&Change::Participants), "{seen:?}");
        assert!(
            seen.contains(&Change::SpaceIndex),
            "archiving a listed room moved the listing: {seen:?}"
        );
        for id in [&live.space.id, &nested] {
            assert!(
                seen.contains(&Change::Space(id.clone())),
                "each closed room announces itself: {seen:?}"
            );
        }

        // And the unchanged half: retiring an agent that owns no live room
        // says nothing about the Library, because the notebook it archives is
        // one the listing never showed.
        let quiet = shared_agent(&core, &parent, "Quiet");
        let mut rx2 = core.subscribe_changes();
        core.runtime()
            .block_on(core.retire_participant(quiet.clone()))
            .expect("retire");
        let seen2 = drain(&mut rx2);
        assert!(seen2.contains(&Change::Participants), "{seen2:?}");
        assert!(
            !seen2.contains(&Change::SpaceIndex),
            "a notebook is not a Library row, so nothing announced one: {seen2:?}"
        );
        assert!(
            core.runtime()
                .block_on(
                    core.test_space_archived(
                        core.runtime()
                            .block_on(core.notebook_space_id(quiet))
                            .expect("notebook")
                            .expect("has one")
                    )
                )
                .expect("archived?"),
            "the notebook arm is exactly as it was"
        );
    });
}

/// **Retirement has to stop the work, not just close the door.**
///
/// Archiving the rooms a retired agent owned takes them out of the Library,
/// but a sub-space's whole point is agents answering each other with no window
/// open — so a helper's turn planned a moment before the retirement would
/// otherwise have run, persisted, and re-planned the next one, billing on in a
/// room that was closed precisely to stop it. This is the scenario the general
/// archival gates exist for; the general regression lives beside the cascade
/// doctrine in `participants_orchestration.rs`, and this one drives it through
/// the door that made it matter.
#[test]
fn retiring_an_owner_stops_the_helpers_mid_cascade() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables for Friday.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();

        // The helper's turn is planned while the room is live — this is the
        // pending work retirement has to catch.
        let pending = match core
            .runtime()
            .block_on(core.mechanical_notification_plan(sub.clone(), out.brief_action_id.clone()))
            .expect("plan")
        {
            NotificationPlan::Turns(t) => t.into_iter().next().expect("the helper's turn"),
            other => panic!("expected turns, got {other:?}"),
        };
        assert_eq!(pending.participant_id, helper);

        core.runtime()
            .block_on(core.retire_participant(owner.clone()))
            .expect("retire");

        let requests_before = mock.chat_bodies().len();

        // The pending turn refuses instead of starting.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let err = core
            .runtime()
            .block_on(core.respond_stream_as(
                sub.clone(),
                helper.clone(),
                out.brief_action_id.clone(),
                tx,
            ))
            .expect_err("a turn planned before the retirement must not start after it");
        match &err {
            AppError::SpaceArchived { space_id } => assert_eq!(space_id, &sub),
            other => panic!("expected SpaceArchived, got {other:?}"),
        }

        // And nothing new is planned there either, so a driver that re-plans
        // finds no work rather than looping.
        assert!(
            matches!(
                core.runtime()
                    .block_on(
                        core.mechanical_notification_plan(
                            sub.clone(),
                            out.brief_action_id.clone(),
                        ),
                    )
                    .expect("plan"),
                NotificationPlan::Turns(t) if t.is_empty()
            ),
            "a closed room schedules nothing"
        );

        // Nothing was spent past the retirement, and the brief is still the
        // only thing in the room.
        assert_eq!(mock.chat_bodies().len(), requests_before);
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(sub))
            .expect("tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].action_type, "brief");
    });
}

/// **A guard that wraps a statement protects nothing if the statement is
/// exported.**
///
/// `db` is a public module, so a `pub` on a raw write is a way for a release
/// dependent to do a guarded thing without the guard. Two kinds of guard are
/// bypassed that way, and both are load-bearing:
///
/// * the ones decided against **live state**, inside a transaction —
///   `remove_space_participant_tx` refuses to end the two structural
///   memberships, `join_space_participant_tx` and `grant_space_membership_tx`
///   refuse a second owner of a sub-space, `retire_participant_tx` archives
///   the rooms a retirement closes; and
/// * the ones decided against the **caller's own values**, in front of the
///   transaction, where a message belongs — `Inner::spawn_subspace` refuses a
///   brief that is empty or nothing but whitespace, which no interleaving can
///   change and which would otherwise commit a room that plans and bills turns
///   off a post saying nothing.
///
/// So the raw writes are crate-private, and this pins the class: adding one, or
/// exporting one for convenience, fails here rather than quietly reopening the
/// hole. Reads stay `pub` — they answer questions, they do not perform acts.
/// It is the sibling of `the_raw_space_insert_has_no_production_caller` in
/// `reap_pristine.rs`, which holds the raw space insert to the same rule.
///
/// **The rule is the shape, not the vintage.** The whole `_tx` family has it —
/// each is a transaction whose refusals or side effects are the reason its
/// door exists — so `promote_participant_tx` (the one-way promotion, with the
/// persona that must travel inside it), `discard_space_if_pristine` (the
/// disposal that decides *inside* the delete whether anything here was worth
/// keeping) and `instantiate_template` (the only path that mints a space with
/// participants from birth) are held here too, and so are the two whose doors
/// are the clearest case of each category above:
///
/// * `archive_space_tx` closes a room and every live delegation beneath it, and
///   `Inner::archive_space` is what **releases each closed room from any wait
///   registered against it** and tells the Library once. That release cannot be
///   done later by anybody: an archived room is never armed, so nothing on the
///   bus can deliver the closure, and there is no un-archive door to recover
///   from it. The transaction alone leaves every waiter outstanding forever.
/// * `update_template_tx` replaces a template's owned participant set, and
///   `Inner::update_template` decides **against the caller's own values in
///   front of it** — a blank title, a `cascade_limit` below 1, an unvalidated
///   participant tuple, a template already removed. That is the same category
///   as `spawn_subspace`'s empty brief, and sound only while the transaction is
///   unreachable.
///
/// **A guard duplicated into the primitive would be the same rule in two
/// places**, so the doors stay the only way in and the primitives stay
/// unreachable. The consequence for tests: a transaction-level regression
/// belongs beside the writer, in `db.rs`'s own test module
/// (`db::tests::tx_contention`, `db::tests::update_template_tx_*`), because an
/// integration test under `tests/` is an external consumer like any dependent.
///
/// **And the `_tx` suffix closes the class by itself.** The enumeration below
/// pins existence and naming, which a sweep cannot; the sweep pins *coverage*,
/// which an enumeration cannot — a writer added to the family tomorrow is
/// caught without anybody remembering to list it, which is exactly what a list
/// of eleven names had already failed at twice.
#[test]
fn the_raw_db_writers_are_not_exported() {
    let source = include_str!("../src/db.rs");
    let production = source
        .split_once("\n#[cfg(test)]\nmod tests {")
        .map(|(before, _)| before)
        .expect("db.rs ends in its test module");

    for raw in [
        // Membership.
        "insert_space_participant",
        "ensure_space_participant",
        "insert_participant_ref",
        "leave_space_participant",
        "soft_remove_participant",
        // The sub-space spawn, whose value-level guards live in `Inner`.
        "spawn_subspace_tx",
        // The capability seam: it mints the one thing attenuation exists to
        // make unmintable, and `test-support` is enable-able by a downstream
        // crate's dev-dependencies.
        "test_insert_space_capability",
        // The `_tx` family: each is a transaction whose refusals (or whose
        // side effects) are the whole reason its door exists.
        "remove_space_participant_tx",
        "join_space_participant_tx",
        "grant_space_membership_tx",
        "retire_participant_tx",
        "promote_participant_tx",
        "archive_space_tx",
        "update_template_tx",
        // Two more of the same shape that carry no `_tx` in their names.
        "discard_space_if_pristine",
        "instantiate_template",
    ] {
        assert!(
            production.contains(&format!("async fn {raw}(")),
            "{raw} should still exist — if it was renamed, rename it here too"
        );
        assert!(
            !production.contains(&format!("pub async fn {raw}(")),
            "{raw} is a raw write and must not be exported from db: reachable from outside, it \
             is a way to end a sub-space owner's membership, mint a second owner, retire an \
             agent without archiving the rooms it owned, promote one without the persona that \
             must travel inside that transaction, close a room and leave every wait registered \
             against it outstanding forever, rebuild a template's roster off values nothing \
             validated, delete a space without asking whether it was pristine, mint a space \
             with no participants, open a room on an empty brief, or hand a space a capability \
             no parent held — each of which its caller exists to refuse"
        );
    }

    // The class, not the members: every `_tx` writer is a transaction with a
    // door in front of it, so finding a `pub` one is enough to fail without
    // knowing which guard it bypasses. This is what the enumeration above kept
    // missing — it can only hold the names somebody remembered to add.
    let exported_tx: Vec<&str> = production
        .lines()
        .filter_map(|line| line.strip_prefix("pub async fn "))
        .map(|rest| {
            rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .next()
                .unwrap_or("")
        })
        .filter(|name| name.ends_with("_tx"))
        .collect();
    assert!(
        exported_tx.is_empty(),
        "these `_tx` writers are exported from db: {exported_tx:?}\n\nA `_tx` name says the \
         function is a transaction, and every transaction here has a door in front of it that \
         adds what the transaction cannot — a refusal decided against the caller's own values, \
         or a side effect (an announcement, a release, a wait torn down) that nothing can \
         perform afterwards. Make it `pub(crate)` and let its door be the only way in."
    );

    // An argument type reachable from outside is the write reachable from
    // outside, so the plan travels with the door.
    assert!(
        !production.contains("pub struct SubspacePlan"),
        "SubspacePlan is spawn_subspace_tx's argument and must not outlive its privacy"
    );

    // The leave is the sharpest of them: it is the guarded statement *minus*
    // the two refusals, so it may not even be compiled into a release build.
    assert!(
        production.contains("#[cfg(test)]\nasync fn leave_space_participant("),
        "leave_space_participant must stay test-only — the real leave is the statement inside \
         remove_space_participant_tx, which carries the notebook-owner and sub-space-owner \
         refusals this one has never had"
    );
}

/// **The owner's notify policy is written at spawn, not inherited.**
///
/// The sub-agents are `all` on purpose — nothing else would ever wake them in
/// a room with no human. The owner must not be, and leaving its override NULL
/// left that to whatever its global row happened to say: a shared agent
/// configured `all` is ordinary, and it would then be scheduled by the first
/// helper's answer, whose own answer would wake every notify-all helper again.
/// The work one spawn schedules would grow with the square of the roster until
/// the cascade guard stopped it.
///
/// `'human'` is the written policy, and the choice only shows once somebody
/// joins: both it and `'explicit'` are silent among agents, but `'human'` means
/// the agent answerable for the delegation answers the human who came to look
/// at it — which is what that agent is for — while `'explicit'` would leave it
/// deaf to them. Both halves are asserted below.
#[test]
fn a_notify_all_owner_is_quiet_among_its_helpers_and_answers_a_human() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);

        // An owner whose **own** policy is `all` — an ordinary configuration,
        // and the one that made the inherited policy a hazard.
        let owner = core
            .runtime()
            .block_on(core.add_space_participant(
                parent.clone(),
                NewParticipant {
                    label: "Loud".into(),
                    model_ref: Some(MODEL.into()),
                    system_prompt: None,
                    notify_policy: "all".into(),
                },
            ))
            .expect("add")
            .id;
        core.runtime()
            .block_on(core.promote_participant(owner.clone(), None, None))
            .expect("promote");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let second = shared_agent(&core, &parent, "Cartographer");

        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tides.",
            vec![helper.clone(), second.clone()],
            vec![],
        )
        .expect("spawn");

        // The room writes the owner's policy down rather than adopting one.
        let roster = core
            .runtime()
            .block_on(core.list_space_participants(out.space.id.clone()))
            .expect("roster");
        let owner_row = roster.iter().find(|p| p.id == owner).expect("the owner");
        assert_eq!(owner_row.notify_policy, "human");
        assert_eq!(
            owner_row
                .reference
                .as_ref()
                .and_then(|r| r.override_notify_policy.as_deref()),
            Some("human"),
            "written as a per-membership override, so the agent's global row is untouched"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.list_space_participants(parent.clone()))
                .expect("parent roster")
                .into_iter()
                .find(|p| p.id == owner)
                .expect("still in the parent")
                .notify_policy,
            "all",
            "and it is still as loud as ever everywhere else"
        );

        // A helper's answer schedules the other helper, and never the owner —
        // which is what keeps one spawn's work linear in the roster.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let answer = core
            .runtime()
            .block_on(core.respond_stream_as(
                out.space.id.clone(),
                helper.clone(),
                out.brief_action_id.clone(),
                tx,
            ))
            .expect("the helper answers")
            .response_action_id
            .expect("an answer");
        let planned = match core
            .runtime()
            .block_on(core.mechanical_notification_plan(out.space.id.clone(), answer))
            .expect("plan")
        {
            NotificationPlan::Turns(t) => {
                t.into_iter().map(|p| p.participant_id).collect::<Vec<_>>()
            }
            other => panic!("expected turns, got {other:?}"),
        };
        assert!(
            !planned.contains(&owner),
            "an agent's answer must not wake the owner: {planned:?}"
        );
        assert_eq!(
            planned,
            vec![second.clone()],
            "only the other helper, and exactly once"
        );

        // The other half of choosing `'human'`: a human who joins and speaks
        // does wake the agent answerable for the room.
        // (Arranged through the roster's add door — the join-on-post surface
        // this anticipates is not built yet, which is exactly what `post`'s own
        // refusal says.)
        core.runtime()
            .block_on(core.add_global_participant(
                out.space.id.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                Some(eidola_app_core::MembershipRole::Member),
            ))
            .expect("a human joins to look at it");
        let asked = core
            .runtime()
            .block_on(core.post("How is this going?".into(), Some(out.space.id.clone())))
            .expect("and posts");
        let planned = match core
            .runtime()
            .block_on(core.mechanical_notification_plan(out.space.id.clone(), asked.action_id))
            .expect("plan")
        {
            NotificationPlan::Turns(t) => {
                t.into_iter().map(|p| p.participant_id).collect::<Vec<_>>()
            }
            other => panic!("expected turns, got {other:?}"),
        };
        assert!(
            planned.contains(&owner),
            "the owner answers the human who came to look: {planned:?}"
        );
    });
}

/// **The owner's quiet policy is not merely written once — it stays written.**
///
/// A spawn writes the owner's membership override as `human`, but an override
/// is ordinary per-space configuration and the inspector can edit it. Setting
/// it to `all` would restore exactly the square fan-out the write was there to
/// prevent, and *clearing* it would be worse than it looks: the membership
/// would fall back to the agent's global row, which another door can flip to
/// `all` afterwards, reopening the hole through a write that never mentioned
/// this conversation.
///
/// So the rule is the airtight one rather than the clever one — the override is
/// **always present and never `all`** for a live sub-space owner — which makes
/// the global policy irrelevant here by construction, and needs no second guard
/// anywhere else. Both halves are proven below, including that the config door
/// cannot reach into the room.
#[test]
fn a_sub_space_owners_policy_cannot_be_edited_back_into_the_cascade() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tides.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();

        let policy_of = |core: &AppCore, space: &str, who: &str| -> String {
            core.runtime()
                .block_on(core.list_space_participants(space.to_string()))
                .expect("roster")
                .into_iter()
                .find(|p| p.id == who)
                .expect("a member")
                .notify_policy
        };
        let set_policy = |core: &AppCore, who: &str, value: Option<&str>| {
            core.runtime().block_on(core.set_space_participant_override(
                sub.clone(),
                who.to_string(),
                eidola_app_core::ParticipantOverride {
                    notify_policy: Some(value.map(str::to_string)),
                    ..Default::default()
                },
            ))
        };

        // Setting it to the agent-triggering value is refused…
        let err = set_policy(&core, &owner, Some("all")).expect_err("`all` is the whole hazard");
        assert!(
            err.to_string().contains("stays quiet"),
            "and the refusal says why: {err}"
        );
        // …and so is handing it back to the global row.
        assert!(
            set_policy(&core, &owner, None).is_err(),
            "inherit is the state the write exists to replace"
        );
        assert_eq!(policy_of(&core, &sub, &owner), "human", "neither landed");

        // What a reader keeps is the choice that matters.
        set_policy(&core, &owner, Some("explicit")).expect("silent is allowed");
        assert_eq!(policy_of(&core, &sub, &owner), "explicit");
        set_policy(&core, &owner, Some("human")).expect("and back again");
        assert_eq!(policy_of(&core, &sub, &owner), "human");

        // A helper is ordinary configuration — the guard is about the owner,
        // not about sub-spaces being frozen.
        set_policy(&core, &helper, Some("human")).expect("a helper is editable");
        assert_eq!(policy_of(&core, &sub, &helper), "human");

        // **The other door cannot reach in.** Flipping the agent's global
        // policy to `all` changes what it does everywhere it inherits — and
        // this room inherits nothing, which is the point of refusing the clear.
        core.runtime()
            .block_on(core.update_space_participant(
                owner.clone(),
                eidola_app_core::ParticipantUpdate {
                    notify_policy: Some("all".into()),
                    ..Default::default()
                },
                eidola_app_core::ExpectedScope::Global,
            ))
            .expect("edit everywhere");
        assert_eq!(
            policy_of(&core, &parent, &owner),
            "all",
            "loud everywhere it inherits"
        );
        assert_eq!(
            policy_of(&core, &sub, &owner),
            "human",
            "and unchanged in the room it opened"
        );

        // Which is the behaviour that actually matters: its helper's answer
        // still schedules nobody.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let answer = core
            .runtime()
            .block_on(core.respond_stream_as(
                sub.clone(),
                helper.clone(),
                out.brief_action_id.clone(),
                tx,
            ))
            .expect("the helper answers")
            .response_action_id
            .expect("an answer");
        match core
            .runtime()
            .block_on(core.mechanical_notification_plan(sub.clone(), answer))
            .expect("plan")
        {
            NotificationPlan::Turns(t) => {
                let ids: Vec<String> = t.into_iter().map(|p| p.participant_id).collect();
                assert!(
                    !ids.contains(&owner),
                    "a global flipped to `all` must not wake the owner here: {ids:?}"
                );
            }
            other => panic!("expected turns, got {other:?}"),
        }
    });
}

/// **Oversight is looking, and the line it stops at is every verb that acts.**
///
/// A human can open any room their agents opened between themselves, and the
/// window that opens is an ordinary one — composer, per-post gutter, retry. An
/// edit, a regeneration and a retry all act on somebody else's work in a
/// conversation nobody joined, and the last two **spend** doing it, so each is
/// refused before any write and before any request. Saying something is the
/// one act that is not refused — it joins the reader instead — which is what
/// makes the refusals a line rather than a wall.
///
/// `respond_stream_as` is deliberately not on that list and is checked here
/// too: it names the participant it acts as, is gated on *that* participant's
/// membership, and is the door a turn driver uses — a human-membership test
/// there would refuse a room's own agents working in their own room.
#[test]
fn an_unjoined_reader_cannot_spend_or_change_what_is_already_there() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tides.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();

        // An agent's answer exists to aim the human's verbs at — driven through
        // the door a driver uses, which the gate must leave alone.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let answer = core
            .runtime()
            .block_on(core.respond_stream_as(
                sub.clone(),
                helper.clone(),
                out.brief_action_id.clone(),
                tx,
            ))
            .expect("the room's own agent works in its own room")
            .response_action_id
            .expect("an answer");

        let requests_before = mock.chat_bodies().len();
        let models_before = mock.models_hits();
        let actions_before = core
            .runtime()
            .block_on(core.test_space_actions(sub.clone()))
            .expect("actions")
            .len();

        // Every door that acts as the human **on work already here** refuses,
        // naming the room. `post` is not among them and never could be: it is
        // how a reader joins, so a reader who has not joined is exactly the one
        // it is for.
        let edit_err = core
            .runtime()
            .block_on(core.edit_post(answer.clone(), "Say it differently.".into()))
            .expect_err("edit");
        let regen_err = core
            .runtime()
            .block_on(core.regenerate(answer.clone(), MODEL.into()))
            .expect_err("regenerate");
        let retry_err = core
            .runtime()
            .block_on(async {
                let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
                core.respond_stream(sub.clone(), MODEL.into(), answer.clone(), tx)
                    .await
            })
            .expect_err("retry");
        for (what, err) in [
            ("edit", edit_err),
            ("regenerate", regen_err),
            ("retry", retry_err),
        ] {
            match &err {
                AppError::NotJoined { space_id } => assert_eq!(space_id, &sub),
                other => panic!("{what} must refuse with NotJoined, got {other:?}"),
            }
        }

        // **Nothing written and nothing spent.** The regeneration and the retry
        // are the ones that would have cost money: `models_hits` catches a gate
        // placed after the backend is resolved, which `chat_bodies` cannot see.
        assert_eq!(
            mock.models_hits(),
            models_before,
            "a refused verb never got as far as building a client"
        );
        assert_eq!(mock.chat_bodies().len(), requests_before, "nor sent one");
        assert_eq!(
            core.runtime()
                .block_on(core.test_space_actions(sub.clone()))
                .expect("actions")
                .len(),
            actions_before,
            "and wrote nothing"
        );

        // Reading is untouched — that is the whole point of the bypass.
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_tree(sub.clone()))
                .expect("tree")
                .len(),
            2
        );

        // And **the join is a post**: speaking is what a reader does, and the
        // three verbs above are open the moment they have.
        core.runtime()
            .block_on(core.post("Now I can speak.".into(), Some(sub.clone())))
            .expect("speaking joins");
        core.runtime()
            .block_on(core.regenerate(answer, MODEL.into()))
            .expect("and may then regenerate an answer");

        // The parent — an ordinary conversation the human is a member of — was
        // never affected by any of this.
        core.runtime()
            .block_on(core.post("Ordinary.".into(), Some(parent)))
            .expect("posting where you are a member is unchanged");
    });
}

/// **You may quote what you can read — and reading a sub-space is what the
/// human bypass grants.**
///
/// The reference gate asked bare membership, so a reader who had opened a room
/// their agents opened between themselves could read a finding and not quote
/// it: the refusal landed after the draft was composed, on the one flow the
/// read bypass exists for — carrying something out of a delegated room into
/// the reader's own conversation.
///
/// Widening the gate to the read question grants nothing: quoting **copies**
/// the excerpt, and the premise is that this author can already read the
/// passage. The one space they cannot read stays refused, by that function's
/// own carve-out.
#[test]
fn a_human_may_quote_out_of_a_room_they_watch_but_not_out_of_a_notebook() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables for Friday.",
            vec![],
            vec![],
        )
        .expect("spawn");

        // The reader is not a member of the sub-space — that is the premise.
        assert!(
            !core
                .runtime()
                .block_on(core.list_space_participants(out.space.id.clone()))
                .expect("roster")
                .iter()
                .any(|p| p.id == eidola_app_core::HUMAN_PARTICIPANT_ID)
        );

        // Quoting the agent's brief into the reader's *own* conversation.
        let block = core
            .runtime()
            .block_on(core.get_space_tree(out.space.id.clone()))
            .expect("tree")[0]
            .blocks[0]
            .id
            .clone();
        let posted = core
            .runtime()
            .block_on(core.post_with_references(
                "Look what it is working on.\n\n{{ embed 1 }}".into(),
                Some(parent.clone()),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: out.brief_action_id.clone(),
                    content_block_id: Some(block),
                    range_start: Some(0),
                    range_end: Some(5),
                    annotation: None,
                }],
            ))
            .expect("a reader may quote what they can read");

        // The reference is really there, and resolves.
        let quoting = core
            .runtime()
            .block_on(core.get_space_tree(parent.clone()))
            .expect("tree")
            .into_iter()
            .find(|n| n.action_id == posted.action_id)
            .expect("the post");
        assert_eq!(quoting.references.len(), 1);
        assert_eq!(
            quoting.references[0].antecedent_action_id,
            out.brief_action_id
        );
        assert!(
            quoting.references[0].snippet.is_some(),
            "the passage came with it: {:?}",
            quoting.references[0]
        );

        // **A notebook is still refused** — the one space the reader may not
        // read, and so the one they may not quote.
        let notebook = core
            .runtime()
            .block_on(core.notebook_space_id(owner.clone()))
            .expect("notebook")
            .expect("a promoted agent has one");
        let private = core
            .runtime()
            .block_on(core.post("A private note to self.".into(), Some(notebook.clone())))
            .expect("the human may write in a notebook");
        let space_of_note = core
            .runtime()
            .block_on(core.create_space(Some("Mine".into())))
            .expect("space")
            .id;
        let err = core
            .runtime()
            .block_on(core.post_with_references(
                "Look at this.\n\n{{ embed 1 }}".into(),
                Some(space_of_note),
                None,
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: private.action_id.clone(),
                    content_block_id: None,
                    range_start: None,
                    range_end: None,
                    annotation: None,
                }],
            ))
            .expect_err("a notebook is not the reader's to quote out of");
        assert!(
            matches!(err, AppError::NotAParticipant { .. }),
            "and it refuses without naming anything: {err:?}"
        );
    });
}

// ===========================================================================
// The tool — how a model reaches the door
// ===========================================================================

/// A core whose upstream answers a scripted tool call. The script is filled in
/// at run time, because the arguments a delegation names — labels, sometimes
/// handles — only exist once the fixture space does.
fn tool_setup() -> (
    MockServer,
    AppCore,
    tempfile::TempDir,
    chat_harness::ToolScript,
) {
    let script = chat_harness::tool_script();
    let (mock, core, dir) = chat_harness::core_for(MockConfig {
        chat: ChatBehavior::ToolScript,
        tool_script: script.clone(),
        ..MockConfig::default()
    });
    core.runtime()
        .block_on(core.add_backend(eidola_app_core::NewBackend {
            id: "ext".into(),
            kind: eidola_app_core::BackendKind::OpenAi,
            display_name: String::new(),
            base_url: Some(mock.base_url.clone()),
            api_key: None,
            models_dir: None,
            model_overrides: None,
            engine_path: None,
            auto_start: true,
        }))
        .expect("add backend");
    (mock, core, dir, script)
}

/// Every tool result of the last request the mock received, in call order.
fn tool_results(mock: &MockServer) -> Vec<String> {
    let bodies = mock.chat_bodies();
    let last = bodies.last().expect("a follow-up round");
    flat_messages(last)
        .into_iter()
        .filter(|(role, _)| role == "tool")
        .map(|(_, c)| c)
        .collect()
}

/// The tool schemas advertised on the round that carried the call — the
/// second-to-last request, since the loop's follow-up round is the last.
fn advertised(mock: &MockServer) -> Vec<String> {
    let bodies = mock.chat_bodies();
    let first = &bodies[bodies.len() - 2];
    first["tools"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Ask a named participant to answer a post — the door a turn takes.
fn ask(core: &AppCore, space_id: &str, participant: &str, target: &str) {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    core.runtime()
        .block_on(core.respond_stream_as(
            space_id.to_string(),
            participant.to_string(),
            target.to_string(),
            tx,
        ))
        .expect("the ask runs");
}

/// The post a space's transcript ends on.
fn last_action(core: &AppCore, space_id: &str) -> String {
    core.runtime()
        .block_on(core.get_space_tree(space_id.to_string()))
        .expect("tree")
        .pop()
        .expect("a post")
        .action_id
}

/// **A delegation's anchor is a generation, so it follows the edit.**
///
/// A turn answering a post carries that post's id raw, and for a regeneration
/// that id comes off the answer's reply edge — the generation that was current
/// when the answer was written. Edit the post since and threading shows the
/// edit while the edge still names what it always named, so every `delegate`
/// call in that regeneration handed the door a generation the parent no longer
/// shows. The door was right to refuse it (an unshowable anchor would put the
/// report at the conversation root); what was wrong was the id, and the model
/// neither chose it nor could correct it.
#[test]
fn a_delegation_anchors_on_the_generation_the_parent_shows() {
    run(|| {
        let (mock, core, _dir, script) = tool_setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");

        // The post, and the agent's answer to it.
        let asked = last_action(&core, &parent);
        ask(&core, &parent, &owner, &asked);
        let answer = last_action(&core, &parent);

        // The reader rewords the post. Its item now shows a new generation; the
        // answer's reply edge still names the old one.
        let edited = core
            .runtime()
            .block_on(core.edit_post(asked.clone(), "What do Friday's tide tables say?".into()))
            .expect("edit the post")
            .action_id;
        assert_ne!(edited, asked, "the edit is a new generation");

        // Regenerate the answer, and let that turn delegate.
        *script.lock().unwrap() = vec![(
            eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
            serde_json::json!({ "brief": "Check the tables and report back." }).to_string(),
        )];
        core.runtime()
            .block_on(core.regenerate(answer, MODEL.to_string()))
            .expect("the regeneration runs");

        let results = tool_results(&mock);
        assert_eq!(results.len(), 1, "{results:?}");
        assert!(
            results[0].starts_with("Opened "),
            "the room opens instead of meeting a refusal nobody can act on: {}",
            results[0]
        );

        let rooms = core
            .runtime()
            .block_on(core.subspaces_of(parent.clone()))
            .expect("rooms");
        assert_eq!(rooms.len(), 1, "{rooms:?}");
        assert_eq!(
            rooms[0].parent_action_id.as_deref(),
            Some(edited.as_str()),
            "anchored on the generation the parent shows, not the one the edge names"
        );
    });
}

/// **The tool is the spawn door, reached from inside a turn.**
///
/// What only the turn knows is what it supplies: the room's owner is the
/// responding participant, the parent is the space it is answering in, and the
/// anchor its report will attach beneath is the post it is answering.
#[test]
fn the_delegate_tool_opens_a_room_from_the_turn_it_was_called_in() {
    run(|| {
        let (mock, core, _dir, script) = tool_setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let anchor = last_action(&core, &parent);

        *script.lock().unwrap() = vec![(
            eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
            serde_json::json!({
                "brief": "Read Friday's tide tables and say when the second high water is.",
                "participants": ["Surveyor"],
            })
            .to_string(),
        )];
        ask(&core, &parent, &owner, &anchor);

        // The schema was advertised and the note joined the system message —
        // both on the round that carried the call.
        assert!(
            advertised(&mock).contains(&eidola_app_core::subspaces::DELEGATE_TOOL_NAME.to_string()),
            "{:?}",
            advertised(&mock)
        );
        let bodies = mock.chat_bodies();
        let system = flat_messages(&bodies[bodies.len() - 2])[0].1.clone();
        assert!(
            system.contains(eidola_app_core::subspaces::DELEGATE_NOTE),
            "{system}"
        );

        // One room, under this conversation, owned by the agent that asked,
        // anchored on the post it was answering.
        let rooms = core
            .runtime()
            .block_on(core.subspaces_of(parent.clone()))
            .expect("rooms");
        assert_eq!(rooms.len(), 1, "{rooms:?}");
        let room = &rooms[0];
        assert_eq!(room.owner_participant_id, owner);
        assert_eq!(room.parent_space_id, parent);
        // A committed spawn *keeps* its record of which turn opened the room —
        // the report attaches beneath that turn's answer, and the driver is the
        // one that clears it when the delegation ends.
        assert_eq!(
            core.test_spawning_answer_record_count(),
            1,
            "the room the turn opened is recorded against the turn"
        );
        assert_eq!(
            room.parent_action_id.as_deref(),
            Some(anchor.as_str()),
            "the report attaches beneath this agent's answer to the post it was asked on"
        );

        // The brief is its first post, and the roster is the two agents.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(room.id.clone()))
            .expect("tree");
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].action_type, "brief");
        assert_eq!(
            tree[0].blocks[0].text.as_deref(),
            Some("Read Friday's tide tables and say when the second high water is.")
        );
        let seats = core
            .runtime()
            .block_on(core.list_space_participants(room.id.clone()))
            .expect("roster");
        assert_eq!(seats.len(), 2, "{seats:?}");
        assert!(seats.iter().any(|p| p.id == helper));
        assert!(seats.iter().all(|p| p.kind != "human"));

        // And the model reads back what it can act on: where the work went,
        // who is in it, and that the answer comes to it rather than being
        // waited for.
        let results = tool_results(&mock);
        assert_eq!(results.len(), 1, "{results:?}");
        assert!(results[0].contains(&room.id), "{}", results[0]);
        assert!(results[0].contains("\"Surveyor\""), "{}", results[0]);
        assert!(
            results[0].contains("told in this conversation when it finishes"),
            "{}",
            results[0]
        );
    });
}

/// **The gate is the one `list_my_spaces` carries, and for the same reason.**
/// A space-owned participant cannot be referenced into another space at all,
/// so it cannot own one either — and is offered no schema that could only be
/// refused.
#[test]
fn a_space_owned_agent_is_offered_no_delegation() {
    run(|| {
        let (mock, core, _dir, script) = tool_setup();
        let parent = space(&core);
        let local = core
            .runtime()
            .block_on(core.add_space_participant(
                parent.clone(),
                NewParticipant {
                    label: "Local".into(),
                    model_ref: Some(MODEL.to_string()),
                    system_prompt: None,
                    notify_policy: "human".into(),
                },
            ))
            .expect("add agent")
            .id;
        let anchor = last_action(&core, &parent);
        // The script stays empty: a turn with no tools at all never calls one.
        assert!(script.lock().unwrap().is_empty());
        ask(&core, &parent, &local, &anchor);

        let bodies = mock.chat_bodies();
        let body = bodies.last().expect("the turn's request");
        assert!(
            body.get("tools").is_none(),
            "a space-owned agent's turn carries no tools field at all: {body}"
        );
        assert!(
            !flat_messages(body)[0]
                .1
                .contains(eidola_app_core::subspaces::DELEGATE_NOTE),
            "nor the note describing an affordance it does not have"
        );
    });
}

/// **The tool is no way around a guard.** Every refusal the spawn door decides
/// inside its transaction arrives at the model as a *tool result* — correctable
/// — and leaves no room behind, which is what stops a model turning a refusal
/// **A room closed before its driver settles takes its record with it.** The
/// record of which turn opened a delegated room is cleared at
/// `drive_subspace`'s terminal exits — and an archived room never reaches
/// them: it is no longer a live delegated room, so the supervisor calls it
/// ordinary and never arms a walk for it. The archival doors are therefore the
/// only clearing there is, and without one a long-lived process opening and
/// archiving delegations grows that map for as long as it runs.
#[test]
fn archiving_a_room_before_it_settles_releases_its_spawning_record() {
    run(|| {
        let (_mock, core, _dir, script) = tool_setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let anchor = last_action(&core, &parent);

        // Three delegations, closed through both doors that reach `close_rooms`
        // — `archive_space`, which hands it whatever its transaction closed (one
        // room, or a whole subtree when the conversation above them goes), and a
        // retirement, which hands it that agent's own set. A departure is the
        // third caller of the same door with a set of the same shape.
        for brief in ["First look.", "Second look.", "Third look."] {
            *script.lock().unwrap() = vec![(
                eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
                serde_json::json!({ "brief": brief }).to_string(),
            )];
            ask(&core, &parent, &owner, &anchor);
        }
        let rooms: Vec<String> = core
            .runtime()
            .block_on(core.subspaces_of(parent.clone()))
            .expect("rooms")
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(rooms.len(), 3, "{rooms:?}");
        assert_eq!(
            core.test_spawning_answer_record_count(),
            3,
            "each open room is recorded against the turn that opened it"
        );

        // Directly.
        core.runtime()
            .block_on(core.archive_space(rooms[0].clone()))
            .expect("archive the room");
        assert_eq!(
            core.test_spawning_answer_record_count(),
            2,
            "a closed room keeps nothing"
        );

        // With its owner — a retirement archives every room that agent owns.
        core.runtime()
            .block_on(core.retire_participant(owner.clone()))
            .expect("retire the owner");
        assert_eq!(
            core.test_spawning_answer_record_count(),
            0,
            "and neither does a room closed with the agent answerable for it"
        );
    });
}

/// into a retry loop that mints anything.
#[test]
fn the_delegate_tool_is_refused_by_every_guard_the_door_holds() {
    run(|| {
        let (mock, core, _dir, script) = tool_setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let anchor = last_action(&core, &parent);

        // Fill the owner's live-room quota through the API, then ask for one
        // more through the tool.
        for i in 0..MAX_LIVE_SUBSPACES_PER_OWNER {
            spawn(
                &core,
                &parent,
                &owner,
                &format!("Room {i}."),
                vec![],
                vec![],
            )
            .expect("spawn");
        }
        *script.lock().unwrap() = vec![
            (
                eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
                serde_json::json!({ "brief": "One more." }).to_string(),
            ),
            // A brief is the whole contract, so there is no such thing as an
            // empty one.
            (
                eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
                serde_json::json!({ "brief": "   " }).to_string(),
            ),
        ];
        ask(&core, &parent, &owner, &anchor);

        let results = tool_results(&mock);
        assert_eq!(results.len(), 2, "{results:?}");
        assert!(
            results[0].contains(&format!("the limit is {MAX_LIVE_SUBSPACES_PER_OWNER}")),
            "{}",
            results[0]
        );
        assert!(results[1].contains("a brief is required"), "{}", results[1]);

        // And the attenuation gate is the door's, not the tool's: nothing in
        // production holds a capability, so asking for one is asking for
        // something unmintable — asked by an agent with a quota to spare, so
        // it is this guard answering and not the one above it.
        let second = shared_agent(&core, &parent, "Surveyor");
        *script.lock().unwrap() = vec![(
            eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
            serde_json::json!({ "brief": "Sandboxed work.", "capabilities": ["sandbox"] })
                .to_string(),
        )];
        ask(&core, &parent, &second, &anchor);
        let results = tool_results(&mock);
        assert_eq!(results.len(), 1, "{results:?}");
        assert!(
            results[0].contains("cannot grant `sandbox`"),
            "{}",
            results[0]
        );

        // A list the model mistyped is a correctable mistake, not an empty one:
        // filtered down to nothing it was indistinguishable from the advertised
        // solo mode, so a typo spent a live-room slot and set a driver working.
        *script.lock().unwrap() = vec![
            (
                eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
                serde_json::json!({ "brief": "Look this over.", "participants": [{"name": "Ada"}] })
                    .to_string(),
            ),
            (
                eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
                serde_json::json!({ "brief": "Look this over.", "participants": ["Surveyor", 7] })
                    .to_string(),
            ),
        ];
        ask(&core, &parent, &second, &anchor);
        let results = tool_results(&mock);
        assert_eq!(results.len(), 2, "{results:?}");
        assert!(results[0].contains("`participants`"), "{}", results[0]);
        assert!(results[0].contains("an object"), "{}", results[0]);
        assert!(results[1].contains("entry 2"), "{}", results[1]);

        // Every one of them left the world exactly as it was.
        assert_eq!(
            core.runtime()
                .block_on(core.subspaces_of(parent.clone()))
                .expect("rooms")
                .len() as i64,
            MAX_LIVE_SUBSPACES_PER_OWNER,
            "a refused call mints nothing"
        );
        // Including in memory. The record of which turn opened a room is
        // written before the spawning transaction and keyed by a room id, so a
        // refusal that left one behind would name a room nothing can ever reach
        // to clear — and the live-rooms ceiling is a *standing* refusal, so it
        // would be one more per attempt for the life of the process.
        assert_eq!(
            core.test_spawning_answer_record_count(),
            0,
            "a refused delegation records nothing about a room it did not open"
        );
    });
}

/// **A delegation can name only this conversation's own roster.**
///
/// A model learns who exists from the roster it is shown, and that is exactly
/// the set the tool resolves against — so an agent working elsewhere in the
/// library is not merely refused, it is unnameable. The refusal says who *is*
/// available, which is a listing the model was already given.
#[test]
fn a_delegation_can_only_name_the_conversations_own_roster() {
    run(|| {
        let (mock, core, _dir, script) = tool_setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        // A shared agent that exists, is eligible in every way the door asks
        // about, and takes part somewhere else entirely.
        let elsewhere = space(&core);
        shared_agent(&core, &elsewhere, "Archivist");
        let anchor = last_action(&core, &parent);

        *script.lock().unwrap() = vec![(
            eidola_app_core::subspaces::DELEGATE_TOOL_NAME.into(),
            serde_json::json!({ "brief": "Dig out the 1953 tables.", "participants": ["Archivist"] })
                .to_string(),
        )];
        ask(&core, &parent, &owner, &anchor);

        let results = tool_results(&mock);
        assert_eq!(results.len(), 1, "{results:?}");
        assert!(
            results[0].contains("no participant of this conversation is called \"Archivist\""),
            "{}",
            results[0]
        );
        assert!(
            results[0].contains("leave `participants` out"),
            "the refusal names the way forward: {}",
            results[0]
        );
        assert!(
            core.runtime()
                .block_on(core.subspaces_of(parent))
                .expect("rooms")
                .is_empty(),
            "and nothing was opened"
        );
    });
}

/// The name is protocol surface: a system note promises it with these
/// semantics, and what executes must be the tool that note describes.
#[test]
fn registering_the_delegate_tool_name_is_refused() {
    run(|| {
        let (_mock, core, _dir, _script) = tool_setup();
        let err = core
            .register_tool(std::sync::Arc::new(eidola_app_core::tools::EchoTool))
            .err();
        assert!(err.is_none(), "an unreserved name registers: {err:?}");
        assert!(eidola_app_core::tools::is_reserved_tool_name(
            eidola_app_core::subspaces::DELEGATE_TOOL_NAME
        ));
    });
}

/// **The roster a delegated room shows is true of the reader who joined it.**
///
/// A room of two agents is not multi-party, so its turns carry no roster at
/// all. A human posting there makes three — and the roster that appears names
/// them, in the same wire bytes every other space's does. That is the whole
/// point of joining at the post: the models are never shown a room whose roster
/// omits somebody who has spoken in it.
#[test]
fn the_roster_of_a_delegated_room_names_the_reader_who_joined_it() {
    run(|| {
        let (mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");
        let helper = shared_agent(&core, &parent, "Surveyor");
        let out = spawn(
            &core,
            &parent,
            &owner,
            "Check the tide tables.",
            vec![helper.clone()],
            vec![],
        )
        .expect("spawn");
        let sub = out.space.id.clone();

        // Two agents, linear: no roster and no trailing message at all.
        ask(&core, &sub, &helper, &out.brief_action_id);
        let bodies = mock.chat_bodies();
        let msgs = flat_messages(bodies.last().expect("the turn"));
        assert!(
            !msgs.iter().any(|(_, c)| c.contains("Participants in this")),
            "a two-party room says nothing about who is in it: {msgs:?}"
        );

        // The reader speaks, which joins them — and the next turn's roster says
        // so, naming them the way every roster names the shared human.
        core.runtime()
            .block_on(core.post("What about Saturday?".into(), Some(sub.clone())))
            .expect("speaking joins");
        let joined = core
            .runtime()
            .block_on(core.get_space_tree(sub.clone()))
            .expect("tree")
            .pop()
            .expect("the reader's post");
        ask(&core, &sub, &helper, &joined.action_id);

        let bodies = mock.chat_bodies();
        let msgs = flat_messages(bodies.last().expect("the turn"));
        let expected = chat_harness::roster(&[
            (chat_harness::HUMAN_LABEL, "human", false),
            ("Navigator", "agent", false),
            ("Surveyor", "agent", true),
        ]);
        let handle = eidola_app_core::post_handle(&joined.item_id);
        assert_eq!(
            msgs.last().expect("a trailing message").1,
            chat_harness::trailing(Some(&expected), None, &handle),
            "the roster names the reader who joined by speaking"
        );
    });
}
