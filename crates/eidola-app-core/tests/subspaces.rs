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
use eidola_app_core::changes::Change;
use eidola_app_core::error::AppError;
use eidola_app_core::{
    AppCore, MAX_LIVE_SUBSPACES_PER_OWNER, MAX_SPAWN_DEPTH, NewParticipant, NotificationPlan,
    SpawnRefusal, SpawnedSubspace,
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
    let (mock, core, dir) = chat_harness::core_for(MockConfig {
        chat: ChatBehavior::OkBlocking,
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

fn space(core: &AppCore) -> String {
    core.runtime()
        .block_on(core.create_space(None))
        .expect("space")
        .id
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
    core.runtime().block_on(core.spawn_subspace(
        parent.to_string(),
        owner.to_string(),
        brief.to_string(),
        participants,
        capabilities,
        None,
    ))
}

fn refusal(err: AppError) -> SpawnRefusal {
    match err {
        AppError::SpawnRefused { refusal } => refusal,
        other => panic!("expected a spawn refusal, got {other:?}"),
    }
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
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
        // no human to post.
        let plan = core
            .runtime()
            .block_on(core.plan_notifications(out.space.id.clone(), out.brief_action_id.clone()))
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
            .block_on(core.plan_notifications(out.space.id, out.brief_action_id))
            .expect("plan")
        {
            NotificationPlan::Paused { depth, limit } => {
                assert_eq!((depth, limit), (1, 1), "the brief itself is the first hop");
            }
            other => panic!("a cascade limit of 1 must pause on the brief, got {other:?}"),
        }
    });
}
