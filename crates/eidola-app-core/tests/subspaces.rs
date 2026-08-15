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
#[test]
fn a_brief_with_no_openable_line_still_names_its_room() {
    run(|| {
        let (_mock, core, _dir) = setup();
        let parent = space(&core);
        let owner = shared_agent(&core, &parent, "Navigator");

        let out = spawn(&core, &parent, &owner, "###", vec![], vec![]).expect("spawn");
        assert_eq!(out.space.title.as_deref(), Some("Delegated by Navigator"));
        let row = core
            .runtime()
            .block_on(core.list_spaces(true))
            .expect("spaces")
            .into_iter()
            .find(|s| s.id == out.space.id)
            .expect("listed");
        assert_eq!(row.title.as_deref(), Some("Delegated by Navigator"));
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

/// Reading a sub-space is oversight; writing into one is membership. Until the
/// join has a surface of its own, the composer's post is refused rather than
/// written by a participant the roster does not carry.
#[test]
fn a_human_cannot_post_into_a_subspace_without_joining_it() {
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

        let err = core
            .runtime()
            .block_on(core.post("Actually, check Saturday.".into(), Some(sub.clone())))
            .expect_err("posting is membership");
        match &err {
            AppError::NotJoined { space_id, message } => {
                assert_eq!(space_id, &sub);
                assert!(
                    message.contains("read") && message.contains("join"),
                    "the refusal says what the join would do: {message}"
                );
            }
            other => panic!("expected NotJoined, got {other:?}"),
        }
        // Zero trace: the room still holds only its brief.
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_tree(sub.clone()))
                .unwrap()
                .len(),
            1
        );

        // `chat` reaches the same gate — it is `post` plus a turn, and the
        // post is what commits first.
        let err = core
            .runtime()
            .block_on(core.chat(
                "Actually, check Saturday.".into(),
                MODEL.into(),
                Some(sub.clone()),
            ))
            .expect_err("chat posts first, so it is refused first");
        assert!(matches!(err, AppError::NotJoined { .. }), "{err:?}");
        assert_eq!(
            core.runtime()
                .block_on(core.get_space_tree(sub.clone()))
                .unwrap()
                .len(),
            1,
            "and no turn was funded either"
        );

        // The human's own conversations are untouched.
        core.runtime()
            .block_on(core.post("Ordinary.".into(), Some(parent)))
            .expect("posting where you are a member is unchanged");
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
            "#[doc(hidden)]\n#[cfg(any(test, feature = \"test-support\"))]\npub async fn \
             test_insert_space_capability("
        ),
        "the capability seam must be compiled out of release builds: it mints the one thing \
         the attenuation gate exists to make unmintable"
    );

    // Every function containing a write against the table, by the same
    // owner-tracking scan the stamp ledger uses.
    let mut current = "<file scope>";
    let mut writers: std::collections::BTreeSet<&str> = Default::default();
    for line in production.lines() {
        let trimmed = line.trim_start();
        for head in ["pub async fn ", "async fn ", "pub fn ", "fn "] {
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

        // The other agent's room is left alone — its owner is still live and
        // still answerable for it — and it reads fine with its parent link
        // pointing at an archived room.
        assert!(
            archived_at(&core, &nested).is_none(),
            "nobody else's delegation is closed by this retirement"
        );
        let relation = core
            .runtime()
            .block_on(core.subspace(nested.clone()))
            .unwrap()
            .expect("still a sub-space");
        assert_eq!(relation.parent_space_id, live.space.id);
        assert_eq!(relation.owner_participant_id, helper);
        assert_eq!(
            core.runtime()
                .block_on(core.live_subspaces_owned_by(helper.clone()))
                .unwrap()
                .len(),
            1,
            "and it still counts against the agent that opened it"
        );

        // An already-archived room is not archived twice: its timestamp is
        // where it was, which is what the write's own rows-affected count is
        // read from.
        assert_eq!(archived_at(&core, &already), Some(already_at));

        // Emissions: the roster changed, and so did the Library — because a
        // sub-space *is* a Library row, unlike the notebook this same
        // transaction also archived.
        let seen = drain(&mut rx);
        assert!(seen.contains(&Change::Participants), "{seen:?}");
        assert!(
            seen.contains(&Change::SpaceIndex),
            "archiving a listed room moved the listing: {seen:?}"
        );

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
