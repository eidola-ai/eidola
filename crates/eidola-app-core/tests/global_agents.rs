//! Global agents and in-place promotion (task 36) on the in-process
//! mock-upstream harness (`chat_harness`).
//!
//! What is pinned here:
//!
//! * **Promotion is in place** — the same participant row, the same id, so
//!   authorship and memory continuity are structural. The pinned
//!   `participant_scope` echo on past `action` rows *and* on `memory_block`
//!   rows follows via `ON UPDATE CASCADE` (proven under this turso build by
//!   `db::tests::turso_enforcement_smoke` case (e)); the former owner space
//!   gains a reference row so the agent stays a member with its persona
//!   unchanged.
//! * **One identity, many spaces** — turns in two different spaces record the
//!   *same* participant id.
//! * **Memory is a no-op for promotion, and then spans spaces** — a `core`
//!   block written before promotion loads in a second space afterwards; a
//!   space-labelled one stays home. Neither moved.
//! * **The notebook space** — created in the promotion transaction, the
//!   residence of core blocks written from then on, hidden from the Library
//!   listing.
//! * **`list_my_spaces`** — the cross-space discovery tool, bounded by
//!   membership, attached only for a global agent (so an install that has
//!   promoted nothing sends byte-identical requests).
//! * **One-way** — every conceivable demotion is a typed error.

mod chat_harness;

use chat_harness::{
    ChatBehavior, MockConfig, MockServer, ToolScript, flat_messages, memory_section,
    system_message, system_message_with, tool_script,
};
use eidola_app_core::AppCore;
use eidola_app_core::changes::Change;
use eidola_app_core::discovery::{GLOBAL_AGENT_NOTE, LIST_MY_SPACES_TOOL_NAME};
use eidola_app_core::memory::MEMORY_NOTE;

/// The external backend's model. The turn-scoped tools attach only where the
/// backend can carry a `tools` field, which excludes the `eidola` backend.
const MODEL: &str = "qwen3-8b@ext";

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

fn setup(script: ToolScript) -> (MockServer, AppCore, tempfile::TempDir) {
    let (mock, core, dir) = chat_harness::core_for(MockConfig {
        chat: ChatBehavior::ToolScript,
        tool_script: script,
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

fn turn(core: &AppCore, prompt: &str, space: Option<String>) -> eidola_app_core::ChatResult {
    core.runtime()
        .block_on(core.chat(prompt.to_string(), MODEL.to_string(), space))
        .unwrap_or_else(|e| panic!("turn {prompt:?} failed: {e}"))
}

/// The space's agent participant answering for [`MODEL`].
fn agent_id(core: &AppCore, space: &str) -> String {
    core.runtime()
        .block_on(core.list_space_participants(space.to_string()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent" && p.model_ref.as_deref() == Some(MODEL))
        .expect("an agent for the model")
        .id
}

fn member(
    core: &AppCore,
    space: &str,
    participant: &str,
) -> Option<eidola_app_core::ParticipantInfo> {
    core.runtime()
        .block_on(core.list_space_participants(space.to_string()))
        .expect("participants")
        .into_iter()
        .find(|p| p.id == participant)
}

fn actions(core: &AppCore, space: &str) -> Vec<eidola_app_core::db::RawActionRow> {
    core.runtime()
        .block_on(core.test_space_actions(space.to_string()))
        .expect("actions")
}

fn blocks(core: &AppCore, participant: &str) -> Vec<eidola_app_core::memory::MemoryBlockInfo> {
    core.runtime()
        .block_on(core.memory_blocks(participant.to_string()))
        .expect("memory blocks")
}

fn promote(core: &AppCore, participant: &str) -> eidola_app_core::PromotionOutcome {
    core.runtime()
        .block_on(core.promote_participant(participant.to_string()))
        .expect("promotion")
}

fn script_remember(script: &ToolScript, arguments: serde_json::Value) {
    *script.lock().unwrap() = vec![("remember".into(), arguments.to_string())];
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
    }
    out
}

/// The whole mechanism in one test: a space-owned agent with a real trail (an
/// inference it wrote, a memory block it holds) is promoted, and afterwards
/// **nothing about its identity moved** — same id, same actions, same blocks —
/// while the pinned scope echo followed everywhere it appears.
#[test]
fn promotion_is_in_place_and_the_scope_echo_cascades() {
    run(|| {
        let script = tool_script();
        let (_mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        script_remember(
            &script,
            serde_json::json!({ "block": "plan", "text": "Ship the parser first." }),
        );
        let first = turn(&core, "What should we build?", None);
        let space = first.space_id.clone();
        let agent = agent_id(&core, &space);

        // Before: owned by this space, and every row it authored echoes 'space'.
        let before = member(&core, &space, &agent).expect("a member");
        assert_eq!(
            (before.scope.as_str(), before.source.as_str()),
            ("space", "owned")
        );
        let authored_before: Vec<_> = actions(&core, &space)
            .into_iter()
            .filter(|a| a.participant_id == agent)
            .collect();
        assert!(
            !authored_before.is_empty()
                && authored_before
                    .iter()
                    .all(|a| a.participant_scope == "space"),
            "the agent authored rows echoing 'space': {authored_before:#?}"
        );
        assert_eq!(blocks(&core, &agent)[0].owner_scope, "space");

        let mut rx = core.subscribe_changes();
        let outcome = promote(&core, &agent);

        // Identity: the same row, the same id — that is the whole point.
        assert_eq!(outcome.participant_id, agent);
        assert_eq!(outcome.home_space_id, space);
        let after = member(&core, &space, &agent).expect("still a member of its home space");
        assert_eq!(
            (after.scope.as_str(), after.source.as_str()),
            ("global", "referenced"),
            "in place: same row, now global and referenced into its former owner space"
        );
        // NULL overrides — the space persona is preserved byte-for-byte.
        assert_eq!(after.label, before.label);
        assert_eq!(after.model_ref, before.model_ref);
        assert_eq!(after.system_prompt, before.system_prompt);
        assert_eq!(after.notify_policy, before.notify_policy);
        let reference = after
            .reference
            .expect("a referenced global carries its detail");
        assert_eq!(
            (
                reference.override_label,
                reference.override_model_ref,
                reference.override_system_prompt,
                reference.override_notify_policy
            ),
            (None, None, None, None),
            "the membership inherits everything"
        );

        // The cascade: every past action's echo followed, and the actions
        // themselves are otherwise untouched.
        let authored_after: Vec<_> = actions(&core, &space)
            .into_iter()
            .filter(|a| a.participant_id == agent)
            .collect();
        assert_eq!(
            authored_after
                .iter()
                .map(|a| a.id.clone())
                .collect::<Vec<_>>(),
            authored_before
                .iter()
                .map(|a| a.id.clone())
                .collect::<Vec<_>>(),
            "no action was rewritten, added or dropped"
        );
        assert!(
            authored_after
                .iter()
                .all(|a| a.participant_scope == "global"),
            "ON UPDATE CASCADE carried the echo onto every past action: {authored_after:#?}"
        );

        // …and onto the memory blocks, which otherwise did not move at all.
        let held = blocks(&core, &agent);
        assert_eq!(held.len(), 1);
        assert_eq!(
            held[0].owner_scope, "global",
            "the block's echo cascaded too"
        );
        assert_eq!(held[0].owner_participant_id, agent, "owner unchanged");
        assert_eq!(held[0].name, "plan");
        assert_eq!(held[0].scope, "space", "the scope label is untouched");
        assert_eq!(held[0].space_id, space, "and so is its residence");
        assert_eq!(held[0].text, "Ship the parser first.");

        assert_eq!(
            drain(&mut rx),
            vec![Change::Participants],
            "membership and the agent library changed; the Library listing did not \
             (the one new space is a hidden notebook)"
        );
    });
}

/// The notebook space is real (a space row, with the agent as a member) and
/// hidden from the Library listing on both axes.
#[test]
fn promotion_creates_a_notebook_space_hidden_from_the_library() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space);

        let outcome = promote(&core, &agent);
        assert_ne!(outcome.notebook_space_id, space);

        for include_archived in [false, true] {
            let listed = core
                .runtime()
                .block_on(core.list_spaces(include_archived))
                .expect("spaces");
            assert!(
                listed.iter().any(|s| s.id == space),
                "ordinary spaces still list (include_archived={include_archived})"
            );
            assert!(
                !listed.iter().any(|s| s.id == outcome.notebook_space_id),
                "the notebook is not a Library entry (include_archived={include_archived})"
            );
        }

        // It is a real space in every other respect, and its agent is a member.
        assert!(
            member(&core, &outcome.notebook_space_id, &agent).is_some(),
            "the agent is a member of its own notebook"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.notebook_space_id(agent.clone()))
                .expect("notebook lookup"),
            Some(outcome.notebook_space_id),
            "and the management surface can find it"
        );
    });
}

/// One identity, many spaces: after promotion the same agent answers in a
/// second space, and both spaces' inference rows name the same participant.
#[test]
fn turns_in_two_spaces_resolve_the_same_participant() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space_a = turn(&core, "First conversation.", None).space_id;
        let agent = agent_id(&core, &space_a);
        promote(&core, &agent);

        let space_b = core
            .runtime()
            .block_on(core.create_space(Some("Second".into())))
            .expect("a second space")
            .id;
        core.runtime()
            .block_on(core.add_global_participant(space_b.clone(), agent.clone()))
            .expect("the shared agent joins");
        turn(&core, "Second conversation.", Some(space_b.clone()));

        for space in [&space_a, &space_b] {
            let inference = actions(&core, space)
                .into_iter()
                .find(|a| a.action_type == "inference")
                .unwrap_or_else(|| panic!("an inference in {space}"));
            assert_eq!(
                (
                    inference.participant_id.as_str(),
                    inference.participant_scope.as_str()
                ),
                (agent.as_str(), "global"),
                "the same identity answered in {space}"
            );
        }
    });
}

/// Memory needs no migration and the loading rule now genuinely spans spaces:
/// a `core` block written **before** promotion loads in a second space
/// afterwards; a `space`-labelled one written in the same breath does not.
/// Both blocks stayed exactly where they were written.
#[test]
fn a_promoted_agents_core_memory_loads_in_another_space() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        script_remember(
            &script,
            serde_json::json!({
                "block": "about you",
                "text": "You prefer terse answers.",
                "scope": "core",
            }),
        );
        let space_a = turn(&core, "Keep it short.", None).space_id;
        script_remember(
            &script,
            serde_json::json!({ "block": "plan", "text": "Ship the parser first." }),
        );
        turn(&core, "And the plan?", Some(space_a.clone()));

        let agent = agent_id(&core, &space_a);
        promote(&core, &agent);
        let space_b = core
            .runtime()
            .block_on(core.create_space(Some("Elsewhere".into())))
            .expect("a second space")
            .id;
        core.runtime()
            .block_on(core.add_global_participant(space_b.clone(), agent.clone()))
            .expect("the shared agent joins");

        // Residence did not move — both blocks still live in space A.
        let held = blocks(&core, &agent);
        assert_eq!(held.len(), 2);
        assert!(
            held.iter().all(|b| b.space_id == space_a),
            "promotion moved no memory: {held:#?}"
        );

        turn(&core, "Where are we?", Some(space_b.clone()));
        let body = mock
            .chat_bodies()
            .pop()
            .expect("the second space's request");
        assert_eq!(
            flat_messages(&body)[0].1,
            format!(
                "{}\n\n{}",
                system_message_with(None, &[MEMORY_NOTE, GLOBAL_AGENT_NOTE]),
                memory_section(&[("about you", "core", "You prefer terse answers.")]),
            ),
            "core memory travels; the block about space A stays there"
        );

        // …and it still loads at home, alongside the space-labelled one.
        turn(&core, "Still here?", Some(space_a));
        let body = mock.chat_bodies().pop().expect("the home space's request");
        assert_eq!(
            flat_messages(&body)[0].1,
            format!(
                "{}\n\n{}",
                system_message_with(None, &[MEMORY_NOTE, GLOBAL_AGENT_NOTE]),
                memory_section(&[
                    ("about you", "core", "You prefer terse answers."),
                    ("plan", "this space", "Ship the parser first."),
                ]),
            ),
        );
    });
}

/// A `core` block a **global** agent writes is about no space, so it resides in
/// that agent's notebook — which is what the notebook is for. A space-labelled
/// one still lands in the space it is about.
#[test]
fn a_global_agents_new_core_block_resides_in_its_notebook() {
    run(|| {
        let script = tool_script();
        let (_mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        let space = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space);
        let notebook = promote(&core, &agent).notebook_space_id;

        script_remember(
            &script,
            serde_json::json!({
                "block": "about you",
                "text": "You prefer terse answers.",
                "scope": "core",
            }),
        );
        turn(&core, "Noted?", Some(space.clone()));
        script_remember(
            &script,
            serde_json::json!({ "block": "plan", "text": "Ship the parser first." }),
        );
        turn(&core, "And the plan?", Some(space.clone()));

        let held = blocks(&core, &agent);
        let core_block = held
            .iter()
            .find(|b| b.name == "about you")
            .expect("core block");
        let space_block = held.iter().find(|b| b.name == "plan").expect("space block");
        assert_eq!(
            core_block.space_id, notebook,
            "a global's core memory lives in its notebook"
        );
        assert_eq!(
            space_block.space_id, space,
            "a block about this space still lives here"
        );
        // The notebook holds the block's generations as real actions, so
        // versioning, the Record and the inspector all work unchanged.
        assert!(
            actions(&core, &notebook)
                .iter()
                .any(|a| a.action_type == "memory" && a.participant_id == agent),
            "the revision is a real action in the notebook"
        );
        // …and it is still loaded in the ordinary space, by the loading rule.
        assert!(
            blocks(&core, &agent)
                .iter()
                .any(|b| b.name == "about you" && b.scope == "core")
        );
    });
}

/// The discovery tool: attached only for a global agent, bounded by
/// membership, and it round-trips through the real turn loop.
#[test]
fn list_my_spaces_reports_membership_and_nothing_else() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());

        let space_a = turn(&core, "First conversation.", None).space_id;
        let agent = agent_id(&core, &space_a);
        let notebook = promote(&core, &agent).notebook_space_id;

        // A space the agent is NOT in — it must never appear.
        let stranger = core
            .runtime()
            .block_on(core.create_space(Some("Not yours".into())))
            .expect("a space")
            .id;
        let space_b = core
            .runtime()
            .block_on(core.create_space(Some("Second".into())))
            .expect("a space")
            .id;
        core.runtime()
            .block_on(core.add_global_participant(space_b.clone(), agent.clone()))
            .expect("joins");

        *script.lock().unwrap() = vec![(LIST_MY_SPACES_TOOL_NAME.into(), "{}".into())];
        turn(&core, "Where else are you?", Some(space_b.clone()));

        let bodies = mock.chat_bodies();
        let followup = bodies.last().expect("a follow-up round");
        let result = flat_messages(followup)
            .into_iter()
            .find(|(role, _)| role == "tool")
            .expect("a tool result")
            .1;
        assert!(
            result.starts_with("You take part in 3 conversations."),
            "{result}"
        );
        assert!(
            result.contains("\n- First conversation. · 2 posts · "),
            "{result}"
        );
        assert!(result.contains("\n- Second · 1 post · "), "{result}");
        assert!(result.contains("(this conversation)"), "{result}");
        assert!(result.contains("(your notebook)"), "{result}");
        assert!(
            !result.contains("Not yours"),
            "membership is the boundary — a space it is not in never appears: {result}"
        );
        assert_eq!(
            [&space_a, &space_b, &notebook, &stranger]
                .iter()
                .filter(|s| result.contains(s.as_str()))
                .count(),
            0,
            "the listing names conversations, not ids: {result}"
        );

        // The schema was advertised, and the note joined the system message —
        // on the last turn's first round (the pre-promotion turn carried
        // neither).
        let first = &bodies[bodies.len() - 2];
        let names: Vec<String> = first["tools"]
            .as_array()
            .expect("a tools array")
            .iter()
            .map(|t| t["function"]["name"].as_str().unwrap().to_string())
            .collect();
        assert!(
            names.contains(&LIST_MY_SPACES_TOOL_NAME.to_string()),
            "{names:?}"
        );
        assert_eq!(
            flat_messages(first)[0].1,
            system_message_with(None, &[GLOBAL_AGENT_NOTE])
        );
    });
}

/// Nothing is enabled until something is promoted: a space-owned agent's turn
/// carries no `tools` field and no global-agent note, byte-identical to an
/// install that has never heard of task 36.
#[test]
fn a_space_owned_agent_is_offered_no_cross_space_tool() {
    run(|| {
        let (mock, core, _dir) = setup(tool_script());
        turn(&core, "How do tides work?", None);

        let body = mock.chat_bodies().pop().expect("one request");
        assert!(
            body.get("tools").is_none(),
            "no tools field before any promotion: {body}"
        );
        assert_eq!(flat_messages(&body)[0].1, system_message(None));
    });
}

/// Promotion is one-way and narrow: every conceivable "demote" or mis-target
/// is a typed error, and none of them writes anything.
#[test]
fn promotion_is_one_way_and_refuses_everything_else() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space);
        let notebook = promote(&core, &agent).notebook_space_id;

        let refuse = |participant: &str| -> String {
            core.runtime()
                .block_on(core.promote_participant(participant.to_string()))
                .expect_err("must be refused")
                .to_string()
        };

        // Already global — this is the only shape a demotion could take, and
        // there is no API for it.
        assert!(
            refuse(&agent).contains("already a shared agent"),
            "{}",
            refuse(&agent)
        );
        // The shared human.
        let human = member(&core, &space, eidola_app_core::db::HUMAN_PARTICIPANT_ID)
            .expect("You is a member")
            .id;
        assert!(
            refuse(&human).contains("already shared"),
            "{}",
            refuse(&human)
        );
        // A template's agent, which belongs to no space.
        let template_agent = core
            .runtime()
            .block_on(core.list_space_templates())
            .expect("templates")
            .into_iter()
            .next()
            .expect("the Default template")
            .participants
            .into_iter()
            .next()
            .expect("its agent")
            .id;
        assert!(
            refuse(&template_agent).contains("space template"),
            "{}",
            refuse(&template_agent)
        );
        // Unknown.
        assert!(
            refuse("00000000-0000-7000-8000-0000000000ff").contains("not found"),
            "{}",
            refuse("00000000-0000-7000-8000-0000000000ff")
        );

        // Nothing was written by any of it: still exactly one notebook, and the
        // promoted agent is still global and still a member of both spaces.
        assert_eq!(
            core.runtime()
                .block_on(core.notebook_space_id(agent.clone()))
                .expect("lookup"),
            Some(notebook.clone())
        );
        assert_eq!(
            member(&core, &space, &agent).expect("member").scope,
            "global"
        );
        assert!(member(&core, &notebook, &agent).is_some());
    });
}

/// The companion refusals on the join side: a space-owned agent cannot be
/// referenced into another space (promote it first), and joining twice is a
/// no-op rather than an error.
#[test]
fn only_a_global_can_join_another_space_and_joining_is_idempotent() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space_a = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space_a);
        let space_b = core
            .runtime()
            .block_on(core.create_space(Some("Second".into())))
            .expect("a space")
            .id;

        let err = core
            .runtime()
            .block_on(core.add_global_participant(space_b.clone(), agent.clone()))
            .expect_err("a space-owned agent cannot be in two places")
            .to_string();
        assert!(err.contains("share it first"), "{err}");
        assert!(member(&core, &space_b, &agent).is_none());

        promote(&core, &agent);
        for _ in 0..2 {
            core.runtime()
                .block_on(core.add_global_participant(space_b.clone(), agent.clone()))
                .expect("joining is idempotent");
        }
        assert_eq!(
            core.runtime()
                .block_on(core.list_space_participants(space_b.clone()))
                .expect("participants")
                .into_iter()
                .filter(|p| p.id == agent)
                .count(),
            1,
            "one membership, not two"
        );
    });
}

/// Shared identity composes with templates for free: a template built from a
/// space that has a global agent carries a *reference* to it (not a copy), so
/// every space instantiated from that template gets the same colleague — the
/// same participant id, the same memory — rather than a fresh stranger with
/// the same name.
#[test]
fn a_template_built_from_a_space_carries_the_shared_agent_by_reference() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space_a = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space_a);
        promote(&core, &agent);

        let template = core
            .runtime()
            .block_on(core.template_from_space(space_a, "With Ada".into()))
            .expect("a template from the space");
        // `SpaceTemplateInfo.participants` lists the template's *owned* agents
        // (what the Templates pane edits), so the reference is invisible there —
        // what matters is that instantiating carries it.

        let fresh = core
            .runtime()
            .block_on(core.create_space_from_template(template.id, Some("New room".into())))
            .expect("a space from the template")
            .id;
        let joined = member(&core, &fresh, &agent).expect("the same agent is a member");
        assert_eq!(
            (joined.scope.as_str(), joined.source.as_str()),
            ("global", "referenced"),
            "the same identity, not a copy"
        );
    });
}

// ---------------------------------------------------------------------------
// Promotion while a turn is in flight
//
// A turn spans HTTP round trips: `prepare_turn` resolves the responding
// participant, then the turn awaits the model, then it writes. Promotion flips
// that participant's scope in between. Any scope captured at preparation time
// is therefore a value that can go stale before it is written — and the pinned
// composite echo is exactly such a value.
//
// This reproduced as a hard failure: the turn died with "failed to insert
// action: FOREIGN KEY constraint failed", losing the model's completed
// response and stranding a `tool_call` with no result. The cure is structural
// rather than a guard — `db::insert_action` / `db::insert_memory_block` derive
// the echo from the participant id inside their own single statement, so there
// is no captured value and no read-then-write window (turso is single-writer,
// so a promotion cannot interleave within the statement). `ActionEntry` and
// `NewMemoryBlock` have no scope field at all: the stale state is
// unrepresentable, not merely unlikely.
// ---------------------------------------------------------------------------

/// A consumer tool that promotes its agent **from inside a turn** — precisely
/// between the turn's preparation and its remaining writes. Deterministic
/// where a wall-clock thread race would not be, and strictly more adversarial:
/// it lands in the narrowest window there is.
struct PromoteMidTurn {
    core: std::sync::Weak<AppCore>,
    participant: std::sync::Mutex<Option<String>>,
}

impl eidola_app_core::tools::Tool for PromoteMidTurn {
    fn name(&self) -> &str {
        "promote_now"
    }
    fn description(&self) -> &str {
        "Promote this agent to a shared identity, right now."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}, "required": []})
    }
    fn call<'a>(&'a self, _a: serde_json::Value) -> eidola_app_core::tools::ToolFuture<'a> {
        Box::pin(async move {
            let core = self.core.upgrade().expect("core outlives the turn");
            let id = self
                .participant
                .lock()
                .unwrap()
                .clone()
                .expect("participant id set");
            core.promote_participant(id).await.map_err(|e| {
                eidola_app_core::tools::ToolError::new(format!("promote failed: {e}"))
            })?;
            Ok("promoted".to_string())
        })
    }
}

/// Promotion mid-turn must not cost the turn its answer. The response
/// persists, with the **post-promotion** scope, and no FK error reaches the
/// caller.
#[test]
fn a_promotion_mid_turn_does_not_lose_the_response() {
    run(|| {
        let script = tool_script();
        let (_mock, core, _dir) = setup(script.clone());
        let core = std::sync::Arc::new(core);

        let space = core
            .runtime()
            .block_on(core.chat("Hello.".into(), MODEL.into(), None))
            .expect("a first turn")
            .space_id;
        let agent = agent_id(&core, &space);

        core.register_tool(std::sync::Arc::new(PromoteMidTurn {
            core: std::sync::Arc::downgrade(&core),
            participant: std::sync::Mutex::new(Some(agent.clone())),
        }))
        .expect("register");

        // Round 1 asks for the tool (which promotes); round 2 answers. Every
        // write after the tool ran — the tool result, the inference — is a
        // write whose captured scope would now be stale.
        *script.lock().unwrap() = vec![("promote_now".into(), "{}".into())];
        let result = core
            .runtime()
            .block_on(core.chat(
                "Promote yourself, then answer.".into(),
                MODEL.into(),
                Some(space.clone()),
            ))
            .expect("the turn survives a promotion landing mid-flight");
        assert!(
            result.response_action_id.is_some(),
            "the model's answer was persisted, not lost"
        );

        // The promotion really did land mid-turn…
        assert_eq!(
            member(&core, &space, &agent).expect("member").scope,
            "global"
        );
        // …and every action in the space — those written before it, and those
        // written after — carries a consistent echo.
        let rows = actions(&core, &space);
        assert!(
            rows.iter()
                .filter(|a| a.participant_id == agent)
                .all(|a| a.participant_scope == "global"),
            "post-promotion writes used the current scope, earlier ones cascaded: {rows:#?}"
        );
        // The trace is whole: the tool round has both halves, and the answer
        // followed it (the reproduction stranded a `tool_call` with no result).
        let types: Vec<&str> = rows.iter().map(|a| a.action_type.as_str()).collect();
        assert_eq!(
            types,
            [
                "user_input",
                "inference",
                "user_input",
                "tool_call",
                "tool_result",
                "inference"
            ],
            "no partial trace: {rows:#?}"
        );
    });
}

/// The same race on the memory path: a `remember` after a mid-turn promotion
/// writes its action **and** its `memory_block` identity row — the failure
/// mode being an action committed with no block row to name it. The new core
/// block also honours what the agent *is now*, residing in the notebook the
/// promotion just created rather than in the space the turn started in.
#[test]
fn a_promotion_mid_turn_does_not_strand_a_memory_write() {
    run(|| {
        let script = tool_script();
        let (_mock, core, _dir) = setup(script.clone());
        let core = std::sync::Arc::new(core);
        core.set_memory_enabled(true);

        let space = core
            .runtime()
            .block_on(core.chat("Hello.".into(), MODEL.into(), None))
            .expect("a first turn")
            .space_id;
        let agent = agent_id(&core, &space);

        core.register_tool(std::sync::Arc::new(PromoteMidTurn {
            core: std::sync::Arc::downgrade(&core),
            participant: std::sync::Mutex::new(Some(agent.clone())),
        }))
        .expect("register");

        // One round, two calls, in order: promote, then write memory.
        *script.lock().unwrap() = vec![
            ("promote_now".into(), "{}".into()),
            (
                "remember".into(),
                serde_json::json!({
                    "block": "about you",
                    "text": "You prefer terse answers.",
                    "scope": "core",
                })
                .to_string(),
            ),
        ];
        core.runtime()
            .block_on(core.chat(
                "Promote yourself, then take a note.".into(),
                MODEL.into(),
                Some(space.clone()),
            ))
            .expect("the turn survives");

        let held = blocks(&core, &agent);
        assert_eq!(held.len(), 1, "the block row was written: {held:#?}");
        assert_eq!(held[0].owner_scope, "global");
        assert_eq!(held[0].text, "You prefer terse answers.");
        // Residence follows what the agent is *now*, not what it was when the
        // turn was prepared.
        let notebook = core
            .runtime()
            .block_on(core.notebook_space_id(agent.clone()))
            .expect("lookup")
            .expect("a notebook");
        assert_eq!(
            held[0].space_id, notebook,
            "the core block landed in the notebook the promotion created"
        );
        // No orphan: the block's own action is in the notebook and resolvable.
        assert!(
            actions(&core, &notebook)
                .iter()
                .any(|a| a.action_type == "memory" && a.participant_scope == "global")
        );
    });
}

/// The **agent library** the management surface reads: every live global agent
/// with its config and its notebook, and nothing that is not a colleague.
#[test]
fn the_agent_library_lists_shared_agents_with_their_notebooks() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());

        // Nothing has been promoted, so the library is empty — even though the
        // shared human and Eidola-the-system are global rows from the first
        // boot. A global that is not an agent is not a colleague.
        assert!(
            core.runtime()
                .block_on(core.list_global_agents())
                .expect("library")
                .is_empty(),
            "an install that promoted nothing has no shared agents"
        );

        let space = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space);
        let outcome = promote(&core, &agent);

        let library = core
            .runtime()
            .block_on(core.list_global_agents())
            .expect("library");
        assert_eq!(library.len(), 1, "{library:#?}");
        let listed = &library[0];
        assert_eq!(listed.id, agent);
        assert_eq!(
            listed.notebook_space_id.as_deref(),
            Some(outcome.notebook_space_id.as_str())
        );
        // The config is the agent's own — the same row the space renders.
        let in_space = member(&core, &space, &agent).expect("still a member");
        assert_eq!(listed.label, in_space.label);
        assert_eq!(listed.model_ref, in_space.model_ref);
        assert_eq!(listed.system_prompt, in_space.system_prompt);
        assert_eq!(listed.notify_policy, in_space.notify_policy);

        // An "edit everywhere" is what the library edits, and it shows here.
        core.runtime()
            .block_on(core.update_space_participant(
                agent.clone(),
                eidola_app_core::ParticipantUpdate {
                    label: Some("Ada".into()),
                    ..Default::default()
                },
            ))
            .expect("edit everywhere");
        assert_eq!(
            core.runtime()
                .block_on(core.list_global_agents())
                .expect("library")[0]
                .label,
            "Ada"
        );
    });
}

/// **Retirement** — the counterpart to promotion, and deliberately not its
/// inverse: the agent leaves the library and its notebook is archived in the
/// same transaction, while the row, its id, its scope and its trail all stand.
#[test]
fn retiring_a_shared_agent_archives_its_notebook_and_keeps_its_trail() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space);
        let notebook = promote(&core, &agent).notebook_space_id;
        let before = actions(&core, &space).len();

        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.retire_participant(agent.clone()))
            .expect("retire");
        let emitted = drain(&mut rx);
        assert!(
            emitted.contains(&Change::Participants),
            "retirement emits Participants: {emitted:?}"
        );
        assert!(
            !emitted.contains(&Change::SpaceIndex),
            "the archived space is a notebook, which the Library never listed: {emitted:?}"
        );

        // Gone from the library…
        assert!(
            core.runtime()
                .block_on(core.list_global_agents())
                .expect("library")
                .is_empty()
        );
        // …its notebook archived in the same breath…
        assert!(
            core.runtime()
                .block_on(core.test_space_archived(notebook.clone()))
                .expect("notebook row"),
            "the notebook is archived with its agent"
        );
        // …and the trail is untouched: the actions it authored are still there.
        assert_eq!(actions(&core, &space).len(), before);
        assert!(
            actions(&core, &space)
                .iter()
                .any(|a| a.participant_id == agent),
            "a retired agent's authorship still resolves"
        );
    });
}

/// Retirement is refusal-first, and every refusal writes nothing.
#[test]
fn retirement_refuses_everything_that_is_not_a_shared_agent() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let space = turn(&core, "Hello.", None).space_id;
        let agent = agent_id(&core, &space);

        let refuse = |participant: &str| -> String {
            core.runtime()
                .block_on(core.retire_participant(participant.to_string()))
                .expect_err("must be refused")
                .to_string()
        };

        // A space-owned agent is removed from its space, not retired.
        assert!(refuse(&agent).contains("one space"), "{}", refuse(&agent));
        // The shared human.
        assert!(
            refuse(eidola_app_core::db::HUMAN_PARTICIPANT_ID).contains("yourself"),
            "{}",
            refuse(eidola_app_core::db::HUMAN_PARTICIPANT_ID)
        );
        // Eidola-the-system is a global row, but not a colleague.
        assert!(
            refuse(eidola_app_core::db::SYSTEM_PARTICIPANT_ID).contains("shared agent"),
            "{}",
            refuse(eidola_app_core::db::SYSTEM_PARTICIPANT_ID)
        );
        // Unknown.
        assert!(
            refuse("00000000-0000-7000-8000-0000000000ff").contains("not found"),
            "{}",
            refuse("00000000-0000-7000-8000-0000000000ff")
        );

        // Nothing above wrote: the agent is still space-owned and still a
        // member. Then a real retirement, and the second one is refused too.
        assert_eq!(
            member(&core, &space, &agent).expect("member").scope,
            "space"
        );
        let notebook = promote(&core, &agent).notebook_space_id;
        core.runtime()
            .block_on(core.retire_participant(agent.clone()))
            .expect("retire");
        assert!(
            refuse(&agent).contains("already been retired"),
            "{}",
            refuse(&agent)
        );
        // The double refusal changed nothing about the notebook either.
        assert!(
            core.runtime()
                .block_on(core.test_space_archived(notebook))
                .expect("notebook row")
        );
    });
}
