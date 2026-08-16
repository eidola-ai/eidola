//! Agent memory (task 35) on the in-process mock-upstream harness
//! (`chat_harness`). The `remember` calls are scripted through
//! `ChatBehavior::ToolScript`, so every test drives the real turn loop: the
//! model asks for the tool, the harness executes it, and the next turn reads
//! what was written back out of the database.
//!
//! What is pinned here:
//!
//! * **The loading rule** — a participant's blocks render at the **head**, in
//!   the system message after the charter and the notes, in exact bytes; and
//!   another participant in the same space sees none of them.
//! * **Revision, not accumulation** — writing a name again supersedes that
//!   block's tip inside one item; the prior generation survives with its own
//!   author, and the loaded set still shows one entry.
//! * **Provenance** — source handles become `reference` antecedent edges at
//!   ordinals `1..=N`, with the annotation recorded; a handle the turn's
//!   snapshot does not know is *reported in the result*, never silently
//!   dropped.
//! * **Budgets** — the per-owner block count and the per-block size refuse as
//!   tool *results* (the model can correct them) and leave **zero** durable
//!   trace.
//! * **Off by default** — with memory not enabled the request is byte-identical
//!   to one from an install that has never heard of the feature: no `tools`
//!   field, no note, no `<memory>` section.
//! * **Emissions** — a committed block emits one extra `Change::Space` (the row
//!   in `tests/bus.rs`'s exit-point table).
//!
//! Memory *quality* (what an agent chooses to remember, and whether it keeps it
//! tidy) is offline-eval material and deliberately absent here.

mod chat_harness;

use chat_harness::{
    ChatBehavior, MockConfig, MockServer, TRAILING_BLOCK_NOTE, ToolScript, flat_messages,
    memory_section, system_message_with, tool_script,
};
use eidola_app_core::AppCore;
use eidola_app_core::changes::{Change, ChangeEvent};
use eidola_app_core::memory::{MAX_MEMORY_BLOCK_BYTES, MAX_MEMORY_BLOCKS, MEMORY_NOTE};

/// The external backend's model. `remember` rides the same learned capability
/// gate the navigation tools do, so any endpoint reaches it; these tests run
/// over an `openai` backend to keep the credential spend out of the way.
const MODEL: &str = "qwen3-8b@ext";

/// The label the minted agent for [`MODEL`] carries (`db::default_agent_label`)
/// — the subject of the task-64 identity line in every request below.
const AGENT_LABEL: &str = "Qwen3 8b";

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// A core wired to the mock through an **openai-kind** backend (no spend, no
/// account), with the script the mock serves. Memory is left **off** — each
/// test switches it on when that is what it is testing.
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

/// One blocking turn.
fn turn(core: &AppCore, prompt: &str, space: Option<String>) -> eidola_app_core::ChatResult {
    turn_as(core, prompt, space, MODEL)
}

/// One blocking turn by whichever agent answers for `model` (a model the space
/// has no agent for mints one — a second participant, cheaply).
fn turn_as(
    core: &AppCore,
    prompt: &str,
    space: Option<String>,
    model: &str,
) -> eidola_app_core::ChatResult {
    core.runtime()
        .block_on(core.chat(prompt.to_string(), model.to_string(), space))
        .unwrap_or_else(|e| panic!("turn {prompt:?} failed: {e}"))
}

/// Script one `remember` call for the next request.
fn script_remember(script: &ToolScript, arguments: serde_json::Value) {
    *script.lock().unwrap() = vec![("remember".into(), arguments.to_string())];
}

/// The space's agent participant answering for `model`.
fn agent_id(core: &AppCore, space: &str, model: &str) -> String {
    core.runtime()
        .block_on(core.list_space_participants(space.to_string()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent" && p.model_ref.as_deref() == Some(model))
        .unwrap_or_else(|| panic!("no agent for {model}"))
        .id
}

fn blocks(core: &AppCore, participant: &str) -> Vec<eidola_app_core::memory::MemoryBlockInfo> {
    core.runtime()
        .block_on(core.memory_blocks(participant.to_string()))
        .expect("memory blocks")
}

/// The `tool`-role message contents of a recorded request, in order — what the
/// model was told about its own calls.
fn tool_results(body: &serde_json::Value) -> Vec<String> {
    flat_messages(body)
        .into_iter()
        .filter(|(role, _)| role == "tool")
        .map(|(_, c)| c)
        .collect()
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<ChangeEvent>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c.change);
    }
    out
}

/// The whole feature is opt-in: with memory off, a turn sends exactly what it
/// sent before task 35 — no `tools` field, no memory note, no `<memory>`
/// section, not even a read. This is the pin that keeps every existing
/// pinned-bytes expectation true.
#[test]
fn memory_off_by_default_leaves_the_request_byte_identical() {
    run(|| {
        let (mock, core, _dir) = setup(tool_script());
        assert!(!core.memory_enabled(), "memory is off unless asked for");

        turn(&core, "How do tides work?", None);

        let body = mock.chat_bodies().pop().expect("one request");
        assert!(
            body.get("tools").is_none(),
            "no tools field without the opt-in: {body}"
        );
        assert_eq!(
            flat_messages(&body)[0].1,
            system_message_with(None, AGENT_LABEL, &[TRAILING_BLOCK_NOTE]),
            "the system message is untouched"
        );
    });
}

/// The whole round trip: the model calls `remember`, the block is committed,
/// and the **next** turn reads it back at the head of the prompt in exact
/// bytes — after the charter and the notes, before any conversation.
#[test]
fn remember_writes_a_block_that_the_next_turn_loads_at_the_head() {
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
        let first = turn(&core, "How do tides work?", None);
        let space = first.space_id.clone();

        // Round 2 of the *same* turn carries the receipt but not the block:
        // the prompt prefix was fixed when the turn was prepared, which is
        // exactly the cache behaviour the head placement promises.
        let bodies = mock.chat_bodies();
        assert_eq!(
            tool_results(&bodies[1]),
            ["Wrote `about you` (core). You now hold 1 of 8 memory blocks."]
        );
        assert!(
            !flat_messages(&bodies[1])[0].1.contains("<memory>"),
            "a mid-turn write does not rewrite the turn's own prefix"
        );

        turn(&core, "And why two per day?", Some(space));

        let body = mock.chat_bodies().pop().expect("the second turn's request");
        let msgs = flat_messages(&body);
        assert_eq!(
            msgs[0].1,
            format!(
                "{}\n\n{}",
                system_message_with(None, AGENT_LABEL, &[TRAILING_BLOCK_NOTE, MEMORY_NOTE]),
                memory_section(&[("about you", "core", "You prefer terse answers.")]),
            ),
            "memory renders at the head, after the charter and the notes"
        );
        assert_eq!(msgs[0].0, "system");
        assert!(
            msgs[1..].iter().all(|(_, c)| !c.contains("<memory>")),
            "and nowhere else: {msgs:#?}"
        );
    });
}

/// Writing a name again is a **revision**: one block, one item, two
/// generations — the prior text stays readable in the trail, and the loaded
/// set still shows exactly one entry.
#[test]
fn a_second_write_of_a_name_supersedes_rather_than_accumulating() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        script_remember(
            &script,
            serde_json::json!({ "block": "plan", "text": "Ship the parser first." }),
        );
        let first = turn(&core, "What should we do?", None);
        let space = first.space_id.clone();

        script_remember(
            &script,
            serde_json::json!({ "block": "plan", "text": "Ship the renderer first." }),
        );
        turn(&core, "Changed my mind.", Some(space.clone()));

        let agent = agent_id(&core, &space, MODEL);
        let held = blocks(&core, &agent);
        assert_eq!(held.len(), 1, "one block, not two: {held:#?}");
        let block = &held[0];
        assert_eq!(block.name, "plan");
        assert_eq!(block.scope, "space", "the default scope is this space");
        assert_eq!(block.space_id, space, "residence is the space it is about");
        assert_eq!(block.owner_participant_id, agent);
        assert_eq!(block.text, "Ship the renderer first.");
        assert_eq!(block.revisions.len(), 2, "append-only history");
        assert_eq!(block.revisions[0].text, "Ship the parser first.");
        // Both generations are the agent's own — this field is what would
        // distinguish a later human correction.
        assert!(
            block
                .revisions
                .iter()
                .all(|r| r.author_participant_id == agent),
            "every revision records its author: {:#?}",
            block.revisions
        );

        // The revision's receipt says so, and the next turn loads one entry.
        // The second turn's follow-up round replays the first turn's round
        // (traces are first-person, task 33) and then its own receipt.
        let bodies = mock.chat_bodies();
        assert_eq!(
            tool_results(&bodies[3]),
            [
                "Wrote `plan` (this space). You now hold 1 of 8 memory blocks.",
                "Revised `plan` (this space, revision 2). You hold 1 of 8 memory blocks.",
            ]
        );
        turn(&core, "And then?", Some(space));
        let body = mock.chat_bodies().pop().expect("third turn");
        assert_eq!(
            flat_messages(&body)[0].1,
            format!(
                "{}\n\n{}",
                system_message_with(None, AGENT_LABEL, &[TRAILING_BLOCK_NOTE, MEMORY_NOTE]),
                memory_section(&[("plan", "this space", "Ship the renderer first.")]),
            ),
        );
    });
}

/// Source handles become real `reference` antecedent edges (ordinals `1..=N`,
/// ordinal 0 left free — memory has no `reply` edge), carrying the annotation.
/// A handle the snapshot does not know is stated in the result.
#[test]
fn a_revision_records_the_posts_it_learned_from() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        let first = turn(&core, "I always want short answers.", None);
        let space = first.space_id.clone();
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree");
        let source_action = tree[0].action_id.clone();
        let source_handle = eidola_app_core::post_handle(&tree[0].item_id);

        script_remember(
            &script,
            serde_json::json!({
                "block": "about you",
                "text": "Prefers short answers.",
                "scope": "core",
                "sources": [format!("#{source_handle}"), "#zzzzzzz"],
                "annotation": "said so directly",
            }),
        );
        turn(&core, "Understood?", Some(space.clone()));

        let agent = agent_id(&core, &space, MODEL);
        let held = blocks(&core, &agent);
        let sources = &held[0].revisions[0].sources;
        assert_eq!(sources.len(), 1, "only the known handle: {sources:#?}");
        assert_eq!(sources[0].ordinal, 1, "provenance starts at ordinal 1");
        assert_eq!(sources[0].action_id, source_action);
        assert_eq!(sources[0].annotation.as_deref(), Some("said so directly"));

        let bodies = mock.chat_bodies();
        let receipt = tool_results(bodies.last().expect("a request")).remove(0);
        assert!(
            receipt.contains("#zzzzzzz") && receipt.contains("not recorded as sources"),
            "an unknown handle is reported, not dropped: {receipt}"
        );
    });
}

/// The per-owner block ceiling refuses as a tool *result* and writes nothing —
/// the refusal is decided before the first write, so the ninth call leaves no
/// action and no block row.
#[test]
fn the_block_budget_refuses_with_zero_trace() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        let calls: Vec<(String, String)> = (0..=MAX_MEMORY_BLOCKS)
            .map(|i| {
                (
                    "remember".to_string(),
                    serde_json::json!({ "block": format!("b{i}"), "text": format!("note {i}") })
                        .to_string(),
                )
            })
            .collect();
        *script.lock().unwrap() = calls;

        let first = turn(&core, "Remember everything.", None);
        let space = first.space_id.clone();
        let agent = agent_id(&core, &space, MODEL);

        let held = blocks(&core, &agent);
        assert_eq!(held.len(), MAX_MEMORY_BLOCKS, "the ceiling holds");
        assert!(
            held.iter()
                .all(|b| b.name != format!("b{MAX_MEMORY_BLOCKS}")),
            "the refused block left no row: {:#?}",
            held.iter().map(|b| &b.name).collect::<Vec<_>>()
        );

        let results = tool_results(&mock.chat_bodies()[1]);
        assert_eq!(results.len(), MAX_MEMORY_BLOCKS + 1);
        let refusal = results.last().expect("the ninth result");
        assert!(
            refusal.starts_with("error: you already hold the maximum of 8 memory blocks")
                && refusal.contains("b0"),
            "the refusal names the limit and what is already held: {refusal}"
        );
    });
}

/// The per-block size budget refuses the same way, and likewise writes nothing.
#[test]
fn an_oversized_block_is_refused_with_zero_trace() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        script_remember(
            &script,
            serde_json::json!({
                "block": "everything",
                "text": "x".repeat(MAX_MEMORY_BLOCK_BYTES + 1),
            }),
        );
        let first = turn(&core, "Remember this.", None);
        let space = first.space_id.clone();
        let agent = agent_id(&core, &space, MODEL);

        assert!(
            blocks(&core, &agent).is_empty(),
            "an over-budget block is not stored"
        );
        let refusal = tool_results(&mock.chat_bodies()[1]).remove(0);
        assert!(
            refusal.contains(&format!("at most {MAX_MEMORY_BLOCK_BYTES}")),
            "the refusal states the budget: {refusal}"
        );

        // And the next turn's prompt is unchanged — no empty `<memory>`.
        turn(&core, "Anything else?", Some(space));
        assert!(
            !flat_messages(&mock.chat_bodies().pop().expect("a request"))[0]
                .1
                .contains("<memory>"),
            "nothing was written, so nothing loads"
        );
    });
}

/// Memory is **per participant**. A second agent in the same space reads its
/// own (empty) memory, never the first agent's — the property the whole
/// ownership model exists to guarantee.
#[test]
fn another_participants_memory_is_never_loaded() {
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
        let first = turn(&core, "How do tides work?", None);
        let space = first.space_id.clone();

        // A different model mints a second space-owned agent, which answers
        // the next turn.
        let other_model = "mistral-7b@ext";
        turn_as(&core, "Second opinion?", Some(space.clone()), other_model);

        let body = mock.chat_bodies().pop().expect("the other agent's request");
        assert!(
            flat_messages(&body)
                .iter()
                .all(|(_, c)| !c.contains("<memory>") && !c.contains("prefer terse")),
            "the other agent sees none of it: {:#?}",
            flat_messages(&body)
        );

        let first_agent = agent_id(&core, &space, MODEL);
        let other_agent = agent_id(&core, &space, other_model);
        assert_eq!(blocks(&core, &first_agent).len(), 1);
        assert!(blocks(&core, &other_agent).is_empty());
    });
}

/// A committed block is a durable commit, so it emits `Change::Space` for its
/// residence — one more than the same turn without a write.
#[test]
fn a_committed_memory_block_emits_space() {
    run(|| {
        let script = tool_script();
        let (_mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);
        let mut rx = core.subscribe_changes();

        // Control: a turn with a tool round that writes nothing durable of its
        // own (an unknown tool name is an ordinary result).
        *script.lock().unwrap() = vec![("nope".into(), "{}".into())];
        let first = turn(&core, "How do tides work?", None);
        let space = first.space_id.clone();
        let control = drain(&mut rx)
            .into_iter()
            .filter(|c| matches!(c, Change::Space(s) if *s == space))
            .count();

        script_remember(
            &script,
            serde_json::json!({ "block": "plan", "text": "Ship the parser first." }),
        );
        turn(&core, "Remember the plan.", Some(space.clone()));
        let with_memory = drain(&mut rx)
            .into_iter()
            .filter(|c| matches!(c, Change::Space(s) if *s == space))
            .count();

        assert_eq!(
            with_memory,
            control + 1,
            "the memory write adds exactly one Space emission"
        );
    });
}

/// `remember` is turn-scoped protocol surface — bound to the responding
/// participant — so the process registry refuses the name outright rather than
/// letting a consumer's tool be silently shadowed the moment memory is on.
#[test]
fn registering_the_remember_tool_name_is_refused() {
    struct Impostor;
    impl eidola_app_core::tools::Tool for Impostor {
        fn name(&self) -> &str {
            "remember"
        }
        fn description(&self) -> &str {
            "a consumer's own tool"
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({ "type": "object", "properties": {} })
        }
        fn call<'a>(
            &'a self,
            _arguments: serde_json::Value,
        ) -> eidola_app_core::tools::ToolFuture<'a> {
            Box::pin(async move { Ok(String::new()) })
        }
    }

    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let err = core
            .register_tool(std::sync::Arc::new(Impostor))
            .expect_err("`remember` is reserved");
        assert!(
            format!("{err}").contains("remember"),
            "the refusal names it: {err}"
        );
        assert!(core.registered_tools().is_empty());
    });
}
