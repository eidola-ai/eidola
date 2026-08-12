//! LLM-written branch summaries (task 21's checkpoint 3), on the in-process
//! mock-upstream harness (`chat_harness`). The summarizer is scripted by a
//! `SummaryBehavior` arm that recognizes a summary request by its system
//! prompt, so one mock serves the turns and the chore in one test.
//!
//! What is pinned here:
//!
//! * **Structural fallback** — an unset (or unresolvable) utility model
//!   generates nothing, makes no request, and leaves the map exactly as it was.
//! * **Rendering** — a generated summary appears on its own indented line under
//!   the branch's structural entry, in the trailing map's exact bytes.
//! * **The cache** — keyed on the branch's tip action id: a second pass over an
//!   unchanged branch makes no call, and growth regenerates as a **new
//!   generation** of the same item (the prior summary survives in the Record).
//!   An edit **anywhere** in the branch — including a non-tip post — moves that
//!   key, because the key is read off the resolved tip generations.
//! * **The slice** — an over-cap branch sends its opening *and* its newest
//!   posts, so a long branch's summary describes where it got to and two
//!   refreshes of a growing branch never send the same prompt twice.
//! * **Cost** — a remote (`eidola`) utility model summarizes and spends through
//!   the same machinery the may-decline router uses.
//! * **Degradation** — a failed or unusable generation leaves the structural
//!   entry alone and writes nothing.
//! * **The cache doctrine** — a summary appearing changes only the trailing map
//!   message; every conversation byte above it is untouched.
//! * **Emissions** — a committed summary emits `Change::Space` (the row in
//!   `tests/bus.rs`'s exit-point table).
//!
//! Every test drives the pass through `test_refresh_branch_summaries` — the
//! same function production spawns. The production trigger is debounced by
//! twenty seconds, so the passes a fixture's own posts and turns queue up are
//! still waiting when the test ends: what is asserted here is the pass itself,
//! not the timer.

mod chat_harness;

use chat_harness::{
    ChatBehavior, HUMAN_LABEL, MODEL, MockConfig, MockServer, SUMMARY_PROMPT_HEAD, SummaryBehavior,
    flat_messages, map_entry, map_entry_summarized, thread_map, with_account,
};
use eidola_app_core::changes::Change;
use eidola_app_core::{AppCore, ChatStreamEvent, post_handle};

const SUMMARY: &str = "They dig into spring tides and settle on the sun-moon alignment.";

fn run<F>(f: F)
where
    F: FnOnce() + Send + 'static,
{
    std::thread::spawn(f).join().unwrap();
}

/// One streaming turn, optionally branching at `reply_to`.
fn turn(
    core: &AppCore,
    prompt: &str,
    space: Option<String>,
    reply_to: Option<String>,
) -> eidola_app_core::ChatResult {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
    core.runtime()
        .block_on(core.chat_stream_reply(prompt.to_string(), MODEL.into(), space, reply_to, tx))
        .unwrap_or_else(|e| panic!("turn {prompt:?} failed: {e}"))
}

/// Point the space's utility model at a model reference. `local` engine
/// registration is what makes the mock answer as that engine.
fn use_utility_model(core: &AppCore, mock: &MockServer, space: &str, model: &str) {
    core.test_register_loaded_local_model("local", chat_harness::ROUTER_SLUG, mock.port());
    core.runtime()
        .block_on(core.set_space_router_model(space.to_string(), Some(model.to_string())))
        .expect("set utility model");
}

/// Run one summary pass to completion (production spawns exactly this).
fn summarize(core: &AppCore, space: &str) {
    core.runtime()
        .block_on(core.test_refresh_branch_summaries(space.to_string()))
        .expect("summary pass");
}

/// The chat requests the mock saw that were *summary* calls.
fn summary_calls(mock: &MockServer) -> Vec<serde_json::Value> {
    mock.chat_bodies()
        .into_iter()
        .filter(|b| {
            b["messages"][0]["content"]
                .as_str()
                .is_some_and(|c| c.starts_with(SUMMARY_PROMPT_HEAD))
        })
        .collect()
}

/// Every persisted branch-summary action, oldest first: `(id, text)`.
fn summary_actions(core: &AppCore, space: &str) -> Vec<(String, String)> {
    core.runtime()
        .block_on(core.test_space_actions(space.to_string()))
        .expect("actions")
        .into_iter()
        .filter(|a| a.action_type == "checkpoint")
        .map(|a| {
            let text = a
                .blocks
                .iter()
                .find(|b| b.block_type == "text")
                .and_then(|b| b.text_content.clone())
                .unwrap_or_default();
            (a.id, text)
        })
        .collect()
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
    }
    out
}

/// A branched space:
///
/// ```text
/// u1 → i1 ─┬─ u2                  (a 1-post branch: nothing to summarize)
///          └─ b1 → b2             (a 2-post branch, worth a précis)
/// ```
///
/// The asymmetry is deliberate: exactly one branch is long enough to be
/// summarized, so a pass costs exactly one call.
struct Fixture {
    space: String,
    /// Item id of the 2-post branch's opening post.
    branch_item: String,
    /// Action id of that branch's newest post — where growth attaches.
    branch_tail: String,
    /// The 1-post branch — the post a spine turn answers.
    spine_post: String,
    /// Item id of the fork point (i1).
    fork_item: String,
}

fn branched_space(core: &AppCore) -> Fixture {
    let first = turn(core, "How do tides work?", None, None);
    let space = first.space_id.clone();
    let i1 = first.response_action_id.clone().expect("an answer");

    // A bare post off i1: one post, no answer.
    let spine_post = core
        .runtime()
        .block_on(core.post_reply(
            "And why two per day?".into(),
            Some(space.clone()),
            Some(i1.clone()),
        ))
        .expect("post")
        .action_id;

    // The other branch: an ask off i1 and the answer it drew.
    let branch = turn(
        core,
        "What about spring tides?",
        Some(space.clone()),
        Some(i1.clone()),
    );
    let branch_tail = branch.response_action_id.clone().expect("an answer");

    Fixture {
        branch_item: item_id_of(core, &space, "What about spring tides?"),
        fork_item: item_of_action(core, &space, &i1),
        branch_tail,
        spine_post,
        space,
    }
}

/// The item id of an action already in the tree.
fn item_of_action(core: &AppCore, space: &str, action_id: &str) -> String {
    core.runtime()
        .block_on(core.get_space_tree(space.to_string()))
        .expect("tree")
        .into_iter()
        .find(|n| n.action_id == action_id)
        .expect("the action is a post in this space")
        .item_id
}

/// The item id of the post whose text is `text`.
fn item_id_of(core: &AppCore, space: &str, text: &str) -> String {
    core.runtime()
        .block_on(core.get_space_tree(space.to_string()))
        .expect("tree")
        .into_iter()
        .find(|n| n.blocks.iter().any(|b| b.text.as_deref() == Some(text)))
        .unwrap_or_else(|| panic!("no post with text {text:?}"))
        .item_id
}

/// Answer an already-persisted post (no new post of our own) — the comparison
/// that isolates what a turn's request bytes depend on.
fn respond(core: &AppCore, space: &str, target: &str) {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<ChatStreamEvent>();
    core.runtime()
        .block_on(core.respond_stream(space.to_string(), MODEL.into(), target.to_string(), tx))
        .expect("respond");
}

/// A branch grown past the summarizer's slice: the 2-post branch of
/// [`branched_space`] plus `extra` human follow-ups. Returns the fixture and
/// the branch's posts as `(author, text)`, oldest first.
fn over_cap_branch(core: &AppCore, extra: usize) -> (Fixture, Vec<(String, String)>) {
    let fx = branched_space(core);
    let mut posts = vec![
        (
            HUMAN_LABEL.to_string(),
            "What about spring tides?".to_string(),
        ),
        (
            chat_harness::DEFAULT_AGENT_LABEL.to_string(),
            chat_harness::STREAM_CONTENT.to_string(),
        ),
    ];
    let mut tail = fx.branch_tail.clone();
    for i in 1..=extra {
        let text = format!("follow-up {i}");
        tail = core
            .runtime()
            .block_on(core.post_reply(text.clone(), Some(fx.space.clone()), Some(tail)))
            .expect("post")
            .action_id;
        posts.push((HUMAN_LABEL.to_string(), text));
    }
    (fx, posts)
}

/// The action id of the post whose text is `text`.
fn branch_tail_action(core: &AppCore, space: &str, text: &str) -> String {
    core.runtime()
        .block_on(core.get_space_tree(space.to_string()))
        .expect("tree")
        .into_iter()
        .find(|n| n.blocks.iter().any(|b| b.text.as_deref() == Some(text)))
        .unwrap_or_else(|| panic!("no post with text {text:?}"))
        .action_id
}

/// The trailing map of the most recent turn request.
fn last_map(mock: &MockServer) -> String {
    flat_messages(mock.chat_bodies().last().expect("a request"))
        .last()
        .expect("a message")
        .1
        .clone()
}

// ===========================================================================
// Structural fallback
// ===========================================================================

/// No utility model (the default) ⇒ no generation, no request, and the map is
/// exactly the structural one. The whole feature is opt-in.
#[test]
fn without_a_utility_model_the_map_stays_structural() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);

        summarize(&core, &fx.space);
        assert!(
            summary_calls(&mock).is_empty(),
            "summaries are off by default"
        );
        assert!(summary_actions(&core, &fx.space).is_empty());

        respond(&core, &fx.space, &fx.spine_post);
        let map = last_map(&mock);
        assert!(map.contains(&map_entry(
            &fx.branch_item,
            HUMAN_LABEL,
            "2 posts",
            "just now",
            // The branch's answer is this responder's own post.
            Some("1 post"),
            "What about spring tides?",
        )));
        assert!(
            !map.contains(SUMMARY),
            "nothing was generated, so nothing is rendered: {map}"
        );
    });
}

/// An unresolvable utility model degrades exactly like an unset one — a
/// diagnostic, no request, no rows.
#[test]
fn an_unresolvable_utility_model_generates_nothing() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);
        // Disabling the backend the utility model names makes it unresolvable
        // *after* it was configured — the honest shape of a broken setting.
        core.runtime()
            .block_on(core.set_backend_enabled("local".into(), false))
            .expect("disable local");

        summarize(&core, &fx.space);
        assert!(summary_calls(&mock).is_empty());
        assert!(summary_actions(&core, &fx.space).is_empty());
    });
}

// ===========================================================================
// Generation and rendering
// ===========================================================================

/// The generated précis renders on its own indented line under the branch's
/// structural entry — which is unchanged. Pinned as exact map bytes.
#[test]
fn a_generated_summary_renders_under_the_structural_entry() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        summarize(&core, &fx.space);
        assert_eq!(
            summary_calls(&mock).len(),
            1,
            "one call: the 2-post branch. The 1-post spine branch is not worth a précis"
        );

        respond(&core, &fx.space, &fx.spine_post);
        let answered = item_of_action(&core, &fx.space, &fx.spine_post);

        assert_eq!(
            last_map(&mock),
            thread_map(
                &[(
                    format!("at #{}", post_handle(&fx.fork_item)),
                    vec![map_entry_summarized(
                        &fx.branch_item,
                        HUMAN_LABEL,
                        "2 posts",
                        "just now",
                        Some("1 post"),
                        "What about spring tides?",
                        SUMMARY,
                    )],
                )],
                &post_handle(&answered),
            ),
        );
    });
}

/// The prompt is the branch, not the space: only the branch's own posts are
/// sent to the summarizer.
#[test]
fn the_summarizer_reads_only_its_branch() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);
        summarize(&core, &fx.space);

        let call = summary_calls(&mock).remove(0);
        let prompt = call["messages"][1]["content"]
            .as_str()
            .expect("user message");
        assert!(prompt.contains("What about spring tides?"), "{prompt}");
        assert!(
            !prompt.contains("How do tides work?"),
            "the trunk is not the branch: {prompt}"
        );
    });
}

/// The summarizer **reads** the branch, so its posts arrive in the one
/// model-facing rendering (`render_post_for_model`) — quoted passages
/// attributed and spliced in at their markers — rather than as the literal
/// `{{ embed N }}` the map's own opening line elides. A branch whose point is
/// the passage it quotes would otherwise be summarized as though the passage
/// were not there, and that summary is written down and read for as long as the
/// branch lives.
#[test]
fn the_summarizer_reads_quoted_passages_not_their_markers() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        // Grow the summarized branch with a post that is nothing but a quote of
        // the trunk's opening and a bare follow-up: elided, it says nothing.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(fx.space.clone()))
            .expect("tree");
        let trunk = tree
            .iter()
            .find(|n| {
                n.blocks
                    .iter()
                    .any(|b| b.text.as_deref() == Some("How do tides work?"))
            })
            .expect("the trunk's opening post");
        let quoting = core
            .runtime()
            .block_on(core.post_with_references(
                "{{ embed 1 }}\n\nStill true?".into(),
                Some(fx.space.clone()),
                Some(fx.branch_tail.clone()),
                vec![eidola_app_core::ReferenceSpec {
                    antecedent_action_id: trunk.action_id.clone(),
                    content_block_id: Some(trunk.blocks[0].id.clone()),
                    range_start: Some(7),
                    range_end: Some(12), // "tides"
                    annotation: None,
                }],
            ))
            .expect("post with a reference");
        core.runtime()
            .block_on(core.post_reply(
                "Ask the tide agent.".into(),
                Some(fx.space.clone()),
                Some(quoting.action_id.clone()),
            ))
            .expect("grow the branch");

        summarize(&core, &fx.space);
        let prompts: Vec<String> = summary_calls(&mock)
            .iter()
            .map(|c| c["messages"][1]["content"].as_str().unwrap().to_string())
            .collect();
        let quoted = format!(
            "[1] #{} · {HUMAN_LABEL}\n> tides",
            post_handle(&trunk.item_id)
        );
        assert!(
            prompts.iter().any(|p| p.contains(&quoted)),
            "the passage reaches the summarizer attributed; got {prompts:?}"
        );
        assert!(
            prompts.iter().all(|p| !p.contains("{{ embed")),
            "no literal marker survives; got {prompts:?}"
        );
    });
}

// ===========================================================================
// The cache: keyed on the branch tip
// ===========================================================================

/// A second pass over an unchanged branch makes no call at all; growth
/// regenerates, and the regeneration is a **new generation** of the same item —
/// the prior summary is superseded, not overwritten.
#[test]
fn an_unchanged_branch_is_not_resummarized_and_growth_supersedes() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        summarize(&core, &fx.space);
        summarize(&core, &fx.space);
        assert_eq!(
            summary_calls(&mock).len(),
            1,
            "the cache is keyed on the branch tip: nothing grew, nothing regenerates"
        );
        let first = summary_actions(&core, &fx.space);
        assert_eq!(first.len(), 1);

        // Grow the branch: a new tip ⇒ a new cache key.
        turn(
            &core,
            "Go on.",
            Some(fx.space.clone()),
            Some(fx.branch_tail.clone()),
        );

        summarize(&core, &fx.space);
        assert_eq!(summary_calls(&mock).len(), 2, "growth regenerates once");

        let both = summary_actions(&core, &fx.space);
        assert_eq!(
            both.len(),
            2,
            "append-only: the prior summary is still there"
        );
        assert_eq!(both[0].0, first[0].0, "the first generation is untouched");

        // The map renders the current generation only.
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(fx.space.clone()))
            .expect("tree");
        assert!(
            tree.iter().all(|n| n.action_type != "checkpoint"),
            "a summary is not a post: it collapses out of the render"
        );
    });
}

/// A branch longer than the slice sends its **opening and its newest posts**,
/// with the elided middle stated. Pinned as exact prompt bytes: an oldest-N
/// slice would describe the branch's opening forever, however far it moved.
#[test]
fn an_over_cap_branch_keeps_its_opening_and_its_newest_posts() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let (fx, posts) = over_cap_branch(&core, 13);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        summarize(&core, &fx.space);
        let call = summary_calls(&mock).remove(0);
        let prompt = call["messages"][1]["content"]
            .as_str()
            .expect("user message");

        // 15 posts, a 4-post opening and the 8 newest: 3 elided in between.
        let mut expected = String::from("BRANCH (oldest first):\n");
        for (author, text) in &posts[..4] {
            expected.push_str(&format!("{author}: {text}\n"));
        }
        expected.push_str("(… 3 posts omitted …)\n");
        for (author, text) in &posts[posts.len() - 8..] {
            expected.push_str(&format!("{author}: {text}\n"));
        }
        expected.push_str("\nSummarize this branch in one or two sentences.");
        assert_eq!(prompt, expected);
    });
}

/// Two refreshes of a *growing* over-cap branch send different prompts. The
/// billing-thoughtfulness half of the same bug: a fixed oldest-N slice would
/// re-send identical bytes on every growth.
#[test]
fn successive_refreshes_of_a_growing_over_cap_branch_differ() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let (fx, _) = over_cap_branch(&core, 13);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);
        summarize(&core, &fx.space);

        let tail = branch_tail_action(&core, &fx.space, "follow-up 13");
        core.runtime()
            .block_on(core.post_reply("follow-up 14".into(), Some(fx.space.clone()), Some(tail)))
            .expect("post");
        summarize(&core, &fx.space);

        let calls = summary_calls(&mock);
        assert_eq!(calls.len(), 2, "growth regenerates");
        let first = calls[0]["messages"][1]["content"].as_str().expect("prompt");
        let second = calls[1]["messages"][1]["content"].as_str().expect("prompt");
        assert_ne!(first, second, "a grown branch is not the same prompt");
        assert!(
            second.contains("follow-up 14") && !first.contains("follow-up 14"),
            "the newest post is what changed"
        );
    });
}

// ===========================================================================
// Edits
// ===========================================================================

/// Editing a post **anywhere** in a branch — here its opening, not its tip —
/// moves the cache key, so the next pass regenerates against the edited text.
/// The key is read off the branch's resolved tip generations, and an edit is a
/// new generation, so a non-tip edit is not invisible to it.
#[test]
fn editing_a_non_tip_branch_post_invalidates_the_cache() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);
        summarize(&core, &fx.space);
        assert_eq!(summary_calls(&mock).len(), 1);

        // The branch's opening post — its *root*, not its tip.
        let opening = branch_tail_action(&core, &fx.space, "What about spring tides?");
        core.runtime()
            .block_on(core.edit_post(opening, "What about neap tides?".into()))
            .expect("edit");

        summarize(&core, &fx.space);
        let calls = summary_calls(&mock);
        assert_eq!(calls.len(), 2, "the edit moved the branch's cache key");
        let prompt = calls[1]["messages"][1]["content"].as_str().expect("prompt");
        assert!(prompt.contains("What about neap tides?"), "{prompt}");
        assert!(!prompt.contains("What about spring tides?"), "{prompt}");
    });
}

/// An edit **schedules** a pass, like a post or a turn does. Without the hook
/// an edited branch keeps its stale summary until some unrelated later write
/// happens to trigger a refresh.
#[test]
fn every_write_that_moves_a_branch_schedules_a_pass() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        let opening = branch_tail_action(&core, &fx.space, "What about spring tides?");
        for (what, write) in [
            (
                "post",
                Box::new(|| {
                    core.runtime()
                        .block_on(core.post_reply(
                            "another".into(),
                            Some(fx.space.clone()),
                            Some(fx.branch_tail.clone()),
                        ))
                        .expect("post");
                }) as Box<dyn Fn()>,
            ),
            (
                "turn",
                Box::new(|| respond(&core, &fx.space, &fx.spine_post)),
            ),
            (
                "edit",
                Box::new(|| {
                    core.runtime()
                        .block_on(core.edit_post(opening.clone(), "edited".into()))
                        .expect("edit");
                }),
            ),
        ] {
            core.test_take_summary_triggers();
            write();
            assert_eq!(
                core.test_take_summary_triggers(),
                vec![fx.space.clone()],
                "a {what} must schedule a summary pass"
            );
        }
    });
}

// ===========================================================================
// Cost
// ===========================================================================

/// A remote utility model summarizes through the same chore runner the router
/// uses: it spends a real credential and settles its refund. Configured
/// behavior, bounded by the tip cache rather than by a policy gate.
#[test]
fn a_remote_utility_model_summarizes_and_spends() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        core.runtime()
            .block_on(core.set_space_router_model(
                fx.space.clone(),
                Some(chat_harness::ROUTER_REMOTE_MODEL.into()),
            ))
            .expect("set remote utility model");

        summarize(&core, &fx.space);

        let calls = summary_calls(&mock);
        assert_eq!(calls.len(), 1, "the remote model is used like any other");
        assert_eq!(calls[0]["model"], chat_harness::ROUTER_REMOTE_MODEL);
        assert_eq!(summary_actions(&core, &fx.space).len(), 1);

        let auths = mock.chat_auth_values();
        assert!(
            auths.last().expect("an auth slot").is_some(),
            "a remote summary call carries a spend token"
        );
        let lifecycle = core
            .runtime()
            .block_on(core.wallet_lifecycle())
            .expect("wallet lifecycle");
        assert!(
            !lifecycle.iter().any(|c| c.state == "spending"),
            "the hold is settled, not stranded: {lifecycle:?}"
        );
    });
}

// ===========================================================================
// Degradation
// ===========================================================================

/// A refusing summarizer, and one that answers with nothing usable, both leave
/// the structural entry alone and write nothing.
#[test]
fn a_failed_or_unusable_generation_degrades_silently() {
    for behavior in [
        SummaryBehavior::Fail(500),
        SummaryBehavior::Reply(String::new()),
    ] {
        run(move || {
            let (mock, core, _dir) = chat_harness::core_for(MockConfig {
                chat: ChatBehavior::OkStreaming,
                summary: behavior,
                ..MockConfig::default()
            });
            with_account(&core);
            let fx = branched_space(&core);
            use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

            summarize(&core, &fx.space);
            assert_eq!(summary_calls(&mock).len(), 1, "it was attempted");
            assert!(
                summary_actions(&core, &fx.space).is_empty(),
                "and nothing was written"
            );

            respond(&core, &fx.space, &fx.spine_post);
            assert!(last_map(&mock).contains(&map_entry(
                &fx.branch_item,
                HUMAN_LABEL,
                "2 posts",
                "just now",
                Some("1 post"),
                "What about spring tides?",
            )));
        });
    }
}

// ===========================================================================
// Cache doctrine + emissions
// ===========================================================================

/// A summary appearing changes **only** the trailing map message. Every
/// conversation byte above it — the system message included — is identical
/// before and after, so no upstream prefix cache is disturbed by a chore.
#[test]
fn a_summary_changes_only_the_map() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        // The same turn — the same post answered twice — before and after the
        // branch has a summary.
        respond(&core, &fx.space, &fx.spine_post);
        let before = flat_messages(mock.chat_bodies().last().expect("a request"));

        summarize(&core, &fx.space);
        respond(&core, &fx.space, &fx.spine_post);
        let after = flat_messages(mock.chat_bodies().last().expect("a request"));

        assert_eq!(
            &before[..before.len() - 1],
            &after[..after.len() - 1],
            "the conversation above the map is byte-identical"
        );
        assert!(!before.last().expect("a map").1.contains(SUMMARY));
        assert!(after.last().expect("a map").1.contains(SUMMARY));
    });
}

/// A summary is invisible where posts are counted: the Library listing's
/// message count and last-activity are unmoved by one.
#[test]
fn a_summary_does_not_show_up_in_the_library_listing() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        let before = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");
        summarize(&core, &fx.space);
        assert_eq!(summary_actions(&core, &fx.space).len(), 1);
        let after = core
            .runtime()
            .block_on(core.list_spaces(false))
            .expect("spaces");

        assert_eq!(before[0].message_count, after[0].message_count);
        assert_eq!(before[0].last_activity_at, after[0].last_activity_at);
    });
}

/// The durable commit's emission (the `tests/bus.rs` exit-point row): a
/// committed summary emits `Change::Space` and nothing else — nothing was
/// posted, so the library listing is untouched.
#[test]
fn a_committed_summary_emits_space() {
    run(|| {
        let (mock, core, _dir) = chat_harness::core_for(MockConfig {
            chat: ChatBehavior::OkStreaming,
            summary: SummaryBehavior::Reply(SUMMARY.into()),
            ..MockConfig::default()
        });
        with_account(&core);
        let fx = branched_space(&core);
        use_utility_model(&core, &mock, &fx.space, chat_harness::ROUTER_MODEL);

        let mut rx = core.subscribe_changes();
        summarize(&core, &fx.space);
        assert_eq!(
            drain(&mut rx),
            vec![Change::Space(fx.space.clone())],
            "one durable commit, one Space emission — no SpaceIndex, no Record"
        );

        // A pass that commits nothing emits nothing.
        let mut rx = core.subscribe_changes();
        summarize(&core, &fx.space);
        assert!(drain(&mut rx).is_empty(), "a cache hit is silent");
    });
}
