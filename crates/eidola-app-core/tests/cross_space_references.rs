//! Cross-space references (task 37, wave 1: the permission model) on the
//! in-process mock-upstream harness (`chat_harness`).
//!
//! The rule being pinned is one sentence — *you may quote what you can read;
//! quoting **copies** the excerpt to the new audience; the link itself is not a
//! capability* — and it decomposes into four checks, each of which has a test
//! here:
//!
//! 1. **Create** — only a participant of the referenced space may quote it.
//!    The shared "You" is referenced into every space the default template
//!    instantiates, so the single-user case is untouched; the one space it is
//!    genuinely not in is an agent's private notebook, and that is exactly what
//!    the refusal protects.
//! 2. **Copy-semantics** — the quoted passage rides along like a paste. A
//!    participant who could not follow the link still reads the excerpt,
//!    expanded into its upstream context, forever.
//! 3. **Existence is public within the referencing space** — which is why a
//!    denial may confirm that a quote came from somewhere.
//! 4. **Follow requires membership** — re-checked per tool call, which is what
//!    makes the blocked → grant → retry loop work with no special machinery.
//!
//! Plus the sharpening that constrains every string in the feature: **a denial
//! leaks nothing** — not a title, not a participant, not a byte of content of
//! the space it refused.

mod chat_harness;

use chat_harness::{ChatBehavior, MockConfig, MockServer, ToolScript, flat_messages, tool_script};
use eidola_app_core::changes::Change;
use eidola_app_core::error::AppError;
use eidola_app_core::tools::FOLLOW_DENIED;
use eidola_app_core::{AppCore, PostResult, ReferenceSpec, post_handle};

/// The external backend's model: the turn-scoped tools ride a learned
/// per-`(backend, wire_model)` capability that excludes no backend kind; these
/// tests run over an `openai` backend to keep the credential spend out of the
/// way.
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

fn post(core: &AppCore, text: &str, space: Option<String>) -> PostResult {
    core.runtime()
        .block_on(core.post(text.to_string(), space))
        .expect("post")
}

fn reply(core: &AppCore, text: &str, space: &str, reply_to: &str) -> PostResult {
    core.runtime()
        .block_on(core.post_reply(
            text.to_string(),
            Some(space.to_string()),
            Some(reply_to.to_string()),
        ))
        .expect("reply")
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

/// Any agent participant of a space — the source space never took a turn, so
/// its agent is the one the default template seeded rather than one minted for
/// [`MODEL`].
fn any_agent_id(core: &AppCore, space: &str) -> String {
    core.runtime()
        .block_on(core.list_space_participants(space.to_string()))
        .expect("participants")
        .into_iter()
        .find(|p| p.kind == "agent")
        .expect("an agent")
        .id
}

/// Quote the substring `passage` of `action`'s first block (byte offsets are
/// the storage form; naming the passage is what the test is about).
fn quote_of(core: &AppCore, space: &str, action_id: &str, passage: &str) -> ReferenceSpec {
    let node = core
        .runtime()
        .block_on(core.get_space_tree(space.to_string()))
        .expect("tree")
        .into_iter()
        .find(|n| n.action_id == action_id)
        .expect("the post in its space's tree");
    let block = node.blocks.first().expect("a content block");
    let text = block.text.as_deref().expect("text");
    let start = text.find(passage).expect("the passage is in the post") as i64;
    ReferenceSpec {
        antecedent_action_id: action_id.to_string(),
        content_block_id: Some(block.id.clone()),
        range_start: Some(start),
        range_end: Some(start + passage.len() as i64),
        annotation: None,
    }
}

fn drain(rx: &mut tokio::sync::broadcast::Receiver<Change>) -> Vec<Change> {
    let mut out = Vec::new();
    while let Ok(c) = rx.try_recv() {
        out.push(c);
    }
    out
}

/// The result of the last round's tool call.
fn last_tool_result(mock: &MockServer) -> String {
    let bodies = mock.chat_bodies();
    let followup = bodies.last().expect("a follow-up round");
    flat_messages(followup)
        .into_iter()
        .rfind(|(role, _)| role == "tool")
        .expect("a tool result")
        .1
}

/// Ask the model to follow quote `ordinal` of the post at `handle` on the next
/// turn.
fn script_follow(script: &ToolScript, handle: &str, ordinal: i64) {
    *script.lock().unwrap() = vec![(
        "read_post".into(),
        serde_json::json!({ "handle": handle, "quote": ordinal }).to_string(),
    )];
}

// ===========================================================================
// Rule 1 — create
// ===========================================================================

/// The single-user common case, which the create gate must not touch: the
/// shared "You" is a referenced global in every space the default template
/// instantiates, so quoting across your own conversations just works.
#[test]
fn the_shared_you_may_quote_across_its_own_conversations() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());

        let source = post(&core, "Tides come from the moon's pull.", None);
        let elsewhere = core
            .runtime()
            .block_on(core.create_space(Some("Elsewhere".into())))
            .expect("a space")
            .id;

        let spec = quote_of(&core, &source.space_id, &source.action_id, "moon's pull");
        let quoting = core
            .runtime()
            .block_on(core.post_with_references(
                "Is this right?\n\n{{ embed 1 }}".into(),
                Some(elsewhere.clone()),
                None,
                vec![spec],
            ))
            .expect("a cross-space quote by a member of both");

        let node = core
            .runtime()
            .block_on(core.get_space_tree(elsewhere))
            .expect("tree")
            .into_iter()
            .find(|n| n.action_id == quoting.action_id)
            .expect("the quoting post");
        assert_eq!(node.references.len(), 1);
        assert_eq!(node.references[0].snippet.as_deref(), Some("moon's pull"));
    });
}

/// Rule 1's teeth. An agent's notebook (task 36) is the one space the shared
/// "You" is not a member of — promotion gives the *agent* the membership and
/// nobody else — so it is the honest fixture for "a conversation you are not
/// in". The refusal is typed, leaves zero durable trace, and names nothing
/// about the space it refused.
#[test]
fn quoting_a_conversation_you_are_not_in_is_refused_with_zero_trace() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());

        // A promoted agent, and something written in its private notebook.
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);
        let notebook = core
            .runtime()
            .block_on(core.promote_participant(agent.clone(), None, None))
            .expect("promotion")
            .notebook_space_id;
        let private = post(&core, "A private note to self.", Some(notebook.clone()));

        let elsewhere = core
            .runtime()
            .block_on(core.create_space(Some("Elsewhere".into())))
            .expect("a space")
            .id;

        let spec = quote_of(&core, &notebook, &private.action_id, "private note");
        let mut rx = core.subscribe_changes();
        let err = core
            .runtime()
            .block_on(core.post_with_references(
                "Look what it wrote.\n\n{{ embed 1 }}".into(),
                Some(elsewhere.clone()),
                None,
                vec![spec],
            ))
            .expect_err("a non-member may not quote");

        match &err {
            AppError::NotAParticipant {
                participant_id,
                action_id,
            } => {
                assert_eq!(participant_id, eidola_app_core::HUMAN_PARTICIPANT_ID);
                assert_eq!(action_id, &private.action_id);
            }
            other => panic!("expected NotAParticipant, got {other:?}"),
        }

        // Non-leaking: the refusal may confirm the action it was handed (the
        // caller named it) and nothing else about where it lives.
        let rendered = err.to_string();
        for secret in [notebook.as_str(), "A private note to self.", "private note"] {
            assert!(
                !rendered.contains(secret),
                "denial leaked {secret:?}: {rendered}"
            );
        }

        // Zero trace: validation runs before any write, so the refused post
        // never existed and nothing was emitted.
        let actions = core
            .runtime()
            .block_on(core.test_space_actions(elsewhere))
            .expect("actions");
        assert!(actions.is_empty(), "a refused post must leave no trace");
        assert!(drain(&mut rx).is_empty(), "a refused post must not emit");
    });
}

// ===========================================================================
// Rules 2–4 — copy-semantics, and the blocked → grant → retry loop
// ===========================================================================

/// Build the canonical scenario: a source conversation, and a second one whose
/// agent is *not* a member of it but whose thread quotes it.
///
/// The second space is branched deliberately: the navigation tools (and so the
/// follow affordance) attach only where there is a map to descend, which is
/// unchanged in this wave. See the AGENTS.md note — widening that gate to "this
/// space contains a quote worth following" is the GUI wave's call, since it
/// changes the wire bytes of spaces that send none today.
struct Scenario {
    source_space: String,
    source_post: String,
    quoted_text: &'static str,
    space: String,
    quoting_handle: String,
    agent: String,
}

const QUOTED: &str = "spring tides at syzygy";

fn scenario(core: &AppCore) -> Scenario {
    let source = post(
        core,
        "The sun and moon align to make spring tides at syzygy, twice a month.",
        None,
    );
    core.runtime()
        .block_on(core.rename_space(source.space_id.clone(), "Tides".into()))
        .expect("rename");

    // A second conversation, branched so the navigation tools attach.
    let opening = turn(core, "What should we read about the sea?", None);
    let space = opening.space_id.clone();
    let first_post = core
        .runtime()
        .block_on(core.get_space_tree(space.clone()))
        .expect("tree")[0]
        .action_id
        .clone();
    reply(core, "Another angle entirely.", &space, &first_post);

    let spec = quote_of(core, &source.space_id, &source.action_id, QUOTED);
    let quoting = core
        .runtime()
        .block_on(core.post_with_references(
            "Someone said this:\n\n{{ embed 1 }}\n\nIs it right?".into(),
            Some(space.clone()),
            Some(first_post),
            vec![spec],
        ))
        .expect("the human quotes a space it belongs to");

    Scenario {
        source_space: source.space_id,
        source_post: source.action_id,
        quoted_text: QUOTED,
        quoting_handle: post_handle(&quoting.item_id),
        agent: agent_id(core, &space),
        space,
    }
}

/// Rule 2. Copy-semantics is the whole privacy model: the excerpt was *copied*
/// into this conversation, so everyone here reads it — including a participant
/// that cannot follow the link back. Nothing about the permission model changed
/// that, and this pins it — now with the attribution the passage travels under
/// (task 63): a bare blockquote inside someone else's post reads as that
/// someone's own words.
#[test]
fn a_quoted_passage_reaches_participants_who_cannot_follow_it() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let s = scenario(&core);

        assert!(
            !core
                .runtime()
                .block_on(core.list_space_participants(s.source_space.clone()))
                .expect("participants")
                .iter()
                .any(|p| p.id == s.agent),
            "the answering agent must not be a member of the source space"
        );

        turn(&core, "So — is it right?", Some(s.space.clone()));

        let body = mock.chat_bodies().pop().expect("a request");
        let sent = flat_messages(&body)
            .into_iter()
            .map(|(_, c)| c)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            sent.contains(&format!(
                "[1] You (a post outside this space, or an earlier version)\n> {}",
                s.quoted_text
            )),
            "the quoted passage rides in the context attributed to its author, and says it \
             came from elsewhere — this space cannot address it by handle: {sent}"
        );
    });
}

/// The subtle half of attribution. A per-space override is that space's name
/// for a participant, so a passage quoted **out of** another space must carry
/// the name it was written under — not the reading space's name for the same
/// participant. Both are the shared "You" here, renamed on each side.
#[test]
fn a_cross_space_passage_is_attributed_by_the_space_it_was_written_in() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let s = scenario(&core);

        let rename = |space: &str, label: &str| {
            core.runtime()
                .block_on(core.set_space_participant_override(
                    space.to_string(),
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                    eidola_app_core::ParticipantOverride {
                        label: Some(Some(label.to_string())),
                        model_ref: None,
                        system_prompt: None,
                        notify_policy: None,
                    },
                ))
                .expect("override");
        };
        rename(&s.source_space, "Tide Watcher");
        rename(&s.space, "Skipper");

        turn(&core, "So — is it right?", Some(s.space.clone()));

        let body = mock.chat_bodies().pop().expect("a request");
        let sent = flat_messages(&body)
            .into_iter()
            .map(|(_, c)| c)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            sent.contains(&format!(
                "[1] Tide Watcher (a post outside this space, or an earlier version)\n> {}",
                s.quoted_text
            )),
            "the byline is the source space's name for the author: {sent}"
        );
        assert!(
            sent.contains("· Skipper\n\nSomeone said this:"),
            "while this space's own header keeps this space's name for them: {sent}"
        );
        assert!(
            !sent.contains("Skipper (a post outside"),
            "the reading space's override must never reach a passage from elsewhere: {sent}"
        );
    });
}

/// Rules 3 and 4, and the sharpening — the canonical loop. A human quotes space
/// A into space B; B's agent tries to follow and is refused in terms it can
/// narrate and that give away nothing; the human grants membership; the very
/// next call resolves. No retry machinery exists — the tool simply re-reads
/// membership.
#[test]
fn an_agent_follows_a_quote_only_once_it_is_granted_membership() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let s = scenario(&core);

        // --- blocked ---------------------------------------------------
        script_follow(&script, &s.quoting_handle, 1);
        turn(&core, "Where is that from?", Some(s.space.clone()));
        let denied = last_tool_result(&mock);
        assert_eq!(denied, FOLLOW_DENIED);
        for secret in [
            s.source_space.as_str(),
            s.source_post.as_str(),
            "Tides",
            "sun and moon align",
            "twice a month",
        ] {
            assert!(
                !denied.contains(secret),
                "the denial leaked {secret:?}: {denied}"
            );
        }

        // --- granted ---------------------------------------------------
        // Through the door the reader actually presses (the inspector's "Invite
        // an agent…"): ordinary membership, which for a space-owned agent means
        // promotion first — anything cross-space implies a shared identity
        // (task 36). Both halves travel in **one** transaction, which also
        // decides *whether* the sharing half is needed at all, so the
        // irreversible one can never land alone or be asked for twice.
        let member = core
            .runtime()
            .block_on(core.grant_space_membership(
                s.source_space.clone(),
                s.agent.clone(),
                eidola_app_core::MembershipRole::Observer,
            ))
            .expect("share and grant");
        assert_eq!(member.scope, "global", "shared on its way in");
        let granted = core
            .runtime()
            .block_on(core.list_space_participants(s.source_space.clone()))
            .expect("participants")
            .into_iter()
            .find(|p| p.id == s.agent)
            .expect("the agent is a member of the source space");
        assert_eq!(granted.role, "observer", "read-only is what was granted");

        // --- retry -----------------------------------------------------
        script_follow(&script, &s.quoting_handle, 1);
        turn(&core, "Try again now.", Some(s.space.clone()));
        let followed = last_tool_result(&mock);
        assert!(
            followed.starts_with("From another conversation you take part in — Tides.\n\n#"),
            "{followed}"
        );
        assert!(
            followed.contains("The sun and moon align to make spring tides at syzygy"),
            "the whole post, not just the excerpt: {followed}"
        );
    });
}

/// **A sibling branch is absent from the turn's context, so a tool result is
/// the model's only view of a post there** — and the author of what that post
/// quotes has to survive the trip. `read_thread` reads a snapshot, and only the
/// source space can name a cross-space author, so the label travels on the
/// reference row rather than being re-derived from the reading space (which has
/// never met that participant, and whose own name for them would misattribute).
#[test]
fn a_sibling_branchs_cross_space_quote_keeps_its_author_through_read_thread() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let s = scenario(&core);

        // The same participant, named differently on each side.
        let rename = |space: &str, label: &str| {
            core.runtime()
                .block_on(core.set_space_participant_override(
                    space.to_string(),
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                    eidola_app_core::ParticipantOverride {
                        label: Some(Some(label.to_string())),
                        model_ref: None,
                        system_prompt: None,
                        notify_policy: None,
                    },
                ))
                .expect("override");
        };
        rename(&s.source_space, "Tide Watcher");
        rename(&s.space, "Skipper");

        // Answer in the *other* branch, so the quoting post is a sibling: it is
        // not in this turn's ancestry and reaches the model only if it asks.
        let other_branch = core
            .runtime()
            .block_on(core.get_space_tree(s.space.clone()))
            .expect("tree")
            .into_iter()
            .find(|n| {
                n.blocks.iter().any(|b| {
                    b.text
                        .as_deref()
                        .is_some_and(|t| t.contains("Another angle entirely"))
                })
            })
            .expect("the other branch")
            .action_id;
        reply(&core, "Let's stay on this one.", &s.space, &other_branch);

        *script.lock().unwrap() = vec![(
            "read_thread".into(),
            serde_json::json!({ "handle": s.quoting_handle }).to_string(),
        )];
        turn(&core, "What else is there?", Some(s.space.clone()));

        // The premise, pinned: the sibling branch's post is nowhere in the
        // context the turn was given.
        let context = mock
            .chat_bodies()
            .iter()
            .flat_map(|b| flat_messages(b).into_iter().map(|(_, c)| c))
            .filter(|c| !c.starts_with("Thread from"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !context.contains(s.quoted_text),
            "a sibling branch must not leak into the context: {context}"
        );

        let read = last_tool_result(&mock);
        assert!(
            read.contains(&format!(
                "[1] Tide Watcher (a post outside this space, or an earlier version)\n> {}",
                s.quoted_text
            )),
            "the only view of that passage keeps the name it was written under: {read}"
        );
        assert!(
            !read.contains("Skipper (a post outside"),
            "and never the reading space's name for the same participant: {read}"
        );
    });
}

/// A followed post is a post. It reaches the model through its own rendering
/// path (`render_followed_post`), and that path renders references the way
/// every other one does: the embedded quote expanded and attributed where the
/// author put it, the un-embedded one footnoted, no literal marker anywhere.
/// Its bylines are addressed from **this** turn's space, so a quote of a post
/// in the followed conversation is named rather than given a handle that would
/// open nothing here.
#[test]
fn a_followed_post_renders_its_own_quotes_like_every_other_post() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let s = scenario(&core);

        // In the source conversation, a post that itself quotes twice: once
        // embedded, once with no marker at all.
        let embedded = quote_of(&core, &s.source_space, &s.source_post, "sun and moon align");
        let orphaned = quote_of(&core, &s.source_space, &s.source_post, "twice a month");
        let quoting_there = core
            .runtime()
            .block_on(core.post_with_references(
                "As I said:\n\n{{ embed 1 }}\n\nAnd there is more.".into(),
                Some(s.source_space.clone()),
                None,
                vec![embedded, orphaned],
            ))
            .expect("a post that quotes in its own space");

        // Here, a post quoting *that* post — the one the model will follow.
        let spec = quote_of(
            &core,
            &s.source_space,
            &quoting_there.action_id,
            "As I said",
        );
        let here = core
            .runtime()
            .block_on(core.post_with_references(
                "Look at this:\n\n{{ embed 1 }}".into(),
                Some(s.space.clone()),
                None,
                vec![spec],
            ))
            .expect("the human quotes a space it belongs to");

        core.runtime()
            .block_on(core.grant_space_membership(
                s.source_space.clone(),
                s.agent.clone(),
                eidola_app_core::MembershipRole::Observer,
            ))
            .expect("share and grant");

        script_follow(&script, &post_handle(&here.item_id), 1);
        turn(&core, "Where is that from?", Some(s.space.clone()));
        let followed = last_tool_result(&mock);

        assert!(
            followed.starts_with("From another conversation you take part in — Tides.\n\n#"),
            "{followed}"
        );
        assert!(
            followed.contains(
                "As I said:\n\n[1] You (a post outside this space, or an earlier version)\n\
                 > sun and moon align\n\nAnd there is more."
            ),
            "the followed post's embedded quote expands in place, attributed: {followed}"
        );
        assert!(
            followed.ends_with(
                "Passages this post quotes:\n\
                 [2] You (a post outside this space, or an earlier version)\n> twice a month"
            ),
            "and the one it never embedded is footnoted: {followed}"
        );
        assert!(
            !followed.contains("{{ embed"),
            "no literal marker survives a follow: {followed}"
        );
    });
}

/// A tool that retires the responding agent from inside its own turn — the
/// interleave the finding is about, made deterministic (the `PromoteMidTurn`
/// idiom from `global_agents.rs`).
struct RetireMidTurn {
    core: std::sync::Weak<AppCore>,
    participant: String,
}

impl eidola_app_core::tools::Tool for RetireMidTurn {
    fn name(&self) -> &str {
        "retire_now"
    }
    fn description(&self) -> &str {
        "Retire this agent, right now."
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {}, "required": []})
    }
    fn call<'a>(&'a self, _a: serde_json::Value) -> eidola_app_core::tools::ToolFuture<'a> {
        Box::pin(async move {
            let core = self.core.upgrade().expect("core outlives the turn");
            core.retire_participant(self.participant.clone())
                .await
                .map_err(|e| {
                    eidola_app_core::tools::ToolError::new(format!("retire failed: {e}"))
                })?;
            Ok("retired".to_string())
        })
    }
}

/// **Retirement ends availability, including mid-turn** (Codex review, PR #279).
///
/// A turn binds its tools to the responding participant when it starts, so a
/// retirement landing between two rounds leaves a live tool holding a retired
/// agent's id. Retirement deliberately leaves the `space_participant` rows
/// standing — the trail still records where the agent worked — so a membership
/// question that asks only about the row goes on answering "member" for an agent
/// the human has just taken out of service, and the next round reads its former
/// conversations. Membership is **live** membership: a live row *and* a live
/// participant, which is the definition `space_participants` already used.
#[test]
fn a_retirement_mid_turn_closes_the_passage_it_had_opened() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let core = std::sync::Arc::new(core);
        let s = scenario(&core);

        // Promote and grant, so the follow is genuinely permitted first.
        core.runtime()
            .block_on(core.promote_participant(s.agent.clone(), None, None))
            .expect("promotion");
        core.runtime()
            .block_on(core.add_global_participant(s.source_space.clone(), s.agent.clone(), None))
            .expect("the grant");
        core.register_tool(std::sync::Arc::new(RetireMidTurn {
            core: std::sync::Arc::downgrade(&core),
            participant: s.agent.clone(),
        }))
        .expect("register");

        // Round 1 retires the agent; round 2 — same turn, same bound
        // participant id — tries to follow the quote it could have followed a
        // moment earlier.
        *script.lock().unwrap() = vec![
            ("retire_now".into(), "{}".into()),
            (
                "read_post".into(),
                serde_json::json!({ "handle": s.quoting_handle, "quote": 1 }).to_string(),
            ),
        ];
        turn(
            &core,
            "Retire yourself, then read that quote.",
            Some(s.space.clone()),
        );

        assert_eq!(
            last_tool_result(&mock),
            FOLLOW_DENIED,
            "a retired agent must not keep reading the spaces it belonged to"
        );
    });
}

/// A quote whose target is in *this* space but no longer current — the
/// generation the reference named was edited away. References name concrete
/// generations and are never remapped, so the honest answer is that generation,
/// labelled as one. Same space, so membership is satisfied by construction.
#[test]
fn following_a_quote_to_a_superseded_generation_says_which_it_is() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());

        let opening = turn(&core, "Tides come from the moon's pull.", None);
        let space = opening.space_id.clone();
        let first_post = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree")[0]
            .action_id
            .clone();
        reply(&core, "Another angle entirely.", &space, &first_post);

        let spec = quote_of(&core, &space, &first_post, "moon's pull");
        let quoting = core
            .runtime()
            .block_on(core.post_with_references(
                "About this:\n\n{{ embed 1 }}".into(),
                Some(space.clone()),
                Some(first_post.clone()),
                vec![spec],
            ))
            .expect("a same-space quote");
        // The quoted generation is superseded, so it is gone from the tree the
        // snapshot is built from — only the reference edge still names it.
        core.runtime()
            .block_on(core.edit_post(first_post, "Tides come from the moon's gravity.".into()))
            .expect("edit");

        script_follow(&script, &post_handle(&quoting.item_id), 1);
        turn(&core, "What did that quote?", Some(space));
        let followed = last_tool_result(&mock);
        assert!(
            followed.starts_with("An earlier version of a post in this conversation.\n\n#"),
            "{followed}"
        );
        assert!(
            followed.contains("Tides come from the moon's pull."),
            "{followed}"
        );
    });
}

/// A quote number the post does not have is a *result*, not a turn failure —
/// the same doctrine as an unknown handle. It says what the post does quote so
/// the model can correct itself.
#[test]
fn following_a_quote_number_that_does_not_exist_is_answered_honestly() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        let s = scenario(&core);

        script_follow(&script, &s.quoting_handle, 7);
        turn(&core, "Follow quote seven.", Some(s.space.clone()));
        let answer = last_tool_result(&mock);
        assert!(answer.contains("has no quote 7"), "{answer}");
        assert!(answer.contains("It quotes: 1."), "{answer}");
    });
}

// ===========================================================================
// Inbound exposure
// ===========================================================================

/// The reverse direction: who quoted *me*. Unfiltered, that announces the
/// existence of a conversation the viewer has no part in; filtered per viewer,
/// "you can see it" and "you can open it" stay the same set. The grant flips it
/// on, exactly as it does for the forward follow.
#[test]
fn inbound_references_are_filtered_to_referrers_the_viewer_takes_part_in() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let s = scenario(&core);
        // The source space's own agent: a member of the quoted space, and not
        // of the quoting one — the viewer the filter exists for.
        let source_agent = any_agent_id(&core, &s.source_space);

        let unfiltered = core
            .runtime()
            .block_on(core.references_to(s.source_post.clone()))
            .expect("reverse index");
        assert_eq!(unfiltered.len(), 1, "one post quotes it");
        assert_eq!(unfiltered[0].space_id, s.space);

        let human = core
            .runtime()
            .block_on(core.references_to_visible_to(
                s.source_post.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            ))
            .expect("filtered");
        assert_eq!(human.len(), 1, "the human takes part in the quoting space");

        let before = core
            .runtime()
            .block_on(core.references_to_visible_to(s.source_post.clone(), source_agent.clone()))
            .expect("filtered");
        assert!(
            before.is_empty(),
            "a viewer that cannot follow the referrer is not told it exists"
        );

        core.runtime()
            .block_on(core.promote_participant(source_agent.clone(), None, None))
            .expect("promotion");
        core.runtime()
            .block_on(core.add_global_participant(s.space.clone(), source_agent.clone(), None))
            .expect("the grant");

        let after = core
            .runtime()
            .block_on(core.references_to_visible_to(s.source_post, source_agent))
            .expect("filtered");
        assert_eq!(after.len(), 1, "membership reveals it, like every follow");
    });
}

// ===========================================================================
// What a reference may name (PR #261 review)
//
// Membership answers *whose* material you may reach. This answers *what kind*
// of material a quote may name at all — and the two are independent: a
// participant of this very space may read every post in it and still must not
// be handed another participant's first-person tool trace, its decision, its
// memory block, or any post's `thinking` block. Those are not transcript, and
// the reference edge was the one place they could be laundered into one.
//
// Both ends are pinned: refused at creation (invalid state unrepresentable),
// and withheld by every read path for an edge that arrives below that gate.
// ===========================================================================

/// A memory block is a real action with real text in this very space, owned by
/// the agent that wrote it. Quoting it would republish it to everyone here.
#[test]
fn a_reference_to_a_non_post_action_is_refused_with_zero_trace() {
    run(|| {
        let script = tool_script();
        let (_mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        let opening = turn(&core, "Tell me about tides.", None);
        let space = opening.space_id.clone();
        let agent = agent_id(&core, &space);

        *script.lock().unwrap() = vec![(
            "remember".into(),
            serde_json::json!({
                "block": "about-this-space",
                "text": "Mike gets impatient when I hedge.",
            })
            .to_string(),
        )];
        turn(&core, "Noted?", Some(space.clone()));
        let memory_action = core
            .runtime()
            .block_on(core.memory_blocks(agent))
            .expect("memory")[0]
            .revisions[0]
            .action_id
            .clone();

        let before = core
            .runtime()
            .block_on(core.test_space_actions(space.clone()))
            .expect("actions")
            .len();
        let mut rx = core.subscribe_changes();
        let err = core
            .runtime()
            .block_on(core.post_with_references(
                "About what you wrote to yourself:".into(),
                Some(space.clone()),
                None,
                vec![ReferenceSpec {
                    antecedent_action_id: memory_action.clone(),
                    content_block_id: None,
                    range_start: None,
                    range_end: None,
                    annotation: None,
                }],
            ))
            .expect_err("a memory block is not a post");
        assert!(
            matches!(&err, AppError::NotConfigured { message }
                if message.contains("is a memory, not a post")),
            "{err:?}"
        );
        assert_eq!(
            core.runtime()
                .block_on(core.test_space_actions(space))
                .expect("actions")
                .len(),
            before,
            "a refused reference must leave no trace"
        );
        assert!(drain(&mut rx).is_empty(), "a refused post must not emit");
    });
}

/// The other axis, and the one reachable through wholly public API: a
/// `thinking` block is a post's block, and `get_space_tree` hands out its id.
/// Both context queries filter reasoning out of the wire by block type; a
/// quote must not be the way back in.
#[test]
fn quoting_a_thinking_block_is_refused() {
    run(|| {
        let (_mock, core, _dir) = chat_harness::core_for(MockConfig::default());
        chat_harness::with_account(&core);

        let res = core
            .runtime()
            .block_on(core.chat("hello".into(), chat_harness::MODEL.into(), None))
            .expect("chat");
        let tree = core
            .runtime()
            .block_on(core.get_space_tree(res.space_id.clone()))
            .expect("tree");
        let inference = tree
            .iter()
            .find(|n| n.action_type == "inference")
            .expect("an inference");
        let thinking = inference
            .blocks
            .iter()
            .find(|b| b.block_type == "thinking")
            .expect("a thinking block");

        let err = core
            .runtime()
            .block_on(core.post_with_references(
                "You wrote:\n\n{{ embed 1 }}".into(),
                Some(res.space_id),
                None,
                vec![ReferenceSpec {
                    antecedent_action_id: inference.action_id.clone(),
                    content_block_id: Some(thinking.id.clone()),
                    range_start: Some(0),
                    range_end: Some(thinking.text.as_deref().unwrap().len() as i64),
                    annotation: None,
                }],
            ))
            .expect_err("reasoning is not quotable");
        assert!(
            matches!(&err, AppError::NotConfigured { message }
                if message.contains("is a thinking block")),
            "{err:?}"
        );
    });
}

/// Belt, for an edge written below the gate. Every read path must withhold the
/// passage on its own account: the upstream expansion (which needs no tool call
/// and no membership — the dangerous one), the rendered reference list that
/// `read_post` prints, and the follow itself.
#[test]
fn an_unvalidated_reference_to_a_non_post_is_withheld_by_every_read() {
    run(|| {
        let script = tool_script();
        let (mock, core, _dir) = setup(script.clone());
        core.set_memory_enabled(true);

        let opening = turn(&core, "Tell me about tides.", None);
        let space = opening.space_id.clone();
        let agent = agent_id(&core, &space);
        let first_post = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree")[0]
            .action_id
            .clone();
        // Branched, so the navigation tools attach and the follow is reachable.
        reply(&core, "Another angle entirely.", &space, &first_post);

        const SECRET: &str = "Mike gets impatient when I hedge.";
        *script.lock().unwrap() = vec![(
            "remember".into(),
            serde_json::json!({ "block": "about-this-space", "text": SECRET }).to_string(),
        )];
        turn(&core, "Noted?", Some(space.clone()));
        let agent_label = core
            .runtime()
            .block_on(core.list_space_participants(space.clone()))
            .expect("participants")
            .into_iter()
            .find(|p| p.id == agent)
            .expect("the agent")
            .label;
        let blocks = core
            .runtime()
            .block_on(core.memory_blocks(agent))
            .expect("memory");
        let memory_action = blocks[0].revisions[0].action_id.clone();
        let memory_item = blocks[0].item_id.clone();

        // A post carrying an embed marker, and — below the gate — the edge the
        // gate would have refused.
        let quoting = post(&core, "About this:\n\n{{ embed 1 }}", Some(space.clone()));
        core.runtime()
            .block_on(core.test_insert_unvalidated_reference(
                quoting.action_id.clone(),
                memory_action,
                1,
            ))
            .expect("the unvalidated edge");

        // (1) The render withholds the passage — the edge still exists.
        let node = core
            .runtime()
            .block_on(core.get_space_tree(space.clone()))
            .expect("tree")
            .into_iter()
            .find(|n| n.action_id == quoting.action_id)
            .expect("the quoting post");
        assert_eq!(node.references.len(), 1, "existence stays public");
        assert_eq!(
            node.references[0].snippet, None,
            "the passage is withheld: {:?}",
            node.references[0].snippet
        );

        // (2) The upstream expansion leaves the marker literal rather than
        // expanding a trace into the next reader's context.
        script_follow(&script, &post_handle(&quoting.item_id), 1);
        turn(&core, "What did that quote?", Some(space));
        // Asserted on the quoting post's own rendered message: the marker must
        // still be literal. (A blanket "SECRET appears nowhere" would be wrong
        // — the agent legitimately reads its *own* memory at the head of its
        // own turn; what must not happen is the quote republishing it as
        // conversation.)
        let quoted_message = mock
            .chat_bodies()
            .iter()
            .flat_map(|b| flat_messages(b).into_iter().map(|(_, c)| c))
            .find(|c| c.contains("About this:"))
            .expect("the quoting post was sent");
        assert!(
            quoted_message.contains("{{ embed 1 }}"),
            "an unquotable edge leaves its marker literal: {quoted_message}"
        );
        assert!(
            !quoted_message.contains(SECRET),
            "a non-post reference must never expand into upstream context: {quoted_message}"
        );
        // The edge is still reported — existence is public — but it is *named*,
        // never addressed: the action is current and lives in this space, yet
        // the snapshot excludes it, so its item handle would resolve to
        // nothing. Only a renderable post earns a handle.
        assert!(
            quoted_message.contains(&format!(
                "Passages this post quotes:\n\
                 [1] {agent_label} (a post outside this space, or an earlier version)"
            )),
            "an unaddressable target is named, not addressed: {quoted_message}"
        );
        assert!(
            !quoted_message.contains(&format!("[1] #{}", post_handle(&memory_item))),
            "no handle for a target read_post cannot return: {quoted_message}"
        );

        // (3) The follow refuses to render it, without saying what it is.
        let followed = last_tool_result(&mock);
        assert_eq!(followed, "That quote does not point at a readable post.");
    });
}

// ===========================================================================
// The grant — promotion and membership as one act
// ===========================================================================

/// The grant is two writes and **one** transaction. A space-owned agent can
/// only join another space by being shared first, and promotion is one-way —
/// so a grant refused after the promotion committed would leave a reader with
/// an irreversible change they never asked for on its own, under a message
/// saying the operation failed. Naming an unknown space is the refusal that
/// makes the point: nothing at all happened.
#[test]
fn a_share_whose_grant_is_refused_shares_nothing() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);

        let mut rx = core.subscribe_changes();
        let err = core
            .runtime()
            .block_on(core.promote_participant(
                agent.clone(),
                None,
                Some(eidola_app_core::SpaceGrant {
                    space_id: "no-such-space".into(),
                    role: eidola_app_core::MembershipRole::Observer,
                }),
            ))
            .expect_err("an unknown space is refused");
        assert!(err.to_string().contains("space not found"), "{err}");
        assert!(drain(&mut rx).is_empty(), "a refusal must not emit");

        let still = core
            .runtime()
            .block_on(core.list_space_participants(home))
            .expect("participants")
            .into_iter()
            .find(|p| p.id == agent)
            .expect("still there");
        assert_eq!(
            still.scope, "space",
            "the irreversible half must not land alone"
        );
        assert_eq!(still.source, "owned");
    });
}

/// A grant naming the promotion's *own* home space asks for a membership the
/// promotion already writes, so it is satisfied rather than refused (a second
/// insert on the same key would fail the whole transaction).
#[test]
fn a_grant_naming_the_home_space_is_satisfied_by_the_promotion() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);

        let outcome = core
            .runtime()
            .block_on(core.promote_participant(
                agent.clone(),
                None,
                Some(eidola_app_core::SpaceGrant {
                    space_id: home.clone(),
                    role: eidola_app_core::MembershipRole::Observer,
                }),
            ))
            .expect("promotion");
        assert_eq!(outcome.granted_space_id, None, "nothing extra was granted");
        let member = core
            .runtime()
            .block_on(core.list_space_participants(home))
            .expect("participants")
            .into_iter()
            .find(|p| p.id == agent)
            .expect("still a member of its home space");
        assert_eq!(member.source, "referenced");
    });
}

/// **The grant decides at the write, not from the picker's snapshot.** The
/// invite form captures whether a candidate is shared when its list lands; a
/// promotion from another window between that read and the confirmation makes
/// the snapshot a lie, and a grant that branches on it asks for a promotion
/// that can only be refused ("already a shared agent") — for a membership that
/// could simply have been added. When the competing promotion granted this very
/// space, the reader is told the operation failed about a state that already
/// holds (Codex review, PR #280).
#[test]
fn a_grant_decides_from_the_row_it_finds_not_from_the_pickers_snapshot() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);
        let elsewhere = core
            .runtime()
            .block_on(core.create_space(Some("Elsewhere".into())))
            .expect("a space")
            .id;

        // The picker read it as space-owned…
        let candidate = core
            .runtime()
            .block_on(core.list_grantable_agents(
                elsewhere.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            ))
            .expect("candidates")
            .into_iter()
            .find(|c| c.id == agent)
            .expect("offered here");
        assert!(!candidate.shared, "the snapshot the form would hold");

        // …and another window shared it before the reader confirmed.
        core.runtime()
            .block_on(core.promote_participant(agent.clone(), None, None))
            .expect("the competing promotion");

        let member = core
            .runtime()
            .block_on(core.grant_space_membership(
                elsewhere.clone(),
                agent.clone(),
                eidola_app_core::MembershipRole::Observer,
            ))
            .expect("the grant lands on the row it finds");
        assert_eq!(member.id, agent);
        assert_eq!(member.role, "observer");
        assert_eq!(
            member.scope, "global",
            "already shared — the grant added the membership and nothing else"
        );
        assert!(
            core.runtime()
                .block_on(core.list_space_participants(elsewhere.clone()))
                .expect("participants")
                .iter()
                .any(|p| p.id == agent),
            "a member here now"
        );

        // And the sharper half: the competing promotion granted **this** space
        // too, so there is nothing left to do — satisfied, not refused (the
        // rule `a_grant_naming_the_home_space_is_satisfied_by_the_promotion`
        // already applies inside the promotion).
        let other = turn(&core, "Another conversation.", None).space_id;
        let second = agent_id(&core, &other);
        let third = core
            .runtime()
            .block_on(core.create_space(Some("Third".into())))
            .expect("a space")
            .id;
        core.runtime()
            .block_on(core.promote_participant(
                second.clone(),
                None,
                Some(eidola_app_core::SpaceGrant {
                    space_id: third.clone(),
                    role: eidola_app_core::MembershipRole::Observer,
                }),
            ))
            .expect("the competing promotion granted the same destination");
        let mut rx = core.subscribe_changes();
        let already = core
            .runtime()
            .block_on(core.grant_space_membership(
                third.clone(),
                second.clone(),
                eidola_app_core::MembershipRole::Observer,
            ))
            .expect("a membership that already holds is satisfied, not refused");
        assert_eq!(already.id, second);
        assert!(
            drain(&mut rx).is_empty(),
            "nothing was written, so nothing was announced"
        );

        // A space-owned candidate still travels as one transaction: the
        // promotion and the membership, or neither.
        let fourth = turn(&core, "A fourth conversation.", None).space_id;
        let owned = agent_id(&core, &fourth);
        let member = core
            .runtime()
            .block_on(core.grant_space_membership(
                elsewhere.clone(),
                owned.clone(),
                eidola_app_core::MembershipRole::Observer,
            ))
            .expect("a space-owned agent is shared on its way in");
        assert_eq!(member.scope, "global", "shared by the grant itself");
        assert_eq!(member.role, "observer");
        assert!(
            core.runtime()
                .block_on(core.list_global_agents())
                .expect("the library")
                .iter()
                .any(|a| a.id == owned),
            "and it is in the shared library now"
        );

        // The refusals that were never about the snapshot are unchanged.
        let err = core
            .runtime()
            .block_on(core.grant_space_membership(
                "no-such-space".into(),
                agent.clone(),
                eidola_app_core::MembershipRole::Observer,
            ))
            .expect_err("an unknown space is refused");
        assert!(err.to_string().contains("space not found"), "{err}");
    });
}

/// An agent that **left** can be invited back — the picker offers it, so the
/// write has to honour the offer.
///
/// A membership is soft-ended (`left_at`), and the row stays on the space's
/// primary key, so an insert-only join struck nothing and the roster read that
/// followed found no member: the reader was told the live agent "has been
/// retired and cannot rejoin a space" — a sentence about the wrong thing, and
/// permanent, since every retry took the same path (Codex review, PR #280).
/// The join is insert-**or-revive**, and the requested role rides the revive.
#[test]
fn an_agent_that_left_can_be_invited_back() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);
        let elsewhere = core
            .runtime()
            .block_on(core.create_space(Some("Elsewhere".into())))
            .expect("a space")
            .id;
        core.runtime()
            .block_on(core.promote_participant(
                agent.clone(),
                None,
                Some(eidola_app_core::SpaceGrant {
                    space_id: elsewhere.clone(),
                    role: eidola_app_core::MembershipRole::Observer,
                }),
            ))
            .expect("share and grant");
        assert!(
            core.runtime()
                .block_on(core.remove_space_participant(elsewhere.clone(), agent.clone()))
                .expect("the removal"),
            "it leaves"
        );

        // The picker offers it again — a departure is not a retirement.
        assert!(
            core.runtime()
                .block_on(core.list_grantable_agents(
                    elsewhere.clone(),
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                ))
                .expect("candidates")
                .iter()
                .any(|c| c.id == agent),
            "an agent that left this space is grantable again"
        );

        let mut rx = core.subscribe_changes();
        let member = core
            .runtime()
            .block_on(core.add_global_participant(
                elsewhere.clone(),
                agent.clone(),
                Some(eidola_app_core::MembershipRole::Observer),
            ))
            .expect("the second grant lands rather than reporting a retirement");
        assert_eq!(member.id, agent);
        assert_eq!(
            member.role, "observer",
            "the requested role rides the revive"
        );
        assert!(
            drain(&mut rx)
                .iter()
                .any(|c| matches!(c, Change::Participants)),
            "a membership that came back is a membership change"
        );

        let roster = core
            .runtime()
            .block_on(core.list_space_participants(elsewhere.clone()))
            .expect("participants");
        let back = roster
            .iter()
            .find(|p| p.id == agent)
            .expect("a member again");
        assert_eq!(back.role, "observer");
        assert_eq!(back.source, "referenced");
        assert_eq!(
            roster.iter().filter(|p| p.id == agent).count(),
            1,
            "one membership, revived — not a second row"
        );

        // Idempotent as ever: adding a live member again changes nothing, and
        // does not rewrite the membership it found.
        let mut rx = core.subscribe_changes();
        core.runtime()
            .block_on(core.add_global_participant(elsewhere.clone(), agent.clone(), None))
            .expect("adding a member again is not an error");
        assert!(
            drain(&mut rx).is_empty(),
            "an idempotent re-add writes nothing, so it says nothing"
        );
    });
}

/// The picker behind the grant obeys the same ACL as everything else: it lists
/// agents a reader could actually add here, and nothing about spaces they take
/// no part in.
#[test]
fn the_grant_picker_offers_only_agents_that_could_join_and_only_ones_you_know_of() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);
        let elsewhere = core
            .runtime()
            .block_on(core.create_space(Some("Elsewhere".into())))
            .expect("a space")
            .id;

        // A space-owned agent of a space the human takes part in: offered
        // *here*, and never in its own space (it is already a member there).
        let offered = core
            .runtime()
            .block_on(core.list_grantable_agents(
                elsewhere.clone(),
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
            ))
            .expect("candidates");
        let row = offered
            .iter()
            .find(|c| c.id == agent)
            .expect("a space-owned agent from elsewhere is offered");
        assert!(!row.shared, "it is not shared yet");
        assert!(row.home_space_title.is_some(), "named by where it works");
        assert!(
            core.runtime()
                .block_on(core.list_grantable_agents(
                    home.clone(),
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                ))
                .expect("candidates")
                .iter()
                .all(|c| c.id != agent),
            "an agent is never offered its own space"
        );

        // Once shared and granted, it drops out of the listing here too — an
        // affordance that could only be a no-op is not offered.
        core.runtime()
            .block_on(core.promote_participant(
                agent.clone(),
                None,
                Some(eidola_app_core::SpaceGrant {
                    space_id: elsewhere.clone(),
                    role: eidola_app_core::MembershipRole::Observer,
                }),
            ))
            .expect("share and grant");
        assert!(
            core.runtime()
                .block_on(core.list_grantable_agents(
                    elsewhere.clone(),
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                ))
                .expect("candidates")
                .iter()
                .all(|c| c.id != agent),
            "a member is not a candidate"
        );

        // Retired: it cannot rejoin a space, so it is not offered either.
        let other = turn(&core, "Another conversation.", None).space_id;
        let other_agent = agent_id(&core, &other);
        core.runtime()
            .block_on(core.promote_participant(other_agent.clone(), None, None))
            .expect("promotion");
        core.runtime()
            .block_on(core.retire_participant(other_agent.clone()))
            .expect("retirement");
        assert!(
            core.runtime()
                .block_on(core.list_grantable_agents(
                    elsewhere.clone(),
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                ))
                .expect("candidates")
                .iter()
                .all(|c| c.id != other_agent),
            "a retired agent cannot rejoin, so it is not offered"
        );

        // And the ACL: the listing is a read like any other. A viewer that
        // takes no part in a space is told nothing about the agents working
        // there — nor, through their home titles, about the conversations
        // themselves. `other`'s seeded agent is the fixture: the viewer below
        // belongs only to `home` and its own notebook.
        let outsider = any_agent_id(&core, &other);
        let seen = core
            .runtime()
            .block_on(core.list_grantable_agents(elsewhere, agent.clone()))
            .expect("candidates");
        assert!(
            seen.iter().any(|c| c.id == any_agent_id(&core, &home)),
            "its own space's agents are its own to know about: {seen:?}"
        );
        assert!(
            seen.iter().all(|c| c.id != outsider),
            "an agent of a space the viewer has no part in is not announced: {seen:?}"
        );
    });
}

// ===========================================================================
// Rule 4, the human arm — a click follows the same rule as a tool call
// ===========================================================================

/// Following is following, whoever does it. The GUI resolves a quoted post's
/// home before navigating to it, and that read is membership-gated: the one
/// space the shared human is genuinely not in is an agent's notebook, and an
/// ungated resolve would have opened its window. The refusal names nothing.
#[test]
fn a_human_following_a_quote_into_a_space_they_are_not_in_is_refused() {
    run(|| {
        let (_mock, core, _dir) = setup(tool_script());
        let home = turn(&core, "Hello there.", None).space_id;
        let agent = agent_id(&core, &home);
        let notebook = core
            .runtime()
            .block_on(core.promote_participant(agent.clone(), None, None))
            .expect("promotion")
            .notebook_space_id;
        let private = post(&core, "A private note to self.", Some(notebook.clone()));
        core.runtime()
            .block_on(core.rename_space(notebook.clone(), "Cartographer — notebook".into()))
            .expect("rename");

        // An ordinary post in a space the human takes part in resolves.
        let ordinary = post(&core, "Ordinary.", Some(home.clone()));
        let located = core
            .runtime()
            .block_on(core.action_location(
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                ordinary.action_id.clone(),
            ))
            .expect("a member resolves it")
            .expect("some location");
        assert_eq!(located.1, home);

        // The notebook post does not.
        let err = core
            .runtime()
            .block_on(core.action_location(
                eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                private.action_id.clone(),
            ))
            .expect_err("a non-member may not follow");
        match &err {
            AppError::NotAParticipant { action_id, .. } => {
                assert_eq!(action_id, &private.action_id);
            }
            other => panic!("expected NotAParticipant, got {other:?}"),
        }
        let rendered = err.to_string();
        for secret in [
            notebook.as_str(),
            "Cartographer — notebook",
            "A private note to self.",
        ] {
            assert!(
                !rendered.contains(secret),
                "the denial leaked {secret:?}: {rendered}"
            );
        }

        // An unknown action is "no such action", not a refusal — conflating the
        // two would make this read a membership oracle.
        assert_eq!(
            core.runtime()
                .block_on(core.action_location(
                    eidola_app_core::HUMAN_PARTICIPANT_ID.to_string(),
                    "no-such-action".into(),
                ))
                .expect("unknown is not a refusal"),
            None
        );

        // The agent, which does take part, resolves its own notebook post.
        assert!(
            core.runtime()
                .block_on(core.action_location(agent, private.action_id))
                .expect("the member resolves it")
                .is_some()
        );
    });
}
