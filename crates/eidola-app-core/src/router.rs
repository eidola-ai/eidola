//! The **may-decline router** — one cheap model call that filters the
//! mechanical notify set down to who should actually respond.
//!
//! # Why a router and not per-agent polling
//!
//! Asking every candidate "do you want to answer this?" costs N full prefills
//! per post (prefill dominates both remote cost and local engine churn), and a
//! trimmed decline-context makes each decision worse-informed than the turn it
//! is deciding about. The deployed prior art (AutoGen speaker selection,
//! SillyTavern group-chat activation) converged on a single cheap router call,
//! and so do we.
//!
//! # Where it sits
//!
//! ```text
//!   post ──► plan_notifications ──► refine_notifications ──► drive turns
//!            (pure read, CI-       (this module: one small-  (unchanged)
//!             deterministic)        model call, degrading)
//! ```
//!
//! [`crate::AppCore::plan_notifications`] stays a **pure read** producing the
//! mechanical candidate set from the notify policies (`all` / `human` /
//! `explicit`) and the data-derived cascade guard. This module adds a step
//! *between* planning and turn-driving:
//! [`crate::AppCore::refine_notifications`], which takes that plan and returns
//! a (possibly empty) subset.
//!
//! Two things it deliberately does **not** touch:
//!
//! * **Explicit asks bypass entirely.** `respond_stream` / `respond_stream_as`
//!   consult no guards and no router — an explicit ask always runs. The router
//!   only ever filters *automatic* notification.
//! * **The cascade guard is unchanged.** It runs first, inside
//!   `plan_notifications`; a `Paused` plan is returned untouched (the router is
//!   not even called). The guard then applies to whatever the router lets
//!   through, because depth is re-derived from the data on the next plan.
//!
//! # Failure doctrine: degrade, never block
//!
//! A post is **never** blocked on router availability. If the router model is
//! unset (the default — the feature is off), unreachable, refuses to load, or
//! answers with something we can't parse, `refine_notifications` returns the
//! **mechanical candidate set unchanged** and logs honestly on stderr. The
//! failure mode is *extra* notifications, never lost ones.
//!
//! Router **non-selections are not persisted**. Not notifying someone is
//! exactly what an `explicit`-policy participant experiences today, and that
//! writes nothing; a row per participant per post would be pure noise.
//!
//! # What the router reads
//!
//! The thread slice is rendered through a [`ThreadSnapshot`] — the same
//! `post_body` rendering `read_thread` and `read_post` answer with, so a
//! quoted passage reaches the router attributed instead of as a literal
//! `{{ embed N }}` marker. The router picks *who answers*, and a post whose
//! meaning lives in what it quotes read as near-empty, which suppresses the
//! participant that should have replied. Eliding the markers (the map's
//! preview rule) would not have fixed that — it removes the noise and the
//! meaning together.
//!
//! It costs one whole-space read per triggering post, taken only for a space
//! that has a router model, and never on the writer's path: the post is
//! committed and emitted before refinement starts. The prompt cost is
//! unchanged in shape — every line was already clipped to
//! [`POST_MAX_BYTES`], so expansion changes *which* bytes the router sees, not
//! how many it may see.
//!
//! It changes *which end* they come from, though: a quoted passage is
//! unbounded, so an over-budget line is spent from both ends
//! ([`clip_middle`]) rather than head-first. A marker before the post's own
//! words — `{{ embed 1 }}\n\nHave the legal reviewer assess this` — would
//! otherwise expand into a quotation that fills the budget on its own, and the
//! router would pick a participant without ever seeing the ask.
//!
//! # Cost
//!
//! The router model is an ordinary qualified `<model>@<backend>` reference, run
//! through the shared chore runner ([`crate::utility`]) — so an engine-backed
//! reference takes the zero-spend path (what makes a local router genuinely
//! free), while a remote (`eidola`) reference is allowed and **bills a normal
//! inference on every triggering post**; any settings surface must say so
//! plainly. The call is not a turn: nothing is persisted (see that module).
//!
//! # Testing
//!
//! The plumbing is CI-deterministic: the router is just another HTTP call, so
//! the chat harness scripts it exactly like a turn (a `RouterBehavior` arm on
//! the mock upstream, selected per test). Whether the router decides *well* is
//! a judgment question — an offline eval over a golden set of
//! (post, cards, slice) → expected notify set, scored against candidate
//! models and prompts. That never runs in CI: real-model output is not
//! deterministic across machines (Metal vs CPU kernels). It would live beside
//! the other offline harnesses, not in `tests/`.

use crate::db;
use crate::error::AppError;
use crate::utility::{clip, clip_middle};
use crate::{Inner, NotificationPlan, ThreadSnapshot, build_post_tree, now_ms};

/// How many posts of the triggering post's upstream thread the router sees.
/// Small on purpose — the router is a routing decision, not a reader.
const THREAD_SLICE_POSTS: usize = 6;

/// Per-post byte budget inside the thread slice. Spent from both ends
/// ([`clip_middle`]): the lines are rendered post bodies, so an over-budget
/// one is usually a quoted passage with the post's own routing cue after it.
const POST_MAX_BYTES: usize = 480;

/// Per-candidate byte budget for the persona digest (the head of the
/// participant's system prompt).
const PERSONA_MAX_BYTES: usize = 240;

/// Completion cap for the router call. The answer is a tiny JSON object; this
/// is generous enough for a chatty small model and small enough that a remote
/// router's hold stays cheap.
const MAX_COMPLETION_TOKENS: u32 = 256;

/// One candidate the router chooses among — a participant the mechanical plan
/// already selected. Addressed by its **1-based index**, not its id: the ids
/// are UUIDv7s, and asking a small model to echo one back verbatim is a
/// needless failure mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RouterCandidate {
    pub participant_id: String,
    pub label: String,
    /// The head of the participant's effective system prompt, one line — what
    /// this agent is *for*, as far as the router needs to know.
    pub persona_digest: Option<String>,
    pub notify_policy: String,
}

/// One rendered line of the thread slice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RouterThreadLine {
    pub author: String,
    pub text: String,
}

/// The router's instruction. Deliberately terse and output-shape-first: a
/// small model that gets nothing else right still has a chance at emitting the
/// object, and an unparseable answer degrades to "notify everyone the policies
/// picked" anyway.
pub(crate) const ROUTER_SYSTEM_PROMPT: &str = "\
You are a routing filter for a group conversation. You do not answer the \
conversation; you decide which of the listed participants should respond to \
the most recent message.

Choose only participants whose stated purpose is genuinely relevant to the \
latest message. Choosing nobody is a valid and often correct answer — silence \
is better than an off-topic reply. Never explain your reasoning.

Reply with exactly one JSON object and nothing else:

{\"notify\": [<candidate numbers>]}

Use the numbers shown in the CANDIDATES list. An empty list means nobody \
responds.";

/// A participant's system prompt reduced to one clipped line — enough for the
/// router to tell a code reviewer from a copy editor, cheap enough to send for
/// every candidate on every post.
pub(crate) fn persona_digest(system_prompt: Option<&str>) -> Option<String> {
    let raw = system_prompt?.trim();
    if raw.is_empty() {
        return None;
    }
    let single_line = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(clip(&single_line, PERSONA_MAX_BYTES))
}

/// Render the router's user message: the thread slice (oldest first, the
/// triggering post last and marked), then the candidate cards.
///
/// Pure and unit-tested — the prompt is the part most likely to be tuned by an
/// offline eval, so it is kept free of I/O.
pub(crate) fn build_router_prompt(
    thread: &[RouterThreadLine],
    candidates: &[RouterCandidate],
) -> String {
    let mut out = String::new();
    out.push_str("CONVERSATION (oldest first):\n");
    if thread.is_empty() {
        out.push_str("(nothing yet)\n");
    }
    let last = thread.len().saturating_sub(1);
    for (i, line) in thread.iter().enumerate() {
        let marker = if i == last { "LATEST — " } else { "" };
        out.push_str(&format!(
            "{marker}{}: {}\n",
            line.author,
            clip_middle(&line.text, POST_MAX_BYTES)
        ));
    }
    out.push_str("\nCANDIDATES:\n");
    for (i, c) in candidates.iter().enumerate() {
        out.push_str(&format!("{}. {}", i + 1, c.label));
        if let Some(digest) = &c.persona_digest {
            out.push_str(&format!(" — {digest}"));
        }
        out.push_str(&format!(" [notify policy: {}]\n", c.notify_policy));
    }
    out.push_str(
        "\nWhich candidates should respond to the LATEST message? \
         Reply with the JSON object only.",
    );
    out
}

/// Read the router's reply into a set of 1-based candidate indices.
///
/// Tolerant of the ways a small model wraps an object (prose before it, a
/// ```` ```json ```` fence, a trailing sentence) by scanning for the first
/// balanced-looking `{ … }` span; strict about the object itself. Accepts
/// numbers or numeric strings in `notify`, ignores out-of-range entries, and
/// de-duplicates. An **empty** list is a valid answer (notify nobody).
///
/// `Err` means "we could not read a decision" — the caller degrades to the
/// mechanical set.
pub(crate) fn parse_router_selection(
    text: &str,
    candidate_count: usize,
) -> Result<Vec<usize>, String> {
    let start = text.find('{').ok_or("no JSON object in the router reply")?;
    let end = text
        .rfind('}')
        .ok_or("no JSON object in the router reply")?;
    if end <= start {
        return Err("no JSON object in the router reply".into());
    }
    let value: serde_json::Value =
        serde_json::from_str(&text[start..=end]).map_err(|e| format!("invalid JSON: {e}"))?;
    let entries = value
        .get("notify")
        .ok_or("the router reply has no `notify` key")?
        .as_array()
        .ok_or("`notify` is not an array")?;

    let mut out: Vec<usize> = Vec::new();
    for entry in entries {
        let n = match entry {
            serde_json::Value::Number(n) => n.as_u64(),
            serde_json::Value::String(s) => s.trim().parse::<u64>().ok(),
            _ => None,
        };
        let Some(n) = n else { continue };
        let n = n as usize;
        if n >= 1 && n <= candidate_count && !out.contains(&n) {
            out.push(n);
        }
    }
    Ok(out)
}

impl Inner {
    /// Filter a mechanical notification plan through the space's router model.
    ///
    /// Never fails: every error path returns `plan` unchanged (see the module
    /// docs' failure doctrine). Returns immediately — without touching the
    /// network — when the plan is `Paused`, when it holds fewer than two turns'
    /// worth of choice to make, or when the space has no router model.
    pub(crate) async fn refine_notifications(
        &self,
        space_id: &str,
        post_action_id: &str,
        plan: NotificationPlan,
    ) -> NotificationPlan {
        let turns = match &plan {
            // The cascade guard already spoke; the router is not consulted.
            NotificationPlan::Paused { .. } => return plan,
            NotificationPlan::Turns(t) if t.is_empty() => return plan,
            NotificationPlan::Turns(t) => t.clone(),
        };

        let Ok(conn) = self.db_conn().await else {
            return plan;
        };
        // Unset (the default) = the feature is off: no router call is made.
        let router_model = match db::space_router_model(&conn, space_id).await {
            Ok(Some(m)) => m,
            Ok(None) => return plan,
            Err(e) => {
                eprintln!("warning: could not read the space's router model: {e}");
                return plan;
            }
        };

        let (thread, candidates) = match self
            .router_inputs(&conn, space_id, post_action_id, &turns)
            .await
        {
            Ok(inputs) => inputs,
            Err(e) => {
                eprintln!("warning: may-decline router inputs unavailable: {e}");
                return plan;
            }
        };
        if candidates.is_empty() {
            return plan;
        }

        let prompt = build_router_prompt(&thread, &candidates);
        let reply = match self
            .router_completion(&conn, &router_model, ROUTER_SYSTEM_PROMPT, &prompt)
            .await
        {
            Ok(reply) => reply,
            Err(e) => {
                eprintln!(
                    "warning: may-decline router `{router_model}` unavailable ({e}); \
                     notifying the mechanical set"
                );
                return plan;
            }
        };

        let selected = match parse_router_selection(&reply, candidates.len()) {
            Ok(selected) => selected,
            Err(e) => {
                eprintln!(
                    "warning: may-decline router `{router_model}` returned unusable output \
                     ({e}); notifying the mechanical set"
                );
                return plan;
            }
        };

        let keep: Vec<&str> = selected
            .iter()
            .map(|i| candidates[i - 1].participant_id.as_str())
            .collect();
        NotificationPlan::Turns(
            turns
                .into_iter()
                .filter(|t| keep.contains(&t.participant_id.as_str()))
                .collect(),
        )
    }

    /// Gather the router's inputs: the triggering post's upstream thread slice
    /// and one card per planned participant.
    ///
    /// The slice is rendered through the snapshot, so the router reads a post
    /// exactly as `read_thread` does — see [`ThreadSnapshot::body_for_action`]
    /// and the note on the whole-space read below.
    async fn router_inputs(
        &self,
        conn: &turso::Connection,
        space_id: &str,
        post_action_id: &str,
        turns: &[crate::PlannedTurn],
    ) -> Result<(Vec<RouterThreadLine>, Vec<RouterCandidate>), AppError> {
        // The same branch-scoped, item-tip-resolved ancestry a turn would send
        // upstream — inclusive of the triggering post — trimmed to its tail.
        let rows = db::get_upstream_context(conn, post_action_id, true).await?;

        // One row per (action, text block), so an action's blocks arrive as one
        // consecutive run and are folded into the one post they are — which is
        // what `THREAD_SLICE_POSTS` has always claimed to count.
        let mut posts: Vec<(String, String, String)> = Vec::new();
        for r in rows {
            let text = r.text_content.unwrap_or_default();
            match posts.last_mut() {
                Some((action_id, _, body)) if *action_id == r.action_id => body.push_str(&text),
                _ => posts.push((r.action_id, r.participant_label, text)),
            }
        }

        // One whole-space read, off the writer's path (the post is committed
        // and emitted before the router runs) and taken only for a space that
        // has a router model at all. It buys the rendering rather than the
        // structure: a post whose meaning lives in a quoted passage read as
        // near-empty here, which suppresses the participant that should have
        // answered it.
        let snapshot = ThreadSnapshot::new(
            build_post_tree(db::get_space_tree_data(conn, space_id).await?),
            now_ms(),
        );
        let skip = posts.len().saturating_sub(THREAD_SLICE_POSTS);
        let thread: Vec<RouterThreadLine> = posts
            .into_iter()
            .skip(skip)
            .filter_map(|(action_id, author, raw)| {
                let text = snapshot.body_for_action(&action_id).unwrap_or(raw);
                (!text.is_empty()).then_some(RouterThreadLine { author, text })
            })
            .collect();

        let members = db::space_participants(conn, space_id).await?;
        let candidates = turns
            .iter()
            .filter_map(|t| {
                let m = members
                    .iter()
                    .find(|m| m.participant_id == t.participant_id)?;
                Some(RouterCandidate {
                    participant_id: m.participant_id.clone(),
                    label: m.label.clone(),
                    persona_digest: persona_digest(m.system_prompt.as_deref()),
                    notify_policy: m.notify_policy.clone(),
                })
            })
            .collect();
        Ok((thread, candidates))
    }

    /// One non-streaming chat completion against the router model, through the
    /// shared chore runner ([`crate::utility`]) — zero-spend for an
    /// engine-backed or `openai` reference, spend-and-settle for an `eidola`
    /// one. Persists nothing.
    async fn router_completion(
        &self,
        db_conn: &turso::Connection,
        model_ref: &str,
        system: &str,
        user: &str,
    ) -> Result<String, AppError> {
        let target = self
            .resolve_utility_target(db_conn, model_ref, "router")
            .await?;
        self.utility_completion(db_conn, &target, system, user, MAX_COMPLETION_TOKENS)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidates(n: usize) -> Vec<RouterCandidate> {
        (1..=n)
            .map(|i| RouterCandidate {
                participant_id: format!("p{i}"),
                label: format!("Agent {i}"),
                persona_digest: None,
                notify_policy: "all".into(),
            })
            .collect()
    }

    #[test]
    fn parses_a_bare_object() {
        assert_eq!(
            parse_router_selection(r#"{"notify": [1, 3]}"#, 3).unwrap(),
            vec![1, 3]
        );
    }

    #[test]
    fn parses_a_fenced_object_with_prose_around_it() {
        let reply = "Sure!\n```json\n{\"notify\": [2]}\n```\nHope that helps.";
        assert_eq!(parse_router_selection(reply, 3).unwrap(), vec![2]);
    }

    #[test]
    fn an_empty_selection_is_valid_and_means_nobody() {
        assert!(
            parse_router_selection(r#"{"notify": []}"#, 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn numeric_strings_are_accepted_and_duplicates_collapse() {
        assert_eq!(
            parse_router_selection(r#"{"notify": ["2", 2, 1]}"#, 3).unwrap(),
            vec![2, 1]
        );
    }

    #[test]
    fn out_of_range_entries_are_ignored_not_fatal() {
        assert_eq!(
            parse_router_selection(r#"{"notify": [0, 1, 9]}"#, 3).unwrap(),
            vec![1]
        );
    }

    #[test]
    fn unusable_output_is_an_error_so_the_caller_degrades() {
        for reply in [
            "I think Agent 2 should answer.",
            "{oops",
            r#"{"notified": [1]}"#,
            r#"{"notify": "everyone"}"#,
        ] {
            assert!(
                parse_router_selection(reply, 3).is_err(),
                "expected an error for {reply:?}"
            );
        }
    }

    #[test]
    fn the_prompt_numbers_candidates_and_marks_the_latest_post() {
        let thread = vec![
            RouterThreadLine {
                author: "You".into(),
                text: "first".into(),
            },
            RouterThreadLine {
                author: "You".into(),
                text: "second".into(),
            },
        ];
        let prompt = build_router_prompt(&thread, &candidates(2));
        assert!(prompt.contains("You: first"));
        assert!(prompt.contains("LATEST — You: second"));
        assert!(prompt.contains("1. Agent 1"));
        assert!(prompt.contains("2. Agent 2"));
        assert!(prompt.contains("[notify policy: all]"));
    }

    /// A marker standing **before** the post's own words expands into a
    /// passage that can be longer than the whole per-post budget — a quote is
    /// a range the author chose, and nothing bounds it. Clipping the tail
    /// would then spend the budget on the quotation and drop the routing cue
    /// the post was written to carry, so the router would pick a participant
    /// from the quoted passage alone.
    #[test]
    fn a_long_quote_does_not_evict_the_posts_own_words() {
        let rendered = format!(
            "[1] #q2m9zzr · Ada\n> {}\n\nHave the legal reviewer assess this",
            "the quoted passage ".repeat(40)
        );
        assert!(
            rendered.len() > POST_MAX_BYTES,
            "the fixture is over budget"
        );
        let thread = vec![RouterThreadLine {
            author: "You".into(),
            text: rendered,
        }];
        let prompt = build_router_prompt(&thread, &candidates(2));
        assert!(
            prompt.contains("Have the legal reviewer assess this"),
            "the post's own ask reaches the router; got {prompt}"
        );
        assert!(
            prompt.contains("[1] #q2m9zzr · Ada"),
            "and so does the passage's attribution; got {prompt}"
        );
        assert!(prompt.contains('…'), "with the cut marked; got {prompt}");
    }

    #[test]
    fn persona_digests_are_one_line_and_clipped() {
        assert_eq!(persona_digest(None), None);
        assert_eq!(persona_digest(Some("   ")), None);
        assert_eq!(
            persona_digest(Some("You are\na careful\treviewer.")).unwrap(),
            "You are a careful reviewer."
        );
        let long = "x".repeat(PERSONA_MAX_BYTES + 50);
        let digest = persona_digest(Some(&long)).unwrap();
        assert!(digest.ends_with('…'));
        assert!(digest.len() <= PERSONA_MAX_BYTES + 4);
    }
}
