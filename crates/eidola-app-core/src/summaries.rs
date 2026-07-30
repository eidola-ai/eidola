//! **LLM-written branch summaries** — a progressive enhancement over the
//! structural thread map (task 21's checkpoint 3).
//!
//! The map's structural line (`#handle · author · posts · last activity —
//! opening line`) is free, deterministic, and **always present**. This module
//! adds one optional line under it: a one-to-two-sentence précis of what the
//! branch is actually about, written by a cheap model. Everything here can fail
//! or be switched off and the map is exactly what it was.
//!
//! # Cache: keyed on the branch tip
//!
//! A summary is generated once per *state* of a branch. The key is the branch's
//! **tip action id** — the newest post in its subtree — so a summary is
//! regenerated when the branch grows (or when a post in it is edited or
//! regenerated, which changes the tip's action id), and never otherwise. A
//! stored summary is rendered even when the branch has since grown past it: it
//! is a snapshot, exactly like the navigation tools' results, and the structural
//! line beside it carries the fresh counts.
//!
//! # Storage: a versioned item, forensically visible
//!
//! A summary is an ordinary action — `action_type = 'checkpoint'`, `intent =
//! 'branch_summary'`, authored by the global [`db::SYSTEM_PARTICIPANT_ID`]
//! (Eidola on its own behalf; attributing it to a human or an agent would be a
//! lie in the Record), carrying the text as its one `text` block and the utility
//! model in `model`. Regeneration **supersedes** the prior summary within one
//! item, so the whole history of what the harness told the model about a branch
//! stays readable — the same generations machinery posts use.
//!
//! Two `reference` antecedent edges carry the keys, in real relations rather
//! than packed into a string: ordinal [`db::BRANCH_SUMMARY_ROOT_ORDINAL`] → the
//! branch root (its *item* is the branch's stable identity, so an edited root
//! does not orphan the summary) and ordinal [`db::BRANCH_SUMMARY_TIP_ORDINAL`]
//! → the tip it read (the cache key). Ordinal 0 stays empty: a summary has no
//! `reply` edge because it is not part of the thread. `checkpoint` is not a post
//! type, so `get_space_tree`, both context queries, and `mechanical_plan`'s post
//! allowlist already collapse it out — a summary is invisible as a post and
//! fully present in the Record.
//!
//! # Model and cost
//!
//! v1 shares the space's existing `router_model` as its utility model (one knob
//! until a real need splits them), through the same chore runner the router
//! uses — so an engine-backed reference is free and a remote `eidola` one bills
//! a normal inference, exactly as it does for routing. That is configured
//! behavior, not a surprise; what keeps it *bounded* is structural rather than
//! a policy gate: generation is lazy, keyed on the branch tip (no regeneration
//! without growth), skips branches too short to be worth a précis, and costs at
//! most one call per branch that actually grew. Unset (the default) or an
//! unresolvable reference ⇒ structural lines only, silently.
//!
//! # Scheduling: never in a turn's way
//!
//! Generation is **never** on the path of a post or a turn.
//! [`Inner::spawn_branch_summaries`] is fire-and-forget, called after a post or
//! a turn has committed and emitted; `prepare_turn` only ever *reads* whatever
//! is already stored. Two things keep a burst cheap: a **trailing debounce**
//! per space (a later trigger supersedes an earlier one, so an exchange —
//! post, then the answer it draws — costs one pass, not two), and a gate that
//! runs one pass at a time with the cache re-read inside it. A pass with
//! nothing stale makes no HTTP call and does not even start an engine: the
//! freshness check happens before the route is opened.

use std::collections::HashMap;

use uuid::Uuid;

use crate::db;
use crate::error::AppError;
use crate::utility::clip;
use crate::{Change, Inner, ThreadSnapshot, build_post_tree, now_ms};

/// What this chore is called in diagnostics.
const CHORE: &str = "branch summary";

/// A branch shorter than this is not summarized: its structural entry already
/// quotes the opening line of its only post, so a précis would restate it.
const MIN_BRANCH_POSTS: usize = 2;

/// How many of a branch's posts the summarizer reads (oldest first — the
/// opening is what a reader needs to place the branch).
const BRANCH_SLICE_POSTS: usize = 12;

/// Per-post byte budget inside that slice.
const POST_MAX_BYTES: usize = 600;

/// Byte budget for the stored summary. Two sentences; the map's line format
/// stays readable and the trailing block stays small.
const SUMMARY_MAX_BYTES: usize = 240;

/// Completion cap for one summary call.
const MAX_COMPLETION_TOKENS: u32 = 160;

/// How long a trigger waits for the space to go quiet before its pass runs.
/// Long enough that a post and the answer it draws are summarized once, short
/// enough that the map catches up within the reader's own pause.
const DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(20);

/// The summarizer's instruction. Output-shape-first, like the router's: a small
/// model that gets nothing else right should still emit one usable line, and an
/// unusable answer degrades to the structural entry anyway.
pub(crate) const SUMMARY_SYSTEM_PROMPT: &str = "\
You summarize one branch of a threaded conversation so that a reader who \
cannot see it knows whether it is worth opening.

Write one or two sentences of plain prose. State what the branch is about and \
where it got to. Do not greet, do not explain what you are doing, do not use \
markdown, do not quote the posts verbatim, and never write more than two \
sentences.";

/// One post of a branch, as the summarizer reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SummaryPost {
    pub author: String,
    pub text: String,
}

/// What identifies a branch and the state it is in — the summarizer's cache
/// key, read off a [`ThreadSnapshot`].
pub(crate) struct BranchKey {
    /// Item of the branch's root post: the branch's identity, stable across
    /// edits of that post.
    pub root_item_id: String,
    /// Current generation of that root — what the summary's root edge names.
    pub root_action_id: String,
    /// The newest post anywhere in the branch: the cache key.
    pub tip_action_id: String,
    /// Posts in the branch (current generations only).
    pub posts: usize,
    /// The branch's handle, for diagnostics.
    pub handle: String,
}

/// Render the summarizer's user message: the branch's posts, oldest first.
///
/// Pure and unit-tested — the prompt is what an offline eval would tune, so it
/// stays free of I/O.
pub(crate) fn build_summary_prompt(posts: &[SummaryPost]) -> String {
    let mut out = String::from("BRANCH (oldest first):\n");
    if posts.is_empty() {
        out.push_str("(no text)\n");
    }
    for p in posts.iter().take(BRANCH_SLICE_POSTS) {
        out.push_str(&format!(
            "{}: {}\n",
            p.author,
            clip(&p.text, POST_MAX_BYTES)
        ));
    }
    out.push_str("\nSummarize this branch in one or two sentences.");
    out
}

/// Reduce a model's answer to the single line the map renders.
///
/// Tolerant of the ways a small model dresses an answer (a fence, surrounding
/// quotes, a `Summary:` label, hard-wrapped lines) and strict about the result
/// being one line: the map's entry format is line-oriented, so a multi-line
/// summary would break it. `None` means "nothing usable" — the caller keeps the
/// structural entry.
pub(crate) fn clean_summary(raw: &str) -> Option<String> {
    let mut text = raw.trim();
    if let Some(rest) = text.strip_prefix("```") {
        // Drop the fence's info string and its closing fence.
        let rest = rest.split_once('\n').map(|(_, r)| r).unwrap_or("");
        text = rest.split("```").next().unwrap_or("").trim();
    }
    let mut single = text.split_whitespace().collect::<Vec<_>>().join(" ");
    for label in ["Summary:", "SUMMARY:", "summary:"] {
        if let Some(rest) = single.strip_prefix(label) {
            single = rest.trim().to_string();
        }
    }
    let single = single.trim_matches(['"', '\'', '“', '”', '*']).trim();
    if single.is_empty() {
        return None;
    }
    Some(clip(single, SUMMARY_MAX_BYTES))
}

impl Inner {
    /// Refresh a space's branch summaries in the background. Fire-and-forget:
    /// the caller has already committed and emitted, and nothing here can delay
    /// or fail a post or a turn.
    ///
    /// **Trailing debounce.** Each call stamps the space and waits
    /// [`DEBOUNCE`]; whoever finds its own stamp still current runs the pass,
    /// so a later trigger supersedes an earlier one. That is what keeps one
    /// exchange — the post, then the answer it draws, each of which moves the
    /// branch tip — from summarizing the same branch twice.
    pub(crate) fn spawn_branch_summaries(&self, space_id: &str) {
        // No runtime (a synchronous unit test) ⇒ nothing to spawn onto.
        if tokio::runtime::Handle::try_current().is_err() {
            return;
        }
        let Some(inner) = self.self_ref.upgrade() else {
            return;
        };
        let space_id = space_id.to_string();
        let stamp = now_ms();
        inner
            .summary_triggers
            .lock()
            .expect("summary trigger lock poisoned")
            .insert(space_id.clone(), stamp);
        tokio::spawn(async move {
            tokio::time::sleep(DEBOUNCE).await;
            {
                let mut triggers = inner
                    .summary_triggers
                    .lock()
                    .expect("summary trigger lock poisoned");
                if triggers.get(&space_id) != Some(&stamp) {
                    return; // a later trigger owns this pass
                }
                triggers.remove(&space_id);
            }
            if let Err(e) = inner.refresh_branch_summaries(&space_id).await {
                eprintln!(
                    "warning: branch summaries for space {space_id} could not be refreshed: {e}"
                );
            }
        });
    }

    /// One summary pass over a space: generate a summary for every branch whose
    /// stored one is missing or stale, and commit each as a new generation.
    ///
    /// Degrades silently at every gate (see the module docs) — an `Err` here
    /// means the local database itself was unreachable.
    pub(crate) async fn refresh_branch_summaries(&self, space_id: &str) -> Result<(), AppError> {
        // One pass at a time. A pass queued behind another re-reads the cache
        // below, so a burst of posts costs one generation per branch, not one
        // per post.
        let _pass = self.summary_gate.lock().await;
        let conn = self.db_conn().await?;

        // v1 shares the router's model. Unset (the default) = summaries off.
        let Some(model_ref) = db::space_router_model(&conn, space_id).await? else {
            return Ok(());
        };
        let target = match self.resolve_utility_target(&conn, &model_ref, CHORE).await {
            Ok(target) => target,
            Err(e) => {
                eprintln!(
                    "warning: branch summaries need the utility model `{model_ref}`, which is \
                     unavailable ({e}); the map keeps its structural entries"
                );
                return Ok(());
            }
        };
        let snapshot = ThreadSnapshot::new(
            build_post_tree(db::get_space_tree_data(&conn, space_id).await?),
            now_ms(),
        );
        let stored: HashMap<String, db::BranchSummaryRow> =
            db::current_branch_summaries(&conn, space_id)
                .await?
                .into_iter()
                .map(|r| (r.branch_item_id.clone(), r))
                .collect();

        // Freshness is decided *before* the route is opened, so a pass with
        // nothing to do starts no engine and makes no request.
        let stale: Vec<usize> = snapshot
            .branch_roots()
            .into_iter()
            .filter(|&idx| {
                let key = snapshot.branch_key(idx);
                key.posts >= MIN_BRANCH_POSTS
                    && !stored
                        .get(&key.root_item_id)
                        .is_some_and(|s| s.summarized_action_id == key.tip_action_id)
            })
            .collect();

        for idx in stale {
            let key = snapshot.branch_key(idx);
            let prompt = build_summary_prompt(&snapshot.branch_posts(idx, BRANCH_SLICE_POSTS));
            // A per-branch failure is per-branch: the next branch may well be
            // summarizable, and the whole feature is optional anyway.
            let reply = match self
                .utility_completion(
                    &conn,
                    &target,
                    SUMMARY_SYSTEM_PROMPT,
                    &prompt,
                    MAX_COMPLETION_TOKENS,
                )
                .await
            {
                Ok(reply) => reply,
                Err(e) => {
                    eprintln!(
                        "warning: branch #{} could not be summarized ({e})",
                        key.handle
                    );
                    continue;
                }
            };
            let Some(summary) = clean_summary(&reply) else {
                eprintln!(
                    "warning: the utility model returned nothing usable for branch #{}",
                    key.handle
                );
                continue;
            };
            self.commit_branch_summary(
                &conn,
                space_id,
                &target.canonical,
                &key,
                stored.get(&key.root_item_id),
                &summary,
            )
            .await?;
            self.bus.emit(Change::Space(space_id.to_string()));
        }
        Ok(())
    }

    /// Persist one summary — a fresh item, or a new generation superseding the
    /// branch's prior summary.
    async fn commit_branch_summary(
        &self,
        conn: &turso::Connection,
        space_id: &str,
        model: &str,
        key: &BranchKey,
        prior: Option<&db::BranchSummaryRow>,
        summary: &str,
    ) -> Result<(), AppError> {
        let action_id = Uuid::now_v7().to_string();
        let (item_id, supersedes) = match prior {
            Some(p) => (p.item_id.clone(), Some(p.action_id.clone())),
            None => (Uuid::now_v7().to_string(), None),
        };
        db::insert_action(
            conn,
            &db::ActionEntry {
                id: action_id.clone(),
                space_id: space_id.to_string(),
                participant_id: db::SYSTEM_PARTICIPANT_ID.to_string(),
                participant_scope: "global".to_string(),
                item_id,
                supersedes_action_id: supersedes,
                action_type: "checkpoint".to_string(),
                status: "complete".to_string(),
                intent: Some(db::BRANCH_SUMMARY_INTENT.to_string()),
                model: Some(model.to_string()),
                input_tokens: None,
                output_tokens: None,
                credits_consumed: None,
                created_at: now_ms(),
            },
        )
        .await?;
        db::insert_action_antecedent(
            conn,
            &action_id,
            &key.root_action_id,
            db::BRANCH_SUMMARY_ROOT_ORDINAL,
            "reference",
        )
        .await?;
        db::insert_action_antecedent(
            conn,
            &action_id,
            &key.tip_action_id,
            db::BRANCH_SUMMARY_TIP_ORDINAL,
            "reference",
        )
        .await?;
        db::insert_text_content_block(
            conn,
            &Uuid::now_v7().to_string(),
            &action_id,
            0,
            "text",
            summary,
        )
        .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_answer_survives_intact() {
        assert_eq!(
            clean_summary("They argue about the intro and settle on cutting it."),
            Some("They argue about the intro and settle on cutting it.".to_string())
        );
    }

    #[test]
    fn fences_labels_quotes_and_wrapping_are_stripped() {
        for raw in [
            "```\nThey argue about the intro.\n```",
            "```text\nThey argue about the intro.\n```",
            "Summary: They argue about the intro.",
            "\"They argue about the intro.\"",
            "They argue\nabout the intro.",
        ] {
            assert_eq!(
                clean_summary(raw).as_deref(),
                Some("They argue about the intro."),
                "for {raw:?}"
            );
        }
    }

    #[test]
    fn an_empty_answer_is_no_summary() {
        assert_eq!(clean_summary(""), None);
        assert_eq!(clean_summary("   \n  "), None);
        assert_eq!(clean_summary("```\n\n```"), None);
    }

    #[test]
    fn a_long_answer_is_clipped_to_one_line() {
        let long = "word ".repeat(200);
        let cleaned = clean_summary(&long).unwrap();
        assert!(cleaned.len() <= SUMMARY_MAX_BYTES + 4);
        assert!(!cleaned.contains('\n'));
    }

    #[test]
    fn the_prompt_lists_posts_oldest_first_and_clips_them() {
        let posts = vec![
            SummaryPost {
                author: "You".into(),
                text: "first".into(),
            },
            SummaryPost {
                author: "Agent".into(),
                text: "x".repeat(POST_MAX_BYTES + 50),
            },
        ];
        let prompt = build_summary_prompt(&posts);
        let first = prompt.find("You: first").unwrap();
        let second = prompt.find("Agent: ").unwrap();
        assert!(first < second);
        assert!(prompt.contains('…'), "the long post is clipped");
    }
}
