//! **Cross-space discovery** (task 36) — the one thing a global agent needs
//! that a space-owned one does not: knowing where else it is.
//!
//! A turn's context is branch-scoped and always will be (`get_upstream_context`
//! sends exactly the thread being answered). Task 21 gave a model the thread
//! map so it could see the branches of *this* space; promotion makes an agent a
//! colleague in several conversations, and without help it cannot know the
//! others exist at all. [`ListMySpacesTool`] is the space-level analogue of
//! `list_branches`: it mirrors the human's Library, bounded by membership.
//!
//! # Membership is the boundary
//!
//! The tool reads [`db::participant_spaces`] — owned rows ∪ live reference rows,
//! the same definition `db::space_participants` reads from the other side — and
//! nothing else. There is no space id, no filter and no argument a model can
//! supply to reach a space it is not in; a space it was removed from
//! (`left_at`) disappears from its view the moment the human removes it. That
//! is the property task 37 will lean on, so it is enforced by the query rather
//! than by a check the caller could forget.
//!
//! Discovery is deliberately **all** this tool does. It reports what and where,
//! not contents: cross-space reading and search-within-membership are their own
//! decisions with their own boundaries, and the result says so plainly rather
//! than leaving a model to infer it from a missing affordance.
//!
//! # Why this is not gated on the memory opt-in
//!
//! Every other automatic tool attaches behind a gate that keeps a request
//! byte-identical when nothing is enabled: the navigation tools need a *branched*
//! space, `remember` needs the process memory opt-in, and all of them need a
//! backend that can carry a `tools` field. This one's gate is **structural**:
//! it attaches only when the responding participant is a global agent, and a
//! global agent can only exist because a human promoted one. There is no
//! install that has global agents by default, so byte-identity is preserved for
//! exactly the same reason a linear space's is — the precondition is a
//! deliberate act, not a setting.
//!
//! Riding the memory opt-in instead was considered and rejected in both
//! directions. Memory is a distinct capability (agent-owned, self-revising,
//! writing durable rows); discovery is a read over membership the human granted
//! by hand. Coupling them would mean promoting an agent without enabling memory
//! leaves a global colleague that cannot see where it is, and enabling memory
//! silently switches on cross-space visibility that was never asked for. Each
//! capability answers for itself.

use std::sync::Weak;

use crate::db;
use crate::tools::{Tool, ToolError, ToolFuture};
use crate::{Inner, derive_space_title, now_ms, plural_posts, relative_time_ms};

/// The tool name the system note promises the model. Reserved — like every
/// turn-scoped tool, it is bound to something (the responding participant's
/// identity) the process registry cannot express.
pub const LIST_MY_SPACES_TOOL_NAME: &str = "list_my_spaces";

/// The note that joins the turn's system message when the tool attaches.
/// Static, so it costs the prefix cache one flip — at promotion — and nothing
/// thereafter.
pub const GLOBAL_AGENT_NOTE: &str = "\
You are a shared participant: the same you takes part in more than one \
conversation here, and what you remember travels with you. Each conversation \
is separate — you are shown only the one you are answering in. Call \
`list_my_spaces` to see which ones you are part of.";

/// How many spaces one result lists. A membership list is small by nature; the
/// bound exists so a pathological one cannot flood a turn's context.
const MAX_SPACES: usize = 40;

/// Render the membership listing. Pure over its inputs (rows, the current
/// space, the agent's notebook, and now) — these are wire bytes, so they are
/// unit-tested.
pub(crate) fn render_spaces(
    rows: &[(db::SpaceListRow, Option<String>)],
    current_space_id: &str,
    notebook_space_id: Option<&str>,
    now: i64,
) -> String {
    if rows.is_empty() {
        return "You are not a member of any space.".to_string();
    }
    let shown = rows.len().min(MAX_SPACES);
    let mut out = format!(
        "You take part in {} conversation{}. This lists them; it does not open them — you can \
         only read and answer in the one you are in.\n",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    );
    for (row, opening) in rows.iter().take(MAX_SPACES) {
        let title = row
            .title
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .or_else(|| opening.as_deref().and_then(derive_space_title))
            .unwrap_or_else(|| "(untitled)".to_string());
        let mut marks = Vec::new();
        if row.id == current_space_id {
            marks.push("this conversation");
        }
        if Some(row.id.as_str()) == notebook_space_id {
            marks.push("your notebook");
        }
        let mark = if marks.is_empty() {
            String::new()
        } else {
            format!(" ({})", marks.join(", "))
        };
        out.push_str(&format!(
            "\n- {title} · {} · {}{mark}",
            plural_posts(row.message_count.max(0) as usize),
            relative_time_ms(row.last_activity_at, now),
        ));
    }
    if rows.len() > shown {
        out.push_str(&format!(
            "\n\nShowing the {shown} most recently active of {}.",
            rows.len()
        ));
    }
    out
}

/// `list_my_spaces` — every space this agent is a member of.
///
/// Turn-scoped like `remember`: it is bound to *this* turn's responding
/// participant, which is the whole boundary. `Weak` back to the core so a tool
/// that somehow outlived its turn can never keep the database open.
pub(crate) struct ListMySpacesTool {
    inner: Weak<Inner>,
    participant_id: String,
    current_space_id: String,
}

impl ListMySpacesTool {
    pub(crate) fn new(
        inner: Weak<Inner>,
        participant_id: String,
        current_space_id: String,
    ) -> Self {
        Self {
            inner,
            participant_id,
            current_space_id,
        }
    }
}

impl Tool for ListMySpacesTool {
    fn name(&self) -> &str {
        LIST_MY_SPACES_TOOL_NAME
    }

    fn description(&self) -> &str {
        "List the conversations you take part in, most recently active first. You are a shared \
         participant, so there may be others besides this one. This is a listing only — it does \
         not let you read them."
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    fn call<'a>(&'a self, _arguments: serde_json::Value) -> ToolFuture<'a> {
        Box::pin(async move {
            let Some(inner) = self.inner.upgrade() else {
                return Err(ToolError::new("your spaces are unavailable in this turn"));
            };
            let conn = inner
                .db_conn()
                .await
                .map_err(|e| ToolError::new(format!("could not read your spaces: {e}")))?;
            let spaces = db::participant_spaces(&conn, &self.participant_id)
                .await
                .map_err(|e| ToolError::new(format!("could not read your spaces: {e}")))?;
            let notebook = db::notebook_space_for(&conn, &self.participant_id)
                .await
                .map_err(|e| ToolError::new(format!("could not read your spaces: {e}")))?;
            // An untitled space still needs a name a model can recognize, and
            // the auto-title heuristic is the same one the Library and the
            // thread map use — one rendering rule, not three.
            let mut rows = Vec::with_capacity(spaces.len());
            for s in spaces {
                let opening = if s.title.as_deref().map(str::trim).unwrap_or("").is_empty() {
                    db::first_user_text(&conn, &s.id)
                        .await
                        .map_err(|e| ToolError::new(format!("could not read your spaces: {e}")))?
                } else {
                    None
                };
                rows.push((s, opening));
            }
            Ok(render_spaces(
                &rows,
                &self.current_space_id,
                notebook.as_deref(),
                now_ms(),
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, title: Option<&str>, posts: i64, age_ms: i64) -> db::SpaceListRow {
        db::SpaceListRow {
            id: id.to_string(),
            title: title.map(str::to_string),
            created_at: 0,
            archived_at: None,
            last_activity_at: 1_000_000 - age_ms,
            message_count: posts,
            // `db::participant_spaces` never fills this in, and this rendering
            // never reads it: a listing bounded by membership must not name a
            // conversation the asking agent may not be in.
            parent: None,
        }
    }

    #[test]
    fn the_listing_marks_this_conversation_and_the_notebook() {
        let rows = vec![
            (row("s1", Some("Parser rewrite"), 12, 7_200_000), None),
            (row("s2", Some("Ada — notebook"), 1, 86_400_000), None),
        ];
        let rendered = render_spaces(&rows, "s1", Some("s2"), 1_000_000);
        assert!(rendered.starts_with("You take part in 2 conversations."));
        assert!(rendered.contains("\n- Parser rewrite · 12 posts · 2h ago (this conversation)"));
        assert!(rendered.contains("\n- Ada — notebook · 1 post · 1d ago (your notebook)"));
    }

    #[test]
    fn an_untitled_space_falls_back_to_its_opening_line() {
        let rows = vec![(
            row("s1", None, 3, 0),
            Some("# Why does the parser drop trailing commas?".to_string()),
        )];
        let rendered = render_spaces(&rows, "other", None, 1_000_000);
        assert!(
            rendered.contains("- Why does the parser drop trailing commas? · 3 posts · just now"),
            "{rendered}"
        );
        // …and an empty space with nothing to derive from still gets a name.
        let rendered = render_spaces(&[(row("s1", Some("  "), 0, 0), None)], "x", None, 1_000_000);
        assert!(rendered.contains("- (untitled) · 0 posts"), "{rendered}");
    }

    #[test]
    fn no_membership_says_so_rather_than_rendering_an_empty_list() {
        assert_eq!(
            render_spaces(&[], "s1", None, 0),
            "You are not a member of any space."
        );
    }
}
