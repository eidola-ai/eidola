//! **Trace visibility** (task 34) — the quiet, subordinate disclosure that
//! lets a reader audit what a turn actually did without leaving the space.
//!
//! Tool rounds ([task 20]), navigation-tool descents (task 21) and decline
//! decisions (task 22) are persisted as `tool_call` / `tool_result` /
//! `decision` actions that the post tree deliberately collapses out — they are
//! not posts. `AppCore::space_traces` is the parallel read that puts them back,
//! each **anchored to a post the tree already renders**, which is what keeps
//! `PostNode` and its virtualization untouched.
//!
//! Two anchors, and the difference is the point:
//!
//! - a turn that **answered** anchors on its own inference (attribution is the
//!   turn's context assembly — see task 33), so the disclosure sits under the
//!   post it explains;
//! - a turn that produced **no post at all** — a decline, a round-cap exit, a
//!   failed loop — anchors on the post it answered. That is the *gap*: the
//!   audit value of a decline is precisely that a non-event is visible.
//!
//! Design register (the footnote rail is the precedent, `references.rs`):
//! collapsed by default to a single quiet line in the reading column; expanded
//! to a ruled rail of one line per round — tool name, terse argument, terse
//! result. **Disclosure, not duplication**: raw payloads stay in the Record,
//! and each round's line links straight through to its own raw exchange there.
//! Nothing floats, so the overlay-containment doctrine has nothing to contain.

use eidola_app_core::{PostTrace, TraceEntry};
use gpui::prelude::FluentBuilder as _;
use gpui::{
    AnyElement, Context, InteractiveElement, IntoElement, ParentElement, SharedString,
    StatefulInteractiveElement, Styled, div, px,
};
use gpui_component::{ActiveTheme, h_flex, v_flex};

use crate::probe::Probe as _;

use super::SpaceView;
use super::model::TreeNode;

/// Longest argument/result summary rendered inline. Past this the line is
/// elided and the Record is one click away — the disclosure summarizes, it
/// never re-renders a payload.
const SUMMARY_CHARS: usize = 120;

impl SpaceView {
    /// Ask the space for its trace index once (idempotent; the entity owns the
    /// fetch slot). Called at the head of `render` beside `sync_references`.
    pub(crate) fn sync_traces(&mut self, cx: &mut Context<Self>) {
        self.space.update(cx, |space, cx| space.ensure_traces(cx));
    }

    /// The trace disclosure for post `i`, if that post anchors any activity.
    ///
    /// Collapsed: one quiet line ("3 tool calls", or "Gemma — declined to
    /// respond" where the turn left no post to carry the byline). Expanded: a
    /// ruled rail of one line per round, each linking into the Record.
    pub(crate) fn render_post_traces(
        &self,
        i: usize,
        node: &TreeNode,
        cx: &Context<Self>,
    ) -> Option<AnyElement> {
        let action_id = self.posts.get(i)?.action_id.clone()?;
        let traces = self.space.read(cx).traces_for(&action_id);
        if traces.is_empty() {
            return None;
        }
        let expanded = self.space.read(cx).trace_expanded(&action_id);
        let theme = cx.theme();
        let (fg, fg_hover, bg_hover, rule) = (
            theme.muted_foreground,
            theme.foreground,
            theme.muted,
            theme.border,
        );

        let toggle_target = action_id.to_string();
        let mut col = v_flex().mt_2().gap_1().child(
            h_flex()
                .id(SharedString::from(format!("space-trace-{}", node.id)))
                .probe(
                    format!("space/post/{i}/trace"),
                    gpui::Role::Button,
                    if expanded {
                        "Hide what this turn did"
                    } else {
                        "Show what this turn did"
                    },
                )
                .aria_expanded(expanded)
                .self_start()
                .px_1()
                .ml_neg_1()
                .rounded_md()
                .cursor_pointer()
                .text_xs()
                .text_color(fg)
                .hover(move |s| s.text_color(fg_hover).bg(bg_hover))
                .child(summary_line(traces))
                .on_click(cx.listener(move |this, _, _, cx| {
                    let target = toggle_target.clone();
                    this.space
                        .update(cx, |space, cx| space.toggle_trace(&target, cx));
                })),
        );

        if expanded {
            let mut rail = v_flex().pt_1p5().gap_0p5().border_t_1().border_color(rule);
            let mut n = 0usize;
            for trace in traces {
                for entry in &trace.entries {
                    n += 1;
                    rail = rail.child(self.trace_row(i, node, n, entry, cx));
                }
            }
            col = col.child(rail);
        }
        Some(col.into_any_element())
    }

    /// One line of the expanded rail: index, what ran, what came back.
    ///
    /// A tool round is a link — clicking opens the Record on that round's own
    /// raw request/response pair, which is where the full payloads live. A
    /// decision has no exchange of its own to open, so it is plain text.
    fn trace_row(
        &self,
        post: usize,
        node: &TreeNode,
        n: usize,
        entry: &TraceEntry,
        cx: &Context<Self>,
    ) -> AnyElement {
        let theme = cx.theme();
        let (name, detail, quiet, request_id) = match entry {
            TraceEntry::Tool {
                request_id,
                name,
                arguments,
                result,
                ..
            } => (
                SharedString::from(if name.is_empty() {
                    "(unnamed tool)".to_string()
                } else {
                    name.clone()
                }),
                SharedString::from(tool_detail(arguments, result.as_deref())),
                result.is_none(),
                request_id.clone(),
            ),
            TraceEntry::Declined { reason, .. } => (
                SharedString::from("declined"),
                SharedString::from(match reason {
                    Some(r) => summarize(r),
                    None => "no reason given".to_string(),
                }),
                reason.is_none(),
                None,
            ),
        };

        let aria = format!("Round {n}: {name} — {detail}");
        let linked = request_id.is_some();
        let row = h_flex()
            // Keyed on the round index, not the action id: one `tool_call`
            // action can carry several parallel calls, and sibling element ids
            // must be unique.
            .id(SharedString::from(format!(
                "space-trace-row-{}-{n}",
                node.id
            )))
            .probe(
                format!("space/post/{post}/trace/{n}"),
                if linked {
                    gpui::Role::Link
                } else {
                    gpui::Role::ListItem
                },
                aria,
            )
            .w_full()
            .items_baseline()
            .gap_1p5()
            .text_xs()
            .when(linked, |d| d.cursor_pointer())
            .child(
                div()
                    .flex_none()
                    .w(px(14.))
                    .text_color(theme.muted_foreground)
                    .child(format!("{n}.")),
            )
            .child(
                div()
                    .flex_none()
                    .text_color(theme.muted_foreground)
                    .child(name),
            )
            .child(
                div()
                    .flex_1()
                    .min_w_0()
                    .truncate()
                    .text_color(if quiet {
                        theme.muted_foreground.opacity(0.75)
                    } else {
                        theme.muted_foreground
                    })
                    .child(detail),
            );
        match request_id {
            Some(id) => row
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.open_in_record(id.clone(), cx);
                }))
                .into_any_element(),
            None => row.into_any_element(),
        }
    }

    /// Open the Record window on a round's raw exchange. The id is recorded
    /// first so behavior tests (which run without `AppGlobal`) can assert the
    /// deep link without a real window — the `last_record_request` seam
    /// mirrors `Space::last_submitted_model`.
    pub fn open_in_record(&mut self, request_id: String, cx: &mut Context<Self>) {
        self.last_record_request = Some(request_id.clone());
        cx.notify();
        // Deferred so the window opens after this update cycle completes (the
        // Library's / the Record's own cross-window idiom).
        cx.defer(move |cx: &mut gpui::App| crate::open_record_request(cx, request_id));
    }
}

/// The collapsed line: what happened, in as few words as stay honest.
///
/// A turn that answered needs no byline — the post above carries it. A turn
/// that left **no** post does, because the disclosure is then hanging under
/// somebody else's post and "declined" with no name would be a riddle.
fn summary_line(traces: &[PostTrace]) -> String {
    let calls: usize = traces
        .iter()
        .flat_map(|t| t.entries.iter())
        .filter(|e| matches!(e, TraceEntry::Tool { .. }))
        .count();
    let declined = traces
        .iter()
        .flat_map(|t| t.entries.iter())
        .any(|e| matches!(e, TraceEntry::Declined { .. }));
    let unanswered = traces.iter().any(|t| t.unanswered);

    let mut parts: Vec<String> = Vec::new();
    if declined {
        parts.push("declined to respond".to_string());
    }
    if calls > 0 {
        parts.push(format!(
            "{calls} tool call{}",
            if calls == 1 { "" } else { "s" }
        ));
    }
    if unanswered && !declined {
        parts.push("no response".to_string());
    }
    if parts.is_empty() {
        parts.push("activity".to_string());
    }
    let body = parts.join(" · ");

    match traces.iter().find(|t| t.unanswered) {
        Some(t) if !t.participant_label.trim().is_empty() => {
            format!("{} — {body}", t.participant_label.trim())
        }
        _ => body,
    }
}

/// A round's one-line detail: the arguments it was called with and what came
/// back, both flattened and elided. Neither is a payload — the Record is.
fn tool_detail(arguments: &str, result: Option<&str>) -> String {
    let args = match summarize(arguments).as_str() {
        // A no-argument call ("{}") says nothing; printing it is noise in a
        // line whose whole job is to be scannable.
        "{}" | "" => String::new(),
        s => s.to_string(),
    };
    match result {
        // A round the loop never executed (the cap withholds a capped round's
        // tools) says so rather than showing a blank result.
        None => {
            if args.is_empty() {
                "not run".to_string()
            } else {
                format!("{args} — not run")
            }
        }
        Some(r) => {
            let out = summarize(r);
            match (args.is_empty(), out.is_empty()) {
                (true, true) => String::new(),
                // The arrow stays even with nothing to its left, so every row
                // in the rail reads the same way: what went in, what came back.
                (true, false) => format!("→ {out}"),
                (false, true) => args,
                (false, false) => format!("{args} → {out}"),
            }
        }
    }
}

/// Flatten a payload to one line and elide it. Whitespace runs collapse (a
/// JSON blob or a multi-paragraph tool result must not smear the rail), and
/// truncation is char-boundary safe.
pub(crate) fn summarize(text: &str) -> String {
    let flat: String = {
        let mut out = String::with_capacity(text.len().min(SUMMARY_CHARS * 2));
        let mut in_space = false;
        for ch in text.chars() {
            if ch.is_whitespace() {
                in_space = true;
                continue;
            }
            if in_space && !out.is_empty() {
                out.push(' ');
            }
            in_space = false;
            out.push(ch);
            if out.chars().count() > SUMMARY_CHARS {
                break;
            }
        }
        out
    };
    if flat.chars().count() > SUMMARY_CHARS {
        let cut = flat
            .char_indices()
            .nth(SUMMARY_CHARS)
            .map(|(i, _)| i)
            .unwrap_or(flat.len());
        format!("{}…", &flat[..cut])
    } else {
        flat
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str, args: &str, result: Option<&str>) -> TraceEntry {
        TraceEntry::Tool {
            action_id: "a".into(),
            request_id: Some("r".into()),
            call_id: "c".into(),
            name: name.into(),
            arguments: args.into(),
            result: result.map(str::to_string),
        }
    }

    fn trace(unanswered: bool, label: &str, entries: Vec<TraceEntry>) -> PostTrace {
        PostTrace {
            anchor_action_id: "anchor".into(),
            participant_label: label.into(),
            unanswered,
            entries,
        }
    }

    #[test]
    fn an_answered_turn_summarizes_its_rounds_without_a_byline() {
        let t = trace(
            false,
            "Gemma",
            vec![
                tool("read_thread", "{}", Some("4 posts")),
                tool("list_branches", "{}", Some("2 branches")),
            ],
        );
        assert_eq!(summary_line(&[t]), "2 tool calls");
    }

    #[test]
    fn one_call_is_singular() {
        let t = trace(false, "Gemma", vec![tool("echo", "{}", Some("hi"))]);
        assert_eq!(summary_line(&[t]), "1 tool call");
    }

    #[test]
    fn a_decline_names_the_agent_that_declined() {
        // The gap case: the disclosure hangs under somebody else's post, so
        // the line has to say whose non-event it is.
        let t = trace(
            true,
            "Gemma",
            vec![TraceEntry::Declined {
                action_id: "d".into(),
                reason: Some("not my area".into()),
            }],
        );
        assert_eq!(summary_line(&[t]), "Gemma — declined to respond");
    }

    #[test]
    fn a_turn_that_ran_tools_and_left_no_post_says_so() {
        let t = trace(true, "Gemma", vec![tool("read_post", "{}", Some("ok"))]);
        assert_eq!(summary_line(&[t]), "Gemma — 1 tool call · no response");
    }

    #[test]
    fn an_unexecuted_round_reads_as_not_run() {
        assert_eq!(tool_detail("{\"a\":1}", None), "{\"a\":1} — not run");
        assert_eq!(tool_detail("", None), "not run");
    }

    #[test]
    fn a_round_shows_arguments_then_result() {
        assert_eq!(
            tool_detail("{\"h\":\"x\"}", Some("4 posts")),
            "{\"h\":\"x\"} → 4 posts"
        );
    }

    #[test]
    fn a_no_argument_call_prints_only_its_result() {
        assert_eq!(tool_detail("{}", Some("2 branches")), "→ 2 branches");
        assert_eq!(tool_detail("  {} ", None), "not run");
    }

    #[test]
    fn summaries_flatten_and_elide() {
        assert_eq!(summarize("a\n\n  b\tc"), "a b c");
        let long = "x".repeat(400);
        let s = summarize(&long);
        assert!(s.ends_with('…'));
        assert_eq!(s.chars().count(), SUMMARY_CHARS + 1);
        // Char-boundary safe on multi-byte input.
        let wide = "é".repeat(400);
        assert!(summarize(&wide).ends_with('…'));
    }
}
