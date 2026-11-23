//! Which posts a conversation shows: one branch of the reply tree.
//!
//! From the first post, at each post with several replies the reply the reader
//! chose (else the newest), to the end; then the reader's post while it saves,
//! on the branch it was written to, and the answers streaming to whatever the
//! branch ends with. An unparented row continues from the row before it
//! (app-core's linear tail), as on desktop. Pure, so its rules are tested
//! without a window.

use std::collections::HashMap;

use gpui::SharedString;

use super::conversation::{Pending, Post, Turn};

/// A row's identity across rebuilds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RowKey {
    Post(String),
    Pending,
    Turn(u64),
}

/// One row of the visible branch.
#[derive(Clone, Debug)]
pub enum Row {
    Post {
        post: Post,
        /// Where this post is one of several replies: (its position, how
        /// many, the parent they share).
        fork: Option<(usize, usize, Option<String>)>,
    },
    Pending {
        text: SharedString,
        created_at: i64,
    },
    Turn {
        seq: u64,
        author: Option<SharedString>,
        reasoning: SharedString,
        content: SharedString,
    },
}

impl Row {
    pub fn key(&self) -> RowKey {
        match self {
            Row::Post { post, .. } => RowKey::Post(post.action_id.clone()),
            Row::Pending { .. } => RowKey::Pending,
            Row::Turn { seq, .. } => RowKey::Turn(*seq),
        }
    }
}

/// The parent a post hangs from: its reply edge, else the row before it.
fn parent_of(posts: &[Post], i: usize) -> Option<String> {
    posts[i]
        .parent
        .clone()
        .or_else(|| i.checked_sub(1).map(|p| posts[p].action_id.clone()))
}

/// The replies to `parent` (`None`: the conversation's first posts), in
/// order.
pub fn replies<'a>(posts: &'a [Post], parent: &Option<String>) -> Vec<&'a Post> {
    (0..posts.len())
        .filter(|&i| parent_of(posts, i) == *parent)
        .map(|i| &posts[i])
        .collect()
}

/// The rows on screen. `chosen` maps a parent to the reply the reader chose
/// there; `label` names a participant for a streaming answer.
pub fn visible(
    posts: &[Post],
    chosen: &HashMap<Option<String>, String>,
    pending: Option<&Pending>,
    turns: &[Turn],
    label: impl Fn(&str) -> Option<SharedString>,
) -> Vec<Row> {
    let mut children: HashMap<Option<String>, Vec<usize>> = HashMap::new();
    for i in 0..posts.len() {
        children.entry(parent_of(posts, i)).or_default().push(i);
    }
    let mut rows = Vec::new();
    let mut cursor: Option<String> = None;
    let mut pending_shown = false;
    loop {
        let kids = children.get(&cursor).map(Vec::as_slice).unwrap_or(&[]);
        let chosen_here = chosen.get(&cursor);
        // The reader's post, not read back yet, on the branch it was written
        // to: where nothing else is, or before its id is known, or where the
        // reader's choice is that post.
        if let Some(p) = pending
            && p.reply_to == cursor
            && (kids.is_empty()
                || p.action_id.is_none()
                || chosen_here.is_some_and(|c| p.action_id.as_ref() == Some(c)))
            && !kids
                .iter()
                .any(|&k| p.action_id.as_deref() == Some(posts[k].action_id.as_str()))
        {
            rows.push(Row::Pending {
                text: p.text.clone(),
                created_at: p.created_at,
            });
            pending_shown = true;
            cursor = p.action_id.clone();
            break;
        }
        if kids.is_empty() {
            break;
        }
        let pick = chosen_here
            .and_then(|c| kids.iter().position(|&k| posts[k].action_id == *c))
            .unwrap_or(kids.len() - 1);
        let post = posts[kids[pick]].clone();
        cursor = Some(post.action_id.clone());
        rows.push(Row::Post {
            post,
            fork: (kids.len() > 1).then(|| (pick, kids.len(), parent_of(posts, kids[pick]))),
        });
    }
    let end = cursor;
    for turn in turns {
        let answers_end = Some(&turn.target) == end.as_ref()
            || (pending_shown && pending.and_then(|p| p.action_id.as_ref()) == Some(&turn.target));
        if answers_end {
            rows.push(Row::Turn {
                seq: turn.seq,
                author: label(&turn.participant_id),
                reasoning: turn.reasoning.clone().into(),
                content: turn.content.clone().into(),
            });
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::conversation::Author;

    fn post(id: &str, parent: Option<&str>) -> Post {
        Post {
            action_id: id.into(),
            parent: parent.map(Into::into),
            author: Author::Reader,
            text: id.to_string().into(),
            reasoning: None,
            created_at: 1,
            truncated: false,
        }
    }

    fn keys(rows: &[Row]) -> Vec<String> {
        rows.iter()
            .map(|r| match r {
                Row::Post { post, fork } => match fork {
                    Some((i, n, _)) => format!("{}[{}/{}]", post.action_id, i + 1, n),
                    None => post.action_id.clone(),
                },
                Row::Pending { .. } => "pending".into(),
                Row::Turn { seq, .. } => format!("turn{seq}"),
            })
            .collect()
    }

    fn turn(seq: u64, target: &str) -> Turn {
        Turn {
            seq,
            participant_id: "agent".into(),
            target: target.into(),
            reasoning: String::new(),
            content: String::new(),
            landed: None,
        }
    }

    fn none(_: &str) -> Option<SharedString> {
        None
    }

    #[test]
    fn an_unparented_conversation_is_one_line() {
        let posts = [post("q1", None), post("a1", None), post("q2", None)];
        let rows = visible(&posts, &HashMap::new(), None, &[], none);
        assert_eq!(keys(&rows), ["q1", "a1", "q2"]);
    }

    #[test]
    fn a_fork_shows_the_newest_reply_unless_another_was_chosen() {
        // q1 has two replies: its answer a1, then a reply q2 written from it.
        let posts = [
            post("q1", None),
            post("a1", Some("q1")),
            post("q2", Some("q1")),
            post("a2", Some("q2")),
        ];
        let rows = visible(&posts, &HashMap::new(), None, &[], none);
        assert_eq!(keys(&rows), ["q1", "q2[2/2]", "a2"]);

        let chosen = HashMap::from([(Some("q1".to_string()), "a1".to_string())]);
        let rows = visible(&posts, &chosen, None, &[], none);
        assert_eq!(keys(&rows), ["q1", "a1[1/2]"]);
        assert_eq!(
            replies(&posts, &Some("q1".into()))
                .iter()
                .map(|p| p.action_id.as_str())
                .collect::<Vec<_>>(),
            ["a1", "q2"]
        );
    }

    #[test]
    fn the_readers_post_shows_on_the_branch_it_was_written_to_until_it_is_read() {
        let posts = [post("q1", None), post("a1", Some("q1"))];
        // A reply to q1 while it saves: the branch switches to it.
        let mut pending = Pending {
            text: "q2".into(),
            reply_to: Some("q1".into()),
            action_id: None,
            created_at: 2,
        };
        let rows = visible(&posts, &HashMap::new(), Some(&pending), &[], none);
        assert_eq!(keys(&rows), ["q1", "pending"]);

        // Saved, its id known and chosen, not read back yet: still the overlay,
        // and the answer streaming to it follows it.
        pending.action_id = Some("q2".into());
        let chosen = HashMap::from([(Some("q1".to_string()), "q2".to_string())]);
        let rows = visible(&posts, &chosen, Some(&pending), &[turn(0, "q2")], none);
        assert_eq!(keys(&rows), ["q1", "pending", "turn0"]);

        // Read back: the post itself, and no overlay.
        let read = [
            post("q1", None),
            post("a1", Some("q1")),
            post("q2", Some("q1")),
        ];
        let rows = visible(&read, &chosen, Some(&pending), &[], none);
        assert_eq!(keys(&rows), ["q1", "q2[2/2]"]);
    }

    #[test]
    fn a_streaming_answer_shows_only_at_the_end_of_the_branch_it_answers() {
        let posts = [
            post("q1", None),
            post("a1", Some("q1")),
            post("q2", Some("q1")),
        ];
        let turns = [turn(0, "q2"), turn(1, "a1")];
        let rows = visible(&posts, &HashMap::new(), None, &turns, |_| {
            Some("Agent".into())
        });
        assert_eq!(keys(&rows), ["q1", "q2[2/2]", "turn0"]);
        assert!(matches!(&rows[2], Row::Turn { author: Some(a), .. } if a == "Agent"));
    }
}
