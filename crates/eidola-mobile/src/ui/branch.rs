//! Which posts a conversation shows: one branch of the reply tree.
//!
//! From the first post, at each post with several replies the reply the reader
//! chose (else the newest), to the end, and the answers streaming to whatever
//! the branch ends with. **The reader's posts not yet read back are part of the
//! tree**: each saved-but-unread post is a node under the post it replies to
//! (which may itself be one of them), so a chain of them, written faster than
//! the reads that show them, stays on the branch in order, and the branch's tip
//! is the last of them. An unparented row continues from the row before it
//! (app-core's linear tail), as on desktop. Pure, so its rules are tested
//! without a window.

use std::collections::HashMap;

use gpui::SharedString;

use super::conversation::{Pending, Post, Turn};

/// A row's identity across rebuilds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RowKey {
    Post(String),
    Pending(u64),
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
        /// Which of the conversation's saves this is (its row's identity).
        seq: u64,
        text: SharedString,
        created_at: i64,
        /// The post's id once its save answered (it is persisted, though no
        /// read has shown it yet).
        action_id: Option<String>,
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
            Row::Pending { seq, .. } => RowKey::Pending(*seq),
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

/// The post the next message on this branch replies to: the last post the
/// branch shows, **the reader's own saved post included** — once its save has
/// answered, its id is the branch's tip whether or not a read has shown it
/// yet. A streaming answer is not a post yet and is passed over; a post still
/// saving has no id (and a second save waits for it).
pub fn tip(rows: &[Row]) -> Option<String> {
    for row in rows.iter().rev() {
        match row {
            Row::Post { post, .. } => return Some(post.action_id.clone()),
            Row::Pending { action_id, .. } => return action_id.clone(),
            Row::Turn { .. } => {}
        }
    }
    None
}

/// Keep the overlays a read has not yet shown: one whose post `posts` holds
/// is done; one still saving, or saved and not read back, stays. A failed read
/// never calls this, so it keeps them all.
pub fn retain_unread(pending: &mut Vec<Pending>, posts: &[Post]) {
    pending.retain(|p| {
        !p.action_id
            .as_deref()
            .is_some_and(|id| posts.iter().any(|post| post.action_id == id))
    });
}

/// A node of the tree a branch is walked through: a persisted post, or one of
/// the reader's posts not read back yet.
enum Node<'a> {
    Post(usize),
    Overlay(&'a Pending),
}

/// The rows on screen. `chosen` maps a parent to the reply the reader chose
/// there; `label` names a participant for a streaming answer.
pub fn visible(
    posts: &[Post],
    chosen: &HashMap<Option<String>, String>,
    pending: &[Pending],
    turns: &[Turn],
    label: impl Fn(&str) -> Option<SharedString>,
) -> Vec<Row> {
    let mut children: HashMap<Option<String>, Vec<Node>> = HashMap::new();
    for i in 0..posts.len() {
        children
            .entry(parent_of(posts, i))
            .or_default()
            .push(Node::Post(i));
    }
    // After the posts, so among a post's replies an overlay is the newest; one
    // a read already holds is shown as that post instead.
    for p in pending {
        let read = p
            .action_id
            .as_deref()
            .is_some_and(|id| posts.iter().any(|post| post.action_id == id));
        if !read {
            children
                .entry(p.reply_to.clone())
                .or_default()
                .push(Node::Overlay(p));
        }
    }
    let id_of = |node: &Node| match node {
        Node::Post(i) => Some(posts[*i].action_id.clone()),
        Node::Overlay(p) => p.action_id.clone(),
    };

    let mut rows = Vec::new();
    let mut cursor: Option<String> = None;
    loop {
        let kids = children.get(&cursor).map(Vec::as_slice).unwrap_or(&[]);
        if kids.is_empty() {
            break;
        }
        // A post still saving is where the reader just wrote: it wins. Else
        // the reader's choice here, else the newest.
        let pick = kids
            .iter()
            .position(|k| matches!(k, Node::Overlay(p) if p.action_id.is_none()))
            .or_else(|| {
                let chosen = chosen.get(&cursor)?;
                kids.iter().position(|k| id_of(k).as_ref() == Some(chosen))
            })
            .unwrap_or(kids.len() - 1);
        match &kids[pick] {
            Node::Post(i) => {
                let post = posts[*i].clone();
                // The switch counts persisted replies (what `replies` lists).
                let persisted: Vec<usize> = kids
                    .iter()
                    .filter_map(|k| match k {
                        Node::Post(j) => Some(*j),
                        Node::Overlay(_) => None,
                    })
                    .collect();
                let fork = (persisted.len() > 1).then(|| {
                    let at = persisted.iter().position(|j| j == i).unwrap_or(0);
                    (at, persisted.len(), cursor.clone())
                });
                cursor = Some(post.action_id.clone());
                rows.push(Row::Post { post, fork });
            }
            Node::Overlay(p) => {
                rows.push(Row::Pending {
                    seq: p.seq,
                    text: p.text.clone(),
                    created_at: p.created_at,
                    action_id: p.action_id.clone(),
                });
                match &p.action_id {
                    Some(id) => cursor = Some(id.clone()),
                    // Nothing can hang under a post with no id yet.
                    None => {
                        cursor = None;
                        break;
                    }
                }
            }
        }
    }
    if let Some(end) = cursor {
        for turn in turns.iter().filter(|t| t.target == end) {
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
                Row::Pending { action_id, .. } => {
                    format!("pending:{}", action_id.as_deref().unwrap_or("-"))
                }
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
        let rows = visible(&posts, &HashMap::new(), &[], &[], none);
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
        let rows = visible(&posts, &HashMap::new(), &[], &[], none);
        assert_eq!(keys(&rows), ["q1", "q2[2/2]", "a2"]);

        let chosen = HashMap::from([(Some("q1".to_string()), "a1".to_string())]);
        let rows = visible(&posts, &chosen, &[], &[], none);
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
            seq: 0,
            text: "q2".into(),
            reply_to: Some("q1".into()),
            action_id: None,
            created_at: 2,
        };
        let rows = visible(
            &posts,
            &HashMap::new(),
            std::slice::from_ref(&pending),
            &[],
            none,
        );
        assert_eq!(keys(&rows), ["q1", "pending:-"]);

        // Saved, its id known and chosen, not read back yet: still the overlay,
        // and the answer streaming to it follows it.
        pending.action_id = Some("q2".into());
        let chosen = HashMap::from([(Some("q1".to_string()), "q2".to_string())]);
        let rows = visible(
            &posts,
            &chosen,
            std::slice::from_ref(&pending),
            &[turn(0, "q2")],
            none,
        );
        assert_eq!(keys(&rows), ["q1", "pending:q2", "turn0"]);

        // Read back: the post itself, and no overlay.
        let read = [
            post("q1", None),
            post("a1", Some("q1")),
            post("q2", Some("q1")),
        ];
        let rows = visible(&read, &chosen, std::slice::from_ref(&pending), &[], none);
        assert_eq!(keys(&rows), ["q1", "q2[2/2]"]);
    }

    #[test]
    fn a_saved_post_not_yet_read_back_is_the_branch_tip() {
        // Q saved as a reply to P; the read that would show it has not landed.
        let posts = [post("p", None)];
        let pending = Pending {
            seq: 0,
            text: "q".into(),
            reply_to: Some("p".into()),
            action_id: Some("q".into()),
            created_at: 2,
        };
        let chosen = HashMap::from([(Some("p".to_string()), "q".to_string())]);
        let rows = visible(
            &posts,
            &chosen,
            std::slice::from_ref(&pending),
            &[turn(0, "q")],
            none,
        );
        assert_eq!(keys(&rows), ["p", "pending:q", "turn0"]);
        // The next message answers Q, not P.
        assert_eq!(tip(&rows).as_deref(), Some("q"));
        // Before its save answers there is no tip to give.
        let unsaved = Pending {
            action_id: None,
            ..pending
        };
        let rows = visible(
            &posts,
            &HashMap::new(),
            std::slice::from_ref(&unsaved),
            &[],
            none,
        );
        assert_eq!(tip(&rows), None);
        // Read back, the post itself is the tip.
        let read = [post("p", None), post("q", Some("p"))];
        let rows = visible(&read, &chosen, &[], &[], none);
        assert_eq!(tip(&rows).as_deref(), Some("q"));
    }

    fn saved(seq: u64, id: &str, reply_to: &str) -> Pending {
        Pending {
            seq,
            text: id.to_string().into(),
            reply_to: Some(reply_to.into()),
            action_id: Some(id.into()),
            created_at: 2,
        }
    }

    /// Posts written faster than the reads that show them: each saved one
    /// stays on the branch, in order, and the last is the tip.
    #[test]
    fn a_chain_of_unread_posts_stays_on_the_branch_in_order() {
        let posts = [post("p", None), post("a", Some("p"))];
        let chain = [saved(0, "q", "a"), saved(1, "r", "q")];
        let rows = visible(&posts, &HashMap::new(), &chain, &[], none);
        assert_eq!(keys(&rows), ["p", "a", "pending:q", "pending:r"]);
        assert_eq!(tip(&rows).as_deref(), Some("r"));

        // A third, still saving: shown last, and no tip until it answers.
        let mut three = chain.to_vec();
        three.push(Pending {
            seq: 2,
            text: "s".into(),
            reply_to: Some("r".into()),
            action_id: None,
            created_at: 3,
        });
        let rows = visible(&posts, &HashMap::new(), &three, &[turn(0, "r")], none);
        assert_eq!(
            keys(&rows),
            ["p", "a", "pending:q", "pending:r", "pending:-"]
        );
        assert_eq!(tip(&rows), None);
        three[2].action_id = Some("s".into());
        let rows = visible(&posts, &HashMap::new(), &three, &[], none);
        assert_eq!(tip(&rows).as_deref(), Some("s"));
    }

    /// A read drops exactly the overlays it holds; the rest stay, and the
    /// branch runs through the read posts into them.
    #[test]
    fn a_partial_read_back_drops_only_what_it_reconciles() {
        let mut chain = vec![saved(0, "q", "p"), saved(1, "r", "q")];
        // A failed read reconciles nothing (and never calls this): unchanged.
        retain_unread(&mut chain, &[post("p", None)]);
        assert_eq!(chain.len(), 2);
        // A read that holds Q but not yet R.
        let read = [post("p", None), post("q", Some("p"))];
        retain_unread(&mut chain, &read);
        assert_eq!(chain.iter().map(|p| p.seq).collect::<Vec<_>>(), [1]);
        let rows = visible(&read, &HashMap::new(), &chain, &[], none);
        assert_eq!(keys(&rows), ["p", "q", "pending:r"]);
        assert_eq!(tip(&rows).as_deref(), Some("r"));
        // And one that holds both.
        let read = [post("p", None), post("q", Some("p")), post("r", Some("q"))];
        retain_unread(&mut chain, &read);
        assert!(chain.is_empty());
    }

    #[test]
    fn a_streaming_answer_shows_only_at_the_end_of_the_branch_it_answers() {
        let posts = [
            post("q1", None),
            post("a1", Some("q1")),
            post("q2", Some("q1")),
        ];
        let turns = [turn(0, "q2"), turn(1, "a1")];
        let rows = visible(&posts, &HashMap::new(), &[], &turns, |_| {
            Some("Agent".into())
        });
        assert_eq!(keys(&rows), ["q1", "q2[2/2]", "turn0"]);
        assert!(matches!(&rows[2], Row::Turn { author: Some(a), .. } if a == "Agent"));
    }
}
