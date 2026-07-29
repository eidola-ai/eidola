//! The space view's UI tree — a minimal, render-shaped projection of the
//! conversation, plus the *pure* structural helpers over it.
//!
//! `Space` loads the conversation as a flat `Vec<ChatMessageView>` in the
//! flattener's pre-order (spine first, then branches), each row carrying its
//! `parent_action_id`. [`build_tree`] relinks that flat list into a navigable
//! tree of [`TreeNode`]s — the shape the recursively-nested scrollers render.
//!
//! Nodes reference their content **by index** into the transcript slice
//! ([`NodeSrc::Msg`]) rather than owning a copy, so rebuilding the tree each
//! frame is cheap integer work (no string clones); the body text is read from
//! the slice at render time. Two synthetic sources — [`NodeSrc::Streaming`] and
//! [`NodeSrc::Draft`] — are appended as a leaf's child by the view to represent
//! the in-flight reply and the active composer.
//!
//! Everything here is pure and unit-tested: the runtime-dependent parts
//! (which sibling a scroller rests on, measured heights) live in
//! [`super::layout`] and the view, which thread a selection/height accessor in.

use eidola_app_core::PostReference;
use gpui::SharedString;

use crate::space::PostBlockSpan;

/// A per-row render snapshot — the minimal, UI-shaped projection of one
/// transcript row. Built once when the transcript changes (not per frame), so
/// the tree and the render path work over cheap `SharedString` clones
/// (refcounted) rather than re-cloning the full message content every frame.
///
/// Only fields the space view actually renders are carried; the rest of
/// `ChatMessageView` (and the DB row behind it) is deliberately left out.
#[derive(Clone, Debug)]
pub struct PostData {
    /// The post's persisted action id, or `None` for an optimistic row.
    pub action_id: Option<SharedString>,
    /// The post's **item** id — stable across every generation of the post
    /// (an edit or a regenerate appends a generation of the same item). This is
    /// what survives an edit, and is what a stale action id is resolved
    /// *through* when a window-local reference to a post outlives the
    /// generation it named (see `SpaceView::rethread_drafts`).
    pub item_id: Option<SharedString>,
    /// The structural reply antecedent, used to relink the flat list.
    pub parent_action_id: Option<SharedString>,
    /// `user` / `assistant` / `error`.
    pub role: SharedString,
    /// The gutter byline: "You" / a model's human display name / "Error".
    pub byline: SharedString,
    /// The serving backend's human display name (assistant rows only) —
    /// the quiet second byline line ("Gemma 4 E2B" over "Local").
    pub byline_backend: Option<SharedString>,
    /// Formatted clock time for the byline, empty when there's no timestamp.
    pub time: SharedString,
    /// The post body as markdown source.
    pub content: SharedString,
    /// The model that produced an inference row (`None` for human rows).
    /// Regenerate re-asks the post's own recorded model.
    pub model: Option<SharedString>,
    /// Total generations of this item (`> 1` shows a `vN` badge).
    pub generation_count: i64,
    /// Captured reasoning for an assistant turn (ephemeral disclosure), if any.
    pub reasoning: Option<SharedString>,
    /// Whether the reasoning disclosure is open.
    pub reasoning_expanded: bool,
    /// The post's quoted references (`reference` edges, ordinals `1..`) — the
    /// embed map's source and the footnote rail's rows.
    pub references: Vec<PostReference>,
    /// Content-block spans within `content` (the selection→quote mapping and
    /// the incoming-highlight range mapping).
    pub blocks: Vec<PostBlockSpan>,
}

/// What a [`TreeNode`] renders from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeSrc {
    /// A persisted (or optimistic) transcript row, by index into the
    /// `Space`'s message slice.
    Msg(usize),
    /// One live in-flight response turn, by its `StreamingTurn::seq`. Several
    /// can render at once (a notification fan-out), each attached at its own
    /// target post.
    Streaming(u64),
    /// The active composer draft (window-local).
    Draft,
}

/// One node of the render tree: where its content comes from, a stable id (the
/// key for its editor state and horizontal scroll handle), and its replies.
#[derive(Clone, Debug)]
pub struct TreeNode {
    pub src: NodeSrc,
    /// Stable across frames: the post's `action_id`, or a positional fallback
    /// for an optimistic row that has no id yet, or a synthetic sentinel for
    /// the streaming/draft overlays.
    pub id: SharedString,
    /// Replies, spine-first (the flattener's order) — the first child is the
    /// spine continuation, later children are branches.
    pub children: Vec<TreeNode>,
}

/// Prefix of every streaming-overlay leaf id (see [`streaming_node_id`]).
pub const STREAMING_ID_PREFIX: &str = "::streaming-";
/// The stable node id for the in-flight turn `seq` — the key for its height
/// cache entry and body editor.
pub fn streaming_node_id(seq: u64) -> SharedString {
    SharedString::from(format!("{STREAMING_ID_PREFIX}{seq}"))
}
/// Sentinel id for the active draft/composer overlay leaf.
pub const DRAFT_ID: &str = "::draft";
/// Synthetic id for the implicit top-level scroller over multiple thread roots.
pub const ROOT_SCROLLER_ID: &str = "::root";

impl TreeNode {
    /// A leaf with no children.
    pub fn leaf(src: NodeSrc, id: impl Into<SharedString>) -> Self {
        Self {
            src,
            id: id.into(),
            children: Vec::new(),
        }
    }
}

/// The stable node id for transcript row `i`: its `action_id` when persisted,
/// else a positional fallback (`idx-{i}`) stable for the row's lifetime in the
/// current transcript (an optimistic user turn before its reload assigns an id).
pub fn node_id(posts: &[PostData], i: usize) -> SharedString {
    posts[i]
        .action_id
        .clone()
        .unwrap_or_else(|| SharedString::from(format!("idx-{i}")))
}

/// Relink the flat transcript into a tree of [`TreeNode`]s by `parent_action_id`.
/// Rows whose parent is absent (or not present in this slice) are thread roots.
/// Children keep the flat slice's order, which the flattener already emits
/// spine-first.
pub fn build_tree(posts: &[PostData]) -> Vec<TreeNode> {
    let n = posts.len();
    // action_id -> row index, for parent resolution.
    let mut by_action: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::with_capacity(n);
    for (i, m) in posts.iter().enumerate() {
        if let Some(a) = m.action_id.as_deref() {
            by_action.insert(a, i);
        }
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut roots: Vec<usize> = Vec::new();
    // Chain an *unparented* row (an optimistic user turn, a synthetic test row,
    // or any row whose parent isn't resolvable) onto the immediately preceding
    // row, so the conversation forms one spine even when only some rows carry
    // links. A real `get_space_tree` result has exactly one unparented row (the
    // thread root); every other row is explicitly parented, so this only affects
    // synthetic/optimistic rows.
    let mut last: Option<usize> = None;
    for (i, m) in posts.iter().enumerate() {
        match m
            .parent_action_id
            .as_deref()
            .and_then(|p| by_action.get(p).copied())
        {
            // Guard against a self/forward reference producing a cycle: only
            // attach to a strictly-earlier row (the flat list is causal order).
            Some(p) if p < i => children[p].push(i),
            _ => match last {
                Some(prev) => children[prev].push(i),
                None => roots.push(i),
            },
        }
        last = Some(i);
    }

    roots
        .into_iter()
        .map(|r| build_node(r, posts, &children))
        .collect()
}

fn build_node(i: usize, posts: &[PostData], children: &[Vec<usize>]) -> TreeNode {
    TreeNode {
        src: NodeSrc::Msg(i),
        id: node_id(posts, i),
        children: children[i]
            .iter()
            .map(|&c| build_node(c, posts, children))
            .collect(),
    }
}

/// Depth-first search for the node with `id` (immutable).
pub fn node_ref<'a>(roots: &'a [TreeNode], id: &str) -> Option<&'a TreeNode> {
    for node in roots {
        if node.id == id {
            return Some(node);
        }
        if let Some(found) = node_ref(&node.children, id) {
            return Some(found);
        }
    }
    None
}

/// The id of `id`'s parent node, or `None` if `id` is a root or absent.
pub fn parent_of(roots: &[TreeNode], id: &str) -> Option<SharedString> {
    for node in roots {
        if node.children.iter().any(|c| c.id == id) {
            return Some(node.id.clone());
        }
        if let Some(found) = parent_of(&node.children, id) {
            return Some(found);
        }
    }
    None
}

/// Ids from the containing root down to `target` (inclusive), or `None` if
/// `target` isn't in the forest. Each adjacent pair is a (parent, selected
/// child) step.
pub fn path_ids(roots: &[TreeNode], target: &str) -> Option<Vec<SharedString>> {
    roots.iter().find_map(|node| path_within(node, target))
}

/// Ids from `node` down to `target` (inclusive), or `None` if `target` is not
/// in `node`'s subtree.
fn path_within(node: &TreeNode, target: &str) -> Option<Vec<SharedString>> {
    if node.id == target {
        return Some(vec![node.id.clone()]);
    }
    for child in &node.children {
        if let Some(mut sub) = path_within(child, target) {
            sub.insert(0, node.id.clone());
            return Some(sub);
        }
    }
    None
}

/// Append `overlay` (a streaming/draft leaf) as the last child of the node with
/// id `parent_id`, in place. Returns whether a parent was found. When the forest
/// is empty (a blank space), the caller pushes the overlay as a lone root.
pub fn attach_overlay(roots: &mut [TreeNode], parent_id: &str, overlay: TreeNode) -> bool {
    for node in roots.iter_mut() {
        if node.id == parent_id {
            node.children.push(overlay);
            return true;
        }
        if attach_overlay(&mut node.children, parent_id, overlay.clone()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal transcript row: only the fields `build_tree` reads.
    fn row(action_id: &str, parent: Option<&str>, kind: &str, text: &str) -> PostData {
        PostData {
            action_id: Some(action_id.into()),
            item_id: Some(format!("item-{action_id}").into()),
            parent_action_id: parent.map(SharedString::from),
            role: if kind == "agent" { "assistant" } else { "user" }.into(),
            byline: if kind == "agent" {
                "kimi".into()
            } else {
                "You".into()
            },
            byline_backend: (kind == "agent").then(|| "Eidola".into()),
            time: "".into(),
            content: text.into(),
            model: (kind == "agent").then(|| "kimi".into()),
            generation_count: 1,
            reasoning: None,
            reasoning_expanded: false,
            references: Vec::new(),
            blocks: Vec::new(),
        }
    }

    fn optimistic(text: &str) -> PostData {
        PostData {
            action_id: None,
            item_id: None,
            parent_action_id: None,
            role: "user".into(),
            byline: "You".into(),
            byline_backend: None,
            time: "".into(),
            content: text.into(),
            model: None,
            generation_count: 1,
            reasoning: None,
            reasoning_expanded: false,
            references: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// a1 ─ a2 ─┬─ a3 ─ a4   (spine)
    ///          └─ a5        (branch)
    fn sample() -> Vec<PostData> {
        vec![
            row("a1", None, "human", "root"),
            row("a2", Some("a1"), "agent", "reply"),
            row("a3", Some("a2"), "human", "spine"),
            row("a4", Some("a3"), "agent", "spine2"),
            row("a5", Some("a2"), "human", "branch"),
        ]
    }

    #[test]
    fn builds_single_root_spine_and_branch() {
        let msgs = sample();
        let tree = build_tree(&msgs);
        assert_eq!(tree.len(), 1, "one thread root");
        let a1 = &tree[0];
        assert_eq!(a1.id, "a1");
        assert_eq!(a1.children.len(), 1);
        let a2 = &a1.children[0];
        assert_eq!(a2.id, "a2");
        // a2's children are spine-first: a3 (spine) then a5 (branch).
        assert_eq!(a2.children.len(), 2);
        assert_eq!(a2.children[0].id, "a3");
        assert_eq!(a2.children[1].id, "a5");
        // a3 continues the spine to a4.
        assert_eq!(a2.children[0].children[0].id, "a4");
    }

    #[test]
    fn optimistic_row_without_id_is_a_positional_root() {
        // A blank-space optimistic user turn: no action_id, no parent.
        let msgs = vec![optimistic("hello")];
        let tree = build_tree(&msgs);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].id, "idx-0");
        assert!(matches!(tree[0].src, NodeSrc::Msg(0)));
    }

    #[test]
    fn node_ref_parent_and_path() {
        let msgs = sample();
        let tree = build_tree(&msgs);
        assert!(node_ref(&tree, "a4").is_some());
        assert!(node_ref(&tree, "nope").is_none());
        assert_eq!(parent_of(&tree, "a5"), Some("a2".into()));
        assert_eq!(parent_of(&tree, "a1"), None);
        assert_eq!(
            path_ids(&tree, "a4"),
            Some(vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()])
        );
    }

    #[test]
    fn forward_reference_does_not_cycle() {
        // A row pointing at a later row must not form a cycle: a1's forward
        // parent is rejected, so a1 is the (first unparented) root; a2 is
        // unparented and chains onto the previous row (a1).
        let msgs = vec![
            row("a1", Some("a2"), "human", "forward"),
            row("a2", None, "human", "later"),
        ];
        let tree = build_tree(&msgs);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].id, "a1");
        assert_eq!(tree[0].children[0].id, "a2");
    }

    #[test]
    fn unparented_rows_chain_into_one_spine() {
        // Synthetic/optimistic rows (no ids, no parents) form a single spine,
        // not N separate roots.
        let msgs = vec![optimistic("one"), optimistic("two"), optimistic("three")];
        let tree = build_tree(&msgs);
        assert_eq!(tree.len(), 1);
        assert_eq!(tree[0].children.len(), 1);
        assert_eq!(tree[0].children[0].children.len(), 1);
    }

    #[test]
    fn attach_overlay_at_leaf() {
        let msgs = sample();
        let mut tree = build_tree(&msgs);
        let ok = attach_overlay(&mut tree, "a4", TreeNode::leaf(NodeSrc::Draft, DRAFT_ID));
        assert!(ok);
        // A streaming turn attaches at its *target* post, so two concurrent
        // turns replying to the same post land as ordered siblings.
        let ok = attach_overlay(
            &mut tree,
            "a3",
            TreeNode::leaf(NodeSrc::Streaming(1), streaming_node_id(1)),
        );
        assert!(ok);
        let ok = attach_overlay(
            &mut tree,
            "a3",
            TreeNode::leaf(NodeSrc::Streaming(2), streaming_node_id(2)),
        );
        assert!(ok);
        let a3 = node_ref(&tree, "a3").unwrap();
        // a4 (the persisted spine child) then the two turns, in start order.
        assert_eq!(a3.children.len(), 3);
        assert_eq!(a3.children[1].id, streaming_node_id(1));
        assert_eq!(a3.children[2].id, streaming_node_id(2));
        let a4 = node_ref(&tree, "a4").unwrap();
        assert_eq!(a4.children.len(), 1);
        assert_eq!(a4.children[0].id, DRAFT_ID);
    }
}
