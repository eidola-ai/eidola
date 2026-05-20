//! Pure state transitions: `update(state, event) -> state`.
//!
//! # Invariants the buffer always satisfies
//!
//! Every state we produce passes through `enforce_invariants` at the end of
//! `update`, so callers don't have to think about these — they just hold:
//!
//! 1. **No soft breaks.** A `\n` in the middle of content is *always* part
//!    of a paragraph break (`\n\n+` run), a hard line break (`  \n` /
//!    `\\\n`), or sits at the document edge (leading / trailing). A lone
//!    mid-content `\n` would render as a space in CommonMark and as a line
//!    break in the editor — that ambiguity breaks pixel fidelity with the
//!    chat transcript renderer, so it's not allowed to exist. If any code
//!    path produces one (typed text, paste, deletion that leaves one
//!    behind), `enforce_invariants` rewrites it into `\n\n`.
//!
//! 2. **Selection offsets sit on UTF-8 char boundaries.** `set_selection`
//!    and `move_` both snap; promotion only inserts ASCII `\n` so it
//!    can't break this.
//!
//! 3. **No cursor inside a structural `\n\n` pair.** The interior byte of
//!    a paragraph-break / empty-paragraph pair is unreachable visually
//!    (the pair renders as one row, not two) and typing there would split
//!    the pair into a stray odd-length run. After every transition,
//!    `avoid_forbidden_positions` snaps any cursor or selection endpoint
//!    that lands on the interior of a structural pair *away* from the
//!    pre-event position: forward if the cursor moved forward (or didn't
//!    move), backward if it moved back. So Right at the end of `p1` in
//!    `p1\n\np2` skips from byte 2 straight to 4 (start of p2), and Left
//!    from byte 4 jumps back to 2. Hard-break `\n`s are exempt — they're
//!    in-paragraph content, not part of a structural pair.
//!
//! # Pairs model
//!
//! The buffer treats `\n\n` as the atomic structural unit. Each Enter
//! inserts exactly `\n\n`; each empty paragraph is rendered from a pair
//! of `\n`s; smart-delete removes pairs. With this discipline the source
//! always carries an even count of `\n`s in any structural run, and the
//! "typing on a trailing empty loses a row" asymmetry disappears for
//! free: typing X at the end of `p1\n\n\n\n` (2 trailing empties)
//! produces `p1\n\n\n\nX`, which the renderer reads as paragraph break +
//! 1 empty + X — same row count as before typing.
//!
//! Hard breaks (`  \n`, `\\\n`) are exempt from the pairs discipline —
//! they're a deliberate single `\n` in mid-paragraph.
//!
//! # Implication for editing actions
//!
//! Most actions don't have to know about either invariant — they just
//! produce whatever markdown they think makes sense, and the post-pass
//! cleans up soft breaks. The only special case is `delete_backward` /
//! `delete_forward`: a generic "delete one byte" at a paragraph break
//! would leave a soft break that the post-pass would immediately
//! re-promote, so the keypress would feel like a no-op. Both handlers
//! detect "I'm in a `\n` run" and delete a *pair* (collapsing the break
//! to merge paragraphs, or removing one empty paragraph from a longer
//! run).

use unicode_segmentation::UnicodeSegmentation;

use crate::analysis::{
    self, count_line_markers, is_forbidden_position, is_in_ranges, is_soft_break,
    line_depth_ending_at, nearest_allowed_position, next_allowed_position, pair_at_end,
    pair_at_start, prev_allowed_position,
};
use crate::event::EditorEvent;
use crate::state::{EditorState, Selection};

pub fn update(state: EditorState, event: EditorEvent) -> EditorState {
    let prev_anchor = match state.selection {
        Selection::Cursor(p) => p,
        Selection::Range { anchor, .. } => anchor,
    };
    let prev_head = state.selection.head();

    let next = match event {
        EditorEvent::InsertText(text) => insert_text(state, &text),
        EditorEvent::InsertNewline => insert_newline(state),
        EditorEvent::InsertLineBreak => insert_line_break(state),

        EditorEvent::DeleteBackward => delete_backward(state),
        EditorEvent::DeleteForward => delete_forward(state),

        EditorEvent::SetSelection(sel) => set_selection(state, sel),

        EditorEvent::IncreaseListDepth => increase_list_depth(state),
        EditorEvent::DecreaseListDepth => decrease_list_depth(state),

        EditorEvent::MoveLeft => move_(state, Move::Left, false),
        EditorEvent::MoveRight => move_(state, Move::Right, false),
        EditorEvent::MoveUp => move_(state, Move::Up, false),
        EditorEvent::MoveDown => move_(state, Move::Down, false),
        EditorEvent::MoveLineStart => move_(state, Move::LineStart, false),
        EditorEvent::MoveLineEnd => move_(state, Move::LineEnd, false),
        EditorEvent::MoveDocumentStart => move_(state, Move::DocStart, false),
        EditorEvent::MoveDocumentEnd => move_(state, Move::DocEnd, false),
        EditorEvent::MoveWordLeft => move_(state, Move::WordLeft, false),
        EditorEvent::MoveWordRight => move_(state, Move::WordRight, false),

        EditorEvent::ExtendLeft => move_(state, Move::Left, true),
        EditorEvent::ExtendRight => move_(state, Move::Right, true),
        EditorEvent::ExtendUp => move_(state, Move::Up, true),
        EditorEvent::ExtendDown => move_(state, Move::Down, true),
        EditorEvent::ExtendLineStart => move_(state, Move::LineStart, true),
        EditorEvent::ExtendLineEnd => move_(state, Move::LineEnd, true),
        EditorEvent::ExtendDocumentStart => move_(state, Move::DocStart, true),
        EditorEvent::ExtendDocumentEnd => move_(state, Move::DocEnd, true),
        EditorEvent::ExtendWordLeft => move_(state, Move::WordLeft, true),
        EditorEvent::ExtendWordRight => move_(state, Move::WordRight, true),

        EditorEvent::DeleteWordBackward => delete_word_backward(state),
        EditorEvent::DeleteWordForward => delete_word_forward(state),
        EditorEvent::DeleteToLineStart => delete_to_line_start(state),
        EditorEvent::DeleteToLineEnd => delete_to_line_end(state),
    };
    let next = enforce_invariants(next);
    avoid_forbidden_positions(next, prev_anchor, prev_head)
}

/// Promote any lone, mid-content `\n` into `\n\n` so the buffer never
/// contains a soft break, normalize every blockquote `>` marker to
/// `> `, canonicalize list source (tight separators, hard-break
/// continuations, marker-width indent on continuation lines), and
/// collapse two consecutive hard breaks into a paragraph break.
/// Idempotent and cheap on already-clean states.
pub fn enforce_invariants(state: EditorState) -> EditorState {
    // Pass order:
    //
    // 1. `collapse_consecutive_hard_breaks` first, because a
    //    user-produced "Shift+Enter Shift+Enter" reaches its
    //    canonical paragraph-break form (`\n\n`) here. The list
    //    normalizer then sees the clean shape, not a paragraph
    //    break to be re-promoted as a hard break.
    //
    // 2. `promote_soft_breaks` *before* `normalize_lists`. A lone
    //    mid-content `\n` at a paragraph→list (or list→paragraph,
    //    or paragraph→paragraph) boundary needs to become `\n\n`
    //    before pulldown's parse will reflect the structural
    //    separation correctly. Running normalize_lists first on
    //    the un-promoted buffer produces a parse where the
    //    boundary block is misclassified (e.g. a list that should
    //    be standalone is folded into the prior paragraph as
    //    lazy continuation), so the list normalizer's per-list
    //    rules — including the split-orphan renumber heuristic —
    //    don't fire on the right structure.
    //
    // 3. `normalize_lists` last among list-affecting passes, on
    //    the canonicalized buffer.
    //
    // 4. `normalize_blockquote_prefixes` is independent and runs
    //    after the others.
    //
    // **Parse-tree cache.** Each pass that needs the parse tree
    // calls `cache.tree(&state.markdown)`; passes that mutate the
    // buffer call `cache.invalidate()` so the next pass parses
    // fresh. Without this cache, the canonical (no-mutation) path
    // re-parses the buffer four times per keystroke, and
    // `promote_soft_breaks`'s per-byte `enclosing_containers_at`
    // call inside fence content can re-parse hundreds of times per
    // keystroke. The cache collapses both to a single parse on the
    // common path.
    let mut cache = ParseCache::new();
    let state = collapse_consecutive_hard_breaks(state); // doesn't parse
    let state = dedupe_orphan_fence_closer(state, &mut cache);
    let state = unify_fence_chain(state, &mut cache);
    let state = promote_soft_breaks(state, &mut cache);
    let state = inject_unordered_marker_space(state, &mut cache);
    let state = normalize_lists(state, &mut cache);
    normalize_blockquote_prefixes(state, &mut cache)
}

/// Inject a space after a bare unordered list marker (`-`, `*`, `+`)
/// when the next byte is non-space, non-newline, and not a repeat of
/// the marker character.
///
/// Rationale: CommonMark requires `<marker><space>` to open a list,
/// but typing `-foo` is a clear list intent that the editor can
/// salvage by injecting the missing space. The repeat-character
/// exception preserves thematic-break candidates (`---`, `***`,
/// `___`) and emphasis runs of `*`/`*` from being prematurely
/// converted into list items.
///
/// Cursor guard: skip injection when the cursor would land at the
/// would-be insertion point (the user is mid-typing — either they
/// just typed the marker and are about to continue, or they're
/// editing the gap directly). Without this, every `-` keystroke
/// would race the next keystroke for the cursor's column.
///
/// Scope guards:
///
/// - Lines whose chain has a `ListItem` are skipped — they're
///   continuation / content of an existing item, where `-foo`
///   means literal text rather than a sibling marker.
/// - Lines inside a verbatim region (fenced code or block-level
///   `$$..$$` math) are skipped — verbatim content is literal source.
/// - Lines starting with extra leading whitespace beyond the chain
///   prefix are skipped — pulldown would see them as code-block
///   indent or lazy continuation, not a fresh list opener.
///
/// The pass runs *before* `normalize_lists` so the canonicalization
/// step sees the just-injected space as part of a regular list.
fn inject_unordered_marker_space(state: EditorState, cache: &mut ParseCache) -> EditorState {
    let inserts: Vec<usize> = {
        let bytes = state.markdown.as_bytes();
        let cursor_head = state.selection.head();
        let cursor_anchor = state.selection.anchor();
        let tree = cache.tree(&state.markdown);
        let fences = analysis::fenced_code_blocks_in_tree(tree, bytes);
        let math_blocks = analysis::display_math_blocks_in_tree(tree, bytes);
        let mut out = Vec::new();
        let mut p = 0usize;
        while p <= bytes.len() {
            let line_start = p;
            // Find the chain at this line's start byte. An empty
            // chain means top-level. A chain containing `ListItem`
            // means we're inside an existing item's source range —
            // skip injection (it'd flip a continuation line into a
            // nested list opener).
            let chain = analysis::enclosing_containers_at_in_tree(tree, bytes, line_start);
            let in_li = chain
                .iter()
                .any(|c| matches!(c, analysis::EnclosingContainer::ListItem(_)));
            if !in_li {
                let prefix_len = analysis::chain_continuation_prefix_bytes(&chain);
                let content_start = line_start + prefix_len;
                if content_start < bytes.len()
                    && !analysis::is_in_verbatim_region_blocks(&fences, &math_blocks, content_start)
                {
                    let c = bytes[content_start];
                    if matches!(c, b'-' | b'*' | b'+') {
                        let next = content_start + 1;
                        if next < bytes.len() {
                            let nb = bytes[next];
                            let cursor_at_gap = cursor_head == next || cursor_anchor == next;
                            if nb != b' ' && nb != b'\n' && nb != c && !cursor_at_gap {
                                out.push(next);
                            }
                        }
                    }
                }
            }
            // Advance to the next line.
            while p < bytes.len() && bytes[p] != b'\n' {
                p += 1;
            }
            if p < bytes.len() {
                p += 1;
            } else {
                break;
            }
        }
        out
    };

    if inserts.is_empty() {
        return state;
    }
    cache.invalidate();
    let mut new_md = String::with_capacity(state.markdown.len() + inserts.len());
    let mut last = 0;
    for &pos in &inserts {
        new_md.push_str(&state.markdown[last..pos]);
        new_md.push(' ');
        last = pos;
    }
    new_md.push_str(&state.markdown[last..]);
    let map = |off: usize| -> usize {
        let shift = inserts.iter().filter(|&&pos| pos <= off).count();
        off + shift
    };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };
    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Propagate a fence's opener-line chain to its body and closer lines.
///
/// When the user types a chain marker (e.g. `>` or `> `) at the start
/// of a fence's opener line, pulldown sometimes reads the opener as
/// being in a deeper chain than the body and closer — the body lacks
/// the BQ marker and parses as either lazy continuation or a separate
/// top-level construct. This pass detects two adjacent unterminated
/// fences with matching fence chars where the first sits in a BQ
/// chain and the second doesn't, and inserts the missing chain prefix
/// on every line between them so the two halves merge into a single
/// terminated BQ-wrapped fence.
///
/// Mirrors the user-rule "context changes to the first line of the
/// code block apply to the entire code block" — applied during
/// `enforce_invariants` so the user sees the fully-nested fence
/// immediately after one keystroke.
fn unify_fence_chain(state: EditorState, cache: &mut ParseCache) -> EditorState {
    let inserts = {
        let tree = cache.tree(&state.markdown);
        let mut blocks: Vec<&crate::syntax::SyntaxNode> = Vec::new();
        collect_code_blocks(tree, &mut blocks);
        let bytes = state.markdown.as_bytes();
        let mut inserts: Vec<(usize, String)> = Vec::new();
        for pair in blocks.windows(2) {
            let (first, second) = (pair[0], pair[1]);
            let (first_delims, _) = match &first.kind {
                crate::syntax::NodeKind::CodeBlock {
                    delimiter_ranges,
                    info_string_range,
                    ..
                } => (delimiter_ranges, info_string_range.as_ref()),
                _ => continue,
            };
            let (second_delims, second_info) = match &second.kind {
                crate::syntax::NodeKind::CodeBlock {
                    delimiter_ranges,
                    info_string_range,
                    ..
                } => (delimiter_ranges, info_string_range.as_ref()),
                _ => continue,
            };
            // Both must be unterminated (1 delimiter range each).
            if first_delims.len() != 1 || second_delims.len() != 1 {
                continue;
            }
            // Second must have no info string (it's the closer the
            // first was waiting for, just at the wrong scope).
            if second_info.is_some() {
                continue;
            }
            // Matching fence chars and length.
            let first_d = &first_delims[0];
            let second_d = &second_delims[0];
            let len_match = first_d.end - first_d.start == second_d.end - second_d.start;
            let chars_match = bytes.get(first_d.start) == bytes.get(second_d.start)
                && matches!(bytes.get(first_d.start), Some(b'`' | b'~'));
            if !len_match || !chars_match {
                continue;
            }
            // First's chain must be deeper than second's at the
            // opener bytes — propagate the difference.
            let first_chain = analysis::enclosing_containers_at_in_tree(tree, bytes, first_d.start);
            let second_chain =
                analysis::enclosing_containers_at_in_tree(tree, bytes, second_d.start);
            if first_chain.len() <= second_chain.len() {
                continue;
            }
            // Build the prefix to add on each in-between line: the
            // chain segment that's in `first` but not in `second`.
            // For the common case `[BQ]` vs `[]`, that's `> `.
            //
            // Conservative implementation: only handle the case where
            // the difference is BQ-only. (LI continuation indents on
            // the body lines would interact with `> ` insertion in
            // ways the simple per-line prepend doesn't get right.)
            let extra_chain = &first_chain[second_chain.len()..];
            if !extra_chain
                .iter()
                .all(|c| matches!(c, analysis::EnclosingContainer::BlockQuote { .. }))
            {
                continue;
            }
            let prefix = analysis::chain_continuation_prefix(extra_chain);
            // Walk lines from `first.range.end` through `second.range.end`,
            // inserting `prefix` at each line's start.
            let mut p = first.range.end;
            let limit = second.range.end;
            while p < limit {
                let line_start = p;
                // Skip blank-only line (no content) — pulldown
                // wouldn't re-classify it anyway, so the chain
                // continuation isn't observably needed there.
                inserts.push((line_start, prefix.clone()));
                while p < bytes.len() && bytes[p] != b'\n' {
                    p += 1;
                }
                if p < bytes.len() {
                    p += 1;
                }
            }
            // One pair per pass — the apply step rebuilds and
            // re-parses, so subsequent passes pick up the next
            // pair if any.
            break;
        }
        inserts
    };

    if inserts.is_empty() {
        return state;
    }
    cache.invalidate();
    let total_added: usize = inserts.iter().map(|(_, s)| s.len()).sum();
    let mut new_md = String::with_capacity(state.markdown.len() + total_added);
    let mut last = 0;
    for (pos, ins) in &inserts {
        new_md.push_str(&state.markdown[last..*pos]);
        new_md.push_str(ins);
        last = *pos;
    }
    new_md.push_str(&state.markdown[last..]);
    let map = |off: usize| -> usize {
        let mut shift = 0usize;
        for (pos, ins) in &inserts {
            if *pos < off {
                shift += ins.len();
            }
        }
        off + shift
    };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };
    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Lazy parse-tree cache shared across `enforce_invariants` passes.
/// Each pass calls [`tree`](Self::tree) on the cache, parses the
/// current buffer if the cache is empty, and returns the cached
/// tree on every subsequent call. A pass that mutates the buffer
/// calls [`invalidate`](Self::invalidate) so the next pass parses
/// the new buffer.
struct ParseCache {
    tree: Option<Vec<crate::syntax::SyntaxNode>>,
}

impl ParseCache {
    fn new() -> Self {
        Self { tree: None }
    }

    fn tree(&mut self, markdown: &str) -> &[crate::syntax::SyntaxNode] {
        self.tree
            .get_or_insert_with(|| crate::parser::parse(markdown))
    }

    fn invalidate(&mut self) {
        self.tree = None;
    }
}

/// Collapse a `[terminated fence] + [identical-line unterminated fence]`
/// pair into a single terminated fence.
///
/// The shape we detect is the artifact of `auto_close_fence_edit`
/// followed by the user typing their own closing fence themselves: the
/// auto-close left a closer one line below the user's body row, the
/// user's typed `\`\`\`` closed the *original* opener, and the
/// auto-close's row became an unterminated *opener* on the line
/// immediately after. Both lines have identical content (same prefix,
/// same fence char, same length).
///
/// This pass deletes the orphan opener line so the construct ends
/// cleanly. Without it, every Backspace through the orphan triggers
/// missing-prefix-repair against an ever-deepening BQ chain (see
/// `bugs.md::backspace_oscillates_inside_corrupted_chain_in_fenced_code`),
/// because pulldown counts each `>` on the orphan-rich tail as opening
/// another BQ scope.
///
/// Restricted to the *exact* shape:
///
/// - First range terminated, second range unterminated.
/// - Second range starts immediately after the first's trailing `\n`
///   (no bytes between).
/// - Second range covers exactly one line (no embedded `\n`) — i.e.
///   the orphan opener has no body. This is the only case where
///   removing it is safe; if the user has typed body content under
///   the orphan, the construct is no longer "just an artifact" and
///   we leave it alone.
/// - The two delimiter lines have the *same fence character and same
///   fence length* and the orphan opener carries no info string. The
///   leading prefix on each line (BQ markers, LI continuation indent)
///   is allowed to differ — `enforce_invariants`'s normalize passes
///   sometimes rewrite an auto-close orphan's prefix to a shallower
///   chain depth than the user's manually-typed closer, so a
///   byte-for-byte match misses the deeper-chain case (see
///   `bugs.md::auto_close_fence_orphan_after_user_typed_closer`).
fn dedupe_orphan_fence_closer(state: EditorState, cache: &mut ParseCache) -> EditorState {
    let bytes = state.markdown.as_bytes();
    // Walk pulldown's parse tree (not the byte scanner) so we see fences
    // that sit inside LI continuation indent — `count_line_markers` only
    // tolerates 3 spaces between consecutive `>` markers, so a fence at
    // chain `[BQ, LI, LI, BQ]` (with 5+ spaces of LI indent before the
    // inner `> `) is invisible to the scanner but visible to pulldown.
    let tree = cache.tree(&state.markdown);
    let mut code_blocks: Vec<&crate::syntax::SyntaxNode> = Vec::new();
    collect_code_blocks(tree, &mut code_blocks);

    let mut delete_range: Option<std::ops::Range<usize>> = None;
    for pair in code_blocks.windows(2) {
        let (first, second) = (pair[0], pair[1]);
        let (first_delims, _) = match &first.kind {
            crate::syntax::NodeKind::CodeBlock {
                delimiter_ranges,
                info_string_range,
                ..
            } => (delimiter_ranges, info_string_range.as_ref()),
            _ => continue,
        };
        let (second_delims, second_info, second_content) = match &second.kind {
            crate::syntax::NodeKind::CodeBlock {
                delimiter_ranges,
                info_string_range,
                content_range,
                ..
            } => (delimiter_ranges, info_string_range.as_ref(), content_range),
            _ => continue,
        };
        // First must be terminated (opener + closer = two delimiters);
        // second must be unterminated (opener only).
        if first_delims.len() != 2 || second_delims.len() != 1 {
            continue;
        }
        // Second must have no info string (otherwise the user intended
        // a *new* code block with that language tag — not a duplicate
        // of the previous closer).
        if second_info.is_some() {
            continue;
        }
        // No body under the orphan: pulldown's `content_range` covers
        // any inner code lines between the opener and closer (or, for
        // an unterminated fence, between the opener and EOF). An
        // empty content_range means the orphan is just a delimiter
        // line — safe to delete. Anything inside it would be user
        // content we shouldn't silently lose.
        if second_content.start != second_content.end {
            continue;
        }
        // Line-based adjacency: the orphan's line must be the line
        // immediately after the first's closer line. We compare by
        // byte position of the `\n` that ends the closer and the
        // `\n` (or buffer start) right before the orphan.
        let first_closer = &first_delims[1];
        let closer_line_end = first_closer.end; // expected `\n` (or EOF)
        let mut orphan_line_start = second.range.start;
        while orphan_line_start > 0 && bytes[orphan_line_start - 1] != b'\n' {
            orphan_line_start -= 1;
        }
        // Adjacency: closer_line_end is the `\n` ending the closer
        // line; orphan_line_start is the byte right after that `\n`.
        if closer_line_end >= bytes.len() || bytes[closer_line_end] != b'\n' {
            continue;
        }
        if orphan_line_start != closer_line_end + 1 {
            continue;
        }
        // Same fence char + length on both delimiter ranges.
        let first_fence_char = bytes.get(first_closer.end - 1).copied();
        let second_fence_char = bytes.get(second_delims[0].start).copied();
        if first_fence_char != second_fence_char || !matches!(first_fence_char, Some(b'`' | b'~')) {
            continue;
        }
        // Count fence chars in the closer (`first_delims[1]` covers
        // leading indent + fence chars).
        let mut first_fence_len = 0usize;
        let mut p = first_closer.end;
        while p > first_closer.start && bytes[p - 1] == first_fence_char.unwrap() {
            first_fence_len += 1;
            p -= 1;
        }
        let second_fence_len = second_delims[0].end - second_delims[0].start;
        if first_fence_len != second_fence_len {
            continue;
        }
        // Delete from the leading `\n` of the orphan line through to
        // the orphan line's end (inclusive of any trailing `\n`).
        let mut orphan_line_end = second.range.end;
        while orphan_line_end < bytes.len() && bytes[orphan_line_end] != b'\n' {
            orphan_line_end += 1;
        }
        if orphan_line_end < bytes.len() && bytes[orphan_line_end] == b'\n' {
            orphan_line_end += 1;
        }
        delete_range = Some((closer_line_end + 1)..orphan_line_end);
        break;
    }
    let Some(range) = delete_range else {
        return state;
    };
    cache.invalidate();
    let mut new_md = String::with_capacity(state.markdown.len() - (range.end - range.start));
    new_md.push_str(&state.markdown[..range.start]);
    new_md.push_str(&state.markdown[range.end..]);
    let map = |off: usize| -> usize {
        if off <= range.start {
            off
        } else if off < range.end {
            range.start
        } else {
            off - (range.end - range.start)
        }
    };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };
    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Apply the list-canonicalization edits computed by
/// [`analysis::list_normalization_edits`] in a single sweep, remapping
/// the cursor through every splice. Threads the live cursor positions
/// in so per-rule guards (don't strip extra-marker-spacing while
/// the user is mid-typing in the gap, …) can fire correctly.
fn normalize_lists(state: EditorState, cache: &mut ParseCache) -> EditorState {
    let cursors: Vec<usize> = match state.selection {
        Selection::Cursor(p) => vec![p],
        Selection::Range { anchor, head } => vec![anchor, head],
    };
    let edits = {
        let tree = cache.tree(&state.markdown);
        analysis::list_normalization_edits_in_tree(tree, state.markdown.as_bytes(), &cursors)
    };
    if edits.is_empty() {
        return state;
    }
    cache.invalidate();
    apply_edits(state, &edits)
}

fn collapse_consecutive_hard_breaks(state: EditorState) -> EditorState {
    let cursors: Vec<usize> = match state.selection {
        Selection::Cursor(p) => vec![p],
        Selection::Range { anchor, head } => vec![anchor, head],
    };
    let edits = analysis::consecutive_hard_break_edits(&state.markdown, &cursors);
    if edits.is_empty() {
        return state;
    }
    apply_edits(state, &edits)
}

fn apply_edits(state: EditorState, edits: &[analysis::SourceEdit]) -> EditorState {
    // Caller-supplied invariant: edits are ordered by `range.start`
    // ascending, with insertions (zero-length ranges) preceding
    // replacements at the same start. The walker below relies on
    // this — `last` only ever moves forward — and a producer that
    // skips the sort would panic the slice operation below with
    // an unhelpful "begin <= end" message. Catch it here instead.
    // Invariant: edits are non-overlapping and ordered such that
    // each edit's range.end <= the next's range.start. Equality is
    // allowed (an insertion at position N can immediately precede a
    // replacement starting at N).
    debug_assert!(
        edits
            .windows(2)
            .all(|w| { w[0].range.end <= w[1].range.start }),
        "edits passed to apply_edits must be non-overlapping and \
         sorted such that each edit's range.end <= the next's range.start; got {:?}",
        edits,
    );

    // Edits are sorted by range.start. Rebuild the buffer
    // interleaving original slices with replacements; the cursor
    // remap walks the same edit sequence and accumulates the byte
    // delta past every prior splice.
    let total_delta: isize = edits
        .iter()
        .map(|e| e.replacement.len() as isize - (e.range.end - e.range.start) as isize)
        .sum();
    let new_len = (state.markdown.len() as isize + total_delta).max(0) as usize;
    let mut new_md = String::with_capacity(new_len);
    let mut last = 0;
    for e in edits {
        new_md.push_str(&state.markdown[last..e.range.start]);
        new_md.push_str(&e.replacement);
        last = e.range.end;
    }
    new_md.push_str(&state.markdown[last..]);

    let map = |off: usize| -> usize {
        let mut shift: isize = 0;
        for e in edits {
            if e.range.end <= off {
                shift += e.replacement.len() as isize - (e.range.end - e.range.start) as isize;
            } else if e.range.start < off && off < e.range.end {
                // Cursor was inside the spliced range — pin to the
                // end of the replacement.
                let new_pos = (e.range.start as isize + shift) + e.replacement.len() as isize;
                return new_pos.max(0) as usize;
            } else {
                break;
            }
        }
        ((off as isize) + shift).max(0) as usize
    };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => {
            let a = map(anchor);
            let h = map(head);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Defensive cap on the chain depth at which `promote_soft_breaks`'s
/// missing-prefix-repair fires. With refactor A's tree-based fence
/// detection in place, the repair only fires when pulldown classifies
/// the line as fence body — so the runaway divergence the cap was
/// originally protecting against (byte scanner missing fences and
/// counting their `>`s as BQ markers, growing the chain on each
/// re-parse) can no longer produce ever-deeper chains. The cap stays
/// as defense-in-depth: 16 `>` markers around a paragraph is well
/// past any human nesting, so capping there is a no-op for legit
/// docs and a guardrail for genuinely corrupted buffers.
const MAX_REPAIR_DEPTH: usize = 16;

/// Recursively gather every `CodeBlock` node in document order.
fn collect_code_blocks<'a>(
    nodes: &'a [crate::syntax::SyntaxNode],
    out: &mut Vec<&'a crate::syntax::SyntaxNode>,
) {
    for node in nodes {
        if matches!(node.kind, crate::syntax::NodeKind::CodeBlock { .. }) {
            out.push(node);
        }
        collect_code_blocks(&node.children, out);
    }
}

fn promote_soft_breaks(state: EditorState, cache: &mut ParseCache) -> EditorState {
    let bytes = state.markdown.as_bytes();
    // Two scopes special-case the soft-break promotion rule:
    //
    //   * Fenced code-block content. A single mid-content `\n` inside
    //     ```/~~~ fences is a literal line separator, not the
    //     ambiguous CommonMark soft break — so we never inflate it
    //     to a paragraph-break pair. But the BQ scope wrapping the
    //     fence (if any) still demands `> ` prefix continuation on
    //     each new line; if the line after the `\n` is missing some
    //     of those markers (a freshly-typed continuation), we splice
    //     in just the missing prefix bytes — no pair, no extra `\n`.
    //   * List ranges. Inside a list pulldown handles line structure
    //     (item separators, marker continuation, indented paragraphs,
    //     lazy continuations). Promoting `\n` to `\n\n` between two
    //     items would split the list.
    let tree = cache.tree(&state.markdown);
    let code_ranges = analysis::fenced_code_ranges_in_tree(tree, bytes);
    let math_ranges = analysis::display_math_block_ranges_in_tree(tree, bytes);
    let list_ranges = analysis::list_content_ranges_in_tree(tree, bytes);

    // Each entry: (insertion_position, inserted_string). Computed in a
    // single forward scan over the *original* buffer so offsets line
    // up with the input — the apply step later remaps cursor offsets
    // exactly once.
    let mut inserts: Vec<(usize, String)> = Vec::new();
    for p in 0..bytes.len() {
        let in_code = is_in_ranges(p, &code_ranges);
        let in_math = is_in_ranges(p, &math_ranges);
        let in_list = is_in_ranges(p, &list_ranges);
        // **Verbatim-opener-boundary exemption.** If `p` is a `\n`
        // whose next byte starts a fenced code or block-math
        // construct, skip the soft-break promotion. The pair-shape
        // `\n[prefix]\n[prefix]` is appropriate between two regular
        // content lines, but right before a verbatim opener it would
        // inject a stray prefix-only line that the user can never
        // delete via Backspace — `delete_backward`'s "eat the line
        // above" path removes it, this pass re-adds it, and the
        // cycle traps the buffer at a fixed point well above zero.
        let next_starts_verbatim = bytes[p] == b'\n'
            && p + 1 < bytes.len()
            && (code_ranges.iter().any(|r| r.start == p + 1)
                || math_ranges.iter().any(|r| r.start == p + 1));
        if next_starts_verbatim && !in_code && !in_math {
            continue;
        }
        // Verbatim content (fenced code + block-level `$$..$$` math) is
        // exempt from soft-break-to-pair promotion — the `\n` is a
        // literal line separator, not a soft break. But a verbatim
        // block sitting inside a BQ still demands each new line carry
        // the surrounding `> ` markers, and a verbatim block inside
        // an LI demands the LI's continuation indent. We compute the
        // missing-prefix repair in both list and non-list cases here
        // so the LI-wrapping case isn't silently skipped by the
        // list_ranges short-circuit. Code and math share the same
        // shape (literal `\n`s with chain prefix on each continuation
        // line) so the repair is identical for both.
        if in_code || in_math {
            if bytes[p] != b'\n' || p + 1 >= bytes.len() {
                continue;
            }
            // Build the *full* chain-aware continuation prefix the
            // line ending at `p` carries, so a missing prefix on the
            // next line is repaired in shape (LI indent + BQ marker,
            // alternating). The chain query takes the cursor at `p`
            // (end of the line) so the parser sees this line's
            // surrounding scope.
            let chain_at = analysis::enclosing_containers_at_in_tree(tree, bytes, p);
            let expected_prefix = analysis::chain_continuation_prefix(&chain_at);
            if expected_prefix.is_empty() {
                continue;
            }
            // What the next line already has (literal leading bytes
            // that form a valid prefix). Compare prefix-wise so we
            // only splice in the missing tail.
            let next_line = &bytes[p + 1..];
            let mut have = 0usize;
            let exp_bytes = expected_prefix.as_bytes();
            while have < exp_bytes.len()
                && have < next_line.len()
                && next_line[have] == exp_bytes[have]
            {
                have += 1;
            }
            if have >= expected_prefix.len() {
                continue;
            }
            // **Anti-runaway guard.** Real documents nest at most a
            // handful of BQ / LI containers; a chain of 16+ entries is
            // a sign the buffer has accumulated phantom markers
            // (typically through repeated re-parsing of an
            // ill-formed-but-tolerated state — e.g. orphan auto-close
            // fences chaining `> > > > >` runs together). At that
            // depth the missing-prefix tail would be huge, which
            // *grows* the buffer on every keystroke, which produces
            // even deeper chains on the next re-parse, which grows
            // the buffer further. Cap the repair: if pulldown reports
            // a chain deeper than [`MAX_REPAIR_DEPTH`], let pulldown's
            // own re-parse pick up whatever scope the bytes actually
            // open instead of forcing an ever-deeper prefix back in.
            //
            // The cap protects against the divergence; legitimate
            // docs with a few levels of nesting are well below it.
            if chain_at.len() > MAX_REPAIR_DEPTH {
                continue;
            }
            let missing = &expected_prefix[have..];
            // Skip the repair when the missing tail consists only of
            // *prefix bytes* (BQ markers, spaces, tabs) AND the rest
            // of the next line is empty / whitespace. Two cases roll
            // up here:
            //
            // 1. Trailing whitespace tail (`" "` or `"   "`): the
            //    space at the end of `chain_continuation_prefix` is
            //    cosmetic on a content-empty row — pulldown parses
            //    both `>` and `> ` as valid BQ continuation when
            //    nothing follows.
            // 2. Missing `> ` markers (`"> "`, `" >"`, `"> > "`, …):
            //    on a prefix-only continuation line the user is
            //    *deleting* the prefix when they Backspace.
            //    Re-injecting the markers undoes their keystroke and
            //    traps the buffer in the trailing-prefix oscillation
            //    documented in
            //    `bugs.md::backspace_oscillates_inside_corrupted_chain_in_fenced_code`.
            //
            // The guard is conservative: we only skip when the next
            // line carries *no* content past the partial prefix. A
            // line with actual content after a partial prefix needs
            // the repair so the BQ scope holds and the content stays
            // inside the code body.
            let next_line_end = next_line
                .iter()
                .position(|&b| b == b'\n')
                .unwrap_or(next_line.len());
            let trailing_only_whitespace_line = next_line[have..next_line_end]
                .iter()
                .all(|&b| b == b' ' || b == b'\t');
            let missing_only_prefix_bytes = missing
                .bytes()
                .all(|b| b == b' ' || b == b'\t' || b == b'>');
            if missing_only_prefix_bytes && trailing_only_whitespace_line {
                continue;
            }
            if !missing.is_empty() {
                // Insert the missing tail *after* whatever prefix the
                // next line already has. Inserting at `p + 1` (before
                // the matched bytes) corrupts the prefix: e.g.
                // expected `"> "`, next line starts with `">"`, missing
                // is `" "` — inserting at `p + 1` produces `" >"` (the
                // bytes swap), which is *not* a valid BQ marker. Insert
                // at `p + 1 + have` instead so the line reads
                // `[matched][missing][rest]`.
                inserts.push((p + 1 + have, missing.to_string()));
            }
            continue;
        }

        if in_list {
            // List structure normalization is `normalize_lists`'s job.
            continue;
        }

        if !is_soft_break(bytes, p) {
            continue;
        }

        // Promotion shape: turn the stray `\n` at p into a complete
        // depth-D pair `\n[prefix]\n[prefix]`. D is the depth of the
        // line ending at `p`; whatever markers already exist on the
        // *next* line are kept in place so we never duplicate them.
        let depth = line_depth_ending_at(bytes, p);
        let (existing_markers, _) = count_line_markers(bytes, p + 1);
        let prefix = "> ".repeat(depth);

        let (insert_at, inserted) = if existing_markers >= depth {
            // The next line already opens with at least the line-
            // before's depth. Splice in `[prefix]\n` right *after*
            // the existing `\n`; the existing markers naturally
            // become the second `[prefix]` of the pair.
            (p + 1, format!("{prefix}\n"))
        } else {
            // Lazy continuation (next line lacks at least one of the
            // markers we'd expect). Insert the full `[prefix]\n
            // [prefix]` so the continuation line gains the missing
            // markers and the pair structure is complete.
            (p + 1, format!("{prefix}\n{prefix}"))
        };

        if inserted.is_empty() {
            continue; // depth 0 + existing_markers >= 0 → nothing to do
        }
        inserts.push((insert_at, inserted));
    }

    if inserts.is_empty() {
        return state;
    }
    cache.invalidate();

    // Apply inserts in order. Each `inserts[i].0` is in *original*
    // coordinates, so we rebuild the buffer by interleaving original
    // slices with inserted strings.
    let total_added: usize = inserts.iter().map(|(_, s)| s.len()).sum();
    let mut new_md = String::with_capacity(state.markdown.len() + total_added);
    let mut last = 0;
    for (pos, ins) in &inserts {
        new_md.push_str(&state.markdown[last..*pos]);
        new_md.push_str(ins);
        last = *pos;
    }
    new_md.push_str(&state.markdown[last..]);

    let map = |off: usize| -> usize {
        let mut shift = 0;
        for (pos, ins) in &inserts {
            if *pos < off {
                shift += ins.len();
            } else if *pos == off {
                // Convention: an offset exactly at an insertion point
                // shifts forward, so the cursor stays "with" the
                // content it sat against. (For Enter-from-soft-break
                // the cursor is typically left-of the broken `\n`,
                // so this branch rarely fires; it's the safe default.)
                shift += ins.len();
            }
        }
        off + shift
    };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };

    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Insert a space after every blockquote `>` marker that isn't already
/// followed by one — *unless* the cursor (or selection anchor / head)
/// is exactly the byte right after the `>`. Mid-typing the user might
/// have just pressed `>` and intends to type a space themselves; we
/// don't second-guess them. Once the cursor moves away, the next
/// `update` call's post-pass normalizes the marker so the parsed
/// shape stays predictable for the renderer.
///
/// Skips `>` bytes that fall inside a verbatim region (fenced code or
/// block-level `$$..$$` math) — those are literal `>` characters, not
/// blockquote markers.
fn normalize_blockquote_prefixes(state: EditorState, cache: &mut ParseCache) -> EditorState {
    let bytes = state.markdown.as_bytes();
    let tree = cache.tree(&state.markdown);
    let code_ranges = analysis::fenced_code_ranges_in_tree(tree, bytes);
    let math_ranges = analysis::display_math_block_ranges_in_tree(tree, bytes);
    let (cursor_head, cursor_anchor) = match state.selection {
        Selection::Cursor(p) => (p, p),
        Selection::Range { anchor, head } => (head, anchor),
    };

    let mut insert_at: Vec<usize> = Vec::new();
    let mut p = 0;
    while p < bytes.len() {
        let line_start = p;
        let mut line_end = p;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        let mut q = line_start;
        loop {
            let mut indent = 0;
            while q < line_end && bytes[q] == b' ' && indent < 3 {
                q += 1;
                indent += 1;
            }
            if q < line_end
                && bytes[q] == b'>'
                && !is_in_ranges(q, &code_ranges)
                && !is_in_ranges(q, &math_ranges)
            {
                let after_gt = q + 1;
                let needs_space =
                    after_gt < line_end && bytes[after_gt] != b' ' && bytes[after_gt] != b'\n';
                if needs_space && after_gt != cursor_head && after_gt != cursor_anchor {
                    insert_at.push(after_gt);
                }
                q = after_gt;
                if q < line_end && bytes[q] == b' ' {
                    q += 1;
                }
                continue;
            }
            break;
        }
        p = line_end + 1;
    }

    if insert_at.is_empty() {
        return state;
    }
    cache.invalidate();

    let mut new_md = String::with_capacity(state.markdown.len() + insert_at.len());
    let mut last = 0;
    for &pos in &insert_at {
        new_md.push_str(&state.markdown[last..pos]);
        new_md.push(' ');
        last = pos;
    }
    new_md.push_str(&state.markdown[last..]);

    let map = |off: usize| -> usize { off + insert_at.iter().filter(|&&p| p < off).count() };
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(map(p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: map(anchor),
            head: map(head),
        },
    };

    EditorState {
        markdown: new_md,
        selection: new_sel,
    }
}

/// Snap any cursor or selection endpoint that landed at a forbidden
/// position (interior of a structural `\n\n` pair) away from where it
/// came from. Forward if the cursor moved forward (or didn't move),
/// backward if it moved back. See module docs for the rationale.
fn avoid_forbidden_positions(
    state: EditorState,
    prev_anchor: usize,
    prev_head: usize,
) -> EditorState {
    let markdown = state.markdown.clone();
    let new_sel = match state.selection {
        Selection::Cursor(p) => Selection::Cursor(snap_off_forbidden(&markdown, p, prev_head)),
        Selection::Range { anchor, head } => {
            let a = snap_off_forbidden(&markdown, anchor, prev_anchor);
            let h = snap_off_forbidden(&markdown, head, prev_head);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    EditorState {
        selection: new_sel,
        ..state
    }
}

fn snap_off_forbidden(markdown: &str, pos: usize, prev: usize) -> usize {
    if !is_forbidden_position(markdown, pos) {
        return pos;
    }
    if pos < prev {
        prev_allowed_position(markdown, pos)
    } else {
        next_allowed_position(markdown, pos)
    }
}

/// Re-exported for behavior tests that look up depth via `update::blockquote_depth_at`.
/// New callers should import directly from [`crate::analysis`].
pub use crate::analysis::blockquote_depth_at;

// `EditorEvent::InsertNewline` / `InsertLineBreak` route through
// `analysis::enter_insertion` / `line_break_insertion`, which know about
// every container kind and emit the right source string for the cursor's
// position. The shell stays a router so keyboard, IME, paste-derived, and
// programmatic dispatch all share one rule.

fn insert_newline(state: EditorState) -> EditorState {
    let cursor = state.selection.head();
    // Empty-item Enter decreases the item's nesting depth by one
    // (analogous to blockquote outdent). This subsumes the
    // "double-Enter exits a list" UX without a dedicated state flag.
    if let Some(edit) = analysis::empty_item_exit_edit(&state.markdown, cursor) {
        return apply_replace(&state.markdown, edit);
    }
    // Empty-BQ-paragraph Enter is the analog for blockquotes — drops
    // the innermost BQ scope on the trailing row. Without this, every
    // Enter on an empty `> ` row just adds another empty `> ` pair,
    // and the user has no Enter-only gesture to leave a BQ.
    if let Some(edit) = analysis::empty_bq_paragraph_exit_edit(&state.markdown, cursor) {
        return apply_replace(&state.markdown, edit);
    }
    // Auto-close-fence: any Enter inside an unterminated fenced code
    // block injects a matching closer below the cursor, with the
    // cursor on a fresh body row in between. After this fires the
    // construct is terminated, so subsequent rules (in-fence
    // continuation, BQ-prefix normalize, soft-break exemption) have a
    // single unambiguous truth to read off `is_in_fenced_code`.
    if let Some(edit) = analysis::auto_close_fence_edit(&state.markdown, cursor) {
        return apply_replace(&state.markdown, edit);
    }
    // Auto-close-math: same shape as auto-close-fence but for an
    // unterminated block-level `$$..$$` construct. Fires on the first
    // Enter after the user types `$$` (or any state where pulldown
    // sees an open math block without a closer), so subsequent rules
    // can rely on the math block being terminated and verbatim from
    // here on.
    if let Some(edit) = analysis::auto_close_math_edit(&state.markdown, cursor) {
        return apply_replace(&state.markdown, edit);
    }
    let insertion = analysis::enter_insertion(&state.markdown, cursor);
    insert_text(state, &insertion)
}

fn insert_line_break(state: EditorState) -> EditorState {
    let insertion = analysis::line_break_insertion(&state.markdown, state.selection.head());
    insert_text(state, &insertion)
}

fn increase_list_depth(state: EditorState) -> EditorState {
    let cursor = state.selection.head();
    if let Some(edits) = analysis::list_item_indent_edits(&state.markdown, cursor) {
        return apply_edits(state, &edits);
    }
    // Fallback: when the indent path doesn't apply (no list item at
    // cursor, or cursor inside a verbatim body where the construct's
    // rules win), Tab inserts a literal `\t`. This is what users
    // expect inside a fence or block-math body — same behavior as any
    // plain text editor, the body is verbatim.
    if analysis::is_in_verbatim_region(&state.markdown, cursor) {
        return insert_text(state, "\t");
    }
    state
}

fn decrease_list_depth(state: EditorState) -> EditorState {
    let cursor = state.selection.head();
    let Some(edits) = analysis::list_item_dedent_edits(&state.markdown, cursor) else {
        return state;
    };
    apply_edits(state, &edits)
}

fn apply_replace(markdown: &str, edit: analysis::DepthDecreaseEdit) -> EditorState {
    let mut buf = String::with_capacity(
        markdown.len() - (edit.range.end - edit.range.start) + edit.replacement.len(),
    );
    buf.push_str(&markdown[..edit.range.start]);
    buf.push_str(&edit.replacement);
    buf.push_str(&markdown[edit.range.end..]);
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(edit.cursor),
    }
}

fn clamp(pos: usize, len: usize) -> usize {
    pos.min(len)
}

fn delete_selection(state: &EditorState) -> (String, usize) {
    let len = state.markdown.len();
    match state.selection {
        Selection::Cursor(p) => (state.markdown.clone(), clamp(p, len)),
        Selection::Range { anchor, head } => {
            let start = clamp(anchor.min(head), len);
            let end = clamp(anchor.max(head), len);
            let mut out = String::with_capacity(len - (end - start));
            out.push_str(&state.markdown[..start]);
            out.push_str(&state.markdown[end..]);
            (out, start)
        }
    }
}

fn insert_text(state: EditorState, text: &str) -> EditorState {
    let (mut buf, cursor) = delete_selection(&state);
    buf.insert_str(cursor, text);
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(cursor + text.len()),
    }
}

fn delete_backward(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    if cursor == 0 {
        return state;
    }

    // List-item depth decrease: Backspace right after a marker
    // takes the same path as Shift+Tab — for a top-level item
    // it becomes a paragraph (with the surrounding scope's
    // canonical break ahead of it), and for a nested item it
    // becomes a sibling of its parent. The two gestures share
    // semantics; see `analysis::list_item_dedent_edits`.
    if analysis::cursor_at_item_marker_end(&state.markdown, cursor)
        && let Some(edits) = analysis::list_item_dedent_edits(&state.markdown, cursor)
    {
        return apply_edits(state, &edits);
    }

    // Fence BQ-outdent: Backspace at the byte right before a
    // fenced code block's opener chars unwraps the innermost BQ
    // around the whole fence in one keystroke (rule: "context
    // changes to the first line of the code block apply to the
    // entire code block"). The LI variant is covered by
    // `list_item_dedent_edits` above when the cursor coincides
    // with the LI's marker_end byte.
    if let Some(edits) = analysis::fence_bq_outdent_edits(&state.markdown, cursor) {
        return apply_edits(state, &edits);
    }

    // At a structural boundary, Backspace has two specialized rules.
    // Both first snap forward over any pair interior — that's where a
    // direct cursor placement (e.g. a click on the visually-collapsed
    // paragraph_gap, or a programmatic SetSelection) might land — then
    // inspect the structure ending at the snapped position. Inside a
    // verbatim region (fenced code or block math), `\n`s are literal
    // line separators — fall through to the regular grapheme-delete
    // path instead.
    let bytes = state.markdown.as_bytes();
    if !analysis::is_in_verbatim_region(&state.markdown, cursor) {
        let snapped = next_allowed_position(&state.markdown, cursor);
        // Blockquote outdent: at the start of a non-first paragraph
        // inside a BQ, Backspace pops one level of nesting from
        // *both halves* of the preceding `\n[prefix]\n[prefix]` pair
        // instead of merging the paragraph with the previous one.
        // Outdenting both halves keeps the pair invariant (depth-D
        // → depth-(D-1) is still a clean balanced pair, and a
        // depth-1 pair becomes a depth-0 `\n\n` break) so the result
        // is never an asymmetric pair the rest of the rules would
        // have to special-case. When the paragraph reaches depth 0
        // the outdent detector returns `None` and the next Backspace
        // falls through to the depth-0 atomic pair delete below,
        // which merges into the previous paragraph as before.
        if let Some((above, below)) = analysis::bq_paragraph_outdent(bytes, snapped) {
            // Apply right-to-left so the earlier range's offsets
            // don't need to be remapped.
            let after_below = splice(&state.markdown, cursor, below.start, below.end);
            return splice(
                &after_below.markdown,
                after_below.selection.head(),
                above.start,
                above.end,
            );
        }
        // Atomic pair delete at depth 0: when the cursor sits at a
        // top-level `\n\n` paragraph break, Backspace removes the
        // whole pair in one step, merging the two paragraphs.
        if let Some(pair_start) = pair_at_end(bytes, snapped) {
            return splice(&state.markdown, cursor, pair_start, snapped);
        }
        // Chain-aware BQ outdent: the BQ-only walker above
        // (`bq_paragraph_outdent`) doesn't see pairs whose prefix
        // includes a list-item continuation indent (e.g.
        // `\n   > \n   > ` for a depth-1 BQ pair inside a `1. `
        // item, or the alternating-chain shape
        // `\n   > \n   >    > ` for `[LI, BQ, LI, BQ]`). Compute the
        // cursor's container chain, build the canonical pair shape
        // for it via [`chain_pair_shape`], and try matching it
        // exactly. When the cursor sits at the end of a chain-aware
        // pair whose chain ends in a `BlockQuote`, drop the innermost
        // BQ scope on the trailing row only — replace the pair with
        // the canonical paragraph-break shape for the *new* chain
        // (chain minus the trailing BQ), which derives from the
        // three-branch [`chain_pair_shape`] rule:
        //
        // - New chain ends in BQ → symmetric `\n[full]\n[full]`
        //   (depth-(D-1) pair; the existing pure-BQ depth-N tests).
        // - New chain has BQ but trails LIs (alternating chain) →
        //   asymmetric `\n[blank]\n[content]` where `blank` is the
        //   prefix through the last BQ and `content` is the full
        //   prefix. This keeps the still-open BQs' markers on the
        //   blank line so pulldown doesn't close intermediate
        //   list-item scopes.
        // - New chain has no BQ (pure LI or empty) → `\n\n[content]`
        //   (top-level paragraph break, LI continuation indent on
        //   the new row only — Path B from
        //   `bugs.md::backspace_on_empty_bq_paragraph_in_li_eats_hidden_chars`).
        let chain = analysis::enclosing_containers_at(&state.markdown, snapped);
        if matches!(
            chain.last(),
            Some(analysis::EnclosingContainer::BlockQuote { .. })
        ) && let Some(pair_start) = analysis::pair_at_end_for_chain(bytes, snapped, &chain)
        {
            let new_chain = &chain[..chain.len() - 1];
            let (blank_prefix, content_prefix) = analysis::chain_pair_shape(new_chain);
            let replacement = format!("\n{blank_prefix}\n{content_prefix}");
            let mut buf = String::with_capacity(
                state.markdown.len() - (snapped - pair_start) + replacement.len(),
            );
            buf.push_str(&state.markdown[..pair_start]);
            buf.push_str(&replacement);
            buf.push_str(&state.markdown[snapped..]);
            let new_cursor = pair_start + replacement.len();
            return EditorState {
                markdown: buf,
                selection: Selection::Cursor(new_cursor),
            };
        }
        // LI-trailing pair (no BQ at the chain's end but possibly BQs
        // earlier): atomic-delete the whole shape so the user doesn't
        // eat hidden prefix bytes one at a time. The forbidden-position
        // predicate (`is_forbidden_position` → `is_list_indent_interior`
        // / `is_chain_pair_interior`) already treats these bytes as
        // pair-interior for navigation; this matches that behavior on
        // the delete path.
        if matches!(
            chain.last(),
            Some(analysis::EnclosingContainer::ListItem(_))
        ) && let Some(pair_start) = analysis::pair_at_end_for_chain(bytes, snapped, &chain)
        {
            return splice(&state.markdown, cursor, pair_start, snapped);
        }
    }

    // **Anti-oscillation: in-verbatim "eat the line above" when
    // grapheme-delete would just strip a `\n` between two prefix-only
    // lines.**
    //
    // Inside a deeply-nested verbatim body (fenced code or block-level
    // `$$..$$` math), the cursor often lands right after a `\n` on a
    // line whose content is just BQ markers (the previous repair
    // filled them in). A bare grapheme-delete there removes the `\n`,
    // the line below merges into the line above, `enforce_invariants`
    // re-parses the merged line as a deeper-BQ opener,
    // `normalize_blockquote_prefixes` re-spaces it, and the buffer
    // ends up the same length. The user's keystroke appears to do
    // nothing.
    //
    // When that pattern is detected (cursor inside a verbatim region,
    // sitting at byte `\n+1`, and the *previous* line consists only
    // of BQ markers + whitespace — no content), delete the whole
    // previous line including its `\n`. Forward progress is
    // guaranteed: the line above can't reappear from any normalize
    // pass because its content was empty to begin with.
    if analysis::is_in_verbatim_region(&state.markdown, cursor)
        && cursor > 0
        && bytes[cursor - 1] == b'\n'
    {
        let cursor_line_end = cursor - 1;
        let mut prev_line_start = cursor_line_end;
        while prev_line_start > 0 && bytes[prev_line_start - 1] != b'\n' {
            prev_line_start -= 1;
        }
        if is_prefix_only_line(bytes, prev_line_start, cursor_line_end) {
            return splice(&state.markdown, cursor, prev_line_start, cursor);
        }
    }

    // **In-verbatim "delete the prefix-only line" gesture.**
    //
    // When the cursor sits anywhere on a prefix-only line inside a
    // verbatim body (fenced code or block-level `$$..$$` math), the
    // line is purely chain prefix — BQ markers, LI continuation
    // indent — with no content; Backspace removes the entire line
    // including its leading `\n`. Without this, the user has to press
    // Backspace once per byte of hidden prefix to remove an empty
    // body row that to them looks like one keystroke worth of content
    // (the user-reported "deleting invisible, forbidden characters"
    // case in `bugs.md`).
    if analysis::is_in_verbatim_region(&state.markdown, cursor) && cursor > 0 {
        let mut line_start = cursor;
        while line_start > 0 && bytes[line_start - 1] != b'\n' {
            line_start -= 1;
        }
        let mut line_end = cursor;
        while line_end < bytes.len() && bytes[line_end] != b'\n' {
            line_end += 1;
        }
        if line_start > 0
            && line_start < line_end
            && is_prefix_only_line(bytes, line_start, line_end)
        {
            return splice(&state.markdown, cursor, line_start - 1, cursor);
        }
    }

    let prev = prev_grapheme_offset(&state.markdown, cursor);
    splice(&state.markdown, cursor, prev, cursor)
}

/// `true` when the byte range `[start, end)` consists only of BQ
/// markers (`>`), spaces, and tabs — i.e. the line carries chain
/// continuation prefix and nothing else. A line that ends in any other
/// byte (including fence chars or code body content) is *not*
/// prefix-only; this is conservative on purpose so the in-code
/// "eat the line above" path never deletes user-typed content.
fn is_prefix_only_line(bytes: &[u8], start: usize, end: usize) -> bool {
    if start >= end {
        return false;
    }
    bytes[start..end]
        .iter()
        .all(|&b| b == b'>' || b == b' ' || b == b'\t')
}

fn delete_forward(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    if cursor >= state.markdown.len() {
        return state;
    }

    let bytes = state.markdown.as_bytes();
    if !analysis::is_in_verbatim_region(&state.markdown, cursor) {
        let snapped = prev_allowed_position(&state.markdown, cursor);
        if let Some(pair_end) = pair_at_start(bytes, snapped) {
            return splice(&state.markdown, cursor, snapped, pair_end);
        }
        // Chain-aware atomic pair delete (forward analog of the
        // backward path in `delete_backward`). See the comment there
        // for why the BQ-only walker above misses LI-wrapped pair
        // shapes.
        let chain = analysis::enclosing_containers_at(&state.markdown, snapped);
        if !chain.is_empty()
            && let Some(pair_end) = analysis::pair_at_start_for_chain(bytes, snapped, &chain)
        {
            return splice(&state.markdown, cursor, snapped, pair_end);
        }
    }

    let next = next_grapheme_offset(&state.markdown, cursor);
    splice(&state.markdown, cursor, cursor, next)
}

/// Delete from the start of the previous word through the cursor.
/// With a non-collapsed selection, falls through to the normal range
/// delete (matches every other deletion's "selection wins" rule).
///
/// **Chain-prefix floor.** When the cursor's line carries a structural
/// chain prefix (`> ` per BQ level, `- ` / `1. ` / etc on a list-item
/// marker line, marker-width spaces on a list-item continuation line),
/// the word-target is clamped to the line's
/// [`analysis::line_chain_prefix_end`]. Without this clamp,
/// `prev_word_offset` at the start of a BQ paragraph would scan past
/// the `> ` (neither `>` nor space contains alphanumeric, so the
/// word walker treats them as separator and walks past them to the
/// previous word on a previous line) and the splice would delete the
/// BQ marker along with content. Top-level lines have no chain
/// prefix, so the clamp is a no-op there and word-delete crosses
/// `\n` naturally — matching plain-text editor behavior.
fn delete_word_backward(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    if cursor == 0 {
        return state;
    }
    let mut target = prev_word_offset(&state.markdown, cursor);
    let prefix_end = analysis::line_chain_prefix_end(&state.markdown, cursor);
    let line_start = line_start_offset(&state.markdown, cursor);
    if prefix_end > line_start && target < prefix_end {
        target = prefix_end;
    }
    if target >= cursor {
        return state;
    }
    splice(&state.markdown, cursor, target, cursor)
}

/// Forward analog of [`delete_word_backward`] — delete from the cursor
/// through the end of the next word.
///
/// **Next-line chain-prefix floor.** When the word-target lies past
/// the cursor's line end *and* the next line carries a chain prefix,
/// the target is clamped to the cursor's line end so the splice
/// doesn't eat the next line's `> ` / `- ` / continuation indent.
/// Top-level next lines have no chain prefix, so word-delete crosses
/// `\n` and devours the next line's word naturally.
fn delete_word_forward(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    if cursor >= state.markdown.len() {
        return state;
    }
    let mut target = next_word_offset(&state.markdown, cursor);
    let cursor_line_end = line_end_offset(&state.markdown, cursor);
    if target > cursor_line_end {
        let next_line_start = cursor_line_end + 1;
        if next_line_start <= state.markdown.len() {
            let next_prefix_end = analysis::line_chain_prefix_end(&state.markdown, next_line_start);
            if next_prefix_end > next_line_start {
                target = cursor_line_end;
            }
        }
    }
    if target <= cursor {
        return state;
    }
    splice(&state.markdown, cursor, cursor, target)
}

/// Delete from the cursor back to the *visible content edge* of the
/// current line — past every byte the renderer treats as chain chrome
/// (BQ `> ` markers, list-item markers on the marker line,
/// list-continuation indent on continuation lines) but before any byte
/// the user thinks of as content.
///
/// At top level (no chain) the floor degenerates to the raw line
/// start, matching the macOS plain-text Cmd+Backspace convention.
/// Inside a BQ paragraph, list item, or nested combination, the chain
/// prefix survives and only the user's content disappears — see
/// [`analysis::line_chain_prefix_end`] for the per-container rules.
///
/// No-op when the cursor already sits at the content edge.
fn delete_to_line_start(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    let prefix_end = analysis::line_chain_prefix_end(&state.markdown, cursor);
    if prefix_end >= cursor {
        return state;
    }
    splice(&state.markdown, cursor, prefix_end, cursor)
}

/// Delete from the cursor forward to the end of the current line
/// (the byte right before its terminating `\n`, if any). Forward
/// analog of [`delete_to_line_start`]; the line `\n` itself is left
/// in place so the surrounding block structure stays intact.
///
/// No chain-prefix concerns: the deletion is bounded by `\n` and
/// therefore never reaches into the next line's prefix bytes.
fn delete_to_line_end(state: EditorState) -> EditorState {
    if !state.selection.is_collapsed() {
        let (buf, cursor) = delete_selection(&state);
        return EditorState {
            markdown: buf,
            selection: Selection::Cursor(cursor),
        };
    }
    let cursor = state.selection.head();
    let target = line_end_offset(&state.markdown, cursor);
    if target <= cursor {
        return state;
    }
    splice(&state.markdown, cursor, cursor, target)
}

/// Splice out `[del_start, del_end)` from `markdown` and re-anchor the
/// cursor relative to the deletion. Caller-supplied `cursor` is the
/// current head; the returned state collapses any range selection to a
/// cursor (which is fine for our delete callers — they all operate on
/// collapsed selections after the range-delete-and-return early exit).
fn splice(markdown: &str, cursor: usize, del_start: usize, del_end: usize) -> EditorState {
    let mut buf = String::with_capacity(markdown.len() - (del_end - del_start));
    buf.push_str(&markdown[..del_start]);
    buf.push_str(&markdown[del_end..]);
    let new_cursor = if cursor <= del_start {
        cursor
    } else if cursor < del_end {
        del_start
    } else {
        cursor - (del_end - del_start)
    };
    EditorState {
        markdown: buf,
        selection: Selection::Cursor(new_cursor),
    }
}

fn set_selection(state: EditorState, sel: Selection) -> EditorState {
    let len = state.markdown.len();
    let normalized = match sel {
        Selection::Cursor(p) => Selection::Cursor(clamp(p, len)),
        Selection::Range { anchor, head } => {
            let a = clamp(anchor, len);
            let h = clamp(head, len);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    let on_boundaries = snap_selection_to_char_boundaries(&state.markdown, normalized);
    // Mouse clicks and host SetSelection calls feed offsets that may
    // land on a forbidden pair interior — most often a synthetic empty
    // paragraph between two real blocks, whose `display_to_source`
    // anchors at the block's source-range start (which is the pair
    // interior). Using the prev-comparison rule here would flip the
    // cursor between two adjacent allowed positions every time a
    // `MouseMoveEvent` re-feeds the same offset, because each call
    // sees the previously-snapped position as `prev`. Nearest-allowed
    // is idempotent: same input → same output.
    let final_sel = match on_boundaries {
        Selection::Cursor(p) => Selection::Cursor(nearest_allowed_position(&state.markdown, p)),
        Selection::Range { anchor, head } => {
            let a = nearest_allowed_position(&state.markdown, anchor);
            let h = nearest_allowed_position(&state.markdown, head);
            if a == h {
                Selection::Cursor(h)
            } else {
                Selection::Range { anchor: a, head: h }
            }
        }
    };
    EditorState {
        selection: final_sel,
        ..state
    }
}

#[derive(Debug, Clone, Copy)]
enum Move {
    Left,
    Right,
    Up,
    Down,
    LineStart,
    LineEnd,
    DocStart,
    DocEnd,
    /// Cursor jumps to the start of the previous word (Unicode word
    /// boundary segments containing at least one alphanumeric char).
    WordLeft,
    /// Cursor jumps to the end of the next word. Symmetric with
    /// [`Move::WordLeft`].
    WordRight,
}

fn move_(state: EditorState, direction: Move, extending: bool) -> EditorState {
    let head = state.selection.head();
    let new_head = match direction {
        Move::Left => prev_grapheme_offset(&state.markdown, head),
        Move::Right => next_grapheme_offset(&state.markdown, head),
        Move::Up => move_vertical(&state.markdown, head, -1),
        Move::Down => move_vertical(&state.markdown, head, 1),
        // Home is content-edge biased: when the raw line-start is a
        // forbidden position (cursor sits inside the line's hidden
        // continuation prefix — the leading `> > ` / list indent), we
        // search *forward* to the visible content edge of the same
        // line rather than letting the post-move forbidden-snap walk
        // backward across the previous `\n`. The forward scan stops at
        // the first allowed byte, which by construction is at most
        // `line_end` (line-end is always allowed); so a line composed
        // entirely of hidden prefix bytes lands at the line terminus
        // and Home never crosses a `\n`.
        Move::LineStart => {
            next_allowed_position(&state.markdown, line_start_offset(&state.markdown, head))
        }
        Move::LineEnd => line_end_offset(&state.markdown, head),
        Move::DocStart => 0,
        Move::DocEnd => state.markdown.len(),
        Move::WordLeft => prev_word_offset(&state.markdown, head),
        Move::WordRight => next_word_offset(&state.markdown, head),
    };

    let new_sel = if extending {
        let anchor = match state.selection {
            Selection::Cursor(p) => p,
            Selection::Range { anchor, .. } => anchor,
        };
        if anchor == new_head {
            Selection::Cursor(new_head)
        } else {
            Selection::Range {
                anchor,
                head: new_head,
            }
        }
    } else if !state.selection.is_collapsed() {
        // Collapse to the appropriate edge (the natural direction the user
        // is moving toward) instead of jumping past the selection.
        match direction {
            Move::Left | Move::WordLeft | Move::Up | Move::LineStart | Move::DocStart => {
                Selection::Cursor(state.selection.lower_bound())
            }
            Move::Right | Move::WordRight | Move::Down | Move::LineEnd | Move::DocEnd => {
                Selection::Cursor(state.selection.upper_bound())
            }
        }
    } else {
        Selection::Cursor(new_head)
    };

    EditorState {
        selection: snap_selection_to_char_boundaries(&state.markdown, new_sel),
        ..state
    }
}

fn snap_to_char_boundary(text: &str, pos: usize) -> usize {
    let len = text.len();
    if pos >= len {
        return len;
    }
    let mut p = pos;
    while p > 0 && !text.is_char_boundary(p) {
        p -= 1;
    }
    p
}

fn snap_selection_to_char_boundaries(text: &str, sel: Selection) -> Selection {
    match sel {
        Selection::Cursor(p) => Selection::Cursor(snap_to_char_boundary(text, p)),
        Selection::Range { anchor, head } => Selection::Range {
            anchor: snap_to_char_boundary(text, anchor),
            head: snap_to_char_boundary(text, head),
        },
    }
}

fn prev_grapheme_offset(text: &str, pos: usize) -> usize {
    if pos == 0 {
        return 0;
    }
    let mut last = 0usize;
    for (idx, _) in text[..pos.min(text.len())].grapheme_indices(true) {
        last = idx;
    }
    last
}

fn next_grapheme_offset(text: &str, pos: usize) -> usize {
    let len = text.len();
    if pos >= len {
        return len;
    }
    let mut iter = text[pos..].grapheme_indices(true);
    iter.next();
    if let Some((idx, _)) = iter.next() {
        pos + idx
    } else {
        len
    }
}

/// Byte position of the start of the previous "word" relative to `pos`,
/// per Unicode word-boundary segmentation (UAX #29 via
/// `unicode-segmentation`). The "previous word" is the most recent
/// segment ending at or before `pos` that contains at least one
/// alphanumeric char; intervening punctuation / whitespace segments
/// are skipped so a cursor sitting after `"foo,  "` jumps back to the
/// `f` of `foo`, not into the comma or one of the spaces.
///
/// Returns `0` when `pos` is at start of buffer or the entire prefix
/// is non-word content (Option+Left from the start of a leading
/// whitespace / punctuation run lands at byte 0).
fn prev_word_offset(text: &str, pos: usize) -> usize {
    let pos = pos.min(text.len());
    if pos == 0 {
        return 0;
    }
    let mut last_word_start = 0;
    for (idx, seg) in text[..pos].split_word_bound_indices() {
        if seg.chars().any(|c| c.is_alphanumeric()) {
            last_word_start = idx;
        }
    }
    last_word_start
}

/// Byte position of the end of the next "word" relative to `pos`,
/// symmetric to [`prev_word_offset`]. Skips any non-word segments
/// (whitespace, punctuation) starting at `pos` and lands at the end
/// of the next segment that contains at least one alphanumeric char.
///
/// Returns `text.len()` when `pos` is at end of buffer or no word-ish
/// segment follows the cursor.
fn next_word_offset(text: &str, pos: usize) -> usize {
    let pos = pos.min(text.len());
    if pos >= text.len() {
        return text.len();
    }
    for (idx, seg) in text[pos..].split_word_bound_indices() {
        if seg.chars().any(|c| c.is_alphanumeric()) {
            return pos + idx + seg.len();
        }
    }
    text.len()
}

fn line_start_offset(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let pos = pos.min(text.len());
    let mut start = pos;
    while start > 0 && bytes[start - 1] != b'\n' {
        start -= 1;
    }
    start
}

fn line_end_offset(text: &str, pos: usize) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut end = pos.min(len);
    while end < len && bytes[end] != b'\n' {
        end += 1;
    }
    end
}

/// Vertical navigation, aware of phantom `\n`-bounded segments.
///
/// `line_start_offset` slices the buffer at every `\n`, so a structural
/// `\n\n` pair shows up as two adjacent zero-length "lines" with the
/// pair interior as the start of the second. That position is forbidden
/// (no visible row to land on), and `text.split('\n')` doesn't know it.
/// We test specifically for [`line_is_phantom`] rather than the general
/// `is_forbidden_position` so we *only* skip phantom no-row segments
/// — list-item line-start bytes are also forbidden (they collapse onto
/// the line's content edge), but the line itself is a real visible row
/// and Up/Down should land on it.
fn move_vertical(text: &str, pos: usize, direction: i32) -> usize {
    let bytes = text.as_bytes();
    let line_start = line_start_offset(text, pos);
    let column = pos - line_start;
    if direction < 0 {
        let mut probe = line_start;
        loop {
            if probe == 0 {
                return 0;
            }
            let prev_line_end = probe - 1;
            let prev_line_start = line_start_offset(text, prev_line_end);
            if line_is_phantom(bytes, prev_line_start) {
                probe = prev_line_start;
                continue;
            }
            let prev_line_len = prev_line_end - prev_line_start;
            let target = prev_line_start + column.min(prev_line_len);
            return snap_within_line(text, target, prev_line_start, prev_line_end);
        }
    } else {
        let mut probe = line_end_offset(text, pos);
        loop {
            if probe >= text.len() {
                return text.len();
            }
            let next_line_start = probe + 1;
            if line_is_phantom(bytes, next_line_start) {
                probe = line_end_offset(text, next_line_start);
                continue;
            }
            let next_line_end = line_end_offset(text, next_line_start);
            let next_line_len = next_line_end - next_line_start;
            let target = next_line_start + column.min(next_line_len);
            return snap_within_line(text, target, next_line_start, next_line_end);
        }
    }
}

/// Snap a `move_vertical` target byte to the nearest cursor-allowed
/// position inside the row `[line_start, line_end]`. The raw target is
/// `line_start + source-byte column`, which on a BQ / list line lands
/// inside the hidden chain prefix bytes (forbidden positions). Naively
/// returning the raw target would defer to `avoid_forbidden_positions`,
/// which uses the *direction of motion* to pick prev- vs next-allowed
/// — and Up arrow's "moving backward" direction snaps backward across
/// the line boundary into a row above, undoing the move. Snapping
/// within the current line keeps the cursor on the row the user
/// arrowed onto and picks the closest visible column instead.
///
/// Forward-first preference: prefer walking toward `line_end` over
/// `line_start`. The row's content edge typically sits at or after the
/// raw target (chain prefix is to the left, content to the right), so
/// forward-first lands at the natural visible column. If forward
/// search exhausts the line without finding an allowed byte, walk
/// backward as a fallback. The final fallback is `line_start`
/// (returned even if forbidden) so the function is always total —
/// `avoid_forbidden_positions` will then resolve the unreachable case
/// via the standard direction rule.
fn snap_within_line(text: &str, target: usize, line_start: usize, line_end: usize) -> usize {
    let target = snap_to_char_boundary(text, target);
    for p in target..=line_end {
        if !analysis::is_forbidden_position(text, p) {
            return p;
        }
    }
    let mut p = target;
    while p > line_start {
        p -= 1;
        if !analysis::is_forbidden_position(text, p) {
            return p;
        }
    }
    line_start
}

/// `true` when the line containing `line_start` is a phantom (no
/// rendered row in the editor's output). Used by [`move_vertical`] to
/// skip past `\n\n`-pair interiors in both top-level and chain-aware
/// (BQ / LI-wrapped-BQ) configurations.
///
/// **Why check the line terminator, not the line start.** At top level
/// a phantom line is a zero-length line, so `line_start == line_end`
/// and the two checks agree. Inside a BQ pair-run, though, every
/// prefix-only `> ` line has an interior `line_start` — including the
/// SYNTHETIC empty paragraph that the renderer paints as a visible
/// row between two paragraph breaks. The line's terminator (`\n` at
/// `line_end`, or end-of-buffer for the last line) is the canonical
/// pair boundary that *is* allowed for that synth row, and is the
/// cursor's natural resting position at "end of empty BQ line".
/// Testing the terminator means a synth row gets recognized as
/// visible and Up/Down correctly stops on it.
///
/// Phantom rows in a multi-pair run still have interior terminators
/// (the `\n` at offset 3, 9, … inside `> a\n> \n> \n> \n> b`), so this
/// rule cleanly distinguishes "row that visually exists" from "row
/// pulldown emits but the renderer collapses into a paragraph_gap".
fn line_is_phantom(bytes: &[u8], line_start: usize) -> bool {
    let len = bytes.len();
    let mut line_end = line_start.min(len);
    while line_end < len && bytes[line_end] != b'\n' {
        line_end += 1;
    }
    analysis::is_paragraph_break_interior(bytes, line_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(s: &str, cursor: usize) -> EditorState {
        EditorState {
            markdown: s.into(),
            selection: Selection::Cursor(cursor),
        }
    }

    #[test]
    fn insert_text_at_cursor() {
        let s = update(st("ab", 1), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "aXb");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn insert_newline_promotes_to_paragraph_break() {
        // Enter in the middle of a paragraph creates a paragraph break, not a
        // soft break — the post-pass enforces the invariant.
        let s = update(st("abc", 1), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "a\n\nbc");
        // Cursor was at 2 in "a\nbc"; the inserted second `\n` shifts it to 3
        // so it stays "right after the break, before 'b'".
        assert_eq!(s.selection, Selection::Cursor(3));
    }

    #[test]
    fn insert_text_replaces_selection() {
        let initial = EditorState {
            markdown: "abcdef".into(),
            selection: Selection::range(1, 4),
        };
        let s = update(initial, EditorEvent::InsertText("XX".into()));
        assert_eq!(s.markdown, "aXXef");
        assert_eq!(s.selection, Selection::Cursor(3));
    }

    #[test]
    fn delete_backward_removes_one_grapheme() {
        let s = update(st("abc", 2), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ac");
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn delete_backward_handles_multibyte() {
        let s = update(st("héllo", 3), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "hllo");
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn delete_forward_removes_one_grapheme() {
        let s = update(st("abc", 1), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "ac");
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn delete_at_start_is_a_noop() {
        let s = update(st("abc", 0), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abc");
        assert_eq!(s.selection, Selection::Cursor(0));
    }

    #[test]
    fn move_left_steps_one_grapheme() {
        let s = update(st("ab", 2), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn move_right_at_end_clamps() {
        let s = update(st("ab", 2), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn move_left_with_selection_collapses_to_lower_bound() {
        let initial = EditorState {
            markdown: "abcdef".into(),
            selection: Selection::range(1, 4),
        };
        let s = update(initial, EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn extend_right_grows_selection() {
        let s = update(st("abcd", 1), EditorEvent::ExtendRight);
        assert_eq!(s.selection, Selection::range(1, 2));
    }

    #[test]
    fn line_start_and_end() {
        // Use already-normalized markdown so move geometry isn't perturbed
        // by the post-pass promoting a soft break.
        let text = "abc\n\ndef";
        let s = update(st(text, 6), EditorEvent::MoveLineStart);
        assert_eq!(s.selection, Selection::Cursor(5));
        let s = update(st(text, 6), EditorEvent::MoveLineEnd);
        assert_eq!(s.selection, Selection::Cursor(8));
    }

    #[test]
    fn move_up_preserves_column_when_possible() {
        // Two soft-wrapped lines coupled by a hard break. MoveUp from
        // column 2 of "ghij" lands on column 2 of "abcdef" → 'c' at offset
        // 2. (Paragraph-break-separated lines are tested separately —
        // crossing the empty inter-paragraph line is its own concern.)
        let text = "abcdef  \nghij";
        let s = update(st(text, 11), EditorEvent::MoveUp);
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn document_start_and_end() {
        let text = "abc\n\ndef";
        let s = update(st(text, 6), EditorEvent::MoveDocumentStart);
        assert_eq!(s.selection, Selection::Cursor(0));
        let s = update(st(text, 0), EditorEvent::MoveDocumentEnd);
        assert_eq!(s.selection, Selection::Cursor(8));
    }

    #[test]
    fn set_selection_clamps_oob() {
        let s = update(
            st("abc", 0),
            EditorEvent::SetSelection(Selection::Cursor(99)),
        );
        assert_eq!(s.selection, Selection::Cursor(3));
    }
}

// ---------------------------------------------------------------------------
// Invariant tests — the no-soft-breaks rule + the smart-delete UX layered on
// top of it. If you're tempted to add a special-case check somewhere in the
// editor, write it as a test here first; the post-pass should be doing the
// work.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod invariant_tests {
    use super::*;

    fn st(s: &str, cursor: usize) -> EditorState {
        EditorState {
            markdown: s.into(),
            selection: Selection::Cursor(cursor),
        }
    }

    fn assert_no_soft_breaks(md: &str) {
        let bytes = md.as_bytes();
        for p in 0..bytes.len() {
            assert!(
                !is_soft_break(bytes, p),
                "soft break at byte {} in {:?}",
                p,
                md
            );
        }
    }

    // ---- Direct enforce_invariants tests --------------------------------------

    #[test]
    fn lone_newline_between_words_is_promoted() {
        let s = enforce_invariants(st("ab\ncd", 0));
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn paragraph_break_already_in_source_is_unchanged() {
        let original = st("ab\n\ncd", 4);
        let s = enforce_invariants(original.clone());
        assert_eq!(s, original);
    }

    #[test]
    fn long_newline_run_is_unchanged() {
        // Paragraph break + 2 empty paragraphs.
        let original = st("ab\n\n\n\ncd", 0);
        let s = enforce_invariants(original.clone());
        assert_eq!(s, original);
    }

    #[test]
    fn leading_newline_left_alone() {
        // CommonMark trims leading whitespace; a single leading `\n` is
        // benign and shouldn't surprise the user with a structural change.
        let s = enforce_invariants(st("\nab", 0));
        assert_eq!(s.markdown, "\nab");
    }

    #[test]
    fn trailing_newline_left_alone() {
        let s = enforce_invariants(st("ab\n", 3));
        assert_eq!(s.markdown, "ab\n");
    }

    #[test]
    fn hard_break_with_trailing_spaces_is_left_alone() {
        let s = enforce_invariants(st("ab  \ncd", 0));
        assert_eq!(s.markdown, "ab  \ncd");
    }

    #[test]
    fn hard_break_with_backslash_is_left_alone() {
        let s = enforce_invariants(st("ab\\\ncd", 0));
        assert_eq!(s.markdown, "ab\\\ncd");
    }

    #[test]
    fn multiple_lone_newlines_all_promoted_in_one_pass() {
        let s = enforce_invariants(st("a\nb\nc\nd", 0));
        assert_eq!(s.markdown, "a\n\nb\n\nc\n\nd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn cursor_at_promotion_site_shifts_with_following_content() {
        // Cursor was right before 'c' in "ab\ncd"; after promotion it
        // should still be right before 'c' (not stranded mid-break).
        let s = enforce_invariants(st("ab\ncd", 3));
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_eq!(s.selection, Selection::Cursor(4));
    }

    #[test]
    fn cursor_before_promotion_site_unchanged() {
        // Cursor before the soft break shouldn't move.
        let s = enforce_invariants(st("ab\ncd", 1));
        assert_eq!(s.selection, Selection::Cursor(1));
    }

    #[test]
    fn range_selection_endpoints_both_remapped() {
        let s = enforce_invariants(EditorState {
            markdown: "ab\ncd".into(),
            selection: Selection::range(0, 5),
        });
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_eq!(s.selection, Selection::range(0, 6));
    }

    // ---- Insert / paste flows --------------------------------------------------

    #[test]
    fn second_enter_at_paragraph_boundary_adds_another_pair() {
        // Pairs model: each Enter inserts `\n\n`. Two Enters mid-content
        // gives 4 `\n`s between content = paragraph break + 1 empty.
        let s1 = update(st("abcd", 2), EditorEvent::InsertNewline);
        assert_eq!(s1.markdown, "ab\n\ncd");
        assert_eq!(s1.selection, Selection::Cursor(4));

        let s2 = update(s1, EditorEvent::InsertNewline);
        assert_eq!(s2.markdown, "ab\n\n\n\ncd");
        assert_no_soft_breaks(&s2.markdown);
    }

    #[test]
    fn paste_with_lone_newlines_normalizes_to_paragraphs() {
        let s = update(
            st("", 0),
            EditorEvent::InsertText("line1\nline2\nline3".into()),
        );
        assert_eq!(s.markdown, "line1\n\nline2\n\nline3");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn paste_with_existing_paragraph_breaks_preserved() {
        let s = update(st("", 0), EditorEvent::InsertText("a\n\nb\n\nc".into()));
        assert_eq!(s.markdown, "a\n\nb\n\nc");
    }

    #[test]
    fn paste_with_mixed_content_normalized() {
        let s = update(st("", 0), EditorEvent::InsertText("a\nb\n\nc\nd".into()));
        assert_eq!(s.markdown, "a\n\nb\n\nc\n\nd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn paste_preserving_trailing_newline() {
        // README-style content often ends with a `\n`. Don't surprise the
        // user with an extra empty paragraph at the bottom.
        let s = update(st("", 0), EditorEvent::InsertText("hello\n".into()));
        assert_eq!(s.markdown, "hello\n");
    }

    #[test]
    fn shift_enter_emits_hard_break_in_middle() {
        let s = update(st("abcd", 2), EditorEvent::InsertLineBreak);
        assert_eq!(s.markdown, "ab  \ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Pairs model: Enter inserts `\n\n`, typing on trailing empty
    //      preserves visible empty count without any prepend trick ----------

    #[test]
    fn enter_inserts_two_newlines_at_end_of_paragraph() {
        let s = update(st("p1", 2), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "p1\n\n");
        assert_eq!(s.selection, Selection::Cursor(4));
    }

    #[test]
    fn second_enter_at_end_of_paragraph_grows_to_two_pairs() {
        let s = update(st("p1", 2), EditorEvent::InsertNewline);
        let s = update(s, EditorEvent::InsertNewline);
        // Two Enters from end: source is two `\n\n` units → 2 trailing
        // empties when rendered (`T / 2 = 2`).
        assert_eq!(s.markdown, "p1\n\n\n\n");
        assert_eq!(s.selection, Selection::Cursor(6));
    }

    #[test]
    fn typing_at_end_of_two_enter_run_preserves_visible_empty() {
        // 2 Enters from "p1" → `p1\n\n\n\n` (3 rows: p1 + 2 empties).
        // Typing X → `p1\n\n\n\nX`, which renders as p1 + 1 inter empty
        // + X = 3 rows. Visible-row count preserved without any
        // editor-side prepend.
        let mut s = st("p1", 2);
        s = update(s, EditorEvent::InsertNewline);
        s = update(s, EditorEvent::InsertNewline);
        s = update(s, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "p1\n\n\n\nX");
    }

    #[test]
    fn typing_at_single_trailing_newline_promotes_via_enforce() {
        // A *single* trailing `\n` (anomalous in the pairs model — not
        // produced by Enter — but possible via paste of "p1\n").
        // Typing X gives `p1\nX`; the soft `\n` is then promoted to
        // `\n\n` by `enforce_invariants`. The user sees 0 empties → 0
        // empties. Stable.
        let s = update(st("p1\n", 3), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "p1\n\nX");
    }

    #[test]
    fn typing_after_trailing_hard_break_fills_the_continuation_line() {
        // After Shift+Enter from end of "ab", source is "ab  \n" with a
        // hard break. The renderer shows that as a paragraph with two
        // visible lines (content + empty trailing). Typing X just fills
        // the continuation line — no extra paragraph injected.
        let s = update(st("ab  \n", 5), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "ab  \nX");
    }

    #[test]
    fn enter_at_trailing_empty_adds_one_more_pair() {
        // Pressing Enter from a trailing-empty position extends the run
        // by one pair (one more visible empty).
        let s = update(st("p1\n\n", 4), EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "p1\n\n\n\n");
    }

    #[test]
    fn shift_enter_inserts_a_hard_break_not_a_pair() {
        // Shift+Enter is the *line-break* keystroke (in-paragraph), not
        // a paragraph break, so it stays at one `\n`.
        let s = update(st("p1\n\n", 4), EditorEvent::InsertLineBreak);
        assert_eq!(s.markdown, "p1\n\n  \n");
    }

    #[test]
    fn typing_at_end_of_doc_without_trailing_newlines_appends_directly() {
        let s = update(st("p1", 2), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "p1X");
    }

    #[test]
    fn typing_in_doc_of_only_newlines_does_not_promote() {
        // No content before the trailing run, so this is leading-only.
        // Inserting X gives `\n\nX` — the leading pair is L=2 → 1 leading
        // empty above X.
        let s = update(st("\n\n", 2), EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "\n\nX");
    }

    // ---- Backspace at paragraph boundaries ------------------------------------

    #[test]
    fn backspace_at_start_of_second_paragraph_merges() {
        // Cursor right before 'c' in "ab\n\ncd" — pressing backspace at the
        // start of the "cd" paragraph collapses the break.
        let s = update(st("ab\n\ncd", 4), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn backspace_inside_two_newline_run_merges() {
        // Cursor sat between the two `\n`s of a paragraph break (a position
        // a click could land on). Backspace still merges the paragraphs in
        // one keystroke rather than feeling like a no-op.
        let s = update(st("ab\n\ncd", 3), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn backspace_through_odd_run_normalizes_via_pair_delete_plus_promotion() {
        // Source `ab\n\n\ncd` is anomalous in the pairs model (3 `\n`s,
        // odd). Backspace removes a pair, leaving `ab\ncd` — one stray
        // `\n` mid-content. `enforce_invariants` promotes that to
        // `\n\n`. Net effect: ends up as a regular paragraph break.
        let s = update(st("ab\n\n\ncd", 5), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    #[test]
    fn backspace_through_empty_paragraphs_drops_one_pair_at_a_time() {
        // Pairs model: source `ab\n\n\n\ncd` is paragraph break + 1 empty.
        // Each backspace deletes a pair (= one paragraph "unit"). First
        // press removes the empty; second press collapses the break.
        let s = update(st("ab\n\n\n\ncd", 6), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\ncd");
        let s = update(s, EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
    }

    #[test]
    fn backspace_through_three_empty_paragraphs_drops_one_pair_at_a_time() {
        // Source `ab\n\n\n\n\n\ncd` is paragraph break + 2 empties.
        let s = update(st("ab\n\n\n\n\n\ncd", 8), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\n\n\ncd");
        let s = update(s, EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab\n\ncd");
        let s = update(s, EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "abcd");
    }

    #[test]
    fn backspace_at_trailing_newline_drops_one() {
        let s = update(st("ab\n", 3), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn backspace_at_leading_paragraph_break_removes_a_pair() {
        // Pairs model: a leading run of 2 `\n`s is one leading empty.
        // Backspace at the start of the first content paragraph removes
        // the entire pair (the empty above it).
        let s = update(st("\n\nab", 2), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Delete-forward symmetric -------------------------------------------

    #[test]
    fn delete_forward_at_end_of_first_paragraph_merges() {
        let s = update(st("ab\n\ncd", 2), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn delete_forward_inside_two_newline_run_merges() {
        let s = update(st("ab\n\ncd", 3), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "abcd");
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    #[test]
    fn delete_forward_through_empty_paragraph_drops_one() {
        let s = update(st("ab\n\n\ncd", 2), EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "ab\n\ncd");
        // Cursor stayed in place; deletion happened to the right.
        assert_eq!(s.selection, Selection::Cursor(2));
    }

    // ---- Range selection deletion -------------------------------------------

    #[test]
    fn selection_replacing_paragraph_break_with_text() {
        let initial = EditorState {
            markdown: "ab\n\ncd".into(),
            selection: Selection::range(1, 5),
        };
        let s = update(initial, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "aXd");
    }

    #[test]
    fn selection_deleting_one_newline_of_break_repromotes() {
        // The user selected exactly one `\n` of a `\n\n` break and pressed
        // delete. The post-pass promotes the lone `\n` back, so the doc is
        // unchanged but the cursor is somewhere sensible. (UX could be
        // smarter here, but consistency-of-source is the load-bearing
        // property; we won't ship a doc with a soft break in it.)
        let initial = EditorState {
            markdown: "ab\n\ncd".into(),
            selection: Selection::range(2, 3),
        };
        let s = update(initial, EditorEvent::DeleteForward);
        assert_eq!(s.markdown, "ab\n\ncd");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Hard breaks --------------------------------------------------------

    #[test]
    fn hard_break_with_typed_content_after_stays_intact() {
        let s1 = update(st("ab", 2), EditorEvent::InsertLineBreak);
        assert_eq!(s1.markdown, "ab  \n");
        let s2 = update(s1, EditorEvent::InsertText("X".into()));
        assert_eq!(s2.markdown, "ab  \nX");
        assert_no_soft_breaks(&s2.markdown);
    }

    #[test]
    fn deleting_one_trailing_space_of_hard_break_promotes_to_paragraph_break() {
        // The user backspaced over one of the trailing spaces, so the
        // sequence is no longer a hard break. The lone `\n` becomes a soft
        // break, and the post-pass turns the break into a paragraph break.
        // This is a deliberate, documented behavior: we never silently
        // ship a soft break.
        let s = update(st("ab  \nX", 4), EditorEvent::DeleteBackward);
        assert_eq!(s.markdown, "ab \n\nX");
        assert_no_soft_breaks(&s.markdown);
    }

    // ---- Movement doesn't introduce soft breaks ----------------------------

    #[test]
    fn navigation_events_preserve_already_clean_markdown() {
        let starts = [
            "ab\n\ncd",
            "ab\n\n\ncd",
            "ab  \ncd",
            "ab\\\ncd",
            "\n\nab",
            "ab\n",
        ];
        for src in starts {
            for evt in [
                EditorEvent::MoveLeft,
                EditorEvent::MoveRight,
                EditorEvent::MoveLineStart,
                EditorEvent::MoveLineEnd,
                EditorEvent::MoveDocumentStart,
                EditorEvent::MoveDocumentEnd,
            ] {
                let s = update(st(src, src.len() / 2), evt);
                assert_eq!(s.markdown, src, "navigation altered markdown {src:?}");
                assert_no_soft_breaks(&s.markdown);
            }
        }
    }

    // ---- Stress / fuzz-ish ---------------------------------------------------

    #[test]
    fn typing_pause_typing_preserves_invariant() {
        // Insert "a" → Enter → "b" → Enter → "c". Each step's output is
        // re-validated.
        let mut s = st("", 0);
        let steps: &[EditorEvent] = &[
            EditorEvent::InsertText("a".into()),
            EditorEvent::InsertNewline,
            EditorEvent::InsertText("b".into()),
            EditorEvent::InsertNewline,
            EditorEvent::InsertText("c".into()),
        ];
        for evt in steps {
            s = update(s, evt.clone());
            assert_no_soft_breaks(&s.markdown);
        }
        assert_eq!(s.markdown, "a\n\nb\n\nc");
    }

    #[test]
    fn alternating_insert_and_delete_preserves_invariant() {
        let mut s = st("hello world", 5);
        let steps = [
            EditorEvent::InsertNewline,
            EditorEvent::DeleteBackward,
            EditorEvent::InsertNewline,
            EditorEvent::InsertNewline,
            EditorEvent::DeleteBackward,
            EditorEvent::DeleteForward,
        ];
        for evt in steps {
            s = update(s, evt);
            assert_no_soft_breaks(&s.markdown);
        }
    }
}

// ---------------------------------------------------------------------------
// Forbidden-position invariant — the cursor must never sit inside a
// structural `\n\n` pair. Pair interiors are visually unreachable (the
// pair renders as one row), and typing there would split the pair into a
// stray odd-length newline run. Every state transition in `update()` runs
// the post-pass that snaps offending cursors away, with the snap
// direction determined by where the cursor came from.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod forbidden_position_tests {
    use super::*;

    fn st(s: &str, cursor: usize) -> EditorState {
        EditorState {
            markdown: s.into(),
            selection: Selection::Cursor(cursor),
        }
    }

    fn assert_no_forbidden(state: &EditorState) {
        let positions: Vec<usize> = match state.selection {
            Selection::Cursor(p) => vec![p],
            Selection::Range { anchor, head } => vec![anchor, head],
        };
        for p in positions {
            assert!(
                !is_forbidden_position(&state.markdown, p),
                "selection endpoint {p} is forbidden in {:?}",
                state.markdown
            );
        }
    }

    // ---- The detector itself -----------------------------------------

    #[test]
    fn paragraph_break_interior_is_forbidden() {
        let src = "p1\n\np2";
        assert!(is_forbidden_position(src, 3));
        assert!(!is_forbidden_position(src, 0));
        assert!(!is_forbidden_position(src, 2));
        assert!(!is_forbidden_position(src, 4));
        assert!(!is_forbidden_position(src, 6));
    }

    #[test]
    fn between_two_pairs_is_allowed() {
        // `p1\n\n\n\np2`: 4-newline run = 2 pairs. Position 4 is the
        // boundary between the two pairs (allowed), positions 3 and 5
        // are pair interiors (forbidden).
        let src = "p1\n\n\n\np2";
        assert!(is_forbidden_position(src, 3));
        assert!(!is_forbidden_position(src, 4));
        assert!(is_forbidden_position(src, 5));
    }

    #[test]
    fn six_newline_run_alternates_forbidden_and_allowed() {
        let src = "p1\n\n\n\n\n\np2";
        assert!(is_forbidden_position(src, 3));
        assert!(!is_forbidden_position(src, 4));
        assert!(is_forbidden_position(src, 5));
        assert!(!is_forbidden_position(src, 6));
        assert!(is_forbidden_position(src, 7));
        assert!(!is_forbidden_position(src, 8)); // start of p2
    }

    #[test]
    fn leading_pair_interior_is_forbidden() {
        let src = "\n\nab";
        assert!(is_forbidden_position(src, 1));
        assert!(!is_forbidden_position(src, 0));
        assert!(!is_forbidden_position(src, 2));
    }

    #[test]
    fn hard_break_alone_has_no_forbidden_positions() {
        // `ab  \n` — single hard break, no structural pair.
        let src = "ab  \n";
        for p in 0..=src.len() {
            assert!(
                !is_forbidden_position(src, p),
                "p={p} unexpectedly forbidden"
            );
        }
    }

    #[test]
    fn hard_break_followed_by_structural_pair_only_marks_pair_interior() {
        // `ab  \n\n\ncd` — hard break (\n at 4) + 2 structural \n's
        // (positions 5, 6). The structural pair is [5, 7). Position 6
        // is the pair interior (forbidden); the hard-break \n doesn't
        // pull position 5 into being forbidden because the back-walk
        // stops at the hard-break terminator.
        let src = "ab  \n\n\ncd";
        assert!(!is_forbidden_position(src, 5));
        assert!(is_forbidden_position(src, 6));
        assert!(!is_forbidden_position(src, 7));
    }

    #[test]
    fn backslash_hard_break_treated_same() {
        // `ab\\\n\n\ncd` — backslash + \n is a hard break; same shape.
        let src = "ab\\\n\n\ncd";
        // bytes: a(0) b(1) \\(2) \n(3) \n(4) \n(5) c(6) d(7)
        // Hard-break \n at 3 (preceded by \\ at 2). Structural pair
        // [4, 6); interior is 5.
        assert!(!is_forbidden_position(src, 4));
        assert!(is_forbidden_position(src, 5));
        assert!(!is_forbidden_position(src, 6));
    }

    #[test]
    fn doc_edges_are_never_forbidden() {
        let src = "\n\n";
        assert!(!is_forbidden_position(src, 0));
        assert!(!is_forbidden_position(src, src.len()));
    }

    // ---- Movement directional snapping --------------------------------

    #[test]
    fn right_arrow_skips_paragraph_break_interior() {
        let s = update(st("p1\n\np2", 2), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn left_arrow_skips_paragraph_break_interior() {
        let s = update(st("p1\n\np2", 4), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(2));
        assert_no_forbidden(&s);
    }

    #[test]
    fn right_arrow_through_extra_pair_lands_at_inter_pair_boundary() {
        // `p1\n\n\n\np2` — Right from end of p1 (byte 2) skips byte 3
        // (forbidden) and lands at byte 4 (between the two pairs,
        // allowed; visually on the empty row).
        let s = update(st("p1\n\n\n\np2", 2), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn left_arrow_through_extra_pair_lands_at_inter_pair_boundary() {
        let s = update(st("p1\n\n\n\np2", 6), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn right_arrow_off_inter_pair_boundary_lands_at_next_real_block() {
        // From the inter-pair boundary (byte 4), Right would land at
        // byte 5 (forbidden, interior of pair 2). Snap forward → 6
        // (start of p2).
        let s = update(st("p1\n\n\n\np2", 4), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(6));
        assert_no_forbidden(&s);
    }

    #[test]
    fn left_arrow_off_inter_pair_boundary_lands_at_prev_real_block() {
        let s = update(st("p1\n\n\n\np2", 4), EditorEvent::MoveLeft);
        assert_eq!(s.selection, Selection::Cursor(2));
        assert_no_forbidden(&s);
    }

    #[test]
    fn down_from_first_paragraph_skips_break_to_second() {
        // `p1\n\np2` MoveDown from byte 0: target byte 3 (forbidden).
        // Snap forward → byte 4 (start of p2).
        let s = update(st("p1\n\np2", 0), EditorEvent::MoveDown);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn up_from_second_paragraph_skips_break_to_first() {
        // `move_vertical` skips the phantom (forbidden) line between
        // paragraphs and preserves column 0, so Up from start of p2
        // lands at start of p1, not at end-of-p1 like a simple
        // post-snap would produce.
        let s = update(st("p1\n\np2", 4), EditorEvent::MoveUp);
        assert_eq!(s.selection, Selection::Cursor(0));
        assert_no_forbidden(&s);
    }

    #[test]
    fn down_from_first_paragraph_lands_on_visible_empty_row() {
        // `p1\n\n\n\np2` MoveDown from p1@0: target → 3 (forbidden) →
        // snap forward to 4. Byte 4 is on the empty row (synthetic
        // empty has range 3..5).
        let s = update(st("p1\n\n\n\np2", 0), EditorEvent::MoveDown);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn up_from_p2_in_long_run_lands_on_empty_row() {
        let s = update(st("p1\n\n\n\np2", 6), EditorEvent::MoveUp);
        assert_eq!(s.selection, Selection::Cursor(4));
        assert_no_forbidden(&s);
    }

    // ---- ExtendLeft / ExtendRight (range selections) ------------------

    #[test]
    fn extend_right_skips_forbidden_interior() {
        // From cursor at byte 2 in `p1\n\np2`, Shift+Right should extend
        // the selection past the forbidden byte 3 to end at byte 4.
        let s = update(st("p1\n\np2", 2), EditorEvent::ExtendRight);
        assert_eq!(s.selection, Selection::range(2, 4));
        assert_no_forbidden(&s);
    }

    #[test]
    fn extend_left_skips_forbidden_interior() {
        let s = update(st("p1\n\np2", 4), EditorEvent::ExtendLeft);
        assert_eq!(s.selection, Selection::range(4, 2));
        assert_no_forbidden(&s);
    }

    // ---- SetSelection (the host API) ----------------------------------
    //
    // SetSelection uses *nearest-allowed* (idempotent), not prev-comparison.
    // That keeps mouse-drag stable: every `mouse_move` fires another
    // `SetSelection` with the same offset, and each must produce the
    // same result regardless of where the previously-snapped cursor
    // ended up. See the comment at the top of `set_selection`.

    #[test]
    fn set_selection_to_forbidden_snaps_to_nearest_allowed() {
        // pos=3 in `p1\n\np2`: equidistant between allowed 2 and 4.
        // Forward wins ties, matching the post-pass default. Same
        // result regardless of where the cursor was before — that's
        // the load-bearing idempotence property.
        for prev in [0, 1, 2, 3, 4, 5, 6] {
            let s = update(
                st("p1\n\np2", prev),
                EditorEvent::SetSelection(Selection::Cursor(3)),
            );
            assert_eq!(
                s.selection,
                Selection::Cursor(4),
                "prev={prev} gave {:?}",
                s.selection
            );
        }
    }

    #[test]
    fn repeated_set_selection_at_forbidden_offset_is_stable() {
        // Regression for the mouse-drag flicker: a `MouseMoveEvent`
        // re-fires `SetSelection` at the same offset on every frame.
        // Under prev-comparison the cursor would oscillate between the
        // two adjacent allowed positions (2 and 4) because each call
        // saw the previous snap as `prev`. Nearest-allowed makes it
        // converge in one step.
        let mut s = st("p1\n\np2", 0);
        for _ in 0..5 {
            s = update(s, EditorEvent::SetSelection(Selection::Cursor(3)));
            assert_eq!(s.selection, Selection::Cursor(4));
        }
    }

    #[test]
    fn set_selection_inside_inter_block_pair_lands_on_visible_empty_row() {
        // The user-reported case: clicking on a synthetic empty row
        // between paragraphs. The renderer's `display_to_source` for
        // the empty maps clicks to the pair interior (forbidden);
        // nearest-allowed snaps to the inter-pair boundary, which is
        // visually the same row.
        let s = update(
            st("p1\n\n\n\np2", 0),
            EditorEvent::SetSelection(Selection::Cursor(3)),
        );
        // Interior of pair 1; equidistant between 2 (end of p1) and 4
        // (visible empty row). Forward wins → 4.
        assert_eq!(s.selection, Selection::Cursor(4));
    }

    // ---- Range endpoints snap independently ---------------------------

    #[test]
    fn range_anchor_and_head_snap_independently() {
        // Both endpoints land on forbidden interiors of pair 1 / pair
        // 2. nearest-allowed snaps each independently — anchor 3 is
        // tied between 2 and 4 (forward → 4); head 5 is tied between
        // 4 and 6 (forward → 6). Result: a real range, not a collapse.
        let initial = EditorState {
            markdown: "p1\n\n\n\np2".into(),
            selection: Selection::Cursor(0),
        };
        let s = update(
            initial,
            EditorEvent::SetSelection(Selection::Range { anchor: 3, head: 5 }),
        );
        assert_eq!(s.selection, Selection::range(4, 6));
    }

    // ---- Hard breaks pass through unchanged --------------------------

    #[test]
    fn navigation_around_hard_break_is_unaffected() {
        // `ab  \nX` cursor right after the hard-break \n. Right arrow
        // moves to byte 6 (after X). No forbidden positions in this doc.
        let s = update(st("ab  \nX", 5), EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(6));
        assert_no_forbidden(&s);
    }

    // ---- Empty-row navigation round-trip ------------------------------

    // ---- Trailing-empty typing semantics ------------------------------
    //
    // The trailing-empty layout shifts each pair by 1 inside the gap so
    // the cursor's resting position when on row N is the typing position
    // that creates content for row N. With offset 0, "cursor on empty
    // row" and "cursor at end of paragraph" share a source offset, and
    // typing extends the paragraph instead.

    #[test]
    fn enter_at_end_of_paragraph_then_type_creates_new_paragraph_for_empty_row() {
        // The user-flow regression. Enter from end of `paragraph` puts
        // the cursor on the empty row below. Typing X must create a new
        // "X" paragraph on that row, not extend "paragraph" to
        // "paragraphX".
        let mut s = st("paragraph", 9);
        s = update(s, EditorEvent::InsertNewline);
        assert_eq!(s.markdown, "paragraph\n\n");
        assert_eq!(s.selection, Selection::Cursor(11));
        s = update(s, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "paragraph\n\nX");
    }

    #[test]
    fn right_arrow_from_end_of_paragraph_into_trailing_empty_then_type() {
        // Cursor at end of paragraph, Right lands on the empty row, X
        // creates a new paragraph for it.
        let mut s = st("paragraph\n\n\n\n\n\n", 9);
        s = update(s, EditorEvent::MoveRight);
        assert_eq!(s.selection, Selection::Cursor(11));
        s = update(s, EditorEvent::InsertText("X".into()));
        assert_eq!(s.markdown, "paragraph\n\nX\n\n\n\n");
        assert_no_forbidden(&s);
    }

    #[test]
    fn click_offset_at_trailing_empty_anchor_snaps_to_typing_position() {
        // Trailing empties have `display_to_source = [block.start]`,
        // and `block.start` is the *forbidden* interior of the
        // structural pair. Clicking on the empty row therefore feeds
        // `SetSelection(block.start)` through `update`, which uses
        // nearest-allowed and lands at the empty's strict-interior
        // typing position.
        let s = update(
            st("paragraph\n\n\n\n\n\n", 0),
            EditorEvent::SetSelection(Selection::Cursor(10)),
        );
        // Empty 1 is range 10..12; click anchor is 10 (forbidden).
        // Nearest allowed: prev=9 (end of paragraph) and next=11
        // (typing position). Forward wins → 11.
        assert_eq!(s.selection, Selection::Cursor(11));
    }

    #[test]
    fn arrow_round_trip_stays_clear_of_forbidden() {
        // Walk Right from the start of `p1\n\n\n\n\n\np2` and back
        // again; every step must produce an allowed cursor.
        let mut s = st("p1\n\n\n\n\n\np2", 0);
        let path = [
            EditorEvent::MoveRight, // 0 → 1
            EditorEvent::MoveRight, // 1 → 2
            EditorEvent::MoveRight, // 2 → skip 3 → 4
            EditorEvent::MoveRight, // 4 → skip 5 → 6
            EditorEvent::MoveRight, // 6 → skip 7 → 8 (start of p2)
            EditorEvent::MoveRight, // 8 → 9
            EditorEvent::MoveRight, // 9 → 10
        ];
        let expected = [1usize, 2, 4, 6, 8, 9, 10];
        for (evt, want) in path.into_iter().zip(expected) {
            s = update(s, evt);
            assert_eq!(s.selection, Selection::Cursor(want));
            assert_no_forbidden(&s);
        }
        // Walk back to the start.
        let back = [
            EditorEvent::MoveLeft, // 10 → 9
            EditorEvent::MoveLeft, // 9 → 8
            EditorEvent::MoveLeft, // 8 → skip 7 → 6
            EditorEvent::MoveLeft, // 6 → skip 5 → 4
            EditorEvent::MoveLeft, // 4 → skip 3 → 2
            EditorEvent::MoveLeft, // 2 → 1
            EditorEvent::MoveLeft, // 1 → 0
        ];
        let expected = [9usize, 8, 6, 4, 2, 1, 0];
        for (evt, want) in back.into_iter().zip(expected) {
            s = update(s, evt);
            assert_eq!(s.selection, Selection::Cursor(want));
            assert_no_forbidden(&s);
        }
    }
}
