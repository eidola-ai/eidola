//! GFM pipe-table geometry and editing computations.
//!
//! # The table model
//!
//! A table is a **structural leaf block** whose internal single `\n`s are
//! sacred: rows are separated by exactly one `\n` (GFM's own shape), which
//! the canonicalizer's soft-break promotion must never inflate to `\n\n`.
//! The source stays *minimal canonical GFM* at all times — `| a | b |`
//! rows with single-space cell padding and a compact delimiter row
//! (`| --- | :-: |`); the editor never writes org-mode-style physical
//! space alignment into the buffer. Visual column alignment is a
//! display-time concern (`element.rs` inserts width-matched pad
//! substitutions), so the file round-trips as clean GFM.
//!
//! # Geometry
//!
//! [`TableGeometry`] is the line/cell map every table rule operates on.
//! It is computed **from the source bytes** (an unescaped-pipe line
//! scanner), not from pulldown's cell events — pulldown never emits the
//! delimiter row, synthesizes degenerate ranges for missing cells, and
//! reports untrimmed cell ranges; scanning the lines ourselves gives one
//! consistent, fully-covering map. Pulldown remains the *authority on
//! whether* a table exists (the scanner only runs inside a parsed
//! `NodeKind::Table` range), so the two can't disagree about what is and
//! isn't a table.
//!
//! # Allowed cursor positions
//!
//! Between-cell chrome (`| ` / ` | ` / ` |`) is forbidden cursor
//! territory, exactly like the hidden list-indent bytes: the caret lives
//! at cell-content edges only, and the standard snapping machinery
//! (`analysis::is_forbidden_position` consumers) walks across the chrome
//! in one step. This is what makes Left/Right hop cell-to-cell and keeps
//! clicks landing inside cells. The delimiter row's cells (`:---:` runs)
//! are ordinary content — the markdown-fluent way to change a column's
//! alignment is to edit its colons in place.
//!
//! V1 scope gate: all *editing* rules (and the render-side grid) apply to
//! **top-level** tables only. A table nested in a blockquote or list item
//! still parses (and its newlines are still protected), but renders as
//! plain source lines — honest raw markdown rather than a half-working
//! grid.

use std::ops::Range;

use crate::analysis::SourceEdit;
use crate::syntax::{NodeKind, SyntaxNode, TableAlignment};

/// Which of the three GFM row kinds a line is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// The first line — the header row.
    Header,
    /// The second line — `| --- | :-: |`. Never navigated by Tab, but
    /// its cells are editable content (colons = alignment).
    Delimiter,
    /// Any later line.
    Body,
}

/// One table row: its source line (exclusive of the trailing `\n`) and
/// the trimmed content range of each cell, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGeometry {
    pub line: Range<usize>,
    pub kind: RowKind,
    /// Trimmed cell-content ranges (absolute byte offsets). An empty
    /// cell contributes an empty range positioned at its canonical
    /// caret point (one byte into the padding when available).
    pub cells: Vec<Range<usize>>,
    /// The *untrimmed* segments between the boundary pipes, one per
    /// cell (absolute). `cells[k]` is `segments[k]` minus its
    /// whitespace padding. The allowed-cursor rule reads these: a
    /// caret may sit anywhere strictly inside a segment (so a
    /// just-typed trailing space is a legal caret home until the
    /// canonicalizer trims it), while the segment edges and the pipe
    /// bytes are chrome.
    pub segments: Vec<Range<usize>>,
}

/// The full line/cell map of one table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableGeometry {
    /// The `NodeKind::Table` node's range — the whole construct
    /// including the trailing `\n` when present.
    pub range: Range<usize>,
    /// Per-column alignment, parsed from the delimiter row. Padded
    /// with `TableAlignment::None` when other rows are wider.
    pub alignments: Vec<TableAlignment>,
    /// Header, delimiter, then body rows, in source order.
    pub rows: Vec<RowGeometry>,
}

impl TableGeometry {
    /// Column count — the widest row wins (the canonicalizer pads all
    /// rows up to this, losslessly).
    pub fn column_count(&self) -> usize {
        self.rows.iter().map(|r| r.cells.len()).max().unwrap_or(0)
    }

    /// Alignment for column `col` (`None` past the parsed set).
    pub fn alignment(&self, col: usize) -> TableAlignment {
        self.alignments
            .get(col)
            .copied()
            .unwrap_or(TableAlignment::None)
    }

    /// Index of the row whose line contains `pos` (inclusive of the
    /// line-end position).
    pub fn row_at(&self, pos: usize) -> Option<usize> {
        self.rows
            .iter()
            .position(|r| pos >= r.line.start && pos <= r.line.end)
    }

    /// `(row, cell)` whose trimmed content range contains `pos`
    /// (inclusive of both edges). Falls back to the nearest cell on
    /// the row when `pos` sits in chrome.
    pub fn cell_at(&self, pos: usize) -> Option<(usize, usize)> {
        let row_idx = self.row_at(pos)?;
        let row = &self.rows[row_idx];
        if row.cells.is_empty() {
            return None;
        }
        // Exact containment first.
        if let Some(c) = row
            .cells
            .iter()
            .position(|c| pos >= c.start && pos <= c.end)
        {
            return Some((row_idx, c));
        }
        // Chrome: nearest cell by distance to content edges.
        let mut best = 0usize;
        let mut best_dist = usize::MAX;
        for (i, c) in row.cells.iter().enumerate() {
            let d = if pos < c.start {
                c.start - pos
            } else {
                pos - c.end
            };
            if d < best_dist {
                best_dist = d;
                best = i;
            }
        }
        Some((row_idx, best))
    }
}

/// Scan one table line into per-cell trimmed content ranges
/// (line-local offsets). Handles optional leading/trailing pipes and
/// `\|` escapes. Empty input yields a single empty cell at 0.
pub fn scan_line_cells(line: &str) -> Vec<Range<usize>> {
    scan_line(line).into_iter().map(|(_, cell)| cell).collect()
}

/// Like [`scan_line_cells`], returning `(segment, trimmed_cell)`
/// pairs — the untrimmed between-pipe segment alongside the trimmed
/// content range.
pub fn scan_line(line: &str) -> Vec<(Range<usize>, Range<usize>)> {
    let bytes = line.as_bytes();
    // Boundary pipe positions (unescaped `|`).
    let mut pipes: Vec<usize> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2, // skip escaped char (incl. `\|`)
            b'|' => {
                pipes.push(i);
                i += 1;
            }
            _ => i += 1,
        }
    }
    // Segment edges: leading pipe (if the first non-blank char is a
    // pipe) starts the first segment after it; otherwise the first
    // segment starts at 0. Symmetric at the end.
    let first_content = bytes
        .iter()
        .position(|b| !matches!(b, b' ' | b'\t'))
        .unwrap_or(bytes.len());
    let last_content = bytes
        .iter()
        .rposition(|b| !matches!(b, b' ' | b'\t'))
        .map(|p| p + 1)
        .unwrap_or(0);
    let has_leading_pipe = pipes.first().copied() == Some(first_content);
    let has_trailing_pipe = last_content > 0
        && pipes.last().copied() == Some(last_content - 1)
        && last_content > first_content + 1;

    let mut seg_start = if has_leading_pipe { pipes[0] + 1 } else { 0 };
    let inner_pipes: &[usize] = {
        let lo = if has_leading_pipe { 1 } else { 0 };
        let hi = if has_trailing_pipe {
            pipes.len().saturating_sub(1)
        } else {
            pipes.len()
        };
        if lo <= hi { &pipes[lo..hi] } else { &[] }
    };
    let mut segments: Vec<Range<usize>> = Vec::new();
    for &p in inner_pipes {
        segments.push(seg_start..p);
        seg_start = p + 1;
    }
    let last_end = if has_trailing_pipe {
        *pipes.last().unwrap()
    } else {
        bytes.len()
    };
    segments.push(seg_start..last_end.max(seg_start));

    // Trim each segment to content; an empty cell keeps a canonical
    // caret point one byte into the padding when available.
    segments
        .into_iter()
        .map(|seg| {
            let mut s = seg.start;
            let mut e = seg.end;
            while s < e && matches!(bytes[s], b' ' | b'\t') {
                s += 1;
            }
            while e > s && matches!(bytes[e - 1], b' ' | b'\t') {
                e -= 1;
            }
            let cell = if s >= e {
                // Empty: place the caret point one byte into the
                // padding (so typing yields `| x |`-shaped source).
                let point = (seg.start + 1).min(seg.end);
                point..point
            } else {
                s..e
            };
            (seg, cell)
        })
        .collect()
}

/// Parse the alignment of one delimiter cell (`:---:` etc.).
pub fn parse_alignment(cell: &str) -> TableAlignment {
    let leading = cell.starts_with(':');
    let trailing = cell.ends_with(':') && cell.len() > 1;
    match (leading, trailing) {
        (true, true) => TableAlignment::Center,
        (true, false) => TableAlignment::Left,
        (false, true) => TableAlignment::Right,
        (false, false) => TableAlignment::None,
    }
}

/// Canonical delimiter-cell text for an alignment.
pub fn alignment_cell(a: TableAlignment) -> &'static str {
    match a {
        TableAlignment::None => "---",
        TableAlignment::Left => ":--",
        TableAlignment::Center => ":-:",
        TableAlignment::Right => "--:",
    }
}

/// Canonical row text for a list of cell contents: `| a | b |`, with
/// an empty cell rendering as `|  |`.
pub fn canonical_row(cells: &[&str]) -> String {
    let mut out = String::from("|");
    for c in cells {
        out.push(' ');
        out.push_str(c);
        out.push(' ');
        out.push('|');
    }
    out
}

/// Canonical delimiter row for `n` columns with the given alignments
/// (padded with `None`).
pub fn canonical_delimiter_row(alignments: &[TableAlignment], n: usize) -> String {
    let cells: Vec<&str> = (0..n)
        .map(|i| alignment_cell(alignments.get(i).copied().unwrap_or(TableAlignment::None)))
        .collect();
    canonical_row(&cells)
}

/// Canonical empty body row for `n` columns (`|  |  |`).
pub fn empty_row(n: usize) -> String {
    let cells: Vec<&str> = (0..n).map(|_| "").collect();
    canonical_row(&cells)
}

/// Byte offset of the first cell's caret point within a canonical
/// empty row — after the leading `| `.
pub const EMPTY_ROW_FIRST_CELL_OFFSET: usize = 2;

/// Build a [`TableGeometry`] from source bytes and a parsed
/// `NodeKind::Table` node's range + alignments.
pub fn geometry_from_range(
    source: &str,
    range: Range<usize>,
    parsed_alignments: &[TableAlignment],
) -> TableGeometry {
    let mut rows = Vec::new();
    let end = range.end.min(source.len());
    let mut pos = range.start.min(end);
    let mut idx = 0usize;
    while pos < end {
        let line_end = source[pos..end].find('\n').map(|i| pos + i).unwrap_or(end);
        let line_str = &source[pos..line_end];
        let kind = match idx {
            0 => RowKind::Header,
            1 => RowKind::Delimiter,
            _ => RowKind::Body,
        };
        let scanned = scan_line(line_str);
        let cells = scanned
            .iter()
            .map(|(_, c)| pos + c.start..pos + c.end)
            .collect();
        let segments = scanned
            .iter()
            .map(|(s, _)| pos + s.start..pos + s.end)
            .collect();
        rows.push(RowGeometry {
            line: pos..line_end,
            kind,
            cells,
            segments,
        });
        idx += 1;
        pos = line_end + 1;
    }
    // Alignment authority: parse the actual delimiter row so an
    // in-progress colon edit is reflected immediately; fall back to
    // pulldown's parsed alignments.
    let alignments = rows
        .iter()
        .find(|r| r.kind == RowKind::Delimiter)
        .map(|r| {
            r.cells
                .iter()
                .map(|c| parse_alignment(&source[c.clone()]))
                .collect()
        })
        .unwrap_or_else(|| parsed_alignments.to_vec());
    TableGeometry {
        range,
        alignments,
        rows,
    }
}

/// Find the **top-level** `Table` node containing `pos` (inclusive of
/// the construct's end) and build its geometry. Tables nested inside
/// blockquotes / lists are deliberately not returned — the v1 editing
/// rules apply to top-level tables only.
pub fn table_at(source: &str, tree: &[SyntaxNode], pos: usize) -> Option<TableGeometry> {
    for node in tree {
        if let NodeKind::Table { alignments } = &node.kind {
            // Inclusive end: a cursor parked at the construct's
            // boundary is "inside" (the click-to-edit rule shared
            // with display math).
            if pos >= node.range.start && pos <= node.range.end {
                return Some(geometry_from_range(source, node.range.clone(), alignments));
            }
        }
    }
    None
}

/// All `Table` node ranges anywhere in the tree (any nesting depth).
/// Used by the soft-break exemption — a table's internal newlines are
/// protected regardless of nesting.
pub fn table_ranges_in_tree(tree: &[SyntaxNode]) -> Vec<Range<usize>> {
    fn walk(nodes: &[SyntaxNode], out: &mut Vec<Range<usize>>) {
        for n in nodes {
            if matches!(n.kind, NodeKind::Table { .. }) {
                out.push(n.range.clone());
            }
            walk(&n.children, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

/// Is the `\n` at byte `p` an **internal** row separator of some
/// table — i.e. exempt from soft-break promotion? The table's final
/// trailing `\n` (its boundary with whatever follows) is *not*
/// internal.
pub fn newline_is_table_internal(ranges: &[Range<usize>], p: usize) -> bool {
    ranges.iter().any(|r| p >= r.start && p + 1 < r.end)
}

/// Is the byte at `pos` escaped by a backslash? Escaping is the
/// **odd-parity** of the run of consecutive backslashes immediately
/// before `pos` — `\|` escapes the pipe, `\\|` does not (the two
/// backslashes escape each other), `\\\|` does again, and so on.
/// This must agree with [`scan_line_cells`]' pipe classification
/// (which consumes backslash-escaped pairs), or a typed `|` could be
/// treated as literal by the keystroke handler while the scanner
/// sees a cell boundary.
pub fn is_escaped_at(bytes: &[u8], pos: usize) -> bool {
    let mut run = 0usize;
    while run < pos && bytes[pos - 1 - run] == b'\\' {
        run += 1;
    }
    run % 2 == 1
}

/// Delimiter-row dash floor: would deleting `del` leave the caret's
/// delimiter cell without a single dash? Deleting the last hyphen of
/// a delimiter cell silently dissolves the whole table (an empty or
/// colon-only delimiter cell fails the GFM shape), so grapheme- and
/// word-level deletion refuse to cross the floor — dissolving a
/// table is an explicit act (delete the delimiter row's line), never
/// a keystroke accident. Colons remain freely deletable (alignment
/// editing).
pub fn delimiter_dash_floor(geo: &TableGeometry, source: &str, del: &Range<usize>) -> bool {
    let Some(row_idx) = geo.row_at(del.start) else {
        return false;
    };
    let row = &geo.rows[row_idx];
    if row.kind != RowKind::Delimiter {
        return false;
    }
    // Find the cell the deletion intersects.
    for cell in &row.cells {
        if del.start < cell.end && del.end > cell.start {
            let text = &source[cell.clone()];
            let deleted_dashes = source[del.start.max(cell.start)..del.end.min(cell.end)]
                .bytes()
                .filter(|&b| b == b'-')
                .count();
            let total_dashes = text.bytes().filter(|&b| b == b'-').count();
            return deleted_dashes >= total_dashes;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Forbidden positions (between-cell chrome)
// ---------------------------------------------------------------------------

/// Is `pos` inside a top-level table's between-cell chrome — i.e. not
/// at a cell-content edge (or inside cell content), and not at the
/// construct's outer boundary? Such positions snap to the nearest
/// content edge via the standard forbidden-position machinery.
pub fn is_table_chrome_interior(source: &str, tree: &[SyntaxNode], pos: usize) -> bool {
    let Some(geo) = table_at(source, tree, pos) else {
        return false;
    };
    // The construct's outer boundaries are always allowed (they're
    // how the cursor approaches from neighboring blocks).
    if pos <= geo.range.start || pos >= geo.range.end {
        return false;
    }
    let Some(row_idx) = geo.row_at(pos) else {
        // Between lines: `pos` sits on a `\n` boundary. The position
        // *after* a line's `\n` is the next line's start, which
        // `row_at` covers; a position outside all rows shouldn't
        // happen, allow it.
        return false;
    };
    let row = &geo.rows[row_idx];
    // Allowed: anywhere in a cell's trimmed content, and anywhere
    // *strictly inside* its untrimmed segment. The strict interior
    // matters mid-typing — a just-typed trailing space parks the
    // caret one byte past the trimmed content, which must remain a
    // legal home or the snap would yank the caret to the next cell
    // (the canonicalizer trims the extra padding once the caret
    // leaves the row). Segment edges (the canonical single-space
    // padding's outer bytes), the pipes themselves, and the row's
    // line-end position stay chrome.
    for (c, seg) in row.cells.iter().zip(row.segments.iter()) {
        if pos >= c.start && pos <= c.end {
            return false;
        }
        if pos > seg.start && pos < seg.end {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Structural edits
// ---------------------------------------------------------------------------

/// A computed structural table edit: source edits (sorted,
/// non-overlapping) plus an explicit caret placement in
/// **post-edit** coordinates, and optionally a selection range
/// (Tab selecting the target cell's content).
#[derive(Debug, Clone)]
pub struct TableEdit {
    pub edits: Vec<SourceEdit>,
    /// Caret (or selection head) in post-edit byte offsets.
    pub cursor: usize,
    /// Selection anchor in post-edit offsets, when the edit selects
    /// a range (Tab landing on a non-empty cell).
    pub anchor: Option<usize>,
}

impl TableEdit {
    fn caret(edits: Vec<SourceEdit>, cursor: usize) -> Self {
        Self {
            edits,
            cursor,
            anchor: None,
        }
    }
}

/// Map a pre-edit offset through `edits` (mirrors
/// `update::apply_edits`' remap so callers can compute post-edit
/// targets from pre-edit geometry).
pub fn map_offset(edits: &[SourceEdit], off: usize) -> usize {
    let mut shift: isize = 0;
    for e in edits {
        if e.range.end <= off {
            shift += e.replacement.len() as isize - (e.range.end - e.range.start) as isize;
        } else if e.range.start < off && off < e.range.end {
            let new_pos = (e.range.start as isize + shift) + e.replacement.len() as isize;
            return new_pos.max(0) as usize;
        } else {
            break;
        }
    }
    ((off as isize) + shift).max(0) as usize
}

/// Navigable cells in Tab order: every row except the delimiter row.
fn nav_cells(geo: &TableGeometry) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (r, row) in geo.rows.iter().enumerate() {
        if row.kind == RowKind::Delimiter {
            continue;
        }
        for c in 0..row.cells.len() {
            out.push((r, c));
        }
    }
    out
}

/// Tab / Shift-Tab: move to the next / previous cell. Lands as a
/// selection of the target cell's content (non-empty) or a caret
/// (empty). Tab past the last cell appends a new empty row and lands
/// in its first cell.
pub fn tab_move(geo: &TableGeometry, pos: usize, forward: bool) -> Option<TableEdit> {
    let (row, cell) = geo.cell_at(pos)?;
    let order = nav_cells(geo);
    let here = order.iter().position(|&(r, c)| (r, c) == (row, cell))?;
    let target = if forward {
        if here + 1 < order.len() {
            Some(order[here + 1])
        } else {
            None
        }
    } else {
        here.checked_sub(1).map(|i| order[i])
    };
    match target {
        Some((r, c)) => {
            let cr = geo.rows[r].cells[c].clone();
            Some(TableEdit {
                edits: Vec::new(),
                cursor: cr.end,
                anchor: if cr.start < cr.end {
                    Some(cr.start)
                } else {
                    None
                },
            })
        }
        None if forward => {
            // Append a fresh empty row after the last row.
            let n = geo.column_count().max(1);
            let last = geo.rows.last()?;
            let row_text = empty_row(n);
            let insert_at = last.line.end;
            let text = format!("\n{row_text}");
            let cursor = insert_at + 1 + EMPTY_ROW_FIRST_CELL_OFFSET;
            Some(TableEdit::caret(
                vec![SourceEdit {
                    range: insert_at..insert_at,
                    replacement: text,
                }],
                cursor,
            ))
        }
        None => None,
    }
}

/// Enter inside a table:
///
/// * on the **last body row with all cells empty** → exit the table:
///   the empty row is removed and a paragraph break opens below;
/// * on the header (or delimiter) row → insert an empty row right
///   after the delimiter row (the first body position);
/// * on any other row → insert an empty row below the current row.
///
/// The caret lands in the new row's first cell (or on the fresh
/// paragraph for the exit case).
pub fn enter_edit(geo: &TableGeometry, source: &str, pos: usize) -> Option<TableEdit> {
    let row_idx = geo.row_at(pos)?;
    let row = &geo.rows[row_idx];

    let row_is_empty = row
        .cells
        .iter()
        .all(|c| source[c.clone()].trim().is_empty());
    let is_last = row_idx + 1 == geo.rows.len();
    if row.kind == RowKind::Body && row_is_empty && is_last {
        // Exit: replace the row's line (and the table's trailing
        // newline, when present) with a paragraph break after the
        // remaining table.
        let del_start = row.line.start.saturating_sub(1); // preceding \n
        let del_end = geo.range.end.min(source.len());
        let edits = vec![SourceEdit {
            range: del_start..del_end,
            replacement: "\n\n".to_string(),
        }];
        let cursor = del_start + 2;
        return Some(TableEdit::caret(edits, cursor));
    }

    // Insert below: header inserts after the delimiter row.
    let after_row_idx = match row.kind {
        RowKind::Header => row_idx + 1,
        _ => row_idx,
    }
    .min(geo.rows.len() - 1);
    let after = &geo.rows[after_row_idx];
    let n = geo.column_count().max(1);
    let text = format!("\n{}", empty_row(n));
    let insert_at = after.line.end;
    let cursor = insert_at + 1 + EMPTY_ROW_FIRST_CELL_OFFSET;
    Some(TableEdit::caret(
        vec![SourceEdit {
            range: insert_at..insert_at,
            replacement: text,
        }],
        cursor,
    ))
}

/// Backspace at a structural point inside a table:
///
/// * in an **empty body row** → delete the row, caret to the previous
///   navigable row's last cell end;
/// * at the **start of cell k > 0** → remove column boundary `k`
///   table-wide (merge cells `k-1` and `k` in every row; the merged
///   delimiter cell keeps column `k-1`'s alignment);
/// * at the **start of the first cell of a body row** → move the
///   caret to the previous navigable row's last cell end (structure
///   is protected, nothing is deleted);
/// * anywhere else → `None` (the generic delete path applies).
pub fn backspace_edit(geo: &TableGeometry, source: &str, pos: usize) -> Option<TableEdit> {
    let (row_idx, cell_idx) = geo.cell_at(pos)?;
    let row = &geo.rows[row_idx];
    let cell = row.cells.get(cell_idx)?;
    if pos != cell.start {
        return None;
    }

    // Empty body row: delete the whole row.
    let row_is_empty = row
        .cells
        .iter()
        .all(|c| source[c.clone()].trim().is_empty());
    if row.kind == RowKind::Body && row_is_empty {
        let del_start = row.line.start.saturating_sub(1);
        let edits = vec![SourceEdit {
            range: del_start..row.line.end,
            replacement: String::new(),
        }];
        let target = prev_nav_cell_end(geo, row_idx).unwrap_or(geo.range.start);
        let cursor = map_offset(&edits, target);
        return Some(TableEdit::caret(edits, cursor));
    }

    if cell_idx > 0 {
        // Remove column boundary `cell_idx` in every row.
        let mut edits: Vec<SourceEdit> = Vec::new();
        for r in &geo.rows {
            let (Some(prev), Some(here)) = (r.cells.get(cell_idx - 1), r.cells.get(cell_idx))
            else {
                continue;
            };
            if r.kind == RowKind::Delimiter {
                // Merged delimiter cell: canonical text for the
                // left column's alignment.
                let align = parse_alignment(&source[prev.clone()]);
                edits.push(SourceEdit {
                    range: prev.start..here.end,
                    replacement: alignment_cell(align).to_string(),
                });
            } else {
                // Delete the chrome between the two cells; contents
                // concatenate (`|` + Backspace round-trips).
                edits.push(SourceEdit {
                    range: prev.end..here.start,
                    replacement: String::new(),
                });
            }
        }
        edits.sort_by_key(|e| e.range.start);
        let cursor = map_offset(&edits, pos);
        return Some(TableEdit::caret(edits, cursor));
    }

    // First cell of a row: protected. Body rows hop to the previous
    // navigable row's last cell; the header falls through to the
    // generic path (merging the table into the preceding block is a
    // legitimate, honest edit).
    if row.kind == RowKind::Header {
        return None;
    }
    let target = prev_nav_cell_end(geo, row_idx)?;
    Some(TableEdit::caret(Vec::new(), target))
}

/// Content end of the last cell of the nearest navigable row above
/// `row_idx`.
fn prev_nav_cell_end(geo: &TableGeometry, row_idx: usize) -> Option<usize> {
    geo.rows[..row_idx]
        .iter()
        .rev()
        .find(|r| r.kind != RowKind::Delimiter && !r.cells.is_empty())
        .map(|r| r.cells.last().unwrap().end)
}

/// Delete-forward at a cell's content end: hop to the next navigable
/// cell's content start (selection only — symmetric to the protected
/// first-cell Backspace). `None` anywhere else.
pub fn delete_forward_hop(geo: &TableGeometry, pos: usize) -> Option<TableEdit> {
    let (row_idx, cell_idx) = geo.cell_at(pos)?;
    let row = &geo.rows[row_idx];
    let cell = row.cells.get(cell_idx)?;
    if pos != cell.end {
        return None;
    }
    // Next cell on this row, else first cell of the next navigable row.
    if let Some(next) = row.cells.get(cell_idx + 1) {
        return Some(TableEdit::caret(Vec::new(), next.start));
    }
    let next_row = geo.rows[row_idx + 1..]
        .iter()
        .find(|r| r.kind != RowKind::Delimiter && !r.cells.is_empty())?;
    Some(TableEdit::caret(
        Vec::new(),
        next_row.cells.first().unwrap().start,
    ))
}

/// Typing `|` inside a cell: insert a column boundary at the caret,
/// table-wide. The caret's cell splits at the caret; the header and
/// delimiter rows gain an empty / `---` cell after the same column so
/// the table keeps parsing within this very event (a header/delimiter
/// count mismatch would dissolve the whole construct before the
/// canonicalizer could repair it). Other body rows are left to the
/// canonicalizer, which pads them on the same keystroke.
pub fn pipe_insert_edit(geo: &TableGeometry, source: &str, pos: usize) -> Option<TableEdit> {
    let (row_idx, cell_idx) = geo.cell_at(pos)?;
    let row = &geo.rows[row_idx];
    let cell = row.cells.get(cell_idx)?;
    // Only intercept inside (or at the edges of) the cell's content.
    if pos < cell.start || pos > cell.end {
        return None;
    }

    let mut edits: Vec<SourceEdit> = Vec::new();
    for (r_idx, r) in geo.rows.iter().enumerate() {
        if r_idx == row_idx {
            continue;
        }
        // Every row that has a cell at the caret's column gains an
        // empty cell right after it, so the new column threads
        // through the whole table at the same index (and Backspace
        // at the fresh cell's start — the exact inverse — round
        // trips). Header and delimiter in particular must stay in
        // lockstep within this very event: a count mismatch between
        // those two dissolves the table at the next parse, before
        // the canonicalizer could repair anything. Shorter body rows
        // (no cell at this column) are left to the canonicalizer's
        // end-padding.
        let Some(after) = r.cells.get(cell_idx) else {
            continue;
        };
        let new_cell = if r.kind == RowKind::Delimiter {
            " | ---"
        } else {
            " |"
        };
        edits.push(SourceEdit {
            range: after.end..after.end,
            replacement: new_cell.to_string(),
        });
    }
    // The caret row's split. On header/body rows a literal ` | ` at
    // the caret splits the cell cleanly; on the **delimiter row** a
    // literal split can produce an invalid cell (an empty left half
    // at the cell's start, a colon-only fragment after a leading
    // `:`), which dissolves the whole table — so the dash cell is
    // replaced wholesale with two canonical cells that distribute
    // its alignment colons (`:-:` → `:-- | --:`, `:--` → `:-- | ---`,
    // `--:` → `--- | --:`).
    if row.kind == RowKind::Delimiter {
        let align = parse_alignment(&source[cell.clone()]);
        let (left, right) = match align {
            TableAlignment::None => (TableAlignment::None, TableAlignment::None),
            TableAlignment::Left => (TableAlignment::Left, TableAlignment::None),
            TableAlignment::Right => (TableAlignment::None, TableAlignment::Right),
            TableAlignment::Center => (TableAlignment::Left, TableAlignment::Right),
        };
        edits.push(SourceEdit {
            range: cell.clone(),
            replacement: format!("{} | {}", alignment_cell(left), alignment_cell(right)),
        });
        edits.sort_by_key(|e| e.range.start);
        // Land the caret at the start of the right-hand cell: the
        // replacement pins an interior offset to its end, so walk
        // back over the right cell's text.
        let cursor = map_offset(&edits, cell.end) - alignment_cell(right).len();
        return Some(TableEdit::caret(edits, cursor));
    }
    edits.push(SourceEdit {
        range: pos..pos,
        replacement: " | ".to_string(),
    });
    edits.sort_by_key(|e| e.range.start);
    // `map_offset` treats an insertion at exactly `pos` as preceding
    // it, so the mapped caret already lands after the inserted
    // ` | ` — at the start of the split-off cell.
    let cursor = map_offset(&edits, pos);
    Some(TableEdit::caret(edits, cursor))
}

/// The scaffold: Enter at the end of a pipe-shaped paragraph line
/// (`| a | b |`) that is not yet a table. Inserts the delimiter row
/// and one empty body row; the caret lands in the first body cell.
pub fn scaffold_edit(source: &str, pos: usize) -> Option<TableEdit> {
    // Cursor must sit at the end of its line.
    let bytes = source.as_bytes();
    if pos < source.len() && bytes[pos] != b'\n' {
        return None;
    }
    let line_start = source[..pos].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line = &source[line_start..pos];
    let trimmed = line.trim();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') || trimmed.len() < 2 {
        return None;
    }
    let cells = scan_line_cells(line);
    if cells.is_empty() {
        return None;
    }
    // Require at least one non-empty cell — scaffolding an all-empty
    // header is never what the user meant.
    if cells.iter().all(|c| line[c.clone()].trim().is_empty()) {
        return None;
    }
    let n = cells.len();
    let delim = canonical_delimiter_row(&[], n);
    let body = empty_row(n);
    let text = format!("\n{delim}\n{body}");
    let cursor = pos + 1 + delim.len() + 1 + EMPTY_ROW_FIRST_CELL_OFFSET;
    Some(TableEdit::caret(
        vec![SourceEdit {
            range: pos..pos,
            replacement: text,
        }],
        cursor,
    ))
}

/// Canonicalization edits for one table: every row whose line does
/// **not** contain the caret is rewritten to canonical form —
/// `| a | b |` single-space padding, rows padded with empty cells up
/// to the column count, and the delimiter row regenerated from its
/// alignments at the header's width. Returns an empty vec when
/// nothing needs rewriting (the idempotent fast path).
pub fn normalize_edits(geo: &TableGeometry, source: &str, cursor: usize) -> Vec<SourceEdit> {
    let mut edits: Vec<SourceEdit> = Vec::new();
    let cursor_row = geo.row_at(cursor);

    // GFM parses the table only when header and delimiter widths
    // agree, so the two share one target. When the header row hosts
    // the caret (and therefore can't be rewritten), the delimiter
    // must match the header's *current* width, not the padded
    // target; body rows always pad up to the overall column count
    // (a wider body row parses fine and gets absorbed once the
    // caret leaves the header).
    let n_all = geo.column_count();
    let header_locked = cursor_row == Some(0);
    let header_target = if header_locked {
        geo.rows.first().map(|r| r.cells.len()).unwrap_or(0)
    } else {
        n_all
    };

    for (idx, row) in geo.rows.iter().enumerate() {
        if cursor_row == Some(idx) {
            continue;
        }
        let target_n = match row.kind {
            RowKind::Header => header_target,
            RowKind::Delimiter => header_target,
            RowKind::Body => n_all,
        }
        .max(1);
        let canonical = match row.kind {
            RowKind::Delimiter => canonical_delimiter_row(&geo.alignments, target_n),
            _ => {
                let mut cells: Vec<&str> =
                    row.cells.iter().map(|c| source[c.clone()].trim()).collect();
                while cells.len() < target_n {
                    cells.push("");
                }
                canonical_row(&cells)
            }
        };
        if source[row.line.clone()] != canonical {
            edits.push(SourceEdit {
                range: row.line.clone(),
                replacement: canonical,
            });
        }
    }
    edits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn geo(src: &str, pos: usize) -> TableGeometry {
        let tree = parse(src);
        table_at(src, &tree, pos).expect("table at pos")
    }

    const T: &str = "| a | b |\n| --- | :-: |\n| c1 | c2 |\n";

    #[test]
    fn scan_cells_basic() {
        let cells = scan_line_cells("| a | b |");
        assert_eq!(cells, vec![2..3, 6..7]);
    }

    #[test]
    fn scan_cells_no_outer_pipes() {
        let cells = scan_line_cells("a | b");
        assert_eq!(cells, vec![0..1, 4..5]);
    }

    #[test]
    fn scan_cells_escaped_pipe_stays_in_cell() {
        let cells = scan_line_cells("| a\\|x | b |");
        assert_eq!(cells, vec![2..6, 9..10]);
    }

    #[test]
    fn scan_cells_empty_cell_gets_padded_point() {
        let cells = scan_line_cells("|  | x |");
        assert_eq!(cells, vec![2..2, 5..6]);
    }

    #[test]
    fn geometry_rows_and_alignments() {
        let g = geo(T, 2);
        assert_eq!(g.rows.len(), 3);
        assert_eq!(g.rows[0].kind, RowKind::Header);
        assert_eq!(g.rows[1].kind, RowKind::Delimiter);
        assert_eq!(g.rows[2].kind, RowKind::Body);
        assert_eq!(
            g.alignments,
            vec![TableAlignment::None, TableAlignment::Center]
        );
        assert_eq!(&T[g.rows[2].cells[1].clone()], "c2");
    }

    #[test]
    fn chrome_interior_detection() {
        let src = T;
        let tree = parse(src);
        // `| a | b |` — offset 1 is between `|` and space: chrome.
        assert!(is_table_chrome_interior(src, &tree, 1));
        // offset 2 = content start of `a`: allowed.
        assert!(!is_table_chrome_interior(src, &tree, 2));
        // offset 3 = content end of `a`: allowed.
        assert!(!is_table_chrome_interior(src, &tree, 3));
        // offset 4 = inside ` | `: chrome.
        assert!(is_table_chrome_interior(src, &tree, 4));
        // outer boundary allowed.
        assert!(!is_table_chrome_interior(src, &tree, 0));
        // Delimiter-row dashes are content (alignment editing).
        let dash_pos = src.find("---").unwrap();
        assert!(!is_table_chrome_interior(src, &tree, dash_pos));
    }

    #[test]
    fn tab_moves_and_appends() {
        let g = geo(T, 2);
        // From header `a` → header `b` (selects content).
        let e = tab_move(&g, 2, true).unwrap();
        assert_eq!((e.anchor, e.cursor), (Some(6), 7));
        // From header `b` forward skips the delimiter row → body c1.
        let e = tab_move(&g, 7, true).unwrap();
        let c1 = T.find("c1").unwrap();
        assert_eq!((e.anchor, e.cursor), (Some(c1), c1 + 2));
        // From last cell → appends a row.
        let c2 = T.find("c2").unwrap();
        let e = tab_move(&g, c2 + 2, true).unwrap();
        assert_eq!(e.edits.len(), 1);
        assert_eq!(e.edits[0].replacement, "\n|  |  |");
        // Shift-Tab from body c1 lands on header `b`.
        let e = tab_move(&g, c1, false).unwrap();
        assert_eq!((e.anchor, e.cursor), (Some(6), 7));
    }

    #[test]
    fn enter_inserts_row_below() {
        let g = geo(T, 2);
        let e = enter_edit(&g, T, T.find("c1").unwrap()).unwrap();
        assert_eq!(e.edits.len(), 1);
        assert_eq!(e.edits[0].replacement, "\n|  |  |");
        // Insert lands after the body row's line.
        let body_line_end = T.rfind(" c2 |").unwrap() + " c2 |".len();
        assert_eq!(e.edits[0].range.start, body_line_end);
    }

    #[test]
    fn enter_on_header_inserts_after_delimiter() {
        let g = geo(T, 2);
        let e = enter_edit(&g, T, 2).unwrap();
        let delim_end = T.find(":-: |").unwrap() + ":-: |".len();
        assert_eq!(e.edits[0].range.start, delim_end);
    }

    #[test]
    fn enter_on_empty_last_row_exits() {
        let src = "| a |\n| --- |\n|  |\n";
        let g = geo(src, 2);
        let point = src.rfind("|  |").unwrap() + 2;
        let e = enter_edit(&g, src, point).unwrap();
        assert_eq!(e.edits[0].replacement, "\n\n");
        // Deleting the row's line plus the preceding newline.
        assert_eq!(e.edits[0].range.start, src.rfind("|  |").unwrap() - 1);
    }

    #[test]
    fn backspace_empty_row_deletes_it() {
        let src = "| a |\n| --- |\n| x |\n|  |\n";
        let g = geo(src, 2);
        let point = src.rfind("|  |").unwrap() + 2;
        let e = backspace_edit(&g, src, point).unwrap();
        assert_eq!(e.edits.len(), 1);
        assert_eq!(e.edits[0].replacement, "");
        // Caret lands at the previous row's `x` end.
        let x_end = src.find('x').unwrap() + 1;
        assert_eq!(e.cursor, x_end);
    }

    #[test]
    fn backspace_at_cell_start_merges_column() {
        let g = geo(T, 2);
        let b_start = 6; // header `b`
        let e = backspace_edit(&g, T, b_start).unwrap();
        // Three rows get an edit (header gap, delimiter merge, body gap).
        assert_eq!(e.edits.len(), 3);
        // Header gap ` | ` between `a` and `b` deleted.
        assert_eq!(e.edits[0].range, 3..6);
        assert_eq!(e.edits[0].replacement, "");
        // Delimiter cells merged to the left column's alignment.
        assert_eq!(e.edits[1].replacement, "---");
    }

    #[test]
    fn backspace_first_cell_of_body_hops_up() {
        let g = geo(T, 2);
        let c1 = T.find("c1").unwrap();
        let e = backspace_edit(&g, T, c1).unwrap();
        assert!(e.edits.is_empty());
        assert_eq!(e.cursor, 7); // header `b` end
    }

    #[test]
    fn pipe_insert_splits_column_through_structure() {
        let g = geo(T, 2);
        // Type `|` at the end of header cell `a` (offset 3).
        let e = pipe_insert_edit(&g, T, 3).unwrap();
        // Header split + delimiter extension + body-row cell at the
        // same column index (so column removal round-trips).
        assert!(e.edits.iter().any(|ed| ed.replacement == " | "));
        assert!(e.edits.iter().any(|ed| ed.replacement == " | ---"));
        let c1_end = T.find("c1").unwrap() + 2;
        assert!(
            e.edits
                .iter()
                .any(|ed| ed.replacement == " |" && ed.range.start == c1_end)
        );
    }

    #[test]
    fn scaffold_from_header_candidate() {
        let src = "| a | b |";
        let e = scaffold_edit(src, src.len()).unwrap();
        assert_eq!(e.edits[0].replacement, "\n| --- | --- |\n|  |  |");
        assert_eq!(e.cursor, src.len() + 1 + "| --- | --- |".len() + 1 + 2);
    }

    #[test]
    fn scaffold_refuses_non_candidates() {
        assert!(scaffold_edit("plain text", 10).is_none());
        assert!(scaffold_edit("| a | b", 7).is_none()); // no trailing pipe
        assert!(scaffold_edit("|  |", 4).is_none()); // all empty
    }

    #[test]
    fn normalize_rewrites_sloppy_rows_but_skips_cursor_row() {
        let src = "| a | b |\n| - | - |\n|c1|   c2   |\n";
        let tree = parse(src);
        let g = table_at(src, &tree, 2).unwrap();
        // Cursor on header row: body + delimiter rewritten.
        let edits = normalize_edits(&g, src, 2);
        assert_eq!(edits.len(), 2);
        assert_eq!(edits[0].replacement, "| --- | --- |");
        assert_eq!(edits[1].replacement, "| c1 | c2 |");
        // Cursor on the body row: body left alone.
        let c1 = src.find("c1").unwrap();
        let edits = normalize_edits(&g, src, c1);
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].replacement, "| --- | --- |");
    }

    #[test]
    fn normalize_pads_short_rows() {
        let src = "| a | b |\n| --- | --- |\n| c |\n";
        let tree = parse(src);
        let g = table_at(src, &tree, 2).unwrap();
        let edits = normalize_edits(&g, src, 2); // cursor on header
        assert_eq!(edits.len(), 1);
        assert_eq!(edits[0].replacement, "| c |  |");
    }

    #[test]
    fn normalize_is_idempotent_on_canonical_tables() {
        let tree = parse(T);
        let g = table_at(T, &tree, 2).unwrap();
        assert!(normalize_edits(&g, T, 2).is_empty());
    }

    #[test]
    fn newline_internal_detection() {
        let tree = parse(T);
        let ranges = table_ranges_in_tree(&tree);
        assert_eq!(ranges.len(), 1);
        let first_nl = T.find('\n').unwrap();
        let last_nl = T.len() - 1;
        assert!(newline_is_table_internal(&ranges, first_nl));
        assert!(!newline_is_table_internal(&ranges, last_nl));
    }

    #[test]
    fn nested_table_not_returned_for_editing() {
        let src = "> | a | b |\n> | - | - |\n> | c | d |\n";
        let tree = parse(src);
        assert!(table_at(src, &tree, 5).is_none());
        // ...but its newlines are still protected.
        assert!(!table_ranges_in_tree(&tree).is_empty());
    }
}
