# Find in the conversation — the ⌘F bar over the visible branch
# (src/space_view/find.rs).
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file.

# The query field's placeholder. It carries the surface's name, because the
# space window's title bar paints nothing and the bar draws no heading of its
# own — the field is where the reader finds out what this is.
find-placeholder = Find in this conversation
# The query field's accessible name. Short, because a screen reader reads it on
# every focus and the placeholder above already says where the search runs.
find-field-label = Find
# The two step arrows. Each says what its click does; a screen reader hears
# them with no arrow glyph to go by.
find-previous = Previous match
find-next = Next match
# Closes the bar and clears the highlights.
find-close = Close find
# The index readout beside the arrows. It is an index rather than a running
# total because stepping wraps: 1 of 3 → 2 of 3 → 3 of 3 → 1 of 3. Digits are
# unlocalized (nothing registers a Fluent NUMBER function — see the
# localization doctrine).
find-count = { $index } of { $total }
# The same readout when the query matches nothing on the visible branch.
# Deliberately not "0 of 0", which reads as a position that exists.
find-no-results = No results
# The **cross-branch** total, beside the index. A different number: the index
# counts the branch the reader is looking at, this counts the whole
# conversation, every branch included — so it is never smaller.
find-total = { $total } total
# The same readout while the cross-branch pass is still walking the space. It
# has to exist and it must not be a number: a partial sum reads on screen
# exactly like a settled one, and the previous query's total is the same lie
# with a longer fuse.
find-total-counting = Counting…
# A minimap sibling column's accessible name gains this: the matches reachable
# only by taking that branch, its whole subtree included. The numeral is painted
# in the cell where it fits and a tint stands in where it does not — this is
# what a screen reader hears either way, and it is always the exact count.
find-branch-count = { $count } more in this branch
# The same readout when the visible branch holds nothing but the conversation
# does. "No results" beside "3 total" is a contradiction on its face — and the
# two are separate nodes, so a screen reader meets them one after the other with
# nothing placing them in one breath. The unqualified line above stays for a
# query nothing anywhere matches.
find-no-results-in-branch = None on this branch
# --- The Find-all overlay (src/space_view/find_overlay.rs) -------------------
# The disclosure beside the cross-branch total. Its name says what the click
# does, in both directions, because the chevron alone says nothing to a screen
# reader — and the total beside it already speaks the number, so this does not.
find-show-all = Show every result
find-hide-all = Hide every result
# The overlay's landmark name. It is the whole of what the window shows while
# it stands, so it is named for what it holds rather than for the bar it came
# from.
find-overlay-label = Every result
# The topological map down the overlay's left: the whole conversation's shape,
# one circle per post, with no regard to how long any of them is. Deliberately
# a different surface from the minimap, which is the scroll handle.
find-map-label = Conversation map
# A map node, labelled by who wrote the post and whether it holds matches —
# never by how many, because the map carries no counts.
find-map-node = { $byline }
find-map-node-matches = { $byline } — has matches
# The results column beside the map.
find-results-label = Results
# A result whose text is a draft the reader has written but not posted. It has
# no byline row of its own, so the overlay names it for what it is.
find-result-draft = Draft
# The overlay with nothing to show. Reachable only when the query stops
# matching while the surface is open, since the disclosure that opens it
# appears with the total.
find-overlay-empty = Nothing matches anywhere in this conversation
