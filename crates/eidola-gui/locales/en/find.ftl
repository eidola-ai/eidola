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
