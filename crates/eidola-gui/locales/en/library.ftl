# The Library window (src/library.rs).
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.

# The badge on a Library row for a conversation one of the reader's agents
# opened from another one, naming the conversation it was delegated from. Such
# a row is a conversation the reader never started, so without this it stands
# in the listing with nothing saying where it came from.
library-row-parent = From { $parent }
# The accessible name of that badge once it can be activated (hovering the row,
# or reaching it with the keyboard cursor, makes it a link to the parent). Names
# what the click does and what it opens.
library-row-parent-open = Open { $parent }, the conversation this was delegated from
# Stands in for the parent's name where that conversation has never been named.
# A noun phrase, because it is composed into the two messages above.
library-row-parent-untitled = an untitled conversation
