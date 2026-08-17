# The conversation window (src/space_view/mod.rs).
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.

# Shown in the conversation's recovery notice when a send is refused because
# the conversation has been archived — by the reader, or by retiring an agent
# that owned it. Archiving does not hide or delete anything: the transcript is
# still there to read, and what has ended is new work. Deliberately says
# "replies" rather than the internal word "turns".
space-error-archived = This conversation is archived, so it can’t take new replies.
# Shown in the conversation's recovery notice when the reader tries to act in
# a conversation their agents opened between themselves — posting, editing,
# regenerating or retrying. Reading one is unconditional oversight; acting in
# one is membership, because the roster the models are shown has to stay
# truthful and a turn driven there spends the reader's credits. Says what
# joining would do, since joining is what would allow it.
space-error-not-joined = Your agents opened this conversation between themselves. You can read all of it; taking part means joining it, which this version can’t do yet.

# The quoted-reference footnote rail (src/space_view/references.rs). When a
# reference is a delegated conversation's report, the edge records how that
# conversation stopped — a value, not a sentence, so it is said here in the
# reader's language rather than stored in one. All four sit in the rail's quiet
# register: they finish the clause "this conversation …".
space-footnote-delegation-concluded = ran to a stop
space-footnote-delegation-paused = reached its reply limit ({ $depth } of { $limit })
space-footnote-delegation-budget = used all { $limit } of its turns
# Why a delegated conversation stopped short, by kind. The reason is a bounded
# category — never an upstream's own words — so this select covers all of them.
space-footnote-delegation-failed = stopped: { $reason ->
        [upstream] the model could not be reached
        [funding] the turn could not be paid for
        [configuration] something in its setup
       *[other] a turn could not be finished
    }
