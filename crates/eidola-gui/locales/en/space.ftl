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
# The same ending, resting on an answer that stopped at its length limit: the
# room ran out of things to say, but not on a finished thought.
space-footnote-delegation-concluded-truncated = ran to a stop, on an answer cut off at its length limit
space-footnote-delegation-paused = reached its reply limit ({ $depth } of { $limit })
# The same pause, resting on an answer that stopped at its length limit —
# resuming would continue from a mid-thought answer, so the row says both.
space-footnote-delegation-paused-truncated = reached its reply limit ({ $depth } of { $limit }), on an answer cut off at its length limit
space-footnote-delegation-budget = used all { $limit } of its turns
# The same spent budget, resting on an answer cut off at its length limit.
space-footnote-delegation-budget-truncated = used all { $limit } of its turns, on an answer cut off at its length limit
# Why a delegated conversation stopped short, by kind. The reason is a bounded
# category — never an upstream's own words — so this select covers all of them.
space-footnote-delegation-failed = stopped: { $reason ->
        [upstream] the model could not be reached
        [funding] the turn could not be paid for
        [configuration] something in its setup
       *[other] a turn could not be finished
    }

# Shown in place of an answer while it is being regenerated, from the frame the
# press was accepted in. A reasoning model can spend minutes before the first
# word of the new answer arrives, so this is often the only thing on screen for
# a long while — it says the request is out, not that anything is nearly done.
space-regenerating = Regenerating…
# Shown in the conversation's recovery notice when a response used its whole
# length allowance on its own reasoning and never began an answer. Nothing was
# recorded and nothing was replaced, so the answer already there is untouched.
space-error-response-truncated = The model used its whole length allowance thinking and never started an answer. No answer was added or replaced — try again, or ask a shorter question.
# A quiet marker beneath a response a regeneration was asked for while one was
# already running against it somewhere this window cannot see. Not a failure:
# the first one is still going, and its result will arrive here.
space-regenerating-elsewhere = This response is already being regenerated.
# A quiet marker beneath an answer that stopped because it reached its length
# allowance rather than because the model was finished.
space-answer-cut-off = This answer reached its length limit and stops mid-thought.
# The source-highlight picker (src/space_view/references.rs): the popover that
# opens when a clicked passage was quoted by more than one post, listing each
# so the reader chooses the target rather than the app guessing.
space-highlight-picker-group = Posts quoting this passage
space-highlight-picker-heading = Quoted by
# A referrer this window holds: the byline the page already shows for that post,
# plus the opening of what it says.
space-highlight-picker-here = { $byline }: { $snippet }
# A referrer from a conversation this window never loaded. The author alone does
# not identify a post — one participant can quote the same passage from two
# conversations — so the row names the conversation the click would open. The
# untitled variants are for a conversation nobody has named, where its existence
# elsewhere is the only true thing left to say.
space-highlight-picker-elsewhere = { $byline }, in { $space }
space-highlight-picker-elsewhere-untitled = { $byline }, in another space
space-highlight-picker-unnamed = A post in { $space }
space-highlight-picker-unnamed-untitled = A post in another space
# Two rows that composed to the same sentence — one author quoting the same
# passage from two conversations that share a title, or from two nobody named.
# A counter, not context: it claims only that these are different rows and this
# is the first of them, in the order the backlinks were written.
# TRANSLATORS: $n is load-bearing, not decoration. It is what makes two
# otherwise identical rows tell apart, and the picker searches upward for a
# number no other row has taken — so a wording that drops it leaves rows the
# reader cannot choose between.
space-highlight-picker-nth = { $label } ({ $n })
# The same number, painted on its own beside the row rather than inside the
# sentence — the row's text truncates at 280px and a discriminator that
# truncates discriminates nothing. Keep it in step with
# `space-highlight-picker-nth`, whose tail it is.
space-highlight-picker-ordinal = ({ $n })
