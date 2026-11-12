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
# Shown in the conversation's recovery notice when the reader tries to act on
# work already in a conversation their agents opened between themselves —
# editing, regenerating or retrying. Reading one is unconditional oversight;
# acting on somebody else's work in one is membership, because the roster the
# models are shown has to stay truthful and a turn driven there spends the
# reader's credits. Posting is deliberately not in that set: it is what joins
# the reader, so this names it as the way through rather than as a step the
# app cannot take.
space-error-not-joined = Your agents opened this conversation between themselves. You can read all of it; to change or re-ask something here, take part first by posting.

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
# Shown in the conversation's recovery notice when the model this turn would
# have used is no longer in its backend's catalog — a row retired there, while
# whoever answers here still names it. Two things carry the sentence, and both
# are load-bearing: **the model is named**, because "a model" is not something
# a reader can go and change, and **nothing answered in its place**, because
# quietly using a different model is the one outcome worse than the failure and
# the reader has to know it did not happen. Retry stays offered — picking a
# different model is what makes the next press work.
# TRANSLATORS: $model is the selection exactly as it was saved, an id rather
# than a display name. Leave it verbatim; it is what the reader looks for.
space-error-model-unavailable = { $model } is no longer offered by its backend, and nothing will answer in its place. Nothing was sent and nothing was charged — pick a different model, then try again.
# Shown in the conversation's recovery notice when the model this turn would
# have used runs on this computer but has not been downloaded — what a new
# profile meets first, since its default agent uses an on-device model. Says
# that nothing was sent, and the two ways on: download that model, or choose
# another one. Retry stays offered, for after the download.
# TRANSLATORS: $model is the model's display name (e.g. "Gemma 4 E4B") or, for
# a model with none, its id. Leave it verbatim. "Settings ▸ Backends ▸ Local"
# names the app's own menus; use the translated names those menus show.
space-error-local-model-not-downloaded = { $model } runs on this computer but isn't downloaded yet, so nothing was sent. Download it in Settings ▸ Backends ▸ Local, or pick a different model, then try again.
# A quiet marker beneath a response a regeneration was asked for while one was
# already running against it somewhere this window cannot see. Not a failure:
# the first one is still going, and its result will arrive here.
space-regenerating-elsewhere = This response is already being regenerated.
# A quiet marker beneath an answer that stopped because it reached its length
# allowance rather than because the model was finished — reasoning that never
# reached an answer included. Where the answer can be continued the marker
# carries Continue beside it, so it states the fact and the way on is the verb.
space-answer-cut-off = This answer stopped at its length limit.
# The verb beside that marker: asks the same model to pick up exactly where the
# answer stopped, mid-sentence or mid-thought, as a new version of the same
# answer. Short, because it sits on the marker's line.
space-continue-answer = Continue
# What a screen reader announces for that verb — the action in full.
space-continue-answer-label = Continue this answer from where it stopped
# Shown above an answer while it is being continued, from the frame the press
# was accepted in, until the continued text starts arriving.
space-continuing = Continuing…
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

# Shown in the conversation's recovery notice when this app refused the
# server's current payment key (the issuer key credits are converted under)
# because its windows do not follow the public key schedule every client can
# compute for itself. Nothing was charged; credits already held still work.
# "Payment key" is deliberately plain: the reader needs to know the app
# protected them, not the protocol's vocabulary.
space-error-issuer-key-off-calendar = Eidola’s server offered a payment key that doesn’t follow the public key schedule, so this app didn’t use it. Nothing was sent and nothing was charged, and the credits you already hold still work.
# The same notice when the server offered a different payment key for this
# week from the one this app already saw. An honest server has one per week,
# and a second one is what a server singling a reader out would show, so the
# app refuses it until the next week's key. Retrying now will not help, which
# is what the last sentence is for.
space-error-issuer-key-second-for-epoch = Eidola’s server offered a different payment key this week from the one this app already has. An honest server uses one key per week, so this app won’t take new credits under it until next week’s key. Nothing was sent and nothing was charged, and the credits you already hold still work.
# The same notice when the server offered one payment key for two different
# weeks. An honest server has a distinct key every week, and a key reused
# across weeks would let the server swap another key in unnoticed later, so
# the app refuses it.
space-error-issuer-key-reused = Eidola’s server offered the same payment key for two different weeks, which an honest server never does, so this app didn’t use it. Nothing was sent and nothing was charged, and the credits you already hold still work.
