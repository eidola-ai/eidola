# The failure surfaces: Settings ▸ Backends ▸ Local's failed-download row
# (src/backends_settings.rs), the space window's transcript-failure surfaces
# (src/space_view/mod.rs), and the startup-failure alert (src/startup.rs).
#
# English is the source locale: every message must exist here. A message missing
# from another locale falls back to English at runtime.

# A managed-store row for a download that failed and left nothing on disk. Its
# status is the same `Available` a real file carries, so the line has to say
# what the row actually is — "downloaded" would be the plainest lie the pane
# could tell. The reason is the error line beside it, not this.
local-model-not-downloaded = not downloaded

# Re-runs the download the row remembers.
local-model-retry = Retry
# The accessible name of that verb: the visible label plus what it acts on.
local-model-retry-label = Retry the download of { $model }
# The transfer is being started. Not a verb — the slot says what is happening
# while its own control is gone, so a second press cannot race the first.
local-model-retry-starting = starting…

# Forgets the standing failure, which is the whole of what such a row is.
local-model-dismiss = Dismiss
local-model-dismiss-label = Dismiss the failed download of { $model }

# The space window could not read its conversation at all — no posts, and so no
# composer either. The panel that stands in the empty reading column.
space-transcript-failed = Couldn't open this conversation.
space-transcript-retry = Retry

# A *refresh* failed over posts already on screen: they stay, and this quiet
# line says the last read is no longer the last word. Same wording as the
# Library's strip — one idiom.
space-transcript-stale-retry = Couldn't refresh — retry
space-transcript-stale-retry-label = Retry loading this conversation

# The startup-failure alert, raised before there is a window. The *body* is the
# typed error's own text and is deliberately not here: app-core stays
# locale-free, so what it reports arrives already worded.
startup-title-already-open = Eidola is already open
startup-title-failed = Eidola can’t start
# The alert's only button. Dismissing it ends the process, so it says so.
startup-quit = Quit
