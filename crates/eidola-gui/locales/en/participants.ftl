# The shared participant field helpers (src/participants.rs) — the model and
# router pickers every surface that edits participants renders through.
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.

# The model picker's accessible name — what the control *is*, as distinct from
# which model it holds (that rides the picker's value).
participants-model-label = Model
# The picker's resting text with nothing selected. An invitation, not a state:
# an unset model is an ordinary starting point for a new participant.
participants-model-unset = Choose a model…
# The dropdown list's accessible name.
participants-model-list = Models
# The dropdown with no selectable model behind it — every backend disabled, or
# nothing installed yet. A fact about this machine, not a failure.
participants-model-list-empty = No models available.

# The router picker's accessible name. A router is the model a space asks which
# participant should answer next, so it is named apart from the chat model.
participants-router-label = Router model
participants-router-list = Router models
# The router picker's first option and its resting label. **Off is the default
# and an ordinary choice**, never a degraded one — a space with no router simply
# notifies by its participants' own policies. Keep the register neutral: a word
# that reads as "disabled" or "broken" would misdescribe the common case.
participants-router-off = Off
