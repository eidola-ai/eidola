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

# Notices about an Eidola model choice, under the participant's name. A tag is
# a name for a set of model weights; a release can move a tag onto newer
# weights. Plain statements of fact, never alarms: the conversation still works.
#
# A followed tag now names newer weights than this participant's last answer
# came from.
participants-model-updated = Model updated: { $tag } now names newer weights.
# A participant pinned to specific weights whose tag has moved on.
participants-model-newer = Newer weights are available as { $tag }. This agent stays on the ones it is pinned to.
# The weights are deprecated and will stop being served after the date.
participants-model-retiring = These weights retire after { $date }.
# The announced retirement date has passed; they may stop answering at any time.
participants-model-retiring-past = These weights were due to retire on { $date }. Choose another model.
# The weights are retired: requests naming them are refused.
participants-model-retired = These weights are retired. Choose another model.
# The model's identity in the editor: its weights hash (shortened) and where
# the weights come from (`owner/name@revision`).
participants-model-weights = Weights { $hash } from { $source }
# Where to learn how to check the weights independently.
participants-model-verify = Check these weights yourself: { $url }
# A model choice pinned to exact weights, in the picker and on the row: the
# tag the weights were chosen under, and the start of their hash.
participants-model-pinned = { $model }, pinned to { $hash }
# The one-click answer to "newer weights are available": re-pin this agent, in
# this conversation, to the weights its tag names now.
participants-model-switch-newer = Switch to the newer weights
# Dismiss a "model updated" notice. It comes back if the tag moves again.
participants-model-dismiss-update = Dismiss
