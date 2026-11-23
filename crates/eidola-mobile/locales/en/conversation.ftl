# A conversation on the phone (src/ui/conversation_view.rs).

# The back button's accessible name: it returns to the Library.
conversation-back = Conversations
# The title of a conversation that has none yet.
conversation-untitled = New conversation
# The byline of the reader's own posts.
conversation-you = You
# The verb beneath a post that starts a reply to it.
conversation-reply = Reply
# That verb's accessible name, naming the post's author.
conversation-reply-label = Reply to { $author }
# Above the composer while a reply is being written.
conversation-replying-to = Replying to { $author }
# The control that drops the reply and writes to the end of the conversation.
conversation-cancel-reply = Cancel reply
# The reasoning disclosure while the model is still thinking.
conversation-thinking = Thinking…
# The reasoning disclosure once the answer has settled: what a tap does.
conversation-show-thinking = Show thinking
conversation-hide-thinking = Hide thinking
# A streaming answer with nothing to show yet.
conversation-waiting = Waiting for a reply…
# Where a post has several replies: which one is shown.
conversation-branch = { $index } of { $count }
conversation-branch-previous = Previous reply
conversation-branch-next = Next reply
# An empty conversation, above the composer.
conversation-empty = Write below to begin.
# The first load of a conversation failed.
conversation-load-failed = Couldn’t load this conversation.
# A refresh failed over posts that are still shown.
conversation-refresh-failed = Couldn’t refresh — retry
conversation-retry = Retry
conversation-dismiss = Dismiss
# A conversation whose row no longer exists (its creation was refused).
conversation-gone = This conversation couldn’t be created.
# The answer is still being written; a second message waits for it.
conversation-busy = A reply is still being written.
# The cascade guard paused replies between agents.
conversation-cascade-paused = The agents paused after { $depth } replies in a row.

# The composer (src/ui/composer.rs).
composer-placeholder = Write a message…
composer-send = Send
# The composer field's accessible name.
composer-label = Message

# The recovery notice when a request failed (typed errors, worded here).
conversation-error-no-account = There’s no account on this device yet, so nothing was sent.
conversation-error-insufficient-balance = There aren’t enough credits to send this. Nothing was charged.
# Shown when a request failed for a reason this app has no words of its own
# for; $reason is the error's own text.
conversation-error-other = Something went wrong: { $reason }
conversation-error-archived = This conversation is archived, so it can’t take new replies.
conversation-error-not-joined = Your agents opened this conversation between themselves. You can read all of it; to change or re-ask something here, take part first by posting.
conversation-error-model-unavailable = { $model } is no longer offered by its backend, and nothing will answer in its place. Nothing was sent and nothing was charged — pick a different model, then try again.
conversation-error-hold-exceeds-denomination = This reply could cost more than a single request is allowed to hold, so the request was not sent and nothing was charged for it. Adding credit won’t change that — try a shorter conversation or a different model.
conversation-error-issuer-key-off-calendar = Eidola’s server offered a payment key that doesn’t follow the public key schedule, so this app didn’t use it. Nothing was sent and nothing was charged, and the credits you already hold still work.
conversation-error-issuer-key-second-for-epoch = Eidola’s server offered a different payment key this week from the one this app already has. An honest server uses one key per week, so this app won’t take new credits under it until next week’s key. Nothing was sent and nothing was charged, and the credits you already hold still work.
conversation-error-issuer-key-reused = Eidola’s server offered the same payment key for two different weeks, which an honest server never does, so this app didn’t use it. Nothing was sent and nothing was charged, and the credits you already hold still work.
conversation-error-uncached-retry-unfunded = Eidola’s engine no longer had this conversation’s earlier context ready, and the credits this app holds can’t cover sending it again in full. Nothing more was converted from your account. Try again to send it as a new request.
