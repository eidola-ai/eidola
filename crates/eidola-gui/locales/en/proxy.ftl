# Settings ▸ Proxy — the local inference proxy (src/proxy_settings.rs).
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.
#
# The pane's *nav* label is deliberately not here: it is an English literal
# beside the other five, and it is also what the nav row's probe name derives
# from — and a probe name is a stable selector that must never localize.

# What the pane is for, above everything else on it. The Record claim is
# about completions, deliberately: a model listing writes no row (nor does the
# app's own model picker), so the backends note below says so where the reader
# chooses which servers a tool may list. Attestation and wallet
# payment belong to the hosted Eidola route alone; an on-device model and an
# external server get neither, so the sentence says which is which rather
# than lending every route the hosted one's properties.
proxy-lead = Let other tools on this computer use the models you reach through Eidola. They speak the OpenAI API, and every completion they ask for is written into the Record. What else a completion gets depends on where it goes: the hosted Eidola service is attested and paid for from your wallet; an on-device model runs on this computer and costs nothing; a server you added yourself is reached directly, with no attestation, on that server's own terms.

# The switch that starts and stops the listener.
proxy-serve = Serve requests
proxy-serve-name = Serve local requests through Eidola

# Where a tool should point. The address the socket is actually bound to,
# never the stored setting — a surface that tells a person where to point
# their tool must not name somewhere nothing is listening.
proxy-listening = Listening on { $address }
proxy-stopped = Not listening
# The write succeeded and the socket did not, which is a different failure
# from a refused setting and gets its own line.
proxy-listen-failed = Couldn't listen on that address — { $reason }

# The address is not loopback, and there is no TLS yet.
proxy-exposed-warning = This address is reachable from your network, and the proxy has no encryption yet. Anything sent through it — your prompts and the answers — travels in the clear.

# The binding row.
proxy-address = Address
proxy-address-name = The IP address the proxy listens on
proxy-port-name = The port the proxy listens on
proxy-binding-change = Change…
proxy-binding-save = Save
proxy-binding-cancel = Cancel

# Which backends may be reached.
proxy-backends = Backends
proxy-backends-note = Only what you tick here can be reached. A tool naming anything else is told the model does not exist. Asking for the model list is not written into the Record, and for a server you added it asks that server for its catalog, with that server's key.
proxy-backend-name = Offer { $backend } through the proxy
proxy-backends-empty = No backends are configured yet.

# What an on-device backend offers.
proxy-exposure = On-device models
proxy-exposure-loaded = Only loaded
proxy-exposure-downloaded = All downloaded
proxy-exposure-note = "All downloaded" starts an engine on the first request that names a model, which takes a while and claims memory. "Only loaded" offers what is already running.

# The keys downstream tools authenticate with.
proxy-keys = API keys
proxy-keys-note = A tool sends its key as a bearer token. Eidola keeps only a hash of it, so a key is shown once and cannot be shown again.
proxy-keys-empty = No keys yet — nothing can reach the proxy until you make one.
proxy-key-label-placeholder = What will use this key?
proxy-key-create = Generate a key
# A key's value exists for exactly one render, so a second generation may not
# start while one is pending or while a minted key is still unread — the verb is
# replaced by whichever of those is true.
proxy-key-creating = Generating…
proxy-key-show-first = Copy the key above and press Done first.
proxy-key-revoked = revoked
proxy-key-unused = never used
proxy-key-used = used
proxy-key-revoke = Revoke
proxy-key-revoke-name = Revoke the key named { $label }

# The one moment the key exists in full.
proxy-key-minted = Copy this now. Eidola stored only its hash and cannot show it again.
proxy-key-copy = Copy
proxy-key-done = Done

# The pane's own failure surface: a read that never answered leaves nothing
# here to act on, so the way back is a retry rather than a plausible-looking
# empty pane.
proxy-failed = Couldn't read the proxy's settings.
proxy-keys-failed = Couldn't list the API keys.
proxy-backends-failed = Couldn't read the backend registry.
proxy-retry = Retry
# A read still in flight is not an empty configuration, and a refresh that
# failed over values still on screen is not a fresh read of them.
proxy-loading = Loading…
proxy-stale = Couldn't refresh — showing the last answer.

# Why a proxy setting or key was refused — one sentence per typed refusal
# (`ProxyRefusal`). Any other failure shows the error's own text, which is
# English: app-core is locale-free, and those variants carry nothing more for a
# sentence of ours to say.
proxy-error-not-an-address = { $value } is not an IP address. The proxy listens on an address, not a name.
proxy-error-no-port = The proxy needs a port. Port 0 would take whatever happened to be free, which is not an address a tool can be told about.
proxy-error-key-needs-name = A key needs a name, so you can tell later which tool holds it.
proxy-error-cannot-listen = Couldn't listen on { $address } — { $reason }
proxy-error-stopped-accepting = The proxy stopped accepting connections — { $reason }

# A refusal stands under the control it was about, so its visible text needs no
# subject. Its accessible name carries one, because a screen reader meets the
# alert without the row above it. `$subject` is the control's own name (a row
# label, a backend's or a key's name); `$message` is the refusal sentence.
proxy-error-about = { $subject }: { $message }
proxy-error-dismiss-name = Dismiss the message about { $subject }
