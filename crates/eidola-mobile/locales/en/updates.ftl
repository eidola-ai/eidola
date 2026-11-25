# Updates (src/ui/updates_view.rs): the desktop's verified update flow, and
# the approval that installs an update on the phone.

updates-title = Updates
updates-checking = Checking for updates…
updates-checking-short = Checking…
updates-check = Check now
updates-up-to-date = Eidola is up to date.
updates-latest = Latest release: v{ $version }
updates-no-latest = No release is marked latest yet.
updates-unbundled = Eidola v{ $version } is out, but it carries no app for phones.
updates-unbundled-note = This app stays on its version until a release carries one.
updates-available = Eidola v{ $version } is available.
updates-verified = Verified against this app’s embedded trust root.
# $names lists who attested the release; $logs their transparency-log entries.
updates-attested-by = Attested by { $names } · built and signed by the release workflow · Rekor { $logs }
updates-attested-by-none = Attested by no one this app trusts by name.
updates-released = Released
# The digest the shell's own sheet will show once the update runs.
updates-next-digest = Its digest, as the shell will show it
updates-install-note = Installing downloads the update, checks every file against the signed release, and restarts the app into it. If it doesn’t start, Eidola goes back to this version on its own.
# The approval. Nothing is downloaded or installed until it is tapped.
updates-install = Install update
updates-installing = Downloading and checking…
updates-restarting = Restarting into v{ $version }…
# $reason is the error's own text (English).
updates-install-failed = The update to v{ $version } was not installed: { $reason }. Nothing changed.
updates-claims-changed = Release v{ $version } verified — but its claims changed.
updates-claims-changed-note = The release is authentically signed, but what it attests differs from what this app expects. It is not treated as an update unless you accept the change.
updates-claims-matching = { $matching } of { $total } expected claims match.
updates-claim-expected = Expected: { $value }
updates-claim-attested = Attested: { $value }
updates-claim-absent = absent
updates-accept-claims = Treat as update
updates-claims-accepted = Its claims differ from this app’s expectations; you chose to treat it as an update.
# $reason is exactly what failed (English).
updates-unverifiable = Security warning: the release marked latest could not be verified. { $reason }
updates-unverifiable-note = This may be a fake release or a compromised update channel. Eidola won’t offer it.
updates-check-failed = Couldn’t check for updates (offline?).
updates-section-running = Running
updates-app-version = This app
updates-serving = Running bundle, as the shell reports it
# The bundle built into the app.
updates-embedded = The one built into the app
updates-serving-digest = Its digest, as the shell reports it
updates-on-trial = On trial: it confirms itself once it has started.
updates-confirmed = Confirmed: it started and confirmed itself.
updates-awaiting-reload = { $version } is installed and starts at the next launch.
# The provisional activation failed: the update never confirmed, and the shell
# went back. $failed is that update, $version the version now running.
updates-fell-back = Your last update ({ $failed }) didn’t start; you’re back on version { $version }.
updates-report-failed = Couldn’t read the shell’s bundle state: { $reason }
updates-revert = Go back to the built-in version
updates-revert-note = The app can always go back to the version built into it, and restarts into it.
updates-reverting = Going back…
updates-revert-failed = Couldn’t go back: { $reason }
