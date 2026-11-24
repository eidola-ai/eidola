# The app's frame (src/ui/app.rs).

# Shown while the app opens its profile.
app-starting = Opening…
# The profile could not be opened; $reason is the error's own text.
app-start-failed = Eidola couldn’t open its data on this device.
app-start-failed-detail = { $reason }

# Shared by every screen (src/ui/widgets.rs).

# The back chevron's accessible name.
nav-back = Back
# The copy verb beside a value, and its accessible name — the same verb with
# the subject its row supplies ($label), so a screen reader hears which.
nav-copy = Copy
nav-copy-label = Copy { $label }
