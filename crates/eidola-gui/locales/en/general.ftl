# Settings ▸ General — appearance, text size, startup and the command line
# (src/general.rs).
#
# English is the source locale: every message must exist here, and the build
# script generates one typed accessor per message from this file. A message
# missing from another locale falls back to English at runtime; a message that
# exists only in another locale is a build error.
#
# The pane's *nav* label is deliberately not here: it is an English literal
# beside the other six, and it is also what the nav row's probe name derives
# from — and a probe name is a stable selector that must never localize.

## Appearance — the circadian theme's axes.

general-appearance = Appearance

general-day-night = Day & night
general-day-night-auto = Auto
general-day-night-system = System
general-day-night-day = Day
general-day-night-night = Night
# A chip's accessible name: the group label beside it is not announced, so the
# name carries it. $choice is one of the four chip labels above.
general-day-night-choice = Day & night: { $choice }
general-day-night-note = System follows macOS. Auto follows the sun — day between sunrise and sunset.

general-time-of-day = Time of day
general-on = On
general-off = Off
# $choice is "On" or "Off" above.
general-time-of-day-choice = Time of day: { $choice }
general-time-of-day-note = Follows the sun's light — cool around sunrise, neutral midday and midnight, warm around sunset. Sunrise and sunset are approximated from your time zone.

general-light = Light
general-light-cool = Cool
general-light-neutral = Neutral
general-light-warm = Warm
# $choice is one of the three labels above.
general-light-choice = Light: { $choice }
general-light-note = A fixed light character for the palette.

## Text size.

general-text-size = Text size
# $percent is the current scale, already formatted ("100%").
general-text-size-smaller-name = Smaller text, currently { $percent }
general-text-size-larger-name = Larger text, currently { $percent }
general-text-size-reset = Reset
general-text-size-reset-name = Reset text size to 100%
# TRANSLATORS: the menu bar is still English as a whole, so "View", "Actual
# Size", "Zoom In" and "Zoom Out" stay as written until the menus localize
# with it — otherwise this sentence points at labels the reader cannot find.
general-text-size-note = Scales all text. Also on the View menu — Actual Size, Zoom In, Zoom Out.

## Startup — the opt-in login item (src/login_item.rs).

general-startup = Startup
general-open-at-login = Open at login
general-login-item-switch = Start Eidola when you log in
general-login-item-name = Open Eidola at login
general-login-item-off = Eidola stays out of the way until you open it.
# Not "ready in the menu bar": the login item launches the ordinary windowed
# app. Say what happens.
general-login-item-on = Eidola opens with your session.
general-login-item-needs-approval = Turned on, but macOS is waiting for you — allow Eidola in System Settings → General → Login Items.
general-login-item-unsupported = Unavailable — this needs macOS 13 or later, and an installed, signed app.
general-login-item-unsupported-linux = Unavailable here — on Linux, `eidola service install` enables the user service.
general-login-item-too-old = Opening at login needs macOS 13 or later — this Mac's system software is older.
general-login-item-not-here = Opening at login is not available on this platform.

## Command line — the `eidola` link on PATH (src/cli_link.rs).
##
## $path is the link's full path (/usr/local/bin/eidola), $dir the directory
## it goes in, $target where an existing link points. All are file paths and
## are shown as they are.

general-command-line = Command line
general-cli-field = Terminal command
general-cli-not-installed = Not installed. Installing puts an `eidola` command in { $dir }, linked to this copy of Eidola, so you can run it from Terminal.
general-cli-needs-admin = macOS will ask for an administrator's password.
general-cli-installed = Installed — `eidola` in Terminal runs this copy of Eidola, through { $path }.
general-cli-dangling = { $path } points at a copy of Eidola that is no longer there ({ $target }) — the app was probably moved. Update the link to use this copy.
general-cli-other-copy = { $path } runs another copy of Eidola, at { $target }. Update the link to use this copy.
general-cli-occupied-file = { $path } is already a file Eidola didn't put there, so Eidola won't replace it. If you no longer want it, move or delete it yourself, then install.
general-cli-occupied-directory = { $path } is a folder, so Eidola won't put a command there. If you no longer want it, move or delete it yourself, then install.
general-cli-occupied-link = { $path } is a link to { $target }, which isn't Eidola, so Eidola won't replace it. If you no longer want it, move or delete it yourself, then install.
general-cli-occupied-other = { $path } is already something Eidola didn't put there, so Eidola won't replace it.
general-cli-unreadable = Couldn't check { $path } — { $reason }
general-cli-not-bundled = Available only when Eidola runs as an installed app.
general-cli-translocated = macOS is running this copy of Eidola from a temporary location, and a link to it would stop working. Move Eidola into your Applications folder, open it from there, then install.
general-cli-missing = This copy of Eidola has no command-line tool inside it.
# The verb. The "…" form is shown when a password prompt follows the press.
general-cli-install = Install
general-cli-install-admin = Install…
general-cli-install-name = Install the eidola command in { $dir }
general-cli-update = Update link
general-cli-update-admin = Update link…
general-cli-update-name = Point { $path } at this copy of Eidola
# Shown in place of the verb while it runs.
general-cli-installing = Installing…
general-cli-waiting = Waiting for an administrator's password…
# After the password prompt was dismissed.
general-cli-declined = Nothing was changed. To install it yourself, run this in Terminal:
general-cli-failed = Couldn't install the command — { $reason }
general-cli-manual = To install it yourself, run this in Terminal:
general-cli-copy = Copy
general-cli-copy-name = Copy the install command
