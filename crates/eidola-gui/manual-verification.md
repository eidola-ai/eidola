# Manual verification — macOS status item, the background app, login item, the startup dialog

The AppKit half of the app lifecycle (task 17, waves 3 and 3b) is a live-system surface no test platform reaches (see AGENTS.md → App lifecycle → "What only a human can verify"). This is the release-verification procedure for it. The cheap oracle for the activation policy:

```text
lsappinfo info -only ApplicationType $(pgrep -x Eidola)
# "Foreground" = Regular (Dock icon); "UIElement" = Accessory (menu-bar-only)
```

Build: `just build gui` (or `cargo build -p eidola-gui && ./scripts/package-gui-app.sh debug`), then `open crates/eidola-gui/build/Eidola.app`.

**The model in one line:** while the app is open it is an ordinary macOS app — Dock icon and menu bar even with no windows. **⌘Q retires it to the background** (windows close, Dock indicator goes, process and engines stay). **Full shutdown is the status menu's Quit Eidola.**

## The status item

1. A hexagon-lattice glyph appears at the right of the menu bar as soon as the app launches. (If the SF Symbol is missing on your macOS it falls back to the word "Eidola" — either is correct, neither should be invisible.)
2. Click it. The menu reads: **Open Eidola · New Space · ─── · engine line(s) · ─── · Quit Eidola ⌘Q**.
3. The engine section is grey and unclickable. With no models loaded it says "No models running". Load a model in Settings → Backends → Local, then reopen the status menu: the line now names the model and says "running" (and "loading…" while its engine warms). **Reopen the menu** to see the change — the menu re-reads at open, not while it is on screen.
4. **New Space** opens a fresh space window. **Open Eidola** focuses an existing window, or opens the Library when there is none.

## Closing windows changes nothing (the wave-3b reversal)

5. With a space window open, close it (⌘W). Expected: the **Dock icon stays**, the menu bar still says "Eidola", and **⌘N opens a new space**. `lsappinfo` still reads `"Foreground"`. (This is the behaviour wave 3 got wrong — the old build went menu-bar-only here and the reflexive ⌘N landed in whatever app was frontmost.)
6. Close every window and leave it a minute. Nothing changes; the app is simply an app with no windows.

## ⌘Q retires to the background

7. With one or more windows open, press **⌘Q** (or Eidola ▸ Quit). Expected: **every window closes**, the **Dock icon disappears**, the app's menu bar goes away, the status item is still in the menu bar, and the **process is still running** — `pgrep -x Eidola` finds it and `lsappinfo` now reads `"UIElement"`. *Watch for the failure this replaced:* the window must actually go — if the app goes menu-bar-only while a window is still on screen (that window then sitting under another app's menu bar), the deferred sweep has regressed.
8. **Loaded engines survive.** Load a local model first (Settings → Backends → Local → Load), confirm `pgrep -fl llama-server` lists a child, then ⌘Q. Expected: the app is `UIElement` **and the `llama-server` child is still running**. Open the status menu — the engine line still says "running", so the bus bridge and the stores are alive too.
9. **The way back.** From the retired state, each of these must bring the app back to `Foreground` with a window, **same pid**: the status menu's **Open Eidola**; the status menu's **New Space**; **Spotlight** "Eidola"; **`open -a Eidola`**; double-clicking the app in Finder. (There is no Dock icon while retired, so there is no Dock click to test.) After any of them, ⌘N / ⌘, / ⌘W all work — that is the check that matters.
10. Retire and reopen a couple of times. No flicker, no stuck state, no duplicate status items.
11. **⌘Q sticks even against work in flight.** Pick "New Space from Template ▸" (or click a Library row) and press ⌘Q immediately. Expected: the app retires and **stays** retired — no window appears a moment later and the app does not come back to the front. (The space itself still commits; only its window is abandoned.)

## Full shutdown

12. With a model loaded (as in 8), pick **Quit Eidola** from the status menu. Expected: process gone, and `pgrep -fl llama-server` empty — the status menu's Quit is the wave-2 `on_app_quit` teardown path.
13. Repeat, but this time open the status menu and press **⌘Q** while it is open, instead of clicking the row. Expected: the same full shutdown. *If nothing happens*, note it — the `NSMenuItem` key equivalent is the "⌘Q while focused on the toolbar app" affordance and the clickable row is the guaranteed door; report it so the doc can be corrected rather than worked around.
14. **⌘Q must not be stolen by the status menu while a window is open.** With a window focused and the status menu **closed**, ⌘Q must *retire* (step 7), never full-quit. (`NSStatusItem` menus are not part of the main menu, so their key equivalents should fire only while the menu is open — this step is the check on that assumption.)
15. From the **already retired** state, ⌘Q (whichever menu takes it) is a **full shutdown**, not a second retire — there is nothing left to retire.
16. **No status item ⇒ ⌘Q is the old full quit.** Hard to force deliberately; if you ever see the app launch without the menu-bar glyph, ⌘Q there must end the process rather than leave an invisible one.

## Open at login

17. Settings → General ends with a **Startup** section: "Open at login" with a switch and one muted line.
18. **Known gap, not a bug to file:** turning this on makes login open the ordinary **windowed** app, not the background/menu-bar layer — `SMAppService.mainAppService` takes no arguments, so `--windowless` is unreachable through it (options recorded in `src/login_item.rs`). The row's copy says "Eidola opens with your session" for exactly that reason; if it ever promises the menu bar again without the LaunchAgent change landing, that is the bug.
19. Running from `crates/eidola-gui/build/Eidola.app` (ad-hoc signed, not in /Applications), expect the honest unavailable state: switch dimmed and inert, line reads "Unavailable — this needs macOS 13 or later, and an installed, signed app." **This is the expected dev-build state**, not a bug. If it *is* settable in your build, so much the better — go on.
20. Copy the bundle to /Applications and launch it from there. Turn the switch on: System Settings → General → Login Items should list Eidola under "Open at Login". Turn it off: the entry disappears.
21. Turn it on, then remove/deny Eidola in System Settings → Login Items, then reopen Eidola's Settings → General. Expected: the switch still reads on and the line says macOS is waiting for you (`RequiresApproval`) — not a silent "off".
22. A refusal (unsigned bundle, denied) shows macOS's own wording in a red banner under the row and leaves the switch where the system actually is — it never flips optimistically.
23. Nothing is remembered app-side: the switch's state comes only from the system, so deleting `~/Library/Application Support/eidola` must not change it.

## The app icon

24. The Dock icon, the ⌘Tab switcher, and Finder's Get Info all show the hexagon-grid mark — a warm cluster on a dark tile, never the generic blank-document icon (which is what a missing or misnamed `Contents/Resources/AppIcon.icns` gives you). `mdls -name kMDItemDisplayName crates/eidola-gui/build/Eidola.app` is not the oracle here; look at it.
25. Shrink a Finder window to list view and check the 16 pt slot: the cells should still be countable, not a smudge. That slot is drawn from the reduced-detail master (`brand/AGENTS.md`), so if it looks like the full-detail mark scaled down, the `.icns` was built from the wrong source.

## The startup-failure dialog

The other AppKit surface no test platform reaches: it is an `NSAlert` presented before `Application::run`, so neither the driver nor the visual tier can render it (see AGENTS.md → Startup failures).

26. With Eidola already running, launch a **second** copy from a terminal: `crates/eidola-gui/build/Eidola.app/Contents/MacOS/Eidola`. Expected: a critical alert titled **"Eidola is already open"** whose body names the holding pid and the database path and says to quit it and try again, with a single **Quit** button. Pressing Quit ends that second process (exit status 1) and leaves the first one untouched — same pid, same windows, engines still loaded.
27. The same line is on stderr either way, so a terminal launch is legible without the dialog.
28. **No crash report.** The failure this replaced was a `panic_cannot_unwind` inside AppKit's `applicationDidFinishLaunching:` — SIGABRT, exit 134, and macOS's "Eidola quit unexpectedly" report. If you see that report, the construction has moved back inside `Application::run`.
29. An ordinary launch with nothing else holding the database is unchanged: no dialog, no extra window, the usual first window.
30. **A refused schema reaches the same dialog.** Hard to stage by hand — turso keeps `user_version` in the WAL, so patching the `.db` header does nothing — so this one is normally taken on trust from `a_refused_schema_surfaces_at_open_database_not_at_construction`. If you do end up with a database from an older build (a `git checkout` across a `LATEST_VERSION` bump), the launch must show "Eidola can't start" quoting the "delete your dev database" message, **not** a window full of failed panes.

## Regression sweep (nothing here should have moved)

31. Every window still opens and closes normally: space, Library, Record, Settings, Participants, About, Updates, onboarding.
32. Dock right-click → New Space / Library… still works while a window is open.
33. `--windowless` still runs with no window and quits on `SIGTERM`. On macOS it starts retired (`UIElement`) with a status item, and its status-menu Quit is a full shutdown.
34. **Linux is unchanged:** no tray, the windowed app quits with its last window / Ctrl+Q (a full shutdown), and the background layer is `eidola service` + `--windowless`.
