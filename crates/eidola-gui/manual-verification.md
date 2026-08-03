# Manual verification — macOS status item, Accessory flip, login item

The AppKit half of the app lifecycle (task 17 wave 3) is a live-system surface no test platform reaches (see AGENTS.md → App lifecycle → "What only a human can verify"). This is the release-verification procedure for it. The cheap oracle for the activation flip:

```text
lsappinfo info -only ApplicationType $(pgrep -x Eidola)
# "Foreground" = Regular (Dock icon); "UIElement" = Accessory (menu-bar-only)
```

Build: `just build gui` (or `cargo build -p eidola-gui && ./scripts/package-gui-app.sh debug`), then `open crates/eidola-gui/build/Eidola.app`.

## The status item

1. A hexagon-lattice glyph appears at the right of the menu bar as soon as the app launches. (If the SF Symbol is missing on your macOS it falls back to the word "Eidola" — either is correct, neither should be invisible.)
2. Click it. The menu reads: **Open Eidola · New Space · ─── · engine line(s) · ─── · Quit Eidola**.
3. The engine section is grey and unclickable. With no models loaded it says "No models running". Load a model in Settings → Backends → Local, then reopen the status menu: the line now names the model and says "running" (and "loading…" while its engine warms). **Reopen the menu** to see the change — the menu re-reads at open, not while it is on screen.
4. **New Space** opens a fresh space window. **Open Eidola** focuses an existing window, or opens the Library when there is none.

### The Accessory flip

5. With a window open, the Dock icon is there and the menu bar says "Eidola".
6. Close every window (⌘W each). Expected: the **Dock icon disappears**, the app's menu bar goes away, the process keeps running, and the status item is still in the menu bar. `lsappinfo info -only ApplicationType $(pgrep -x Eidola)` should now read `"UIElement"`.
7. From the status menu pick **New Space**. Expected: the Dock icon comes back, the app comes to the front, the menu bar is Eidola's again, and ⌘N / ⌘, / ⌘W all work (this is the check that matters — a window opened without the flip would sit under another app's menu bar).
8. Repeat close-to-zero → reopen a couple of times; no flicker, no stuck state, no duplicate status items.
9. Reopen paths while Accessory: Spotlight "Eidola", `open -a Eidola`, and double-clicking the app in Finder should each raise a window (the Dock icon is gone, so there is no Dock click to test).

### Quit kills everything

10. Load a local model (Settings → Backends → Local → Load) and confirm `pgrep -fl llama-server` lists a child.
11. Quit from the **status menu's** Quit Eidola. Expected: process gone, and `pgrep -fl llama-server` empty — the status menu dispatches the same `Quit` action ⌘Q does, so wave 2's `on_app_quit` engine teardown runs.
12. Repeat with ⌘Q for the control.

### Open at login

13. Settings → General now ends with a **Startup** section: "Open at login" with a switch and one muted line.
14. Running from `crates/eidola-gui/build/Eidola.app` (ad-hoc signed, not in /Applications), expect the honest unavailable state: switch dimmed and inert, line reads "Unavailable — this needs macOS 13 or later, and an installed, signed app." **This is the expected dev-build state**, not a bug. If it *is* settable in your build, so much the better — go on.
15. Copy the bundle to /Applications and launch it from there. Turn the switch on: System Settings → General → Login Items should list Eidola under "Open at Login". Turn it off: the entry disappears.
16. Turn it on, then remove/deny Eidola in System Settings → Login Items, then reopen Eidola's Settings → General. Expected: the switch still reads on and the line says macOS is waiting for you (`RequiresApproval`) — not a silent "off".
17. A refusal (unsigned bundle, denied) shows macOS's own wording in a red banner under the row and leaves the switch where the system actually is — it never flips optimistically.
18. Nothing is remembered app-side: the switch's state comes only from the system, so deleting `~/Library/Application Support/eidola` must not change it.

### Regression sweep (nothing here should have moved)

19. Every window still opens and closes normally: space, Library, Record, Settings, Participants, About, Updates, onboarding.
20. Dock right-click → New Space / Library… still works while a window is open.
21. `--windowless` still runs with no window and quits on `SIGTERM`.
