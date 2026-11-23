#!/usr/bin/env python3
"""The phone interface's UI test on the Android emulator: real taps and typing through
`adb shell input`, elements found by probe name in the WebView's accessibility tree
(`uiautomator dump`), as the iOS test (ios/UITests/PhoneUITests.swift) does with XCUITest:
the first run with a new account against the local dev stack and a checkout handed to the
browser, a whole conversation, settings applied live, the system back gesture, a cold
relaunch, and the update approved, fallen back from and reverted.

  ui_test.py ADB_SERIAL PACKAGE OUT

A development bundle mirrors its probes into the page as labelled, touch-transparent
elements; each one's bounds come from the dump, and a tap at its centre lands on the
canvas beneath. Screenshots go to OUT. Exits non-zero on the first failed step.
"""

import re
import struct
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

SERIAL, PKG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def adb(*args, capture=False):
    cmd = ["adb", "-s", SERIAL, *args]
    if capture:
        return subprocess.run(cmd, check=True, capture_output=True).stdout
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return None


def dump():
    """Every node with a content description: name -> (cx, cy)."""
    for _ in range(3):
        try:
            adb("shell", "uiautomator", "dump", "/sdcard/ui.xml")
            xml = adb("shell", "cat", "/sdcard/ui.xml", capture=True).decode("utf-8", "replace")
            root = ET.fromstring(xml[xml.index("<"):])
            break
        except (subprocess.CalledProcessError, ET.ParseError, ValueError):
            time.sleep(1)
    else:
        return {}, False
    found = {}
    keyboard = False
    for node in root.iter("node"):
        if node.get("package", "").endswith("inputmethod.latin"):
            keyboard = True
        # A mirrored text field reports its name as its hint.
        name = node.get("content-desc") or node.get("text") or node.get("hint") or ""
        bounds = re.findall(r"\d+", node.get("bounds", ""))
        if name and len(bounds) == 4:
            x1, y1, x2, y2 = map(int, bounds)
            found[name] = ((x1 + x2) // 2, (y1 + y2) // 2)
    return found, keyboard


def wait_for(name, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        nodes, _ = dump()
        if name in nodes:
            return nodes[name]
        time.sleep(1)
    fail(f"no {name}")


def wait_gone(name, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        nodes, _ = dump()
        if name not in nodes:
            return
        time.sleep(1)
    fail(f"{name} stayed")


def exists(name):
    nodes, _ = dump()
    return name in nodes


def tap(name, timeout=30):
    x, y = wait_for(name, timeout)
    adb("shell", "input", "tap", str(x), str(y))
    time.sleep(0.6)


def band(nodes):
    """The transcript's visible band: below the header, above the composer."""
    top = nodes.get("conversation/back", (0, 150))[1] + 90
    bottom = nodes.get("composer/field", (0, 1900))[1] - 110
    return top, max(top + 100, bottom)


def drag_transcript(dy):
    """Drag the transcript by dy pixels (positive: toward the start), inside the band."""
    nodes, _ = dump()
    top, bottom = band(nodes)
    mid = (top + bottom) // 2
    step = max(-(bottom - top) * 4 // 5, min((bottom - top) * 4 // 5, dy))
    adb("shell", "input", "swipe", "540", str(mid - step // 2), "540", str(mid + step // 2), "600")
    time.sleep(1.0)


def tap_in_transcript(name, toward_end=False, tries=14):
    """Scroll a transcript probe into the visible band, then tap it."""
    for _ in range(tries):
        nodes, _ = dump()
        top, bottom = band(nodes)
        if name in nodes:
            x, y = nodes[name]
            if top <= y <= bottom:
                adb("shell", "input", "tap", str(x), str(y))
                time.sleep(0.6)
                return
            drag_transcript(min(600, top - y + 150) if y < top else -min(600, y - bottom + 150))
        else:
            drag_transcript(-600 if toward_end else 600)
    fail(f"no {name} in view")


def keyboard_up(timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = adb("shell", "dumpsys", "input_method", capture=True).decode()
        if "mInputShown=true" in out or "mIsInputViewShown=true" in out:
            return True
        time.sleep(0.5)
    return False


def type_text(text):
    # `input text` takes %s for a space.
    adb("shell", "input", "text", text.replace(" ", "%s").replace("?", "\\?"))
    time.sleep(0.5)


def shot(name):
    png = adb("exec-out", "screencap", "-p", capture=True)
    with open(f"{OUT}/{name}.png", "wb") as f:
        f.write(png)
    print(f"shot {name}", flush=True)


def fail(message):
    shot("failure")
    print(f"FAIL: {message}", flush=True)
    sys.exit(1)


def send(text, answer):
    tap("composer/field")
    if not keyboard_up():
        fail("the keyboard did not come up")
    type_text(text)
    tap("composer/send")
    wait_for(f"conversation/post/{answer}", 90)


def launch():
    adb("shell", "am", "start", "-W", "-n", f"{PKG}/ai.eidola.shell.MainActivity")


def resumed():
    """The package of the activity in front."""
    out = adb("shell", "dumpsys", "activity", "activities", capture=True).decode()
    m = re.search(r"(?:topResumedActivity|mResumedActivity)[^\n]*?\s([a-zA-Z0-9_.]+)/", out)
    return m.group(1) if m else ""


def paper_luminance():
    """Mean luminance (0..1) of a row near the bottom of the page, above the system bar: the palette in force."""
    raw = adb("exec-out", "screencap", capture=True)
    w, h, _fmt = struct.unpack("<III", raw[:12])
    header = len(raw) - w * h * 4
    y = h * 9 // 10
    total, n = 0.0, 0
    for x in range(w // 10, w * 9 // 10, 7):
        i = header + (y * w + x) * 4
        r, g, b = raw[i], raw[i + 1], raw[i + 2]
        total += (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        n += 1
    return total / n


def swipe_body(dy):
    """Scroll a screen's body (not a transcript) by dy pixels (negative: toward the end), in
    strokes that stay well inside the screen (a stroke from an edge is a system gesture)."""
    mid = 1200
    left = dy
    while abs(left) > 1:
        step = max(-900, min(900, left))
        adb("shell", "input", "swipe", "540", str(mid - step // 2), "540", str(mid + step // 2), "500")
        time.sleep(0.8)
        left -= step


def tap_scrolling(name, tries=8):
    """Tap a probe on a scrolling screen, scrolling toward the end until it is in view."""
    for _ in range(tries):
        nodes, _ = dump()
        if name in nodes and 200 < nodes[name][1] < 2150:
            break
        swipe_body(-500)
    tap(name)


def back():
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    time.sleep(1.0)


def introduction(prefix):
    wait_for("onboarding/cta/understood", 90)
    shot(f"{prefix}-about")
    tap("onboarding/cta/understood")
    wait_for("onboarding/cta/understood")
    shot(f"{prefix}-responsibility")
    tap("onboarding/cta/understood")
    wait_for("onboarding/cta/new-account")
    shot(f"{prefix}-get-started")


def first_run():
    """A fresh install: a new account under the terms rendered, the plans, the browser."""
    introduction("20-onboarding")
    tap("onboarding/cta/new-account")
    wait_for("onboarding/terms/terms-of-service", 60)
    wait_for("onboarding/terms/privacy-policy")
    shot("23-onboarding-consent")
    tap_scrolling("onboarding/cta/create")
    time.sleep(2)
    if exists("onboarding/copy/account-id/copy"):
        fail("an account was created without agreement")
    tap_scrolling("onboarding/agree")
    shot("24-onboarding-agreed")
    tap_scrolling("onboarding/cta/create")
    wait_for("onboarding/copy/account-id/copy", 90)
    shot("25-onboarding-new-account")
    tap_scrolling("onboarding/cta/saved")
    wait_for("onboarding/cta/plans")
    shot("26-onboarding-add-credit")
    tap("onboarding/cta/plans")
    wait_for("purchase/plan/0", 60)
    shot("27-plans")
    tap("purchase/plan/0")
    deadline = time.time() + 30
    while resumed() in ("", PKG):
        if time.time() > deadline:
            fail("the browser did not open")
        time.sleep(1)
    time.sleep(4)
    print(f"checkout opened in {resumed()}", flush=True)
    shot("28-checkout-in-the-browser")
    launch()
    wait_for("purchase/opened", 30)
    time.sleep(4)
    shot("29-back-from-the-browser")
    # The system back gesture pops a screen inside the app.
    back()
    wait_for("onboarding/cta/purchase-later")
    tap("onboarding/cta/purchase-later")
    wait_for("library/settings", 30)
    shot("30-library-after-the-first-run")
    tap("library/settings")
    wait_for("settings/account")
    shot("31-settings")
    tap("settings/account")
    wait_for("account/balance", 60)
    wait_for("account/pool/0", 30)
    shot("32-account")
    tap_scrolling("account/reset")
    wait_for("account/reset/warning")
    shot("34-account-reset-armed")
    tap_scrolling("account/reset/keep")
    back()
    wait_for("settings/account")
    back()
    wait_for("library/new")


def settings():
    """Settings applied live, the wallet, About, the licences; back by the system gesture."""
    tap("library/settings")
    tap("settings/appearance")
    wait_for("appearance/day-night/night")
    time.sleep(1)
    day = paper_luminance()
    shot("40-appearance-day")
    tap("appearance/day-night/night")
    time.sleep(2)
    night = paper_luminance()
    shot("41-appearance-night-live")
    if not (day > 0.6 and night < 0.3):
        fail(f"the night palette did not apply live ({day:.2f} -> {night:.2f})")
    tap("appearance/text-size/larger")
    tap("appearance/text-size/larger")
    shot("42-appearance-larger-text-night")
    tap("appearance/text-size/reset")
    tap("appearance/day-night/system")
    time.sleep(2)
    if paper_luminance() < 0.6:
        fail("System did not follow the day setting")
    back()
    wait_for("settings/wallet")
    tap("settings/wallet")
    wait_for("wallet/refresh")
    time.sleep(2)
    shot("43-wallet")
    back()
    tap("settings/about")
    wait_for("about/licences")
    time.sleep(2)
    shot("44-about")
    tap_scrolling("about/licences")
    wait_for("licences/row/0", 30)
    shot("46-licences")
    tap("licences/row/0")
    shot("47-licence-text")
    back()
    back()
    back()
    wait_for("library/new")


def update():
    """Approved, restarted into and confirmed; a broken one falling back; reverted."""
    tap("library/settings")
    tap("settings/updates")
    wait_for("updates/serving/embedded", 30)
    deadline = time.time() + 30
    while time.time() < deadline and not (exists("updates/accept-claims") or exists("updates/approve")):
        time.sleep(1)
    if exists("updates/accept-claims"):
        shot("50-update-claims-changed")
        tap_scrolling("updates/accept-claims")
    wait_for("updates/approve", 30)
    shot("51-update-available")
    time.sleep(3)
    if not exists("updates/serving/embedded"):
        fail("something was staged before the approval")
    tap_scrolling("updates/approve")
    wait_for("library/settings", 120)
    tap("library/settings")
    tap("settings/updates")
    wait_for("updates/serving/update-confirmed", 30)
    swipe_body(-1500)
    shot("52-update-running-confirmed")
    swipe_body(1500)
    deadline = time.time() + 30
    while time.time() < deadline and not (exists("updates/accept-claims") or exists("updates/approve")):
        time.sleep(1)
    if exists("updates/accept-claims"):
        tap_scrolling("updates/accept-claims")
    wait_for("updates/approve", 60)
    tap_scrolling("updates/approve")
    time.sleep(8)
    shot("53-broken-update-on-trial")
    adb("shell", "am", "force-stop", PKG)
    launch()
    wait_for("library/settings", 90)
    tap("library/settings")
    wait_for("settings/updates")
    shot("54-settings-update-failed")
    tap("settings/updates")
    wait_for("updates/fell-back", 30)
    wait_for("updates/serving/update-confirmed")
    time.sleep(2)
    shot("55-update-fell-back")
    swipe_body(-6000)
    shot("55b-update-fell-back-running")
    tap_scrolling("updates/revert")
    wait_for("library/settings", 60)
    tap("library/settings")
    tap("settings/updates")
    wait_for("updates/serving/embedded", 30)
    swipe_body(-1500)
    shot("56-reverted-to-embedded")
    back()
    back()
    wait_for("library/new")


def main():
    adb("shell", "cmd", "uimode", "night", "no")
    launch()

    first_run()
    shot("01-library-empty")

    tap("library/new")
    wait_for("composer/field")
    tap("composer/field")
    if not keyboard_up():
        fail("the keyboard did not come up")
    shot("02-empty-keyboard-up")

    type_text("How do tides work, briefly?")
    shot("03-typed")
    tap("composer/send")
    wait_for("conversation/post/0")
    wait_for("conversation/streaming/0")
    adb("shell", "input", "keyevent", "KEYCODE_BACK")  # the keyboard down
    tap_in_transcript("conversation/row/1/thinking")
    shot("04-streaming-with-reasoning")

    wait_gone("conversation/streaming/0")
    wait_for("conversation/post/1")
    if exists("conversation/error"):
        fail("the turn failed")
    drag_transcript(900)
    shot("05-markdown-answer-top")
    drag_transcript(-2000)
    shot("05-markdown-answer-end")

    tap_in_transcript("conversation/post/0/reply")
    if not keyboard_up():
        fail("the keyboard did not come up")
    wait_for("composer/cancel-reply")
    shot("06-replying")
    type_text("And why are there two a day?")
    tap("composer/send")
    wait_for("conversation/streaming/1", 60)
    wait_gone("conversation/streaming/1")
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    tap_in_transcript("conversation/post/1/branch-previous")
    shot("07-branch-switched")
    tap_in_transcript("conversation/post/1/branch-next")
    shot("08-branch-back")
    send("Tell me more.", 4)
    send("And the moon?", 6)
    adb("shell", "input", "keyevent", "KEYCODE_BACK")
    drag_transcript(900)
    drag_transcript(900)
    shot("09-long-scrolled-up")
    for _ in range(4):
        drag_transcript(-900)
    shot("10-long-at-the-end")

    adb("shell", "cmd", "uimode", "night", "yes")
    time.sleep(2)
    shot("11-night-conversation")

    tap("conversation/back")
    wait_for("library/row/0")
    shot("12-library-night")
    adb("shell", "cmd", "uimode", "night", "no")
    time.sleep(2)
    shot("13-library-day")

    tap("library/new")
    wait_for("composer/field")
    tap("conversation/back")
    wait_for("library/row/0")
    time.sleep(1)
    if exists("library/row/1"):
        fail("an untouched conversation was kept")

    # A cold relaunch: the conversation reopens at its newest post.
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-W", "-n", f"{PKG}/ai.eidola.shell.MainActivity")
    wait_for("library/row/0", 90)
    tap("library/row/0")
    deadline = time.time() + 30
    while not any(n.startswith("conversation/post/") for n in dump()[0]):
        if time.time() > deadline:
            fail("the conversation did not reopen")
        time.sleep(1)
    shot("14-relaunched")
    back()
    wait_for("library/new")

    # The other screens at night.
    adb("shell", "cmd", "uimode", "night", "yes")
    time.sleep(2)
    tap("library/settings")
    wait_for("settings/account")
    shot("70-night-settings")
    tap("settings/account")
    wait_for("account/balance", 60)
    shot("71-night-account")
    tap_scrolling("account/purchase")
    wait_for("purchase/plan/0", 60)
    shot("72-night-plans")
    back()
    back()
    tap("settings/updates")
    wait_for("updates/check")
    time.sleep(3)
    shot("74-night-updates")
    back()
    back()
    adb("shell", "cmd", "uimode", "night", "no")
    time.sleep(2)
    wait_for("library/new")

    settings()
    update()
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
