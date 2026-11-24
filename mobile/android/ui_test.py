#!/usr/bin/env python3
"""The phone interface's UI test on the Android emulator: real taps and typing through
`adb shell input`, elements found by probe name in the WebView's accessibility tree
(`uiautomator dump`), as the iOS test (ios/UITests/PhoneUITests.swift) does with XCUITest.

  ui_test.py ADB_SERIAL PACKAGE OUT

A development bundle mirrors its probes into the page as labelled, touch-transparent
elements; each one's bounds come from the dump, and a tap at its centre lands on the
canvas beneath. Screenshots go to OUT. Exits non-zero on the first failed step.
"""

import re
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


def main():
    adb("shell", "cmd", "uimode", "night", "no")
    adb("shell", "am", "start", "-W", "-n", f"{PKG}/ai.eidola.shell.MainActivity")

    wait_for("library/new", 90)
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
    print("PASS", flush=True)


if __name__ == "__main__":
    main()
