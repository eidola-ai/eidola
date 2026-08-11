#!/usr/bin/env python3
"""Generate every Eidola identity asset from one description of the geometry.

Run via `just update-brand`. The masters in this directory and every derived
asset listed in `INSTALL` are committed; this script is how they are rebuilt,
never a build step.

Rasterization needs `rsvg-convert` (preferred, if on PATH) or Google Chrome;
the macOS `.icns` additionally needs `iconutil`, so a full run is macOS-only.
"""

import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

# --- Palette ----------------------------------------------------------------
# The app's night anchors and brand warm, verbatim from
# crates/eidola-gui/src/theme.rs. The warm-on-cool tension is deliberate.
NIGHT_TOP = "#232a33"
NIGHT_BOTTOM = "#11151a"
WARM = "#c39669"
INK = "#15191e"
NIGHT_TEXT = "#d4d0c8"

# --- Geometry ---------------------------------------------------------------
# The mark is a rosette of seven regular pointy-top hexagons: one centre cell
# and a ring of six. The twelve outermost vertices of the underlying cells are
# exactly co-circular -- at radius sqrt(7)*r for cell circumradius r -- so the
# circle of the motif is a property of the geometry rather than a drawn ring.
#
# GAP is the gutter between neighbouring cells as a fraction of the lattice
# pitch; CORNER is each cell's corner radius as a fraction of its circumradius.
GAP = 0.16
CORNER = 0.12

# Reduced detail for the 16 pt macOS slot: a wider gutter and square corners
# survive a mark that is only a handful of pixels across.
GAP_SMALL = 0.24
CORNER_SMALL = 0.0

# macOS icon grid: an 824 px body on a 1024 px canvas. The body is a
# superellipse rather than a rounded rectangle -- at n = 5 it tracks Apple's
# 185.4 px continuous-corner radius to within about a pixel while keeping the
# corner curvature continuous.
CANVAS = 1024.0
BODY = 824.0
SUPERELLIPSE_N = 5.0
MARK_FRAC = 0.66
MARK_FRAC_SMALL = 0.78

# Home-screen tiles are masked by the OS, so the ground is full-bleed and the
# mark keeps well inside the mask's safe circle.
TOUCH_MARK_FRAC = 0.58

SQRT3 = math.sqrt(3.0)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def fmt(v):
    t = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t


def poly(pts):
    d = f"M{fmt(pts[0][0])} {fmt(pts[0][1])}"
    for x, y in pts[1:]:
        d += f"L{fmt(x)} {fmt(y)}"
    return d + "Z"


def hexagon(cx, cy, r):
    """Pointy-top regular hexagon: a vertex at the top, flat left/right sides."""
    return [
        (
            cx + r * math.cos(math.radians(90.0 + 60.0 * k)),
            cy - r * math.sin(math.radians(90.0 + 60.0 * k)),
        )
        for k in range(6)
    ]


def rosette(cx, cy, outer, gap, corner, fill, indent="  ", cls=None):
    """The mark, sized so its cells' outer vertices sit on radius `outer`.

    Rounded cells are drawn as an inset polygon dilated by a round join: a
    stroke of width 2q with `stroke-linejoin="round"` grows the polygon by q
    and rounds every corner to radius q, so insetting the circumradius by one
    apothem's worth of q lands the cell edges exactly where they belong. That
    keeps the whole mark a handful of straight-line paths -- no arcs, no
    filters, nothing a minimal SVG renderer has to guess at.
    """
    s = 1.0 - gap
    r = outer / math.hypot(SQRT3 * (1.0 + s / 2.0), s / 2.0)
    pitch = SQRT3 * r
    cell = r * s
    q = corner * cell
    inset = cell - 2.0 * q / SQRT3
    centres = [(cx, cy)] + [
        (
            cx + pitch * math.cos(math.radians(60.0 * i)),
            cy - pitch * math.sin(math.radians(60.0 * i)),
        )
        for i in range(6)
    ]
    # With a class the paint comes from CSS (the favicon's day/night rule);
    # otherwise it is spelled out, so the file needs no stylesheet at all.
    attrs = f'class="{cls}"' if cls else f'fill="{fill}"'
    if q > 0:
        if not cls:
            attrs += f' stroke="{fill}"'
        attrs += f' stroke-width="{fmt(2 * q)}" stroke-linejoin="round"'
    lines = [f"{indent}<g {attrs}>"]
    for x, y in centres:
        lines.append(f'{indent}  <path d="{poly(hexagon(x, y, inset))}"/>')
    lines.append(f"{indent}</g>")
    return "\n".join(lines)


def superellipse(cx, cy, half, n=SUPERELLIPSE_N, steps=180):
    pts = []
    for i in range(steps):
        t = 2.0 * math.pi * i / steps
        c, s = math.cos(t), math.sin(t)
        pts.append(
            (
                cx + half * math.copysign(abs(c) ** (2.0 / n), c),
                cy - half * math.copysign(abs(s) ** (2.0 / n), s),
            )
        )
    return poly(pts)


def svg_open(size, extra=""):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fmt(size)} {fmt(size)}"'
        f' width="{fmt(size)}" height="{fmt(size)}"{extra}>'
    )


GROUND_DEFS = (
    '  <defs>\n    <linearGradient id="ground" x1="0" y1="0" x2="0" y2="1">\n'
    f'      <stop offset="0" stop-color="{NIGHT_TOP}"/>\n'
    f'      <stop offset="1" stop-color="{NIGHT_BOTTOM}"/>\n'
    "    </linearGradient>\n  </defs>"
)


# --- Documents --------------------------------------------------------------


def mark_svg():
    """The bare mark in `currentColor`, for inlining."""
    return "\n".join(
        [
            "<!-- Generated by brand/generate.py; do not edit. -->",
            svg_open(64.0),
            rosette(32.0, 32.0, 30.0, GAP, CORNER, "currentColor"),
            "</svg>",
            "",
        ]
    )


def favicon_svg():
    """The mark, ink on paper by day and warm grey on the night ground by night."""
    body = rosette(32.0, 32.0, 30.0, GAP, CORNER, "currentColor", cls="mark")
    return "\n".join(
        [
            "<!-- Generated by brand/generate.py; do not edit. -->",
            svg_open(64.0),
            "  <style>",
            f"    .mark {{ fill: {INK}; stroke: {INK}; }}",
            "    @media (prefers-color-scheme: dark) {",
            f"      .mark {{ fill: {NIGHT_TEXT}; stroke: {NIGHT_TEXT}; }}",
            "    }",
            "  </style>",
            body,
            "</svg>",
            "",
        ]
    )


def app_icon_svg(small=False):
    """The desktop app icon on the macOS icon grid."""
    c = CANVAS / 2.0
    gap, corner, frac = (
        (GAP_SMALL, CORNER_SMALL, MARK_FRAC_SMALL)
        if small
        else (GAP, CORNER, MARK_FRAC)
    )
    return "\n".join(
        [
            "<!-- Generated by brand/generate.py; do not edit. -->",
            svg_open(CANVAS),
            GROUND_DEFS,
            f'  <path d="{superellipse(c, c, BODY / 2.0)}" fill="url(#ground)"/>',
            rosette(c, c, frac * BODY / 2.0, gap, corner, WARM),
            "</svg>",
            "",
        ]
    )


def touch_icon_svg():
    """The home-screen tile: full-bleed ground, the OS supplies the mask."""
    c = CANVAS / 2.0
    return "\n".join(
        [
            "<!-- Generated by brand/generate.py; do not edit. -->",
            svg_open(CANVAS),
            GROUND_DEFS,
            f'  <rect width="{fmt(CANVAS)}" height="{fmt(CANVAS)}" fill="url(#ground)"/>',
            rosette(c, c, TOUCH_MARK_FRAC * CANVAS / 2.0, GAP, CORNER, WARM),
            "</svg>",
            "",
        ]
    )


# --- Rasterization ----------------------------------------------------------

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _rsvg(svg_path, out_png, size):
    subprocess.run(
        [
            "rsvg-convert",
            "-w",
            str(size),
            "-h",
            str(size),
            "-o",
            out_png,
            svg_path,
        ],
        check=True,
    )


def _chrome(svg_path, out_png, size, work):
    """Screenshot the SVG at exactly `size` device pixels.

    Chrome writes the screenshot and then keeps running, so we poll for the
    output and stop it ourselves. The profile directory must not live under
    the system temp tree -- headless Chrome hangs there on macOS.
    """
    html = os.path.join(work, f"page-{size}.html")
    with open(html, "w") as f:
        f.write(
            "<!doctype html><meta charset=utf-8>"
            "<style>html,body{margin:0;padding:0;background:transparent;"
            "overflow:hidden}"
            f"img{{display:block;width:{size}px;height:{size}px}}</style>"
            f'<img src="file://{svg_path}">'
        )
    proc = subprocess.Popen(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            f"--user-data-dir={os.path.join(work, 'profile')}",
            "--virtual-time-budget=1500",
            f"--screenshot={out_png}",
            f"--window-size={size},{size}",
            "--force-device-scale-factor=1",
            "--default-background-color=00000000",
            "--hide-scrollbars",
            f"file://{html}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline, seen = time.time() + 90, -1
    while time.time() < deadline:
        if os.path.exists(out_png):
            cur = os.path.getsize(out_png)
            if cur > 0 and cur == seen:
                break
            seen = cur
        time.sleep(0.25)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


class Raster:
    def __init__(self):
        self.rsvg = shutil.which("rsvg-convert")
        if not self.rsvg and not os.path.exists(CHROME):
            sys.exit("need rsvg-convert on PATH or Google Chrome installed")
        self.work = os.path.join(HERE, ".render")
        os.makedirs(self.work, exist_ok=True)

    def png(self, svg_path, out_png, size):
        if os.path.exists(out_png):
            os.remove(out_png)
        if self.rsvg:
            _rsvg(svg_path, out_png, size)
        else:
            _chrome(svg_path, out_png, size, self.work)
        if not os.path.exists(out_png):
            sys.exit(f"failed to rasterize {svg_path} at {size}px")
        return out_png

    def cleanup(self):
        shutil.rmtree(self.work, ignore_errors=True)


# The macOS `.icns` slots. The 16 pt slot -- 16 and 32 device pixels -- is
# drawn from the reduced-detail master; everything larger uses the standard
# one. Reducing by *point* size rather than pixel size is why 32 appears in
# both columns.
ICNS_SLOTS = [
    ("icon_16x16.png", 16, True),
    ("icon_16x16@2x.png", 32, True),
    ("icon_32x32.png", 32, False),
    ("icon_32x32@2x.png", 64, False),
    ("icon_128x128.png", 128, False),
    ("icon_128x128@2x.png", 256, False),
    ("icon_256x256.png", 256, False),
    ("icon_256x256@2x.png", 512, False),
    ("icon_512x512.png", 512, False),
    ("icon_512x512@2x.png", 1024, False),
]

# Sizes shipped under share/icons/hicolor for Linux desktops, alongside the
# scalable SVG every modern shell prefers.
LINUX_PNG_SIZES = [48, 128, 256]


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    print("  ", os.path.relpath(path, ROOT))


def main():
    print("masters")
    masters = {
        "mark.svg": mark_svg(),
        "app-icon.svg": app_icon_svg(False),
        "app-icon-small.svg": app_icon_svg(True),
        "touch-icon.svg": touch_icon_svg(),
        "favicon.svg": favicon_svg(),
    }
    for name, text in masters.items():
        write(os.path.join(HERE, name), text)

    print("website")
    write(os.path.join(ROOT, "www/static/favicon.svg"), masters["favicon.svg"])

    raster = Raster()
    try:
        raster.png(
            os.path.join(HERE, "touch-icon.svg"),
            os.path.join(ROOT, "www/static/apple-touch-icon.png"),
            180,
        )
        print("   www/static/apple-touch-icon.png")

        print("linux")
        icons = os.path.join(ROOT, "releases/linux/icons/hicolor")
        scalable = os.path.join(icons, "scalable/apps/tech.m6i.Eidola.svg")
        write(scalable, masters["app-icon.svg"])
        for size in LINUX_PNG_SIZES:
            out = os.path.join(icons, f"{size}x{size}/apps/tech.m6i.Eidola.png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            raster.png(os.path.join(HERE, "app-icon.svg"), out, size)
            print("  ", os.path.relpath(out, ROOT))

        print("macos")
        if not shutil.which("iconutil"):
            sys.exit("iconutil not found: the .icns can only be built on macOS")
        iconset = tempfile.mkdtemp(dir=HERE, prefix=".iconset-")
        try:
            stage = os.path.join(iconset, "AppIcon.iconset")
            os.makedirs(stage)
            for name, size, small in ICNS_SLOTS:
                src = os.path.join(
                    HERE, "app-icon-small.svg" if small else "app-icon.svg"
                )
                raster.png(src, os.path.join(stage, name), size)
            icns = os.path.join(ROOT, "crates/eidola-gui/Support/AppIcon.icns")
            subprocess.run(["iconutil", "-c", "icns", stage, "-o", icns], check=True)
            print("   crates/eidola-gui/Support/AppIcon.icns")
        finally:
            shutil.rmtree(iconset, ignore_errors=True)
    finally:
        raster.cleanup()
    print("done")


if __name__ == "__main__":
    main()
