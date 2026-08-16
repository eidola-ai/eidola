#!/usr/bin/env python3
"""Generate every Eidola identity asset from one description of the geometry.

Run via `just update-brand`. The masters in this directory and every derived
asset listed in `INSTALL` are committed; this script is how they are rebuilt,
never a build step.

PNGs are rasterized from the same geometry that emits the SVGs (no
Chrome/`rsvg-convert`); the macOS `.icns` needs `iconutil` and the Liquid
Glass `Assets.car` needs `actool` (full Xcode), so a full run is macOS-only.
"""

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

# --- Palette ----------------------------------------------------------------
# Anchors from crates/eidola-gui/src/theme.rs. Light and dark follow the
# circadian pair; tinted is Apple's monochrome rendition, so the mark is
# white and the system supplies the tint.
#
# Light: paper ground, ink mark — day's "good paper at noon", and the same
# ink the favicon uses by day. Dark: the night tile the icon always was —
# cool ground, brand warm mark. Tinted: the mark alone, in white.
NIGHT_TOP = "#232a33"
NIGHT_BOTTOM = "#11151a"
DAY_TOP = "#ffffff"
DAY_BOTTOM = "#f1f1f1"  # theme.sidebar — a hair off paper, lit from above
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
# corner curvature continuous. Flattened `.icns` / Linux tiles still draw this
# body themselves; the Icon Composer `.icon` leaves the squircle to the OS
# and only paints the fill + mark.
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

# Icon Composer.app ships next to the active developer dir. `actool` is
# xcrun-resolved so a beta Xcode still works.
ICTOOL_REL = "Applications/Icon Composer.app/Contents/Executables/ictool"


def fmt(v):
    t = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t


def srgb(hex_color):
    """`extended-srgb:r,g,b,a` for Icon Composer fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return f"extended-srgb:{r:.5f},{g:.5f},{b:.5f},1.00000"


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
    # With a class the paint comes from CSS (the favicon's day/night rule,
    # the Linux scalable icon's light/dark rule); otherwise it is spelled
    # out, so the file needs no stylesheet at all.
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


def ground_defs(top, bottom, css=False):
    if css:
        return (
            "  <defs>\n    <linearGradient id=\"ground\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n"
            '      <stop class="ground-top" offset="0"/>\n'
            '      <stop class="ground-bottom" offset="1"/>\n'
            "    </linearGradient>\n  </defs>"
        )
    return (
        "  <defs>\n    <linearGradient id=\"ground\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">\n"
        f'      <stop offset="0" stop-color="{top}"/>\n'
        f'      <stop offset="1" stop-color="{bottom}"/>\n'
        "    </linearGradient>\n  </defs>"
    )


# --- Documents --------------------------------------------------------------


def mark_svg():
    """The bare mark in `currentColor`, for inlining and the Linux symbolic icon."""
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


def _icon_geometry(small):
    if small:
        return GAP_SMALL, CORNER_SMALL, MARK_FRAC_SMALL
    return GAP, CORNER, MARK_FRAC


def app_icon_svg(small=False, appearance="dark"):
    """A flattened desktop tile on the macOS icon grid.

    `appearance` is `light`, `dark`, or `adaptive`. Adaptive carries both
    palettes behind `prefers-color-scheme` (the Linux scalable icon); the
    flattened files are what we rasterize, because a PNG cannot ask.
    """
    c = CANVAS / 2.0
    gap, corner, frac = _icon_geometry(small)
    mark_r = frac * BODY / 2.0
    if appearance == "adaptive":
        style = "\n".join(
            [
                "  <style>",
                f"    .ground-top {{ stop-color: {DAY_TOP}; }}",
                f"    .ground-bottom {{ stop-color: {DAY_BOTTOM}; }}",
                f"    .mark {{ fill: {INK}; stroke: {INK}; }}",
                "    @media (prefers-color-scheme: dark) {",
                f"      .ground-top {{ stop-color: {NIGHT_TOP}; }}",
                f"      .ground-bottom {{ stop-color: {NIGHT_BOTTOM}; }}",
                f"      .mark {{ fill: {WARM}; stroke: {WARM}; }}",
                "    }",
                "  </style>",
            ]
        )
        return "\n".join(
            [
                "<!-- Generated by brand/generate.py; do not edit. -->",
                svg_open(CANVAS),
                style,
                ground_defs(DAY_TOP, DAY_BOTTOM, css=True),
                f'  <path d="{superellipse(c, c, BODY / 2.0)}" fill="url(#ground)"/>',
                rosette(c, c, mark_r, gap, corner, INK, cls="mark"),
                "</svg>",
                "",
            ]
        )
    if appearance == "light":
        top, bottom, mark = DAY_TOP, DAY_BOTTOM, INK
    else:
        top, bottom, mark = NIGHT_TOP, NIGHT_BOTTOM, WARM
    return "\n".join(
        [
            "<!-- Generated by brand/generate.py; do not edit. -->",
            svg_open(CANVAS),
            ground_defs(top, bottom),
            f'  <path d="{superellipse(c, c, BODY / 2.0)}" fill="url(#ground)"/>',
            rosette(c, c, mark_r, gap, corner, mark),
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
            ground_defs(NIGHT_TOP, NIGHT_BOTTOM),
            f'  <rect width="{fmt(CANVAS)}" height="{fmt(CANVAS)}" fill="url(#ground)"/>',
            rosette(c, c, TOUCH_MARK_FRAC * CANVAS / 2.0, GAP, CORNER, WARM),
            "</svg>",
            "",
        ]
    )


def mark_layer_svg():
    """White-on-transparent mark for Icon Composer. Fills recolor through alpha."""
    c = CANVAS / 2.0
    return "\n".join(
        [
            "<!-- Generated by brand/generate.py; do not edit. -->",
            svg_open(CANVAS),
            rosette(c, c, MARK_FRAC * BODY / 2.0, GAP, CORNER, "#ffffff"),
            "</svg>",
            "",
        ]
    )


def icon_composer_document():
    """icon.json: light / dark / tinted specializations, macOS squares only."""
    return {
        "fill-specializations": [
            {
                "value": {
                    "linear-gradient": [srgb(DAY_TOP), srgb(DAY_BOTTOM)],
                }
            },
            {
                "appearance": "dark",
                "value": {
                    "linear-gradient": [srgb(NIGHT_TOP), srgb(NIGHT_BOTTOM)],
                },
            },
        ],
        "groups": [
            {
                "name": "Mark",
                "layers": [
                    {
                        "name": "Hexagon grid",
                        "image-name": "mark.svg",
                        "glass": True,
                        "fill-specializations": [
                            {"value": {"solid": srgb(INK)}},
                            {
                                "appearance": "dark",
                                "value": {"solid": srgb(WARM)},
                            },
                            {
                                "appearance": "tinted",
                                "value": {
                                    "solid": "extended-srgb:1.00000,1.00000,1.00000,1.00000"
                                },
                            },
                        ],
                    }
                ],
                "shadow": {"kind": "neutral", "opacity": 0.35},
                "specular": True,
                "translucency": {"enabled": False, "value": 0.5},
            }
        ],
        "supported-platforms": {"squares": ["macOS"]},
    }


# --- Rasterization ----------------------------------------------------------
# The vector masters are SVGs; the PNGs are painted from the same geometry
# so a run does not need Chrome or rsvg-convert, and the bytes do not move
# when a rasterizer's version does.


def _hex_rgb(hex_color):
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _dist_to_filled_poly(px, py, verts):
    """Signed Euclidean distance to a convex CCW polygon: negative inside."""
    n = len(verts)
    inside = True
    min_d2 = 1e300
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        if ex * (py - y1) - ey * (px - x1) < 0:
            inside = False
        elen2 = ex * ex + ey * ey
        t = 0.0 if elen2 == 0 else max(0.0, min(1.0, ((px - x1) * ex + (py - y1) * ey) / elen2))
        dx = px - (x1 + t * ex)
        dy = py - (y1 + t * ey)
        min_d2 = min(min_d2, dx * dx + dy * dy)
    d = math.sqrt(min_d2)
    return -d if inside else d


def _cell_params(cx, cy, outer, gap, corner):
    s = 1.0 - gap
    r = outer / math.hypot(SQRT3 * (1.0 + s / 2.0), s / 2.0)
    pitch = SQRT3 * r
    cell = r * s
    q = corner * cell
    inset = cell - (0.0 if q == 0 else 2.0 * q / SQRT3)
    centres = [(cx, cy)] + [
        (
            cx + pitch * math.cos(math.radians(60.0 * i)),
            cy - pitch * math.sin(math.radians(60.0 * i)),
        )
        for i in range(6)
    ]
    # hexagon() walks clockwise in y-down SVG space; the distance
    # function wants CCW so the interior is to the left of each edge.
    return [(list(reversed(hexagon(x, y, inset))), q) for x, y in centres]


def _coverage_tile(px, py, cx, cy, half, cells, full_bleed):
    """Coverage of the tile body and of the mark, at one sample point."""
    if full_bleed:
        body = 1.0
    else:
        ax = abs(px - cx) / half
        ay = abs(py - cy) / half
        body = 1.0 if ax ** SUPERELLIPSE_N + ay ** SUPERELLIPSE_N <= 1.0 else 0.0
    mark = 0.0
    if body > 0:
        for verts, q in cells:
            if _dist_to_filled_poly(px, py, verts) <= q:
                mark = 1.0
                break
    return body, mark


def paint_tile(size, appearance, small=False, full_bleed=False):
    """RGBA bytes of one flattened tile at `size` device pixels."""
    if appearance == "light":
        top, bottom, mark_rgb = _hex_rgb(DAY_TOP), _hex_rgb(DAY_BOTTOM), _hex_rgb(INK)
    else:
        top, bottom, mark_rgb = _hex_rgb(NIGHT_TOP), _hex_rgb(NIGHT_BOTTOM), _hex_rgb(WARM)
    scale = size / CANVAS
    cx = cy = CANVAS / 2.0
    half = BODY / 2.0
    gap, corner, frac = _icon_geometry(small)
    outer = (CANVAS / 2.0 * TOUCH_MARK_FRAC) if full_bleed else (frac * BODY / 2.0)
    cells = _cell_params(cx, cy, outer, gap, corner)
    samples = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    n_samp = float(len(samples))
    out = bytearray(size * size * 4)
    i = 0
    for y in range(size):
        for x in range(size):
            body_acc = mark_acc = 0.0
            for ox, oy in samples:
                # SVG y grows down; our hexagon() already uses canvas y-down.
                px = (x + ox) / scale
                py = (y + oy) / scale
                b, m = _coverage_tile(px, py, cx, cy, half, cells, full_bleed)
                body_acc += b
                mark_acc += m
            body_acc /= n_samp
            mark_acc /= n_samp
            t = (y + 0.5) / size
            gr = int(top[0] + (bottom[0] - top[0]) * t)
            gg = int(top[1] + (bottom[1] - top[1]) * t)
            gb = int(top[2] + (bottom[2] - top[2]) * t)
            # Over-composite the mark on the ground, then multiply by body
            # coverage so the superellipse edge anti-aliases against clear.
            a_mark = mark_acc
            r = int(gr * (1.0 - a_mark) + mark_rgb[0] * a_mark)
            g = int(gg * (1.0 - a_mark) + mark_rgb[1] * a_mark)
            b = int(gb * (1.0 - a_mark) + mark_rgb[2] * a_mark)
            a = int(round(255 * body_acc))
            out[i] = r
            out[i + 1] = g
            out[i + 2] = b
            out[i + 3] = a
            i += 4
    return out


def write_png(path, rgba, size):
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    raw = b"".join(b"\x00" + bytes(rgba[y * size * 4 : (y + 1) * size * 4]) for y in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)


def scale_rgba(rgba, src, dst):
    """Integer box-downsample; `src` must be a multiple of `dst`."""
    step = src // dst
    if step * dst != src:
        raise ValueError(f"cannot box-scale {src} to {dst}")
    inv = 1.0 / (step * step)
    out = bytearray(dst * dst * 4)
    for y in range(dst):
        for x in range(dst):
            r = g = b = a = 0
            for oy in range(step):
                row = ((y * step + oy) * src + x * step) * 4
                for ox in range(step):
                    i = row + ox * 4
                    r += rgba[i]
                    g += rgba[i + 1]
                    b += rgba[i + 2]
                    a += rgba[i + 3]
            j = (y * dst + x) * 4
            out[j] = int(r * inv)
            out[j + 1] = int(g * inv)
            out[j + 2] = int(b * inv)
            out[j + 3] = int(a * inv)
    return out


def write_tile_png(path, size, appearance, small=False, full_bleed=False, master=None, master_size=None):
    if master is not None and master_size is not None and master_size % size == 0:
        rgba = scale_rgba(master, master_size, size) if size != master_size else master
    else:
        rgba = paint_tile(size, appearance, small=small, full_bleed=full_bleed)
    write_png(path, rgba, size)


# The macOS `.icns` slots. The 16 pt slot -- 16 and 32 device pixels -- is
# drawn from the reduced-detail master; everything larger uses the standard
# one. Reducing by *point* size rather than pixel size is why 32 appears in
# both columns. Pre-Tahoe reads this file; Tahoe 26+ prefers Assets.car.
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


def _xcode_developer():
    try:
        return subprocess.check_output(["xcode-select", "-p"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def find_ictool():
    dev = _xcode_developer()
    if not dev:
        return None
    path = os.path.join(os.path.dirname(dev), ICTOOL_REL)
    return path if os.path.isfile(path) and os.access(path, os.X_OK) else None


def find_actool():
    try:
        path = subprocess.check_output(
            ["xcrun", "--find", "actool"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return path if path and os.path.isfile(path) else None


def verify_icon(ictool, icon_dir, work):
    """Render every appearance `ictool` will ship. A failed rendition is a
    broken `.icon` — Icon Composer has no separate validate command."""
    renditions = [
        "Default",
        "Dark",
        "TintedLight",
        "TintedDark",
        "ClearLight",
        "ClearDark",
    ]
    for rendition in renditions:
        out = os.path.join(work, f"verify-{rendition}.png")
        proc = subprocess.run(
            [
                ictool,
                icon_dir,
                "--export-image",
                "--output-file",
                out,
                "--platform",
                "macOS",
                "--rendition",
                rendition,
                "--width",
                "64",
                "--height",
                "64",
                "--scale",
                "1",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stdout or proc.stderr or "").strip()
            sys.exit(
                f"ictool failed to render {rendition} from {icon_dir}"
                + (f": {detail}" if detail else "")
            )


def compile_assets_car(actool, icon_dir, dest_car, work):
    """Compile the `.icon` into the Assets.car Tahoe reads via CFBundleIconName.

    actool also emits a legacy .icns; we keep our own (16 pt reduced-detail)
    and only install the catalog.
    """
    out = os.path.join(work, "actool")
    os.makedirs(out)
    plist = os.path.join(out, "partial.plist")
    proc = subprocess.run(
        [
            actool,
            icon_dir,
            "--compile",
            out,
            "--output-format",
            "human-readable-text",
            "--notices",
            "--warnings",
            "--errors",
            "--output-partial-info-plist",
            plist,
            "--app-icon",
            "AppIcon",
            "--include-all-app-icons",
            "--enable-on-demand-resources",
            "NO",
            "--development-region",
            "en",
            "--target-device",
            "mac",
            "--minimum-deployment-target",
            "13.0",
            "--platform",
            "macosx",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        sys.exit("actool failed to compile AppIcon.icon" + (f":\n{detail}" if detail else ""))
    produced = os.path.join(out, "Assets.car")
    if not os.path.isfile(produced):
        sys.exit(f"actool did not write Assets.car in {out}")
    os.makedirs(os.path.dirname(dest_car), exist_ok=True)
    shutil.copy2(produced, dest_car)


def main():
    print("masters")
    masters = {
        "mark.svg": mark_svg(),
        "app-icon.svg": app_icon_svg(False, "adaptive"),
        "app-icon-light.svg": app_icon_svg(False, "light"),
        "app-icon-dark.svg": app_icon_svg(False, "dark"),
        "app-icon-small.svg": app_icon_svg(True, "light"),
        "app-icon-small-dark.svg": app_icon_svg(True, "dark"),
        "touch-icon.svg": touch_icon_svg(),
        "favicon.svg": favicon_svg(),
    }
    for name, text in masters.items():
        write(os.path.join(HERE, name), text)

    print("icon composer")
    icon_dir = os.path.join(HERE, "AppIcon.icon")
    assets = os.path.join(icon_dir, "Assets")
    os.makedirs(assets, exist_ok=True)
    write(os.path.join(assets, "mark.svg"), mark_layer_svg())
    write(
        os.path.join(icon_dir, "icon.json"),
        json.dumps(icon_composer_document(), indent=2) + "\n",
    )

    print("website")
    write(os.path.join(ROOT, "www/static/favicon.svg"), masters["favicon.svg"])

    print("rasters")
    light_1024 = paint_tile(1024, "light")
    small_32 = paint_tile(32, "light", small=True)
    write_tile_png(
        os.path.join(ROOT, "www/static/apple-touch-icon.png"),
        180,
        "dark",
        full_bleed=True,
    )
    print("   www/static/apple-touch-icon.png")

    print("linux")
    icons = os.path.join(ROOT, "releases/linux/icons/hicolor")
    write(os.path.join(icons, "scalable/apps/ai.eidola.app.svg"), masters["app-icon.svg"])
    write(os.path.join(icons, "symbolic/apps/ai.eidola.app-symbolic.svg"), masters["mark.svg"])
    for size in LINUX_PNG_SIZES:
        out = os.path.join(icons, f"{size}x{size}/apps/ai.eidola.app.png")
        write_tile_png(
            out,
            size,
            "light",
            master=light_1024 if 1024 % size == 0 else None,
            master_size=1024 if 1024 % size == 0 else None,
        )
        print("  ", os.path.relpath(out, ROOT))

    print("macos")
    if not shutil.which("iconutil"):
        sys.exit("iconutil not found: the .icns can only be built on macOS")
    actool = find_actool()
    if not actool:
        sys.exit("actool not found: full Xcode is required to compile Assets.car")
    ictool = find_ictool()
    if not ictool:
        sys.exit(
            "ictool not found: Icon Composer (full Xcode) is required to verify AppIcon.icon"
        )
    work = tempfile.mkdtemp(dir=HERE, prefix=".render-")
    iconset = tempfile.mkdtemp(dir=HERE, prefix=".iconset-")
    try:
        stage = os.path.join(iconset, "AppIcon.iconset")
        os.makedirs(stage)
        for name, size, small in ICNS_SLOTS:
            if small:
                write_tile_png(
                    os.path.join(stage, name),
                    size,
                    "light",
                    small=True,
                    master=small_32 if 32 % size == 0 else None,
                    master_size=32 if 32 % size == 0 else None,
                )
            else:
                write_tile_png(
                    os.path.join(stage, name),
                    size,
                    "light",
                    master=light_1024 if 1024 % size == 0 else None,
                    master_size=1024 if 1024 % size == 0 else None,
                )
        icns = os.path.join(ROOT, "crates/eidola-gui/Support/AppIcon.icns")
        subprocess.run(["iconutil", "-c", "icns", stage, "-o", icns], check=True)
        print("   crates/eidola-gui/Support/AppIcon.icns")

        verify_icon(ictool, icon_dir, work)
        print("   AppIcon.icon (ictool ok)")

        car = os.path.join(ROOT, "crates/eidola-gui/Support/Assets.car")
        compile_assets_car(actool, icon_dir, car, work)
        print("   crates/eidola-gui/Support/Assets.car")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(iconset, ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
