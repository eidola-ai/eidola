# brand — the Eidola mark

The single home for Eidola's identity geometry. Everything square that ships with the product — the macOS app icon (light / dark / tinted), the Linux themed icon (including the symbolic mark), the website favicon and home-screen tile — is generated from this directory by `just update-brand`. Workspace context lives in the top-level `AGENTS.md`.

## The mark

A rosette of seven regular **pointy-top hexagons**: one centre cell and a ring of six edge-sharing neighbours, separated by an even gutter.

The circle of the motif is a property of the geometry rather than a drawn ring: the twelve outermost vertices of the seven cells are **exactly co-circular**, at radius `sqrt(7) * r` for cell circumradius `r`. The mark is therefore a hexagon grid inscribed in a circle without anything being inscribed in anything — which is both the more original drawing and the more honest one.

Two knobs, both fractions so the mark is resolution-free:

| Constant | Value | Meaning |
|---|---|---|
| `GAP` | 0.16 | gutter between cells, as a fraction of the lattice pitch |
| `CORNER` | 0.12 | each cell's corner radius, as a fraction of its circumradius |

`GAP` was tuned against real 16 px and 20 px rasters (not scaled previews): below ~0.12 the gutters close into a blob at 16 px, above ~0.20 the mark reads as scattered dots at every size. Rounded cells are drawn as an inset polygon dilated by a round stroke join, so the whole mark stays a handful of straight-line paths — no arcs, no filters, nothing a minimal SVG renderer has to guess at.

**Reduced detail for the 16 pt macOS slot** (`GAP_SMALL` 0.24, square corners, mark scaled to 0.78 of the icon body): a mark eight pixels across cannot carry a 0.16 gutter. The reduction is by *point* size, not pixel size, which is why 32 px is rasterized twice — once for `icon_16x16@2x`, once for `icon_32x32`. That slot lives only in the flattened `.icns` (pre-Tahoe, and Tahoe's fallback); Icon Composer has no per-point-size master, so `Assets.car` always ships the full-detail mark.

## Colour

Anchors from the app's circadian pair (`crates/eidola-gui/src/theme.rs`). Light and dark follow that pair; tinted is Apple's monochrome rendition, so the mark is white and the system supplies the tint.

| Appearance | Ground | Mark |
|---|---|---|
| Light (Default) | paper `#ffffff` → `#f1f1f1` (sidebar, lit from above) | ink `#15191e` |
| Dark | night `#232a33` → `#11151a` (the night anchor, "reading lamp at midnight") | brand warm `#c39669` |
| Tinted | system-derived — do not set a top-level fill | white |

Favicon: no ground at all. The bare mark is `#15191e` ink by day and `#d4d0c8` warm grey under `prefers-color-scheme: dark`, which is the website's circadian doctrine applied to a tab.

The Linux scalable icon carries both desktop palettes behind the same `prefers-color-scheme` rule. The symbolic icon is the bare mark in `currentColor`. Flattened PNGs (`.icns` slots, Linux raster sizes, the home-screen tile) cannot ask, so each is one appearance: light for the desktop tiles, night for the touch icon.

## The macOS icon grid

Flattened tiles (`app-icon-*.svg`, the `.icns`, Linux PNGs) are an 824 px body on a 1024 px canvas — Apple's icon grid. The body is a **superellipse**, not a rounded rectangle: at `n = 5` it tracks Apple's 185.4 px continuous-corner radius to within about a pixel while keeping the corner curvature continuous, which is what makes the tile read as native beside system icons. No drop shadow: `.icns` icons are composited as-is, and a flat tile is the honest choice for a flat mark.

The Icon Composer document (`brand/AppIcon.icon`) leaves the squircle to the OS and only paints the fill + mark. That is what lets Tahoe apply Liquid Glass, dark, and tinted appearances. Do not flatten the superellipse into the `.icon` layer.

## SF Symbols is not a source

The GUI's status-item glyph is Apple's `circle.hexagongrid` SF Symbol (`crates/eidola-gui/src/status_item/macos.rs`), which is licensed for use as a symbol *inside* an app and explicitly **not** as an app icon, logo, or trademark. Nothing in this directory derives from that asset: the geometry here is computed from scratch and shares only the motif, which is not Apple's. **Never export, trace, or embed an SF Symbol into a brand asset.**

## Regenerating

```sh
just update-brand      # macOS only: iconutil + actool + ictool (full Xcode)
```

`generate.py` writes the masters in this directory *and* installs every derived asset, so a run is the whole update:

| Destination | What |
|---|---|
| `brand/{mark,favicon,app-icon,app-icon-light,app-icon-dark,app-icon-small,app-icon-small-dark,touch-icon}.svg` | vector masters |
| `brand/AppIcon.icon/` | Icon Composer document (fill + white-on-transparent mark) |
| `crates/eidola-gui/Support/AppIcon.icns` | flattened macOS app icon (10 slots, light, 16 pt reduced) |
| `crates/eidola-gui/Support/Assets.car` | compiled catalog Tahoe reads via `CFBundleIconName` |
| `www/static/favicon.svg` | site favicon |
| `www/static/apple-touch-icon.png` | 180 px home-screen tile (night, full-bleed) |
| `releases/linux/icons/hicolor/scalable/apps/ai.eidola.app.svg` | adaptive desktop icon |
| `releases/linux/icons/hicolor/symbolic/apps/ai.eidola.app-symbolic.svg` | symbolic (mono) mark |
| `releases/linux/icons/hicolor/{48,128,256}x{48,128,256}/apps/ai.eidola.app.png` | light flattened tiles |

PNGs are painted from the same geometry that emits the SVGs, so a run does not need Chrome or `rsvg-convert`, and the bytes do not move when a rasterizer's version does. The `.icns` is assembled with `iconutil`; `ictool` must render Default, Dark, TintedLight, TintedDark, ClearLight, and ClearDark before `actool` compiles `Assets.car`. Every output is committed, so no build step needs any of them — but the committed rasters and catalog move if the geometry or the compiler changes, which (like any asset change) moves the desktop narHashes. That is a release-time reconciliation, never part of a feature change.
