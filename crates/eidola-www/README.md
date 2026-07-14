# eidola-www

The website generator for [www.eidola.ai](https://www.eidola.ai) — a small, deliberate static-site generator kept in the workspace (pure Rust, pinned in `Cargo.lock`, gated by the normal cargo checks) rather than an external SSG binary.

## What it builds

One site from three content sources:

- `www/pages/*.md` — standalone pages; `index.md` is the home page, anything else publishes at `/<name>/`.
- `www/blog/YYYY-MM-DD-<slug>.md` — blog posts at `/blog/<slug>/`, plus a generated index at `/blog/` and an Atom feed at `/blog/atom.xml`. Front matter is TOML between `+++` fences (`title`, `description`, `date`, `draft`); see the committed draft example post for the format.
- `docs/**/*.md` — the repo docs rendered under `/docs/`, straight from the tree (no copying, no front matter). Relative `.md` links that resolve inside `docs/` become site routes; links to anything else in the repo become GitHub blob URLs. Each docs page gets the docs sidebar (structure in `www/docs-nav.toml` — `[path, label]` entries in titled sections; the build fails if an entry is missing or stale, so **add a nav entry when adding a doc**), an edit-on-GitHub footer link, and — like blog posts — an in-page "On this page" rail when it has three or more h2/h3 headings (scroll-spy in `www/static/toc.js`; shown on wide screens only, with the sidebar collapsing to a `<details>` disclosure on narrow ones).

`www/static/**` is copied to `/assets/` (`robots.txt`, `favicon.svg`, and `CNAME` land at the site root). The circadian palette stylesheet (`/assets/circadian.css`) is *generated* by `src/circadian.rs` — a unit-tested port of the app's `theme.rs` palettes and tint math — so it is never edited by hand; the handwritten styles live in `www/static/site.css`, and the runtime palette selection (solar math, canonical-hour warp, character schedule, all ported from `theme.rs`/`solar.rs`) in `www/static/circadian.js`.

## Usage

```bash
just build www              # build to target/www (drafts excluded)
just run www                # dev server at http://127.0.0.1:8000 (drafts included, rebuilds on change)
cargo run -p eidola-www -- build --out _site --drafts
```

Deployment is GitHub Pages via `.github/workflows/website.yml`: PRs touching site inputs get a validating build; pushes to `main` build and deploy.

## Regenerating `www/static/zones.js`

The circadian runtime infers geography from the IANA timezone name (never geolocation), mirroring the GUI's `solar.rs` — but browsers have no tzdb, so the site ships a snapshot of the zone → representative-coordinate table. To refresh it after a tzdb update, run this from the repo root:

```bash
python3 - <<'EOF'
import re
rows = {}
path = '/var/db/timezone/zoneinfo/zone.tab'  # /usr/share/zoneinfo/zone.tab on Linux
for line in open(path):
    if line.startswith('#'): continue
    parts = line.rstrip('\n').split('\t')
    if len(parts) < 3: continue
    coord, zone = parts[1], parts[2]
    m = re.match(r'^([+-])(\d{2})(\d{2})(\d{2})?([+-])(\d{3})(\d{2})(\d{2})?$', coord)
    if not m: continue
    lat = (int(m.group(2)) + int(m.group(3))/60 + (int(m.group(4) or 0))/3600) * (1 if m.group(1)=='+' else -1)
    lon = (int(m.group(6)) + int(m.group(7))/60 + (int(m.group(8) or 0))/3600) * (1 if m.group(5)=='+' else -1)
    rows[zone] = [round(lat,2), round(lon,2)]
body = ",\n".join(f'"{z}":[{v[0]},{v[1]}]' for z,v in sorted(rows.items()))
header = """// Generated from the IANA tzdb zone.tab (representative coordinates for
// each timezone, ISO 6709 -> decimal degrees). This mirrors the GUI's
// solar.rs, which reads the same table from the OS zoneinfo directory; the
// browser has no tzdb access, so the site ships a snapshot instead.
// Regenerate with the snippet in crates/eidola-www/README.md.
"""
open('www/static/zones.js','w').write(header + "window.EIDOLA_ZONES={\n" + body + "\n};\n")
print(len(rows), "zones")
EOF
```

## Fonts

`www/static/fonts/` holds the variable Newsreader woff2 subsets (weight 200–800, normal + italic, latin/latin-ext/vietnamese) from Google Fonts, under the same SIL OFL 1.1 license as the app's bundled static instances (`OFL.txt` alongside). Browsers apply the optical-size axis automatically, so at prose size this matches the app's Newsreader 16pt faces.
