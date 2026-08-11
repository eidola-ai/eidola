# eidola-www — Agent Development Guide

A small in-workspace static-site generator (pure Rust: `pulldown-cmark` + `toml`; pinned in `Cargo.lock`, no external SSG binary) that builds the public site ([www.eidola.ai](https://www.eidola.ai)).

## Content model

- Content lives in the top-level `www/` — `pages/` for standalone pages, `blog/` for `YYYY-MM-DD-<slug>.md` posts with TOML `+++` front matter, `static/` for assets.
- The docs section renders **directly from `docs/`** — single source of truth, with relative `.md` links rewritten to site routes and out-of-docs repo links rewritten to GitHub blob URLs.
- Output: pretty URLs (`/docs/client/`, `/blog/<slug>/`), a blog index + Atom feed, `sitemap.xml`, GitHub-convention heading anchors (authored `#fragment` links keep working). Deterministic (sorted input order, no timestamps).
- **Docs navigation:** the sidebar structure lives in `www/docs-nav.toml` (`[path, label]` entries in titled sections). **Adding a doc requires adding a nav entry** — the build fails on a missing or stale entry. Rendered as a sticky left rail on wide screens, a `<details>` disclosure on narrow ones. Docs and posts with ≥3 h2/h3 headings get an "On this page" right-gutter rail (scroll-spy in `www/static/toc.js`); every docs page ends with an edit-on-GitHub link.
- **Three pages are Stripe's landing targets**, not ordinary content: `payment-complete.md`, `payment-canceled.md`, and `billing.md` (the billing-portal return). Their routes are pinned by constants in `crates/eidola-server/src/account.rs`; renaming or removing one breaks a redirect a live checkout depends on, so the two move together. Each says plainly that the reader may close the window and go back to the app, and claims nothing about timing the webhook actually owns.
- Legal documents publish their exact source bytes at `/terms/source.md` and `/privacy/source.md` for the server's terms-feed poller (see `crates/eidola-server/AGENTS.md`); rendered pages carry `eidola:version` / `eidola:source-sha256` meta tags.
- **Hash-versioned docs** (`HASH_PUBLISHED_DOCS` in `lib.rs`, today just `privacy-guarantees.md`) get the same source-bytes + `eidola:source-sha256` treatment but no version number — the content hash *is* the version, and it's the same whole-file hash `release-tool attest` signs into each release attestation. Never add `version` front matter to these files: front matter would change the attested bytes, and a monotonic number implies ongoing-conduct (legal-doc) semantics. The byline links `source.md` and the file's GitHub history; release-pinned copies are reachable via the `git_commit` in any attestation.
- GFM alert blockquotes (`> [!NOTE]` / `[!TIP]` / `[!IMPORTANT]` / `[!WARNING]` / `[!CAUTION]`, via `ENABLE_GFM`) render through pulldown-cmark's `markdown-alert-*` classes; `site.css` styles them in the hairline blockquote idiom using the app's status slots (`info`/`success`/`warning`/`danger` in `circadian.rs`; important = `accent-foreground`).

## Visual system

**The app's circadian theme, ported.** `src/circadian.rs` duplicates the palette constants and tint math from `crates/eidola-gui/src/theme.rs` (day/night neutral anchors, cool/warm targets at 0.08/0.12, the 0.6× day-paper soft-tint list) and generates `/assets/circadian.css` at build time — unit tests pin the derived hex values so drift from the app palette is loud; **if the app palette changes, update `circadian.rs` to match**. `www/static/circadian.js` ports `solar.rs` + the `theme.rs` resolution logic (NOAA sunrise equation, canonical-hour warp, ±2h character windows); geography comes from the IANA timezone via a committed `zones.js` snapshot of `zone.tab` (regeneration snippet in `README.md`), never geolocation. Site defaults mirror the app: appearance `auto`, tint on; a footer control can pin day/night (localStorage). Typography is the app's book metaphor: variable Newsreader woff2 (same OFL license as the app's bundled faces) at 17px/1.65 on a 600px measure, flat heading ramp, system-ui 14px chrome, the background→transparent title-band fade.

## Build & deploy

`just build www` builds to `target/www`; `just run www` serves at `127.0.0.1:8000` with drafts + rebuild-on-change. Deployment via `.github/workflows/website.yml` (PR validating build; `main` pushes deploy to GitHub Pages). `tests/build.rs` builds the real site as a regression test.
