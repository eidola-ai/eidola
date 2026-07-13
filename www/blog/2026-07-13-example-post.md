+++
title = "How blog posts work"
description = "A draft that documents the post format; it never publishes."
draft = true
+++

# How blog posts work

This draft documents the format so the first real post has a template; `draft = true` keeps it out of the published site (`eidola-www serve` and `build --drafts` include it).

Posts live in `www/blog/` and are named `YYYY-MM-DD-slug.md` — the date prefix becomes the post date (a `date` front-matter key overrides it) and the slug becomes the URL, so this file would publish at `/blog/example-post/`.

Front matter is TOML between `+++` fences. `title` names the post (otherwise the first `# heading` is used) and `description` is the snippet shown on the blog index and in the feed; if omitted, the first paragraph is used.

The body is markdown: *emphasis*, **strong**, `inline code`, tables, footnotes, and block quotes all render in the site's book typography. Published posts appear on [the index](/blog/) and in the Atom feed automatically.
