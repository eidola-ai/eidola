//! End-to-end build test against the real repo content: builds the actual
//! site into a temp dir and asserts the structural invariants (routes,
//! link rewriting, feed, assets). Content-only regressions are also
//! caught in CI by the website workflow's validating build.

use std::fs;
use std::path::{Path, PathBuf};

use eidola_www::{BuildOptions, build};

fn repo_root() -> PathBuf {
    // crates/eidola-www -> repo root
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace layout")
        .to_path_buf()
}

fn out_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("eidola-www-test-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    dir
}

fn read(out: &Path, rel: &str) -> String {
    fs::read_to_string(out.join(rel)).unwrap_or_else(|e| panic!("missing {rel} in built site: {e}"))
}

#[test]
fn builds_the_real_site() {
    let out = out_dir("real");
    let stats = build(&BuildOptions {
        root: repo_root(),
        out: out.clone(),
        include_drafts: false,
    })
    .expect("site builds");

    // The three content sources all produced pages.
    assert!(stats.pages >= 3, "home + privacy + terms");
    assert!(stats.docs >= 10, "the docs tree");

    // Home page.
    let home = read(&out, "index.html");
    assert!(home.contains("<title>Eidola</title>"));
    assert!(home.contains("Intelligence should be intimate"));
    assert!(home.contains("/assets/circadian.css"));
    assert!(home.contains("class=\"prose home\""));

    // Legal placeholders.
    assert!(read(&out, "privacy/index.html").contains("Privacy policy"));
    assert!(read(&out, "terms/index.html").contains("Terms of service"));

    // Docs: index + a nested page, with .md links rewritten to routes and
    // out-of-docs links rewritten to GitHub.
    let docs_index = read(&out, "docs/index.html");
    assert!(docs_index.contains("href=\"/docs/paradigm/\""));
    assert!(docs_index.contains("https://github.com/eidola-ai/eidola/blob/main/README.md"));
    assert!(out.join("docs/architecture/state/index.html").exists());
    assert!(out.join("docs/contributing/index.html").exists());

    // Docs sidebar: on every docs page (both renderings), current page
    // marked, driven by www/docs-nav.toml — and never on non-doc pages.
    let client = read(&out, "docs/client/index.html");
    assert!(client.contains("class=\"docs-sidebar\""));
    assert!(client.contains("class=\"docs-nav-inline\""));
    assert!(client.contains("<a href=\"/docs/client/\" aria-current=\"page\">The client</a>"));
    assert!(client.contains("<p class=\"docs-nav-title\">Start here</p>"));
    assert!(docs_index.contains("<a href=\"/docs/\" aria-current=\"page\">Overview</a>"));
    assert!(!home.contains("docs-sidebar"));
    assert!(!read(&out, "blog/index.html").contains("docs-sidebar"));

    // In-page ToC: on structured docs (entries link the generated
    // heading ids), never on plain pages like privacy.
    let trust_root = read(&out, "docs/trust-root/index.html");
    assert!(trust_root.contains("class=\"toc\""));
    assert!(trust_root.contains("<a href=\"#whats-pinned\">"));
    assert!(trust_root.contains("class=\"toc-h1\""));
    assert!(trust_root.contains("/assets/toc.js"));
    assert!(!read(&out, "privacy/index.html").contains("class=\"toc\""));
    assert!(read(&out, "assets/toc.js").contains("aria-current"));

    // Every docs page carries an edit link to its own source; nothing
    // else does.
    assert!(client.contains("https://github.com/eidola-ai/eidola/edit/main/docs/client.md"));
    assert!(
        read(&out, "docs/architecture/state/index.html")
            .contains("edit/main/docs/architecture/state.md")
    );
    assert!(!home.contains("page-edit"));
    assert!(!read(&out, "privacy/index.html").contains("page-edit"));

    // Blog: index and feed exist even with no published posts; the draft
    // example post must not publish.
    let blog = read(&out, "blog/index.html");
    assert!(blog.contains("<h1>Blog</h1>"));
    assert!(!blog.contains("example-post"));
    assert!(!out.join("blog/example-post/index.html").exists());
    assert!(read(&out, "blog/atom.xml").contains("<feed"));

    // Assets: generated palette css, handwritten css, runtime js, fonts,
    // root-level static files.
    let palette = read(&out, "assets/circadian.css");
    assert!(palette.contains(":root[data-palette=\"night-warm\"]"));
    assert!(read(&out, "assets/site.css").contains("Newsreader"));
    assert!(read(&out, "assets/circadian.js").contains("canonicalHour"));
    assert!(read(&out, "assets/zones.js").contains("America/Los_Angeles"));
    assert!(out.join("assets/fonts/newsreader-latin.woff2").exists());
    assert!(out.join("robots.txt").exists());
    assert!(out.join("favicon.svg").exists());

    // Sitemap covers the key routes.
    let sitemap = read(&out, "sitemap.xml");
    for route in ["/", "/docs/client/", "/blog/", "/privacy/", "/terms/"] {
        assert!(
            sitemap.contains(&format!("<loc>https://www.eidola.ai{route}</loc>")),
            "sitemap missing {route}"
        );
    }

    fs::remove_dir_all(&out).ok();
}

#[test]
fn drafts_are_included_when_requested() {
    let out = out_dir("drafts");
    build(&BuildOptions {
        root: repo_root(),
        out: out.clone(),
        include_drafts: true,
    })
    .expect("site builds");

    let post = read(&out, "blog/example-post/index.html");
    assert!(post.contains("draft-notice"));
    // Drafts render but never enter the feed or sitemap.
    assert!(!read(&out, "blog/atom.xml").contains("example-post"));
    assert!(!read(&out, "sitemap.xml").contains("example-post"));

    fs::remove_dir_all(&out).ok();
}
