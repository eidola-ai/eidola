//! The HTML shell: one layout for every page, mirroring the app's chrome —
//! a transparent title band that content fades under, a centered reading
//! column, and a quiet footer.

use crate::content::{Page, PageKind};

pub const BASE_URL: &str = "https://www.eidola.ai";

/// One sidebar entry: a resolved docs route and its short label.
pub struct NavPage {
    pub route: String,
    pub label: String,
}

/// A titled group of sidebar entries (title may be empty for the
/// unlabeled top group).
pub struct NavSection {
    pub title: String,
    pub pages: Vec<NavPage>,
}

/// The docs-nav list markup, shared by the wide-screen sidebar and the
/// narrow-screen disclosure. The current page is marked with
/// `aria-current` (which the CSS styles) rather than a class.
fn docs_nav_list(sections: &[NavSection], current_route: &str) -> String {
    let mut html = String::new();
    for section in sections {
        html.push_str("<div class=\"docs-nav-group\">\n");
        if !section.title.is_empty() {
            html.push_str(&format!(
                "<p class=\"docs-nav-title\">{}</p>\n",
                escape_html(&section.title)
            ));
        }
        html.push_str("<ul>\n");
        for page in &section.pages {
            let current = if page.route == current_route {
                " aria-current=\"page\""
            } else {
                ""
            };
            html.push_str(&format!(
                "<li><a href=\"{}\"{}>{}</a></li>\n",
                escape_html(&page.route),
                current,
                escape_html(&page.label)
            ));
        }
        html.push_str("</ul>\n</div>\n");
    }
    html
}

pub fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// Human date for bylines: `2026-07-13` -> `July 13, 2026`.
pub fn human_date(iso: &str) -> String {
    let parts: Vec<&str> = iso.splitn(3, '-').collect();
    let [year, month, day] = parts[..] else {
        return iso.to_string();
    };
    const MONTHS: [&str; 12] = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ];
    let m: usize = month.parse().unwrap_or(0);
    if m == 0 || m > 12 {
        return iso.to_string();
    }
    format!(
        "{} {}, {}",
        MONTHS[m - 1],
        day.trim_start_matches('0'),
        year
    )
}

/// Render a full page. `docs_nav` supplies the docs sidebar, rendered on
/// `Doc` pages only.
pub fn layout(page: &Page, docs_nav: Option<&[NavSection]>) -> String {
    let title = match page.kind {
        PageKind::Home => "Eidola".to_string(),
        _ => format!("{} · Eidola", escape_html(&page.title)),
    };
    let description = page
        .description
        .as_deref()
        .map(|d| {
            format!(
                "<meta name=\"description\" content=\"{}\">\n",
                escape_html(d)
            )
        })
        .unwrap_or_default();
    let feed = "<link rel=\"alternate\" type=\"application/atom+xml\" title=\"Eidola blog\" href=\"/blog/atom.xml\">\n";
    let canonical = format!("{}{}", BASE_URL, page.route);
    let nav_class = |prefix: &str| {
        if page.route.starts_with(prefix) {
            " class=\"active\""
        } else {
            ""
        }
    };
    let byline = match (&page.kind, &page.date) {
        (PageKind::Post, Some(date)) => format!(
            "<p class=\"byline\"><time datetime=\"{}\">{}</time></p>\n",
            escape_html(date),
            escape_html(&human_date(date))
        ),
        _ => String::new(),
    };
    let draft_notice = if page.draft {
        "<p class=\"draft-notice\">Draft — excluded from the published site.</p>\n"
    } else {
        ""
    };
    let main_class = match page.kind {
        PageKind::Home => "prose home",
        _ => "prose",
    };
    // The docs sidebar renders twice from one source: a sticky rail in
    // the left gutter (wide screens) and a native <details> disclosure
    // above the content (narrow screens) — CSS shows exactly one.
    let (sidebar, inline_nav) = match (page.kind, docs_nav) {
        (PageKind::Doc, Some(sections)) => {
            let list = docs_nav_list(sections, &page.route);
            (
                format!(
                    "<nav class=\"docs-sidebar\" aria-label=\"Documentation\">\n{list}</nav>\n"
                ),
                format!(
                    "<details class=\"docs-nav-inline\">\n<summary>Documentation</summary>\n<nav aria-label=\"Documentation\">\n{list}</nav>\n</details>\n"
                ),
            )
        }
        _ => (String::new(), String::new()),
    };
    // The in-page ToC rail: docs and posts with enough structure to be
    // worth navigating (the h1 doesn't count toward the threshold — it's
    // the scroll-to-top entry, not structure). toc.js drives the
    // scroll-spy state.
    let structure = page.headings.iter().filter(|h| h.level >= 2).count();
    let toc = if matches!(page.kind, PageKind::Doc | PageKind::Post) && structure >= 3 {
        let mut items = String::new();
        for heading in &page.headings {
            items.push_str(&format!(
                "<li class=\"toc-h{}\"><a href=\"#{}\">{}</a></li>\n",
                heading.level,
                escape_html(&heading.id),
                escape_html(&heading.text)
            ));
        }
        format!("<nav class=\"toc\" aria-label=\"On this page\">\n<ul>\n{items}</ul>\n</nav>\n")
    } else {
        String::new()
    };
    let toc_script = if toc.is_empty() {
        ""
    } else {
        "<script defer src=\"/assets/toc.js\"></script>\n"
    };
    // Docs render straight from the repo, so offer the door back into it.
    let edit = match (page.kind, &page.source_path) {
        (PageKind::Doc, Some(source)) => format!(
            "<p class=\"page-edit\"><a href=\"https://github.com/eidola-ai/eidola/edit/main/{}\">Edit this page on GitHub</a></p>\n",
            escape_html(source)
        ),
        _ => String::new(),
    };

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{description}<link rel="canonical" href="{canonical}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="/assets/circadian.css">
<link rel="stylesheet" href="/assets/site.css">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self';">
{feed}<meta property="og:title" content="{title}">
<meta property="og:url" content="{canonical}">
<script defer src="/assets/zones.js"></script>
<script defer src="/assets/circadian.js"></script>
{toc_script}</head>
<body>
<header class="site-header">
<a class="wordmark" href="/">Eidola</a>
<nav>
<a href="/blog/"{nav_blog}>Blog</a>
<a href="/docs/"{nav_docs}>Docs</a>
<a href="https://github.com/eidola-ai/eidola">GitHub</a>
</nav>
</header>
<div class="layout">
{sidebar}<main class="{main_class}">
{inline_nav}{draft_notice}{byline}{body}{edit}</main>
{toc}</div>
<footer class="site-footer">
<span>© 2026 <a href="/about/">Eidola, Inc.</a></span>
<nav>
<a href="/privacy/">Privacy</a>
<a href="/terms/">Terms</a>
</nav>
<details class="appearance">
<summary aria-label="Appearance" title="Appearance">☀</summary>
<div class="appearance-options" role="group" aria-label="Appearance">
<button data-appearance="auto">auto</button><button data-appearance="system">system</button><button data-appearance="day">day</button><button data-appearance="night">night</button>
</div>
</details>
</footer>
</body>
</html>
"#,
        title = title,
        description = description,
        canonical = canonical,
        feed = feed,
        nav_blog = nav_class("/blog/"),
        nav_docs = nav_class("/docs/"),
        main_class = main_class,
        sidebar = sidebar,
        inline_nav = inline_nav,
        toc = toc,
        toc_script = toc_script,
        edit = edit,
        draft_notice = draft_notice,
        byline = byline,
        body = page.html,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_humanize() {
        assert_eq!(human_date("2026-07-13"), "July 13, 2026");
        assert_eq!(human_date("2026-01-05"), "January 5, 2026");
        assert_eq!(human_date("garbage"), "garbage");
    }

    fn doc_page() -> Page {
        Page {
            kind: PageKind::Doc,
            route: "/docs/client/".into(),
            title: "The client".into(),
            description: None,
            date: None,
            draft: false,
            html: "<h1>The client</h1>".into(),
            headings: Vec::new(),
            source_path: Some("docs/client.md".into()),
        }
    }

    #[test]
    fn layout_marks_active_nav_and_titles() {
        let html = layout(&doc_page(), None);
        assert!(html.contains("<title>The client · Eidola</title>"));
        assert!(html.contains("<a href=\"/docs/\" class=\"active\">"));
        assert!(html.contains("atom.xml"));
        assert!(html.contains("rel=\"canonical\" href=\"https://www.eidola.ai/docs/client/\""));
    }

    #[test]
    fn docs_pages_get_sidebar_with_current_marker() {
        let nav = vec![NavSection {
            title: "Start here".into(),
            pages: vec![
                NavPage {
                    route: "/docs/client/".into(),
                    label: "The client".into(),
                },
                NavPage {
                    route: "/docs/server/".into(),
                    label: "The server".into(),
                },
            ],
        }];
        let html = layout(&doc_page(), Some(&nav));
        assert!(html.contains("class=\"docs-sidebar\""));
        assert!(html.contains("class=\"docs-nav-inline\""));
        assert!(html.contains("<a href=\"/docs/client/\" aria-current=\"page\">The client</a>"));
        assert!(html.contains("<a href=\"/docs/server/\">The server</a>"));
        assert!(html.contains("<p class=\"docs-nav-title\">Start here</p>"));

        // Non-doc pages never get the sidebar, even when nav is supplied.
        let mut home = doc_page();
        home.kind = PageKind::Home;
        home.route = "/".into();
        let html = layout(&home, Some(&nav));
        assert!(!html.contains("docs-sidebar"));
    }

    #[test]
    fn docs_pages_get_edit_links() {
        let html = layout(&doc_page(), None);
        assert!(html.contains(
            "<a href=\"https://github.com/eidola-ai/eidola/edit/main/docs/client.md\">Edit this page on GitHub</a>"
        ));

        // Posts and pages don't (their sources aren't repo docs).
        let mut post = doc_page();
        post.kind = PageKind::Post;
        post.source_path = None;
        assert!(!layout(&post, None).contains("page-edit"));
    }

    #[test]
    fn toc_renders_for_structured_docs_and_posts_only() {
        use crate::content::Heading;
        let heading = |level: u8, id: &str, text: &str| Heading {
            level,
            id: id.into(),
            text: text.into(),
        };

        let mut page = doc_page();
        page.headings = vec![
            heading(1, "the-client", "The client"),
            heading(2, "one", "One"),
            heading(3, "one-a", "One A"),
            heading(2, "two", "Two"),
        ];
        let html = layout(&page, None);
        assert!(html.contains("class=\"toc\""));
        assert!(html.contains("<li class=\"toc-h1\"><a href=\"#the-client\">The client</a></li>"));
        assert!(html.contains("<li class=\"toc-h3\"><a href=\"#one-a\">One A</a></li>"));
        assert!(html.contains("/assets/toc.js"));

        // Too little structure -> no rail, no script. The h1 is the
        // scroll-to-top entry, so it doesn't count toward the threshold.
        let mut short = doc_page();
        short.headings = vec![
            heading(1, "title", "Title"),
            heading(2, "only", "Only"),
            heading(2, "other", "Other"),
        ];
        let html = layout(&short, None);
        assert!(!html.contains("class=\"toc\""));
        assert!(!html.contains("/assets/toc.js"));

        // Plain pages never get one, however long.
        let mut plain = doc_page();
        plain.kind = PageKind::Page;
        plain.headings = vec![
            heading(2, "a", "A"),
            heading(2, "b", "B"),
            heading(2, "c", "C"),
        ];
        assert!(!layout(&plain, None).contains("class=\"toc\""));
    }
}
