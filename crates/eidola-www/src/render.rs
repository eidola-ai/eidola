//! The HTML shell: one layout for every page, mirroring the app's chrome —
//! a transparent title band that content fades under, a centered reading
//! column, and a quiet footer.

use crate::content::{Page, PageKind};

pub const BASE_URL: &str = "https://www.eidola.ai";

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

/// Render a full page.
pub fn layout(page: &Page) -> String {
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
{feed}<meta property="og:title" content="{title}">
<meta property="og:url" content="{canonical}">
<script defer src="/assets/zones.js"></script>
<script defer src="/assets/circadian.js"></script>
</head>
<body>
<header class="site-header">
<a class="wordmark" href="/">Eidola</a>
<nav>
<a href="/blog/"{nav_blog}>Blog</a>
<a href="/docs/"{nav_docs}>Docs</a>
<a href="https://github.com/eidola-ai/eidola">GitHub</a>
</nav>
</header>
<main class="{main_class}">
{draft_notice}{byline}{body}</main>
<footer class="site-footer">
<span>© 2026 <a href="/about/">Eidola, Inc.</a></span>
<nav>
<a href="/privacy/">Privacy</a>
<a href="/terms/">Terms</a>
</nav>
<div class="appearance" role="group" aria-label="Appearance">
<button data-appearance="auto">auto</button><button data-appearance="day">day</button><button data-appearance="night">night</button>
</div>
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

    #[test]
    fn layout_marks_active_nav_and_titles() {
        let page = Page {
            kind: PageKind::Doc,
            route: "/docs/client/".into(),
            title: "The client".into(),
            description: None,
            date: None,
            draft: false,
            html: "<h1>The client</h1>".into(),
        };
        let html = layout(&page);
        assert!(html.contains("<title>The client · Eidola</title>"));
        assert!(html.contains("<a href=\"/docs/\" class=\"active\">"));
        assert!(html.contains("atom.xml"));
        assert!(html.contains("rel=\"canonical\" href=\"https://www.eidola.ai/docs/client/\""));
    }
}
