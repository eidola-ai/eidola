//! Markdown rendering: pulldown-cmark with heading anchors and repo-aware
//! link rewriting.
//!
//! Docs pages are rendered straight from the repo's `docs/` tree, so their
//! relative links need translation: a `.md` link that resolves inside
//! `docs/` becomes a site route (`client.md` -> `/docs/client/`); anything
//! else that resolves inside the repo becomes a GitHub blob URL (the site
//! doesn't host source files). Absolute URLs, fragments, and site-absolute
//! paths pass through untouched.

use pulldown_cmark::{CowStr, Event, HeadingLevel, Options, Parser, Tag, TagEnd, html};

use crate::content::{Heading, docs_route};

const GITHUB_BLOB_BASE: &str = "https://github.com/eidola-ai/eidola/blob/main";

/// Rendered page body plus metadata extracted along the way.
pub struct Rendered {
    pub html: String,
    /// Text of the first `# h1`, used as the page title fallback.
    pub first_heading: Option<String>,
    /// Plain text of the first paragraph, for blog-index snippets.
    pub first_paragraph: Option<String>,
    /// h2/h3 headings in document order, for the in-page table of contents.
    pub headings: Vec<Heading>,
}

/// Rewrite one link destination. `source_dir` is the repo-relative
/// directory of the file being rendered (e.g. `docs` or
/// `docs/contributing`; empty for site-authored pages, whose links should
/// already be site-absolute or external).
fn rewrite_link(dest: &str, source_dir: &str) -> String {
    if dest.starts_with('#')
        || dest.starts_with('/')
        || dest.contains("://")
        || dest.starts_with("mailto:")
    {
        return dest.to_string();
    }
    let (path, fragment) = match dest.split_once('#') {
        Some((p, f)) => (p, format!("#{f}")),
        None => (dest, String::new()),
    };
    // Resolve `path` against `source_dir`, normalizing `.` and `..`.
    let mut parts: Vec<&str> = source_dir.split('/').filter(|s| !s.is_empty()).collect();
    for seg in path.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            _ => parts.push(seg),
        }
    }
    let repo_path = parts.join("/");
    if repo_path.ends_with(".md")
        && let Some(rel) = repo_path.strip_prefix("docs/")
    {
        return format!("{}{}", docs_route(rel), fragment);
    }
    format!("{GITHUB_BLOB_BASE}/{repo_path}{fragment}")
}

/// Slugify heading text into an anchor id, matching GitHub's convention —
/// the docs' authored fragment links (e.g. `gaps.md#first-install-downgrade`,
/// `#whats-pinned`) were written against GitHub's rendering: lowercase,
/// punctuation dropped (not dashed), each space/hyphen becomes a dash.
fn slugify(text: &str) -> String {
    let mut slug = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || c == '_' {
            slug.extend(c.to_lowercase());
        } else if c == ' ' || c == '-' {
            slug.push('-');
        }
    }
    slug
}

fn plain_text(events: &[Event]) -> String {
    let mut out = String::new();
    for ev in events {
        match ev {
            Event::Text(t) | Event::Code(t) => out.push_str(t),
            Event::SoftBreak | Event::HardBreak => out.push(' '),
            _ => {}
        }
    }
    out
}

/// Render a markdown body to HTML.
pub fn render(body: &str, source_dir: &str) -> Rendered {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_FOOTNOTES);
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_SMART_PUNCTUATION);
    options.insert(Options::ENABLE_HEADING_ATTRIBUTES);

    let mut events: Vec<Event> = Parser::new_ext(body, options).collect();

    // Rewrite link and image destinations.
    for ev in &mut events {
        match ev {
            Event::Start(Tag::Link { dest_url, .. })
            | Event::Start(Tag::Image { dest_url, .. }) => {
                *dest_url = CowStr::from(rewrite_link(dest_url, source_dir));
            }
            _ => {}
        }
    }

    // Assign ids to headings that lack one, and capture metadata.
    let mut first_heading = None;
    let mut first_paragraph = None;
    let mut headings: Vec<Heading> = Vec::new();
    let mut used_ids: Vec<String> = Vec::new();
    let mut i = 0;
    while i < events.len() {
        match &events[i] {
            Event::Start(Tag::Heading { level, id, .. }) => {
                let level = *level;
                let explicit = id.clone();
                let end = events[i..]
                    .iter()
                    .position(|e| matches!(e, Event::End(TagEnd::Heading(_))))
                    .map(|off| i + off)
                    .unwrap_or(events.len());
                let text = plain_text(&events[i + 1..end]);
                if level == HeadingLevel::H1 && first_heading.is_none() {
                    first_heading = Some(text.clone());
                }
                let mut slug = match explicit {
                    Some(id) => id.to_string(),
                    None => slugify(&text),
                };
                let base = slug.clone();
                let mut n = 1;
                while used_ids.contains(&slug) {
                    n += 1;
                    slug = format!("{base}-{n}");
                }
                used_ids.push(slug.clone());
                if matches!(level, HeadingLevel::H2 | HeadingLevel::H3) {
                    headings.push(Heading {
                        level: if level == HeadingLevel::H2 { 2 } else { 3 },
                        id: slug.clone(),
                        text: text.clone(),
                    });
                }
                if let Event::Start(Tag::Heading { id, .. }) = &mut events[i] {
                    *id = Some(CowStr::from(slug));
                }
                i = end;
            }
            Event::Start(Tag::Paragraph) if first_paragraph.is_none() => {
                let end = events[i..]
                    .iter()
                    .position(|e| matches!(e, Event::End(TagEnd::Paragraph)))
                    .map(|off| i + off)
                    .unwrap_or(events.len());
                first_paragraph = Some(plain_text(&events[i + 1..end]));
                i = end;
            }
            _ => i += 1,
        }
    }

    let mut out = String::new();
    html::push_html(&mut out, events.into_iter());
    Rendered {
        html: out,
        first_heading,
        first_paragraph,
        headings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_md_links_become_routes() {
        assert_eq!(rewrite_link("client.md", "docs"), "/docs/client/");
        assert_eq!(
            rewrite_link("architecture/state.md", "docs"),
            "/docs/architecture/state/"
        );
        assert_eq!(
            rewrite_link("../paradigm.md", "docs/contributing"),
            "/docs/paradigm/"
        );
        assert_eq!(
            rewrite_link("client.md#trust", "docs"),
            "/docs/client/#trust"
        );
        assert_eq!(rewrite_link("README.md", "docs"), "/docs/");
    }

    #[test]
    fn out_of_docs_links_go_to_github() {
        assert_eq!(
            rewrite_link("../README.md", "docs"),
            "https://github.com/eidola-ai/eidola/blob/main/README.md"
        );
        assert_eq!(
            rewrite_link("../crates/eidola-server/schema/schema.sql", "docs"),
            "https://github.com/eidola-ai/eidola/blob/main/crates/eidola-server/schema/schema.sql"
        );
    }

    #[test]
    fn absolute_and_fragment_links_untouched() {
        assert_eq!(
            rewrite_link("https://x.example/", "docs"),
            "https://x.example/"
        );
        assert_eq!(rewrite_link("#anchor", "docs"), "#anchor");
        assert_eq!(rewrite_link("/blog/", ""), "/blog/");
    }

    #[test]
    fn headings_get_ids_and_metadata_is_extracted() {
        let r = render(
            "# The Title\n\nFirst para.\n\n## A section\n\n## A section\n",
            "",
        );
        assert_eq!(r.first_heading.as_deref(), Some("The Title"));
        assert_eq!(r.first_paragraph.as_deref(), Some("First para."));
        assert!(r.html.contains("<h1 id=\"the-title\">"));
        assert!(r.html.contains("<h2 id=\"a-section\">"));
        assert!(r.html.contains("<h2 id=\"a-section-2\">"));
    }

    #[test]
    fn slugs_match_github() {
        assert_eq!(
            slugify("Trust root: technical specification"),
            "trust-root-technical-specification"
        );
        // Apostrophes drop rather than dash — the convention the docs'
        // authored anchors rely on (smart punctuation may have curled the
        // quote by the time it reaches slugify).
        assert_eq!(slugify("What's pinned"), "whats-pinned");
        assert_eq!(slugify("What\u{2019}s pinned"), "whats-pinned");
        // Em dashes drop, leaving a dash per surrounding space.
        assert_eq!(slugify("URL index — no hashes"), "url-index--no-hashes");
    }
}
