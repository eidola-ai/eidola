//! Content loading: front matter, routes, and page metadata.

use serde::Deserialize;

/// Optional TOML front matter, fenced by `+++` lines (Zola-style).
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrontMatter {
    pub title: Option<String>,
    pub description: Option<String>,
    /// `YYYY-MM-DD`. For blog posts this overrides the filename prefix.
    pub date: Option<String>,
    /// Drafts are excluded from `build` unless `--drafts` is passed
    /// (`serve` always includes them).
    #[serde(default)]
    pub draft: bool,
}

/// Split a source file into front matter and markdown body. Files without
/// a leading `+++` fence are all body (docs pages have no front matter).
pub fn split_front_matter(src: &str) -> Result<(FrontMatter, &str), String> {
    let Some(rest) = src.strip_prefix("+++") else {
        return Ok((FrontMatter::default(), src));
    };
    let Some((raw, body)) = rest.split_once("\n+++") else {
        return Err("unterminated +++ front matter fence".into());
    };
    let matter: FrontMatter =
        toml::from_str(raw).map_err(|e| format!("invalid front matter: {e}"))?;
    // Skip the remainder of the closing fence line.
    let body = body.strip_prefix('\n').unwrap_or(body);
    Ok((matter, body))
}

/// What kind of page this is; drives layout details (title format, nav
/// highlight, whether a date byline renders).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    Home,
    Page,
    Post,
    Doc,
}

/// A fully-loaded page, ready to render.
pub struct Page {
    pub kind: PageKind,
    /// Site route with leading and trailing slash, e.g. `/docs/client/`.
    pub route: String,
    pub title: String,
    pub description: Option<String>,
    /// `YYYY-MM-DD`, blog posts only.
    pub date: Option<String>,
    pub draft: bool,
    /// Rendered markdown body.
    pub html: String,
}

/// Derive the site route for a docs source path relative to `docs/`
/// (e.g. `architecture/state.md` -> `/docs/architecture/state/`,
/// `README.md` -> `/docs/`).
pub fn docs_route(rel: &str) -> String {
    let stripped = rel.strip_suffix(".md").unwrap_or(rel);
    if stripped == "README" {
        return "/docs/".to_string();
    }
    if let Some(dir) = stripped.strip_suffix("/README") {
        return format!("/docs/{dir}/");
    }
    format!("/docs/{stripped}/")
}

/// Parse `YYYY-MM-DD-slug` from a blog post file stem.
pub fn post_stem_parts(stem: &str) -> Option<(&str, &str)> {
    if stem.len() < 12 {
        return None;
    }
    let (date, rest) = stem.split_at(10);
    let bytes = date.as_bytes();
    let digits = |r: std::ops::Range<usize>| bytes[r].iter().all(u8::is_ascii_digit);
    if digits(0..4) && bytes[4] == b'-' && digits(5..7) && bytes[7] == b'-' && digits(8..10) {
        rest.strip_prefix('-').map(|slug| (date, slug))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_roundtrip() {
        let (m, body) =
            split_front_matter("+++\ntitle = \"Hi\"\ndraft = true\n+++\n\n# Body\n").unwrap();
        assert_eq!(m.title.as_deref(), Some("Hi"));
        assert!(m.draft);
        assert_eq!(body, "\n# Body\n");
    }

    #[test]
    fn no_front_matter_is_all_body() {
        let (m, body) = split_front_matter("# Just a doc\n").unwrap();
        assert!(m.title.is_none());
        assert_eq!(body, "# Just a doc\n");
    }

    #[test]
    fn unknown_front_matter_keys_are_errors() {
        assert!(split_front_matter("+++\nttile = \"typo\"\n+++\n").is_err());
    }

    #[test]
    fn docs_routes() {
        assert_eq!(docs_route("README.md"), "/docs/");
        assert_eq!(docs_route("client.md"), "/docs/client/");
        assert_eq!(
            docs_route("architecture/state.md"),
            "/docs/architecture/state/"
        );
        assert_eq!(docs_route("contributing/README.md"), "/docs/contributing/");
    }

    #[test]
    fn post_stems() {
        assert_eq!(
            post_stem_parts("2026-07-13-hello-world"),
            Some(("2026-07-13", "hello-world"))
        );
        assert_eq!(post_stem_parts("not-a-date-post"), None);
    }
}
