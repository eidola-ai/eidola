//! The Eidola website generator.
//!
//! A small, deliberate static-site generator (see the workspace
//! conventions: pure Rust, pinned in `Cargo.lock`, no external SSG
//! binary). It renders three content sources into one site:
//!
//! - `www/pages/*.md` — standalone pages (`index.md` is the home page)
//! - `www/blog/YYYY-MM-DD-<slug>.md` — blog posts (`/blog/<slug>/`), plus
//!   a generated index and Atom feed
//! - `docs/**/*.md` — the repo docs, rendered under `/docs/` with `.md`
//!   links rewritten to routes (or GitHub for files the site doesn't host)
//!
//! `www/static/**` is copied verbatim to `/assets/`, except `robots.txt`
//! and `favicon.svg`-style root files listed in [`ROOT_STATIC`]. The
//! circadian palette stylesheet is generated (not committed) by
//! [`circadian::stylesheet`], so its values stay pinned by unit tests.
//!
//! Output is deterministic: inputs are read in sorted order and no
//! timestamps are embedded beyond content dates.

pub mod circadian;
pub mod content;
pub mod markdown;
pub mod render;
pub mod serve;

use std::fs;
use std::path::{Path, PathBuf};

use content::{Page, PageKind, post_stem_parts, split_front_matter};
use render::{BASE_URL, NavPage, NavSection, escape_html, layout};
use serde::Deserialize;

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// Static files that belong at the site root rather than under `/assets/`.
const ROOT_STATIC: &[&str] = &["robots.txt", "favicon.svg", "CNAME"];

pub struct BuildOptions {
    /// Repo root (the directory containing `www/` and `docs/`).
    pub root: PathBuf,
    /// Output directory; recreated on each build.
    pub out: PathBuf,
    /// Include pages/posts marked `draft = true`.
    pub include_drafts: bool,
}

pub struct BuildStats {
    pub pages: usize,
    pub posts: usize,
    pub docs: usize,
}

fn read_dir_sorted(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("reading {}: {e}", dir.display()))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    entries.sort();
    Ok(entries)
}

fn load_page(path: &Path, kind: PageKind, route: String, source_dir: &str) -> Result<Page, Error> {
    let src = fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let (matter, body) =
        split_front_matter(&src).map_err(|e| format!("{}: {e}", path.display()))?;
    let rendered = markdown::render(body, source_dir);
    let title = matter
        .title
        .or(rendered.first_heading)
        .ok_or_else(|| format!("{}: no title (front matter or # heading)", path.display()))?;
    let description = matter.description.or(match kind {
        PageKind::Post => rendered.first_paragraph,
        _ => None,
    });
    Ok(Page {
        kind,
        route,
        title,
        description,
        date: matter.date,
        draft: matter.draft,
        html: rendered.html,
        headings: rendered.headings,
        source_path: None,
    })
}

fn write_page(out: &Path, page: &Page, docs_nav: Option<&[NavSection]>) -> Result<(), Error> {
    let rel = page.route.trim_matches('/');
    let dir = if rel.is_empty() {
        out.to_path_buf()
    } else {
        out.join(rel)
    };
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("index.html"), layout(page, docs_nav))?;
    Ok(())
}

/// `www/docs-nav.toml` — the docs sidebar structure. Each page entry is
/// `[path-relative-to-docs, sidebar-label]`.
#[derive(Deserialize)]
struct DocsNavFile {
    sections: Vec<DocsNavSection>,
}

#[derive(Deserialize)]
struct DocsNavSection {
    title: String,
    pages: Vec<(String, String)>,
}

/// Load and validate the docs nav: every entry must point at a real docs
/// page, and every docs page must appear exactly once — so the sidebar
/// can never silently drift from the docs tree.
fn load_docs_nav(www: &Path, doc_paths: &[String]) -> Result<Vec<NavSection>, Error> {
    let path = www.join("docs-nav.toml");
    let raw = fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let parsed: DocsNavFile =
        toml::from_str(&raw).map_err(|e| format!("{}: {e}", path.display()))?;

    let mut seen: Vec<&str> = Vec::new();
    for section in &parsed.sections {
        for (page, _) in &section.pages {
            if !doc_paths.iter().any(|p| p == page) {
                return Err(format!("docs-nav.toml lists {page}, which is not in docs/").into());
            }
            if seen.contains(&page.as_str()) {
                return Err(format!("docs-nav.toml lists {page} twice").into());
            }
            seen.push(page);
        }
    }
    let missing: Vec<&str> = doc_paths
        .iter()
        .map(String::as_str)
        .filter(|p| !seen.contains(p))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "docs pages missing from www/docs-nav.toml: {} (every doc must appear in the sidebar)",
            missing.join(", ")
        )
        .into());
    }

    Ok(parsed
        .sections
        .into_iter()
        .map(|section| NavSection {
            title: section.title,
            pages: section
                .pages
                .into_iter()
                .map(|(path, label)| NavPage {
                    route: content::docs_route(&path),
                    label,
                })
                .collect(),
        })
        .collect())
}

fn copy_tree(from: &Path, to: &Path) -> Result<(), Error> {
    fs::create_dir_all(to)?;
    for path in read_dir_sorted(from)? {
        let name = path.file_name().expect("read_dir yields named entries");
        if path.is_dir() {
            copy_tree(&path, &to.join(name))?;
        } else {
            fs::copy(&path, to.join(name))?;
        }
    }
    Ok(())
}

/// Collect `.md` files under `docs/`, returning paths relative to it.
fn collect_docs(dir: &Path, prefix: &str, into: &mut Vec<String>) -> Result<(), Error> {
    for path in read_dir_sorted(dir)? {
        let name = path
            .file_name()
            .expect("read_dir yields named entries")
            .to_string_lossy()
            .into_owned();
        let rel = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        if path.is_dir() {
            collect_docs(&path, &rel, into)?;
        } else if name.ends_with(".md") {
            into.push(rel);
        }
    }
    Ok(())
}

fn blog_index(posts: &[Page]) -> Page {
    let mut html = String::from("<h1>Blog</h1>\n");
    if posts.is_empty() {
        html.push_str("<p>Nothing here yet. Subscribe to the <a href=\"/blog/atom.xml\">feed</a> — articles are coming.</p>\n");
    } else {
        html.push_str("<ul class=\"post-list\">\n");
        for post in posts {
            let date = post.date.as_deref().unwrap_or("");
            let snippet = post
                .description
                .as_deref()
                .map(|d| format!("<p>{}</p>", escape_html(d)))
                .unwrap_or_default();
            html.push_str(&format!(
                "<li><time datetime=\"{date}\">{human}</time><a href=\"{route}\">{title}</a>{snippet}</li>\n",
                date = escape_html(date),
                human = escape_html(&render::human_date(date)),
                route = escape_html(&post.route),
                title = escape_html(&post.title),
                snippet = snippet,
            ));
        }
        html.push_str("</ul>\n");
    }
    Page {
        kind: PageKind::Page,
        route: "/blog/".into(),
        title: "Blog".into(),
        description: Some("Notes from the people building Eidola.".into()),
        date: None,
        draft: false,
        html,
        headings: Vec::new(),
        source_path: None,
    }
}

fn atom_feed(posts: &[&Page]) -> String {
    let updated = posts
        .iter()
        .filter_map(|p| p.date.as_deref())
        .max()
        .unwrap_or("2026-01-01");
    let mut entries = String::new();
    for post in posts {
        let url = format!("{}{}", BASE_URL, post.route);
        let date = post.date.as_deref().unwrap_or(updated);
        entries.push_str(&format!(
            r#"<entry>
<title>{title}</title>
<link href="{url}"/>
<id>{url}</id>
<updated>{date}T00:00:00Z</updated>
<content type="html">{content}</content>
</entry>
"#,
            title = escape_html(&post.title),
            url = url,
            date = date,
            content = escape_html(&post.html),
        ));
    }
    format!(
        r#"<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<title>Eidola blog</title>
<link href="{base}/blog/"/>
<link rel="self" href="{base}/blog/atom.xml"/>
<id>{base}/blog/</id>
<updated>{updated}T00:00:00Z</updated>
{entries}</feed>
"#,
        base = BASE_URL,
        updated = updated,
        entries = entries,
    )
}

fn sitemap(routes: &[String]) -> String {
    let mut xml = String::from(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n",
    );
    for route in routes {
        xml.push_str(&format!("<url><loc>{BASE_URL}{route}</loc></url>\n"));
    }
    xml.push_str("</urlset>\n");
    xml
}

pub fn build(opts: &BuildOptions) -> Result<BuildStats, Error> {
    let www = opts.root.join("www");
    let docs = opts.root.join("docs");
    if !www.is_dir() || !docs.is_dir() {
        return Err(format!(
            "{} does not look like the repo root (missing www/ or docs/)",
            opts.root.display()
        )
        .into());
    }
    if opts.out.exists() {
        fs::remove_dir_all(&opts.out)?;
    }
    fs::create_dir_all(&opts.out)?;

    let mut routes: Vec<String> = Vec::new();
    let mut pages: Vec<Page> = Vec::new();

    // Standalone pages.
    for path in read_dir_sorted(&www.join("pages"))? {
        if path.extension().is_none_or(|e| e != "md") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        let (kind, route) = if stem == "index" {
            (PageKind::Home, "/".to_string())
        } else {
            (PageKind::Page, format!("/{stem}/"))
        };
        pages.push(load_page(&path, kind, route, "")?);
    }

    // Blog posts, newest first.
    let mut posts: Vec<Page> = Vec::new();
    let blog_dir = www.join("blog");
    if blog_dir.is_dir() {
        for path in read_dir_sorted(&blog_dir)? {
            if path.extension().is_none_or(|e| e != "md") {
                continue;
            }
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            let Some((date, slug)) = post_stem_parts(&stem) else {
                return Err(format!(
                    "{}: blog posts must be named YYYY-MM-DD-<slug>.md",
                    path.display()
                )
                .into());
            };
            let route = format!("/blog/{slug}/");
            let mut post = load_page(&path, PageKind::Post, route, "")?;
            if post.date.is_none() {
                post.date = Some(date.to_string());
            }
            posts.push(post);
        }
    }
    posts.retain(|p| opts.include_drafts || !p.draft);
    posts.sort_by(|a, b| b.date.cmp(&a.date));

    // Docs.
    let mut doc_paths = Vec::new();
    collect_docs(&docs, "", &mut doc_paths)?;
    let docs_nav = load_docs_nav(&www, &doc_paths)?;
    let mut doc_pages: Vec<Page> = Vec::new();
    for rel in &doc_paths {
        let route = content::docs_route(rel);
        let source_dir = match rel.rsplit_once('/') {
            Some((dir, _)) => format!("docs/{dir}"),
            None => "docs".to_string(),
        };
        let mut page = load_page(&docs.join(rel), PageKind::Doc, route, &source_dir)?;
        page.source_path = Some(format!("docs/{rel}"));
        doc_pages.push(page);
    }

    pages.retain(|p| opts.include_drafts || !p.draft);

    let published_posts: Vec<&Page> = posts.iter().filter(|p| !p.draft).collect();

    // Write everything.
    let stats = BuildStats {
        pages: pages.len(),
        posts: posts.len(),
        docs: doc_pages.len(),
    };
    for page in &pages {
        write_page(&opts.out, page, None)?;
        routes.push(page.route.clone());
    }
    let index = blog_index(&posts);
    write_page(&opts.out, &index, None)?;
    routes.push(index.route.clone());
    for post in &posts {
        write_page(&opts.out, post, None)?;
        if !post.draft {
            routes.push(post.route.clone());
        }
    }
    for page in &doc_pages {
        write_page(&opts.out, page, Some(&docs_nav))?;
        routes.push(page.route.clone());
    }

    fs::create_dir_all(opts.out.join("blog"))?;
    fs::write(opts.out.join("blog/atom.xml"), atom_feed(&published_posts))?;

    routes.sort();
    fs::write(opts.out.join("sitemap.xml"), sitemap(&routes))?;

    // Static assets: root files at /, the rest under /assets/.
    let static_dir = www.join("static");
    let assets = opts.out.join("assets");
    fs::create_dir_all(&assets)?;
    for path in read_dir_sorted(&static_dir)? {
        let name = path
            .file_name()
            .expect("read_dir yields named entries")
            .to_string_lossy()
            .into_owned();
        if path.is_dir() {
            copy_tree(&path, &assets.join(&name))?;
        } else if ROOT_STATIC.contains(&name.as_str()) {
            fs::copy(&path, opts.out.join(&name))?;
        } else {
            fs::copy(&path, assets.join(&name))?;
        }
    }

    // The generated circadian palette stylesheet.
    fs::write(assets.join("circadian.css"), circadian::stylesheet())?;

    Ok(stats)
}
