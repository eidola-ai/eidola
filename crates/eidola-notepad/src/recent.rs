//! File ▸ Open Recent.
//!
//! gpui does not surface AppKit's own recent-documents menu, so the list is
//! kept here: most recent first, de-duplicated, bounded, and persisted as one
//! path per line under the platform config directory. Every opened or saved
//! document is also handed to `App::add_recent_document`, which is what feeds
//! the Dock icon's own recents.

use std::path::{Path, PathBuf};

/// How many documents the menu remembers.
pub const LIMIT: usize = 12;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecentDocuments {
    paths: Vec<PathBuf>,
    store: Option<PathBuf>,
}

impl RecentDocuments {
    /// The list persisted at the platform's config directory, or an empty
    /// in-memory one when there is none.
    pub fn load_default() -> Self {
        let store = dirs::config_dir().map(|d| d.join("Eidola Notepad").join("recent.txt"));
        match store {
            Some(path) => Self::load_from(path),
            None => Self::default(),
        }
    }

    pub fn load_from(store: PathBuf) -> Self {
        let paths = std::fs::read_to_string(&store)
            .map(|s| parse(&s))
            .unwrap_or_default();
        Self {
            paths,
            store: Some(store),
        }
    }

    /// The remembered documents that still exist, most recent first.
    pub fn paths(&self) -> impl Iterator<Item = &Path> {
        self.paths
            .iter()
            .map(PathBuf::as_path)
            .filter(|p| p.exists())
    }

    /// Move `path` to the front, then persist.
    pub fn note(&mut self, path: &Path) {
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.paths.retain(|p| p != &path);
        self.paths.insert(0, path);
        self.paths.truncate(LIMIT);
        self.persist();
    }

    pub fn clear(&mut self) {
        self.paths.clear();
        self.persist();
    }

    /// Best effort: a list that cannot be written is still right for this
    /// session, and nothing about it is worth interrupting the reader for.
    fn persist(&self) {
        let Some(store) = &self.store else { return };
        if let Some(dir) = store.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let body: String = self
            .paths
            .iter()
            .map(|p| format!("{}\n", p.display()))
            .collect();
        let _ = crate::file::write_atomically(store, body.as_bytes());
    }
}

fn parse(s: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for line in s.lines().map(str::trim).filter(|l| !l.is_empty()) {
        let path = PathBuf::from(line);
        if !out.contains(&path) {
            out.push(path);
        }
    }
    out.truncate(LIMIT);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_document_leads_and_a_reopen_moves_it_up() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.md");
        let b = dir.path().join("b.md");
        std::fs::write(&a, "").unwrap();
        std::fs::write(&b, "").unwrap();
        let store = dir.path().join("recent.txt");

        let mut recent = RecentDocuments::load_from(store.clone());
        recent.note(&a);
        recent.note(&b);
        recent.note(&a);

        let names: Vec<_> = recent
            .paths()
            .map(|p| p.file_name().unwrap().to_owned())
            .collect();
        assert_eq!(names, ["a.md", "b.md"]);
        assert_eq!(
            RecentDocuments::load_from(store).paths().count(),
            2,
            "the list survives a relaunch"
        );
    }

    #[test]
    fn a_document_that_has_gone_is_not_offered() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.md");
        std::fs::write(&a, "").unwrap();
        let mut recent = RecentDocuments::load_from(dir.path().join("recent.txt"));
        recent.note(&a);
        std::fs::remove_file(&a).unwrap();
        assert_eq!(recent.paths().count(), 0);
    }

    #[test]
    fn the_list_is_bounded() {
        let dir = tempfile::tempdir().unwrap();
        let mut recent = RecentDocuments::load_from(dir.path().join("recent.txt"));
        for i in 0..LIMIT + 5 {
            let p = dir.path().join(format!("{i}.md"));
            std::fs::write(&p, "").unwrap();
            recent.note(&p);
        }
        assert_eq!(recent.paths().count(), LIMIT);
    }
}
