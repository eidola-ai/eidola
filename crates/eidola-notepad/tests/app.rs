//! The app-level flows: opening files into windows, and quitting with
//! unsaved work.

use std::path::{Path, PathBuf};

use eidola_notepad::recent::RecentDocuments;
use gpui::TestAppContext;

fn start(cx: &mut TestAppContext, dir: &Path) {
    let recent = RecentDocuments::load_from(dir.join("recent.txt"));
    cx.update(|cx| eidola_notepad::install(recent, cx));
}

fn note(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, format!("# {name}\n")).unwrap();
    path
}

fn titles(cx: &mut TestAppContext) -> Vec<String> {
    cx.update(|cx| {
        let docs = eidola_notepad::documents(cx);
        let mut names: Vec<String> = docs.iter().map(|d| d.read(cx).name()).collect();
        names.sort();
        names
    })
}

#[gpui::test]
fn a_file_takes_the_place_of_an_untouched_untitled_document(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    start(cx, dir.path());
    cx.update(eidola_notepad::new_document);
    assert_eq!(titles(cx), ["Untitled"]);

    let a = note(dir.path(), "a.md");
    cx.update(|cx| eidola_notepad::open_paths(vec![a.clone()], cx));
    cx.run_until_parked();
    assert_eq!(titles(cx), ["a.md"], "no second window for a blank page");

    // Opening it again raises the window it is in.
    cx.update(|cx| eidola_notepad::open_paths(vec![a], cx));
    cx.run_until_parked();
    assert_eq!(titles(cx), ["a.md"]);

    let b = note(dir.path(), "b.md");
    cx.update(|cx| eidola_notepad::open_paths(vec![b], cx));
    cx.run_until_parked();
    assert_eq!(titles(cx), ["a.md", "b.md"]);
}

#[gpui::test]
fn opened_documents_are_remembered_for_open_recent(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    start(cx, dir.path());
    let a = note(dir.path(), "a.md");
    cx.update(|cx| eidola_notepad::open_paths(vec![a.clone()], cx));
    cx.run_until_parked();

    let recent = RecentDocuments::load_from(dir.path().join("recent.txt"));
    let remembered: Vec<_> = recent.paths().map(Path::to_path_buf).collect();
    assert_eq!(remembered, [std::fs::canonicalize(&a).unwrap()]);
}

#[gpui::test]
fn a_file_that_is_not_text_is_refused_with_a_reason(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    start(cx, dir.path());
    let bin = dir.path().join("photo.md");
    std::fs::write(&bin, [0xff, 0xfe, 0x00]).unwrap();
    cx.update(|cx| eidola_notepad::open_paths(vec![bin], cx));
    cx.run_until_parked();

    let (message, _) = cx.pending_prompt().expect("the refusal says why");
    assert!(message.contains("photo.md"), "{message}");
    assert_eq!(titles(cx), ["Untitled"], "said on a window of its own");
}

#[gpui::test]
fn quitting_with_unsaved_work_asks_and_cancel_stays(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    start(cx, dir.path());
    cx.update(eidola_notepad::new_document);
    let doc = cx.update(|cx| eidola_notepad::documents(cx)[0].clone());
    cx.update(|cx| {
        doc.read(cx)
            .editor()
            .clone()
            .update(cx, |editor, cx| editor.append_at_end("draft", cx))
    });
    assert!(cx.update(|cx| doc.read(cx).is_dirty()));

    cx.update(eidola_notepad::quit);
    cx.run_until_parked();
    let (message, _) = cx.pending_prompt().expect("quit asks first");
    assert!(message.contains("Untitled"), "{message}");
    cx.simulate_prompt_answer("Cancel");
    cx.run_until_parked();
    assert_eq!(titles(cx), ["Untitled"], "the app and its document stay");
}
