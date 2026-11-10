//! A document window's life on disk, driven through real keystrokes, real
//! actions and gpui's simulated sheets and save panels.

use std::path::{Path, PathBuf};

use eidola_notepad::actions::{CloseWindow, RevertToSaved, Save, SaveAs};
use eidola_notepad::document::{Contents, DocumentView};
use eidola_notepad::file;
use gpui::{Entity, TestAppContext, VisualTestContext};
// The editor's own actions rather than keystrokes, whose bindings differ
// between macOS and Linux.
use gpui_markdown_editor::{DocumentEnd, Undo};

fn init(cx: &mut TestAppContext) {
    cx.update(|cx| {
        gpui_component::init(cx);
        eidola_gui::theme::install(cx);
        gpui_markdown_editor::init(cx);
    });
}

fn open(
    contents: Contents,
    cx: &mut TestAppContext,
) -> (Entity<DocumentView>, &mut VisualTestContext) {
    init(cx);
    let (view, cx) = cx.add_window_view(|window, cx| DocumentView::new(contents, window, cx));
    cx.update(|window, cx| {
        let focus = view.read(cx).editor().read(cx).focus_handle.clone();
        window.focus(&focus, cx);
    });
    cx.run_until_parked();
    (view, cx)
}

fn open_file<'a>(
    path: &Path,
    cx: &'a mut TestAppContext,
) -> (Entity<DocumentView>, &'a mut VisualTestContext) {
    let (text, format) = file::decode(&std::fs::read(path).unwrap()).unwrap();
    open(
        Contents::File {
            path: path.to_path_buf(),
            text,
            format,
        },
        cx,
    )
}

fn value(view: &Entity<DocumentView>, cx: &mut VisualTestContext) -> String {
    cx.update(|_, cx| view.read(cx).editor().read(cx).value().to_string())
}

fn dirty(view: &Entity<DocumentView>, cx: &mut VisualTestContext) -> bool {
    cx.update(|_, cx| view.read(cx).is_dirty())
}

fn note(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

#[gpui::test]
fn typing_marks_the_document_edited_and_undo_takes_it_back(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"hello\n");
    let (view, cx) = open_file(&path, cx);
    assert!(!dirty(&view, cx));
    assert_eq!(cx.window_title().as_deref(), Some("a.md"));
    assert_eq!(cx.document_path(), Some(path.clone()));

    cx.dispatch_action(DocumentEnd);
    cx.simulate_input("!");
    assert_eq!(value(&view, cx), "hello!");
    assert!(dirty(&view, cx), "an edit marks the document");

    cx.dispatch_action(Undo);
    assert_eq!(value(&view, cx), "hello");
    assert!(
        !dirty(&view, cx),
        "undoing back to the saved text reads clean"
    );
}

#[gpui::test]
fn save_writes_the_buffer_in_the_files_own_format(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "crlf.md", b"\xEF\xBB\xBFone\r\n\r\ntwo\r\n");
    let (view, cx) = open_file(&path, cx);

    cx.dispatch_action(DocumentEnd);
    cx.simulate_input(" more");
    cx.dispatch_action(Save);
    cx.run_until_parked();

    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"\xEF\xBB\xBFone\r\n\r\ntwo more\r\n",
        "BOM, CRLF and the final newline survive an edit"
    );
    assert!(!dirty(&view, cx));
    assert!(
        !cx.did_prompt_for_new_path(),
        "a titled document saves in place"
    );
}

#[gpui::test]
fn editing_the_body_leaves_frontmatter_alone(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let body = "---\ntitle: A note\ntags: [a, b]\ndate: 2026-10-30\n---\n\nBody";
    let path = note(dir.path(), "a.md", format!("{body}\n").as_bytes());
    let (view, cx) = open_file(&path, cx);

    cx.dispatch_action(DocumentEnd);
    cx.simulate_input(" text");

    assert_eq!(value(&view, cx), format!("{body} text"));
}

#[gpui::test]
fn saving_an_untitled_document_asks_where_and_names_the_window(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let (view, cx) = open(Contents::Untitled(2), cx);
    assert_eq!(cx.window_title().as_deref(), Some("Untitled 2"));

    cx.simulate_input("# Idea");
    cx.dispatch_action(Save);
    cx.run_until_parked();
    assert!(cx.did_prompt_for_new_path());

    let target = dir.path().join("Idea");
    cx.simulate_new_path_selection(|_| Some(target.clone()));
    cx.run_until_parked();

    let saved = dir.path().join("Idea.md");
    assert_eq!(std::fs::read(&saved).unwrap(), b"# Idea\n");
    assert_eq!(cx.window_title().as_deref(), Some("Idea.md"));
    assert_eq!(
        cx.update(|_, cx| view.read(cx).path().map(Path::to_path_buf)),
        Some(saved)
    );
    assert!(!dirty(&view, cx));
}

#[gpui::test]
fn a_save_name_that_gains_an_extension_asks_before_replacing(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let existing = note(dir.path(), "Idea.md", b"keep me\n");
    let (view, cx) = open(Contents::Untitled(1), cx);
    cx.simulate_input("new");

    for (answer, expected) in [("Cancel", &b"keep me\n"[..]), ("Replace", &b"new\n"[..])] {
        cx.dispatch_action(Save);
        cx.run_until_parked();
        let typed = dir.path().join("Idea");
        cx.simulate_new_path_selection(|_| Some(typed.clone()));
        cx.run_until_parked();
        let (message, _) = cx.pending_prompt().expect("replacing Idea.md asks");
        assert!(message.contains("Idea.md"), "{message}");
        cx.simulate_prompt_answer(answer);
        cx.run_until_parked();
        assert_eq!(
            std::fs::read(&existing).unwrap(),
            expected,
            "after {answer}"
        );
    }
    assert!(!dirty(&view, cx));
}

#[gpui::test]
fn saves_in_quick_succession_leave_the_newest_text(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (view, cx) = open_file(&path, cx);
    cx.dispatch_action(DocumentEnd);

    cx.simulate_input("1");
    cx.dispatch_action(Save);
    cx.simulate_input("2");
    cx.dispatch_action(Save);
    cx.run_until_parked();

    assert_eq!(std::fs::read(&path).unwrap(), b"text12\n");
    assert!(!dirty(&view, cx));
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

#[gpui::test]
fn a_save_queued_behind_save_as_follows_the_document_to_its_new_path(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let old = note(dir.path(), "a.md", b"text\n");
    let (view, cx) = open_file(&old, cx);
    cx.dispatch_action(DocumentEnd);

    cx.dispatch_action(SaveAs);
    cx.run_until_parked();
    let new = dir.path().join("b.md");
    cx.simulate_new_path_selection(|_| Some(new.clone()));
    // Before the Save As has written, edit and save again.
    cx.update(|window, cx| {
        let editor = view.read(cx).editor().clone();
        editor.update(cx, |editor, cx| {
            editor.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText(" more".into()),
                cx,
            )
        });
        view.update(cx, |view, cx| view.save(false, window, cx))
            .detach();
    });
    cx.run_until_parked();

    assert_eq!(std::fs::read(&new).unwrap(), b"text more\n");
    assert_eq!(
        std::fs::read(&old).unwrap(),
        b"text\n",
        "the old file is left alone"
    );
    assert_eq!(
        cx.update(|_, cx| view.read(cx).path().map(Path::to_path_buf)),
        Some(new)
    );
    assert!(!dirty(&view, cx));
}

#[gpui::test]
fn save_as_moves_a_titled_document(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "old.md", b"text\n");
    let (view, cx) = open_file(&path, cx);

    cx.dispatch_action(SaveAs);
    cx.run_until_parked();
    let target = dir.path().join("new.markdown");
    cx.simulate_new_path_selection(|_| Some(target.clone()));
    cx.run_until_parked();

    assert_eq!(std::fs::read(&target).unwrap(), b"text\n");
    assert_eq!(
        cx.update(|_, cx| view.read(cx).path().map(Path::to_path_buf)),
        Some(target)
    );
}

#[gpui::test]
fn closing_an_edited_document_asks_and_cancel_keeps_it(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (view, cx) = open_file(&path, cx);
    cx.simulate_input("x");

    cx.dispatch_action(CloseWindow);
    cx.run_until_parked();
    let (message, _) = cx.pending_prompt().expect("a sheet asks first");
    assert!(message.contains("a.md"), "{message}");

    cx.simulate_prompt_answer("Cancel");
    cx.run_until_parked();
    assert!(dirty(&view, cx), "the edit is still there");
    assert_eq!(cx.windows().len(), 1, "the window stayed");
}

#[gpui::test]
fn the_close_button_on_an_edited_document_can_save_then_close(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (_view, cx) = open_file(&path, cx);
    cx.dispatch_action(DocumentEnd);
    cx.simulate_input("!");

    assert!(!cx.simulate_close(), "the window waits for the answer");
    cx.run_until_parked();
    cx.simulate_prompt_answer("Save");
    cx.run_until_parked();

    assert_eq!(std::fs::read(&path).unwrap(), b"text!\n");
    assert!(cx.windows().is_empty(), "saved, then closed");
}

#[gpui::test]
fn typing_during_the_close_time_save_keeps_the_window(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (view, cx) = open_file(&path, cx);
    cx.dispatch_action(DocumentEnd);
    cx.simulate_input("!");

    cx.dispatch_action(CloseWindow);
    cx.run_until_parked();
    // Step the save until its bytes are down, then type before it reports
    // back: the edit is newer than what was written.
    cx.simulate_prompt_answer("Save");
    while std::fs::read(&path).unwrap() != b"text!\n" {
        assert!(cx.executor().tick(), "the save stalled");
    }
    cx.update(|_, cx| {
        view.read(cx).editor().clone().update(cx, |editor, cx| {
            editor.apply_event_for_test(
                gpui_markdown_editor::EditorEvent::InsertText("?".into()),
                cx,
            )
        })
    });
    cx.run_until_parked();

    assert!(dirty(&view, cx), "the newer edit is unsaved");
    assert_eq!(cx.windows().len(), 1, "so the window stayed");
    assert_eq!(value(&view, cx), "text!?");
}

#[gpui::test]
fn dont_save_closes_without_writing(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (_view, cx) = open_file(&path, cx);
    cx.simulate_input("x");

    cx.dispatch_action(CloseWindow);
    cx.run_until_parked();
    cx.simulate_prompt_answer("Don’t Save");
    cx.run_until_parked();

    assert_eq!(std::fs::read(&path).unwrap(), b"text\n");
    assert!(cx.windows().is_empty());
}

#[gpui::test]
fn a_clean_document_closes_without_asking(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (_view, cx) = open_file(&path, cx);
    assert!(cx.simulate_close());
    assert!(!cx.has_pending_prompt());
}

#[gpui::test]
fn revert_restores_what_is_on_disk(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"text\n");
    let (view, cx) = open_file(&path, cx);
    cx.simulate_input("junk ");

    cx.dispatch_action(RevertToSaved);
    cx.run_until_parked();
    cx.simulate_prompt_answer("Revert");
    cx.run_until_parked();

    assert_eq!(value(&view, cx), "text");
    assert!(!dirty(&view, cx));
}

#[gpui::test]
fn a_clean_document_follows_a_change_made_elsewhere(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"before\n");
    let (view, cx) = open_file(&path, cx);

    cx.deactivate_window();
    // Another program writes the file; a later mtime is what it leaves.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&path, b"after\n").unwrap();
    cx.update(|window, _| window.activate_window());
    cx.run_until_parked();

    assert_eq!(value(&view, cx), "after");
    assert!(!cx.has_pending_prompt(), "nothing to ask about");
}

#[gpui::test]
fn an_edited_document_asks_before_taking_a_change_made_elsewhere(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"before\n");
    let (view, cx) = open_file(&path, cx);
    cx.simulate_input("mine ");

    cx.deactivate_window();
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&path, b"theirs\n").unwrap();
    cx.update(|window, _| window.activate_window());
    cx.run_until_parked();

    assert!(cx.has_pending_prompt());
    cx.simulate_prompt_answer("Keep This Version");
    cx.run_until_parked();
    assert_eq!(value(&view, cx), "mine before");

    // Asked once per change on disk, not on every activation.
    cx.deactivate_window();
    cx.update(|window, _| window.activate_window());
    cx.run_until_parked();
    assert!(!cx.has_pending_prompt());
}

#[gpui::test]
fn an_image_beside_the_document_resolves_against_its_folder(cx: &mut TestAppContext) {
    let dir = tempfile::tempdir().unwrap();
    let path = note(dir.path(), "a.md", b"![](fig.png)\n");
    let (view, cx) = open_file(&path, cx);
    let base = cx.update(|_, cx| {
        view.read(cx)
            .editor()
            .read(cx)
            .image_base_dir()
            .map(Path::to_path_buf)
    });
    assert_eq!(base, Some(dir.path().to_path_buf()));
}
